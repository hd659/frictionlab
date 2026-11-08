"""Lemons and certification (fl-056): how far does the pool unravel, and who pays to escape?

The fl-055 `LemonsMarket` is swept over certification_cost x buyer_premium with a
fixed seeded quality draw (common random numbers: every cell sees the same
sellers). Without certification the pooled price is the Akerlof fixed point

    p_(k+1) = floor(mean{q : q <= p_k}) + premium,   p_0 = floor(mean q) + premium

which `akerlof_pooled_price` computes by direct iteration. A seller certifies iff
q + premium - c >= max(p, q). Its gain over keeping is premium, and any pooled price
is at least q_min + premium, so the gain over the pooled sale is at most top quality
minus that pooled price, q_top - q_min. Hence nobody certifies at any cost
>= `certification_bound` = min(premium, q_top - q_min) + 1.
Surplus is measured against the fl-022 full-information competitive equilibrium.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..analysis.sweep import SweepSpec
from ..analysis.welfare import competitive_equilibrium
from ..costs import CertificationPolicy
from ..markets import CERTIFIED, LemonsMarket
from .base import ExperimentResult, run_experiment

COST, PREMIUM = "certification_cost", "buyer_premium"
COSTS, PREMIUMS = (0, 2, 4, 8, 16, 32), (0, 5, 10, 20, 40, 90)
QUINTILES = 5


@dataclass(frozen=True)
class Config:
    """Discrete quality schedule, seller count and whether certification is offered."""
    schedule: Tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    n_sellers: int = 40
    certification: bool = True


def _iterate(qs: Sequence[int], premium: int, cost: Optional[int]) -> Tuple[Optional[int], Tuple[int, ...], Tuple[int, ...]]:
    """Mirror of the fl-055 price loop: (final pooled price, offered qualities, certified qualities)."""
    qs, prev, seen, pool, cert = [int(q) for q in qs], None, set(), (), ()
    for _ in range(10_000):
        cert = () if prev is None or cost is None else tuple(
            q for q in qs if q + premium - cost >= max(prev, q))
        rest = [q for q in qs if q not in cert] if cert else qs
        pool = tuple(rest if prev is None else (q for q in rest if q <= prev))
        if not pool:
            return None, (), cert
        p = sum(pool) // len(pool) + premium
        if p in seen:
            return p, pool, cert
        seen.add(p)
        prev = p
    return prev, pool, cert


def akerlof_pooled_price(quality_schedule: Sequence[int], buyer_premium: int) -> Tuple[int, int]:
    """(integer-tick fixed-point pooled price, highest quality still offered), no market."""
    p, pool, _ = _iterate(quality_schedule, int(buyer_premium), None)
    return int(p), max(pool)


def certification_cutoff(quality_schedule: Sequence[int], buyer_premium: int,
                         cost: int) -> Optional[int]:
    """Smallest quality that certifies at the fixed point (None if nobody certifies)."""
    cert = _iterate(quality_schedule, int(buyer_premium), int(cost))[2]
    return min(cert) if cert else None


def certification_bound(quality_schedule: Sequence[int], buyer_premium: int) -> int:
    """Smallest cost at which no seller can certify: min(premium, q_top - q_min) + 1."""
    qs = [int(q) for q in quality_schedule]
    return min(int(buyer_premium), max(qs) - min(qs)) + 1


class LemonsCell:
    """One seeded market; `run()` returns the cell so fl-023 metrics see the market."""

    def __init__(self, market: LemonsMarket) -> None:
        self.market = market

    def run(self) -> "LemonsCell":
        self.market.run()
        return self


def build(cfg: Config, params: Dict[str, Any], seed: int) -> LemonsCell:
    """fl-023 factory: same seeded qualities in every cell, priced at (cost, premium)."""
    cert = CertificationPolicy(int(params[COST])) if cfg.certification else None
    return LemonsCell(LemonsMarket.from_schedule(cfg.n_sellers, cfg.schedule, seed=seed,
                                                 premium=int(params[PREMIUM]), certification=cert))


def lemons_metrics(cell: LemonsCell) -> Dict[str, Any]:
    """Traded share (overall, by quality quintile), price, uptake, cutoff (q_top + 1 if none), welfare."""
    m = cell.market
    n, prem, sold = len(m.qualities), m.premium, set(m.sellers_sold())
    ranked = sorted(m.qualities, key=lambda i: (m.qualities[i], i))
    out: Dict[str, Any] = {}
    for k in range(QUINTILES):
        ids = ranked[k * n // QUINTILES:(k + 1) * n // QUINTILES]
        out[f"q{k + 1}_traded_share"] = sum(i in sold for i in ids) / len(ids) if ids else 0.0
    ce = competitive_equilibrium([q + prem for q in m.qualities.values()], list(m.qualities.values()))
    realized = sum(t.price - m.qualities[t.seller_id] for t in m.trades) \
        + sum(m.qualities[t.seller_id] + prem - t.price for t in m.trades) - m.certification_costs
    certified = m.sellers_sold(CERTIFIED)
    return {"traded_share": len(sold) / n, **out, "pooled_price": m.price or 0,
            "rounds": len(m.rounds), "certified": len(certified),
            "certification_uptake": len(certified) / n,
            "cutoff": min((m.qualities[i] for i in certified), default=max(m.qualities.values()) + 1),
            "surplus_share": realized / ce.max_surplus if ce.max_surplus else 0.0,
            "certification_paid": m.certification_costs,
            "violations": len(m.ledger.violations())}


def run_lemons_grid(cfg: Config = Config(), certification_costs: Sequence[int] = COSTS,
                    buyer_premiums: Sequence[int] = PREMIUMS, *, replicates: int = 1,
                    master_seed: int = 56, n_workers: int = 1) -> ExperimentResult:
    """Sweep certification_cost x buyer_premium through the fl-023 engine (CRN)."""
    costs, prems = [int(c) for c in certification_costs], [int(p) for p in buyer_premiums]

    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        at = lambda p, k: max(r[k] for r in rows if r[PREMIUM] == p)
        return {"lowest_premium_traded_share": at(prems[0], "traded_share"),
                "highest_premium_traded_share": at(prems[-1], "traded_share"),
                "max_certification_uptake": max(r["certification_uptake"] for r in rows),
                "total_certification_paid": sum(r["certification_paid"] for r in rows)}

    return run_experiment("lemons", partial(build, cfg), lemons_metrics, {COST: costs, PREMIUM: prems},
                          replicates=replicates, master_seed=master_seed, crn=True,
                          n_workers=n_workers, findings=headline)


@dataclass(frozen=True)
class CertificationThreshold:
    """Smallest cost at which uptake falls below `fraction`, with the analytic bound."""
    threshold: int
    analytic: int
    fraction: float
    probes: Dict[int, float] = field(repr=False)


def find_certification_threshold(cfg: Config = Config(), buyer_premium: int = 20,
                                 fraction: float = 0.05, *, master_seed: int = 56) -> CertificationThreshold:
    """Integer bisection on [0, bound]; every probe reuses the same master seed."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    bound, probes = certification_bound(cell_qualities(cfg, master_seed), buyer_premium), {}

    def uptake(c: int) -> float:
        if c not in probes:
            probes[c] = run_lemons_grid(cfg, [c], [buyer_premium],
                                        master_seed=master_seed).rows[0]["certification_uptake"]
        return probes[c]

    lo, hi = 0, bound                     # uptake(bound) == 0 < fraction by the bound
    if uptake(0) < fraction:
        hi = 0
    while hi - lo > 1:                    # invariant: uptake(lo) >= fraction > uptake(hi)
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if uptake(mid) < fraction else (mid, hi)
    return CertificationThreshold(hi, bound, float(fraction), dict(sorted(probes.items())))


def cell_qualities(cfg: Config, master_seed: int = 56) -> List[int]:
    """The seeded quality draw every CRN cell (replicate 0) sees under `master_seed`."""
    seed = SweepSpec(build, lemons_metrics, grid={COST: [0]}, master_seed=master_seed).child_seeds()[0][0]
    return list(build(cfg, {COST: 0, PREMIUM: 0}, seed).market.qualities.values())


def run_lemons(cfg: Config = Config(), certification_costs: Sequence[int] = COSTS,
               buyer_premiums: Sequence[int] = PREMIUMS, *, threshold_premium: int = 20,
               uptake_fraction: float = 0.05, replicates: int = 1, master_seed: int = 56,
               n_workers: int = 1) -> ExperimentResult:
    """Grid + analytic Akerlof prices per premium + bisected certification threshold."""
    res = run_lemons_grid(cfg, certification_costs, buyer_premiums, replicates=replicates,
                          master_seed=master_seed, n_workers=n_workers)
    qs = cell_qualities(cfg, master_seed)
    analytic = {int(p): akerlof_pooled_price(qs, p)[0] for p in buyer_premiums}
    th = find_certification_threshold(cfg, threshold_premium, uptake_fraction, master_seed=master_seed)
    res.findings.update({"akerlof_pooled_price": analytic, "threshold_premium": int(threshold_premium),
                         "certification_threshold": th.threshold, "certification_bound": th.analytic})
    return res
