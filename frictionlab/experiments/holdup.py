"""Hold-up and costly-contracting experiment (fl-074) on the fl-073 `HoldupMarket`.

Schedule, outside value o and horizon are fixed; the grid is contracting_cost x
seller_weight_bps w (every seller a `SpecificInvestor`), each cell beside a zero-cost twin
on the same seed. Exact integer analytics (b = buyer_share, S(i) = Q(i) - c(i), h = hold-up
level, i* = first best):

    holdup_surplus_loss   L = S(i*) - S(h)               (0 iff h attains the first best)
    seller contract gain  G = seller_share(S(i*)) - (seller_share(Q(h)) - c(h))
                            = L + b(Q(h)) - b(S(i*))
    max_contracting_cost    = G - 1  (0 if G <= 0)       (ties go to no contract)

G is the hold-up loss the seller recovers plus the buyer's contract-time share of the
sunk cost, so G > 0 is possible at h == i* when w < 10000; at w = 10000 both are 0. G is
hump-shaped in w; the default weight grid sits on its falling side (w >= 5000).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Sequence, Tuple

from ..agents.specific_investor import (
    SpecificInvestor, check_schedule, first_best_level, holdup_level, seller_share,
)
from ..analysis.pnl import holdup_buyer_surplus, holdup_pnl
from ..config.schema import Param
from ..costs.contracting import ContractingCost
from ..markets import CONTRACT, HoldupMarket
from .base import ExperimentResult, run_experiment

COST, WEIGHT, BPS = "contracting_cost", "seller_weight_bps", 10_000
COSTS, WEIGHTS = (0, 1, 2, 4, 6, 8, 10, 12), (5_000, 6_000, 7_500, 9_000, 10_000)


@dataclass(frozen=True)
class Config:
    """Fixed market: schedule (costs, values), outside value, pairs, buyers, rounds, noise."""
    investment_costs: Tuple[int, ...] = (0, 6, 14, 24, 36, 50)
    buyer_values: Tuple[int, ...] = (40, 58, 74, 88, 98, 104)
    outside_value: int = 30
    n_pairs: int = 4
    n_buyers: int = 4
    rounds: int = 40
    value_noise: Tuple[int, ...] = (-4, 0, 4)

    @property
    def schedule(self) -> Tuple[Tuple[int, int], ...]:
        if len(self.investment_costs) != len(self.buyer_values):
            raise ValueError("investment_costs and buyer_values must have equal length")
        return check_schedule(tuple(zip(self.investment_costs, self.buyer_values)))


def _surplus(schedule, o: int, i: int) -> int:
    c, v = schedule[i]
    return v - o - c


def holdup_investment(schedule, outside_value: int, seller_weight_bps: int) -> int:
    """Integer investment level chosen under ex-post bargaining (smallest argmax)."""
    return holdup_level(check_schedule(schedule), int(outside_value), int(seller_weight_bps))


def first_best_investment(schedule, outside_value: int) -> int:
    """Smallest surplus-maximizing investment level."""
    return first_best_level(check_schedule(schedule), int(outside_value))


def holdup_surplus_loss(schedule, outside_value: int, seller_weight_bps: int) -> int:
    """Joint surplus destroyed by hold-up, S(i*) - S(h), in ticks."""
    s, o = check_schedule(schedule), int(outside_value)
    return _surplus(s, o, first_best_investment(s, o)) - _surplus(
        s, o, holdup_investment(s, o, seller_weight_bps))


def max_contracting_cost(schedule, outside_value: int, seller_weight_bps: int) -> int:
    """Largest integer contracting cost at which the seller still strictly prefers to contract."""
    s, o, w = check_schedule(schedule), int(outside_value), int(seller_weight_bps)
    h, c_h = holdup_investment(s, o, w), s[holdup_investment(s, o, w)][0]
    gain = (seller_share(_surplus(s, o, first_best_investment(s, o)), w)
            - (seller_share(s[h][1] - o, w) - c_h))
    return max(gain - 1, 0)


def noisy_bound(cfg: Config, seller_weight_bps: int) -> int:
    """`max_contracting_cost` maximized over the value-noise shifts the market can draw."""
    return max(max_contracting_cost(tuple((c, v + e) for c, v in cfg.schedule),
                                    cfg.outside_value, seller_weight_bps) for e in cfg.value_noise)


def market(cfg: Config, cost: int, w: int, seed: int) -> HoldupMarket:
    """Seeded market: n_pairs `SpecificInvestor`s (ids 1..), buyers 101.., flat contract cost."""
    n = cfg.n_pairs
    return HoldupMarket([SpecificInvestor(i + 1) for i in range(n)], [cfg.schedule] * n,
                        [101 + j for j in range(cfg.n_buyers)], rounds=cfg.rounds,
                        outside_value=cfg.outside_value, seller_weight_bps=w,
                        value_noise=cfg.value_noise, contracting=ContractingCost(cost), seed=seed)


class HoldupCell:
    """One seeded market at (cost, w) plus its zero-contracting-cost twin on the same draws."""

    def __init__(self, cfg: Config, m: HoldupMarket, twin: HoldupMarket) -> None:
        self.cfg, self.market, self.twin = cfg, m, twin

    def run(self) -> "HoldupCell":
        self.market.run()
        self.twin.run()
        return self


def build(cfg: Config, params: Dict[str, Any], seed: int) -> HoldupCell:
    """fl-023 factory: the cell market and its zero-cost twin, both on `seed`."""
    w = int(params[WEIGHT])
    return HoldupCell(cfg, market(cfg, int(params[COST]), w, seed), market(cfg, 0, w, seed))


def _first_best(m: HoldupMarket, r) -> Tuple[int, int]:
    sched = tuple((c, v + r.noise) for c, v in m.schedules[r.pair])
    i = first_best_level(sched, m.outside)
    return i, _surplus(sched, m.outside, i)


def welfare(m: HoldupMarket) -> int:
    """fl-022 welfare over the outside option: value - outside - investment - contracting."""
    return sum(r.value - m.outside - r.investment - r.contract_cost for r in m.log)


def holdup_metrics(cell: HoldupCell) -> Dict[str, Any]:
    """Contracting rate, investment vs first best, gap, price, costs, surpluses, welfare."""
    m, log = cell.market, cell.market.log
    fb = [_first_best(m, r) for r in log]
    max_s, n = sum(s for _, s in fb), len(log)
    seller, buyer = (sum(x.net for x in d.values()) for d in (holdup_pnl(m), holdup_buyer_surplus(m)))
    w = welfare(m)          # surplus over the outside option, net of investment and contracts
    return {"trades": n, "contracts": m.contracts, "contracting_rate": m.contracts / n,
            "mean_investment": sum(r.level for r in log) / n,
            "first_best_investment": sum(i for i, _ in fb) / n,
            "underinvestment_gap": sum(s - (r.value - m.outside - r.investment)
                                       for (_, s), r in zip(fb, log)) / n,
            "mean_price": sum(r.price for r in log) / n,
            "contracting_cost_paid": m.contracting.revenue,
            "seller_surplus": seller - n * m.outside, "buyer_surplus": buyer,
            "welfare": w, "max_surplus": max_s, "surplus_share": w / max_s if max_s else 1.0,
            "welfare_change_vs_free_contracting": w - welfare(cell.twin),
            "max_contracting_cost": noisy_bound(cell.cfg, m.w),
            "violations": len(m.ledger.violations()) + len(cell.twin.ledger.violations())}


def run_holdup_grid(cfg: Config = Config(), costs: Sequence[int] = COSTS,
                    weights: Sequence[int] = WEIGHTS, *, replicates: int = 1,
                    master_seed: int = 74, n_workers: int = 1) -> ExperimentResult:
    """Sweep contracting_cost x seller_weight_bps through the fl-023 engine (CRN)."""
    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"max_contracting_rate": max(r["contracting_rate"] for r in rows),
                "min_surplus_share": min(r["surplus_share"] for r in rows),
                "max_underinvestment_gap": max(r["underinvestment_gap"] for r in rows),
                "total_contracting_cost_paid": sum(r["contracting_cost_paid"] for r in rows),
                "worst_welfare_change": min(r["welfare_change_vs_free_contracting"] for r in rows)}

    return run_experiment("holdup", partial(build, cfg), holdup_metrics,
                          {COST: [int(x) for x in costs], WEIGHT: [int(x) for x in weights]},
                          replicates=replicates, master_seed=master_seed, crn=True,
                          n_workers=n_workers, findings=headline)


@dataclass(frozen=True)
class ContractingThreshold:
    """Smallest contracting cost whose simulated contracting rate is < collapse_bps / 10000."""
    contracting_cost: int
    analytic: int                                  # noise-adjusted max_contracting_cost
    seller_weight_bps: int
    probes: Dict[int, float] = field(repr=False)   # cost -> simulated contracting rate


def find_contracting_threshold(cfg: Config = Config(), seller_weight_bps: int = 5_000, *,
                               collapse_bps: int = 5_000, master_seed: int = 74,
                               max_cost: int = 1 << 20) -> ContractingThreshold:
    """Integer bisection on contracting cost; every probe reuses the master seed."""
    if not 0 < collapse_bps <= BPS:
        raise ValueError("need 0 < collapse_bps <= 10000")
    probes: Dict[int, float] = {}

    def collapsed(c: int) -> bool:
        if c not in probes:
            probes[c] = run_holdup_grid(cfg, [c], [seller_weight_bps],
                                        master_seed=master_seed).rows[0]["contracting_rate"]
        return probes[c] * BPS < collapse_bps

    lo, hi = (0, 0) if collapsed(0) else (0, 1)
    if hi:
        while not collapsed(hi):
            lo, hi = hi, 2 * hi
            if hi > max_cost:
                raise ValueError(f"no contracting cost <= {max_cost} collapses contracting")
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if collapsed(mid) else (mid, hi)
    return ContractingThreshold(hi, noisy_bound(cfg, seller_weight_bps), int(seller_weight_bps),
                                dict(sorted(probes.items())))


def run_holdup(cfg: Config = Config(), costs: Sequence[int] = COSTS,
               weights: Sequence[int] = WEIGHTS, *, probe_weight: int = 5_000,
               collapse_bps: int = 5_000, replicates: int = 1, master_seed: int = 74,
               n_workers: int = 1) -> ExperimentResult:
    """Grid + analytic levels + bisected contracting threshold."""
    res = run_holdup_grid(cfg, costs, weights, replicates=replicates, master_seed=master_seed,
                          n_workers=n_workers)
    thr = find_contracting_threshold(cfg, probe_weight, collapse_bps=collapse_bps,
                                     master_seed=master_seed)
    s, o = cfg.schedule, cfg.outside_value
    res.findings.update({
        "probe_weight": int(probe_weight), "collapse_bps": int(collapse_bps),
        "first_best_investment": first_best_investment(s, o),
        "holdup_investment": holdup_investment(s, o, probe_weight),
        "holdup_surplus_loss": holdup_surplus_loss(s, o, probe_weight),
        "max_contracting_cost": max_contracting_cost(s, o, probe_weight),
        "bisected_threshold": thr.contracting_cost, "analytic_threshold": thr.analytic})
    return res


# ---- fl-036 registry spec (kept here so the registry module stays under the line ceiling)
_POS, _SEQ = Param("int", min=1), Param("int", seq=True, min=0)
PARAM_SCHEMA: Dict[str, Param] = {
    "contracting_costs": Param("int", seq=True, friction=True),
    "seller_weights": Param("int", seq=True, min=0, max=BPS), "investment_costs": _SEQ,
    "buyer_values": _SEQ, "outside_value": Param("int", min=0), "n_pairs": _POS, "n_buyers": _POS,
    "rounds": _POS, "value_noise": Param("int", seq=True), "probe_weight": Param("int", min=0, max=BPS),
    "collapse_bps": Param("int", min=1, max=BPS)}
SMALL: Dict[str, Any] = {"contracting_costs": [0, 2, 6, 10], "seller_weights": [5000, 7500, 10000],
                         "n_pairs": 2, "rounds": 12, "value_noise": [0]}
_TUPLES = ("investment_costs", "buyer_values", "value_noise")


def defaults() -> Dict[str, Any]:
    """Registered defaults: `Config()` fields plus the grid and threshold-probe settings."""
    base = {f: getattr(Config(), f) for f in Config.__dataclass_fields__}
    return {"contracting_costs": list(COSTS), "seller_weights": list(WEIGHTS),
            **{k: list(v) if k in _TUPLES else v for k, v in base.items()},
            "probe_weight": 5_000, "collapse_bps": 5_000}


def registry_runner(p: Dict[str, Any], master_seed: int, replicates: int, n_workers: int) -> ExperimentResult:
    """fl-036 runner: resolved params -> `run_holdup`."""
    cfg = Config(**{k: tuple(p[k]) if k in _TUPLES else p[k] for k in Config.__dataclass_fields__})
    return run_holdup(cfg, p["contracting_costs"], p["seller_weights"], probe_weight=p["probe_weight"],
                      collapse_bps=p["collapse_bps"], replicates=replicates,
                      master_seed=master_seed, n_workers=n_workers)
