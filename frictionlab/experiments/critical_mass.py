"""Critical-mass experiment (fl-065): when does a two-sided platform escape cold start?

The fl-064 `PlatformMarket` is run through the fl-023 engine with the standalone
schedules, cross-side benefits b, fees f and horizon T fixed, sweeping
adoption_cost x seed_members on common random numbers (every cell draws the same
standalone values). `seed_members = k` pre-enrols the first min(k, n_X) ids of each
side at the platform's expense.

With u the per-member standalone values (member-id order), the best-response map
on membership acts only through the counts (n_a, n_b) of the previous round:

    stay iff u + b * n_other - f >= 0,     join iff u + b * n_other - f >= c

It is monotone (b >= 0), so nested seeds give nested paths. Starting from the
full set no one ever joins, so the greatest fixed point H does not depend on c.
If the empty state is not a fixed point (max u - f >= c on some side) there is no
cold start and `critical_mass` is 0. Otherwise it is the smallest symmetric seed
whose path ends at H; both cases are non-decreasing in c because a higher c only
removes joins. `tipping_fixed_points` lists the attracting fixed points reached
from symmetric seeds 0 .. max(n). Schedules are per-member values in id order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..analysis.pnl import member_surplus, platform_pnl
from ..config.schema import Param
from ..markets import PLATFORM_ID, PlatformMarket
from .base import ExperimentResult, run_experiment

COST, SEED = "adoption_cost", "seed_members"
COSTS, SEEDS = (0, 2, 4, 8, 16, 40), (0, 2, 4, 6, 8, 10, 14)
Counts = Tuple[int, int]


@dataclass(frozen=True)
class Config:
    """Fixed platform: side sizes, standalone schedules, benefits, fees, horizon."""
    n_a: int = 12
    n_b: int = 14
    standalone_a: Tuple[int, ...] = (-30, -20, -10, -4, 0, 3, 8)
    standalone_b: Tuple[int, ...] = (-25, -12, -6, -1, 2, 6)
    benefit_a: int = 3
    benefit_b: int = 2
    fee_a: int = 2
    fee_b: int = -1
    horizon: int = 40


def _path(u_a, u_b, b_a, b_b, f_a, f_b, c, k_a, k_b) -> Tuple[Counts, int]:
    """Iterate the map from the first (k_a, k_b) ids; returns final counts and cycle length."""
    ua, ub = np.asarray(u_a, dtype=np.int64), np.asarray(u_b, dtype=np.int64)
    ma, mb = np.arange(ua.size) < k_a, np.arange(ub.size) < k_b
    seen: Dict[bytes, int] = {}
    while True:
        key = ma.tobytes() + mb.tobytes()
        if key in seen:
            return (int(ma.sum()), int(mb.sum())), len(seen) - seen[key]
        seen[key] = len(seen)
        na_, nb_ = int(ma.sum()), int(mb.sum())
        na, nb = ua + b_a * nb_ - f_a, ub + b_b * na_ - f_b
        ma, mb = np.where(ma, na >= 0, na >= c), np.where(mb, nb >= 0, nb >= c)


def _attractors(u_a, u_b, b_a, b_b, f_a, f_b, c) -> List[Tuple[Counts, int]]:
    n = max(len(u_a), len(u_b))
    return [_path(u_a, u_b, b_a, b_b, f_a, f_b, c, min(k, len(u_a)), min(k, len(u_b)))
            for k in range(n + 1)]


def tipping_fixed_points(standalone_schedule_a: Sequence[int], standalone_schedule_b: Sequence[int],
                         benefit_a: int, benefit_b: int, fee_a: int, fee_b: int,
                         adoption_cost: int) -> List[Counts]:
    """Every stable (attracting) membership fixed point, in increasing order."""
    pts = _attractors(standalone_schedule_a, standalone_schedule_b, benefit_a, benefit_b,
                      fee_a, fee_b, int(adoption_cost))
    return sorted({p for p, cyc in pts if cyc == 1}, key=lambda p: (p[0] + p[1], p))


def critical_mass(standalone_schedule_a: Sequence[int], standalone_schedule_b: Sequence[int],
                  benefit_a: int, benefit_b: int, fee_a: int, fee_b: int,
                  adoption_cost: int) -> int:
    """Smallest symmetric seed count whose best-response path tips to the greatest fixed point."""
    if not empty_is_fixed_point(standalone_schedule_a, standalone_schedule_b, fee_a, fee_b,
                                int(adoption_cost)):
        return 0                                      # no cold-start trap to escape
    pts = _attractors(standalone_schedule_a, standalone_schedule_b, benefit_a, benefit_b,
                      fee_a, fee_b, int(adoption_cost))
    return next(k for k, (p, cyc) in enumerate(pts) if cyc == 1 and p == pts[-1][0])


def empty_is_fixed_point(u_a, u_b, f_a: int, f_b: int, adoption_cost: int) -> bool:
    """True iff nobody joins an empty platform: max(u) - f < c on both sides."""
    return max(u_a) - f_a < adoption_cost and max(u_b) - f_b < adoption_cost


class PlatformCell:
    """One seeded platform plus its zero-adoption-cost twin on the same draws."""

    def __init__(self, market: PlatformMarket, twin: PlatformMarket, k: Counts) -> None:
        self.market, self.twin, self.k = market, twin, k

    def run(self) -> "PlatformCell":
        self.market.run()
        self.twin.run()
        return self


def build(cfg: Config, params: Dict[str, Any], seed: int) -> PlatformCell:
    """fl-023 factory: market at (c, k) and its c = 0 twin, both on `seed`."""
    k = (min(int(params[SEED]), cfg.n_a), min(int(params[SEED]), cfg.n_b))
    kw = dict(standalone=(cfg.standalone_a, cfg.standalone_b), benefit=(cfg.benefit_a, cfg.benefit_b),
              fee=(cfg.fee_a, cfg.fee_b), seed_members=k, rounds=cfg.horizon, seed=seed)
    return PlatformCell(PlatformMarket((cfg.n_a, cfg.n_b), adoption=int(params[COST]), **kw),
                        PlatformMarket((cfg.n_a, cfg.n_b), adoption=0, **kw), k)


def side_values(m: PlatformMarket) -> Tuple[List[int], List[int]]:
    """Realized standalone values per side in member-id order."""
    return tuple([m.values[i] for i in sorted(m.side_of) if m.side_of[i] == s] for s in (0, 1))


def _welfare(m: PlatformMarket) -> int:
    return sum(r.net for r in member_surplus(m).values()) + platform_pnl(m).net


def outcome(m: PlatformMarket) -> Tuple[str, int]:
    """('fixed_point' | 'cycle' | 'horizon', cycle length) of a finished run."""
    h = m.history
    if h[-1] not in h[:-1]:
        return "horizon", 0
    cyc = len(h) - 1 - max(i for i, s in enumerate(h[:-1]) if s == h[-1])
    return ("fixed_point" if cyc == 1 else "cycle"), cyc


def platform_metrics(cell: PlatformCell) -> Dict[str, Any]:
    """Final membership, convergence, adoption paid, seed subsidy, PnL, surplus, welfare."""
    m, tw = cell.market, cell.twin
    c = m.adoption.cost_for(PLATFORM_ID)
    u_a, u_b = side_values(m)
    args = (u_a, u_b, m.benefit[0], m.benefit[1], m.fee[0], m.fee[1], c)
    kind, cyc = outcome(m)
    final = (m.count(0), m.count(1))
    seeded = m.history[0]
    pnl, ms = platform_pnl(m), sum(r.net for r in member_surplus(m).values())
    return {"members_a": final[0], "members_b": final[1], "members": sum(final),
            "share": sum(final) / (m.n[0] + m.n[1]), "rounds": len(m.log), "outcome": kind,
            "cycle_length": cyc, "adoption_paid": m.adoption.revenue,
            "seed_subsidy": sum(f.amount for f in m.flows if f.payer == PLATFORM_ID and f.member in seeded),
            "fee_revenue": pnl.fee_revenue, "subsidy_outlay": pnl.subsidy_outlay, "platform_pnl": pnl.net,
            "member_surplus": ms, "welfare_change_vs_free_adoption": ms + pnl.net - _welfare(tw),
            "on_fixed_point": final in tipping_fixed_points(*args), "critical_mass": critical_mass(*args),
            "empty_fixed_point": empty_is_fixed_point(u_a, u_b, m.fee[0], m.fee[1], c),
            "violations": len(m.ledger.violations()) + len(tw.ledger.violations())}


def run_platform_grid(cfg: Config = Config(), adoption_costs: Sequence[int] = COSTS,
                      seed_members: Sequence[int] = SEEDS, *, replicates: int = 1,
                      master_seed: int = 65, n_workers: int = 1) -> ExperimentResult:
    """Sweep adoption_cost x seed_members through the fl-023 engine (CRN)."""
    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        cold = [r for r in rows if r[SEED] == 0 and r["members"] == 0]
        return {"cold_start_traps": len(cold), "max_members": max(r["members"] for r in rows),
                "max_critical_mass": max(r["critical_mass"] for r in rows),
                "total_seed_subsidy": sum(r["seed_subsidy"] for r in rows),
                "min_welfare_change": min(r["welfare_change_vs_free_adoption"] for r in rows)}

    return run_experiment("critical_mass", partial(build, cfg), platform_metrics,
                          {COST: [int(x) for x in adoption_costs], SEED: [int(x) for x in seed_members]},
                          replicates=replicates, master_seed=master_seed, crn=True,
                          n_workers=n_workers, findings=headline)


@dataclass(frozen=True)
class TippingSeed:
    """Smallest seed_members whose final membership exceeds `fraction` of the population."""
    seed_members: int
    analytic: int
    adoption_cost: int
    fraction: float
    probes: Dict[int, int] = field(repr=False)


def find_tipping_seed(cfg: Config = Config(), adoption_cost: int = 8, fraction: float = 0.9, *,
                      master_seed: int = 65) -> TippingSeed:
    """Integer bisection on seed_members in [0, max(n)]; every probe reuses the master seed."""
    target, probes, rows = fraction * (cfg.n_a + cfg.n_b), {}, {}

    def members(k: int) -> int:
        if k not in probes:
            rows[k] = run_platform_grid(cfg, [adoption_cost], [k], master_seed=master_seed).rows[0]
            probes[k] = rows[k]["members"]
        return probes[k]

    lo, hi = -1, max(cfg.n_a, cfg.n_b)                 # invariant: fails at lo, tips at hi
    if members(hi) <= target:
        raise ValueError(f"membership never exceeds {fraction:.0%} of the population")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if members(mid) > target else (mid, hi)
    return TippingSeed(hi, rows[hi]["critical_mass"], int(adoption_cost), float(fraction),
                       dict(sorted(probes.items())))


def run_critical_mass(cfg: Config = Config(), adoption_costs: Sequence[int] = COSTS,
                      seed_members: Sequence[int] = SEEDS, *, tipping_cost: int = 8,
                      tipping_fraction: float = 0.9, replicates: int = 1, master_seed: int = 65,
                      n_workers: int = 1) -> ExperimentResult:
    """Grid + analytic critical mass + bisected tipping seed."""
    res = run_platform_grid(cfg, adoption_costs, seed_members, replicates=replicates,
                            master_seed=master_seed, n_workers=n_workers)
    tip = find_tipping_seed(cfg, tipping_cost, tipping_fraction, master_seed=master_seed)
    res.findings.update({"tipping_cost": tip.adoption_cost, "tipping_fraction": tip.fraction,
                         "tipping_seed": tip.seed_members, "analytic_critical_mass": tip.analytic})
    return res


# ---- fl-036 registry spec (kept here so the registry module stays under the line ceiling)
_TICKS, _INT, _POS, _FRIC = Param("int", seq=True, friction=True), Param("int"), Param("int", min=1), Param("int", friction=True)
PARAM_SCHEMA: Dict[str, Param] = {
    "adoption_costs": _TICKS, "seed_members": Param("int", seq=True, min=0), "n_a": _POS, "n_b": _POS,
    "standalone_a": Param("int", seq=True), "standalone_b": Param("int", seq=True), "benefit_a": _FRIC,
    "benefit_b": _FRIC, "fee_a": _INT, "fee_b": _INT, "horizon": _POS, "tipping_cost": _FRIC,
    "tipping_fraction": Param("float", min=0, max=0.999)}
SMALL: Dict[str, Any] = {"adoption_costs": [0, 2, 8, 40], "seed_members": [0, 4, 6, 8], "n_a": 8, "n_b": 8,
                         "standalone_a": [-6], "standalone_b": [-6], "benefit_a": 1, "benefit_b": 1,
                         "fee_a": 0, "fee_b": 0, "horizon": 20, "tipping_cost": 2}


def defaults() -> Dict[str, Any]:
    """Registered defaults: `Config()` fields plus the grid and tipping-probe settings."""
    base = {f: getattr(Config(), f) for f in Config.__dataclass_fields__}
    base = {k: list(v) if isinstance(v, tuple) else v for k, v in base.items()}
    return {"adoption_costs": list(COSTS), "seed_members": list(SEEDS), **base,
            "tipping_cost": 8, "tipping_fraction": 0.9}


def registry_runner(p: Dict[str, Any], master_seed: int, replicates: int, n_workers: int) -> ExperimentResult:
    """fl-036 runner: resolved params -> `run_critical_mass`."""
    cfg = Config(**{k: tuple(p[k]) if isinstance(p[k], list) else p[k] for k in Config.__dataclass_fields__})
    return run_critical_mass(cfg, p["adoption_costs"], p["seed_members"], tipping_cost=p["tipping_cost"],
                             tipping_fraction=p["tipping_fraction"], replicates=replicates,
                             master_seed=master_seed, n_workers=n_workers)
