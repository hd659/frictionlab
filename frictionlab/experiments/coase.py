"""Coase threshold: at what per-trade transaction cost does trading collapse?

A ZI-C double auction with a fixed, seed-free valuation/cost schedule (one
unit per trader) is run under one total per-trade friction level F, in
integer ticks: a fixed cost f charged to both legs plus a taker fee t, with
F = 2f + t exactly. Traders are friction-aware and role-aware: a resting
quote budgets for f, a crossing quote for f + t, so no fill is ever at a
loss net of friction. A buyer (value v) and seller (cost c) can trade iff

    v - c >= F

so pairs drop out in increasing order of their gap, and no trade is
individually rational once F > max(values) - min(costs), the analytic bound.
With unit traders every fill is a distinct pair, so realized volume is
bounded by `max_volume(F)`, the largest matching of pairs with gap >= F.
`find_coase_threshold` locates the smallest F at which mean volume falls
below `fraction` x the frictionless volume: a coarse grid scan, then integer
bisection inside the bracket, every probe on the same per-replicate seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..agents import ZICTrader
from ..agents.base import Cancel, MarketView, Submit
from ..analysis.welfare import competitive_equilibrium, pnl_by_account, welfare
from ..core.runner import MarketRunner
from ..core.types import Side
from ..costs import FixedCost, FrictionStack
from ..costs.base import TAKER_ONLY, CostModel, Fill
from .base import ExperimentResult, run_experiment

KEY = "friction"


@dataclass(frozen=True)
class Config:
    values: Tuple[int, ...] = (160, 150, 140, 130, 120, 110)   # buyer redemption values
    costs: Tuple[int, ...] = (60, 70, 80, 90, 100, 110)        # seller unit costs
    price_range: Tuple[int, int] = (1, 400)
    taker_share: float = 0.5        # share of F levied as the taker fee
    max_wakeups: int = 4_000

    @property
    def analytic_bound(self) -> int:
        """Largest valuation-cost gap; above it no trade is individually rational."""
        return max(self.values) - min(self.costs)


def split_friction(level: int, taker_share: float = 0.5) -> Tuple[int, int]:
    """(fixed per leg, taker fee) in ticks with 2 * fixed + taker == level."""
    if int(level) != level or level < 0:
        raise ValueError(f"friction level must be a non-negative integer, got {level}")
    fixed = (int(level) - int(round(level * taker_share))) // 2
    return fixed, int(level) - 2 * fixed


class TakerTickFee(CostModel):
    """Flat taker fee in ticks per lot (integer, so F stays exact)."""
    name = "taker_ticks"

    def __init__(self, ticks: int) -> None:
        super().__init__(TAKER_ONLY)
        self.ticks = Fraction(int(ticks))

    def raw_cost(self, fill: Fill) -> Fraction:
        return self.ticks * fill.qty


def friction_stack(level: int, taker_share: float = 0.5) -> FrictionStack:
    fixed, taker = split_friction(level, taker_share)
    return FrictionStack([FixedCost(fixed), TakerTickFee(taker)])


class CoaseZIC(ZICTrader):
    """ZI-C whose budget is net of the friction its role would pay."""

    def __init__(self, agent_id: int, side: Side, rng, fixed: int, taker: int, **kw) -> None:
        super().__init__(agent_id, side, rng, **kw)
        self.fixed, self.taker = fixed, taker

    def decide(self, view: MarketView):
        out = [Cancel(o) for o in view.open_orders]
        if self.exhausted:
            return out
        lo, hi = self.price_range
        v, f, t, buy = self.unit_value, self.fixed, self.taker, self.side is Side.BUY
        lo, hi = (lo, v - f) if buy else (v + f, hi)
        if lo > hi:
            return out                         # priced out even as a maker
        p = int(self.rng.integers(lo, hi + 1))
        touch = view.best_ask if buy else view.best_bid
        if touch is not None and (p >= touch if buy else p <= touch):
            if (touch + f + t > v) if buy else (touch - f - t < v):
                p = touch - 1 if buy else touch + 1   # rest instead of a losing take
                if not lo <= p <= hi:
                    return out
        self.quotes.append(p)
        return out + [Submit(self.side, 1, p)]


def build(cfg: Config, params: Dict[str, Any], seed: int) -> MarketRunner:
    """Sweep factory: the fixed schedule under friction level params['friction']."""
    level = int(params[KEY])
    fixed, taker = split_friction(level, cfg.taker_share)
    sides = [(Side.BUY, v) for v in cfg.values] + [(Side.SELL, c) for c in cfg.costs]
    streams = np.random.SeedSequence(seed).spawn(len(sides))
    agents = [CoaseZIC(i + 1, s, np.random.default_rng(ss), fixed, taker, schedule=[u],
                       price_range=cfg.price_range, value_range=cfg.price_range)
              for i, ((s, u), ss) in enumerate(zip(sides, streams))]
    endow = {a.agent_id: (0, 1 if a.side is Side.SELL else 0) for a in agents}
    return MarketRunner(agents, friction_stack(level, cfg.taker_share), seed=seed,
                        endowments=endow)


def no_rational_trade(runner: MarketRunner) -> bool:
    """Stop rule: no remaining buyer/seller pair has a gap >= F."""
    bids, asks, level = [], [], 0
    for a in runner.agents.values():
        level = 2 * a.fixed + a.taker
        held = runner.ledger[a.agent_id].position
        if a.side is Side.BUY and held == 0:
            bids.append(a.schedule[0])
        elif a.side is Side.SELL and held == 1:
            asks.append(a.schedule[0])
    return not bids or not asks or max(bids) - min(asks) < level


def max_volume(values: Sequence[int], costs: Sequence[int], level: int) -> int:
    """Largest k such that k buyers and k sellers pair up with every gap >= level.

    The best k-matching pairs the top-k values with the bottom-k costs in the
    same order (largest value with largest selected cost), which maximizes
    the smallest gap; feasibility is monotone in k.
    """
    d, s = sorted(values, reverse=True), sorted(costs)
    ok = lambda k: all(d[i] - s[k - 1 - i] >= level for i in range(k))
    return max((k for k in range(min(len(d), len(s)) + 1) if ok(k)), default=0)


def coase_metrics(res) -> Dict[str, float]:
    a0 = next(iter(res.agents.values()))
    level = 2 * a0.fixed + a0.taker
    w, volume = welfare(res), sum(t.qty for t in res.trades)
    mark = (w.equilibrium.price_lo + w.equilibrium.price_hi) // 2
    pop = list(res.agents.values())
    cap = max_volume([a.schedule[0] for a in pop if a.side is Side.BUY],
                     [a.schedule[0] for a in pop if a.side is Side.SELL], level)
    return {"volume": volume, "max_volume": cap, "surplus_share": w.net_efficiency, "gross_share": w.efficiency,
            "friction_collected": res.fees, "friction_residual": res.fees - level * volume,
            "violations": len(res.ledger.violations()),
            "zero_sum_residual": sum(pnl_by_account(res, mark).values()), "wakeups": res.wakeups}


def default_levels(cfg: Config = Config(), n: int = 6) -> List[int]:
    """n + 1 evenly spaced integer levels from 0 to one step past the bound."""
    b = cfg.analytic_bound
    step = max(1, -(-b // n))
    return list(range(0, b + step + 1, step))


def run_coase(cfg: Config = Config(), levels: Optional[Sequence[int]] = None, *,
              replicates: int = 4, master_seed: int = 28, n_workers: int = 1) -> ExperimentResult:
    levels = list(default_levels(cfg) if levels is None else levels)
    ce = competitive_equilibrium(cfg.values, cfg.costs)

    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        vol = {lv: float(np.mean([r["volume"] for r in rows if r[KEY] == lv])) for lv in levels}
        return {"analytic_bound": cfg.analytic_bound, "ce_quantity": ce.quantity,
                "ce_max_surplus": ce.max_surplus, "frictionless_volume": vol.get(0),
                "first_zero_volume_level": next((lv for lv in levels if vol[lv] == 0), None)}

    return run_experiment("coase_threshold", partial(build, cfg), coase_metrics, {KEY: levels},
                          replicates=replicates, master_seed=master_seed, crn=True,
                          run_kwargs={"max_wakeups": cfg.max_wakeups,
                                      "stop_when": no_rational_trade},
                          n_workers=n_workers, findings=headline)


@dataclass(frozen=True)
class CoaseThreshold:
    threshold: int                  # smallest F with mean volume < target
    analytic_bound: int
    baseline_volume: float
    target: float
    bracket: Tuple[int, int]        # (last ok, first collapsed) from the grid scan
    probes: Dict[int, float] = field(repr=False)   # F -> mean volume


def find_coase_threshold(cfg: Config = Config(), fraction: float = 0.5, *,
                         grid: Optional[Sequence[int]] = None, replicates: int = 4,
                         master_seed: int = 28, n_workers: int = 1) -> CoaseThreshold:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    """Smallest F with mean volume < fraction x volume(F=0), or zero volume.

    At F = bound only the widest pair can trade (volume <= 1), so the result
    is <= bound whenever fraction x baseline > 1 and <= bound + 1 always.
    """
    probes: Dict[int, float] = {}

    def volume(levels: Sequence[int]) -> None:
        todo = sorted({int(x) for x in levels} - set(probes))
        if todo:
            res = run_coase(cfg, todo, replicates=replicates, master_seed=master_seed,
                            n_workers=n_workers)
            probes.update(zip(todo, res.column("volume")))

    bound = cfg.analytic_bound
    scan = sorted({0, bound + 1, *(int(g) for g in (grid or default_levels(cfg)))})
    volume(scan)
    target = fraction * probes[0]
    collapsed = lambda lv: probes[lv] < target or probes[lv] == 0
    if collapsed(0):
        return CoaseThreshold(0, bound, probes[0], target, (0, 0), probes)
    hi = next(lv for lv in scan if collapsed(lv))
    lo = max(lv for lv in scan if lv < hi)
    bracket = (lo, hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        volume([mid])
        lo, hi = (lo, mid) if collapsed(mid) else (mid, hi)
    return CoaseThreshold(hi, bound, probes[0], target, bracket, dict(sorted(probes.items())))
