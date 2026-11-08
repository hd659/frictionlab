"""Switching-cost experiment (fl-062): how much market power does lock-in buy?

The fl-061 `SwitchingMarket` is run through the fl-023 engine with horizon T,
the valuation schedule and marginal cost c held fixed, sweeping
mean_switching_cost x target_share on common random numbers (every cell sees
the same per-consumer cost stream). Both sellers are `LockinSeller`s that
acquire at c and harvest once their share reaches the target. Consumer
switching costs come from the mean-preserving schedule (m - k, m, m + k) with
k = min(spread, m); spread = 0 is the homogeneous case.

A locked-in seller facing a rival priced at c can sustain (Klemperer)

    markup = max(0, min(s_min, v_min - c))          (`klemperer_harvest_markup`)

and a seller harvesting that markup for T - 1 rounds on `target` bps of the
market will rationally cut its round-0 price by at most

    d = (T - 1) * markup * target // 10000          (`acquisition_discount`)

The harvest price is switch-proof, so every buyer keeps buying: lock-in moves
surplus from consumers to the seller and the only deadweight loss against the
zero-switching-cost Bertrand twin (fl-022 welfare) is the switching cost paid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..agents.lockin_seller import ACQUIRE, HARVEST, LockinSeller
from ..analysis.pnl import consumer_surplus, switching_pnl
from ..analysis.welfare import competitive_equilibrium
from ..markets import SwitchingMarket
from ..markets.switching import BPS
from .base import ExperimentResult, run_experiment

MEAN, TARGET = "mean_switching_cost", "target_share"
COSTS, TARGETS = (0, 1, 2, 4, 8, 16, 32, 48), (6000, 8000, 9000)


@dataclass(frozen=True)
class Config:
    """Fixed market: valuation schedule, consumers, horizon, marginal cost, cost spread."""
    valuations: Tuple[int, ...] = (20, 70, 80, 90, 100, 110, 120, 130)
    n_consumers: int = 40
    horizon: int = 10
    marginal_cost: int = 40
    spread: int = 2


def klemperer_harvest_markup(min_switching_cost: int, valuation: int, marginal_cost: int) -> int:
    """Integer-tick markup over c a locked-in seller sustains against a rival at c."""
    return max(0, min(int(min_switching_cost), int(valuation) - int(marginal_cost)))


def acquisition_discount(horizon: int, harvest_markup: int, target_share: int) -> int:
    """Largest round-0 price cut worth paying to lock in `target_share` bps for T - 1 rounds."""
    return max(0, int(horizon) - 1) * max(0, int(harvest_markup)) * int(target_share) // BPS


def cost_schedule(mean: int, spread: int) -> Tuple[int, ...]:
    """Mean-preserving integer schedule (m - k, m, m + k), k = min(spread, m)."""
    k = min(int(spread), int(mean))
    return (mean - k, mean, mean + k)


def consumer_values(cfg: Config) -> List[int]:
    return [cfg.valuations[i % len(cfg.valuations)] for i in range(cfg.n_consumers)]


def buyer_floor(cfg: Config) -> int:
    """Lowest valuation that buys at c: the harvest cap of a seller holding every buyer."""
    return min(v for v in cfg.valuations if v >= cfg.marginal_cost)


class SwitchingCell:
    """One lock-in market plus its zero-switching-cost Bertrand twin."""

    def __init__(self, market: SwitchingMarket, twin: SwitchingMarket, mean: int, spread: int) -> None:
        self.market, self.twin, self.mean, self.spread = market, twin, mean, spread

    def run(self) -> "SwitchingCell":
        self.market.run()
        self.twin.run()
        return self


def build(cfg: Config, params: Dict[str, Any], seed: int) -> SwitchingCell:
    """fl-023 factory: two LockinSellers at target share; Bertrand twin on the same seed."""
    m, c, vals = int(params[MEAN]), cfg.marginal_cost, consumer_values(cfg)
    sellers = [LockinSeller(i, c, target_bps=int(params[TARGET])) for i in (1, 2)]
    mkt = SwitchingMarket(vals, sellers, switch_costs=cost_schedule(m, cfg.spread),
                          rounds=cfg.horizon, seed=seed)
    twin = SwitchingMarket(vals, [LockinSeller.bertrand_seller(i, c) for i in (1, 2)],
                           switch_costs=(0,), rounds=cfg.horizon, seed=seed)
    return SwitchingCell(mkt, twin, m, cfg.spread)


def _mean(xs: Sequence[int]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def switching_metrics(cell: SwitchingCell) -> Dict[str, Any]:
    """Prices by phase, markups, churn, shares, switching paid, surplus and welfare change."""
    m, tw = cell.market, cell.twin
    c, T = m.sellers[1].marginal_cost, m.rounds
    acq = [t.price for t in m.trades if m.phase_of[t.trade_id] == ACQUIRE]
    har = [t.price for t in m.trades if m.phase_of[t.trade_id] == HARVEST]
    cs = sum(r[2] for r in consumer_surplus(m).values())
    profit = sum(p.net for p in switching_pnl(m).values())
    cs_b = sum(r[2] for r in consumer_surplus(tw).values())
    welfare_b = cs_b + sum(p.net for p in switching_pnl(tw).values())
    vals = list(m.values.values())
    ce = competitive_equilibrium(vals, [c] * len(vals))
    harvester = max(m.sellers, key=lambda s: (m.share_bps(s), -s))
    own = m.customers(harvester)          # realized lock-in: min s and min v of its customers
    analytic = klemperer_harvest_markup(min(m.costs[k] for k in own), min(m.values[k] for k in own),
                                        c) if har else 0
    return {"trades": len(m.trades), "mean_price": _mean([t.price for t in m.trades]),
            "mean_markup": _mean([t.price - c for t in m.trades]),
            "acquisition_trades": len(acq), "harvest_trades": len(har),
            "acquisition_price": _mean(acq), "harvest_price": _mean(har),
            "harvest_markup": _mean([p - c for p in har]), "analytic_harvest_markup": analytic,
            "acquisition_discount": acquisition_discount(T, analytic, m.sellers[harvester].target_bps),
            "first_harvest_round": next((r.t for r in m.log if HARVEST in r.phases), T),
            "switches": m.switch_count, "churn_rate": m.switch_count / len(m.trades) if m.trades else 0.0,
            "share_1": m.share_bps(1), "share_2": m.share_bps(2),
            "switching_paid": m.switching_costs_paid, "consumer_surplus": cs, "seller_profit": profit,
            "surplus_share": (cs + profit) / (T * ce.max_surplus) if ce.max_surplus else 0.0,
            "cs_change_vs_bertrand": cs - cs_b, "welfare_change_vs_bertrand": cs + profit - welfare_b,
            "violations": len(m.ledger.violations()) + len(tw.ledger.violations())}


def run_switching_grid(cfg: Config = Config(), mean_costs: Sequence[int] = COSTS,
                       target_shares: Sequence[int] = TARGETS, *, replicates: int = 1,
                       master_seed: int = 62, n_workers: int = 1) -> ExperimentResult:
    """Sweep mean_switching_cost x target_share through the fl-023 engine (CRN)."""
    costs, targets = [int(x) for x in mean_costs], [int(x) for x in target_shares]

    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        locked = [r for r in rows if r["harvest_trades"]]
        return {"lockin_cells": len(locked), "max_harvest_markup": max(r["harvest_markup"] for r in rows),
                "max_cs_loss_vs_bertrand": -min(r["cs_change_vs_bertrand"] for r in rows),
                "min_welfare_change_vs_bertrand": min(r["welfare_change_vs_bertrand"] for r in rows),
                "total_switching_paid": sum(r["switching_paid"] for r in rows)}

    return run_experiment("switching_cost", partial(build, cfg), switching_metrics,
                          {MEAN: costs, TARGET: targets}, replicates=replicates,
                          master_seed=master_seed, crn=True, n_workers=n_workers, findings=headline)


@dataclass(frozen=True)
class LockinThreshold:
    """Smallest mean switching cost whose harvest markup exceeds `min_markup` ticks."""
    threshold: int
    analytic: int
    target: int
    min_markup: int
    probes: Dict[int, float] = field(repr=False)


def analytic_lockin_threshold(cfg: Config, min_markup: int) -> Optional[int]:
    """Smallest m with klemperer markup at s_min = m - min(spread, m) above `min_markup`."""
    v = buyer_floor(cfg)
    if klemperer_harvest_markup(v, v, cfg.marginal_cost) <= min_markup:
        return None                                   # the valuation cap blocks it
    return int(min_markup) + 1 + int(cfg.spread)


def find_lockin_threshold(cfg: Config = Config(), target: int = 6000, min_markup: int = 4, *,
                          master_seed: int = 62) -> LockinThreshold:
    """Integer bisection on [0, analytic]; every probe reuses the same master seed."""
    bound, probes = analytic_lockin_threshold(cfg, min_markup), {}
    if bound is None:
        raise ValueError(f"valuation cap keeps every markup <= {min_markup} ticks")

    def markup(m: int) -> float:
        if m not in probes:
            probes[m] = run_switching_grid(cfg, [m], [target], master_seed=master_seed).rows[0]["harvest_markup"]
        return probes[m]

    if markup(bound) <= min_markup:
        raise ValueError(f"no lock-in at target {target}: the seller never reaches its target share")
    lo, hi = 0, bound
    if markup(0) > min_markup:
        hi = 0
    while hi - lo > 1:                                # invariant: markup(lo) <= k < markup(hi)
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if markup(mid) > min_markup else (mid, hi)
    return LockinThreshold(hi, bound, int(target), int(min_markup), dict(sorted(probes.items())))


def run_switching_cost(cfg: Config = Config(), mean_costs: Sequence[int] = COSTS,
                       target_shares: Sequence[int] = TARGETS, *, threshold_target: int = 6000,
                       min_markup: int = 4, replicates: int = 1, master_seed: int = 62,
                       n_workers: int = 1) -> ExperimentResult:
    """Grid + analytic Klemperer markup + bisected lock-in threshold."""
    res = run_switching_grid(cfg, mean_costs, target_shares, replicates=replicates,
                             master_seed=master_seed, n_workers=n_workers)
    th = find_lockin_threshold(cfg, threshold_target, min_markup, master_seed=master_seed)
    res.findings.update({"threshold_target": th.target, "min_markup": th.min_markup,
                         "lockin_threshold": th.threshold, "analytic_lockin_threshold": th.analytic})
    return res
