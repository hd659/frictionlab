"""Reputation experiment (fl-059): can public ratings replace fl-056 certification?

The fl-058 `ReputationMarket` is run through the fl-023 engine with the quality
schedule, horizon T, buyer premium and opportunist share held fixed, sweeping
rating_accuracy_bps x whitewash_cost on common random numbers (every cell sees the
same sellers, the same types and the same rating streams). Buyers' prior that a
fresh id is honest equals the true honest share.

An opportunist with saving s = q - f and a constant per-round reputation premium
gap g cheats from the end-game round

    t* = max(0, T - 1 - (s - 1) // g)        (`milking_switch_round`)

and any seller whitewashes iff p_fresh - c > p_now. Since p_now >= p_floor,
re-entry is never profitable once c >= p_fresh - p_floor = (m - f) * prior // 10000
(`min_deterrent_whitewash_cost`, m the fl-055 pooled mean quality, f the floor).
Every trade creates exactly `premium` of surplus (buyer value of the delivered
quality minus its cost), so realized surplus is premium x trades minus whitewash
cost, measured against the fl-022 full-information maximum over T rounds and
compared with the matching fl-056 pooled lemons cell (certification disabled).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..agents.reputation_seller import switch_round
from ..analysis.sweep import SweepSpec
from ..analysis.welfare import competitive_equilibrium
from ..markets import ReputationMarket
from ..markets.reputation import BPS, score_quality
from . import lemons
from .base import ExperimentResult, run_experiment

ACC, WASH = "rating_accuracy_bps", "whitewash_cost"
ACCURACIES, WASH_COSTS = (6000, 7000, 8500, 9500, 10000), (0, 1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class Config:
    """Fixed market: schedule, sellers, horizon, buyer premium, opportunist share (bps)."""
    schedule: Tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    n_sellers: int = 40
    horizon: int = 8
    premium: int = 45
    opportunist_bps: int = 2500


def opportunist_ids(n: int, share_bps: int) -> List[int]:
    """n * share // 10000 evenly spaced seller ids (deterministic, no RNG)."""
    return [i + 1 for i in range(n) if (i + 1) * share_bps // BPS > i * share_bps // BPS]


def milking_switch_round(horizon: int, cheat_saving: int, reputation_premium_per_round: int) -> int:
    """First cheating round of an opportunist; `horizon` means it never cheats in-horizon."""
    t = switch_round(horizon, cheat_saving, reputation_premium_per_round)
    return int(horizon) if t is None else t


def min_deterrent_whitewash_cost(prior: int, buyer_premium: int, quality_schedule: Sequence[int],
                                 floor: Optional[int] = None) -> int:
    """Smallest whitewash cost at which a fresh id never beats any score: p_fresh - p_floor."""
    qs = [int(q) for q in quality_schedule]
    f = min(qs) if floor is None else int(floor)
    m = lemons.akerlof_pooled_price(qs, buyer_premium)[0] - int(buyer_premium)
    return score_quality(0, 0, m, f, BPS, int(prior)) - f


def predicted_milking_round(market: ReputationMarket, seller_id: int) -> int:
    """Analytic t* at perfect accuracy: fresh-score gap in round 0, established gap after."""
    s, T = market.sellers[seller_id], market.rounds
    saving, g0 = s.saving(market.floor), market.fresh_price - market.floor_price
    if (T - 1) * g0 < saving or s.quality > market.fresh_price:
        return milking_switch_round(T, saving, g0)        # cheats at once, or never rated honest
    return milking_switch_round(T, saving, market.price_for((1, 0)) - market.floor_price)


class ReputationCell:
    """One seeded market plus its fl-056 pooled lemons twin on the same qualities."""

    def __init__(self, market: ReputationMarket, twin: "lemons.LemonsCell") -> None:
        self.market, self.twin = market, twin

    def run(self) -> "ReputationCell":
        self.market.run()
        self.twin.run()
        return self


def build(cfg: Config, params: Dict[str, Any], seed: int) -> ReputationCell:
    """fl-023 factory: same seeded qualities and types in every cell."""
    prior = BPS - cfg.opportunist_bps
    mkt = ReputationMarket.from_schedule(
        cfg.n_sellers, cfg.schedule, seed=seed, opportunists=opportunist_ids(cfg.n_sellers, cfg.opportunist_bps),
        rounds=cfg.horizon, premium=cfg.premium, rating_accuracy_bps=int(params[ACC]),
        whitewash_cost=int(params[WASH]), prior_honest_bps=prior)
    twin = lemons.build(lemons.Config(cfg.schedule, cfg.n_sellers, certification=False),
                        {lemons.COST: 0, lemons.PREMIUM: cfg.premium}, seed)
    return ReputationCell(mkt, twin)


def reputation_metrics(cell: ReputationCell) -> Dict[str, Any]:
    """Honest delivery, cheating, milking, whitewashing, price-quality gap and welfare."""
    m, T = cell.market, cell.market.rounds
    opp = [s for s in m.sellers.values() if s.opportunist]
    n_tr = len(m.trades)
    cheats = sum(m.delivered[t.trade_id] < m.sellers[t.seller_id].quality for t in m.trades)
    qs = [s.quality for s in m.sellers.values()]
    ce = competitive_equilibrium([q + m.premium for q in qs], qs)
    realized = sum(m.buyer_value(m.delivered[t.trade_id]) - m.delivered[t.trade_id]
                   for t in m.trades) - m.whitewash_costs
    share = realized / (T * ce.max_surplus) if ce.max_surplus else 0.0
    lemons_share = lemons.lemons_metrics(cell.twin)["surplus_share"]
    return {"trades": n_tr, "true_quality_share": (n_tr - cheats) / n_tr if n_tr else 1.0,
            "cheats": cheats, "cheating_rate": cheats / (len(opp) * T) if opp else 0.0,
            "mean_milking_round": sum(T if s.milking_round is None else s.milking_round
                                      for s in opp) / len(opp) if opp else float(T),
            "whitewashes": sum(s.whitewashes for s in m.sellers.values()),
            "price_quality_gap": sum(abs(t.price - m.buyer_value(m.delivered[t.trade_id]))
                                     for t in m.trades) / n_tr if n_tr else 0.0,
            "surplus_share": share, "whitewash_paid": m.whitewash_costs,
            "lemons_surplus_share": lemons_share, "surplus_gain_vs_lemons": share - lemons_share,
            "violations": len(m.ledger.violations())}


def run_reputation_grid(cfg: Config = Config(), accuracies: Sequence[int] = ACCURACIES,
                        whitewash_costs: Sequence[int] = WASH_COSTS, *, replicates: int = 1,
                        master_seed: int = 59, n_workers: int = 1) -> ExperimentResult:
    """Sweep rating_accuracy_bps x whitewash_cost through the fl-023 engine (CRN)."""
    accs, costs = [int(a) for a in accuracies], [int(c) for c in whitewash_costs]

    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        at = lambda a, k: max(r[k] for r in rows if r[ACC] == a)
        return {"lowest_accuracy_cheating_rate": at(accs[0], "cheating_rate"),
                "highest_accuracy_cheating_rate": at(accs[-1], "cheating_rate"),
                "max_surplus_share": max(r["surplus_share"] for r in rows),
                "max_surplus_gain_vs_lemons": max(r["surplus_gain_vs_lemons"] for r in rows),
                "total_whitewash_paid": sum(r["whitewash_paid"] for r in rows)}

    return run_experiment("reputation", partial(build, cfg), reputation_metrics, {ACC: accs, WASH: costs},
                          replicates=replicates, master_seed=master_seed, crn=True,
                          n_workers=n_workers, findings=headline)


@dataclass(frozen=True)
class ReputationThreshold:
    """Smallest whitewash cost with zero whitewashes at `accuracy`, with the analytic bound."""
    threshold: int
    analytic: int
    accuracy: int
    probes: Dict[int, int] = field(repr=False)


def cell_qualities(cfg: Config, master_seed: int = 59) -> List[int]:
    """The seeded quality draw every CRN cell (replicate 0) sees under `master_seed`."""
    seed = SweepSpec(build, reputation_metrics, grid={ACC: [BPS]}, master_seed=master_seed).child_seeds()[0][0]
    return [s.quality for s in build(cfg, {ACC: BPS, WASH: 0}, seed).market.sellers.values()]


def deterrent_bound(cfg: Config, master_seed: int = 59) -> int:
    return min_deterrent_whitewash_cost(BPS - cfg.opportunist_bps, cfg.premium,
                                        cell_qualities(cfg, master_seed), floor=min(cfg.schedule))


def find_reputation_threshold(cfg: Config = Config(), accuracy: int = 8500, *,
                              master_seed: int = 59) -> ReputationThreshold:
    """Integer bisection on [0, bound]; every probe reuses the same master seed."""
    bound, probes = deterrent_bound(cfg, master_seed), {}

    def washes(c: int) -> int:
        if c not in probes:
            probes[c] = run_reputation_grid(cfg, [accuracy], [c], master_seed=master_seed).rows[0]["whitewashes"]
        return probes[c]

    lo, hi = 0, bound                     # washes(bound) == 0 by the deterrent bound
    if washes(0) == 0:
        hi = 0
    while hi - lo > 1:                    # invariant: washes(lo) > 0 == washes(hi)
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if washes(mid) == 0 else (mid, hi)
    return ReputationThreshold(hi, bound, int(accuracy), dict(sorted(probes.items())))


def run_reputation(cfg: Config = Config(), accuracies: Sequence[int] = ACCURACIES,
                   whitewash_costs: Sequence[int] = WASH_COSTS, *, threshold_accuracy: int = 8500,
                   replicates: int = 1, master_seed: int = 59, n_workers: int = 1) -> ExperimentResult:
    """Grid + analytic milking round and deterrent bound + bisected whitewash threshold."""
    res = run_reputation_grid(cfg, accuracies, whitewash_costs, replicates=replicates,
                              master_seed=master_seed, n_workers=n_workers)
    th = find_reputation_threshold(cfg, threshold_accuracy, master_seed=master_seed)
    res.findings.update({"threshold_accuracy": int(threshold_accuracy),
                         "whitewash_threshold": th.threshold, "deterrent_bound": th.analytic})
    return res
