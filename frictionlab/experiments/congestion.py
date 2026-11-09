"""Congestion and optimal-toll experiment (fl-068): Naor (1969) over-joining on the fl-067 queue.

The fl-067 `QueueMarket` is run through the fl-023 engine with arrival / service
probabilities, reward R and horizon T fixed, sweeping toll x waiting_cost on common
random numbers (every cell sees the same arrival and service draws). Queue length
means customers waiting behind the one in service, so a customer who sees n in the
system and joins leaves a queue of exactly n.

Thresholds are "largest n at which an arrival joins" (-1: nobody joins). With
k = floor((R - toll) / c) the ceil join rule gives n_ind = floor(k * q / 10000) - 1.
For a policy "join iff n <= s" the start-of-step system size X is a birth-death
chain on 0 .. s + 1 with up a = p(1 - q), down d = (1 - p)q inside and q at the cap
(arrivals join before the same-step service draw), so pi(n) ~ a^n d^(s-n) q for
n <= s and pi(s + 1) ~ a^(s+1). Tolls are transfers, so welfare per step is
    W(s) = R * p * P(X <= s) - c * (E[X] + p * P(X <= s))
evaluated exactly in `Fraction`s. Self-interested arrivals ignore the delay they
impose on later ones, so the social threshold is <= the individual one and the
optimal toll is the smallest that closes the gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

from ..analysis.pnl import operator_pnl, queue_surplus
from ..config.schema import Param
from ..markets import QueueMarket
from .base import ExperimentResult, run_experiment

TOLL, WAIT, BPS = "toll", "waiting_cost", 10_000
TOLLS, WAITS = (0, 2, 4, 8, 12, 16, 24, 32), (1, 2, 3, 4)


@dataclass(frozen=True)
class Config:
    """Fixed queue: reward, arrival / service probabilities (bps) and horizon."""
    reward: int = 40
    arrival_prob_bps: int = 6000
    service_prob_bps: int = 5000
    horizon: int = 400


def naor_individual_threshold(reward: int, waiting_cost: int, service_prob_bps: int,
                              toll: int = 0) -> Optional[int]:
    """Largest queue length n at which an arrival still joins; -1 if none, None if c == 0."""
    slack = int(reward) - int(toll)
    if slack < 0:
        return -1
    if waiting_cost == 0:
        return None
    return (slack // int(waiting_cost)) * int(service_prob_bps) // BPS - 1


def welfare_per_step(reward: int, waiting_cost: int, arrival_prob_bps: int,
                     service_prob_bps: int, threshold: int) -> Fraction:
    """Exact long-run welfare per step under "join iff n <= threshold"."""
    if threshold < 0:
        return Fraction(0)
    p, q = Fraction(arrival_prob_bps, BPS), Fraction(service_prob_bps, BPS)
    a, d, s = p * (1 - q), (1 - p) * q, threshold
    w = [a ** n * d ** (s - n) * q for n in range(s + 1)] + [a ** (s + 1)]
    tot = sum(w)
    pi = [x / tot for x in w] if tot else [Fraction(1)] + [Fraction(0)] * (s + 1)
    join = p * sum(pi[:-1])
    return reward * join - waiting_cost * (sum(n * x for n, x in enumerate(pi)) + join)


def naor_social_threshold(reward: int, waiting_cost: int, arrival_prob_bps: int,
                          service_prob_bps: int) -> int:
    """Smallest threshold in -1 .. n_ind(toll=0) maximizing `welfare_per_step`."""
    if waiting_cost <= 0:
        raise ValueError("the social threshold needs a positive waiting cost")
    top = naor_individual_threshold(reward, waiting_cost, service_prob_bps)
    vals = [welfare_per_step(reward, waiting_cost, arrival_prob_bps, service_prob_bps, s)
            for s in range(-1, top + 1)]
    return vals.index(max(vals)) - 1


def naor_optimal_toll(reward: int, waiting_cost: int, arrival_prob_bps: int,
                      service_prob_bps: int) -> int:
    """Smallest integer toll whose individual threshold equals the social threshold."""
    s = naor_social_threshold(reward, waiting_cost, arrival_prob_bps, service_prob_bps)
    return next(t for t in range(int(reward) + 2)
                if naor_individual_threshold(reward, waiting_cost, service_prob_bps, t) <= s)


class QueueCell:
    """One seeded queue plus its zero-toll twin on the same draws."""

    def __init__(self, market: QueueMarket, twin: QueueMarket) -> None:
        self.market, self.twin = market, twin

    def run(self) -> "QueueCell":
        self.market.run()
        self.twin.run()
        return self


def build(cfg: Config, params: Dict[str, Any], seed: int) -> QueueCell:
    """fl-023 factory: queue at (toll, c) and its toll = 0 twin, both on `seed`."""
    kw = dict(reward=cfg.reward, arrival_prob_bps=cfg.arrival_prob_bps, waiting=int(params[WAIT]),
              service_prob_bps=cfg.service_prob_bps, steps=cfg.horizon, seed=seed)
    return QueueCell(QueueMarket(toll=int(params[TOLL]), **kw), QueueMarket(toll=0, **kw))


def welfare(m: QueueMarket) -> int:
    """fl-022 welfare: customer surplus plus toll revenue (tolls are transfers)."""
    return sum(s.net for s in queue_surplus(m).values()) + operator_pnl(m).net


def queue_metrics(cell: QueueCell) -> Dict[str, Any]:
    """Join / balk rates, queue lengths, time in system, tolls, waiting, surplus, welfare."""
    m, cc = cell.market, cell.market.congestion
    q = [max(n - 1, 0) for n in m.lengths]
    arr, joins = len(m.arrivals), m.joins
    cs, rev = sum(s.net for s in queue_surplus(m).values()), operator_pnl(m).net
    c = cc.wait_for(0)
    return {"arrivals": arr, "joins": joins, "join_rate": joins / arr if arr else 0.0,
            "balk_rate": (arr - joins) / arr if arr else 0.0, "served": m.served,
            "mean_queue_length": sum(q) / len(q), "max_queue_length": max(q),
            "mean_steps_in_system": cc.customer_steps / joins if joins else 0.0,
            "toll_revenue": rev, "waiting_paid": cc.waiting_paid, "customer_surplus": cs,
            "welfare": cs + rev, "welfare_change_vs_zero_toll": cs + rev - welfare(cell.twin),
            "individual_threshold": naor_individual_threshold(m.reward, c, m.service_bps,
                                                              cc.toll_for(0)),
            "violations": len(m.ledger.violations()) + len(cell.twin.ledger.violations())}


def run_queue_grid(cfg: Config = Config(), tolls: Sequence[int] = TOLLS,
                   waiting_costs: Sequence[int] = WAITS, *, replicates: int = 1,
                   master_seed: int = 68, n_workers: int = 1) -> ExperimentResult:
    """Sweep toll x waiting_cost through the fl-023 engine (CRN)."""
    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = max(rows, key=lambda r: (r["welfare_change_vs_zero_toll"], -r[TOLL]))
        return {"max_welfare_gain": best["welfare_change_vs_zero_toll"], "best_toll": best[TOLL],
                "best_toll_waiting_cost": best[WAIT],
                "total_toll_revenue": sum(r["toll_revenue"] for r in rows),
                "max_queue_length": max(r["max_queue_length"] for r in rows)}

    return run_experiment("congestion", partial(build, cfg), queue_metrics,
                          {TOLL: [int(x) for x in tolls], WAIT: [int(x) for x in waiting_costs]},
                          replicates=replicates, master_seed=master_seed, crn=True,
                          n_workers=n_workers, findings=headline)


@dataclass(frozen=True)
class CongestionToll:
    """Smallest toll whose simulated max queue length is <= the social threshold."""
    toll: int
    analytic: int
    social_threshold: int
    waiting_cost: int
    probes: Dict[int, int] = field(repr=False)


def find_congestion_toll(cfg: Config = Config(), waiting_cost: int = 2, *,
                         master_seed: int = 68) -> CongestionToll:
    """Integer bisection on toll in [0, R + 1]; every probe reuses the master seed."""
    s = naor_social_threshold(cfg.reward, waiting_cost, cfg.arrival_prob_bps, cfg.service_prob_bps)
    if s < 0:
        raise ValueError("social optimum admits nobody; no toll caps the queue at -1")
    probes: Dict[int, int] = {}

    def capped(t: int) -> bool:
        if t not in probes:
            row = run_queue_grid(cfg, [t], [waiting_cost], master_seed=master_seed).rows[0]
            probes[t] = row["max_queue_length"]
        return probes[t] <= s

    lo, hi = -1, cfg.reward + 1                       # fails at lo (virtual), capped at hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if capped(mid) else (mid, hi)
    opt = naor_optimal_toll(cfg.reward, waiting_cost, cfg.arrival_prob_bps, cfg.service_prob_bps)
    return CongestionToll(hi, opt, s, int(waiting_cost), dict(sorted(probes.items())))


def run_congestion(cfg: Config = Config(), tolls: Sequence[int] = TOLLS,
                   waiting_costs: Sequence[int] = WAITS, *, probe_waiting_cost: int = 2,
                   replicates: int = 1, master_seed: int = 68, n_workers: int = 1) -> ExperimentResult:
    """Grid + Naor thresholds + bisected congestion toll."""
    res = run_queue_grid(cfg, tolls, waiting_costs, replicates=replicates,
                         master_seed=master_seed, n_workers=n_workers)
    tol = find_congestion_toll(cfg, probe_waiting_cost, master_seed=master_seed)
    res.findings.update({"probe_waiting_cost": tol.waiting_cost, "social_threshold": tol.social_threshold,
                         "individual_threshold": naor_individual_threshold(
                             cfg.reward, tol.waiting_cost, cfg.service_prob_bps),
                         "bisected_toll": tol.toll, "naor_optimal_toll": tol.analytic})
    return res


# ---- fl-036 registry spec (kept here so the registry module stays under the line ceiling)
_POS = Param("int", min=1)
PARAM_SCHEMA: Dict[str, Param] = {
    "tolls": Param("int", seq=True, friction=True), "waiting_costs": Param("int", seq=True, friction=True),
    "reward": _POS, "arrival_prob_bps": Param("int", min=0, max=BPS),
    "service_prob_bps": Param("int", min=1, max=BPS), "horizon": _POS, "probe_waiting_cost": _POS}
SMALL: Dict[str, Any] = {"tolls": [0, 4, 8, 16], "waiting_costs": [1, 2, 4], "reward": 24,
                         "arrival_prob_bps": 7000, "service_prob_bps": 5000, "horizon": 200}


def defaults() -> Dict[str, Any]:
    """Registered defaults: `Config()` fields plus the grid and toll-probe settings."""
    return {"tolls": list(TOLLS), "waiting_costs": list(WAITS),
            **{f: getattr(Config(), f) for f in Config.__dataclass_fields__}, "probe_waiting_cost": 2}


def registry_runner(p: Dict[str, Any], master_seed: int, replicates: int, n_workers: int) -> ExperimentResult:
    """fl-036 runner: resolved params -> `run_congestion`."""
    cfg = Config(**{k: p[k] for k in Config.__dataclass_fields__})
    return run_congestion(cfg, p["tolls"], p["waiting_costs"], probe_waiting_cost=p["probe_waiting_cost"],
                          replicates=replicates, master_seed=master_seed, n_workers=n_workers)
