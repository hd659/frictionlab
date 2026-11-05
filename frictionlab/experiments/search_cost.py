"""Search costs and perishable value: how much trade survives costly bilateral search?

The fl-046 `SearchMarket` is populated from a unit valuation/cost schedule. Each
agent posts a search-inclusive reservation price (`reservation_price`): a buyer
bids the highest price p at which paying p plus the expected search bill of
finding a seller willing at p still does not exceed its value; a seller mirrors
this with an ask. The markup is fixed at entry and asks track the seller's
current (perishable) value, so decay lowers them. Agents whose reservation no
counterparty can meet stay out and pay nothing (perishable sellers stay in). The fl-023 engine sweeps
search_cost x decay_per_step; each cell reports volume, time to match, surplus
vs the fl-022 competitive maximum, search cost paid, value lost to decay and the
sellers' share of realized surplus.

Both markups are at least k = search_cost / meeting_rate, so a pair (v, s) can
trade only if v - s >= 2 k. `analytic_bound` is the smallest integer cost at
which no meeting can cover its cost; `find_search_collapse` bisects for the
smallest cost at which volume falls below a fraction of the zero-cost baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..analysis.welfare import competitive_equilibrium
from ..core.types import Side
from ..markets import ReservationPolicy, SearchAgent, SearchMarket
from ..markets.search_market import SEARCH_SINK_ID
from .base import ExperimentResult, run_experiment

COST, DECAY = "search_cost", "decay_per_step"
FIRST_BUYER, FIRST_SELLER = 1, 1_001


@dataclass(frozen=True)
class Config:
    """Search market: unit schedules, meeting rate, bargaining weight, value floor, horizon."""
    values: Tuple[int, ...] = (120, 110, 100, 90, 80, 70)
    costs: Tuple[int, ...] = (20, 30, 40, 50, 60, 70)
    meeting_rate: float = 1.0
    buyer_weight_bps: int = 5_000
    floor: int = 20
    until: float = 200.0


@dataclass(frozen=True)
class ShadedPolicy(ReservationPolicy):
    """Reservation = current value shaded by a fixed search markup (down for buyers, up for sellers)."""
    shade: int = 0

    def reservation(self, side: Side, value: int) -> int:
        """Shaded reservation of an agent whose current value is `value`."""
        return value - self.shade if side is Side.BUY else value + self.shade


def reservation_price(value: int, search_cost: int, meeting_rate: float,
                      counterparty_range: Tuple[int, int], side: Side = Side.BUY) -> int:
    """Stopping-rule reservation price in ticks against counterparties ~ U{lo..hi}.

    A meeting costs k = search_cost / meeting_rate. A buyer bidding p meets an
    acceptable seller with probability m(p) / n (m = #{x <= p}, n = hi - lo + 1),
    so it expects to pay k n / m(p) before it stops; its reservation is the largest
    p <= value with (value - p) m(p) >= k n. A seller mirrors this: the smallest
    ask a >= value with (a - value) m(a) >= k n, m = #{x >= a}. Exact integer
    arithmetic; zero cost returns `value`; lo - 1 (buyer) or hi + 1 (seller) when no
    price covers the expected search bill.
    """
    lo, hi = (int(x) for x in counterparty_range)
    if hi < lo or not meeting_rate > 0:
        raise ValueError(f"need lo <= hi and meeting_rate > 0, got {counterparty_range}, {meeting_rate}")
    k, n = Fraction(int(search_cost)) / Fraction(meeting_rate), hi - lo + 1
    if k == 0:
        return int(value)
    if side is Side.BUY:
        return next((p for p in range(value, lo - 1, -1)
                     if (value - p) * (min(p, hi) - lo + 1) >= k * n), lo - 1)
    return next((a for a in range(value, hi + 1)
                 if (a - value) * (hi - max(a, lo) + 1) >= k * n), hi + 1)


def analytic_bound(cfg: Config, decay: int = 0) -> int:
    """Smallest integer search cost at which no buyer-seller meeting can cover its cost."""
    low = min(*cfg.costs, cfg.floor) if decay else min(cfg.costs)
    slack = Fraction(max(cfg.values) - low) * Fraction(cfg.meeting_rate) / 2
    return slack.numerator // slack.denominator + 1


class SearchCell:
    """One sweep cell: the market plus initial values, so metrics can price decay and surplus."""

    def __init__(self, market: SearchMarket, start: Dict[int, int]) -> None:
        self.market, self.start, self.until = market, start, 0.0

    def run(self, until: float) -> "SearchCell":
        """Run the market to `until` (or exhaustion) and return self for the metrics."""
        self.market.run(until)
        self.until = float(until)
        return self


def build(cfg: Config, params: Dict[str, Any], seed: int) -> SearchCell:
    """Shade every agent's reservation for `search_cost`; non-viable agents stay out."""
    c, d = int(params[COST]), int(params[DECAY])
    agents: List[SearchAgent] = []
    plan = ((Side.BUY, cfg.values, (min(cfg.costs), max(cfg.costs)), FIRST_BUYER),
            (Side.SELL, cfg.costs, (min(cfg.values), max(cfg.values)), FIRST_SELLER))
    for side, vals, (lo, hi), first in plan:
        for i, v in enumerate(vals):
            r = reservation_price(v, c, cfg.meeting_rate, (lo, hi), side)
            if c and (r < lo if side is Side.BUY else (r > hi and not d)):
                continue                      # a perishable seller may become viable later
            pol = ShadedPolicy(shade=abs(v - r))
            agents.append(SearchAgent(first + i, side, v, pol, min(cfg.floor, v)))
    m = SearchMarket(agents, meeting_rate=cfg.meeting_rate, search_cost=c,
                     buyer_weight_bps=cfg.buyer_weight_bps, decay=d, seed=seed)
    return SearchCell(m, {a.agent_id: a.value for a in agents})


def search_metrics(cfg: Config, cell: SearchCell) -> Dict[str, Any]:
    """Per-run volume, timing, welfare, search and decay costs, split and ledger residuals."""
    m, v0 = cell.market, cell.start
    sold = {t.seller_id: t for t in m.trades}
    sellers = [a for a in m.agents.values() if a.side is Side.SELL]
    surplus = sum(v0[t.buyer_id] - v0[t.seller_id] for t in m.trades)
    seller_gain = sum(t.price - v0[t.seller_id] for t in m.trades)
    times = [sold[a.agent_id].time if a.agent_id in sold else cell.until for a in sellers]
    best = competitive_equilibrium(cfg.values, cfg.costs).max_surplus
    return {"volume": len(m.trades), "meetings": len(m.meetings), "participants": len(m.agents),
            "mean_time_to_match": sum(times) / len(times) if times else cell.until,
            "surplus": surplus, "surplus_share": surplus / best if best else 0.0,
            "search_cost_paid": m.search_costs, "net_surplus": surplus - m.search_costs,
            "decay_loss": sum(v0[a.agent_id] - a.value for a in sellers),
            "seller_share": seller_gain / surplus if surplus else 0.0,
            "sink_residual": abs(m.ledger[SEARCH_SINK_ID].cash - m.search_costs),
            "position_residual": abs(sum(a.position for a in m.ledger) - len(sellers)),
            "violations": len(m.ledger.violations())}


def run_search_cost(cfg: Config = Config(), search_costs: Sequence[int] = (0, 5, 10, 20, 30, 40, 52),
                    decays: Sequence[int] = (0, 1, 2, 4), *, replicates: int = 1,
                    master_seed: int = 47, n_workers: int = 1) -> ExperimentResult:
    """Sweep search_cost x decay_per_step through the fl-023 engine (common random numbers)."""
    costs, decs = [int(c) for c in search_costs], [int(d) for d in decays]
    ce = competitive_equilibrium(cfg.values, cfg.costs)

    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def cell(c: int, d: int, k: str) -> float:
            return float(np.mean([r[k] for r in rows if r[COST] == c and r[DECAY] == d]))
        out: Dict[str, Any] = {"ce_quantity": ce.quantity, "ce_max_surplus": ce.max_surplus,
                               "analytic_bound": analytic_bound(cfg)}
        for tag, c in (("baseline", costs[0]), ("costliest", costs[-1])):
            for k in ("volume", "surplus_share", "search_cost_paid"):
                out[f"{tag}_{k}"] = cell(c, decs[0], k)
        out["fastest_decay_time_to_match"] = cell(costs[0], decs[-1], "mean_time_to_match")
        return out

    return run_experiment("search_cost", partial(build, cfg), partial(search_metrics, cfg),
                          {COST: costs, DECAY: decs}, replicates=replicates,
                          master_seed=master_seed, crn=True, run_kwargs={"until": cfg.until},
                          n_workers=n_workers, findings=headline)


@dataclass(frozen=True)
class SearchCollapse:
    """Smallest search cost at which mean volume drops below `target`, with the analytic bound."""
    threshold: Optional[int]
    analytic_bound: int
    decay: int
    baseline_volume: float
    target: float
    probes: Dict[int, float] = field(repr=False)

    @property
    def within_bound(self) -> bool:
        """True when the bisected threshold does not exceed the analytic bound."""
        return self.threshold is not None and self.threshold <= self.analytic_bound


def find_search_collapse(cfg: Config = Config(), fraction: float = 0.5, decay: int = 0, *,
                         replicates: int = 1, master_seed: int = 47,
                         n_workers: int = 1) -> SearchCollapse:
    """Integer bisection on [0, analytic bound]; every probe reuses the same master seed."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    probes: Dict[int, float] = {}

    def vol(c: int) -> float:
        if c not in probes:
            probes[c] = run_search_cost(cfg, [c], [decay], replicates=replicates,
                                        master_seed=master_seed, n_workers=n_workers).column("volume")[0]
        return probes[c]

    base, bound = vol(0), analytic_bound(cfg, decay)
    target, hi = fraction * base, None
    if base > 0 and vol(bound) < target:          # bisect the integer bracket [0, bound]
        lo, hi = 0, bound
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if vol(mid) < target else (mid, hi)
    return SearchCollapse(hi, bound, decay, base, target, dict(sorted(probes.items())))
