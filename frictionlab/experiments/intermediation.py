"""Intermediation: when is a middleman's spread worth the search time it saves?

The fl-046 `SearchMarket` gets an fl-049 `Middleman` quoting bid = P - h and ask =
P + h around the competitive mid P of the unit schedule (h = middleman half-spread).
As in fl-047, search cost is a flow: reaching a match through direct search takes
`mpm` meetings at rate `meeting_rate` (lambda_d), the middleman one meeting at rate
`middleman_meeting_rate` (lambda_m), so waiting for the middleman saves

    saving = search_cost * (mpm / lambda_d - 1 / lambda_m)

and a trader weakly prefers the middleman iff h <= floor(saving)
(`middleman_indifference_spread`). Trader i has mpm_i = n / m_i, m_i the number of
the n counterparties it could trade with directly. Its fixed reservation is P plus
its own indifference spread (capped at value) for a buyer, P minus it (floored at
cost) for a seller: it fills with the middleman exactly when h is within its
spread, and costlier search makes it accept worse direct prices too. The fl-023
engine sweeps search_cost x middleman_half_spread; each cell is also run without
the middleman on the same seed, giving the surplus change of intermediation.
`find_intermediation_threshold` bisects for the smallest search cost at which the
intermediated share exceeds a fraction, against `analytic_bound`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from math import ceil, floor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..agents.middleman import DECAY_SINK_ID, HOLD_SINK_ID, Middleman
from ..analysis.pnl import middleman_pnl
from ..analysis.welfare import competitive_equilibrium
from ..core.types import Side
from ..markets import ReservationPolicy, SearchAgent, SearchMarket
from ..markets.search_market import SEARCH_SINK_ID
from .base import ExperimentResult, run_experiment

COST, SPREAD = "search_cost", "middleman_half_spread"
FIRST_BUYER, FIRST_SELLER, MM_ID = 1, 1_001, 5_000


@dataclass(frozen=True)
class Config:
    """Unit schedules, fixed meeting rates, middleman capacity / holding / decay, horizon."""
    values: Tuple[int, ...] = (130, 125, 120, 115, 110, 105)
    costs: Tuple[int, ...] = (40, 45, 50, 55, 60, 65)
    meeting_rate: float = 0.5
    middleman_meeting_rate: float = 2.0
    capacity: int = 2
    holding_cost: int = 1
    decay: int = 1
    floor: int = 20
    buyer_weight_bps: int = 5_000
    until: float = 200.0

    @property
    def mid(self) -> int:
        """Integer mid of the competitive-equilibrium price band: the middleman's quote centre."""
        ce = competitive_equilibrium(self.values, self.costs)
        return (ce.price_lo + ce.price_hi) // 2


def middleman_indifference_spread(search_cost: int, direct_meeting_rate: float,
                                  middleman_meeting_rate: float,
                                  meetings_per_match: Fraction = Fraction(1)) -> int:
    """Largest integer half-spread at which waiting for the middleman weakly beats direct search.

    floor(search_cost * (meetings_per_match / lambda_d - 1 / lambda_m)), clamped at 0:
    zero at zero search cost, non-decreasing in search cost; exact rational arithmetic.
    """
    if search_cost < 0 or not direct_meeting_rate > 0 or not middleman_meeting_rate > 0:
        raise ValueError("need search_cost >= 0 and positive meeting rates")
    saving = int(search_cost) * (Fraction(meetings_per_match) / Fraction(direct_meeting_rate)
                                 - 1 / Fraction(middleman_meeting_rate))
    return max(0, floor(saving))


def analytic_bound(cfg: Config, half_spread: int) -> Optional[int]:
    """Smallest search cost at which a trader whose every meeting can trade prefers the middleman."""
    gap = 1 / Fraction(cfg.meeting_rate) - 1 / Fraction(cfg.middleman_meeting_rate)
    if half_spread <= 0:
        return 0
    return ceil(Fraction(int(half_spread)) / gap) if gap > 0 else None


def traders(cfg: Config, search_cost: int) -> List[SearchAgent]:
    """Unit traders whose fixed reservations embed their own indifference spread."""
    p, out = cfg.mid, []
    plan = ((Side.BUY, cfg.values, cfg.costs, FIRST_BUYER), (Side.SELL, cfg.costs, cfg.values, FIRST_SELLER))
    for side, own, other, first in plan:
        for i, v in enumerate(own):
            m = sum(1 for x in other if (x <= v if side is Side.BUY else x >= v))
            if not m:
                pol = ReservationPolicy.fixed()        # no direct counterparty: truthful
            else:
                h = middleman_indifference_spread(search_cost, cfg.meeting_rate,
                                                  cfg.middleman_meeting_rate, Fraction(len(other), m))
                pol = ReservationPolicy.fixed(p + h if side is Side.BUY else max(1, p - h))
            out.append(SearchAgent(first + i, side, int(v), pol))
    return out


def market(cfg: Config, search_cost: int, half_spread: Optional[int], seed: int) -> SearchMarket:
    """One seeded search market; `half_spread=None` drops the middleman (the direct twin)."""
    mm = None if half_spread is None else Middleman(
        MM_ID, max(1, cfg.mid - half_spread), cfg.mid + half_spread, cfg.capacity,
        holding_cost=cfg.holding_cost, decay=cfg.decay, floor=cfg.floor)
    return SearchMarket(traders(cfg, search_cost), meeting_rate=cfg.meeting_rate,
                        search_cost=search_cost, buyer_weight_bps=cfg.buyer_weight_bps, seed=seed,
                        middleman=mm, middleman_meeting_rate=cfg.middleman_meeting_rate)


class IntermediationCell:
    """A sweep cell: the market with the middleman and its no-middleman twin on the same seed."""

    def __init__(self, market: SearchMarket, twin: SearchMarket) -> None:
        self.market, self.twin, self.until = market, twin, 0.0

    def run(self, until: float) -> "IntermediationCell":
        """Run both markets to `until` (or exhaustion) and return self for the metrics."""
        self.market.run(until)
        self.twin.run(until)
        self.until = float(until)
        return self


def build(cfg: Config, params: Dict[str, Any], seed: int) -> IntermediationCell:
    """Factory for the fl-023 engine: one (search_cost, half_spread) cell."""
    c, h = int(params[COST]), int(params[SPREAD])
    return IntermediationCell(market(cfg, c, h, seed), market(cfg, c, None, seed))


def outcome(m: SearchMarket, until: float) -> Tuple[int, int, int, float]:
    """(units delivered to buyers, of which via the middleman, realized surplus, mean time to match)."""
    mid = m.middleman.agent_id if m.middleman is not None else None
    direct = [t for t in m.trades if mid not in (t.buyer_id, t.seller_id)]
    buys = [t for t in m.trades if t.buyer_id == mid]
    sells = [t for t in m.trades if t.seller_id == mid]
    val = {i: a.value for i, a in m.agents.items()}
    served = [t.buyer_id for t in direct + sells]
    delivered = [t.seller_id for t in direct + buys[:len(sells)]]   # FIFO inventory
    surplus = sum(val[i] for i in served) - sum(val[i] for i in delivered)
    when = {i: t.time for t in m.trades for i in (t.buyer_id, t.seller_id)}
    wait = [when.get(i, until) for i in m.agents]
    return len(served), len(sells), surplus, sum(wait) / len(wait)


def intermediation_metrics(cfg: Config, cell: IntermediationCell) -> Dict[str, Any]:
    """Per-run shares, volume, timing, welfare, middleman PnL split, search cost, ledger residuals."""
    m, mm, led = cell.market, cell.market.middleman, cell.market.ledger
    vol, via, surplus, wait = outcome(m, cell.until)
    d_vol, _, d_surplus, d_wait = outcome(cell.twin, cell.until)
    pnl, best = middleman_pnl(m), competitive_equilibrium(cfg.values, cfg.costs).max_surplus
    sellers = sum(1 for a in m.agents.values() if a.side is Side.SELL)
    return {"volume": vol, "intermediated": via, "intermediated_share": via / vol if vol else 0.0,
            "mm_bought": pnl.bought, "mm_held": mm.units, "mean_time_to_match": wait,
            "surplus": surplus, "surplus_share": surplus / best if best else 0.0,
            "mm_spread_capture": pnl.spread_capture, "mm_holding_cost": pnl.holding_cost,
            "mm_decay_loss": pnl.decay_loss, "mm_net_pnl": pnl.net,
            "search_cost_paid": m.search_costs, "direct_volume": d_vol,
            "direct_time_to_match": d_wait, "direct_surplus": d_surplus,
            "surplus_change": surplus - d_surplus, "net_surplus_change": surplus - d_surplus
            - m.search_costs + cell.twin.search_costs,
            "sink_residual": abs(led[SEARCH_SINK_ID].cash - m.search_costs)
            + abs(led[HOLD_SINK_ID].cash - mm.holding_paid) + abs(led[DECAY_SINK_ID].cash - mm.decay_paid),
            "pnl_residual": abs(pnl.net - led[mm.agent_id].cash - pnl.held_basis),
            "position_residual": abs(sum(a.position for a in led) - sellers)
            + abs(led[mm.agent_id].position - mm.units),
            "violations": len(led.violations()) + len(cell.twin.ledger.violations())}


def run_intermediation(cfg: Config = Config(), search_costs: Sequence[int] = (0, 1, 2, 4, 8, 16, 32),
                       half_spreads: Sequence[int] = (0, 2, 4, 8, 16, 24), *, replicates: int = 1,
                       master_seed: int = 50, n_workers: int = 1) -> ExperimentResult:
    """Sweep search_cost x middleman_half_spread through the fl-023 engine (common random numbers)."""
    costs, spreads = [int(c) for c in search_costs], [int(h) for h in half_spreads]
    ce = competitive_equilibrium(cfg.values, cfg.costs)

    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def cell(c: int, h: int, k: str) -> float:
            vals = [r[k] for r in rows if r[COST] == c and r[SPREAD] == h]
            return sum(vals) / len(vals)
        out: Dict[str, Any] = {"ce_quantity": ce.quantity, "ce_max_surplus": ce.max_surplus,
                               "mid": cfg.mid, "indifference_spread_at_max_cost":
                               middleman_indifference_spread(costs[-1], cfg.meeting_rate,
                                                             cfg.middleman_meeting_rate),
                               "analytic_bound_widest": analytic_bound(cfg, spreads[-1])}
        for tag, c in (("baseline", costs[0]), ("costliest", costs[-1])):
            for k in ("intermediated_share", "surplus_share", "mean_time_to_match", "surplus_change"):
                out[f"{tag}_{k}"] = cell(c, spreads[0], k)
        out["costliest_widest_mm_net_pnl"] = cell(costs[-1], spreads[-1], "mm_net_pnl")
        return out

    return run_experiment("intermediation", partial(build, cfg), partial(intermediation_metrics, cfg),
                          {COST: costs, SPREAD: spreads}, replicates=replicates,
                          master_seed=master_seed, crn=True, run_kwargs={"until": cfg.until},
                          n_workers=n_workers, findings=headline)


@dataclass(frozen=True)
class IntermediationThreshold:
    """Smallest search cost at which the intermediated share exceeds `fraction`, with the bound."""
    threshold: Optional[int]
    analytic_bound: Optional[int]
    half_spread: int
    fraction: float
    probes: Dict[int, float] = field(repr=False)

    @property
    def within_bound(self) -> bool:
        """True when the bisected threshold does not exceed the analytic bound."""
        return None not in (self.threshold, self.analytic_bound) and self.threshold <= self.analytic_bound


def find_intermediation_threshold(cfg: Config = Config(), half_spread: int = 4, fraction: float = 0.5,
                                  *, upper: Optional[int] = None, replicates: int = 1,
                                  master_seed: int = 50, n_workers: int = 1) -> IntermediationThreshold:
    """Integer bisection on [0, upper]; every probe reuses the same master seed."""
    if not 0 <= fraction < 1:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")
    bound, probes = analytic_bound(cfg, half_spread), {}
    upper = int(upper if upper is not None else max(8, 4 * bound) if bound is not None else 64)

    def share(c: int) -> float:
        if c not in probes:
            probes[c] = run_intermediation(cfg, [c], [half_spread], replicates=replicates,
                                           master_seed=master_seed,
                                           n_workers=n_workers).column("intermediated_share")[0]
        return probes[c]

    hi: Optional[int] = None
    if share(0) > fraction:
        hi = 0
    elif share(upper) > fraction:              # invariant: share(lo) <= fraction < share(hi)
        lo, hi = 0, upper
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if share(mid) > fraction else (mid, hi)
    return IntermediationThreshold(hi, bound, int(half_spread), float(fraction), dict(sorted(probes.items())))
