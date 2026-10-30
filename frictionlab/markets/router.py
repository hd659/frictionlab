"""Deterministic multi-venue order router (smart order routing with frictions).

Each `RouteVenue` is a continuous `OrderBook` plus its own `Ledger`, whose
friction stack (e.g. a tiered `FeeComponent` + `FixedCost`) prices fills.

All-in cost of buying (selling) `q` lots at one venue, in integer tick-lots,
as the taker's cash outflow:
    all_in = side * q * best  +  slippage  +  fees
    slippage = side * (walked notional - q * best) >= 0
    fees     = sum over maker orders hit of stack.charge(taker fill)
The walk visits the same resting orders, in the same price-time order and with
the same self-trade handling, as `OrderBook.submit`, so `fees` includes the
per-trade fixed cost and the account's current fee tier exactly: the
prediction equals the realized cash change of executing the leg.

Policy: route the full quantity to the venue with minimum `all_in`, ties
broken by venue id. Split only when no single venue can fill the full size
(within the optional limit price): venues are then filled greedily by
all-in cost per lot of their available depth, ties by venue id, and the
total routed never exceeds the requested quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..agents.base import Submit
from ..core.ledger import Ledger
from ..core.order_book import OrderBook
from ..core.types import IdGen, Order, OrderType, Side, Trade, _check_int
from ..costs.base import Fill, Role


@dataclass
class RouteVenue:
    venue_id: str
    book: OrderBook
    ledger: Ledger
    order_ids: IdGen = field(default_factory=IdGen)

    def __post_init__(self) -> None:
        if not isinstance(self.venue_id, str) or not self.venue_id:
            raise TypeError("venue_id must be a non-empty str")

    def charge(self, fill: Fill) -> int:
        fr = self.ledger.frictions
        return 0 if fr is None else fr.charge(fill)


@dataclass(frozen=True)
class RouteLeg:
    venue_id: str
    side: Side
    qty: int
    best: int
    notional: int
    slippage: int
    fees: int
    n_fills: int

    @property
    def all_in(self) -> int:
        return self.side.value * self.notional + self.fees


@dataclass(frozen=True)
class RoutePlan:
    agent_id: int
    side: Side
    requested: int
    legs: Tuple[RouteLeg, ...]
    quotes: Dict[str, Optional[RouteLeg]]   # full-size cost per venue (None: no depth)

    @property
    def qty(self) -> int:
        return sum(l.qty for l in self.legs)

    @property
    def all_in(self) -> int:
        return sum(l.all_in for l in self.legs)

    @property
    def split(self) -> bool:
        return len(self.legs) > 1


def _walk(book: OrderBook, agent_id: int, side: Side, qty: int,
          limit: Optional[int]) -> Iterator[Tuple[int, int]]:
    """Yield (price, lots) per resting order a taker order would hit."""
    opp = book._halves[side.opposite]
    left = qty
    for price, _ in opp.levels():
        if limit is not None and (price > limit if side is Side.BUY else price < limit):
            return
        for o in opp.queues[price]:
            if not o.is_active:
                continue
            if o.agent_id == agent_id:
                if book.stp == "cancel_incoming":
                    return
                continue  # cancel_resting: own order is dropped, keep walking
            lots = min(left, o.remaining)
            yield price, lots
            left -= lots
            if not left:
                return


def quote_leg(venue: RouteVenue, agent_id: int, side: Side, qty: int,
              limit: Optional[int] = None) -> Optional[RouteLeg]:
    """Cost of taking up to `qty` lots at one venue (None if nothing fills)."""
    chunks = list(_walk(venue.book, agent_id, side, qty, limit))
    if not chunks:
        return None
    best, q = chunks[0][0], sum(l for _, l in chunks)
    notional = sum(p * l for p, l in chunks)
    fees = sum(venue.charge(Fill(agent_id, side, Role.TAKER, p, l)) for p, l in chunks)
    return RouteLeg(venue.venue_id, side, q, best, notional,
                    side.value * (notional - q * best), fees, len(chunks))


class VenueRouter:
    def __init__(self, venues: Sequence[RouteVenue]) -> None:
        ids = [v.venue_id for v in venues]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("router needs >= 1 venue with unique ids")
        self.venues: Dict[str, RouteVenue] = {v.venue_id: v for v in sorted(venues, key=lambda v: v.venue_id)}

    def plan(self, agent_id: int, intent: Submit) -> RoutePlan:
        side, qty = Side(intent.side), _check_int("qty", intent.qty, positive=True)
        quotes = {vid: quote_leg(v, agent_id, side, qty, intent.price)
                  for vid, v in self.venues.items()}
        full = [l for l in quotes.values() if l is not None and l.qty == qty]
        if full:
            legs: List[RouteLeg] = [min(full, key=lambda l: (l.all_in, l.venue_id))]
        else:
            avail = sorted((l for l in quotes.values() if l is not None),
                           key=lambda l: (Fraction(l.all_in, l.qty), l.venue_id))
            legs, left = [], qty
            for l in avail:
                if not left:
                    break
                take = min(left, l.qty)
                legs.append(l if take == l.qty else
                            quote_leg(self.venues[l.venue_id], agent_id, side, take, intent.price))
                left -= take
        return RoutePlan(agent_id, side, qty, tuple(legs), quotes)

    def execute(self, plan: RoutePlan, time: float = 0.0) -> Dict[str, List[Trade]]:
        """Send each leg as an IOC market order and settle on the venue ledger."""
        out: Dict[str, List[Trade]] = {}
        for leg in plan.legs:
            v = self.venues[leg.venue_id]
            order = Order(v.order_ids(), plan.agent_id, plan.side, leg.qty, None,
                          float(time), OrderType.MARKET)
            trades = v.book.submit(order)
            v.ledger.settle_all(trades)
            out[leg.venue_id] = trades
        return out

    def route(self, agent_id: int, intent: Submit,
              time: float = 0.0) -> Tuple[RoutePlan, Dict[str, List[Trade]]]:
        plan = self.plan(agent_id, intent)
        return plan, self.execute(plan, time)
