"""Rubinstein-Wolinsky middleman: posted integer-tick quotes over perishable inventory.

A `Middleman` is an optional counterparty of the fl-046 `SearchMarket`. Traders
meet it at `middleman_meeting_rate` through the market's seeded matching
process. It posts a standing `bid` and `ask` (integer ticks, `bid <= ask`) and
holds at most `capacity` units:

    buys one unit at `bid` from a seller with reservation <= bid   (if units < K)
    sells one unit at `ask` to a buyer with reservation >= ask      (if units > 0)

Units leave FIFO. Every carry step (the market's decay interval) each held unit
j pays, through two friction stacks priced on a one-lot `Fill` at its value:

    holding_j = holding stack charge                 -> hold sink
    decay_j   = min(decay, value_j - floor)+          -> decay sink (`PerishCost`)
    value_j  <- value_j - decay_j                     (the fl-046 perishability rule)

Units are carried at cost basis (the bid paid) and spoilage is paid in cash, so

    net = cash change + basis of held units = spread capture - holding - decay

with spread capture = sum(ask) - sum(basis of units sold), exact in ticks.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from ..core.ledger import Ledger
from ..core.types import Side, _check_int
from ..costs.base import BOTH, CostModel, Fill, Role
from ..costs.fees import FixedCost
from ..costs.stack import FrictionStack
from .base import Action, Agent, MarketView

HOLD_SINK_ID = -4
DECAY_SINK_ID = -5


class PerishCost(CostModel):
    """Spoilage of one held unit per carry step: min(decay, value - floor) ticks, floor-bounded."""
    name = "perish"

    def __init__(self, decay: int, floor: int = 1) -> None:
        super().__init__(BOTH)
        self.decay = _check_int("decay", decay, nonneg=True)
        self.floor = _check_int("floor", floor, positive=True)

    def raw_cost(self, fill: Fill) -> Fraction:
        return Fraction(min(self.decay, max(0, fill.price - self.floor)) * fill.qty)


@dataclass
class HeldUnit:
    """One unit in inventory: `basis` is the bid paid, `value` its perishable carrying value."""
    basis: int
    value: int
    since: float


def as_stack(cost: Union[int, CostModel, FrictionStack], name: str) -> FrictionStack:
    """Integer -> per-unit `FixedCost`; a bare `CostModel` -> one-knob stack."""
    if isinstance(cost, int):
        cost = FixedCost(_check_int(name, cost, nonneg=True))
    return cost if isinstance(cost, FrictionStack) else FrictionStack([cost])


class Middleman(Agent):
    """Market-making intermediary with bid/ask quotes, capacity K, holding cost and decay."""

    def __init__(self, agent_id: int, bid: int, ask: int, capacity: int = 1, *,
                 holding_cost: Union[int, CostModel, FrictionStack] = 0, decay: int = 0,
                 floor: int = 1, hold_sink_id: int = HOLD_SINK_ID,
                 decay_sink_id: int = DECAY_SINK_ID) -> None:
        super().__init__(agent_id)
        self.bid = _check_int("bid", bid, positive=True)
        self.ask = _check_int("ask", ask, positive=True)
        if self.ask < self.bid:
            raise ValueError(f"ask must be >= bid, got bid={bid} ask={ask}")
        self.capacity = _check_int("capacity", capacity, positive=True)
        self.holding = as_stack(holding_cost, "holding_cost")
        self.perish = FrictionStack([PerishCost(decay, floor)])
        self.hold_sink_id, self.decay_sink_id = hold_sink_id, decay_sink_id
        if len({agent_id, hold_sink_id, decay_sink_id}) != 3:
            raise ValueError("middleman id and its sink ids must be distinct")
        self.inventory: List[HeldUnit] = []
        self.holding_paid = self.decay_paid = 0
        self.bought = self.sold = 0
        self.basis_sold = 0

    @property
    def units(self) -> int:
        """Units currently held, always in [0, capacity]."""
        return len(self.inventory)

    def buys_from(self, reservation: int) -> bool:
        """True if a seller with `reservation` sells to the posted bid now."""
        return self.units < self.capacity and self.bid >= reservation

    def sells_to(self, reservation: int) -> bool:
        """True if a buyer with `reservation` buys at the posted ask now."""
        return self.units > 0 and self.ask <= reservation

    def alive(self, buy_reservations: Iterable[int], sell_floors: Sequence[int]) -> bool:
        """Could any future middleman fill happen (seller reservations only fall to `sell_floors`)?"""
        can_buy = self.units < self.capacity and any(r <= self.bid for r in sell_floors)
        can_sell = (self.units > 0 or can_buy) and any(r >= self.ask for r in buy_reservations)
        return can_buy or can_sell

    def take(self, time: float) -> int:
        """Book one bought unit at the bid; returns the trade price."""
        if self.units >= self.capacity:
            raise ValueError("middleman at capacity")
        self.inventory.append(HeldUnit(self.bid, self.bid, float(time)))
        self.bought += 1
        return self.bid

    def give(self) -> int:
        """Release the oldest unit at the ask; returns the trade price."""
        if not self.inventory:
            raise ValueError("middleman has no inventory")
        self.basis_sold += self.inventory.pop(0).basis
        self.sold += 1
        return self.ask

    def meet(self, side: Side, reservation: int, time: float) -> Optional[int]:
        """Fill price for a trader on `side` with `reservation`, or None (trader keeps searching)."""
        if side is Side.BUY:
            return self.give() if self.sells_to(reservation) else None
        return self.take(time) if self.buys_from(reservation) else None

    def carry(self, ledger: Ledger, time: float) -> Tuple[int, int]:
        """One carry step: charge holding and decay per held unit to the sinks."""
        hold = spoil = 0
        for u in self.inventory:
            fill = Fill(self.agent_id, Side.SELL, Role.MAKER, u.value, 1, float(time))
            h, d = self.holding.charge(fill), self.perish.charge(fill)
            u.value -= d
            hold, spoil = hold + h, spoil + d
        if hold:
            ledger.transfer(self.agent_id, self.hold_sink_id, hold)
        if spoil:
            ledger.transfer(self.agent_id, self.decay_sink_id, spoil)
        self.holding_paid += hold
        self.decay_paid += spoil
        return hold, spoil

    def decide(self, view: MarketView) -> Sequence[Action]:
        """Quotes are standing; the search venue routes meetings, so no book intents."""
        return ()
