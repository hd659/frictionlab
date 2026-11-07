"""Grossman-Stiglitz information acquirer: buy a costly signal only when it pays.

On every wake-up (after the base class charges `decision_cost`) the agent
prices the `SignalSource` signal against the current touch:

    E      = source.edge(spread) * order_size     expected gross edge, floored once
    hurdle = source.cost(agent_id) + decision_cost

It buys iff E > hurdle (ties go to "do not buy": a zero-net purchase only pays
the source). After buying it trades like the fl-018 `InformedTrader`: a
marketable limit at the touch in the signal's direction (BUY on HIGH, SELL on
LOW), sized by touch depth and an optional |position| cap, provided the edge
still clears the taker friction of that fill. In every other case it is a
value-blind noise trader driven by an embedded `NoiseTrader` stream, so with
an uninformative source (5000 bps) it reproduces that noise trader exactly.

The signal is bought in `step` through the venue's ledger (an information-sink
transfer like the decision-cost sink); `decide` stays a pure intent builder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..core.types import Side, _check_int
from ..costs.base import Fill, Role
from ..costs.information import HIGH, SignalSource
from .base import Action, Agent, Cancel, MarketView, Submit
from .informed import _as_stack
from .noise import NoiseTrader, RngLike


@dataclass(frozen=True)
class InfoChoice:
    """One wake-up's acquisition decision and its outcome."""
    time: float
    expected_edge: Optional[int]   # None when the book has no two-sided quote
    hurdle: int                    # signal cost + decision cost
    bought: bool
    signal: Optional[int] = None   # HIGH / LOW when bought, else None
    traded: bool = False           # an informed order was sent


class InfoAcquirer(Agent):
    """Pays for a `SignalSource` signal iff expected edge > signal cost + decision cost."""

    def __init__(self, agent_id: int, source: SignalSource, *, frictions=None,
                 order_size: int = 1, max_position: Optional[int] = None,
                 wake_rate: float = 1.0, decision_cost: int = 0, rng: RngLike = None,
                 participation: float = 1.0, buy_prob: float = 0.5) -> None:
        super().__init__(agent_id, decision_cost)
        if not isinstance(source, SignalSource):
            raise TypeError("source must be a SignalSource")
        self.source = source
        self.frictions = _as_stack(frictions)
        self.order_size = _check_int("order_size", order_size, positive=True)
        self.max_position = (None if max_position is None
                             else _check_int("max_position", max_position, positive=True))
        self.noise = NoiseTrader(agent_id, rng, arrival_rate=wake_rate,
                                 participation=participation, buy_prob=buy_prob)
        self.wake_rate = self.noise.wake_rate
        self.history: List[InfoChoice] = []
        self.n_signals = 0
        self.signal_costs_paid = 0
        self.n_informed_orders = 0
        self._ledger = None
        self._pending: Optional[int] = None

    # -- model ----------------------------------------------------------------
    @property
    def hurdle(self) -> int:
        """Signal price plus decision cost, in ticks."""
        return self.source.cost(self.agent_id) + self.decision_cost

    def expected_edge(self, view: MarketView) -> Optional[int]:
        """Floored expected gross edge of trading `order_size` lots on one signal."""
        if view.spread is None:
            return None
        return self.source.edge(view.spread) * self.order_size

    def should_buy(self, view: MarketView) -> bool:
        """True iff the expected edge strictly exceeds signal cost + decision cost."""
        e = self.expected_edge(view)
        return e is not None and e > self.hurdle

    def _room(self, side: Side, position: int) -> int:
        if self.max_position is None:
            return self.order_size
        return max(0, min(self.order_size, self.max_position - side.value * position))

    def informed_order(self, view: MarketView, signal: int) -> Optional[Submit]:
        """Marketable limit at the touch in the signal's direction, if edge > taker friction."""
        side = Side.BUY if signal == HIGH else Side.SELL
        price = view.best_ask if side is Side.BUY else view.best_bid
        levels = view.asks if side is Side.BUY else view.bids
        if price is None or not levels:
            return None
        qty = min(self._room(side, view.position), levels[0][1])
        if qty == 0:
            return None
        edge = self.source.edge(view.spread) * qty
        if edge <= self.frictions.charge(Fill(self.agent_id, side, Role.TAKER, price, qty)):
            return None
        return Submit(side, qty, price)

    # -- lifecycle ------------------------------------------------------------
    def step(self, venue, time: float):
        """Base step with the venue's ledger exposed so `decide` can buy the signal."""
        self._ledger = venue.ledger
        try:
            return super().step(venue, time)
        finally:
            self._ledger = None

    def _buy(self, view: MarketView) -> Optional[int]:
        if self._ledger is None:  # bare decide() without a venue: nothing to pay with
            return None
        before = self.source.paid.get(self.agent_id, 0)
        sig = self.source.sell(self._ledger, self.agent_id, view.time)
        if sig is not None:
            self.n_signals += 1
            self.signal_costs_paid += self.source.paid[self.agent_id] - before
        return sig

    def decide(self, view: MarketView) -> List[Action]:
        actions: List[Action] = [Cancel(oid) for oid in view.open_orders]
        e = self.expected_edge(view)
        hurdle = self.hurdle
        sig = self._buy(view) if (e is not None and e > hurdle) else None
        order = self.informed_order(view, sig) if sig is not None else None
        self.history.append(InfoChoice(view.time, e, hurdle, sig is not None, sig,
                                       order is not None))
        if order is not None:
            self.n_informed_orders += 1
            return actions + [order]
        noise = self.noise.draw()
        if noise is not None:
            self.noise.orders.append(noise)
            actions.append(noise)
        return actions

    @property
    def buy_rate(self) -> float:
        """Share of decisions on which a signal was bought."""
        return self.n_signals / len(self.history) if self.history else 0.0

    def __repr__(self) -> str:
        return (f"InfoAcquirer(id={self.agent_id}, accuracy_bps={self.source.accuracy_bps}, "
                f"decision_cost={self.decision_cost}, signals={self.n_signals})")
