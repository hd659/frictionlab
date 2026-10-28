"""Informed trader (Glosten-Milgrom / Kyle 1985 style) that respects friction.

The trader observes the latent fundamental V(t) from a `FundamentalProcess`
and compares it with the touch. For q lots at the best quote:

    gross edge  buy:  (V - best_ask) * q        sell: (best_bid - V) * q
    friction    F = stack charge of a TAKER fill (taker fee in bps of
                notional + half-spread + any fixed/latency knob, summed as
                exact rationals and rounded once, venue-favourable)
              + decision_cost  (the price of the thought that found the edge)

It submits a marketable limit order at the touch only if gross edge > F, so
with edge <= F it never trades. Ties go to "no trade": a zero-net-edge trade
only transfers surplus to the venue. Size is capped by the touch depth and an
optional |position| limit; leftovers that would rest are cancelled next step.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional

from ..core.types import Side, _check_int
from ..costs.base import CostModel, Fill, Role
from ..costs.stack import FrictionStack
from .base import Action, Agent, Cancel, MarketView, Submit
from .fundamental import FundamentalProcess


@dataclass(frozen=True)
class EdgeQuote:
    """One side's opportunity as evaluated by `InformedTrader.evaluate`."""
    side: Side
    price: int
    qty: int
    gross_edge: int            # tick-lots before friction
    friction: Fraction         # tick-lots incl. decision cost
    breakdown: Dict[str, Fraction]

    @property
    def net_edge(self) -> Fraction:
        return self.gross_edge - self.friction

    @property
    def profitable(self) -> bool:
        return self.qty > 0 and self.gross_edge > self.friction


def _as_stack(frictions) -> FrictionStack:
    if frictions is None:
        return FrictionStack()
    if isinstance(frictions, CostModel):
        return FrictionStack([frictions])
    if isinstance(frictions, FrictionStack):
        return frictions
    raise TypeError(f"frictions must be a FrictionStack or CostModel, got {type(frictions).__name__}")


class InformedTrader(Agent):
    def __init__(self, agent_id: int, fundamental: FundamentalProcess, *,
                 frictions=None, order_size: int = 1,
                 max_position: Optional[int] = None, wake_rate: float = 1.0,
                 decision_cost: int = 0) -> None:
        super().__init__(agent_id, decision_cost)
        if not isinstance(fundamental, FundamentalProcess):
            raise TypeError("fundamental must be a FundamentalProcess")
        self.fundamental = fundamental
        self.frictions = _as_stack(frictions)
        self.order_size = _check_int("order_size", order_size, positive=True)
        self.max_position = (None if max_position is None
                             else _check_int("max_position", max_position, positive=True))
        if not float(wake_rate) > 0:
            raise ValueError(f"wake_rate must be > 0, got {wake_rate}")
        self.wake_rate = float(wake_rate)
        self.history: List[Optional[EdgeQuote]] = []
        self.n_trades_sent = 0

    # -- model --------------------------------------------------------------
    def friction(self, side: Side, price: int, qty: int) -> Dict[str, Fraction]:
        """Exact per-component friction of one taker fill plus decision cost."""
        fill = Fill(self.agent_id, side, Role.TAKER, price, qty)
        parts = dict(self.frictions.breakdown(fill))
        parts["decision_cost"] = Fraction(self.decision_cost)
        return parts

    def total_friction(self, side: Side, price: int, qty: int) -> Fraction:
        fill = Fill(self.agent_id, side, Role.TAKER, price, qty)
        return Fraction(self.frictions.charge(fill) + self.decision_cost)

    def _room(self, side: Side, position: int) -> int:
        if self.max_position is None:
            return self.order_size
        return max(0, min(self.order_size, self.max_position - side.value * position))

    def evaluate(self, view: MarketView, side: Side) -> Optional[EdgeQuote]:
        price = view.best_ask if side is Side.BUY else view.best_bid
        if price is None:
            return None
        levels = view.asks if side is Side.BUY else view.bids
        touch = levels[0][1] if levels else self.order_size
        qty = min(self._room(side, view.position), touch)
        v = self.fundamental.value(view.time)
        gross = side.value * (v - price) * qty
        if qty == 0:
            return EdgeQuote(side, price, 0, 0, Fraction(0), {})
        return EdgeQuote(side, price, qty, gross, self.total_friction(side, price, qty),
                         self.friction(side, price, qty))

    def best_opportunity(self, view: MarketView) -> Optional[EdgeQuote]:
        cands = [q for q in (self.evaluate(view, s) for s in (Side.BUY, Side.SELL))
                 if q is not None and q.profitable]
        return max(cands, key=lambda q: q.net_edge) if cands else None

    # -- lifecycle ----------------------------------------------------------
    def decide(self, view: MarketView) -> List[Action]:
        actions: List[Action] = [Cancel(oid) for oid in view.open_orders]
        opp = self.best_opportunity(view)
        self.history.append(opp)
        if opp is not None:
            actions.append(Submit(opp.side, opp.qty, opp.price))
            self.n_trades_sent += 1
        return actions

    def __repr__(self) -> str:
        return (f"InformedTrader(id={self.agent_id}, size={self.order_size}, "
                f"decision_cost={self.decision_cost}, frictions={len(self.frictions)})")
