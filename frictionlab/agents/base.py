"""Agent base class: observe -> decide -> act, with an explicit decision cost.

Lifecycle of one `Agent.step(venue, time)`:
  1. observe: the venue builds a read-only `MarketView` snapshot (top of book,
     depth, last trade, the agent's own account and open orders).
  2. decide:  before `decide()` runs, `deliberation_cost(view)` integer
     tick-lots move from the agent to a dedicated cost-sink account through
     `Ledger.transfer`, so cash stays conserved. If the agent cannot fund the
     thought (ledger forbids overdraft) it is priced out: no decision, no act.
  3. act:     `decide()` returns typed intents (`Submit` / `Cancel`); only the
     venue touches the book, enforces order ownership and settles every fill
     through the ledger (friction stack included).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple, Union

from ..core.ledger import Ledger, LedgerError
from ..core.order_book import OrderBook
from ..core.types import IdGen, Order, OrderType, Side, Trade, _check_int

COST_SINK_ID = -1
Levels = Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class MarketView:
    time: float
    agent_id: int
    best_bid: Optional[int]
    best_ask: Optional[int]
    bids: Levels
    asks: Levels
    last_trade: Optional[Trade]
    cash: int
    position: int
    open_orders: Tuple[int, ...] = ()
    recent_trades: Tuple[Trade, ...] = ()  # tape tail, oldest first

    @property
    def spread(self) -> Optional[int]:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else a - b

    @property
    def mid(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else (a + b) / 2


@dataclass(frozen=True)
class Submit:
    side: Side
    qty: int
    price: Optional[int] = None  # None -> market order

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", Side(self.side))
        _check_int("qty", self.qty, positive=True)
        if self.price is not None:
            _check_int("price", self.price, positive=True)

    @property
    def otype(self) -> OrderType:
        return OrderType.MARKET if self.price is None else OrderType.LIMIT


@dataclass(frozen=True)
class Cancel:
    order_id: int
    qty: Optional[int] = None  # None -> cancel the full remainder

    def __post_init__(self) -> None:
        _check_int("order_id", self.order_id)
        if self.qty is not None:
            _check_int("qty", self.qty, positive=True)


Action = Union[Submit, Cancel]


class ActionError(ValueError):
    """An intent is malformed or targets an order the agent does not own."""


@dataclass
class Venue:
    """The only writer of book and ledger; agents see it through `MarketView`."""
    book: OrderBook
    ledger: Ledger
    order_ids: IdGen = field(default_factory=IdGen)
    cost_sink_id: int = COST_SINK_ID
    depth: int = 5
    tape: int = 64  # trades exposed via MarketView.recent_trades

    def __post_init__(self) -> None:
        if self.cost_sink_id not in self.ledger:
            self.ledger.open_account(self.cost_sink_id)

    @property
    def decision_costs_collected(self) -> int:
        return self.ledger[self.cost_sink_id].cash

    def view(self, agent: "Agent", time: float) -> MarketView:
        acct, b = self.ledger[agent.agent_id], self.book
        agent.open_orders = {o for o in agent.open_orders if o in b}
        return MarketView(float(time), agent.agent_id, b.best_bid, b.best_ask,
                          tuple(b.depth(Side.BUY, self.depth)),
                          tuple(b.depth(Side.SELL, self.depth)),
                          b.trades[-1] if b.trades else None,
                          acct.cash, acct.position, tuple(sorted(agent.open_orders)),
                          tuple(b.trades[-self.tape:]) if self.tape else ())

    def charge(self, agent_id: int, amount: int) -> bool:
        """Route a decision cost to the sink; False if the agent cannot pay."""
        try:
            self.ledger.transfer(agent_id, self.cost_sink_id, amount)
        except LedgerError:
            return False
        return True

    def execute(self, agent: "Agent", action: Action, time: float) -> List[Trade]:
        if isinstance(action, Submit):
            oid = self.order_ids()
            order = Order(oid, agent.agent_id, action.side, action.qty,
                          action.price, float(time), action.otype)
            trades = self.book.submit(order)
            self.ledger.settle_all(trades)
            if oid in self.book:
                agent.open_orders.add(oid)
            return trades
        if isinstance(action, Cancel):
            resting = self.book.get(action.order_id)
            if resting is not None and resting.agent_id != agent.agent_id:
                raise ActionError(f"agent {agent.agent_id} cannot cancel order "
                                  f"{action.order_id} owned by {resting.agent_id}")
            self.book.cancel(action.order_id, action.qty)
            return []
        raise ActionError(f"unknown action {action!r}")


class Agent(ABC):
    def __init__(self, agent_id: int, decision_cost: int = 0) -> None:
        self.agent_id = _check_int("agent_id", agent_id)
        self.decision_cost = _check_int("decision_cost", decision_cost, nonneg=True)
        self.open_orders: Set[int] = set()
        self.view: Optional[MarketView] = None
        self.n_decisions = 0
        self.n_priced_out = 0
        self.decision_costs_paid = 0

    def observe(self, view: MarketView) -> None:
        """Hook for belief updates; the default just keeps the latest view."""
        self.view = view

    def deliberation_cost(self, view: MarketView) -> int:
        """Cost of one decide() call; override for state-dependent thinking."""
        return self.decision_cost

    @abstractmethod
    def decide(self, view: MarketView) -> Sequence[Action]:
        """Pure intent: return actions, never mutate book or ledger."""

    def act(self, venue: Venue, actions: Sequence[Action], time: float) -> List[Trade]:
        trades: List[Trade] = []
        for a in actions:
            trades += venue.execute(self, a, time)
        return trades

    def step(self, venue: Venue, time: float) -> List[Trade]:
        view = venue.view(self, time)
        self.observe(view)
        cost = _check_int("deliberation_cost", self.deliberation_cost(view), nonneg=True)
        if cost and not venue.charge(self.agent_id, cost):
            self.n_priced_out += 1
            return []
        self.decision_costs_paid += cost
        self.n_decisions += 1
        return self.act(venue, list(self.decide(view)), time)
