"""Decentralized bilateral search market with search costs and perishable goods.

Unit buyers (valuation `value`) and unit sellers (cost `value`) meet pairwise on
the simulation clock: meetings arrive as a Poisson process with rate
`meeting_rate`, and each meeting draws one active buyer and one active seller
uniformly from the seeded clock RNG. Every meeting charges both sides a search
cost, priced by the friction stack on a one-lot `Fill` and moved to the
search-cost sink account by `Ledger.transfer`. A pair trades one lot iff
`b = buyer reservation >= s = seller reservation`, at the Nash split

    surplus = b - s,  buyer_share = floor(surplus * w_bps / 10_000),
    price   = b - buyer_share   (seller gets the remainder: price - s)

so `buyer_share + seller_share == surplus` exactly. With perishability, every
`decay_interval` an unsold seller's value drops by `decay` ticks, never below
its `floor`. Fills are `Trade`s settled through the ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..core.clock import SimClock
from ..core.ledger import Ledger
from ..core.types import Side, Trade, _check_int
from ..costs.base import CostModel, Fill, Role
from ..costs.fees import FixedCost
from ..costs.stack import FrictionStack

BPS_DENOM = 10_000
SEARCH_SINK_ID = -3


@dataclass(frozen=True)
class ReservationPolicy:
    """How an agent sets its reservation price: `fixed` or myopic optimal `stopping`.

    `fixed(price)` posts `price` (own value if None); `stopping(lo, hi, cost)` is the
    one-step-lookahead rule against a counterparty reservation X ~ U{lo..hi}: a buyer
    accepts x <= r with r the smallest tick where E[(r - X)+] >= cost, a seller x >= r
    with r the largest tick where E[(X - r)+] >= cost. Buyers never post above their
    value and sellers never below it, so no trade is at a loss.
    """
    kind: str = "fixed"
    price: Optional[int] = None
    lo: int = 1
    hi: int = 1
    cost: int = 0

    @classmethod
    def fixed(cls, price: Optional[int] = None) -> "ReservationPolicy":
        """Constant reservation `price`, or truthful (own current value) when None."""
        return cls("fixed", None if price is None else _check_int("price", price, positive=True))

    @classmethod
    def stopping(cls, lo: int, hi: int, cost: int) -> "ReservationPolicy":
        """Myopic optimal stopping rule for a known uniform counterparty distribution."""
        lo, hi = _check_int("lo", lo, positive=True), _check_int("hi", hi, positive=True)
        if hi < lo:
            raise ValueError(f"hi must be >= lo, got [{lo}, {hi}]")
        return cls("stopping", None, lo, hi, _check_int("cost", cost, nonneg=True))

    def threshold(self, side: Side) -> int:
        """Integer stopping threshold for `side` (exact: sums times n vs cost times n)."""
        need = self.cost * (self.hi - self.lo + 1)
        if side is Side.BUY:
            return next((r for r in range(self.lo, self.hi + 1)
                         if sum(r - x for x in range(self.lo, r + 1)) >= need), self.hi)
        return next((r for r in range(self.hi, self.lo - 1, -1)
                     if sum(x - r for x in range(r, self.hi + 1)) >= need), self.lo)

    def reservation(self, side: Side, value: int) -> int:
        """Reservation price of an agent with current `value` on `side`."""
        r = value if self.kind == "fixed" and self.price is None else (
            self.price if self.kind == "fixed" else self.threshold(side))
        return min(value, r) if side is Side.BUY else max(value, r)


@dataclass
class SearchAgent:
    """A unit trader: `value` is a buyer's valuation or a seller's (perishable) cost."""
    agent_id: int
    side: Side
    value: int
    policy: ReservationPolicy = field(default_factory=ReservationPolicy)
    floor: int = 1
    active: bool = True

    @property
    def reservation(self) -> int:
        """Current reservation price under `policy`."""
        return self.policy.reservation(self.side, self.value)


@dataclass(frozen=True)
class Meeting:
    """One bilateral meeting: search costs charged and the trade (if any)."""
    time: float
    buyer_id: int
    seller_id: int
    buyer_cost: int
    seller_cost: int
    trade: Optional[Trade] = None


class SearchMarket:
    """Seeded bilateral search venue over a `Ledger` with a search-cost sink."""

    def __init__(self, agents: List[SearchAgent], *, meeting_rate: float = 1.0,
                 search_cost: Union[int, CostModel, FrictionStack] = 0, buyer_weight_bps: int = 5_000,
                 decay: int = 0, decay_interval: float = 1.0, seed: int = 0, cash: int = 0) -> None:
        if not meeting_rate > 0 or not decay_interval > 0:
            raise ValueError("meeting_rate and decay_interval must be > 0")
        self.w = _check_int("buyer_weight_bps", buyer_weight_bps, nonneg=True)
        if self.w > BPS_DENOM:
            raise ValueError(f"buyer_weight_bps must be <= {BPS_DENOM}")
        self.decay = _check_int("decay", decay, nonneg=True)
        if isinstance(search_cost, int):
            search_cost = FixedCost(_check_int("search_cost", search_cost, nonneg=True))
        self.stack = search_cost if isinstance(search_cost, FrictionStack) else FrictionStack([search_cost])
        self.agents: Dict[int, SearchAgent] = {}
        self.ledger = Ledger()
        self.ledger.open_account(SEARCH_SINK_ID)
        for a in agents:
            _check_int("value", a.value, positive=True)
            _check_int("floor", a.floor, positive=True)
            if a.agent_id in self.agents or a.agent_id == SEARCH_SINK_ID:
                raise ValueError(f"duplicate or reserved agent id {a.agent_id}")
            self.agents[a.agent_id] = a
            self.ledger.open_account(a.agent_id, cash, 1 if a.side is Side.SELL else 0)
        self.rate, self.interval = float(meeting_rate), float(decay_interval)
        self.clock = SimClock(seed)
        self.clock.on("meet", self._on_meet)
        self.clock.on("decay", self._on_decay)
        self.meetings: List[Meeting] = []
        self.trades: List[Trade] = []
        self._started = False

    # ---- state ---------------------------------------------------------
    def active(self, side: Side) -> List[SearchAgent]:
        """Unmatched agents on `side`, in id order (the draw order)."""
        return [a for _, a in sorted(self.agents.items()) if a.active and a.side is side]

    def exhausted(self) -> bool:
        """True when no remaining buyer-seller pair could ever trade (values only fall)."""
        sells = self.active(Side.SELL)
        lowest = min((s.policy.reservation(Side.SELL, s.floor if self.decay else s.value)
                      for s in sells), default=None)
        return lowest is None or all(b.reservation < lowest for b in self.active(Side.BUY))

    @property
    def search_costs(self) -> int:
        """Total search cost charged over all meetings (both sides)."""
        return sum(m.buyer_cost + m.seller_cost for m in self.meetings)

    def split(self, b: int, s: int) -> int:
        """Nash bargaining price for reservations b >= s; remainder goes to the seller."""
        return b - (b - s) * self.w // BPS_DENOM

    # ---- event handlers ------------------------------------------------
    def _charge(self, a: SearchAgent) -> int:
        fee = self.stack.charge(Fill(a.agent_id, a.side, Role.TAKER, a.reservation, 1, self.clock.now))
        if fee:
            self.ledger.transfer(a.agent_id, SEARCH_SINK_ID, fee)
        return fee

    def _on_meet(self, clock: SimClock, ev) -> None:
        if self.exhausted():  # stop the meeting process; decay keeps its own schedule
            return
        buys, sells = self.active(Side.BUY), self.active(Side.SELL)
        b = buys[int(clock.rng.integers(len(buys)))]
        s = sells[int(clock.rng.integers(len(sells)))]
        bc, sc, trade = self._charge(b), self._charge(s), None
        rb, rs = b.reservation, s.reservation
        if rb >= rs:
            tid = len(self.trades) + 1
            trade = Trade(tid, clock.now, self.split(rb, rs), 1, tid, tid, b.agent_id, s.agent_id, Side.BUY)
            self.ledger.settle(trade)
            self.trades.append(trade)
            b.active = s.active = False
        self.meetings.append(Meeting(clock.now, b.agent_id, s.agent_id, bc, sc, trade))
        clock.schedule_poisson(self.rate, "meet", None, 1)

    def _on_decay(self, clock: SimClock, ev) -> None:
        for s in self.active(Side.SELL):
            s.value = max(s.floor, s.value - self.decay)
        if self.decay:
            clock.schedule(self.interval, "decay", None, 0)

    # ---- driver --------------------------------------------------------
    def run(self, until: float) -> int:
        """Advance to `until` (or exhaustion); returns the number of trades so far."""
        if not self._started:
            self._started = True
            self.clock.schedule_poisson(self.rate, "meet", None, 1)
            if self.decay:
                self.clock.schedule(self.interval, "decay", None, 0)
        self.clock.run(until=until)
        return len(self.trades)

    def records(self) -> list:
        """Fills as fl-022 `TradeRecord`s with venue "search"."""
        from ..analysis.metrics import TradeRecord  # lazy: analysis sits above markets
        return [TradeRecord.from_trade(t, "search") for t in self.trades]
