"""Perpetual futures venue with a periodic, clamped funding rate.

Prices (all on the integer tick grid, exact `Fraction` arithmetic):
    index        = index(t)                      e.g. a `FundamentalProcess`
    mark         = order-book mid (else last trade, else index)
    premium_bps  = (mark - index) * 10_000 / index
    rate_bps     = clamp(premium_bps * k, -cap_bps, cap_bps)

Funding at each interval: an account with signed contract position `q` owes
    owed = q * mark * rate_bps / 10_000        (> 0 pays, < 0 receives)
so longs pay shorts when mark > index and shorts pay longs when mark < index.
Payers pay ceil(owed), receivers get floor(-owed); every transfer is an
internal `Ledger.transfer` via the venue's funding sink account, which keeps
the rounding remainder. Since open interest is balanced (sum q == 0) the
exact amounts cancel, so the remainder is >= 0 and cash stays conserved.

Contract positions are ingested from fills on `book` (buyer +qty, seller
-qty); entry prices are volume-weighted while a position grows, kept while
it shrinks and reset to the fill price when it flips sign.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Optional, Union

from ..core.ledger import Ledger, LedgerError
from ..core.order_book import OrderBook
from ..core.types import Trade, _check_int

BPS_DENOM = 10_000
PERP_SINK_ID = -2
Num = Union[int, Fraction]


@dataclass(frozen=True)
class FundingEvent:
    time: float
    index: int
    mark: Fraction
    premium_bps: Fraction
    rate_bps: Fraction
    payments: Dict[int, int]    # signed cash delta per account (+ = received)
    remainder: int              # rounding residue kept by the sink (>= 0)

    @property
    def paid(self) -> int:
        return -sum(v for v in self.payments.values() if v < 0)

    @property
    def received(self) -> int:
        return sum(v for v in self.payments.values() if v > 0)


def _ceil(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def _floor(x: Fraction) -> int:
    return x.numerator // x.denominator


class PerpMarket:
    def __init__(self, ledger: Ledger, index, book: Optional[OrderBook] = None, *,
                 k: Num = 1, cap_bps: Num = 75, interval: float = 8.0,
                 sink_id: int = PERP_SINK_ID) -> None:
        self.index_fn: Callable[[float], int] = getattr(index, "value", index)
        self.k = Fraction(k)
        self.cap_bps = Fraction(cap_bps)
        if self.k < 0 or self.cap_bps < 0:
            raise ValueError("k and cap_bps must be >= 0")
        if not interval > 0:
            raise ValueError(f"interval must be > 0, got {interval}")
        self.interval = float(interval)
        self.ledger, self.book = ledger, book if book is not None else OrderBook()
        self.sink_id = _check_int("sink_id", sink_id)
        if sink_id not in ledger:
            ledger.open_account(sink_id)
        self.positions: Dict[int, int] = {}
        self.entry: Dict[int, Fraction] = {}
        self.history: List[FundingEvent] = []
        self._cursor = 0

    # ---- positions -----------------------------------------------------
    def _fill(self, aid: int, dq: int, price: int) -> None:
        q0 = self.positions.get(aid, 0)
        q1 = q0 + dq
        if q1 == 0:
            self.positions.pop(aid, None)
            self.entry.pop(aid, None)
            return
        if q0 == 0 or (q0 > 0) != (q1 > 0):
            self.entry[aid] = Fraction(price)          # open or flip
        elif abs(q1) > abs(q0):
            self.entry[aid] = (self.entry[aid] * abs(q0) + price * abs(dq)) / abs(q1)
        self.positions[aid] = q1

    def apply_trade(self, trade: Trade) -> None:
        self._fill(trade.buyer_id, trade.qty, trade.price)
        self._fill(trade.seller_id, -trade.qty, trade.price)

    def sync(self, trades: Optional[Iterable[Trade]] = None) -> int:
        """Ingest new fills (default: the book tape since the last sync)."""
        if trades is None:
            trades = self.book.trades[self._cursor:]
            self._cursor = len(self.book.trades)
        n = 0
        for t in trades:
            self.apply_trade(t)
            n += 1
        return n

    def position(self, aid: int) -> int:
        return self.positions.get(aid, 0)

    @property
    def open_interest(self) -> int:
        return sum(q for q in self.positions.values() if q > 0)

    @property
    def short_interest(self) -> int:
        return -sum(q for q in self.positions.values() if q < 0)

    def unrealized_pnl(self, aid: int, mark: Optional[Num] = None) -> Fraction:
        q = self.position(aid)
        m = self.mark_price() if mark is None else Fraction(mark)
        return Fraction(0) if q == 0 else q * (m - self.entry[aid])

    # ---- prices & rate -------------------------------------------------
    def index_price(self, t: float) -> int:
        return _check_int("index", self.index_fn(t), positive=True)

    def mark_price(self, t: float = 0.0) -> Fraction:
        b, a = self.book.best_bid, self.book.best_ask
        if b is not None and a is not None:
            return Fraction(a + b, 2)
        if self.book.trades:
            return Fraction(self.book.trades[-1].price)
        return Fraction(self.index_price(t))

    def funding_rate_bps(self, mark: Num, index: int) -> tuple:
        premium = (Fraction(mark) - index) * BPS_DENOM / index
        rate = max(-self.cap_bps, min(self.cap_bps, premium * self.k))
        return premium, rate

    # ---- funding -------------------------------------------------------
    def fund(self, time: float, mark: Optional[Num] = None) -> FundingEvent:
        """Settle one funding period; atomic (validates before any transfer)."""
        self.sync()
        if self.open_interest != self.short_interest:
            raise LedgerError("open interest unbalanced: longs != shorts")
        index = self.index_price(time)
        m = self.mark_price(time) if mark is None else Fraction(mark)
        premium, rate = self.funding_rate_bps(m, index)
        pays: Dict[int, int] = {}
        gets: Dict[int, int] = {}
        for aid in sorted(self.positions):
            owed = self.positions[aid] * m * rate / BPS_DENOM
            if owed > 0:
                pays[aid] = _ceil(owed)
            elif owed < 0:
                gets[aid] = _floor(-owed)
        lg = self.ledger
        for aid in list(pays) + list(gets):
            lg[aid]  # unknown account -> LedgerError before mutation
        if not lg.allow_overdraft:
            short = [a for a, v in pays.items() if lg[a].cash < v]
            if short:
                raise LedgerError(f"accounts {short} cannot fund payment")
        for aid, v in pays.items():
            if v:
                lg.transfer(aid, self.sink_id, v)
        for aid, v in gets.items():
            if v:
                lg.transfer(self.sink_id, aid, v)
        payments = {a: -v for a, v in pays.items() if v}
        payments.update({a: v for a, v in gets.items() if v})
        ev = FundingEvent(float(time), index, m, premium, rate, payments,
                          sum(pays.values()) - sum(gets.values()))
        self.history.append(ev)
        return ev

    def attach(self, clock, start: Optional[float] = None, priority: int = -1) -> None:
        """Schedule recurring `funding` events on a `SimClock`."""
        def on_funding(c, ev) -> None:
            self.fund(c.now)
            c.schedule(self.interval, "funding", None, priority)
        clock.on("funding", on_funding)
        clock.schedule_at(clock.now + self.interval if start is None else start,
                          "funding", None, priority)

    @property
    def sink_cash(self) -> int:
        return self.ledger[self.sink_id].cash
