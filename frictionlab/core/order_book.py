"""Continuous limit order book with price-time priority.

Matching contract:
- Price priority: best opposite price first; trades print at the *resting* price.
- Time priority: FIFO within a level (arrival order at the book).
- Partial fills: an aggressor walks levels until filled or no longer crossing;
  a limit remainder rests, a market remainder is cancelled (IOC semantics).
- Cancels: full or partial (`qty=`); a partial reduce keeps queue priority.
- Self-trade prevention (Trade forbids buyer == seller):
  "cancel_resting" drops the own resting order and keeps matching,
  "cancel_incoming" kills the aggressor's remainder.

Cancelled orders are removed lazily from level queues; per-level aggregate
quantity is maintained eagerly so depth queries are exact at all times.
"""
from __future__ import annotations

import bisect
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from .types import IdGen, Order, OrderType, Side, Trade, _check_int

STP_MODES = ("cancel_resting", "cancel_incoming")


class _Half:
    """One side of the book: sorted price keys + FIFO queue per level."""

    def __init__(self, side: Side) -> None:
        self.side = side
        self._keys: List[int] = []  # ascending; bids stored negated
        self.queues: Dict[int, Deque[Order]] = {}
        self.qty: Dict[int, int] = {}

    def _key(self, price: int) -> int:
        return -price if self.side is Side.BUY else price

    def best(self) -> Optional[int]:
        return abs(self._keys[0]) if self._keys else None

    def add(self, order: Order) -> None:
        p = order.price
        if p not in self.queues:
            bisect.insort(self._keys, self._key(p))
            self.queues[p] = deque()
            self.qty[p] = 0
        self.queues[p].append(order)
        self.qty[p] += order.remaining

    def head(self, price: int) -> Order:
        q = self.queues[price]
        while not q[0].is_active:  # lazy purge; qty>0 guarantees a live order
            q.popleft()
        return q[0]

    def reduce(self, price: int, lots: int) -> None:
        self.qty[price] -= lots
        if self.qty[price] == 0:
            del self.queues[price], self.qty[price]
            i = bisect.bisect_left(self._keys, self._key(price))
            del self._keys[i]

    def levels(self, n: Optional[int] = None) -> List[Tuple[int, int]]:
        keys = self._keys if n is None else self._keys[:n]
        return [(abs(k), self.qty[abs(k)]) for k in keys]


class OrderBook:
    def __init__(self, trade_ids: Optional[IdGen] = None,
                 stp: str = "cancel_resting") -> None:
        if stp not in STP_MODES:
            raise ValueError(f"stp must be one of {STP_MODES}, got {stp!r}")
        self.stp = stp
        self._tid = trade_ids or IdGen()
        self._halves = {Side.BUY: _Half(Side.BUY), Side.SELL: _Half(Side.SELL)}
        self._live: Dict[int, Order] = {}
        self.trades: List[Trade] = []

    # ---- queries -------------------------------------------------------
    @property
    def best_bid(self) -> Optional[int]:
        return self._halves[Side.BUY].best()

    @property
    def best_ask(self) -> Optional[int]:
        return self._halves[Side.SELL].best()

    @property
    def spread(self) -> Optional[int]:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else a - b

    @property
    def mid(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else (a + b) / 2

    def depth(self, side: Side, levels: Optional[int] = None) -> List[Tuple[int, int]]:
        """[(price, total_qty)] from best outward."""
        return self._halves[Side(side)].levels(levels)

    def volume_at(self, side: Side, price: int) -> int:
        return self._halves[Side(side)].qty.get(price, 0)

    def get(self, order_id: int) -> Optional[Order]:
        return self._live.get(order_id)

    def __contains__(self, order_id: int) -> bool:
        return order_id in self._live

    def __len__(self) -> int:
        return len(self._live)

    # ---- mutations -----------------------------------------------------
    def submit(self, order: Order) -> List[Trade]:
        """Match `order` against the book; rest any limit remainder."""
        if order.order_id in self._live:
            raise ValueError(f"duplicate order id {order.order_id}")
        if not order.is_active:
            raise ValueError(f"order {order.order_id} is not active")
        trades = self._match(order)
        if order.is_active:
            if order.otype is OrderType.LIMIT:
                self._halves[order.side].add(order)
                self._live[order.order_id] = order
            else:
                order.cancel()
        self.trades.extend(trades)
        return trades

    def cancel(self, order_id: int, qty: Optional[int] = None) -> int:
        """Cancel all (qty=None) or `qty` lots of a resting order.

        Returns lots removed; 0 if the id is unknown or already done.
        """
        order = self._live.get(order_id)
        if order is None:
            return 0
        half = self._halves[order.side]
        if qty is None or _check_int("cancel qty", qty, positive=True) >= order.remaining:
            lots = order.cancel()
            del self._live[order_id]
        else:
            order.qty -= qty  # shrink in place: queue position preserved
            lots = qty
        half.reduce(order.price, lots)
        return lots

    def _match(self, taker: Order) -> List[Trade]:
        opp = self._halves[taker.side.opposite]
        trades: List[Trade] = []
        while taker.is_active:
            price = opp.best()
            if price is None or not taker.crosses(price):
                break
            maker = opp.head(price)
            if maker.agent_id == taker.agent_id:
                if self.stp == "cancel_incoming":
                    taker.cancel()
                    break
                self.cancel(maker.order_id)
                continue
            lots = min(taker.remaining, maker.remaining)
            taker.fill(lots)
            maker.fill(lots)
            opp.reduce(price, lots)
            if not maker.is_active:
                del self._live[maker.order_id]
            buy, sell = (taker, maker) if taker.side is Side.BUY else (maker, taker)
            trades.append(Trade(self._tid(), taker.time, price, lots,
                                buy.order_id, sell.order_id,
                                buy.agent_id, sell.agent_id, taker.side))
        return trades
