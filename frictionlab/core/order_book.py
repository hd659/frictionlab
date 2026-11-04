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

Storage (fl-039, see docs/phase6-performance.md): each side is a
`book_levels.BookSide` -- integer-tick dict of FIFO deques plus a heapq
best-price index with lazy deletion of emptied levels. `_live` indexes resting
orders by id, so cancel is an O(1) lookup plus an O(1) level update.
Cancelled orders are removed lazily from level queues; per-level aggregate
quantity is maintained eagerly so depth queries are exact at all times.

Results are byte-for-byte those of the pre-fl-039 sorted-list book, which is
frozen as the oracle in tests/reference_order_book.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .book_levels import BookSide, depth_array
from .types import IdGen, Order, OrderType, Side, Trade, _check_int

STP_MODES = ("cancel_resting", "cancel_incoming")
_BUY, _SELL, _LIMIT = Side.BUY, Side.SELL, OrderType.LIMIT


class OrderBook:
    """Continuous limit order book with price-time priority over integer-tick levels.

    Supports limit and market orders, partial fills, O(1) cancel by order id and a
    self-trade prevention mode (`stp`). Every fill is appended to `trades`.
    """
    def __init__(self, trade_ids: Optional[IdGen] = None,
                 stp: str = "cancel_resting") -> None:
        if stp not in STP_MODES:
            raise ValueError(f"stp must be one of {STP_MODES}, got {stp!r}")
        self.stp = stp
        self._tid = trade_ids or IdGen()
        self._halves = {_BUY: BookSide(_BUY), _SELL: BookSide(_SELL)}
        self._live: Dict[int, Order] = {}
        self.trades: List[Trade] = []

    # ---- queries -------------------------------------------------------
    @property
    def best_bid(self) -> Optional[int]:
        return self._halves[_BUY].best()

    @property
    def best_ask(self) -> Optional[int]:
        return self._halves[_SELL].best()

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

    def depth_array(self, side: Side, levels: Optional[int] = None) -> np.ndarray:
        """int64 array of [price, qty, cumulative qty] rows from best outward.

        Vectorised view of `depth` for MarketView-style depth queries (e.g.
        lots available within k ticks); the numba kernel is opt-in via the
        FRICTIONLAB_NUMBA env flag and never changes the integers returned.
        """
        return depth_array(self._halves[Side(side)], levels)

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
            if order.otype is _LIMIT:
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
        tside = taker.side
        opp = self._halves[tside.opposite]
        live, tid, stp = self._live, self._tid, self.stp
        is_buy = tside is _BUY
        limit = taker.price
        trades: List[Trade] = []
        while taker.is_active:
            price = opp.best()
            if price is None:
                break
            if limit is not None and (price > limit if is_buy else price < limit):
                break
            maker = opp.head(price)
            if maker.agent_id == taker.agent_id:
                if stp == "cancel_incoming":
                    taker.cancel()
                    break
                self.cancel(maker.order_id)
                continue
            t_left = taker.qty - taker.filled
            m_left = maker.qty - maker.filled
            lots = t_left if t_left < m_left else m_left
            taker.filled += lots  # == Order.fill(lots): 0 < lots <= remaining
            maker.filled += lots
            opp.reduce(price, lots)
            if lots == m_left:
                del live[maker.order_id]
            if is_buy:
                trades.append(Trade(tid(), taker.time, price, lots, taker.order_id,
                                    maker.order_id, taker.agent_id, maker.agent_id, tside))
            else:
                trades.append(Trade(tid(), taker.time, price, lots, maker.order_id,
                                    taker.order_id, maker.agent_id, taker.agent_id, tside))
        return trades
