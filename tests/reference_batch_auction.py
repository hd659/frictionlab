"""FROZEN ORACLE: the pre-fl-041 batch auction clear, verbatim.

Do not optimise or "fix" this file. tests/test_batch_auction_oracle.py replays
seeded random batches through `ReferenceBatchAuction` and through
`frictionlab.markets.BatchAuction` with twin orders; any difference in the
clearing price, volume, trade list or per-order fill is a regression. Only the
class name differs from the original, and the query helpers the oracle test
does not need (spread, mid, volume_at, ...) plus `BatchVenue` / `batch_runner`
were dropped. Each clear re-sorts every live order by `_priority`, and
`clearing_price` evaluates D and S by bisection once per constant segment.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import accumulate
from typing import Dict, List, Optional, Sequence, Tuple

from frictionlab.core.types import IdGen, Order, OrderType, Side, Trade, _check_int

Quote = Tuple[Optional[int], int]  # (limit price or None for market, lots)


@dataclass(frozen=True)
class Clearing:
    time: float
    price: Optional[int]   # None if nothing crossed
    volume: int            # lots executed (after self-trade netting)
    n_orders: int          # orders in the batch
    trades: Tuple[Trade, ...]


def _limit(o: Order) -> float:
    if o.price is not None:
        return o.price
    return float("inf") if o.side is Side.BUY else 0.0


def _priority(o: Order):
    return (-o.side.value * _limit(o), o.time, o.order_id)


def clearing_price(bids: Sequence[Quote], asks: Sequence[Quote],
                   ref: Optional[int] = None) -> Tuple[Optional[int], int]:
    """(p*, V) for the batch; (None, 0) if no positive volume can execute."""
    bl = sorted((p, q) for p, q in bids if p is not None)
    al = sorted((p, q) for p, q in asks if p is not None)
    mb = sum(q for p, q in bids if p is None)
    ms = sum(q for p, q in asks if p is None)
    prices = [p for p, _ in bl + al] or ([] if ref is None else [ref])
    if not prices:
        return None, 0
    lo, hi = min(prices), max(prices)
    bp, bsuf = [p for p, _ in bl], list(accumulate(q for _, q in reversed(bl)))[::-1] + [0]
    ap, apre = [p for p, _ in al], [0] + list(accumulate(q for _, q in al))
    D = lambda p: mb + bsuf[bisect_left(bp, p)]
    S = lambda p: ms + apre[bisect_right(ap, p)]
    starts = sorted({lo} | {p + 1 for p in bp if p + 1 <= hi} | {p for p in ap if p > lo})
    segs = [(s, (starts[i + 1] - 1) if i + 1 < len(starts) else hi)
            for i, s in enumerate(starts)]
    score = {s: (-min(D(s), S(s)), abs(D(s) - S(s))) for s, _ in segs}
    best = min(score.values())
    if best[0] == 0:
        return None, 0
    tied = [(s, e) for s, e in segs if score[s] == best]
    r = ref if ref is not None else (tied[0][0] + tied[-1][1]) // 2
    p = min((min(max(r, s), e) for s, e in tied), key=lambda c: (abs(c - r), c))
    return p, -best[0]


def pro_rata(total: int, qtys: Sequence[int]) -> List[int]:
    """Split `total` lots over `qtys` exactly; never exceeds any q_i."""
    Q = sum(qtys)
    if total >= Q:
        return list(qtys)
    out = [total * q // Q for q in qtys]
    rank = sorted(range(len(qtys)), key=lambda i: (-(total * qtys[i] % Q), i))
    for i in rank[:total - sum(out)]:
        out[i] += 1
    return out


def _allocate(orders: List[Order], volume: int) -> List[List]:
    out, left, i = [], volume, 0
    while left > 0 and i < len(orders):
        j = i
        while j < len(orders) and _limit(orders[j]) == _limit(orders[i]):
            j += 1
        lots = pro_rata(left, [o.remaining for o in orders[i:j]])
        out += [[o, n] for o, n in zip(orders[i:j], lots) if n]
        left, i = left - sum(lots), j
    return out


class ReferenceBatchAuction:
    """Pre-fl-041 `BatchAuction`: submit/cancel/depth/clear, unchanged."""

    def __init__(self, trade_ids: Optional[IdGen] = None, *, carry: bool = True,
                 last_price: Optional[int] = None) -> None:
        self._tid = trade_ids or IdGen()
        self.carry = carry
        self.last_price = last_price
        self._live: Dict[int, Order] = {}
        self.trades: List[Trade] = []
        self.history: List[Clearing] = []

    def _orders(self, side: Side) -> List[Order]:
        return sorted((o for o in self._live.values() if o.side is side and o.is_active),
                      key=_priority)

    def depth(self, side: Side, levels: Optional[int] = None) -> List[Tuple[int, int]]:
        agg: Dict[int, int] = {}
        for o in self._orders(Side(side)):
            if o.price is not None:
                agg[o.price] = agg.get(o.price, 0) + o.remaining
        out = list(agg.items())  # dicts keep priority order: best level first
        return out if levels is None else out[:levels]

    def __len__(self) -> int:
        return len(self._live)

    def submit(self, order: Order) -> List[Trade]:
        if order.order_id in self._live:
            raise ValueError(f"duplicate order id {order.order_id}")
        if not order.is_active:
            raise ValueError(f"order {order.order_id} is not active")
        self._live[order.order_id] = order
        return []  # nothing prints until the batch clears

    def cancel(self, order_id: int, qty: Optional[int] = None) -> int:
        order = self._live.get(order_id)
        if order is None:
            return 0
        if qty is None or _check_int("cancel qty", qty, positive=True) >= order.remaining:
            del self._live[order_id]
            return order.cancel()
        order.qty -= qty
        return qty

    def clear(self, time: float) -> Clearing:
        bids, asks = self._orders(Side.BUY), self._orders(Side.SELL)
        n = len(bids) + len(asks)
        p, vol = clearing_price([(o.price, o.remaining) for o in bids],
                                [(o.price, o.remaining) for o in asks], self.last_price)
        trades: List[Trade] = []
        if vol:
            buys = _allocate([o for o in bids if _limit(o) >= p], vol)
            sells = _allocate([o for o in asks if _limit(o) <= p], vol)
            trades = self._pair(buys, sells, p, float(time))
        if trades:
            self.last_price = p
        for oid, o in list(self._live.items()):
            if not o.is_active or o.otype is OrderType.MARKET or not self.carry:
                o.cancel()
                del self._live[oid]
        self.trades.extend(trades)
        c = Clearing(float(time), p if trades else None, sum(t.qty for t in trades),
                     n, tuple(trades))
        self.history.append(c)
        return c

    def _pair(self, buys, sells, p: int, time: float) -> List[Trade]:
        out: List[Trade] = []
        for b in buys:
            while b[1]:
                s = next((s for s in sells if s[1] and s[0].agent_id != b[0].agent_id), None)
                own = s is None
                s = s or next(s for s in sells if s[1])  # only own orders left: net out
                lots = min(b[1], s[1])
                b[1] -= lots
                s[1] -= lots
                if own:
                    continue
                bo, so = b[0], s[0]
                bo.fill(lots)
                so.fill(lots)
                aggr = Side.BUY if (bo.time, bo.order_id) > (so.time, so.order_id) else Side.SELL
                out.append(Trade(self._tid(), time, p, lots, bo.order_id, so.order_id,
                                 bo.agent_id, so.agent_id, aggr))
        return out
