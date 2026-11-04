"""Frequent batch auction (FBA): the book's order-intent interface, cleared in batches.

Orders collected during one batch interval clear together at one uniform
integer-tick price p*. With demand D(p) = lots bid at >= p and supply
S(p) = lots offered at <= p (market orders always count), the executable
volume is V(p) = min(D(p), S(p)) and

    p* = argmax V(p);  ties -> min |D(p) - S(p)| -> min |p - ref| -> lower p

over the integers between the lowest and highest limit price. `ref` is the
previous clearing price (first batch: midpoint of the tied region).

Clearing is one sorted sweep (fl-041). Bids and asks are aggregated into
integer-tick demand / supply maps; D starts at all bid lots and drops by the
lots at b when p passes b + 1, S grows by the lots at a when p reaches a, so
cumulative demand (descending) and supply (ascending) are pure-Python running
sums over the sorted change points. Each constant segment is scored once; the
integer closest to `ref` inside the tied segments wins. The pre-fl-041 version
is frozen in tests/reference_batch_auction.py and must agree exactly.

Allocation walks each side from the most aggressive level (market first,
then price; FIFO by arrival): better levels fill in full, the marginal level
shares the remainder pro-rata, floor(R * q_i / Q) each plus one lot to the
largest fractional remainders (ties: earlier arrival), so exactly V lots are
allocated per side. Only orders that can execute at p* are sorted. Allocations
are paired in priority order into `Trade`s at p*; the later arrival is the
aggressor (taker). Lots an agent would cross with itself are netted out.

After a batch, market remainders are cancelled; limit remainders carry into
the next batch with their arrival priority (`carry=True`) or are cancelled.
`BatchVenue` settles each batch through the friction stack and ledger like
book trades; `batch_runner` drives it from `MarketRunner` on a fixed grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..agents.base import Venue
from ..core.types import IdGen, Order, OrderType, Side, Trade, _check_int

Quote = Tuple[Optional[int], int]  # (limit price or None for market, lots)
_INF = float("inf")


@dataclass(frozen=True)
class Clearing:
    """Result of one batch: uniform clearing price (None if nothing crossed), volume, order count and trades."""
    time: float
    price: Optional[int]   # None if nothing crossed
    volume: int            # lots executed (after self-trade netting)
    n_orders: int          # orders in the batch
    trades: Tuple[Trade, ...]


def _limit(o: Order) -> float:
    if o.price is not None:
        return o.price
    return _INF if o.side is Side.BUY else 0.0


def _bid_key(o: Order):
    return (-_INF if o.price is None else -o.price, o.time, o.order_id)


def _ask_key(o: Order):
    return (0.0 if o.price is None else o.price, o.time, o.order_id)


def _sweep(dem: Dict[int, int], sup: Dict[int, int], mb: int, ms: int,
           ref: Optional[int]) -> Tuple[Optional[int], int]:
    """Clearing price from aggregated limit lots plus market lots `mb` / `ms`."""
    if dem or sup:
        keys = (dem.keys() | sup.keys()) if dem and sup else (dem or sup).keys()
        lo, hi = min(keys), max(keys)
    elif ref is None:
        return None, 0
    else:
        lo = hi = ref
    step: Dict[int, List[int]] = {}
    for b, q in dem.items():
        if b < hi:
            step.setdefault(b + 1, [0, 0])[0] += q
    for a, q in sup.items():
        if a > lo:
            step.setdefault(a, [0, 0])[1] += q
    d, s = mb + sum(dem.values()), ms + sup.get(lo, 0)
    best, tied, start = None, [], lo
    for x in sorted(step) + [hi + 1]:
        v = d if d < s else s
        if v:
            key = (v, -abs(d - s))
            if best is None or key > best:
                best, tied = key, [(start, x - 1)]
            elif key == best:
                tied.append((start, x - 1))
        if x <= hi:
            dd, ds = step[x]
            d, s, start = d - dd, s + ds, x
    if best is None:
        return None, 0
    r = ref if ref is not None else (tied[0][0] + tied[-1][1]) // 2
    p = min((min(max(r, a), b) for a, b in tied), key=lambda c: (abs(c - r), c))
    return p, best[0]


def _aggregate(quotes: Sequence[Quote]) -> Tuple[Dict[int, int], int]:
    agg: Dict[int, int] = {}
    mkt = 0
    for p, q in quotes:
        if p is None:
            mkt += q
        else:
            agg[p] = agg.get(p, 0) + q
    return agg, mkt


def clearing_price(bids: Sequence[Quote], asks: Sequence[Quote],
                   ref: Optional[int] = None) -> Tuple[Optional[int], int]:
    """(p*, V) for the batch; (None, 0) if no positive volume can execute."""
    (dem, mb), (sup, ms) = _aggregate(bids), _aggregate(asks)
    return _sweep(dem, sup, mb, ms, ref)


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


class BatchAuction:
    """Duck-types `OrderBook` (submit/cancel/queries); trades print on `clear`."""

    def __init__(self, trade_ids: Optional[IdGen] = None, *, carry: bool = True,
                 last_price: Optional[int] = None) -> None:
        self._tid = trade_ids or IdGen()
        self.carry = carry
        self.last_price = last_price
        self._live: Dict[int, Order] = {}
        self.trades: List[Trade] = []
        self.history: List[Clearing] = []

    # ---- queries (pending batch) --------------------------------------
    def depth(self, side: Side, levels: Optional[int] = None) -> List[Tuple[int, int]]:
        side, agg = Side(side), {}
        for o in self._live.values():
            if o.side is side and o.price is not None:
                q = o.remaining
                if q > 0:
                    agg[o.price] = agg.get(o.price, 0) + q
        out = sorted(agg.items(), reverse=side is Side.BUY)  # best level first
        return out if levels is None else out[:levels]

    def _best(self, side: Side) -> Optional[int]:
        d = self.depth(side, 1)
        return d[0][0] if d else None

    best_bid = property(lambda self: self._best(Side.BUY))
    best_ask = property(lambda self: self._best(Side.SELL))

    @property
    def spread(self) -> Optional[int]:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else a - b

    @property
    def mid(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else (a + b) / 2

    def volume_at(self, side: Side, price: int) -> int:
        return dict(self.depth(side)).get(price, 0)

    def get(self, order_id: int) -> Optional[Order]:
        return self._live.get(order_id)

    def __contains__(self, order_id: int) -> bool:
        return order_id in self._live

    def __len__(self) -> int:
        return len(self._live)

    # ---- mutations -----------------------------------------------------
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
        bids: List[Order] = []
        asks: List[Order] = []
        dem: Dict[int, int] = {}
        sup: Dict[int, int] = {}
        drop: List[Order] = []          # inactive or market: never carried
        mb = ms = 0
        buy = Side.BUY
        for o in self._live.values():   # one pass: aggregate, no per-order sort
            q, p = o.remaining, o.price
            if q <= 0 or p is None:
                drop.append(o)
                if q <= 0:
                    continue
            if o.side is buy:
                bids.append(o)
                if p is None:
                    mb += q
                else:
                    dem[p] = dem.get(p, 0) + q
            else:
                asks.append(o)
                if p is None:
                    ms += q
                else:
                    sup[p] = sup.get(p, 0) + q
        p, vol = _sweep(dem, sup, mb, ms, self.last_price)
        trades: List[Trade] = []
        if vol:
            eb = sorted((o for o in bids if o.price is None or o.price >= p), key=_bid_key)
            ea = sorted((o for o in asks if o.price is None or o.price <= p), key=_ask_key)
            trades = self._pair(_allocate(eb, vol), _allocate(ea, vol), p, float(time))
            drop += [o for o in eb + ea if o.price is not None and not o.is_active]
        if trades:
            self.last_price = p
        # only non-executable limits are untouched by a clear, so with `carry` the
        # drop list is exactly what a full scan of the live set would remove
        for o in (list(self._live.values()) if not self.carry else drop):
            o.cancel()
            del self._live[o.order_id]
        self.trades.extend(trades)
        c = Clearing(float(time), p if trades else None, sum(t.qty for t in trades),
                     len(bids) + len(asks), tuple(trades))
        self.history.append(c)
        return c

    def _pair(self, buys, sells, p: int, time: float) -> List[Trade]:
        out: List[Trade] = []
        k, n = 0, len(sells)      # sells[:k] are exhausted: lots only ever decrease
        for b in buys:
            aid = b[0].agent_id
            while b[1]:
                while not sells[k][1]:
                    k += 1
                j = k
                while j < n and (not sells[j][1] or sells[j][0].agent_id == aid):
                    j += 1
                own = j == n
                s = sells[k] if own else sells[j]  # only own orders left: net out
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
                                 aid, so.agent_id, aggr))
        return out


@dataclass
class BatchVenue(Venue):
    """`Venue` over a `BatchAuction`: intents queue, `clear` settles the batch."""

    def clear(self, time: float) -> Clearing:
        c = self.book.clear(time)
        self.ledger.settle_all(c.trades)
        return c


def batch_runner(agents, frictions=None, *, interval: float = 1.0, carry: bool = True,
                 start: Optional[float] = None, **kw):
    """`MarketRunner` whose venue is a `BatchVenue` cleared every `interval`."""
    from ..core.runner import MarketRunner  # lazy: runner imports markets
    if kw.get("pools"):
        raise ValueError("batch_runner does not support AMM pools")
    if not interval > 0:
        raise ValueError(f"interval must be > 0, got {interval}")
    r = MarketRunner(agents, frictions, **kw)
    old = r.venue
    r.book = BatchAuction(IdGen(), carry=carry)
    r.venue = BatchVenue(r.book, r.ledger, old.order_ids, old.cost_sink_id, old.depth, old.tape)

    def on_batch(clock, ev) -> None:
        r.venue.clear(clock.now)
        clock.schedule(interval, "batch", None, -1)

    r.clock.on("batch", on_batch)
    r.clock.schedule_at(interval if start is None else start, "batch", None, -1)
    return r
