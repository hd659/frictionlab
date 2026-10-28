import numpy as np
import pytest

from frictionlab.core.order_book import OrderBook
from frictionlab.core.types import IdGen, Order, OrderType, Side


@pytest.fixture
def ids():
    return IdGen()


def lim(ids, agent, side, qty, price, t=0.0):
    return Order(ids(), agent, side, qty, price=price, time=t)


def mkt(ids, agent, side, qty, t=0.0):
    return Order(ids(), agent, side, qty, time=t, otype=OrderType.MARKET)


def test_empty_book_queries():
    book = OrderBook()
    assert book.best_bid is None and book.best_ask is None
    assert book.spread is None and book.mid is None and len(book) == 0
    assert book.submit(mkt(IdGen(), 1, Side.BUY, 5)) == []


def test_resting_orders_and_top_of_book(ids):
    book = OrderBook()
    for p in (99, 98, 99):
        book.submit(lim(ids, 1, Side.BUY, 5, p))
    for p in (101, 103):
        book.submit(lim(ids, 2, Side.SELL, 4, p))
    assert book.best_bid == 99 and book.best_ask == 101
    assert book.spread == 2 and book.mid == 100.0
    assert book.depth(Side.BUY) == [(99, 10), (98, 5)]
    assert book.depth(Side.SELL, levels=1) == [(101, 4)]
    assert book.volume_at(Side.BUY, 99) == 10 and book.volume_at(Side.BUY, 50) == 0
    assert len(book) == 5 and book.trades == []


def test_price_priority_and_trade_at_resting_price(ids):
    book = OrderBook()
    a1 = lim(ids, 1, Side.SELL, 3, 102)
    a2 = lim(ids, 2, Side.SELL, 3, 101)
    book.submit(a1), book.submit(a2)
    trades = book.submit(lim(ids, 9, Side.BUY, 4, 105))
    assert [(t.price, t.qty, t.sell_order_id) for t in trades] == [
        (101, 3, a2.order_id), (102, 1, a1.order_id)]
    assert all(t.aggressor is Side.BUY and t.buyer_id == 9 for t in trades)
    assert book.best_ask == 102 and book.volume_at(Side.SELL, 102) == 2
    assert a2.order_id not in book and a1.remaining == 2


def test_time_priority_fifo_within_level(ids):
    book = OrderBook()
    first, second = lim(ids, 1, Side.BUY, 2, 100), lim(ids, 2, Side.BUY, 2, 100)
    book.submit(first), book.submit(second)
    trades = book.submit(lim(ids, 3, Side.SELL, 3, 100))
    assert [(t.buy_order_id, t.qty) for t in trades] == [
        (first.order_id, 2), (second.order_id, 1)]
    assert second.remaining == 1 and book.depth(Side.BUY) == [(100, 1)]


def test_limit_remainder_rests_after_partial_fill(ids):
    book = OrderBook()
    book.submit(lim(ids, 1, Side.SELL, 2, 100))
    taker = lim(ids, 2, Side.BUY, 5, 100)
    trades = book.submit(taker)
    assert sum(t.qty for t in trades) == 2
    assert book.get(taker.order_id) is taker and taker.remaining == 3
    assert book.best_bid == 100 and book.best_ask is None


def test_non_crossing_limit_does_not_trade(ids):
    book = OrderBook()
    book.submit(lim(ids, 1, Side.SELL, 2, 101))
    assert book.submit(lim(ids, 2, Side.BUY, 2, 100)) == []
    assert book.spread == 1


def test_market_order_sweeps_and_remainder_is_cancelled(ids):
    book = OrderBook()
    book.submit(lim(ids, 1, Side.BUY, 2, 100))
    book.submit(lim(ids, 1, Side.BUY, 2, 99))
    m = mkt(ids, 2, Side.SELL, 10)
    trades = book.submit(m)
    assert [(t.price, t.qty) for t in trades] == [(100, 2), (99, 2)]
    assert m.cancelled and m.remaining == 0 and m.order_id not in book
    assert book.best_bid is None and len(book) == 0


def test_full_and_partial_cancel(ids):
    book = OrderBook()
    a, b = lim(ids, 1, Side.BUY, 5, 100), lim(ids, 2, Side.BUY, 5, 100)
    book.submit(a), book.submit(b)
    assert book.cancel(a.order_id, qty=3) == 3
    assert a.remaining == 2 and book.volume_at(Side.BUY, 100) == 7
    trades = book.submit(lim(ids, 3, Side.SELL, 2, 100))
    assert trades[0].buy_order_id == a.order_id  # partial cancel keeps priority
    assert book.cancel(b.order_id) == 5 and b.cancelled
    assert book.cancel(b.order_id) == 0 and book.cancel(12345) == 0
    assert book.best_bid is None and len(book) == 0
    c = lim(ids, 1, Side.BUY, 1, 90)
    book.submit(c)
    with pytest.raises(ValueError):
        book.cancel(c.order_id, qty=0)


def test_cancelled_head_is_skipped_lazily(ids):
    book = OrderBook()
    a, b = lim(ids, 1, Side.SELL, 3, 101), lim(ids, 2, Side.SELL, 3, 101)
    book.submit(a), book.submit(b)
    book.cancel(a.order_id)
    trades = book.submit(lim(ids, 3, Side.BUY, 1, 101))
    assert trades[0].sell_order_id == b.order_id and trades[0].qty == 1


def test_self_trade_prevention_modes(ids):
    book = OrderBook(stp="cancel_resting")
    own = lim(ids, 1, Side.SELL, 2, 100)
    other = lim(ids, 2, Side.SELL, 2, 101)
    book.submit(own), book.submit(other)
    trades = book.submit(lim(ids, 1, Side.BUY, 2, 101))
    assert own.cancelled and [(t.seller_id, t.price) for t in trades] == [(2, 101)]

    book = OrderBook(stp="cancel_incoming")
    book.submit(lim(ids, 1, Side.SELL, 2, 100))
    taker = lim(ids, 1, Side.BUY, 2, 100)
    assert book.submit(taker) == [] and taker.cancelled
    assert book.best_ask == 100 and taker.order_id not in book
    with pytest.raises(ValueError):
        OrderBook(stp="nope")


def test_rejects_duplicate_and_inactive_orders(ids):
    book = OrderBook()
    o = lim(ids, 1, Side.BUY, 1, 100)
    book.submit(o)
    with pytest.raises(ValueError):
        book.submit(o)
    dead = lim(ids, 1, Side.BUY, 1, 100)
    dead.cancel()
    with pytest.raises(ValueError):
        book.submit(dead)


def test_randomized_invariants_conservation_and_no_cross():
    rng = np.random.default_rng(7)
    ids, book, orders = IdGen(), OrderBook(), []
    for step in range(3000):
        r = rng.random()
        if r < 0.15 and len(book):
            oid = int(rng.choice(list(book._live)))
            if rng.random() < 0.5:
                book.cancel(oid)
            else:
                book.cancel(oid, qty=int(rng.integers(1, 4)))
            continue
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        qty = int(rng.integers(1, 10))
        if r < 0.25:
            o = mkt(ids, int(rng.integers(0, 20)), side, qty, t=step)
        else:
            o = lim(ids, int(rng.integers(0, 20)), side, qty,
                    int(rng.integers(95, 106)), t=step)
        orders.append(o)
        book.submit(o)
        if book.best_bid is not None and book.best_ask is not None:
            assert book.best_bid < book.best_ask
    resting = sum(q for s in Side for _, q in book.depth(s))
    assert resting == sum(o.remaining for o in book._live.values())
    for s in Side:
        prices = [p for p, _ in book.depth(s)]
        assert prices == sorted(prices, reverse=(s is Side.BUY))
    # every traded lot is a fill on exactly one buy and one sell order
    assert sum(o.filled for o in orders) == 2 * sum(t.qty for t in book.trades)
    for s in Side:
        assert sum(o.filled for o in orders if o.side is s) == sum(
            t.qty for t in book.trades)
    live = {o.order_id for o in orders if o.is_active}
    assert live == set(book._live)
    assert all(t.buyer_id != t.seller_id for t in book.trades)
    assert [t.trade_id for t in book.trades] == list(range(1, len(book.trades) + 1))
