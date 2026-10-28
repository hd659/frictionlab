import numpy as np
import pytest

from frictionlab.core.types import (
    Account, IdGen, Order, OrderType, Side, Trade, from_ticks, settle, to_ticks,
)


def _trade(price=100, qty=5, buyer=1, seller=2, tid=1):
    return Trade(tid, 0.0, price, qty, 10, 20, buyer, seller, Side.BUY)


def test_tick_conversion_roundtrip_and_validation():
    assert to_ticks(100.25, 0.05) == 2005
    assert from_ticks(2005, 0.05) == pytest.approx(100.25)
    assert to_ticks(1.0000001, 0.01) == 100
    with pytest.raises(ValueError):
        to_ticks(1.0, 0.0)
    with pytest.raises(TypeError):
        from_ticks(1.5, 0.01)


def test_side_opposite_and_sign():
    assert Side.BUY.opposite is Side.SELL
    assert Side.SELL.opposite is Side.BUY
    assert int(Side.BUY) == 1 and int(Side.SELL) == -1


def test_idgen_is_monotone():
    gen = IdGen()
    assert [gen() for _ in range(3)] == [1, 2, 3]
    assert IdGen(start=10)() == 10


def test_order_validation():
    with pytest.raises(ValueError):
        Order(1, 1, Side.BUY, 5)  # limit without price
    with pytest.raises(ValueError):
        Order(1, 1, Side.BUY, 5, price=100, otype=OrderType.MARKET)
    with pytest.raises(ValueError):
        Order(1, 1, Side.BUY, 0, price=100)
    with pytest.raises(ValueError):
        Order(1, 1, Side.BUY, 5, price=-1)
    with pytest.raises(TypeError):
        Order(1, 1, Side.BUY, 5, price=100.5)
    with pytest.raises(TypeError):
        Order(1, 1, Side.BUY, True, price=100)
    o = Order(1, 1, 1, np.int64(5), price=np.int64(100))
    assert o.side is Side.BUY and type(o.qty) is int and type(o.price) is int


def test_partial_fills_and_overfill_guard():
    o = Order(1, 7, Side.SELL, 10, price=101)
    assert o.fill(3) == 7 and o.filled == 3 and o.is_active
    with pytest.raises(ValueError):
        o.fill(8)
    assert o.fill(7) == 0 and not o.is_active
    with pytest.raises(ValueError):
        o.fill(1)


def test_cancel_returns_open_remainder_once():
    o = Order(1, 7, Side.BUY, 10, price=99)
    o.fill(4)
    assert o.cancel() == 6
    assert o.remaining == 0 and not o.is_active
    assert o.cancel() == 0
    with pytest.raises(ValueError):
        o.fill(1)


def test_crosses_semantics():
    bid = Order(1, 1, Side.BUY, 1, price=100)
    ask = Order(2, 2, Side.SELL, 1, price=100)
    assert bid.crosses(100) and bid.crosses(99) and not bid.crosses(101)
    assert ask.crosses(100) and ask.crosses(101) and not ask.crosses(99)
    mkt = Order(3, 3, Side.BUY, 1, otype=OrderType.MARKET)
    assert mkt.crosses(10 ** 9)


def test_orders_compare_by_identity():
    a = Order(1, 1, Side.BUY, 1, price=100)
    b = Order(1, 1, Side.BUY, 1, price=100)
    assert a != b and a == a


def test_trade_validation_and_notional():
    t = _trade(price=250, qty=4)
    assert t.notional == 1000
    with pytest.raises(ValueError):
        _trade(buyer=3, seller=3)
    with pytest.raises(ValueError):
        _trade(qty=0)
    with pytest.raises(Exception):
        t.price = 1  # frozen


def test_account_fill_accounting_with_fee_and_rebate():
    acc = Account(1, cash=10_000)
    acc.apply_fill(Side.BUY, 100, 10, fee=5)
    assert acc.position == 10 and acc.cash == 10_000 - 1000 - 5
    acc.apply_fill(Side.SELL, 110, 10, fee=-2)  # maker rebate
    assert acc.position == 0 and acc.cash == 10_000 - 1005 + 1100 + 2
    assert acc.fees_paid == 3 and acc.volume == 20 and acc.n_trades == 2
    assert acc.equity(mark=500) == acc.cash


def test_account_rejects_float_cash():
    with pytest.raises(TypeError):
        Account(1, cash=1.5)


def test_short_position_mark_to_market():
    acc = Account(1)
    acc.apply_fill(Side.SELL, 100, 3)
    assert acc.position == -3 and acc.cash == 300
    assert acc.equity(90) == 30 and acc.equity(110) == -30


def test_settle_conserves_cash_and_inventory():
    rng = np.random.default_rng(42)
    accts = {i: Account(i, cash=1_000_000, position=100) for i in range(5)}
    cash0 = sum(a.cash for a in accts.values())
    pos0 = sum(a.position for a in accts.values())
    total_fees = 0
    for tid in range(200):
        b, s = rng.choice(5, size=2, replace=False)
        price, qty = int(rng.integers(90, 111)), int(rng.integers(1, 20))
        fb, fs = int(rng.integers(-2, 6)), int(rng.integers(-2, 6))
        settle(_trade(price, qty, int(b), int(s), tid), accts[int(b)], accts[int(s)], fb, fs)
        total_fees += fb + fs
    assert sum(a.position for a in accts.values()) == pos0
    assert sum(a.cash for a in accts.values()) == cash0 - total_fees
    assert sum(a.fees_paid for a in accts.values()) == total_fees


def test_settle_rejects_mismatched_accounts():
    with pytest.raises(ValueError):
        settle(_trade(buyer=1, seller=2), Account(2), Account(1))
