import numpy as np
import pytest

from frictionlab.core import (
    IdGen, InvariantViolation, Ledger, LedgerError, Order, OrderBook, OrderType, Side, Trade,
)
from frictionlab.costs import FixedCost, HalfSpreadCost, MakerTakerFee, zero_friction


def trade(tid=1, price=1000, qty=10, buyer=1, seller=2, aggressor=Side.BUY):
    return Trade(tid, 0.0, price, qty, 100 + tid, 200 + tid, buyer, seller, aggressor)


def ledger(frictions=None, **kw):
    lg = Ledger(frictions, **kw)
    lg.open_account(1, cash=100_000)
    lg.open_account(2, cash=50_000, position=100)
    return lg


def test_frictionless_settlement_moves_cash_and_inventory():
    lg = ledger(zero_friction())
    rec = lg.settle(trade())
    assert (lg[1].cash, lg[1].position) == (90_000, 10)
    assert (lg[2].cash, lg[2].position) == (60_000, 90)
    assert rec.total_fee == 0 and lg.fees_collected == 0
    lg.check()


def test_fees_route_to_venue_sink_and_conserve_cash():
    stack = MakerTakerFee(maker_bps=0, taker_bps=10) + FixedCost(3)
    lg = ledger(stack)
    rec = lg.settle(trade())  # buyer is taker: 10bps of 10_000 = 10, + 3 each
    assert (rec.buyer_fee, rec.seller_fee) == (13, 3)
    assert lg.venue.cash == lg.fees_collected == 16
    assert lg[1].cash == 100_000 - 10_000 - 13
    assert sum(a.cash for a in lg) + lg.venue.cash == lg.cash_in
    lg.check()


def test_maker_rebate_is_paid_by_venue():
    lg = ledger(MakerTakerFee(maker_bps=-2, taker_bps=5))
    rec = lg.settle(trade(price=10_000, qty=10))  # notional 100_000
    assert (rec.buyer_fee, rec.seller_fee) == (50, -20)
    assert lg.venue.cash == 30 and lg[2].fees_paid == -20
    lg.check()


def test_explicit_fees_override_frictions():
    lg = ledger(FixedCost(100))
    rec = lg.settle(trade(), fees=(1, 2))
    assert rec.total_fee == 3 and lg.fees_collected == 3


def test_total_equity_drops_by_exactly_fees():
    lg = ledger(HalfSpreadCost(4) + FixedCost(1))
    before = lg.total_equity(1000)
    lg.settle_all([trade(1), trade(2, aggressor=Side.SELL)])
    assert lg.total_equity(1000) == before - lg.fees_collected
    assert lg.fees_collected == 2 * (20 + 2)


def test_double_settlement_and_unknown_accounts_rejected():
    lg = ledger()
    lg.settle(trade(1))
    with pytest.raises(LedgerError, match="already settled"):
        lg.settle(trade(1))
    with pytest.raises(LedgerError, match="unknown account"):
        lg.settle(trade(2, buyer=9))
    assert len(lg.journal) == 1


def test_account_registry_rules():
    lg = ledger(venue_id=0)
    with pytest.raises(LedgerError, match="reserved"):
        lg.open_account(0)
    with pytest.raises(LedgerError, match="already exists"):
        lg.open_account(1)
    with pytest.raises(TypeError):
        lg.open_account(3, cash=1.5)
    assert 3 not in lg and len(lg) == 2


def test_no_short_rejects_atomically():
    lg = ledger(FixedCost(5), allow_short=False)
    snap = (lg[1].cash, lg[2].cash, lg[2].position, lg.venue.cash)
    with pytest.raises(LedgerError, match="lacks inventory"):
        lg.settle(trade(qty=101, price=1))
    assert (lg[1].cash, lg[2].cash, lg[2].position, lg.venue.cash) == snap
    assert lg.journal == [] and lg.settle(trade(qty=100, price=1)).qty == 100


def test_no_overdraft_includes_fee():
    lg = Ledger(FixedCost(1), allow_overdraft=False)
    lg.open_account(1, cash=10_000)
    lg.open_account(2, position=10)
    with pytest.raises(LedgerError, match="cannot fund"):
        lg.settle(trade())  # needs 10_000 + 1 fee
    lg.deposit(1, cash=1)
    lg.settle(trade())
    assert lg[1].cash == 0
    with pytest.raises(LedgerError, match="overdraws"):
        lg.deposit(1, cash=-1)
    with pytest.raises(LedgerError, match="overdraws"):
        lg.open_account(3, cash=-5)
    assert 3 not in lg


def test_seller_fee_exceeding_proceeds_rejected():
    lg = Ledger(FixedCost(2_000), allow_overdraft=False)
    lg.open_account(1, cash=100_000)
    lg.open_account(2, position=10)
    with pytest.raises(LedgerError, match="cannot pay fee"):
        lg.settle(trade(price=10, qty=10))


def test_tampering_is_detected():
    lg = ledger(FixedCost(2))
    lg.settle(trade())
    lg[1].cash += 1
    with pytest.raises(InvariantViolation, match="cash not conserved"):
        lg.check()
    lg[1].cash -= 1
    lg.venue.position = 1
    assert any("venue holds" in v for v in lg.violations())


def test_withdrawals_are_external_flows():
    lg = ledger()
    lg.deposit(2, cash=-50_000, position=-100)
    assert lg.cash_in == 100_000 and lg.inventory_in == 0
    lg.check()


def test_random_book_flow_conserves_with_frictions():
    rng = np.random.default_rng(7)
    stack = MakerTakerFee(maker_bps=-1, taker_bps=7) + HalfSpreadCost(1) + FixedCost(2)
    lg = Ledger(stack, check_every=True)
    for aid in range(1, 9):
        lg.open_account(aid, cash=int(rng.integers(0, 10**6)), position=int(rng.integers(0, 500)))
    book, oid = OrderBook(), IdGen()
    for t in range(600):
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        aid, qty = int(rng.integers(1, 9)), int(rng.integers(1, 20))
        if rng.random() < 0.2:
            o = Order(oid(), aid, side, qty, None, float(t), OrderType.MARKET)
        else:
            o = Order(oid(), aid, side, qty, int(rng.integers(95, 106)), float(t))
        lg.settle_all(book.submit(o))
    assert len(lg.journal) == len(book.trades) > 50
    assert lg.fees_collected == sum(stack.trade_fees(t)[0] + stack.trade_fees(t)[1]
                                    for t in book.trades)
    assert lg.total_equity(100) == lg.cash_in + 100 * lg.inventory_in - lg.fees_collected
