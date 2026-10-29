"""PerpMarket: funding direction, clamp, conservation, OI balance, runner."""
from fractions import Fraction

import numpy as np
import pytest

from frictionlab.agents import FundamentalProcess, make_zi_population
from frictionlab.core import Ledger, LedgerError, Order, OrderBook, Side, IdGen
from frictionlab.core.runner import MarketRunner
from frictionlab.markets import PERP_SINK_ID, PerpMarket

LONG, SHORT, MAKER = 1, 2, 3


def make(index=100, k=1, cap_bps=75, cash=10**6):
    lg = Ledger(check_every=True)
    for aid in (LONG, SHORT, MAKER):
        lg.open_account(aid, cash=cash)
    book = OrderBook(trade_ids=IdGen())
    perp = PerpMarket(lg, FundamentalProcess.constant(index), book, k=k, cap_bps=cap_bps)
    return lg, book, perp


_oid = IdGen()


def trade(book, lg, buyer, seller, qty, price):
    book.submit(Order(_oid(), seller, Side.SELL, qty, price))
    lg.settle_all(book.submit(Order(_oid(), buyer, Side.BUY, qty, price)))


def quote(book, bid, ask):
    book.submit(Order(_oid(), MAKER, Side.BUY, 1, bid))
    book.submit(Order(_oid(), MAKER, Side.SELL, 1, ask))


def total_cash(lg):
    return sum(a.cash for a in lg) + lg.venue.cash


def test_transfers_sum_to_zero_with_sink():
    lg, book, perp = make()
    trade(book, lg, LONG, SHORT, 7, 100)
    quote(book, 100, 103)                       # mid 101.5 -> premium 150 bps
    c0, s0 = total_cash(lg), perp.sink_cash
    ev = perp.fund(8.0)
    assert ev.payments[LONG] < 0 < ev.payments[SHORT]
    assert sum(ev.payments.values()) + ev.remainder == 0
    assert perp.sink_cash - s0 == ev.remainder >= 0
    assert total_cash(lg) == c0
    lg.check()


@pytest.mark.parametrize("bid,ask,long_pays", [(101, 103, True), (97, 99, False)])
def test_direction_follows_sign_of_premium(bid, ask, long_pays):
    lg, book, perp = make(cap_bps=10_000)
    trade(book, lg, LONG, SHORT, 5, 100)
    quote(book, bid, ask)
    l0, s0 = lg[LONG].cash, lg[SHORT].cash
    ev = perp.fund(8.0)
    assert (ev.premium_bps > 0) is long_pays
    if long_pays:
        assert lg[LONG].cash < l0 and lg[SHORT].cash > s0
    else:
        assert lg[LONG].cash > l0 and lg[SHORT].cash < s0
    assert ev.paid >= ev.received > 0


def test_rate_clamped_at_cap():
    lg, book, perp = make(k=2, cap_bps=30)
    trade(book, lg, LONG, SHORT, 10, 100)
    quote(book, 109, 111)                       # premium 1000 bps * k=2
    ev = perp.fund(8.0)
    assert ev.premium_bps == 1000 and ev.rate_bps == 30
    # owed = 10 * 110 * 30 / 10000 = 3.3 -> long pays 4, short gets 3
    assert ev.payments == {LONG: -4, SHORT: 3} and ev.remainder == 1
    _, rate = perp.funding_rate_bps(Fraction(50), 100)
    assert rate == -30


def test_zero_premium_zero_transfer():
    lg, book, perp = make()
    trade(book, lg, LONG, SHORT, 9, 100)
    quote(book, 99, 101)                        # mid == index
    cash = {a.agent_id: a.cash for a in lg}
    ev = perp.fund(8.0)
    assert ev.rate_bps == 0 and ev.payments == {} and ev.remainder == 0
    assert {a.agent_id: a.cash for a in lg} == cash


def test_positions_entry_and_open_interest_balanced():
    lg, book, perp = make()
    trade(book, lg, LONG, SHORT, 4, 100)
    trade(book, lg, LONG, MAKER, 2, 103)
    trade(book, lg, SHORT, LONG, 8, 98)         # LONG flips to -2
    perp.sync()
    assert perp.positions == {LONG: -2, SHORT: 4, MAKER: -2}
    assert perp.entry[LONG] == 98 and perp.entry[SHORT] == 98
    assert perp.entry[MAKER] == 103
    assert perp.open_interest == perp.short_interest == 4
    assert sum(perp.positions.values()) == 0
    assert perp.unrealized_pnl(SHORT, 100) == 4 * 2


def test_funding_rejected_atomically_without_overdraft():
    lg = Ledger(allow_overdraft=False)
    lg.open_account(LONG, cash=0)
    lg.open_account(SHORT, cash=0)
    perp = PerpMarket(lg, lambda t: 100, k=1, cap_bps=10_000)
    perp.positions.update({LONG: 5, SHORT: -5})
    with pytest.raises(LedgerError):
        perp.fund(8.0, mark=120)
    assert lg[LONG].cash == lg[SHORT].cash == 0 and perp.history == []


def _run(seed):
    agents = make_zi_population("zi-c", 4, 4, seed=seed, n_units=6)
    runner = MarketRunner(agents, seed=seed, cash=10**6, position=10, check_every=True)
    index = FundamentalProcess.random_walk(100, jump_rate=0.5, jump_size=2, seed=seed)
    perp = PerpMarket(runner.ledger, index, runner.book, k=1, cap_bps=50, interval=2.0)
    perp.attach(runner.clock)
    res = runner.run(until=40.0)
    perp.sync()
    return res, perp


def test_runner_invariants_and_balanced_oi():
    res, perp = _run(3)
    assert res.trades and len(perp.history) == 20
    assert perp.open_interest == perp.short_interest
    assert sum(perp.positions.values()) == 0
    remainders = sum(ev.remainder for ev in perp.history)
    assert perp.sink_cash == remainders >= 0
    for ev in perp.history:
        assert -50 <= ev.rate_bps <= 50
        assert sum(ev.payments.values()) + ev.remainder == 0
    res.ledger.check()
    assert PERP_SINK_ID in res.ledger


def test_deterministic_under_seed():
    (a, pa), (b, pb) = _run(11), _run(11)
    assert [(t.price, t.qty, t.buyer_id) for t in a.trades] == \
        [(t.price, t.qty, t.buyer_id) for t in b.trades]
    assert [(e.rate_bps, e.payments) for e in pa.history] == \
        [(e.rate_bps, e.payments) for e in pb.history]
    assert pa.positions == pb.positions
    assert np.array_equal(np.array([x.cash for x in a.ledger]),
                          np.array([y.cash for y in b.ledger]))
