"""BatchAuction: uniform price, volume maximality, pro-rata, carry, ledger, runner."""
import numpy as np
import pytest

from frictionlab.agents import Cancel, Submit, make_zi_population
from frictionlab.core import IdGen, Ledger, Order, OrderType, Side
from frictionlab.costs import MakerTakerFee
from frictionlab.markets import BatchAuction, BatchVenue, batch_runner, clearing_price, pro_rata

_oid = IdGen()


def order(aid, side, qty, price=None, t=0.0):
    ot = OrderType.LIMIT if price is not None else OrderType.MARKET
    return Order(_oid(), aid, side, qty, price, t, ot)


def brute(bids, asks):
    prices = [p for p, _ in bids + asks if p is not None]
    best = 0
    for p in range(min(prices), max(prices) + 1):
        d = sum(q for b, q in bids if b is None or b >= p)
        s = sum(q for a, q in asks if a is None or a <= p)
        best = max(best, min(d, s))
    return best


def random_batch(rng, n=6):
    bids = [(int(rng.integers(95, 106)), int(rng.integers(1, 6))) for _ in range(n)]
    asks = [(int(rng.integers(95, 106)), int(rng.integers(1, 6))) for _ in range(n)]
    return bids, asks


def fill_batch(auc, bids, asks):
    for i, (p, q) in enumerate(bids):
        auc.submit(order(1 + i, Side.BUY, q, p, t=i))
    for i, (p, q) in enumerate(asks):
        auc.submit(order(100 + i, Side.SELL, q, p, t=i + 0.5))


@pytest.mark.parametrize("seed", range(25))
def test_uniform_price_limits_respected_and_volume_is_maximal(seed):
    rng = np.random.default_rng(seed)
    bids, asks = random_batch(rng)
    auc = BatchAuction(IdGen())
    fill_batch(auc, bids, asks)
    snap = {o.order_id: o for o in auc._live.values()}
    c = auc.clear(1.0)
    assert c.volume == brute(bids, asks)
    assert {t.price for t in c.trades} <= {c.price}
    for t in c.trades:
        assert snap[t.buy_order_id].price >= c.price   # no buy executed below its limit
        assert snap[t.sell_order_id].price <= c.price  # no sell executed above its limit


def test_tie_breaks_imbalance_then_reference_price():
    # V = 4 on [98, 100]; imbalance is minimal (1) only at 100
    assert clearing_price([(100, 5), (99, 3)], [(98, 4), (101, 2)]) == (100, 4)
    # V = 2 and imbalance 0 on [97, 103]: closest to ref, else the midpoint
    flat = ([(103, 2)], [(97, 2)])
    assert clearing_price(*flat, ref=101) == (101, 2)
    assert clearing_price(*flat, ref=200) == (103, 2)
    assert clearing_price(*flat) == (100, 2)


def test_pro_rata_is_exact_and_bounded():
    rng = np.random.default_rng(7)
    for _ in range(200):
        qs = rng.integers(1, 20, size=int(rng.integers(1, 6))).tolist()
        total = int(rng.integers(0, sum(qs) + 3))
        out = pro_rata(total, qs)
        assert sum(out) == min(total, sum(qs))
        assert all(0 <= a <= q for a, q in zip(out, qs))
    assert pro_rata(5, [3, 3, 3, 3]) == [2, 1, 1, 1]  # remainder to earliest on ties


def test_marginal_level_shares_pro_rata_and_quantity_is_conserved():
    auc = BatchAuction(IdGen(), last_price=100)  # 99 and 100 tie; ref picks 100
    auc.submit(order(1, Side.BUY, 6, 101))   # better level: filled in full
    auc.submit(order(2, Side.BUY, 6, 100))   # marginal level: 6 and 3 share 6 lots
    auc.submit(order(3, Side.BUY, 3, 100))
    auc.submit(order(4, Side.SELL, 12, 99))
    c = auc.clear(1.0)
    got = {a: sum(t.qty for t in c.trades if t.buyer_id == a) for a in (1, 2, 3)}
    assert c.price == 100 and c.volume == 12 and got == {1: 6, 2: 4, 3: 2}
    assert sum(t.qty for t in c.trades if t.seller_id == 4) == 12
    rest = {o.agent_id: o.remaining for o in auc._live.values()}
    assert rest == {2: 2, 3: 1}  # 15 bid - 12 executed = 3 carried, none lost


@pytest.mark.parametrize("bids,asks", [([], []), ([(100, 3)], []), ([], [(100, 3)]),
                                       ([(99, 2)], [(101, 2)])])
def test_empty_one_sided_or_uncrossed_batch_has_no_trades(bids, asks):
    auc = BatchAuction(IdGen())
    fill_batch(auc, bids, asks)
    c = auc.clear(1.0)
    assert c.trades == () and c.price is None and c.volume == 0
    assert len(auc) == len(bids) + len(asks)  # limits carry over


def test_carry_vs_cancel_and_market_remainders():
    for carry in (True, False):
        auc = BatchAuction(IdGen(), carry=carry)
        auc.submit(order(1, Side.BUY, 5, 100))
        auc.submit(order(2, Side.BUY, 4))         # market: fills first, never rests
        auc.submit(order(3, Side.SELL, 6, 100))
        c = auc.clear(1.0)
        assert c.volume == 6 and len(auc) == (1 if carry else 0)
    assert clearing_price([(None, 3)], [(None, 3)], ref=50) == (50, 3)


def test_self_cross_is_netted_out():
    auc = BatchAuction(IdGen())
    auc.submit(order(1, Side.SELL, 2, 100, t=0))  # own ask has time priority
    auc.submit(order(2, Side.SELL, 2, 100, t=1))
    auc.submit(order(1, Side.BUY, 4, 100, t=2))
    c = auc.clear(1.0)
    assert all(t.buyer_id != t.seller_id for t in c.trades)
    assert sum(t.qty for t in c.trades) == 2


def test_batch_venue_settles_through_friction_stack_and_ledger():
    lg = Ledger(MakerTakerFee(maker_bps=10, taker_bps=30), check_every=True)
    for aid in (1, 2):
        lg.open_account(aid, cash=10**6, position=10)
    venue = BatchVenue(BatchAuction(IdGen()), lg)

    class A:  # minimal agent stub
        def __init__(self, aid):
            self.agent_id, self.open_orders = aid, set()

    a, b = A(1), A(2)
    assert venue.execute(a, Submit(Side.SELL, 3, 1000), 0.1) == []
    venue.execute(b, Submit(Side.BUY, 3, 1002), 0.2)
    c = venue.clear(1.0)
    lg.check()
    assert c.volume == 3 and lg[2].position == 13 and lg[1].position == 7
    assert lg.fees_collected == sum(lg.fees_for(t)[0] + lg.fees_for(t)[1] for t in c.trades)
    assert lg[2].fees_paid > lg[1].fees_paid  # later arrival pays the taker rate


def _run(seed):
    agents = make_zi_population("zi-c", 5, 5, seed=seed, n_units=3)
    r = batch_runner(agents, MakerTakerFee(0, 10), interval=0.5, seed=seed,
                     cash=10**6, position=10, check_every=True)
    return r, r.run(until=30.0)


def test_runner_invariants_uniform_prints_and_determinism():
    r, res = _run(3)
    assert res.trades and r.book.history
    for c in r.book.history:
        assert all(t.price == c.price and t.time == c.time for t in c.trades)
    _, again = _run(3)
    key = lambda rr: [(t.time, t.price, t.qty, t.buyer_id, t.seller_id) for t in rr.trades]
    assert key(res) == key(again)
    assert key(res) != key(_run(4)[1])
    with pytest.raises(ValueError):
        batch_runner(make_zi_population("zi-c", 1, 1), interval=0)
