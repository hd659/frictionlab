from fractions import Fraction

import numpy as np
import pytest

from frictionlab.agents import (
    AvellanedaStoikovMM, Cancel, FundamentalProcess, InformedTrader, MarketView,
    NoiseTrader, Submit, Venue, make_noise_population, order_flow,
)
from frictionlab.core import Ledger, Order, OrderBook, Side
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import FixedCost, HalfSpreadCost, MakerTakerFee


def view(bid=99, ask=101, t=0.0, position=0, depth=5, open_orders=()):
    bids = ((bid, depth),) if bid is not None else ()
    asks = ((ask, depth),) if ask is not None else ()
    return MarketView(float(t), 1, bid, ask, bids, asks, None, 10**6, position,
                      tuple(open_orders))


def stack():
    # taker at ~100: 10 bps -> 0.1, half-spread 1 -> total ceil(1.1) = 2 per lot
    return MakerTakerFee(taker_bps=10) + HalfSpreadCost(2)


def informed(v, decision_cost=0, **kw):
    return InformedTrader(1, FundamentalProcess.constant(v), frictions=stack(),
                          decision_cost=decision_cost, **kw)


# -- fundamental -------------------------------------------------------------
def test_fundamental_constant_and_seeded_random_walk():
    c = FundamentalProcess.constant(120)
    assert c.is_constant and c.path([0, 5, 1e6]).tolist() == [120, 120, 120]
    a = FundamentalProcess.random_walk(100, jump_rate=2.0, jump_size=3, seed=7)
    b = FundamentalProcess.random_walk(100, jump_rate=2.0, jump_size=3, seed=7)
    ts = np.linspace(0, 50, 201)
    assert a.value(40.0) == b.value(40.0)  # out-of-order query on `a` first
    assert a.path(ts).tolist() == b.path(ts).tolist()
    jumps = a.jumps(50.0)
    assert len(jumps) > 20 and all(t > 0 for t, _ in jumps)
    steps = np.diff([100] + [v for _, v in jumps])
    assert set(np.abs(steps).tolist()) <= {3, 0}  # 0 only when floored
    assert FundamentalProcess.random_walk(100, 2.0, 3, seed=8).path(ts).tolist() != a.path(ts).tolist()
    with pytest.raises(ValueError):
        FundamentalProcess(5, floor=10)


# -- informed ----------------------------------------------------------------
@pytest.mark.parametrize("decision_cost", [0, 1, 3])
def test_informed_never_trades_when_edge_le_friction(decision_cost):
    for v in range(90, 111):
        trader = informed(v, decision_cost)
        acts = trader.decide(view(99, 101))
        for side, px in ((Side.BUY, 101), (Side.SELL, 99)):
            q = trader.evaluate(view(99, 101), side)
            assert q.friction == trader.total_friction(side, px, 1) == 2 + decision_cost
            assert q.profitable == (side.value * (v - px) > 2 + decision_cost)
        sent = [a for a in acts if isinstance(a, Submit)]
        edge = max(v - 101, 99 - v)
        assert bool(sent) == (edge > 2 + decision_cost)


def test_informed_buys_below_and_sells_above_fundamental():
    buy = informed(110).decide(view(99, 101))
    assert buy == [Submit(Side.BUY, 1, 101)]
    sell = informed(90).decide(view(99, 101))
    assert sell == [Submit(Side.SELL, 1, 99)]
    q = informed(110).evaluate(view(99, 101), Side.BUY)
    assert q.gross_edge == 9 and q.net_edge == 7
    assert set(q.breakdown) == {"maker_taker", "half_spread", "decision_cost"}
    assert sum(q.breakdown.values()) == Fraction(1101, 1000)  # 10 bps of 101 + 1


def test_informed_respects_depth_position_cap_and_cancels_leftovers():
    t = informed(120, order_size=5, max_position=4)
    assert t.decide(view(99, 101, depth=2)) == [Submit(Side.BUY, 2, 101)]
    assert t.decide(view(99, 101, position=3)) == [Submit(Side.BUY, 1, 101)]
    assert t.decide(view(99, 101, position=4, open_orders=(7,))) == [Cancel(7)]
    assert informed(120).decide(view(None, None)) == []


def test_informed_trades_in_a_live_book_only_past_friction():
    for v, expect_fill in ((103, False), (104, True)):  # edge 2 == F, edge 3 > F
        lg = Ledger(stack(), check_every=True)
        venue = Venue(OrderBook(), lg)
        lg.open_account(1, cash=10**6)
        lg.open_account(2, cash=10**6, position=10)
        venue.book.submit(Order(900, 2, Side.SELL, 3, 101, 0.0))
        trades = informed(v).step(venue, 1.0)
        assert bool(trades) == expect_fill
        assert lg[1].position == (1 if expect_fill else 0)
        lg.check()


# -- noise -------------------------------------------------------------------
def test_noise_flow_is_side_balanced_over_many_seeded_draws():
    pop = make_noise_population(20, seed=11, max_size=3)
    for t in pop:
        for _ in range(500):
            t.decide(view())
    flow = order_flow(pop)
    assert len(flow) == 10_000 and set(np.abs(flow).tolist()) == {1, 2, 3}
    buys = int((flow > 0).sum())
    assert abs(buys - 5_000) < 4 * np.sqrt(10_000 * 0.25)  # 4-sigma binomial band
    assert abs(flow.sum() / np.abs(flow).sum()) < 0.03
    assert all(isinstance(o, Submit) and o.price is None for t in pop for o in t.orders)


def test_noise_is_value_blind_and_rate_configurable():
    a, b = NoiseTrader(1, 5), NoiseTrader(1, 5)
    sa = [a.decide(view(99, 101)) for _ in range(200)]
    sb = [b.decide(view(10, 300, t=9.0, position=50)) for _ in range(200)]
    assert sa == sb  # book state and time never enter the decision
    half = NoiseTrader(2, 3, arrival_rate=4.0, participation=0.5)
    assert half.wake_rate == 4.0 and half.effective_rate == 2.0
    n = sum(bool(half.decide(view())) for _ in range(4000))
    assert 1850 < n < 2150
    skew = NoiseTrader(3, 3, buy_prob=0.8)
    for _ in range(2000):
        skew.decide(view())
    assert 0.5 < skew.imbalance() < 0.7
    with pytest.raises(ValueError):
        NoiseTrader(4, arrival_rate=0.0)


# -- integration ---------------------------------------------------------------
def run_market(seed, decision_cost=0):
    fund = FundamentalProcess.random_walk(100, jump_rate=0.2, jump_size=4, seed=seed)
    fr = stack() + FixedCost(0)
    agents = [AvellanedaStoikovMM(1, gamma=0.01, sigma=1.0, horizon=300.0,
                                         inventory_limit=50, rng=seed),
              InformedTrader(2, fund, frictions=fr, decision_cost=decision_cost,
                             max_position=20),
              *make_noise_population(4, seed=seed, first_id=3, arrival_rate=0.5)]
    runner = MarketRunner(agents, fr, seed=seed, cash=10**7)
    return runner.run(until=300.0)


def test_market_runs_are_deterministic_and_informed_profits():
    r1, r2 = run_market(42), run_market(42)
    key = lambda r: [(t.price, t.qty, t.buyer_id, t.seller_id, t.time) for t in r.trades]
    assert key(r1) == key(r2) and len(r1.trades) > 50
    assert key(run_market(43)) != key(r1)
    inf = r1.agents[2]
    assert inf.n_trades_sent > 0
    assert all(q is None or q.profitable for q in inf.history)
    v_end = inf.fundamental.value(r1.time)
    assert r1.ledger.equity(v_end)[2] > 10**7  # marked at V, information pays
    # pricier thinking can only shrink the informed trader's opportunity set
    assert run_market(42, decision_cost=50).agents[2].n_trades_sent <= inf.n_trades_sent
