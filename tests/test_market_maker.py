import math

import pytest

from frictionlab.agents import (
    AvellanedaStoikovMM, Cancel, MarketView, Submit, Venue, ZICTrader,
)
from frictionlab.core import Ledger, Order, OrderBook, Side
from frictionlab.core.types import IdGen


def view(position=0, t=0.0, bid=99, ask=101, open_orders=(), agent_id=1):
    return MarketView(float(t), agent_id, bid, ask, (), (), None, 100_000,
                      position, tuple(open_orders))


def mm(**kw):
    base = dict(gamma=0.1, sigma=2.0, k=1.5, horizon=100.0, inventory_limit=5)
    base.update(kw)
    return AvellanedaStoikovMM(1, **base)


def submits(actions):
    return {a.side: a for a in actions if isinstance(a, Submit)}


def test_reservation_price_matches_closed_form():
    m = mm()
    q = m.quote(view(position=3, t=40.0))
    risk = 0.1 * 4.0 * 60.0
    assert q.reservation == pytest.approx(100 - 3 * risk)
    assert q.spread == pytest.approx(risk + 20.0 * math.log1p(0.1 / 1.5))
    assert q.bid == math.floor(q.reservation - q.spread / 2)
    assert q.ask == math.ceil(q.reservation + q.spread / 2)


def test_quotes_skew_down_when_long_and_up_when_short():
    m = mm()
    flat, long_, short = (m.quote(view(position=p)) for p in (0, 3, -3))
    assert long_.reservation < flat.reservation < short.reservation
    assert long_.bid < flat.bid < short.bid
    assert long_.ask < flat.ask < short.ask


def test_spread_widens_with_gamma_and_remaining_horizon():
    spreads = [mm(gamma=g).quote(view()).spread for g in (0.05, 0.1, 0.5)]
    assert spreads == sorted(spreads) and spreads[0] < spreads[-1]
    m = mm()
    by_time = [m.quote(view(t=t)).spread for t in (0.0, 50.0, 99.0, 150.0)]
    assert by_time[0] > by_time[1] > by_time[2] > by_time[3]
    # past the horizon only the arrival-intensity term remains
    assert by_time[3] == pytest.approx(20.0 * math.log1p(0.1 / 1.5))


def test_quotes_are_integer_ticks_and_never_cross():
    m = mm(gamma=0.01, sigma=0.1, min_spread=2)
    for p in range(-5, 6):
        q = m.quote(view(position=p, t=99.9))
        for px in (q.bid, q.ask):
            assert px is None or (isinstance(px, int) and px >= 1)
        if q.bid is not None and q.ask is not None:
            assert q.ask - q.bid >= 2


def test_inventory_cap_withdraws_the_accumulating_side():
    m = mm(inventory_limit=5, order_size=2)
    at_cap = m.quote(view(position=5))
    assert at_cap.bid is None and at_cap.bid_qty == 0 and at_cap.ask is not None
    assert m.quote(view(position=4)).bid_qty == 1  # partial room left
    floor = m.quote(view(position=-5))
    assert floor.ask is None and floor.bid is not None
    assert set(submits(m.decide(view(position=5)))) == {Side.SELL}


def test_inventory_cap_blocks_accumulation_in_a_live_market():
    lg = Ledger(check_every=True)
    venue = Venue(OrderBook(), lg)
    maker = mm(inventory_limit=3, sigma=0.5, gamma=0.05)
    lg.open_account(1, cash=1_000_000)
    lg.open_account(2, cash=1_000_000, position=100)
    seller_ids = IdGen(10_000)
    for t in range(20):
        maker.step(venue, float(t))
        bid = venue.book.best_bid
        if bid is not None:  # a persistent seller hits the maker's bid
            trades = venue.book.submit(Order(seller_ids(), 2, Side.SELL, 1, bid, float(t)))
            lg.settle_all(trades)
        assert lg[1].position <= 3
    assert lg[1].position == 3
    lg.check()


def test_cancel_and_replace_only_stale_quotes():
    venue = Venue(OrderBook(), Ledger())
    venue.ledger.open_account(1, cash=100_000)
    venue.book.submit(Order(900, 99, Side.BUY, 1, 99))
    venue.book.submit(Order(901, 99, Side.SELL, 1, 101))
    venue.ledger.open_account(99)
    m = mm()
    m.step(venue, 0.0)
    first = m.quotes
    assert sorted(s for s, _ in first.values()) == [Side.SELL, Side.BUY]
    m.step(venue, 0.0)  # same state -> keep both quotes, no churn
    assert m.quotes == first
    acts = m.decide(view(position=4, t=0.0, open_orders=tuple(first)))
    cancels = {a.order_id for a in acts if isinstance(a, Cancel)}
    assert cancels == set(first)  # skew moved both prices
    assert set(submits(acts)) == {Side.BUY, Side.SELL}
    unknown = m.decide(view(open_orders=(12345,)))
    assert Cancel(12345) in unknown


def test_quotes_are_deterministic_under_a_fixed_seed():
    def run(seed):
        lg = Ledger(check_every=True)
        venue = Venue(OrderBook(), lg)
        maker = AvellanedaStoikovMM(1, jitter=2, rng=seed, inventory_limit=4)
        lg.open_account(1, cash=1_000_000, position=10)
        agents = [maker]
        for i, side in enumerate([Side.BUY, Side.SELL] * 3):
            z = ZICTrader(2 + i, side, seed + i, n_units=5, value_range=(85, 115),
                          price_range=(1, 200))
            lg.open_account(z.agent_id, cash=1_000_000, position=10)
            agents.append(z)
        for t in range(300):
            agents[t % len(agents)].step(venue, float(t) * 0.3)
        lg.check()
        return [(q.bid, q.ask) for q in maker.history], len(venue.book.trades)

    a, b, c = run(7), run(7), run(8)
    assert a == b and a[1] > 0
    assert a[0] != c[0]


def test_rejects_bad_parameters():
    for kw in ({"gamma": 0}, {"sigma": -1}, {"k": float("nan")}, {"horizon": 0}):
        with pytest.raises(ValueError):
            mm(**kw)
    with pytest.raises(ValueError):
        mm(inventory_limit=0)
