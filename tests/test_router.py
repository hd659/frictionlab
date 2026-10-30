"""VenueRouter: min all-in cost by brute force, ties, splits, ledger, determinism."""
import numpy as np
import pytest

from frictionlab.agents import Submit
from frictionlab.core import Side
from frictionlab.core.ledger import Ledger
from frictionlab.core.order_book import OrderBook
from frictionlab.core.types import IdGen, Order
from frictionlab.costs import FixedCost, FrictionStack
from frictionlab.markets import (FeeComponent, FeeTier, FeeTierSchedule, RouteVenue,
                                 VenueRouter, quote_leg)

TAKER = 1
MAKERS = (10, 11, 12)


def _venue(vid, levels, taker_bps=10, fixed=0, tiers=None, seed_volume=0, oid0=1):
    """levels: [(side, price, qty, maker_id)] resting on a fresh book."""
    sched = FeeTierSchedule(tiers or [FeeTier(0, 0, taker_bps)], window=5)
    fc = FeeComponent(sched)
    led = Ledger(FrictionStack([fc, FixedCost(fixed)]), check_every=True)
    for aid in (TAKER,) + MAKERS:
        led.open_account(aid, 10**7, 10**4)
    if seed_volume:
        fc.record(TAKER, seed_volume, step=0)
        fc.advance(1)
    book, ids = OrderBook(trade_ids=IdGen()), IdGen(oid0)
    for side, price, qty, mid in levels:
        book.submit(Order(ids(), mid, side, qty, price))
    return RouteVenue(vid, book, led, ids)


def _random_specs(rng, n=3, side=Side.SELL):
    out = []
    for k in range(n):
        lv = [(side, int(rng.integers(95, 106)), int(rng.integers(1, 6)),
               int(rng.choice(MAKERS))) for _ in range(int(rng.integers(1, 5)))]
        out.append(dict(vid=f"v{k}", levels=lv, taker_bps=int(rng.integers(0, 60)),
                        fixed=int(rng.integers(0, 8))))
    return out


def _random_venues(rng, n=3, side=Side.SELL):
    return [_venue(**spec) for spec in _random_specs(rng, n, side)]


def _realized_cost(spec, side, qty):
    """Brute force: execute the full order on a fresh copy, return taker cash outflow."""
    v = _venue(**spec)
    cash0 = v.ledger[TAKER].cash
    plan, _ = VenueRouter([v]).route(TAKER, Submit(side, qty))
    return cash0 - v.ledger[TAKER].cash, plan.qty


def test_router_validation():
    with pytest.raises(ValueError):
        VenueRouter([])
    a = _venue("a", [])
    with pytest.raises(ValueError):
        VenueRouter([a, a])
    with pytest.raises(TypeError):
        RouteVenue(3, OrderBook(), Ledger())


def test_quote_matches_realized_cash_and_breakdown():
    v = _venue("a", [(Side.SELL, 100, 2, 10), (Side.SELL, 102, 5, 11)], taker_bps=25, fixed=3)
    leg = quote_leg(v, TAKER, Side.BUY, 4)
    assert (leg.best, leg.notional, leg.slippage, leg.n_fills) == (100, 404, 4, 2)
    assert leg.fees == (1 + 3) + (1 + 3)                # ceil(25bps * notional) + fixed
    spec = dict(vid="a", levels=[(Side.SELL, 100, 2, 10), (Side.SELL, 102, 5, 11)],
                taker_bps=25, fixed=3)
    assert _realized_cost(spec, Side.BUY, 4) == (leg.all_in, 4)


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_router_picks_min_all_in_venue_by_brute_force(side):
    rng = np.random.default_rng(2031 + side.value)
    checked = 0
    for _ in range(60):
        specs = _random_specs(rng, side=side.opposite)
        qty = int(rng.integers(1, 6))
        plan = VenueRouter([_venue(**sp) for sp in specs]).plan(TAKER, Submit(side, qty))
        full = {sp["vid"]: _realized_cost(sp, side, qty) for sp in specs}
        full = {k: c for k, (c, q) in full.items() if q == qty}
        if not full:
            assert plan.qty <= qty
            continue
        best = min(full.items(), key=lambda kv: (kv[1], kv[0]))
        assert not plan.split and plan.legs[0].venue_id == best[0]
        assert plan.all_in == best[1]
        checked += 1
    assert checked > 20


def test_ties_resolve_by_venue_id():
    lv = [(Side.SELL, 100, 5, 10)]
    venues = [_venue(x, lv, taker_bps=10, fixed=1) for x in ("zeta", "alpha", "mid")]
    for order in (venues, venues[::-1]):
        plan = VenueRouter(order).plan(TAKER, Submit(Side.BUY, 3))
        assert plan.legs[0].venue_id == "alpha"
        assert len({l.all_in for l in plan.quotes.values()}) == 1


def test_tier_discount_moves_flow_to_venue_where_account_is_vip():
    tiers = [FeeTier(0, 0, 50), FeeTier(100, -2, 5)]
    lv = [(Side.SELL, 100, 10, 10)]
    a = _venue("a", lv, tiers=tiers)
    b = _venue("b", [(Side.SELL, 100, 10, 10)], tiers=tiers, seed_volume=150)
    plan = VenueRouter([a, b]).plan(TAKER, Submit(Side.BUY, 10))
    assert plan.legs[0].venue_id == "b"
    assert plan.quotes["a"].fees == 5 and plan.quotes["b"].fees == 1  # 50bps vs ceil(5bps)


def test_split_only_when_single_venue_lacks_depth_and_never_overfills():
    a = _venue("a", [(Side.SELL, 100, 3, 10)], taker_bps=0)
    b = _venue("b", [(Side.SELL, 101, 4, 11)], taker_bps=0)
    r = VenueRouter([a, b])
    assert not r.plan(TAKER, Submit(Side.BUY, 3)).split
    plan, trades = r.route(TAKER, Submit(Side.BUY, 5))
    assert plan.split and [(l.venue_id, l.qty) for l in plan.legs] == [("a", 3), ("b", 2)]
    assert sum(t.qty for ts in trades.values() for t in ts) == 5
    over = VenueRouter([_venue("c", [(Side.SELL, 100, 2, 10)])]).plan(TAKER, Submit(Side.BUY, 9))
    assert over.qty == 2 <= over.requested
    rng = np.random.default_rng(5)
    for _ in range(40):
        venues, qty = _random_venues(rng), int(rng.integers(1, 30))
        plan, trades = VenueRouter(venues).route(TAKER, Submit(Side.BUY, qty))
        assert plan.qty <= qty
        assert sum(t.qty for ts in trades.values() for t in ts) == plan.qty
        for v in venues:
            v.ledger.check()


def test_limit_price_and_self_trade_are_respected():
    v = _venue("a", [(Side.SELL, 100, 2, TAKER), (Side.SELL, 101, 2, 10), (Side.SELL, 105, 9, 11)])
    leg = quote_leg(v, TAKER, Side.BUY, 6, limit=103)
    assert leg.qty == 2 and leg.best == 101             # own order skipped, 105 > limit
    assert quote_leg(v, TAKER, Side.BUY, 1, limit=99) is None


def test_routing_is_deterministic_under_fixed_seed():
    def run(seed):
        rng, out = np.random.default_rng(seed), []
        for _ in range(15):
            plan, _ = VenueRouter(_random_venues(rng)).route(TAKER, Submit(Side.BUY, int(rng.integers(1, 12))))
            out.append(tuple((l.venue_id, l.qty, l.all_in) for l in plan.legs))
        return out
    assert run(99) == run(99)
