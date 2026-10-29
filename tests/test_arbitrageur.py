"""CrossVenueArbitrageur: book <-> constant-product AMM, net of total friction."""
import pytest

from frictionlab.agents import (
    Agent, ArbPair, CrossVenueArbitrageur, MarketView, Submit, Swap, cp_optimal_size,
)
from frictionlab.core import Ledger, Side
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import FixedCost, FrictionStack, HalfSpreadCost, MakerTakerFee
from frictionlab.markets import ConstantProductAMM

ARB, MAKER, POOL = 1, 2, 900
X, Y, FEE = 10_000, 1_000_000, 30  # spot 100 ticks, 30 bps LP fee


def book_fees():
    return MakerTakerFee(taker_bps=10) + HalfSpreadCost(1)  # 0.1 + 0.5 per lot


class Maker(Agent):
    """Posts a fixed ladder once (deterministic quotes), then idles."""

    def __init__(self, quotes, agent_id=MAKER):
        super().__init__(agent_id)
        self.quotes, self.wake_rate = list(quotes), 50.0

    def decide(self, view):
        out = [Submit(s, q, p) for s, q, p in self.quotes]
        self.quotes = []
        return out


def pool_factory(gas=0, pid=POOL, x=X, y=Y):
    stack = FrictionStack([FixedCost(gas)]) if gas else None
    return lambda lg: ConstantProductAMM(lg, pid, x, y, fee_bps=FEE, frictions=stack)


def run(quotes, frictions=None, pools=None, seed=3, wakeups=200, **kw):
    arb = CrossVenueArbitrageur(ARB, **kw)
    r = MarketRunner([Maker(quotes), arb], frictions, seed=seed, cash=10**8, position=10**5,
                     pools=pools or [pool_factory()], check_every=True)
    return r.run(max_wakeups=wakeups), arb


def view(bid=None, ask=None, depth=1000):
    bids = ((bid, depth),) if bid else ()
    asks = ((ask, depth),) if ask else ()
    return MarketView(0.0, ARB, bid, ask, bids, asks, None, 10**8, 0)


def pool(gas=0):
    lg = Ledger()
    return pool_factory(gas)(lg)


def first_trading_gap(gas, side=Side.BUY, decision_cost=0):
    """Smallest venue price gap (ticks) at which the agent trades."""
    arb = CrossVenueArbitrageur(ARB, frictions=book_fees(), max_size=400,
                                decision_cost=decision_cost)
    p = pool(gas)
    for gap in range(0, 30):
        v = view(ask=100 - gap) if side is Side.BUY else view(bid=100 + gap)
        if arb.best_opportunity(v, {POOL: p}) is not None:
            return gap
    return 30


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_no_trade_when_gap_within_friction(side):
    # gap of 0 / 1 tick cannot cover taker fee + half-spread + 30 bps LP fee
    for gap in (0, 1):
        v = view(ask=100 - gap) if side is Side.BUY else view(bid=100 + gap)
        quotes = [(Side.SELL, 1000, 100 - gap)] if side is Side.BUY else [(Side.BUY, 1000, 100 + gap)]
        arb = CrossVenueArbitrageur(ARB, frictions=book_fees())
        assert arb.best_opportunity(v, {POOL: pool()}) is None
        res, arb = run(quotes, book_fees())
        assert res.arbs == [] and res.pools[POOL].reserves == (X, Y)
    # a huge decision cost alone also blocks an otherwise profitable gap
    assert first_trading_gap(0, side, decision_cost=10**6) == 30


def test_trade_narrows_venue_gap_both_directions():
    for side, quotes in ((Side.BUY, [(Side.SELL, 1000, 95)]), (Side.SELL, [(Side.BUY, 1000, 105)])):
        res, arb = run(quotes, book_fees())
        assert len(res.arbs) >= 1 and all(a.hedged for a in res.arbs)
        assert res.arbs[0].pair.book.side is side
        spot = res.pools[POOL].spot_fraction
        touch = res.book.best_ask if side is Side.BUY else res.book.best_bid
        assert abs(spot - touch) < 5
        assert side.value * (spot - touch) > 0  # narrowed, not overshot past the touch


@pytest.mark.parametrize("ask", [90, 95, 98])
def test_size_never_exceeds_cp_analytic_optimum(ask):
    arb = CrossVenueArbitrageur(ARB, max_size=5000)  # no book fees -> pure CP optimum
    opp = arb.best_opportunity(view(ask=ask, depth=5000), {POOL: pool()})
    q_star = cp_optimal_size(X, Y, FEE, ask, Side.BUY)
    assert opp is not None and opp.qty <= q_star
    # the integer search lands near q*; any shortfall is only output-flooring dust
    assert opp.qty >= 0.85 * q_star
    assert opp.net_edge >= arb.quote_size(view(ask=ask, depth=5000), pool(), Side.BUY,
                                          int(q_star)).net_edge
    bid = 200 - ask
    opp = arb.best_opportunity(view(bid=bid, depth=5000), {POOL: pool()})
    assert opp is not None and opp.qty <= cp_optimal_size(X, Y, FEE, bid, Side.SELL)


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_fixed_cost_widens_no_arbitrage_band_monotonically(side):
    bands = [first_trading_gap(g, side) for g in (0, 10, 100, 400, 1500, 4000)]
    assert all(a <= b for a, b in zip(bands, bands[1:]))
    assert bands[-1] > bands[0]


def test_friction_breakdown_and_planned_pnl_are_exact():
    arb = CrossVenueArbitrageur(ARB, frictions=book_fees(), decision_cost=7)
    opp = arb.best_opportunity(view(ask=95), {POOL: pool(gas=5)})
    assert opp.gross_edge > opp.friction and opp.price_impact >= 0 and opp.lp_fee > 0
    assert opp.friction == opp.book_fees + opp.gas + opp.lp_fee + opp.price_impact + 7
    assert opp.pnl.net == opp.net_edge and opp.gas == 5
    assert {"book_maker_taker", "book_half_spread", "lp_fee", "price_impact",
            "amm_gas", "decision_cost"} <= set(opp.breakdown)


def test_net_pnl_per_trade_nonnegative_and_matches_plan():
    quotes = [(Side.SELL, 20, 94), (Side.SELL, 30, 96), (Side.SELL, 40, 97)]
    res, arb = run(quotes, book_fees(), pools=[pool_factory(gas=3)], decision_cost=2,
                   max_size=15)  # small clip -> several pairs walk the ladder
    assert len(res.arbs) >= 3 and len(res.arbs) == len(arb.sent)
    for a, q in zip(res.arbs, arb.sent):
        assert a.hedged and a.qty == q.qty
        assert a.pnl.net >= 0 and a.pnl.net == q.net_edge
        assert a.pnl.decision_costs == 2
    assert res.arb_pnl(ARB).net == sum(a.pnl.net for a in res.arbs) > 0
    assert not arb.open_orders  # IOC: hedge legs never rest on the book


def test_ledger_invariants_and_pool_k_with_two_pools():
    quotes = [(Side.SELL, 500, 95)]  # pool 901 is richer (spot 105) and is hit first
    pools = [pool_factory(gas=2), pool_factory(pid=901, x=2_000, y=210_000)]
    res, arb = run(quotes, book_fees(), pools=pools)
    lg = res.ledger
    lg.check()
    assert res.arbs[0].pair.amm.pool_id == 901
    assert {a.pair.amm.pool_id for a in res.arbs} == {POOL, 901}
    assert res.pools[POOL].k >= X * Y and res.pools[901].k >= 2_000 * 210_000
    assert lg.cash_in == sum(a.cash for a in lg) + lg.venue.cash
    assert sum(a.position for a in lg) == lg.inventory_in


def test_deterministic_under_fixed_seed():
    quotes = [(Side.SELL, 50, 95), (Side.SELL, 50, 97), (Side.BUY, 60, 104)]
    sig = lambda r: ([(t.price, t.qty, t.buyer_id) for t in r.trades],
                     [a.pnl for a in r.arbs], r.pools[POOL].reserves)
    a, _ = run(quotes, book_fees(), seed=11)
    b, _ = run(quotes, book_fees(), seed=11)
    assert sig(a) == sig(b)


def test_intent_validation():
    with pytest.raises(ValueError):
        ArbPair(Submit(Side.BUY, 1), Swap(POOL, Side.SELL, 1))  # book leg must be priced
    with pytest.raises(ValueError):
        ArbPair(Submit(Side.BUY, 1, 95), Swap(POOL, Side.BUY, 100))
    with pytest.raises(ValueError):
        Swap(POOL, Side.SELL, 0)
