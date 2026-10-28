import importlib
import sys
from fractions import Fraction

import numpy as np
import pytest

from frictionlab.core import IdGen, Ledger, LedgerError, Side
from frictionlab.costs import FixedCost, FrictionStack
from frictionlab.markets import ConcentratedLiquidityAMM as CL
from frictionlab.markets import ConstantProductAMM, SwapRejected

TRADER = 1


def ledger():
    lg = Ledger(check_every=True)
    lg.open_account(TRADER, cash=10**12, position=10**9)
    return lg


def active_L(pool):
    return sum(p.liquidity for p in pool.positions if p.tick_lower <= pool.tick < p.tick_upper)


def _cp_cl(fee):
    lg, ids = ledger(), IdGen(start=10**9)  # one id source per ledger
    cp = ConstantProductAMM(lg, 900, 10**6, 10**8 + 12_345, fee_bps=fee, ids=ids)
    cl = CL.full_range(lg, 901, 10**6, 10**8 + 12_345, fee_bps=fee, ids=ids)
    return cp, cl


def close(a, b, rel=10**5):
    return abs(a - b) <= max(2, a // rel)


@pytest.mark.parametrize("fee", [0, 5, 30, 100])
def test_full_range_quotes_match_constant_product(fee):
    cp, cl = _cp_cl(fee)
    assert abs(cl.spot_fraction - cp.spot_fraction) < Fraction(1, 10**9)
    for side, amt in ((Side.BUY, 10**4), (Side.BUY, 10**7), (Side.SELL, 50), (Side.SELL, 2 * 10**5)):
        assert close(cp.quote_exact_in(side, amt), cl.quote_exact_in(side, amt))


def test_full_range_zero_fee_path_matches_constant_product():
    # with fee > 0 paths diverge by design: CP compounds fees into k, CL books them apart
    cp, cl = _cp_cl(0)
    rng = np.random.default_rng(3)
    for _ in range(25):
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        amt = int(rng.integers(10**4, 10**7)) if side is Side.BUY else int(rng.integers(100, 10**5))
        assert close(cp.swap_exact_in(TRADER, side, amt).amount_out,
                     cl.swap_exact_in(TRADER, side, amt).amount_out)
    assert all(close(a, b) for a, b in zip(cp.reserves, cl.reserves))
    assert cl.violations() == [] and cl.liquidity == active_L(cl)


def test_concentrated_liquidity_has_lower_price_impact_for_same_capital():
    lg = ledger()
    conc = CL(lg, 900, 100, fee_bps=30)
    b, q = conc.add_position(90, 110, 10**7)
    capital = b * 100 + q
    full = CL.full_range(lg, 901, capital // 200, capital // 2, fee_bps=30)
    fb, fq = full.reserves
    assert fb * 100 + fq <= capital  # full range gets no more capital
    for side, amt in ((Side.BUY, 10**6), (Side.SELL, 10**4)):
        ic, if_ = conc.price_impact(side, amt), full.price_impact(side, amt)
        assert 0 < ic < if_
        assert ic * 2 < if_  # range width ~ +/-10% => several-x capital efficiency


def make_two_ranges(backstop=0, **kw):
    lg = ledger()
    pool = CL(lg, 900, 100, fee_bps=30, **kw)
    pool.add_position(90, 110, 10**7)    # A: active at start
    pool.add_position(105, 130, 3 * 10**7)  # B: above the price, base only
    if backstop:
        pool.add_position(1, 10**6, backstop)  # wide range keeps random walks fillable
    return lg, pool


def test_crossing_range_boundaries_switches_active_liquidity():
    lg, pool = make_two_ranges()
    assert pool.amounts_for(105, 130, 1)[1] == 0 and pool.liquidity == 10**7
    pool.swap_exact_in(TRADER, Side.BUY, 10**5)  # small: stays inside A
    assert pool.liquidity == 10**7 and pool.tick < 105
    plan = pool.simulate(Side.BUY, 5 * 10**6)
    assert plan.crossed == (105,) and plan.liquidity == 4 * 10**7
    pool.swap_exact_in(TRADER, Side.BUY, 5 * 10**6)
    assert 105 <= pool.tick < 110 and pool.liquidity == active_L(pool) == 4 * 10**7
    pool.swap_exact_in(TRADER, Side.BUY, 2 * 10**7)  # through A's upper bound
    assert 110 <= pool.tick < 130 and pool.liquidity == active_L(pool) == 3 * 10**7
    assert pool.simulate(Side.SELL, 25 * 10**4).crossed == (110, 105)
    pool.swap_exact_in(TRADER, Side.SELL, 25 * 10**4)  # back down past 110 and 105
    assert 90 <= pool.tick < 105 and pool.liquidity == active_L(pool) == 10**7
    assert pool.violations() == []
    lg.check()


def test_swap_beyond_all_liquidity_partial_or_rejected():
    lg, pool = make_two_ranges()
    cash0, base0 = lg[TRADER].cash, pool.reserves[0]
    res = pool.swap_exact_in(TRADER, Side.BUY, 10**11)
    assert res.amount_in < 10**11 and lg[TRADER].cash == cash0 - res.amount_in
    assert pool.liquidity == 0 and pool.tick == 130 and pool.reserves[0] <= base0 - res.amount_out
    with pytest.raises(SwapRejected):
        pool.swap_exact_in(TRADER, Side.BUY, 10**6)  # nothing left above
    lg2, strict = make_two_ranges(partial_fill=False)
    before = (strict.reserves, strict.sqrt_price, lg2[TRADER].cash, len(lg2.journal))
    with pytest.raises(SwapRejected):
        strict.swap_exact_in(TRADER, Side.BUY, 10**11)
    with pytest.raises(SwapRejected):
        strict.swap_exact_in(TRADER, Side.SELL, 10**9)
    assert (strict.reserves, strict.sqrt_price, lg2[TRADER].cash, len(lg2.journal)) == before
    assert strict.swap_exact_in(TRADER, Side.BUY, 10**6).amount_in == 10**6


def test_reserves_plus_fees_conservation_through_ledger():
    lg, pool = make_two_ranges(backstop=10**6, frictions=FrictionStack([FixedCost(3)]))
    rng = np.random.default_rng(5)
    fees = {Side.BUY: 0, Side.SELL: 0}
    for i in range(40):
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        amt = int(rng.integers(10**5, 2 * 10**7)) if side is Side.BUY else int(rng.integers(10**3, 2 * 10**5))
        r = pool.swap_exact_in(TRADER, side, amt, time=float(i))
        fees[side] += r.lp_fee
        assert pool.violations() == [] and pool.liquidity == active_L(pool)
    assert (pool.fees_quote, pool.fees_base) == (fees[Side.BUY], fees[Side.SELL])
    assert pool.fees_quote > 0 and pool.fees_base > 0
    assert pool.reserves == (pool.reserve_base + pool.fees_base, pool.reserve_quote + pool.fees_quote)
    assert lg.venue.cash == 3 * len(lg.journal) and lg[900].fees_paid == 0
    lg.check()


def test_quote_equals_swap_and_slippage_guard():
    lg, pool = make_two_ranges()
    q = pool.quote_exact_in(Side.BUY, 5 * 10**7)
    with pytest.raises(SwapRejected):
        pool.swap_exact_in(TRADER, Side.BUY, 5 * 10**7, min_out=q + 1)
    assert pool.swap_exact_in(TRADER, Side.BUY, 5 * 10**7, min_out=q).amount_out == q
    assert pool.marginal_price(Side.BUY) >= pool.spot_price >= pool.marginal_price(Side.SELL)


def test_no_overdraft_rejects_without_moving_pool():
    lg = Ledger(allow_overdraft=False, allow_short=False)
    lg.open_account(TRADER, cash=1_000, position=0)
    pool = CL.full_range(lg, 900, 10**4, 10**6)
    s0 = pool.sqrt_price
    with pytest.raises(LedgerError):
        pool.swap_exact_in(TRADER, Side.BUY, 1_001)
    with pytest.raises(LedgerError):
        pool.swap_exact_in(TRADER, Side.SELL, 5)
    assert pool.sqrt_price == s0 and pool.violations() == []


def test_invalid_inputs():
    lg = ledger()
    with pytest.raises(ValueError):
        CL(lg, 900, 0)
    with pytest.raises(ValueError):
        CL(lg, 901, 100, fee_bps=10_000)
    pool = CL(lg, 902, 100)
    with pytest.raises(ValueError):
        pool.add_position(110, 90, 10)
    with pytest.raises(ValueError):
        pool.add_position(90, 110, 0)
    with pytest.raises(SwapRejected):
        pool.quote_exact_in(Side.BUY, 100)  # no liquidity at all


def test_legacy_module_warns_but_works():
    sys.modules.pop("frictionlab.markets.amm_v3_legacy", None)
    with pytest.warns(DeprecationWarning, match="amm_v3_legacy is deprecated"):
        legacy = importlib.import_module("frictionlab.markets.amm_v3_legacy")
    assert legacy.ConcentratedLiquidityAMM is CL
    pool = legacy.ConcentratedLiquidityAMM.full_range(ledger(), 900, 10**4, 10**6)
    assert pool.quote_exact_in(Side.SELL, 10) > 0
    with pytest.raises(AttributeError, match="removed"):
        legacy.V3Pool


def _seeded_run(seed):
    lg, pool = make_two_ranges(backstop=10**6, frictions=FrictionStack([FixedCost(1)]))
    rng = np.random.default_rng(seed)
    for _ in range(30):
        if rng.random() < 0.5:
            pool.swap_exact_in(TRADER, Side.BUY, int(rng.integers(10**5, 2 * 10**7)))
        else:
            pool.swap_exact_in(TRADER, Side.SELL, int(rng.integers(10**3, 2 * 10**5)))
    return pool.reserves, pool.sqrt_price, pool.tick, [(s.price, s.qty) for s in lg.journal]


def test_deterministic_under_seed():
    assert _seeded_run(17) == _seeded_run(17)
    assert _seeded_run(17) != _seeded_run(18)
