from fractions import Fraction

import numpy as np
import pytest

from frictionlab.core import Ledger, LedgerError, Side
from frictionlab.costs import FixedCost, FrictionStack
from frictionlab.markets import ConstantProductAMM, SwapRejected

POOL, TRADER = 900, 1


def make(fee_bps=30, x=10_000, y=1_000_000, frictions=None, min_reserve=1):
    lg = Ledger(check_every=True)
    lg.open_account(TRADER, cash=10_000_000, position=50_000)
    amm = ConstantProductAMM(lg, POOL, x, y, fee_bps=fee_bps, frictions=frictions,
                             min_reserve=min_reserve)
    return lg, amm


def test_pool_is_ledger_account_and_spot_price():
    lg, amm = make()
    assert amm.reserves == (10_000, 1_000_000) and amm.k == 10**10
    assert amm.spot_price == 100
    assert amm.marginal_price(Side.BUY) > 100 > amm.marginal_price(Side.SELL)
    assert lg[POOL] is amm.account
    lg.check()


@pytest.mark.parametrize("fee", [0, 5, 30, 100])
def test_k_never_decreases_and_strictly_rises_with_fee(fee):
    lg, amm = make(fee)
    rng = np.random.default_rng(7)
    for _ in range(40):
        k0 = amm.k
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        amt = int(rng.integers(1_000, 20_000)) if side is Side.BUY else int(rng.integers(10, 200))
        amm.swap_exact_in(TRADER, side, amt)
        assert amm.k >= k0
        if fee > 0:
            assert amm.k > k0
    lg.check()


@pytest.mark.parametrize("side,amt", [(Side.BUY, 50_000), (Side.SELL, 500)])
def test_output_decreases_as_fee_rises(side, amt):
    outs = [make(f)[1].quote_exact_in(side, amt) for f in (0, 5, 30, 100, 300)]
    assert all(a > b for a, b in zip(outs, outs[1:]))


@pytest.mark.parametrize("side,amt", [(Side.BUY, 12_345), (Side.SELL, 321)])
def test_quote_equals_executed_swap(side, amt):
    lg, amm = make()
    q = amm.quote_exact_in(side, amt)
    x0, y0 = amm.reserves
    cash0, pos0 = lg[TRADER].cash, lg[TRADER].position
    res = amm.swap_exact_in(TRADER, side, amt)
    assert res.amount_out == q
    if side is Side.BUY:
        assert amm.reserves == (x0 - q, y0 + amt)
        assert (lg[TRADER].cash, lg[TRADER].position) == (cash0 - amt, pos0 + q)
    else:
        assert amm.reserves == (x0 + amt, y0 - q)
        assert (lg[TRADER].cash, lg[TRADER].position) == (cash0 + q, pos0 - amt)


def test_quote_exact_out_is_tight_inverse():
    _, amm = make()
    need = amm.quote_exact_out(Side.BUY, 100)
    assert amm.quote_exact_in(Side.BUY, need) >= 100
    assert amm.quote_exact_in(Side.BUY, need - 1) < 100


def test_convex_price_impact():
    _, amm = make(x=10**8, y=10**10)  # outputs >> 1 lot: floor dust negligible
    sizes = [10**6, 10**7, 10**8, 10**9]
    buy_px = [Fraction(s, amm.quote_exact_in(Side.BUY, s)) for s in sizes]
    assert all(a < b for a, b in zip(buy_px, buy_px[1:]))
    sell_px = [Fraction(amm.quote_exact_in(Side.SELL, s), s) for s in (10**4, 10**5, 10**6, 10**7)]
    assert all(a > b for a, b in zip(sell_px, sell_px[1:]))
    impacts = [amm.price_impact(Side.BUY, s) for s in sizes]
    assert impacts[0] > 0 and all(a < b for a, b in zip(impacts, impacts[1:]))


def test_drain_and_dust_are_rejected_atomically():
    lg, amm = make(min_reserve=100)
    before = (amm.reserves, lg[TRADER].cash, len(lg.journal))
    with pytest.raises(SwapRejected):
        amm.quote_exact_out(Side.BUY, 10_000)  # entire base reserve
    with pytest.raises(SwapRejected):
        amm.swap_exact_in(TRADER, Side.BUY, 10**12)  # would leave < min_reserve base
    with pytest.raises(SwapRejected):
        amm.swap_exact_in(TRADER, Side.SELL, 10**9)  # would drain quote reserve
    with pytest.raises(SwapRejected):
        amm.swap_exact_in(TRADER, Side.BUY, 50)  # output rounds to zero
    with pytest.raises(SwapRejected):
        amm.swap_exact_in(TRADER, Side.BUY, 10_000, min_out=1_000)  # slippage
    assert (amm.reserves, lg[TRADER].cash, len(lg.journal)) == before


def test_round_trip_with_fee_loses_value():
    for fee, strict in ((0, False), (30, True)):
        lg, amm = make(fee)
        cash0, pos0 = lg[TRADER].cash, lg[TRADER].position
        got = amm.swap_exact_in(TRADER, Side.BUY, 100_000).amount_out
        amm.swap_exact_in(TRADER, Side.SELL, got)
        assert lg[TRADER].position == pos0
        loss = cash0 - lg[TRADER].cash
        assert loss > 0 if strict else loss >= 0


def test_fixed_friction_routes_to_venue_sink_and_invariants_hold():
    stack = FrictionStack([FixedCost(7)])
    lg, amm = make(frictions=stack)
    cash0 = lg[TRADER].cash
    r1 = amm.swap_exact_in(TRADER, Side.BUY, 10_000)
    r2 = amm.swap_exact_in(TRADER, Side.SELL, 50)
    assert r1.friction_fee == r2.friction_fee == 7
    assert lg.venue.cash == lg.fees_collected == 14
    assert lg[POOL].fees_paid == 0 and lg[TRADER].fees_paid == 14
    assert lg[TRADER].cash == cash0 - 10_000 - 7 + r2.amount_out - 7
    assert sum(a.cash for a in lg) + lg.venue.cash == lg.cash_in
    assert sum(a.position for a in lg) == lg.inventory_in
    lg.check()


def test_no_overdraft_mode_rejects_unfunded_swap():
    lg = Ledger(allow_overdraft=False, allow_short=False)
    lg.open_account(TRADER, cash=500, position=0)
    amm = ConstantProductAMM(lg, POOL, 10_000, 1_000_000, frictions=FrictionStack([FixedCost(5)]))
    with pytest.raises(LedgerError):
        amm.swap_exact_in(TRADER, Side.BUY, 499)  # 499 + 5 friction > 500
    with pytest.raises(LedgerError):
        amm.swap_exact_in(TRADER, Side.SELL, 10)
    amm.swap_exact_in(TRADER, Side.BUY, 495)
    lg.check()


def _seeded_run(seed):
    lg, amm = make(fee_bps=25, frictions=FrictionStack([FixedCost(2)]))
    rng = np.random.default_rng(seed)
    for _ in range(30):
        if rng.random() < 0.5:
            amm.swap_exact_in(TRADER, Side.BUY, int(rng.integers(500, 30_000)))
        else:
            amm.swap_exact_in(TRADER, Side.SELL, int(rng.integers(5, 300)))
    return amm.reserves, lg[TRADER].cash, [(s.price, s.qty) for s in lg.journal]


def test_deterministic_under_seed():
    assert _seeded_run(11) == _seeded_run(11)
    assert _seeded_run(11) != _seeded_run(12)


def test_invalid_construction():
    lg = Ledger()
    with pytest.raises(ValueError):
        ConstantProductAMM(lg, POOL, 0, 100)
    with pytest.raises(ValueError):
        ConstantProductAMM(lg, POOL + 1, 10, 100, fee_bps=10_000)
    with pytest.raises(LedgerError):
        ConstantProductAMM(lg, 0, 10, 100)  # venue id reserved
