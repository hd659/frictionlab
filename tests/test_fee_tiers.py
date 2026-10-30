"""Volume-tiered fee schedules: monotone tiers, trailing window, rebates, parity."""
import numpy as np
import pytest

from frictionlab.agents import make_zi_population
from frictionlab.core import Side
from frictionlab.core.ledger import Ledger
from frictionlab.core.runner import MarketRunner
from frictionlab.core.types import Trade
from frictionlab.costs import FixedCost, FrictionStack, MakerTakerFee
from frictionlab.costs.base import Fill, Role
from frictionlab.markets import FeeComponent, FeeTier, FeeTierSchedule, settle_tiered

TIERS = [FeeTier(0, 0, 30), FeeTier(10, -2, 20), FeeTier(50, -5, 10), FeeTier(200, -8, 4)]


def _trade(tid, buyer, seller, price=100, qty=1, aggressor=Side.BUY):
    return Trade(tid, 0.0, price, qty, tid * 2, tid * 2 + 1, buyer, seller, aggressor)


def test_schedule_validation():
    with pytest.raises(ValueError):
        FeeTierSchedule([])
    with pytest.raises(ValueError):
        FeeTierSchedule([FeeTier(5, 0, 10)])            # must start at 0
    with pytest.raises(ValueError):
        FeeTierSchedule([FeeTier(0, 0, 10), FeeTier(0, 0, 5)])
    with pytest.raises(ValueError):
        FeeTierSchedule([FeeTier(0, 0, 10), FeeTier(5, 0, 12)])  # fee rises
    with pytest.raises(ValueError):
        FeeTier(0, 0, -1)
    with pytest.raises(ValueError):
        FeeTierSchedule(TIERS, window=0)


def test_tier_lookup_is_monotone():
    sched = FeeTierSchedule(TIERS)
    assert sched.tier_for(0) is TIERS[0] and sched.tier_for(9) is TIERS[0]
    assert sched.tier_for(10) is TIERS[1] and sched.tier_for(10**9) is TIERS[-1]
    rng = np.random.default_rng(31)
    vols = np.sort(rng.integers(0, 500, size=200))
    taker = [sched.tier_for(int(v)).rate(Role.TAKER) for v in vols]
    maker = [sched.tier_for(int(v)).rate(Role.MAKER) for v in vols]
    assert all(a >= b for a, b in zip(taker, taker[1:]))
    assert all(a >= b for a, b in zip(maker, maker[1:]))


def test_trailing_window_expires_old_volume():
    fc = FeeComponent(FeeTierSchedule(TIERS, window=3))
    fc.record(1, 12, step=0)
    assert fc.trailing_volume(1) == 0                  # current step excluded
    fc.advance(1)
    assert fc.trailing_volume(1) == 12 and fc.tier_of(1) is TIERS[1]
    fc.record(1, 40)                                   # at step 1
    fc.advance(3)
    assert fc.trailing_volume(1) == 52 and fc.tier_of(1) is TIERS[2]
    fc.advance(4)                                      # step 0 leaves [1, 4)
    assert fc.trailing_volume(1) == 40 and fc.tier_of(1) is TIERS[1]
    fc.advance(5)
    assert fc.trailing_volume(1) == 0 and fc.tier_of(1) is TIERS[0]
    with pytest.raises(ValueError):
        fc.advance(2)
    with pytest.raises(ValueError):
        fc.record(1, 1, step=0)


def test_sync_reads_ledger_journal_and_lowers_fees():
    fc = FeeComponent(FeeTierSchedule(TIERS, window=10))
    led = Ledger(FrictionStack([fc]))
    for a in (1, 2):
        led.open_account(a, 10**6, 10**3)
    led.settle(_trade(1, 1, 2, qty=60))
    fill = Fill(1, Side.BUY, Role.TAKER, 100, 10)
    before = fc.charge(fill)
    fc.advance(1, led)
    assert fc.trailing_volume(1) == 60 == fc.trailing_volume(2)
    assert fc.charge(fill) < before
    assert fc.sync(led) == 0                           # idempotent cursor
    with pytest.raises(ValueError):
        fc.sync(Ledger())


def test_rebates_net_to_zero_across_accounts_and_sink():
    fc = FeeComponent(FeeTierSchedule([FeeTier(0, -5, 12)]))
    led = Ledger(FrictionStack([fc]))
    for a in (1, 2, 3):
        led.open_account(a, 10**7, 10**4)
    rng = np.random.default_rng(7)
    for tid in range(1, 41):
        b, s = rng.choice([1, 2, 3], size=2, replace=False)
        agg = Side.BUY if rng.random() < 0.5 else Side.SELL
        led.settle(_trade(tid, int(b), int(s), int(rng.integers(90, 110)),
                          int(rng.integers(50, 300)), agg))
    led.check()
    assert any(s.buyer_fee < 0 or s.seller_fee < 0 for s in led.journal)
    assert sum(a.cash for a in led) + led.venue.cash == led.cash_in
    assert led.venue.cash == sum(a.fees_paid for a in led) > 0


def test_sink_policy_rejects_rebate_that_overdraws_sink():
    fc = FeeComponent(FeeTierSchedule([FeeTier(0, -30, 5)]))
    for policy, venue_cash in (("allow", -25), ("reject", 5)):
        led = Ledger(FrictionStack([fc]))
        led.open_account(1, 10**6, 0)
        led.open_account(2, 10**6, 100)
        # notional 10_000: buyer taker +5, seller maker rebate -30
        res, rejected = settle_tiered(led, [_trade(1, 1, 2, 100, 100)], policy)
        led.check()
        assert led.venue.cash == venue_cash
        assert rejected == (30 if policy == "reject" else 0)
    with pytest.raises(ValueError):
        settle_tiered(led, [], "nope")


def _zi_run(stack):
    agents = make_zi_population("zi-c", 4, 4, seed=11, n_units=2)
    r = MarketRunner(agents, stack, seed=11, cash=10**6, position=10)
    return r.run(max_wakeups=400)


def test_one_tier_schedule_reproduces_flat_fee_bit_for_bit():
    flat = _zi_run(FrictionStack([MakerTakerFee(-1, 7), FixedCost(2)]))
    tier = _zi_run(FrictionStack([FeeComponent(FeeTierSchedule.flat(-1, 7)), FixedCost(2)]))
    assert len(flat.trades) > 0 and flat.trades == tier.trades
    assert flat.ledger.journal == tier.ledger.journal
    key = lambda res: [(a.agent_id, a.cash, a.position, a.fees_paid) for a in res.ledger]
    assert key(flat) == key(tier) and flat.fees == tier.fees
