"""Tests for fl-067 `CongestionCost`: join rule, toll / waiting sinks, reconciliation."""
from __future__ import annotations

import pytest

from frictionlab.core.ledger import Ledger, LedgerError
from frictionlab.costs import (
    TOLL_SINK_ID, WAIT_SINK_ID, CongestionCost, FixedCost, FrictionStack, HalfSpreadCost,
)
from frictionlab.markets import QueueMarket


def test_join_rule_ties_join():
    cc = CongestionCost(toll=5, waiting=3)
    assert cc.toll_for(2) == 5 and cc.wait_for(2) == 3
    assert cc.expected_net(20, 5) == 0 and cc.joins(20, 5)
    assert not cc.joins(19, 5) and cc.joins(21, 5)
    assert CongestionCost().joins(0, 100) and not CongestionCost().joins(-1, 0)


def test_accepts_cost_models_and_stacks():
    cc = CongestionCost(FixedCost(4), FrictionStack([FixedCost(2), HalfSpreadCost(0)]))
    assert cc.toll_for(3) == 4 and cc.wait_for(3) == 2
    assert CongestionCost(HalfSpreadCost(0)).toll_for(3) == 0


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        CongestionCost(toll=-1)
    with pytest.raises(ValueError):
        CongestionCost(waiting=-2)
    with pytest.raises(ValueError):
        CongestionCost(toll_sink_id=-20, waiting_sink_id=-20)
    with pytest.raises(TypeError):
        CongestionCost(toll=1.5)
    with pytest.raises(ValueError):
        CongestionCost(1, 1).expected_net(10, -1)


def test_charges_route_to_separate_sinks_and_conserve():
    led = Ledger()
    led.open_account(7, 100)
    cc = CongestionCost(toll=9, waiting=2)
    assert cc.charge_toll(led, 7) == 9
    for _ in range(4):
        cc.charge_wait(led, 7)
    assert led[TOLL_SINK_ID].cash == 9 == cc.toll_revenue
    assert led[WAIT_SINK_ID].cash == 8 == cc.waiting_paid
    assert cc.customer_steps == 4 and cc.wait_steps[7] == 4
    assert led[7].cash == 83 and sum(a.cash for a in led) == 100
    led.check()


def test_attach_is_idempotent_and_custom_sinks():
    led = Ledger()
    cc = CongestionCost(1, 1, toll_sink_id=-31, waiting_sink_id=-32)
    cc.attach(led)
    cc.attach(led)
    assert -31 in led and -32 in led and TOLL_SINK_ID not in led


def test_overdraft_guard_blocks_unaffordable_toll():
    led = Ledger(allow_overdraft=False)
    led.open_account(3, 4)
    cc = CongestionCost(toll=5)
    with pytest.raises(LedgerError):
        cc.charge_toll(led, 3)
    assert led[3].cash == 4 and cc.toll_revenue == 0


@pytest.mark.parametrize("seed", [1, 2, 67])
def test_market_tolls_reconcile_with_toll_sink(seed):
    m = QueueMarket(reward=50, arrival_prob_bps=8000, service_prob_bps=5000, steps=250,
                    toll=7, waiting=3, seed=seed)
    m.run()
    assert m.joins > 0
    assert m.congestion.toll_revenue == 7 * m.joins == m.ledger[TOLL_SINK_ID].cash
    assert set(m.congestion.tolls) == set(m.customers)
    balkers = {a.customer for a in m.arrivals if not a.joined}
    assert all(m.ledger[b].cash == 0 for b in balkers)


@pytest.mark.parametrize("seed", [1, 2, 67])
def test_market_waiting_reconciles_with_waiting_sink(seed):
    m = QueueMarket(reward=50, arrival_prob_bps=8000, service_prob_bps=5000, steps=250,
                    toll=7, waiting=3, seed=seed)
    m.run()
    steps = sum(c.steps for c in m.customers.values())
    assert steps == sum(m.lengths) == m.congestion.customer_steps
    assert m.congestion.waiting_paid == 3 * steps == m.ledger[WAIT_SINK_ID].cash


def test_shared_congestion_instance_used_by_market():
    cc = CongestionCost(toll=2, waiting=1)
    m = QueueMarket(reward=30, arrival_prob_bps=6000, service_prob_bps=6000, steps=50,
                    congestion=cc, seed=5)
    m.run()
    assert m.congestion is cc and cc.toll_revenue == 2 * m.joins
    assert "CongestionCost" in repr(cc) and "QueueMarket" in repr(m)
