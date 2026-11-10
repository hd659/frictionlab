"""Tests for the fl-073 `ContractingCost` component and its sink reconciliation."""
from __future__ import annotations

import pytest

from frictionlab.agents import SpecificInvestor
from frictionlab.core.ledger import Ledger
from frictionlab.costs import CONTRACT_SINK_ID, ContractingCost, FixedCost, FrictionStack
from frictionlab.markets import HoldupMarket

SCHED = ((0, 20), (4, 30), (10, 40), (18, 48), (30, 55))


def _market(cost, seed=73, enabled=True):
    cc = ContractingCost(cost) if enabled else ContractingCost.disabled()
    sellers = [SpecificInvestor(i) for i in (1, 2, 3)]
    return HoldupMarket(sellers, [SCHED] * 3, [10, 11, 12], rounds=20, outside_value=15,
                        seller_weight_bps=5_000, value_noise=(-2, 0, 1, 4),
                        contracting=cc, seed=seed)


def test_flat_cost_and_stack_inputs():
    assert ContractingCost(7).cost_for(1) == 7
    stack = FrictionStack([FixedCost(5)])
    assert ContractingCost(stack).cost_for(4) == 5
    assert ContractingCost(FixedCost(4)).cost_for(4) == 4
    with pytest.raises(ValueError):
        ContractingCost(-1)


def test_charge_moves_cash_to_sink():
    led = Ledger()
    led.open_account(1, 100)
    cc = ContractingCost(6)
    assert cc.charge(led, 1) == 6 and cc.charge(led, 1) == 6
    assert led[1].cash == 88 and led[CONTRACT_SINK_ID].cash == 12
    assert cc.contracts == 2 and cc.contracts_of(1) == 2 and cc.revenue == 12
    led.check()


def test_disabled_component_refuses_to_charge():
    led = Ledger()
    led.open_account(1, 10)
    cc = ContractingCost.disabled()
    assert not cc.enabled
    with pytest.raises(RuntimeError):
        cc.charge(led, 1)


@pytest.mark.parametrize("cost", [0, 1, 3])   # contracting gain is 4 ticks per pair
def test_contract_count_times_cost_reconciles_with_sink(cost):
    m = _market(cost)
    m.run()
    assert m.contracts > 0
    assert m.contracting.contracts == m.contracts
    assert m.ledger[CONTRACT_SINK_ID].cash == cost * m.contracts == m.contracting.revenue
    for s in m.sellers:
        n = sum(1 for r in m.log if r.seller == s.agent_id and r.segment == "contract")
        assert m.contracting.contracts_of(s.agent_id) == n


@pytest.mark.parametrize("cost", [4, 10_000])  # tie at the 4-tick gain: no contract
def test_cost_at_or_above_gain_means_no_contracts(cost):
    m = _market(cost)
    assert m.run() == 0
    assert m.ledger[CONTRACT_SINK_ID].cash == 0
    assert all(r.segment == "holdup" for r in m.log)


def test_disabled_market_has_no_contract_sink():
    m = _market(0, enabled=False)
    m.run()
    assert m.contracts == 0 and CONTRACT_SINK_ID not in m.ledger
    assert "enabled=False" in repr(m.contracting)
