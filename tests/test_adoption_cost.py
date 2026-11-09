"""Tests for fl-064 `AdoptionCost`: join rule, sink settlement, rejoin charging."""
from __future__ import annotations

import pytest

from frictionlab.core.ledger import Ledger, LedgerError
from frictionlab.costs import (
    ADOPTION_SINK_ID, AdoptionCost, FixedCost, FrictionStack, HalfSpreadCost,
)
from frictionlab.markets import PlatformMarket


def test_join_rule_ties_join():
    ac = AdoptionCost(5)
    assert ac.cost_for(3) == 5
    assert ac.joins(5) and ac.joins(9) and not ac.joins(4)
    assert AdoptionCost(0).joins(0) and not AdoptionCost(0).joins(-1)


def test_accepts_cost_models_and_stacks():
    assert AdoptionCost(FixedCost(4)).cost_for(2) == 4
    stack = FrictionStack([FixedCost(5), HalfSpreadCost(0)])
    assert AdoptionCost(stack).cost_for(2) == 5
    assert AdoptionCost(HalfSpreadCost(0)).cost_for(2) == 0


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        AdoptionCost(-1)
    with pytest.raises(TypeError):
        AdoptionCost(True)
    with pytest.raises(TypeError):
        AdoptionCost(2).joins(1.5)


def test_charge_moves_cash_to_sink_and_conserves():
    led = Ledger()
    led.open_account(1, 100)
    led.open_account(2, 10)
    ac = AdoptionCost(7)
    assert ac.charge(led, 2) == 7 and ac.charge(led, 2, payer=1) == 7
    assert led[ADOPTION_SINK_ID].cash == 14 == ac.revenue
    assert led[2].cash == 3 and led[1].cash == 93
    assert ac.count == 2 and ac.charges == {2: 2} and ac.paid == {2: 7, 1: 7}
    led.check()


def test_attach_is_idempotent_and_custom_sink():
    led = Ledger()
    ac = AdoptionCost(1, sink_id=-42)
    ac.attach(led)
    ac.attach(led)
    assert -42 in led and ADOPTION_SINK_ID not in led


def test_overdraft_guard_blocks_unaffordable_charge():
    led = Ledger(allow_overdraft=False)
    led.open_account(5, 2)
    with pytest.raises(LedgerError):
        AdoptionCost(3).charge(led, 5)
    assert led[5].cash == 2


def test_adoption_wedge_separates_join_and_stay():
    """A member whose net value sits in [0, adoption) stays but would not join."""
    mkt = PlatformMarket((1, 1), standalone=([2], [2]), benefit=(0, 0), fee=(0, 0),
                         adoption=5, seed_members=(1, 1), rounds=5)
    mkt.run()
    assert mkt.members == {2, 3} and mkt.adoption.count == 2
    cold = PlatformMarket((1, 1), standalone=([2], [2]), adoption=5, rounds=5)
    cold.run()
    assert cold.members == set()


def test_rejoin_pays_adoption_again():
    """2-cycle: seeded A leaves, B joins, then A rejoins (pays again) and B leaves."""
    mkt = PlatformMarket((1, 1), standalone=([-1], [-1]), benefit=(2, 2), fee=(0, 0),
                         adoption=1, seed_members=(1, 0), rounds=8)
    assert mkt.run() == 2
    assert [r.members for r in mkt.log] == [(0, 1), (1, 0)]
    assert mkt.history[-1] == mkt.history[0] == frozenset({2})
    assert mkt.adoption.charges == {2: 2, 3: 1}
    assert mkt.adoption.paid == {1: 1, 2: 1, 3: 1}
    assert mkt.ledger[ADOPTION_SINK_ID].cash == 3 == mkt.adoption.revenue
    mkt.ledger.check()
