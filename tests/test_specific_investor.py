"""Tests for the fl-073 `SpecificInvestor`: hold-up level, first best, contract choice."""
from __future__ import annotations

import pytest

from frictionlab.agents import (
    Agent, SpecificInvestor, buyer_share, first_best_level, holdup_level, seller_share,
)
from frictionlab.agents.specific_investor import check_schedule

SCHED = check_schedule(((0, 20), (4, 30), (10, 40), (18, 48), (30, 55)))
O = 15


def test_shares_sum_to_rent_and_follow_floor_convention():
    for rent in range(0, 60):
        for w in (0, 1, 3333, 5000, 7777, 10_000):
            b, s = buyer_share(rent, w), seller_share(rent, w)
            assert b + s == rent and b == (10_000 - w) * rent // 10_000
            assert 0 <= b <= rent and 0 <= s <= rent


def test_first_best_maximizes_joint_surplus():
    vals = [v - O - c for c, v in SCHED]
    i = first_best_level(SCHED, O)
    assert vals[i] == max(vals) and i == vals.index(max(vals))


@pytest.mark.parametrize("w", [0, 1000, 2500, 5000, 7500, 9999, 10_000])
def test_holdup_level_never_above_first_best(w):
    assert holdup_level(SCHED, O, w) <= first_best_level(SCHED, O)


def test_full_seller_weight_restores_first_best():
    assert holdup_level(SCHED, O, 10_000) == first_best_level(SCHED, O)
    assert holdup_level(SCHED, O, 5_000) < first_best_level(SCHED, O)


def test_smallest_index_wins_ties():
    flat = check_schedule(((0, 20), (5, 25)))       # joint surplus 5 at both levels
    assert first_best_level(flat, 15) == 0
    assert holdup_level(flat, 15, 10_000) == 0


def test_contract_choice_strict_and_ties_go_to_no_contract():
    s = SpecificInvestor(1)
    gap = s.contract_payoff(SCHED, O, 5_000) - s.holdup_payoff(SCHED, O, 5_000)
    assert gap > 0
    assert s.wants_contract(SCHED, O, 5_000, gap - 1)
    assert not s.wants_contract(SCHED, O, 5_000, gap)          # tie: no contract
    assert not s.wants_contract(SCHED, O, 10_000, 0)            # no hold-up: nothing to buy
    assert s.choose(SCHED, O, 5_000, 0, allow_contract=True) == (
        "contract", first_best_level(SCHED, O))
    assert s.choose(SCHED, O, 5_000, 0, allow_contract=False) == (
        "holdup", holdup_level(SCHED, O, 5_000))
    assert len(s.choices) == 2


def test_agent_base_and_no_book_intents():
    s = SpecificInvestor(3, decision_cost=1)
    assert isinstance(s, Agent) and s.decision_cost == 1
    assert tuple(s.decide(None)) == ()
    assert "SpecificInvestor" in repr(s)


@pytest.mark.parametrize("bad", [(), ((0, 10), (0, 12)), ((0, 10), (2, 9)), ((-1, 10),)])
def test_bad_schedules_rejected(bad):
    with pytest.raises(ValueError):
        check_schedule(bad)


def test_bad_weight_rejected():
    with pytest.raises(ValueError):
        seller_share(10, 10_001)
    with pytest.raises(ValueError):
        holdup_level(SCHED, O, -1)
