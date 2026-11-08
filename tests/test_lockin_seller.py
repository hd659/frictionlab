"""Tests for fl-061 `LockinSeller`: phases, switch-proof harvest pricing, Bertrand mode."""
from __future__ import annotations

import numpy as np
import pytest

from frictionlab.agents import (
    ACQUIRE, BERTRAND, HARVEST, Agent, LockinSeller, harvest_price,
)
from frictionlab.markets import SwitchingMarket

COSTS = [2, 4, 7]


def _values(n=40, seed=3):
    return [int(x) for x in np.random.default_rng(seed).integers(30, 90, size=n)]


def _duopoly(seed=61, rounds=12, entry=0):
    a = LockinSeller(1, 20, acquisition_price=14, target_bps=6000)
    b = LockinSeller(2, 20, acquisition_price=17, target_bps=6000, entry_round=entry)
    return SwitchingMarket(_values(), [a, b], switch_costs=COSTS, rounds=rounds, seed=seed)


def test_is_agent_and_decides_no_book_intents():
    s = LockinSeller(1, 20)
    assert isinstance(s, Agent) and s.decide(None) == ()
    assert s.acquisition_price == 20 and not s.bertrand


def test_phase_switches_at_target_share():
    s = LockinSeller(1, 20, acquisition_price=12, target_bps=7000)
    assert s.phase(6999) == ACQUIRE and s.phase(7000) == HARVEST
    assert LockinSeller.bertrand_seller(2, 20).phase(10_000) == BERTRAND
    late = LockinSeller(3, 20, entry_round=4)
    assert late.phase(0, 3) == "absent" and late.post_price("absent") is None
    assert late.phase(0, 4) == ACQUIRE


def test_harvest_price_is_rival_plus_min_cost_capped_by_valuation():
    assert harvest_price(30, [(80, 5), (90, 3)]) == 33
    assert harvest_price(30, [(31, 9), (90, 3)]) == 31
    assert harvest_price(None, [(50, 1), (60, 2)]) == 50
    with pytest.raises(ValueError):
        harvest_price(30, [])


def test_harvest_price_is_largest_switch_proof_price():
    custs = [(70, 4), (85, 6), (60, 9)]
    p = harvest_price(25, custs)
    assert all(not (25 + s < p) and p <= v for v, s in custs)
    assert any(25 + s < p + 1 for _, s in custs)          # one tick more loses someone


def test_harvest_never_induces_a_locked_in_customer_to_switch():
    for entry in (0, 3):
        mkt = _duopoly(entry=entry)
        mkt.run()
        harvested = 0
        by_round = {}
        for t in mkt.trades:
            by_round.setdefault(int(t.time), {})[t.buyer_id] = t.seller_id
        prev = {}
        for r in mkt.log:
            sold = by_round.get(r.t, {})
            for i, phase in enumerate(r.phases):
                if phase != HARVEST:
                    continue
                sid = i + 1
                locked = [c for c, k in prev.items() if k == sid]
                harvested += len(locked)
                assert all(sold.get(c) == sid for c in locked), (entry, r.t)
            prev.update(sold)
        assert harvested > 0


def test_invest_then_harvest_price_path():
    mkt = _duopoly(entry=3)
    mkt.run()
    s1 = mkt.sellers[1]
    assert s1.phases[0] == ACQUIRE and s1.prices[0] == 14
    assert HARVEST in s1.phases
    first = s1.phases.index(HARVEST)
    assert all(p == HARVEST for p in s1.phases[first:])
    assert max(s1.prices[first:]) > s1.marginal_cost > s1.prices[0]


def test_bertrand_mode_posts_marginal_cost_every_round():
    s = LockinSeller.bertrand_seller(1, 25)
    mkt = SwitchingMarket(_values(), [s, LockinSeller.bertrand_seller(2, 25)],
                          switch_costs=COSTS, rounds=5, seed=2)
    mkt.run()
    assert s.prices == [25] * 5 and set(s.phases) == {BERTRAND}
    assert all(r.prices == (25, 25) for r in mkt.log)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        LockinSeller(1, 20, target_bps=5000)
    with pytest.raises(ValueError):
        LockinSeller(1, 0)
    with pytest.raises(ValueError):
        LockinSeller(1, 20).post_price("nope")
