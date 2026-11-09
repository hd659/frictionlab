"""Tests for fl-070 `SsPricer`: (S, s) inaction band, overshoot and flexible mode."""
from __future__ import annotations

import pytest

from frictionlab.agents import SsPricer, half_life, overshoot
from frictionlab.markets import MenuCostMarket


def test_half_life_and_overshoot():
    assert half_life(10, 0) == 0 and overshoot(10, 0) == 0
    assert half_life(10, 1) == 5 and overshoot(10, 1) == 5
    assert half_life(10, 2) == 2 and overshoot(10, 2) == 4
    assert overshoot(10, -2) == -4
    assert overshoot(0, 3) == 0
    for band in range(0, 20):
        for d in (-3, -1, 1, 2, 5):
            assert abs(overshoot(band, d)) <= band // 2


def test_band_rule_ties_keep_price():
    s = SsPricer(1, 4)
    s.start(100)
    assert not s.should_adjust(100, 104) and not s.should_adjust(100, 96)
    assert s.should_adjust(100, 105) and s.should_adjust(100, 95)
    assert s.reprice(104) is False and s.price == 100
    assert s.reprice(105) is True and s.price == 105
    assert s.prices == [100, 105] and s.adjusted == [False, True]


def test_reset_lands_inside_band_with_drift():
    s = SsPricer(1, 10, drift=1)
    s.start(100)
    assert s.reprice(111) and s.price == 116
    assert abs(s.price - 111) <= s.band


def test_flexible_mode():
    s = SsPricer.flexible(2, drift=3)
    assert s.is_flexible and s.overshoot == 0
    s.start(50)
    assert s.reprice(50) is False
    assert s.reprice(51) is True and s.price == 51


def test_reprice_requires_start_and_rejects_bad_inputs():
    with pytest.raises(RuntimeError):
        SsPricer(1, 2).reprice(10)
    with pytest.raises(ValueError):
        SsPricer(1, -1)
    with pytest.raises(TypeError):
        SsPricer(1, 1.5)
    assert SsPricer(1).decide(None) == ()


@pytest.mark.parametrize("seed", [3, 4, 70])
def test_never_adjusts_within_band_in_market(seed):
    sellers = [SsPricer(1, 6, drift=1), SsPricer(2, 12, drift=1)]
    m = MenuCostMarket(sellers, [100, 120], steps=150, inflation_ticks_per_step=1,
                       shocks=(-2, 0, 0, 2), menu=4, curvature=1, base_profit=30, seed=seed)
    m.run()
    for r in m.log:
        band = m.sellers[r.seller].band
        assert r.adjusted == (abs(r.gap_before) > band)
        if not r.adjusted:
            assert r.gap == r.gap_before
        else:
            assert r.price == r.target + m.sellers[r.seller].overshoot
    wide = sum(r.adjusted for r in m.path(2))
    narrow = sum(r.adjusted for r in m.path(1))
    assert 0 < wide < narrow


def test_menu_cost_creates_lumpy_adjustment():
    sellers = [SsPricer(1, 0, drift=1), SsPricer(2, 8, drift=1)]
    m = MenuCostMarket(sellers, [100, 100], steps=100, inflation_ticks_per_step=1,
                       menu=3, curvature=1, base_profit=10, seed=5)
    m.run()
    flex, sticky = m.path(1), m.path(2)
    assert all(r.adjusted for r in flex)
    assert sum(r.adjusted for r in sticky) <= 100 // 9 + 1
