import math

import pytest

from frictionlab.analysis import LinearScale, extent, format_tick, nice_ticks, ramp
from frictionlab.analysis.scales import fmt, luminance, nice_number


@pytest.mark.parametrize("lo,hi", [
    (0.0, 1.0), (0.13, 0.87), (-3.0, 47.0), (-250.0, -7.5), (99.0, 101.0),
    (1e-4, 3e-4), (0.0, 12345.0), (-1.0, 1.0),
])
def test_nice_ticks_bracket_data(lo, hi):
    ticks = nice_ticks(lo, hi)
    assert ticks[0] <= lo and ticks[-1] >= hi
    assert 2 <= len(ticks) <= 12
    steps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert all(math.isclose(s, steps[0], rel_tol=1e-9) for s in steps)
    mant = steps[0] / 10 ** math.floor(math.log10(steps[0]))
    assert any(math.isclose(mant, m, rel_tol=1e-6) for m in (1, 2, 5, 10))


def test_nice_ticks_degenerate_and_reversed():
    t = nice_ticks(5.0, 5.0)
    assert t[0] < 5.0 < t[-1]
    assert nice_ticks(10.0, 0.0) == nice_ticks(0.0, 10.0)
    assert nice_ticks(0.0, 0.0)[0] <= 0.0 <= nice_ticks(0.0, 0.0)[-1]


def test_nice_number():
    assert nice_number(7.3, False) == 10
    assert nice_number(1.2, True) == 1
    assert nice_number(0.0, True) == 1.0


def test_extent_handles_empty_constant_nan():
    assert extent([]) == (0.0, 1.0)
    assert extent([float("nan"), float("inf")]) == (0.0, 1.0)
    lo, hi = extent([3.0, 3.0, 3.0])
    assert lo < 3.0 < hi
    lo, hi = extent([0.0, 0.0])
    assert lo < 0.0 < hi
    assert extent([1.0, float("nan"), -2.0], [5.0]) == (-2.0, 5.0)


def test_linear_scale_maps_endpoints_and_inverts():
    s = LinearScale(0.0, 10.0, 100.0, 0.0)
    assert s(0.0) == 100.0 and s(10.0) == 0.0 and s(5.0) == 50.0
    assert LinearScale(2.0, 2.0, 0.0, 10.0)(2.0) == 5.0
    n = LinearScale.nice(0.13, 0.87, 0.0, 100.0)
    assert n.d0 <= 0.13 and n.d1 >= 0.87
    assert n(n.d0) == 0.0 and n(n.d1) == 100.0


def test_format_tick_integers_and_decimals():
    assert format_tick(20.0, 10.0) == "20"
    assert format_tick(-10.0, 10.0) == "-10"
    assert format_tick(0.0, 5.0) == "0"
    assert format_tick(0.2, 0.2) == "0.2"
    assert format_tick(0.05, 0.05) == "0.05"
    assert format_tick(1.0, 0.5) == "1"


def test_fmt_fixed_precision_no_negative_zero():
    assert fmt(1.0) == "1"
    assert fmt(1.23456) == "1.23"
    assert fmt(-0.0001) == "0"
    assert fmt(-0.0) == "0"
    assert fmt(2.5) == "2.5"


def test_ramp_monotone():
    colors = [ramp(i / 20) for i in range(21)]
    lum = [luminance(c) for c in colors]
    assert all(a > b for a, b in zip(lum, lum[1:]))
    assert ramp(-1.0) == ramp(0.0) and ramp(2.0) == ramp(1.0)
    assert all(len(c) == 7 and c.startswith("#") for c in colors)
