"""fl-071: exact (S, s) band, adjustment frequency and the bisected rigidity threshold."""
from fractions import Fraction
from pathlib import Path

import pytest

from frictionlab.experiments import (
    MenuCostConfig, adjustment_frequency, find_rigidity_threshold, optimal_ss_band, per_step_loss,
)
from frictionlab.experiments.menu_cost import COST, INFL, SMALL, _rigid, build

ROOT = Path(__file__).resolve().parents[1]
COSTS, DRIFTS = range(0, 130), (1, 2, 3, 4, -2)
PRESET = MenuCostConfig(n_sellers=SMALL["n_sellers"], horizon=SMALL["horizon"],
                        shocks=tuple(SMALL["shocks"]))


@pytest.mark.parametrize("d", DRIFTS)
def test_band_zero_at_zero_cost_and_non_decreasing(d):
    bands = [optimal_ss_band(f, 1, d) for f in COSTS]
    assert bands[0] == 0 and bands == sorted(bands)
    assert all(optimal_ss_band(f, 3, d) <= b for f, b in zip(COSTS, bands))   # steeper loss, tighter band
    freqs = [adjustment_frequency(b, d) for b in bands]
    assert freqs[0] == 1 and freqs == sorted(freqs, reverse=True)


@pytest.mark.parametrize("f,d", [(8, 1), (32, 2), (64, 3), (100, 2), (5, 1)])
def test_band_is_smallest_global_minimizer(f, d):
    b = optimal_ss_band(f, 1, d)
    vals = [per_step_loss(x, f, 1, d) for x in range(60)]
    assert vals[b] == min(vals) and all(v > vals[b] for v in vals[:b])
    assert isinstance(vals[b], Fraction) and per_step_loss(0, f, 1, d) == f


def test_frequency_closed_forms_and_bad_inputs():
    assert adjustment_frequency(0, 3) == 1 and adjustment_frequency(5, 0) == 0
    assert adjustment_frequency(4, 2) == Fraction(1, 4)       # overshoot 2: gaps 2, 0, -2, -4
    assert adjustment_frequency(3, 1) == Fraction(1, 5)
    assert optimal_ss_band(50, 2, 0) == 0
    with pytest.raises(ValueError):
        optimal_ss_band(-1, 1, 1)
    with pytest.raises(ValueError):
        optimal_ss_band(4, 0, 1)


@pytest.mark.parametrize("f,d", [(0, 1), (2, 1), (8, 1), (32, 1), (8, 2), (32, 2), (64, 3), (32, 4)])
def test_zero_shock_frequency_equals_analytic(f, d):
    cfg = MenuCostConfig(n_sellers=2, horizon=840, shocks=(0,))       # 840 = lcm(1..8)
    m = build(cfg, {COST: f, INFL: d}, 5).run().market
    band = optimal_ss_band(f, cfg.curvature, d)
    assert Fraction(m.adjustments, len(m.log)) == adjustment_frequency(band, d)
    loss = sum(r.loss for r in m.log) + m.menu.revenue
    assert Fraction(loss, len(m.log)) == per_step_loss(band, f, cfg.curvature, d)


@pytest.mark.parametrize("f", SMALL["menu_costs"])
@pytest.mark.parametrize("d", SMALL["inflations"])
def test_analytic_band_beats_neighbours_on_small_preset(f, d):
    def sim(band):
        m = build(PRESET, {COST: f, INFL: d, "band": band}, 71).run().market
        return Fraction(sum(r.loss for r in m.log) + m.menu.revenue, len(m.log))

    b = optimal_ss_band(f, PRESET.curvature, d)
    here = per_step_loss(b, f, PRESET.curvature, d)
    for nb in (b - 1, b + 1):
        if nb >= 0:
            assert here <= per_step_loss(nb, f, PRESET.curvature, d)
            assert sim(b) <= sim(nb), (f, d, b, nb)


@pytest.mark.parametrize("cfg,d,bps", [(PRESET, 2, 5_000), (PRESET, 1, 3_000),
                                       (MenuCostConfig(), 2, 5_000)])
def test_bisected_threshold_is_tight(cfg, d, bps):
    thr = find_rigidity_threshold(cfg, d, rigidity_bps=bps, master_seed=71)
    base = thr.probes[0]
    assert base == thr.baseline_adjustments and thr.menu_cost >= 1
    assert _rigid(thr.probes[thr.menu_cost], base, bps)
    assert not _rigid(thr.probes[thr.menu_cost - 1], base, bps)
    assert thr.analytic_frequency == adjustment_frequency(thr.band, d)
    if cfg.shocks == (0,):
        assert thr.menu_cost == thr.analytic
        assert thr.analytic_frequency < Fraction(bps, 10_000)


def test_threshold_bad_inputs():
    with pytest.raises(ValueError):
        find_rigidity_threshold(PRESET, 0)
    with pytest.raises(ValueError):
        find_rigidity_threshold(PRESET, 2, rigidity_bps=0)


def test_new_modules_under_line_ceiling():
    for rel in ("frictionlab/experiments/menu_cost.py", "frictionlab/experiments/registry.py",
                "frictionlab/experiments/__init__.py"):
        assert len((ROOT / rel).read_text("utf-8").splitlines()) <= 250, rel
