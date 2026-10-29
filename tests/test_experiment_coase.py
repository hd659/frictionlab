"""Tests for frictionlab.experiments.coase and examples/phase5_coase_threshold.py."""
import importlib.util
import sys
import time
from pathlib import Path

import pytest

from frictionlab.costs.base import Fill, Role
from frictionlab.core.types import Side
from frictionlab.experiments import (
    CoaseConfig, coase_friction_stack, default_levels, find_coase_threshold, max_volume,
    run_coase, split_friction,
)

PATH = Path(__file__).resolve().parents[1] / "examples" / "phase5_coase_threshold.py"
CFG = CoaseConfig()
BOUND = CFG.analytic_bound
LEVELS = sorted({*default_levels(CFG), BOUND, BOUND + 1, BOUND + 7})


def _load():
    if str(PATH.parent) not in sys.path:   # spawned sweep workers re-import by name
        sys.path.insert(0, str(PATH.parent))
    spec = importlib.util.spec_from_file_location("phase5_coase_threshold", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def res():
    return run_coase(CFG, LEVELS, replicates=4, master_seed=28)


@pytest.fixture(scope="module")
def threshold():
    return find_coase_threshold(CFG, 0.5, replicates=4, master_seed=28)


@pytest.mark.parametrize("level", [0, 1, 7, 33, 100, 101])
@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_split_friction_is_exact_and_charges_level_per_trade(level, share):
    fixed, taker = split_friction(level, share)
    assert fixed >= 0 and taker >= 0 and 2 * fixed + taker == level
    st = coase_friction_stack(level, share)
    maker = st.charge(Fill(1, Side.BUY, Role.MAKER, 120, 1))
    taker_fee = st.charge(Fill(2, Side.SELL, Role.TAKER, 120, 1))
    assert maker + taker_fee == level
    with pytest.raises(ValueError):
        split_friction(-1)


def test_analytic_bound_and_max_volume():
    assert BOUND == 160 - 60 == 100
    assert max_volume(CFG.values, CFG.costs, 0) == 6
    assert max_volume(CFG.values, CFG.costs, BOUND) == 1
    assert max_volume(CFG.values, CFG.costs, BOUND + 1) == 0
    caps = [max_volume(CFG.values, CFG.costs, f) for f in range(BOUND + 2)]
    assert all(b <= a for a, b in zip(caps, caps[1:]))


def test_ledger_invariants_hold_in_every_cell(res):
    assert len(res.rows) == len(LEVELS) * 4 and not res.errors
    for r in res.rows:
        assert r["violations"] == 0 and r["zero_sum_residual"] == 0
        assert r["friction_residual"] == 0              # sink == F * volume exactly
        assert r["volume"] <= r["max_volume"]           # every fill is individually rational
        assert -1e-12 <= r["surplus_share"] <= r["gross_share"] <= 1 + 1e-12


def test_volume_and_surplus_non_increasing_in_friction(res):
    vol, share = res.column("volume"), res.column("surplus_share")
    assert all(b <= a for a, b in zip(vol, vol[1:])), vol
    assert all(b <= a + 1e-12 for a, b in zip(share, share[1:])), share
    assert vol[0] > vol[LEVELS.index(BOUND)]


def test_volume_exactly_zero_above_bound(res):
    for r in res.rows:
        if r["friction"] > BOUND:
            assert r["volume"] == 0 and r["friction_collected"] == 0
            assert r["surplus_share"] == 0
        if r["friction"] == BOUND:
            assert r["volume"] <= 1


def test_findings(res):
    f = res.findings
    assert f["analytic_bound"] == BOUND and f["ce_quantity"] == 6 and f["ce_max_surplus"] == 300
    assert f["first_zero_volume_level"] == BOUND + 1
    assert f["frictionless_volume"] == res.column("volume")[0]


def test_threshold_within_bound_and_consistent_with_grid(threshold):
    th = threshold
    assert th.analytic_bound == BOUND and 0 <= th.threshold <= BOUND
    lo, hi = th.bracket
    assert lo < th.threshold <= hi
    assert th.target == pytest.approx(0.5 * th.baseline_volume)
    p = th.probes
    assert p[th.threshold] < th.target and p[th.threshold - 1] >= th.target
    below = [lv for lv in p if lv < th.threshold]
    assert all(p[lv] >= th.target for lv in below)          # first crossing
    grid_hit = min(lv for lv in default_levels(CFG) if p[lv] < th.target)
    assert grid_hit == hi                                    # scan and bisection agree


def test_threshold_moves_with_fraction():
    loose = find_coase_threshold(CFG, 0.9, replicates=4, master_seed=28)
    strict = find_coase_threshold(CFG, 0.1, replicates=4, master_seed=28)
    assert loose.threshold <= strict.threshold <= BOUND + 1
    with pytest.raises(ValueError):
        find_coase_threshold(CFG, 0.0)


def test_serial_and_parallel_rows_identical(res):
    assert run_coase(CFG, LEVELS, replicates=4, master_seed=28, n_workers=2).rows == res.rows


def test_reproducible_from_master_seed(res, threshold):
    assert run_coase(CFG, LEVELS, replicates=4, master_seed=28).rows == res.rows
    assert run_coase(CFG, LEVELS, replicates=4, master_seed=29).rows != res.rows
    assert find_coase_threshold(CFG, 0.5, replicates=4, master_seed=28) == threshold


def test_example_main_small_runs_fast(capsys, tmp_path):
    ex = _load()
    t0 = time.perf_counter()
    assert ex.main(["--small", "--report", str(tmp_path / "coase.html")]) == 0
    assert time.perf_counter() - t0 < 20
    out = capsys.readouterr().out
    assert "Coase threshold" in out and "analytic bound" in out
    html = (tmp_path / "coase.html").read_text()
    assert "coase_threshold" in html and "Coase threshold: sweep" in html
