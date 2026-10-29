"""Regression tests for examples/phase4_analysis_toolkit.py (Phase 4 close-out)."""
import importlib.util
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab.analysis import aggregate, run_sweep

PATH = Path(__file__).resolve().parents[1] / "examples" / "phase4_analysis_toolkit.py"


def _load():
    if str(PATH.parent) not in sys.path:   # spawned sweep workers re-import by name
        sys.path.insert(0, str(PATH.parent))
    spec = importlib.util.spec_from_file_location("phase4_analysis_toolkit", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses / pickling resolve the module here
    spec.loader.exec_module(mod)
    return mod


ex = _load()
CFG = ex.Config()


@pytest.fixture(scope="module")
def rows():
    return run_sweep(ex.sweep_spec(CFG, ex.FULL))


@pytest.fixture(scope="module")
def runs():
    return {c: ex.replicator(CFG, c) for c in CFG.costs}


@pytest.fixture(scope="module")
def report_html(rows, runs):
    return ex.build_report(CFG, ex.FULL, rows, runs).to_html()


def test_grid_is_fully_covered_without_errors(rows):
    n = len(ex.FULL["taker_bps"]) * len(ex.FULL["latency"])
    assert len(rows) == n * CFG.replicates
    assert all(r["error"] is None for r in rows)


def test_ledger_invariants_hold_in_every_cell(rows):
    for r in rows:
        assert r["violations"] == 0
        assert r["zero_sum_residual"] == 0          # accounts + sinks close the books
        assert r["identity_residual"] == 0          # effective = realized + impact per trade


def test_spread_identity_holds_for_cell_means(rows):
    for r in rows:
        if r["volume"]:
            assert math.isclose(r["effective_spread"], r["realized_spread"] + r["price_impact"],
                                abs_tol=1e-9)


def test_reproducible_from_master_seed(rows):
    again = run_sweep(ex.sweep_spec(CFG, ex.FULL))
    assert repr(again) == repr(rows)                # repr: nan-safe exact comparison
    other = run_sweep(ex.sweep_spec(ex.Config(seed=CFG.seed + 1), ex.FULL))
    assert repr(other) != repr(rows)


def test_serial_and_parallel_rows_are_identical():
    spec = ex.sweep_spec(ex.Config(replicates=2), ex.SMALL)
    assert repr(run_sweep(spec, n_workers=1)) == repr(run_sweep(spec, n_workers=2))


def test_cells_share_seeds_across_friction_levels(rows):
    seeds = {}
    for r in rows:
        seeds.setdefault(r["replicate"], set()).add(r["seed"])
    assert all(len(s) == 1 for s in seeds.values())     # CRN: one seed per replicate
    assert len({next(iter(s)) for s in seeds.values()}) == CFG.replicates


def test_mean_welfare_is_non_increasing_in_taker_fee(rows):
    table = aggregate(rows, by=("taker_bps", "latency"))
    for lat in ex.FULL["latency"]:
        series = [r["welfare_surplus_mean"] for r in sorted(table, key=lambda r: r["taker_bps"])
                  if r["latency"] == lat]
        assert len(series) == len(ex.FULL["taker_bps"])
        assert all(b <= a for a, b in zip(series, series[1:])), (lat, series)
        fees = [r["fees_mean"] for r in sorted(table, key=lambda r: r["taker_bps"])
                if r["latency"] == lat]
        assert fees == sorted(fees) and fees[-1] > fees[0]
        if lat == 0:
            assert fees[0] == 0                       # no fee, no latency drift


def test_high_decision_cost_favours_satisficers(runs):
    low, high = (ex.settled_share(runs[c]) for c in CFG.costs)
    assert high >= low
    assert high > 0.5 and runs[CFG.costs[1]].converged
    assert runs[CFG.costs[1]].fitness[0][0] > runs[CFG.costs[1]].fitness[0][1]
    for r in runs.values():
        assert all(abs(x.sum() - 1) < 1e-12 and (x >= 0).all() for x in r.shares)


def test_report_is_well_formed_with_one_section_per_part(report_html):
    assert report_html.startswith("<!DOCTYPE html>\n")
    root = ET.fromstring(report_html.split("\n", 1)[1])
    sections = root.findall(".//section")
    assert [s.get("class") for s in sections] == ["sweep", "equilibrium"]
    titles = [s.find("h2").text for s in sections]
    assert titles[0].startswith("Part 1") and titles[1].startswith("Part 2")
    assert "http://" not in report_html and "https://" not in report_html
    assert "<script" not in report_html
    assert str(CFG.seed) in report_html and "taker_bps" in report_html
    for c in CFG.costs:
        assert f"cost_per_step = {c}" in report_html


def test_report_is_byte_identical_for_same_inputs(rows, runs, report_html):
    assert ex.build_report(CFG, ex.FULL, rows, runs).to_html() == report_html


def test_main_small_grid_writes_report_within_budget(tmp_path, capsys):
    t0 = time.perf_counter()
    assert ex.main(["--small", "--out", str(tmp_path)]) == 0
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, elapsed
    out = capsys.readouterr().out
    report = tmp_path / "phase4_report.html"
    assert report.is_file() and report.stat().st_size > 0
    assert f"report: {report}" in out and "satisficer share=" in out
    ET.fromstring(report.read_text(encoding="utf-8").split("\n", 1)[1])


def test_main_defaults_to_a_temp_dir(capsys):
    assert ex.main(["--small", "--seed", "3"]) == 0
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("report: ")][0]
    path = Path(line.split(" ", 1)[1])
    assert path.is_file() and "frictionlab-phase4-" in str(path.parent)
