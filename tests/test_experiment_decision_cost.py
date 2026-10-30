"""Tests for frictionlab.experiments.decision_cost and examples/phase5_decision_cost.py."""
import importlib.util
import sys
import time
from fractions import Fraction
from pathlib import Path

import pytest

from frictionlab.experiments import (
    BreakEven, DecisionCostConfig, find_break_even_cost, population, run_decision_cost,
    run_decision_cost_study, settled_share,
)

ROOT = Path(__file__).resolve().parents[1] / "examples"
CFG = DecisionCostConfig()
COSTS, ASPS = (0, 1, 4, 16), (100.0, 500.0, 2000.0)


def _load(name):
    if str(ROOT) not in sys.path:   # spawned sweep workers re-import by name
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def res():
    return run_decision_cost(CFG, COSTS, ASPS, replicates=4, master_seed=29)


@pytest.fixture(scope="module")
def table():
    return {a: find_break_even_cost(CFG, a, replicates=4, master_seed=29) for a in ASPS}


def _cells(res):
    out = {}
    for r in res.aggregate():
        out.setdefault((r["aspiration"], r["cost_per_step"]), {})[r["mode"]] = r
    return out


def test_ledger_invariants_hold_in_every_cell(res):
    assert len(res.rows) == len(ASPS) * len(COSTS) * 2 * 4 and not res.errors
    for r in res.rows:
        assert r["violations"] == 0 and r["zero_sum_residual"] == 0
        assert r["decision_residual"] == 0 and r["decision"] == r["cost_per_step"] * r["steps"]
        assert r["net"] == r["gross"] - r["fees"] - r["decision"] and r["fees"] == 0
        assert r["volume"] > 0


def test_paired_modes_gross_and_steps_flat_in_cost(res):
    by = {(r["aspiration"], r["mode"], r["replicate"]): set() for r in res.rows}
    for r in res.rows:
        by[(r["aspiration"], r["mode"], r["replicate"])].add((r["gross"], r["steps"]))
    assert all(len(v) == 1 for v in by.values())       # step cost never moves prices
    for (asp, rep_mode, rep), v in by.items():
        if rep_mode == "satisficing":
            (_, s_steps), = v
            (_, o_steps), = by[(asp, "optimizing", rep)]
            assert s_steps <= o_steps                   # optimizer searches a superset


def test_zero_cost_optimizer_net_at_least_satisficer(res):
    for (asp, c), m in _cells(res).items():
        if c == 0:
            assert m["optimizing"]["net_mean"] >= m["satisficing"]["net_mean"], asp


def test_break_even_condition_holds_at_c_and_fails_below(table):
    for asp, b in table.items():
        assert isinstance(b, BreakEven) and b.cost is not None and b.cost > 0
        assert b.satisficer_wins(b.cost) and not b.satisficer_wins(b.cost - 1)
        s, o = b.probes[b.cost]
        assert o - s == b.gross_gain - b.cost * b.extra_steps        # net linear in c
        assert b.gross_gain < b.extra_steps * b.cost                  # dG < dS * c
        assert b.gross_gain >= b.extra_steps * (b.cost - 1)
        assert b.cost == b.analytic and isinstance(b.gross_gain, Fraction)


def test_break_even_consistent_with_grid(res, table):
    for (asp, c), m in _cells(res).items():
        sat_wins = m["satisficing"]["net_mean"] > m["optimizing"]["net_mean"]
        assert sat_wins == (c >= table[asp].cost), (asp, c)


def test_break_even_non_decreasing_in_max_steps():
    for asp in (100.0, 500.0):
        cs = [find_break_even_cost(CFG, asp, k, replicates=4, master_seed=29).cost
              for k in (2, 5, 10, 20)]
        assert all(b >= a for a, b in zip(cs, cs[1:])), (asp, cs)
        assert cs[-1] > cs[0]


def test_high_cost_satisficer_share_at_least_zero_cost():
    zero, high = population(CFG, 0), population(CFG, 64)
    assert high.converged and settled_share(high) >= settled_share(zero)
    assert settled_share(high) > 0.5
    for eq in (zero, high):
        assert all(abs(x.sum() - 1) < 1e-12 and (x >= 0).all() for x in eq.shares)


def test_study_findings_and_reproducible_from_master_seed():
    kw = dict(costs=(0, 8), aspirations=(500.0,), replicates=2, generations=4)
    a = run_decision_cost_study(CFG, **kw, master_seed=29)
    assert a.rows == run_decision_cost_study(CFG, **kw, master_seed=29).rows
    assert a.findings == run_decision_cost_study(CFG, **kw, master_seed=29).findings
    assert run_decision_cost(CFG, (0,), (500.0,), replicates=2, master_seed=30).rows != \
        run_decision_cost(CFG, (0,), (500.0,), replicates=2, master_seed=29).rows
    be, pop = a.findings["break_even"], a.findings["population"]
    assert [b["aspiration"] for b in be] == [500.0] and be[0]["cost"] == be[0]["analytic"]
    assert list(pop) == ["zero", "break_even", "high"]
    assert pop["zero"]["cost"] == 0 < pop["break_even"]["cost"] < pop["high"]["cost"]


def test_serial_and_parallel_rows_identical():
    kw = dict(replicates=2, master_seed=29)
    assert run_decision_cost(CFG, (0, 4), (500.0,), n_workers=2, **kw).rows == \
        run_decision_cost(CFG, (0, 4), (500.0,), **kw).rows


def test_phase2_seed_shares_the_market_and_still_crosses():
    ph2 = _load("phase2_decision_cost")
    p2 = ph2.Config()
    for k in ("zi_values", "zi_prices", "br_values", "max_steps", "aspiration",
              "search_width", "kappa", "wake_rate", "horizon"):
        assert getattr(p2, k) == getattr(CFG, k), k
    rows = ph2.sweep(ph2.Config(markets=3), (0, 16))
    gaps = [r["satisficing"]["net"] - r["optimizing"]["net"] for r in rows]
    assert ph2.crossover(rows) is not None and gaps[1] > max(gaps[0], 0)
    # the full phase 2 regression lives unchanged in test_example_phase2_decision_cost.py


def test_example_main_small_runs_fast(capsys, tmp_path):
    ex = _load("phase5_decision_cost")
    t0 = time.perf_counter()
    assert ex.main(["--small", "--report", str(tmp_path / "dc.html")]) == 0
    assert time.perf_counter() - t0 < 20
    out = capsys.readouterr().out
    assert "break-even cost_per_step vs aspiration" in out and "replicator fixed point" in out
    html = (tmp_path / "dc.html").read_text()
    assert "decision_cost" in html and "Decision cost: sweep" in html
