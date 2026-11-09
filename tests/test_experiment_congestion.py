"""fl-068: congestion experiment over toll x waiting_cost on the fl-067 queue."""
import json
from pathlib import Path

import pytest

from frictionlab.config.schema import load_config, validate_params
from frictionlab.experiments import (
    CongestionConfig, get_experiment, list_experiments, resolve_config, run_congestion,
    run_from_config, run_queue_grid,
)
from frictionlab.experiments.congestion import SMALL as PRESET, TOLL, TOLLS, WAIT, WAITS, build

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "examples" / "configs" / "congestion_small.json"
CFG, MASTER = CongestionConfig(), 68


@pytest.fixture(scope="module")
def grid():
    return run_queue_grid(CFG, master_seed=MASTER)


def _cell(res, t, c):
    (row,) = [r for r in res.rows if r[TOLL] == t and r[WAIT] == c]
    return row


def test_registered_defaults_validate_against_schema():
    exp = get_experiment("congestion")
    assert "congestion" in list_experiments() and exp.master_seed == 68 and "toll" in exp.doc
    assert validate_params(exp.defaults, exp.param_schema) == exp.defaults
    assert set(exp.defaults) == set(exp.param_schema)
    assert exp.defaults["tolls"] == list(TOLLS) and exp.defaults["waiting_costs"] == list(WAITS)
    assert exp.defaults["reward"] == CFG.reward and exp.defaults["horizon"] == CFG.horizon


def test_shipped_small_config_equals_resolved_preset():
    raw, cfg = load_config(SMALL), resolve_config(SMALL)
    preset = resolve_config({"experiment": "congestion", "small": True})
    assert cfg.experiment == "congestion" and cfg.small and cfg.output is None
    assert raw["params"] == cfg.params == preset.params
    assert {k: cfg.params[k] for k in PRESET} == PRESET


def test_ledger_invariants_and_accounting_in_every_cell(grid):
    assert len(grid.rows) == len(TOLLS) * len(WAITS) and not grid.errors
    for r in grid.rows:
        assert r["violations"] == 0
        assert r["toll_revenue"] == r["joins"] * r[TOLL]
        assert r["welfare"] == r["customer_surplus"] + r["toll_revenue"]
        assert r["join_rate"] + r["balk_rate"] == pytest.approx(1.0)
        assert r["max_queue_length"] <= max(r["individual_threshold"], 0)
        if r[TOLL] == 0:
            assert r["welfare_change_vs_zero_toll"] == 0
    for t, c in ((8, 2), (32, 4)):
        cell = build(CFG, {TOLL: t, WAIT: c}, 3).run()
        for m in (cell.market, cell.twin):
            m.ledger.check()
            assert sum(a.cash for a in m.ledger) == 0
            assert m.congestion.waiting_paid == m.congestion.customer_steps * c


@pytest.mark.parametrize("key", ["join_rate", "mean_queue_length"])
def test_non_increasing_in_toll_and_waiting_cost(grid, key):
    for c in WAITS:
        xs = [_cell(grid, t, c)[key] for t in TOLLS]
        assert xs == sorted(xs, reverse=True), (c, key, xs)
    for t in TOLLS:
        xs = [_cell(grid, t, c)[key] for c in WAITS]
        assert xs == sorted(xs, reverse=True), (t, key, xs)


def test_findings_and_reproducibility_from_master_seed():
    small = CongestionConfig(horizon=120)
    a, b = (run_congestion(small, (0, 8, 29), (2,), master_seed=MASTER) for _ in range(2))
    assert a.rows == b.rows and a.findings == b.findings
    assert {"bisected_toll", "naor_optimal_toll", "social_threshold", "max_welfare_gain"} <= set(a.findings)
    assert a.findings["max_welfare_gain"] > 0
    other = run_congestion(small, (0, 8, 29), (2,), master_seed=MASTER + 1)
    assert [r["seed"] for r in other.rows] != [r["seed"] for r in a.rows]


def test_serial_and_parallel_rows_identical():
    ser = run_queue_grid(CFG, (0, 8, 32), (1, 4), master_seed=MASTER)
    par = run_queue_grid(CFG, (0, 8, 32), (1, 4), master_seed=MASTER, n_workers=2)
    assert ser.rows == par.rows


def test_run_from_config_rows_equal_direct_call():
    res = run_from_config(SMALL)
    raw = json.loads(SMALL.read_text())
    p = raw["params"]
    cfg = CongestionConfig(**{k: v for k, v in p.items() if k in CongestionConfig.__dataclass_fields__})
    direct = run_congestion(cfg, p["tolls"], p["waiting_costs"],
                            probe_waiting_cost=p["probe_waiting_cost"], master_seed=raw["master_seed"])
    assert res.rows == direct.rows and res.findings == direct.findings
    assert res.config["experiment"] == "congestion"
    assert res.findings["bisected_toll"] == res.findings["naor_optimal_toll"]
