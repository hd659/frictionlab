"""fl-071: menu-cost experiment over menu_cost x inflation on the fl-070 market."""
import json
from pathlib import Path

import pytest

from frictionlab.analysis.pnl import menu_pnl
from frictionlab.config.schema import load_config, validate_params
from frictionlab.experiments import (
    MenuCostConfig, get_experiment, list_experiments, resolve_config, run_from_config,
    run_menu_cost, run_menu_grid,
)
from frictionlab.experiments.menu_cost import COST, COSTS, INFL, INFLS, SMALL as PRESET, build

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "examples" / "configs" / "menu_cost_small.json"
CFG, MASTER = MenuCostConfig(), 71


@pytest.fixture(scope="module")
def grid():
    return run_menu_grid(CFG, master_seed=MASTER)


def _cell(res, f, d):
    (row,) = [r for r in res.rows if r[COST] == f and r[INFL] == d]
    return row


def test_registered_defaults_validate_against_schema():
    exp = get_experiment("menu_cost")
    assert "menu_cost" in list_experiments() and exp.master_seed == 71 and "menu cost" in exp.doc
    assert validate_params(exp.defaults, exp.param_schema) == exp.defaults
    assert set(exp.defaults) == set(exp.param_schema)
    assert exp.defaults["menu_costs"] == list(COSTS) and exp.defaults["inflations"] == list(INFLS)
    assert exp.defaults["shocks"] == list(CFG.shocks) and exp.defaults["horizon"] == CFG.horizon


def test_shipped_small_config_equals_resolved_preset():
    raw, cfg = load_config(SMALL), resolve_config(SMALL)
    preset = resolve_config({"experiment": "menu_cost", "small": True})
    assert cfg.experiment == "menu_cost" and cfg.small and cfg.output is None
    assert raw["params"] == cfg.params == preset.params
    assert {k: cfg.params[k] for k in PRESET} == PRESET


def test_ledger_invariants_and_accounting_in_every_cell(grid):
    assert len(grid.rows) == len(COSTS) * len(INFLS) and not grid.errors
    for r in grid.rows:
        assert r["violations"] == 0 and r["seller_steps"] == CFG.n_sellers * CFG.horizon
        assert r["menu_cost_paid"] == r["adjustments"] * r[COST]
        assert r["seller_net_pnl"] == (r["seller_steps"] * CFG.base_profit
                                       - r["misalignment_loss"] - r["menu_cost_paid"])
        if r[COST] == 0:
            assert r["band"] == 0 and r["welfare_change_vs_flexible"] == 0 and r["mean_abs_gap"] == 0
        else:
            assert r["welfare_change_vs_flexible"] < 0          # menu costs are deadweight
    for f, d in ((8, 2), (64, 1)):
        cell = build(CFG, {COST: f, INFL: d}, 3).run()
        for m in (cell.market, cell.twin):
            m.ledger.check()
            assert sum(a.cash for a in m.ledger) == 0
            assert sum(p.net for p in menu_pnl(m).values()) == sum(m.ledger[i].cash for i in m.sellers)


def test_frequency_monotone_in_menu_cost_and_inflation(grid):
    for d in INFLS:
        xs = [_cell(grid, f, d)["adjustment_frequency"] for f in COSTS]
        assert xs == sorted(xs, reverse=True), (d, xs)
    for f in COSTS:
        xs = [_cell(grid, f, d)["adjustment_frequency"] for d in INFLS]
        assert xs == sorted(xs), (f, xs)


def test_mean_abs_gap_non_decreasing_in_menu_cost(grid):
    for d in INFLS:
        xs = [_cell(grid, f, d)["mean_abs_gap"] for f in COSTS]
        assert xs == sorted(xs), (d, xs)
    big = [r for r in grid.rows if r[COST] == max(COSTS)]
    assert all(r["mean_adjustment_size"] > r[INFL] for r in big)      # lumpy adjustment


def test_findings_and_reproducibility_from_master_seed():
    small = MenuCostConfig(n_sellers=2, horizon=80)
    a, b = (run_menu_cost(small, (0, 8, 32), (1, 2), master_seed=MASTER) for _ in range(2))
    assert a.rows == b.rows and a.findings == b.findings
    assert {"bisected_threshold", "analytic_threshold", "threshold_band", "max_band"} <= set(a.findings)
    other = run_menu_cost(small, (0, 8, 32), (1, 2), master_seed=MASTER + 1)
    assert [r["seed"] for r in other.rows] != [r["seed"] for r in a.rows]


def test_serial_and_parallel_rows_identical():
    ser = run_menu_grid(CFG, (0, 8, 32), (1, 4), master_seed=MASTER)
    par = run_menu_grid(CFG, (0, 8, 32), (1, 4), master_seed=MASTER, n_workers=2)
    assert ser.rows == par.rows


def test_run_from_config_rows_equal_direct_call():
    res = run_from_config(SMALL)
    raw = json.loads(SMALL.read_text())
    p = raw["params"]
    cfg = MenuCostConfig(**{k: tuple(v) if k == "shocks" else v for k, v in p.items()
                            if k in MenuCostConfig.__dataclass_fields__})
    direct = run_menu_cost(cfg, p["menu_costs"], p["inflations"], probe_inflation=p["probe_inflation"],
                           rigidity_bps=p["rigidity_bps"], master_seed=raw["master_seed"])
    assert res.rows == direct.rows and res.findings == direct.findings
    assert res.config["experiment"] == "menu_cost"
    assert res.findings["bisected_threshold"] == res.findings["analytic_threshold"]
    assert all(r["adjustment_frequency"] == r["analytic_frequency"] for r in res.rows)   # zero shocks
