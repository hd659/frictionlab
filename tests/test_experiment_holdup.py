"""fl-074: hold-up experiment over contracting_cost x seller_weight_bps on the fl-073 market."""
import json
from pathlib import Path

import pytest

from frictionlab.analysis.pnl import holdup_pnl
from frictionlab.config.schema import load_config, validate_params
from frictionlab.costs import CONTRACT_SINK_ID
from frictionlab.experiments import (
    HoldupConfig, get_experiment, list_experiments, resolve_config, run_from_config, run_holdup,
    run_holdup_grid,
)
from frictionlab.experiments.holdup import COST, COSTS, SMALL as PRESET, WEIGHT, WEIGHTS, build
from frictionlab.markets import INVEST_SINK_ID

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "examples" / "configs" / "holdup_small.json"
CFG, MASTER = HoldupConfig(), 74


@pytest.fixture(scope="module")
def grid():
    return run_holdup_grid(CFG, master_seed=MASTER)


def _cell(res, c, w):
    (row,) = [r for r in res.rows if r[COST] == c and r[WEIGHT] == w]
    return row


def test_registered_defaults_validate_against_schema():
    exp = get_experiment("holdup")
    assert "holdup" in list_experiments() and exp.master_seed == 74 and "hold-up" in exp.doc
    assert validate_params(exp.defaults, exp.param_schema) == exp.defaults
    assert set(exp.defaults) == set(exp.param_schema)
    assert exp.defaults["contracting_costs"] == list(COSTS)
    assert exp.defaults["seller_weights"] == list(WEIGHTS)
    assert exp.defaults["buyer_values"] == list(CFG.buyer_values) and exp.defaults["rounds"] == CFG.rounds


def test_shipped_small_config_equals_resolved_preset():
    raw, cfg = load_config(SMALL), resolve_config(SMALL)
    preset = resolve_config({"experiment": "holdup", "small": True})
    assert cfg.experiment == "holdup" and cfg.small and cfg.output is None
    assert raw["params"] == cfg.params == preset.params
    assert {k: cfg.params[k] for k in PRESET} == PRESET


def test_ledger_invariants_and_accounting_in_every_cell(grid):
    assert len(grid.rows) == len(COSTS) * len(WEIGHTS) and not grid.errors
    n = CFG.n_pairs * CFG.rounds
    for r in grid.rows:
        assert r["violations"] == 0 and r["trades"] == n
        assert r["contracting_cost_paid"] == r["contracts"] * r[COST]
        assert r["seller_surplus"] + r["buyer_surplus"] == r["welfare"]
        assert r["max_surplus"] - r["welfare"] == pytest.approx(
            n * r["underinvestment_gap"] + r["contracting_cost_paid"])
        assert 0 <= r["mean_investment"] <= r["first_best_investment"]
        if r[COST] == 0:
            assert r["welfare_change_vs_free_contracting"] == 0
    for c, w in ((4, 5_000), (10, 7_500)):
        cell = build(CFG, {COST: c, WEIGHT: w}, 3).run()
        for m in (cell.market, cell.twin):
            m.ledger.check()
            assert sum(a.cash for a in m.ledger) == 0
            assert sum(p.net for p in holdup_pnl(m).values()) == sum(
                m.ledger[s.agent_id].cash for s in m.sellers)
            assert -m.ledger[INVEST_SINK_ID].cash <= 0
            if CONTRACT_SINK_ID in m.ledger:
                assert m.ledger[CONTRACT_SINK_ID].cash == m.contracting.revenue


def test_contracting_rate_monotone_in_cost_and_weight(grid):
    for w in WEIGHTS:
        xs = [_cell(grid, c, w)["contracting_rate"] for c in COSTS]
        assert xs == sorted(xs, reverse=True), (w, xs)
    for c in COSTS:
        xs = [_cell(grid, c, w)["contracting_rate"] for w in WEIGHTS]
        assert xs == sorted(xs, reverse=True), (c, xs)


def test_surplus_share_one_at_zero_cost_and_rate_zero_past_bound(grid):
    for w in WEIGHTS:
        free = _cell(grid, 0, w)
        assert free["surplus_share"] == 1.0 and free["underinvestment_gap"] == 0
        for c in COSTS:
            r = _cell(grid, c, w)
            if c > r["max_contracting_cost"]:
                assert r["contracting_rate"] == 0.0 and r["contracting_cost_paid"] == 0
            assert r["welfare_change_vs_free_contracting"] <= 0      # costly contracts never help
    assert _cell(grid, max(COSTS), 5_000)["surplus_share"] < 1.0     # hold-up bites


def test_findings_and_reproducibility_from_master_seed():
    small = HoldupConfig(n_pairs=2, rounds=10)
    a, b = (run_holdup(small, (0, 4, 12), (5_000, 10_000), master_seed=MASTER) for _ in range(2))
    assert a.rows == b.rows and a.findings == b.findings
    assert {"bisected_threshold", "analytic_threshold", "max_contracting_cost",
            "holdup_investment", "first_best_investment"} <= set(a.findings)
    other = run_holdup(small, (0, 4, 12), (5_000, 10_000), master_seed=MASTER + 1)
    assert [r["seed"] for r in other.rows] != [r["seed"] for r in a.rows]


def test_serial_and_parallel_rows_identical():
    ser = run_holdup_grid(CFG, (0, 4, 10), (5_000, 9_000), master_seed=MASTER)
    par = run_holdup_grid(CFG, (0, 4, 10), (5_000, 9_000), master_seed=MASTER, n_workers=2)
    assert ser.rows == par.rows


def test_run_from_config_rows_equal_direct_call():
    res = run_from_config(SMALL)
    raw = json.loads(SMALL.read_text())
    p = raw["params"]
    cfg = HoldupConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in p.items()
                          if k in HoldupConfig.__dataclass_fields__})
    direct = run_holdup(cfg, p["contracting_costs"], p["seller_weights"],
                        probe_weight=p["probe_weight"], collapse_bps=p["collapse_bps"],
                        master_seed=raw["master_seed"])
    assert res.rows == direct.rows and res.findings == direct.findings
    assert res.config["experiment"] == "holdup"
    assert res.findings["bisected_threshold"] == res.findings["max_contracting_cost"] + 1   # no noise
