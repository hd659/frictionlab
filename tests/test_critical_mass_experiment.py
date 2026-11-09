"""fl-065: critical-mass experiment over adoption_cost x seed_members on the fl-064 platform."""
import json
from pathlib import Path

import pytest

from frictionlab.config.schema import load_config, validate_params
from frictionlab.experiments import (
    CriticalMassConfig, get_experiment, list_experiments, resolve_config, run_critical_mass,
    run_from_config, run_platform_grid, tipping_fixed_points,
)
from frictionlab.experiments.critical_mass import COST, COSTS, SEED, SEEDS, build, side_values

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "examples" / "configs" / "critical_mass_small.json"
CFG, MASTER = CriticalMassConfig(), 65


@pytest.fixture(scope="module")
def grid():
    return run_platform_grid(CFG, master_seed=MASTER)


def _cell(res, c, k):
    (row,) = [r for r in res.rows if r[COST] == c and r[SEED] == k]
    return row


def test_registered_defaults_validate_against_schema():
    exp = get_experiment("critical_mass")
    assert "critical_mass" in list_experiments() and exp.master_seed == 65 and "critical mass" in exp.doc
    assert validate_params(exp.defaults, exp.param_schema) == exp.defaults
    assert set(exp.defaults) == set(exp.param_schema)
    assert exp.defaults["adoption_costs"] == list(COSTS) and exp.defaults["seed_members"] == list(SEEDS)
    assert tuple(exp.defaults["standalone_a"]) == CFG.standalone_a and exp.defaults["n_b"] == CFG.n_b


def test_shipped_small_config_equals_resolved_preset():
    raw, cfg = load_config(SMALL), resolve_config(SMALL)
    preset = resolve_config({"experiment": "critical_mass", "small": True})
    assert cfg.experiment == "critical_mass" and cfg.small and cfg.output is None
    assert raw["params"] == cfg.params == preset.params
    assert list(cfg.params) == list(get_experiment("critical_mass").param_schema)


def test_ledger_invariants_and_accounting_in_every_cell(grid):
    assert len(grid.rows) == len(COSTS) * len(SEEDS) and not grid.errors
    for r in grid.rows:
        assert r["violations"] == 0
        assert r["platform_pnl"] == r["fee_revenue"] - r["subsidy_outlay"]
        assert r["members"] == r["members_a"] + r["members_b"] and 0 <= r["share"] <= 1
        assert r["seed_subsidy"] <= r["subsidy_outlay"]
        if r[COST] == 0:
            assert r["adoption_paid"] == 0 and r["welfare_change_vs_free_adoption"] == 0
        if r[SEED] == 0:
            assert r["seed_subsidy"] == 0
    for c, k in ((8, 4), (40, 10)):
        cell = build(CFG, {COST: c, SEED: k}, 3).run()
        for m in (cell.market, cell.twin):
            m.ledger.check()
            assert sum(a.cash for a in m.ledger) == 0


def test_membership_monotone_in_seed_and_adoption_cost(grid):
    for c in COSTS:
        ms = [_cell(grid, c, k)["members"] for k in SEEDS]
        assert ms == sorted(ms), (c, ms)
    for k in SEEDS:
        ms = [_cell(grid, c, k)["members"] for c in COSTS]
        assert ms == sorted(ms, reverse=True), (k, ms)
    assert grid.findings["cold_start_traps"] > 0 and grid.findings["max_members"] == CFG.n_a + CFG.n_b


def test_simulated_fixed_point_is_an_analytic_fixed_point(grid):
    hit = 0
    for r in grid.rows:
        if r["outcome"] == "fixed_point":
            hit += 1
            assert r["on_fixed_point"], r
    assert hit > 0
    for c, k in ((0, 0), (16, 6), (40, 14)):
        cell = build(CFG, {COST: c, SEED: k}, 9).run()
        m = cell.market
        pts = tipping_fixed_points(*side_values(m), *m.benefit, *m.fee, c)
        assert (m.count(0), m.count(1)) in pts and pts == sorted(pts, key=lambda p: (sum(p), p))


def test_findings_and_reproducibility_from_master_seed():
    small = CriticalMassConfig(horizon=12)
    a, b = (run_critical_mass(small, (0, 16), (0, 3), tipping_fraction=0.5, master_seed=MASTER)
            for _ in range(2))
    assert a.rows == b.rows and a.findings == b.findings
    assert {"tipping_seed", "analytic_critical_mass", "cold_start_traps"} <= set(a.findings)
    other = run_critical_mass(small, (0, 16), (0, 3), tipping_fraction=0.5, master_seed=MASTER + 1)
    assert [r["seed"] for r in other.rows] != [r["seed"] for r in a.rows]


def test_serial_and_parallel_rows_identical():
    ser = run_platform_grid(CFG, (0, 8, 40), (0, 6), master_seed=MASTER)
    par = run_platform_grid(CFG, (0, 8, 40), (0, 6), master_seed=MASTER, n_workers=2)
    assert ser.rows == par.rows


def test_run_from_config_rows_equal_direct_call():
    res = run_from_config(SMALL)
    raw = json.loads(SMALL.read_text())
    p = raw["params"]
    cfg = CriticalMassConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in p.items()
                                if k in CriticalMassConfig.__dataclass_fields__})
    direct = run_critical_mass(cfg, p["adoption_costs"], p["seed_members"], tipping_cost=p["tipping_cost"],
                               tipping_fraction=p["tipping_fraction"], master_seed=raw["master_seed"])
    assert res.rows == direct.rows and res.findings == direct.findings
    assert res.config["experiment"] == "critical_mass"
    assert res.findings["tipping_seed"] == res.findings["analytic_critical_mass"]
