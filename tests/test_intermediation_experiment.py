"""fl-050: intermediation experiment (experiments/intermediation.py)."""
from fractions import Fraction
from pathlib import Path

import pytest

from frictionlab.analysis.welfare import competitive_equilibrium
from frictionlab.config.schema import validate_params
from frictionlab.experiments import (
    ExperimentResult, IntermediationConfig, find_intermediation_threshold, get_experiment,
    intermediation_bound, middleman_indifference_spread, run_from_config, run_intermediation,
)
from frictionlab.experiments.intermediation import COST, SPREAD, market

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "examples" / "configs" / "intermediation_small.json"
CFG = IntermediationConfig()
COSTS, SPREADS = [0, 1, 2, 4, 8, 16, 32], [0, 2, 4, 8, 16, 24]
RESIDUALS = ("sink_residual", "pnl_residual", "position_residual", "violations")


@pytest.fixture(scope="module")
def res() -> ExperimentResult:
    return run_intermediation(CFG, COSTS, SPREADS, replicates=2, master_seed=50)


def cell(res, c, h, k, rep=0):
    return [r[k] for r in res.rows if r[COST] == c and r[SPREAD] == h][rep]


def test_registered_defaults_validate_against_schema():
    exp = get_experiment("intermediation")
    assert set(exp.defaults) == set(exp.param_schema)
    assert validate_params(exp.defaults, exp.param_schema) == exp.defaults
    assert set(exp.resolve_params(small=True)) == set(exp.param_schema)
    assert exp.master_seed == 50 and exp.small["search_costs"] == [0, 2, 8, 32]


def test_result_shape_and_headline_findings(res):
    assert res.name == "intermediation" and res.params == {COST: COSTS, SPREAD: SPREADS}
    assert len(res.rows) == len(COSTS) * len(SPREADS) * 2 and not res.errors
    f, ce = res.findings, competitive_equilibrium(CFG.values, CFG.costs)
    assert f["ce_quantity"] == ce.quantity and f["ce_max_surplus"] == ce.max_surplus
    assert f["mid"] == CFG.mid and f["analytic_bound_widest"] == intermediation_bound(CFG, 24)
    for k in ("intermediated_share", "volume", "mean_time_to_match", "surplus_share",
              "mm_net_pnl", "mm_spread_capture", "mm_holding_cost", "mm_decay_loss",
              "search_cost_paid", "surplus_change"):
        assert k in res.metric_names()


def test_ledger_invariants_hold_in_every_cell(res):
    for r in res.rows:
        assert all(r[k] == 0 for k in RESIDUALS), {k: r[k] for k in RESIDUALS}
        assert r["mm_net_pnl"] == r["mm_spread_capture"] - r["mm_holding_cost"] - r["mm_decay_loss"]
        assert 0 <= r["intermediated_share"] <= 1 and 0 <= r["surplus_share"] <= 1
        assert r["surplus_change"] == r["surplus"] - r["direct_surplus"]
        assert r["search_cost_paid"] >= 0 and (r[COST] or r["search_cost_paid"] == 0)


def test_share_monotone_in_search_cost_and_half_spread(res):
    for rep in (0, 1):
        for h in SPREADS:
            col = [cell(res, c, h, "intermediated_share", rep) for c in COSTS]
            assert all(b >= a for a, b in zip(col, col[1:])), (h, col)
        for c in COSTS:
            col = [cell(res, c, h, "intermediated_share", rep) for h in SPREADS]
            assert all(b <= a for a, b in zip(col, col[1:])), (c, col)
    assert cell(res, 0, 4, "intermediated_share") == 0 < cell(res, 32, 4, "intermediated_share")


def test_no_middleman_twin_matches_a_plain_market(res):
    seed = res.rows[0]["seed"]
    for c in (0, 8, 32):
        m = market(CFG, c, None, seed)
        m.run(CFG.until)
        assert m.middleman is None and m.ledger.violations() == []
        assert len(m.trades) == cell(res, c, 4, "direct_volume")
        assert m.search_costs == 2 * c * len(m.meetings)


def test_indifference_spread_zero_at_zero_cost_and_non_decreasing():
    for rd, rm in ((0.5, 2.0), (1.0, 1.0), (1.0, 4.0), (2.0, 1.0)):
        col = [middleman_indifference_spread(c, rd, rm) for c in range(0, 80)]
        assert col[0] == 0 and all(b >= a for a, b in zip(col, col[1:])) and min(col) >= 0
    # c * (1 / 0.5 - 1 / 2) = 1.5 c, floored; more meetings per match save more
    assert [middleman_indifference_spread(c, 0.5, 2.0) for c in (1, 2, 3, 8)] == [1, 3, 4, 12]
    assert middleman_indifference_spread(8, 0.5, 2.0, Fraction(3, 2)) == 20
    assert middleman_indifference_spread(10, 1.0, 1.0) == 0     # equal rates: never worth a spread
    with pytest.raises(ValueError):
        middleman_indifference_spread(1, 0.0, 1.0)


@pytest.mark.parametrize("half_spread", [2, 4, 8])
def test_bisected_threshold_holds_at_level_and_fails_at_level_minus_one(half_spread):
    t = find_intermediation_threshold(CFG, half_spread, fraction=0.25, master_seed=50)
    assert t.threshold is not None and t.threshold > 0 and t.within_bound
    share = run_intermediation(CFG, [t.threshold - 1, t.threshold], [half_spread],
                               master_seed=50).column("intermediated_share")
    assert share[0] <= t.fraction < share[1]
    assert t.probes[t.threshold] == share[1] and t.probes[t.threshold - 1] == share[0]
    bound = t.analytic_bound
    assert middleman_indifference_spread(bound, CFG.meeting_rate, CFG.middleman_meeting_rate) \
        >= half_spread > middleman_indifference_spread(bound - 1, CFG.meeting_rate,
                                                       CFG.middleman_meeting_rate)


def test_run_from_config_matches_direct_call():
    res = run_from_config(SMALL)
    p = res.config["params"]
    direct = run_intermediation(IntermediationConfig(until=p["until"]), p["search_costs"],
                                p["half_spreads"], replicates=res.config["replicates"],
                                master_seed=res.config["master_seed"])
    assert res.rows == direct.rows and res.findings == direct.findings


def test_serial_and_parallel_rows_are_identical():
    kw = dict(search_costs=[0, 4, 32], half_spreads=[0, 8], replicates=2, master_seed=9)
    assert run_intermediation(CFG, **kw).rows == run_intermediation(CFG, n_workers=2, **kw).rows


def test_reproducible_from_master_seed(res):
    again = run_intermediation(CFG, COSTS, SPREADS, replicates=2, master_seed=50)
    assert again.rows == res.rows and again.findings == res.findings
    other = run_intermediation(CFG, COSTS[:1], SPREADS[:1], master_seed=51)
    assert other.rows[0]["seed"] != res.rows[0]["seed"]


def test_validation():
    with pytest.raises(ValueError):
        find_intermediation_threshold(CFG, 4, fraction=1.0)
    assert intermediation_bound(CFG, 0) == 0
    assert intermediation_bound(IntermediationConfig(middleman_meeting_rate=0.5), 4) is None
