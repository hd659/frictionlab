"""fl-047: search-cost and perishable-value experiment (experiments/search_cost.py)."""
from pathlib import Path

import pytest

from frictionlab.analysis.welfare import competitive_equilibrium
from frictionlab.config.schema import validate_params
from frictionlab.core import Side
from frictionlab.experiments import (
    ExperimentResult, SearchCostConfig, find_search_collapse, get_experiment, reservation_price,
    run_from_config, run_search_cost, search_cost_bound,
)
from frictionlab.experiments.search_cost import COST, DECAY

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "examples" / "configs" / "search_cost_small.json"
CFG = SearchCostConfig()
COSTS, DECAYS = [0, 5, 10, 20, 30, 40, 52], [0, 1, 2, 4]
RESIDUALS = ("sink_residual", "position_residual", "violations")


@pytest.fixture(scope="module")
def res() -> ExperimentResult:
    return run_search_cost(CFG, COSTS, DECAYS, replicates=2, master_seed=47)


def agg_at(res, key, level):
    return [a for a in res.aggregate() if a[key] == level]


def test_registered_defaults_validate_against_schema():
    exp = get_experiment("search_cost")
    assert set(exp.defaults) == set(exp.param_schema)
    assert validate_params(exp.defaults, exp.param_schema) == exp.defaults
    assert set(exp.resolve_params(small=True)) == set(exp.param_schema)
    assert exp.master_seed == 47


def test_result_shape_and_headline_findings(res):
    assert res.name == "search_cost" and res.params == {COST: COSTS, DECAY: DECAYS}
    assert len(res.rows) == len(COSTS) * len(DECAYS) * 2 and not res.errors
    f, ce = res.findings, competitive_equilibrium(CFG.values, CFG.costs)
    assert f["ce_quantity"] == ce.quantity and f["ce_max_surplus"] == ce.max_surplus
    assert f["baseline_volume"] == ce.quantity and f["baseline_surplus_share"] == 1.0
    assert f["costliest_volume"] == 0 and f["analytic_bound"] == search_cost_bound(CFG)
    for k in ("volume", "mean_time_to_match", "surplus_share", "search_cost_paid",
              "decay_loss", "seller_share"):
        assert k in res.metric_names()


def test_ledger_invariants_hold_in_every_cell(res):
    for r in res.rows:
        assert all(r[k] == 0 for k in RESIDUALS), {k: r[k] for k in RESIDUALS}
        assert 0 <= r["seller_share"] <= 1 and r["decay_loss"] >= 0
        assert r["search_cost_paid"] == 2 * r[COST] * r["meetings"]
        assert r["decay_loss"] == 0 or r[DECAY] > 0


def test_volume_and_surplus_share_non_increasing_in_search_cost(res):
    for d in DECAYS:
        rows = agg_at(res, DECAY, d)
        for k in ("volume_mean", "surplus_share_mean"):
            col = [a[k] for a in rows]
            assert all(b <= a for a, b in zip(col, col[1:])), (d, k, col)
        assert rows[0]["volume_mean"] > rows[-1]["volume_mean"] == 0


def test_time_to_match_non_increasing_in_decay(res):
    for c in COSTS:
        t = [a["mean_time_to_match_mean"] for a in agg_at(res, COST, c) if a[DECAY] > 0]
        assert all(b <= a for a, b in zip(t, t[1:])), (c, t)


def test_reservation_price_non_increasing_in_cost_and_value_at_zero():
    for v in (40, 70, 120):
        bids = [reservation_price(v, c, 1.0, (20, 70)) for c in range(0, 60)]
        asks = [reservation_price(v, c, 1.0, (70, 120), Side.SELL) for c in range(0, 60)]
        assert bids[0] == asks[0] == v
        assert all(b <= a for a, b in zip(bids, bids[1:]))
        assert all(b >= a for a, b in zip(asks, asks[1:]))   # seller asks mirror the bids
    # exact boundary: (100 - p) * m(p) >= k * n with n = 51, k = 6 -> p = 94 (m = 51)
    assert reservation_price(100, 6, 1.0, (30, 80)) == 94
    assert reservation_price(100, 3, 0.5, (30, 80)) == 94      # k = cost / rate
    assert reservation_price(25, 1, 1.0, (30, 80)) == 29       # no seller ever acceptable
    with pytest.raises(ValueError):
        reservation_price(10, 1, 0.0, (1, 5))


def test_bisected_threshold_satisfies_collapse_and_lies_within_bound():
    t = find_search_collapse(CFG, fraction=0.5, master_seed=47)
    assert t.threshold is not None and t.within_bound and t.target == 0.5 * t.baseline_volume
    below, at = run_search_cost(CFG, [t.threshold - 1, t.threshold], [0],
                                master_seed=47).column("volume")
    assert at < t.target <= below
    assert t.probes[t.threshold] == at and t.probes[t.threshold - 1] == below
    bound = search_cost_bound(CFG)
    assert run_search_cost(CFG, [bound, bound + 10], [0, 2]).column("volume") == [0, 0, 0, 0]
    assert run_search_cost(CFG, [bound - 1], [0]).column("volume")[0] > 0


def test_run_from_config_matches_direct_call():
    res = run_from_config(SMALL)
    p = res.config["params"]
    direct = run_search_cost(SearchCostConfig(until=p["until"]), p["search_costs"], p["decays"],
                             replicates=res.config["replicates"],
                             master_seed=res.config["master_seed"])
    assert res.rows == direct.rows and res.findings == direct.findings


def test_serial_and_parallel_rows_are_identical():
    kw = dict(search_costs=[0, 20, 52], decays=[0, 2], replicates=2, master_seed=9)
    assert run_search_cost(CFG, **kw).rows == run_search_cost(CFG, n_workers=2, **kw).rows


def test_reproducible_from_master_seed(res):
    again = run_search_cost(CFG, COSTS, DECAYS, replicates=2, master_seed=47)
    assert again.rows == res.rows and again.findings == res.findings
    other = run_search_cost(CFG, COSTS[:1], DECAYS[:1], master_seed=48)
    assert other.rows[0]["seed"] != res.rows[0]["seed"]


def test_validation():
    with pytest.raises(ValueError):
        find_search_collapse(CFG, fraction=0)
