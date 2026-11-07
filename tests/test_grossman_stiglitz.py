"""fl-053: Grossman-Stiglitz costly-information experiment (experiments/grossman_stiglitz.py)."""
from pathlib import Path

import pytest

from frictionlab.config.schema import validate_params
from frictionlab.costs import expected_edge
from frictionlab.experiments import (
    ExperimentResult, GrossmanStiglitzConfig, GSMaker, analytic_collapse_cost,
    find_information_collapse, get_experiment, gs_equilibrium_informed_share, gs_population,
    run_from_config, run_grossman_stiglitz, run_gs_grid,
)
from frictionlab.experiments.grossman_stiglitz import COST, NOISE

SMALL = Path(__file__).resolve().parents[1] / "examples" / "configs" / "grossman_stiglitz_small.json"
CFG = GrossmanStiglitzConfig()
COSTS, NOISES = [0, 1, 2, 4, 6, 8, 10], [1.0, 4.0, 16.0]
RESIDUALS = ("sink_residual", "pnl_residual", "violations")


@pytest.fixture(scope="module")
def res() -> ExperimentResult:
    return run_gs_grid(CFG, COSTS, NOISES, master_seed=53)


def col(res, key, *, cost=None, noise=None):
    return [r[key] for r in res.rows if cost in (None, r[COST]) and noise in (None, r[NOISE])]


def nonincreasing(xs):
    return all(b <= a for a, b in zip(xs, xs[1:]))


def test_registered_defaults_validate_against_schema():
    exp = get_experiment("grossman_stiglitz")
    assert set(exp.defaults) == set(exp.param_schema)
    assert validate_params(exp.defaults, exp.param_schema) == exp.defaults
    assert set(exp.resolve_params(small=True)) == set(exp.param_schema)
    assert exp.master_seed == 53 and exp.defaults["accuracy_bps"] == CFG.accuracy_bps


def test_result_shape_and_headline_findings(res):
    assert res.name == "grossman_stiglitz" and res.params == {COST: COSTS, NOISE: NOISES}
    assert len(res.rows) == len(COSTS) * len(NOISES) and not res.errors
    for k in ("buyer_fraction", "quoted_spread", "efficiency_error", "informed_gain_per_buyer",
              "buyer_net_pnl", "non_buyer_net_pnl", "sink_revenue"):
        assert k in res.metric_names()
    f = res.findings
    assert f["analytic_collapse_cost"] == analytic_collapse_cost(CFG) == 8
    assert f["cheapest_buyer_fraction"] > 0.5 and f["costliest_buyer_fraction"] == 0.0


def test_ledger_invariants_hold_in_every_cell(res):
    for r in res.rows:
        assert all(r[k] == 0 for k in RESIDUALS), {k: r[k] for k in RESIDUALS}
        assert 0 <= r["buyer_fraction"] <= 1 and r["quoted_spread"] >= 0
        assert r["sink_revenue"] == r[COST] * r["signals"]
        assert r["n_buyers"] or (r["signals"] == 0 and r["buyer_net_pnl"] == 0)


def test_buyer_fraction_monotone_in_cost_and_noise(res):
    for n in NOISES:
        assert nonincreasing(col(res, "buyer_fraction", noise=n)), n
    for c in COSTS:
        assert nonincreasing(col(res, "buyer_fraction", cost=c)[::-1]), c
    assert col(res, "buyer_fraction", cost=10) == [0.0] * len(NOISES)


def test_price_efficiency_error_non_decreasing_in_cost(res):
    for n in NOISES:
        err = col(res, "efficiency_error", noise=n)
        assert nonincreasing(err[::-1]), (n, err)
        assert err[0] < 1 and err[-1] == pytest.approx(CFG.jump)   # collapse: mid stays at prior


def test_analytic_share_non_increasing_and_zero_past_full_accuracy_edge():
    for a in (5_000, 7_000, 9_000, 10_000):
        for jump in (4, 10, 25):
            full = expected_edge(a, jump, 0)
            shares = [gs_equilibrium_informed_share(c, a, jump) for c in range(0, full + 4)]
            assert nonincreasing(shares) and all(0 <= s <= 1 for s in shares)
            assert all(s == 0 for s in shares[max(full, 0) + 1:])
    assert gs_equilibrium_informed_share(0, 10_000, 10) == 1.0
    assert gs_equilibrium_informed_share(1, 5_000, 10) == 0.0


def test_gm_maker_quotes_widen_with_informed_share_and_learn():
    mm = GSMaker(1, CFG, [], 10.0)
    assert mm.informed_share() == 0.0 and mm.quote(0.0) == (99, 101)
    (b0, a0), (b1, a1) = mm.quote(0.2), mm.quote(0.6)
    assert a1 - b1 >= a0 - b0 > 0
    mm.update(True, 0.5)
    assert mm.p > 0.5 and sum(mm.quote(0.5)) > 2 * CFG.center


def test_replicator_buyer_share_falls_with_cost():
    lo, hi = (gs_population(CFG, c, 4.0, master_seed=53) for c in (0, 9))
    share = lambda eq: (eq.fixed_point if eq.fixed_point is not None else eq.shares[-1])[0]
    assert share(hi) <= share(lo) and share(lo) > 0.5


def test_bisected_collapse_holds_at_level_and_fails_at_level_minus_one():
    col_ = find_information_collapse(CFG, 4.0, floor=0.05, master_seed=53)
    t = col_.threshold
    assert t is not None and 0 < t <= col_.analytic
    frac = run_gs_grid(CFG, [t - 1, t], [4.0], master_seed=53).column("buyer_fraction")
    assert frac[1] < col_.floor <= frac[0]
    assert col_.probes[t] == frac[1] and col_.probes[t - 1] == frac[0]
    with pytest.raises(ValueError):
        find_information_collapse(CFG, floor=0.0)


def test_full_run_packages_population_and_collapse():
    out = run_grossman_stiglitz(CFG, [0, 4, 8], [1.0, 4.0], population_costs=(0, 9), master_seed=53)
    pop = out.findings["population"]
    assert [p["signal_cost"] for p in pop] == [0, 9] and pop[1]["buyer_share"] <= pop[0]["buyer_share"]
    assert pop[0]["analytic_share"] >= pop[1]["analytic_share"] == 0.0
    assert out.findings["collapse_noise_intensity"] == 4.0
    assert out.findings["collapse_threshold"] <= out.findings["collapse_analytic"]


def test_run_from_config_matches_direct_call():
    res = run_from_config(SMALL)
    p = res.config["params"]
    kw = {k: p[k] for k in ("center", "jump", "accuracy_bps", "n_acquirers", "acquirer_rate",
                            "n_noise", "noise_buy_prob", "maker_rate", "quote_size", "horizon")}
    direct = run_grossman_stiglitz(GrossmanStiglitzConfig(**kw), p["signal_costs"],
                                   p["noise_intensities"], population_costs=p["population_costs"],
                                   generations=p["generations"], collapse_floor=p["collapse_floor"],
                                   replicates=res.config["replicates"],
                                   master_seed=res.config["master_seed"])
    assert res.rows == direct.rows and res.findings == direct.findings


def test_serial_and_parallel_rows_are_identical():
    kw = dict(signal_costs=[0, 4, 8], noise_intensities=[1.0, 16.0], replicates=2, master_seed=9)
    assert run_gs_grid(CFG, **kw).rows == run_gs_grid(CFG, n_workers=2, **kw).rows


def test_reproducible_from_master_seed(res):
    again = run_gs_grid(CFG, COSTS, NOISES, master_seed=53)
    assert again.rows == res.rows and again.findings == res.findings
    other = run_gs_grid(CFG, COSTS[:1], NOISES[:1], master_seed=54)
    assert other.rows[0]["seed"] != res.rows[0]["seed"]
