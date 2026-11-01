"""fl-034: funding-rate arbitrage under gas shocks (experiments/funding_arb.py)."""
from fractions import Fraction

import pytest

from frictionlab.experiments import (
    ExperimentError, ExperimentResult, FundingArbConfig, entry_threshold_gas, find_gas_threshold, funding_arb_bound,
    run_funding_arb,
)
from frictionlab.experiments.funding_arb import LEVEL, PROB

CFG = FundingArbConfig()
BOUND = funding_arb_bound(CFG)
LEVELS = [1, 5, 11, 58, BOUND + 1]
PROBS = [0.0, 0.3, 0.6]
RESIDUALS = ("recon_residual", "fee_residual", "gas_residual", "funding_residual",
             "pool_residual", "oi_residual", "violations", "net_delta")


@pytest.fixture(scope="module")
def res() -> ExperimentResult:
    return run_funding_arb(CFG, LEVELS, PROBS, master_seed=34)


def rows_at(res, p):
    return sorted((r for r in res.rows if r[PROB] == p), key=lambda r: r[LEVEL])


def test_entry_threshold_gas_is_exact_on_the_boundary():
    # slack = 10 * 4 - 0 = 40 -> 4 g < 40 -> g <= 9
    assert entry_threshold_gas(10, 4, 0, 0, 0) == 9
    assert entry_threshold_gas(Fraction(41, 4), 4, 0, 0, 0) == 10    # slack 41
    assert entry_threshold_gas(10, 4, 20, 10, 10) == -1               # slack 0: never
    assert entry_threshold_gas(10, 4, 30, 0, 0) == 2                  # slack 10 -> g < 2.5
    for g in range(-1, 20):
        slack = 10 * 4 - 3 - Fraction(1, 2) - 1
        assert (entry_threshold_gas(10, 4, 3, Fraction(1, 2), 1) >= g) == (g < 0 or slack - 4 * g > 0)


def test_result_is_one_experiment_with_headline_findings(res):
    assert res.name == "funding_arb" and res.params == {LEVEL: LEVELS, PROB: PROBS}
    assert len(res.rows) == len(LEVELS) * len(PROBS) and not res.errors
    f = res.findings
    assert f["analytic_bound"] == BOUND
    assert f["baseline_entries"] > f["shock_entries"]
    assert f["shock_mean_abs_premium"] > f["baseline_mean_abs_premium"]
    for k in ("entries", "mean_holding", "gross_funding", "perp_fee", "amm_cost", "gas",
              "decision_cost", "net_pnl", "mean_abs_premium"):
        assert k in res.metric_names()


def test_ledger_invariants_and_tick_reconciliation_hold_in_every_cell(res):
    for r in res.rows:
        assert all(r[k] == 0 for k in RESIDUALS), {k: r[k] for k in RESIDUALS}


def test_zero_shock_probability_reproduces_constant_gas_baseline(res):
    base = rows_at(res, 0.0)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("cell", LEVEL)}
    assert all(strip(r) == strip(base[0]) for r in base)
    assert base[0]["entries"] > 0 and base[0]["entry_gas_max"] == CFG.calm_gas


def test_entries_and_net_pnl_fall_and_premium_rises_with_shock_level(res):
    for p in PROBS[1:]:
        rows = rows_at(res, p)
        for a, b in zip(rows, rows[1:]):
            assert b["entries"] <= a["entries"]
            assert b["net_pnl"] <= a["net_pnl"]
            assert b["mean_abs_premium"] >= a["mean_abs_premium"]
        assert rows[-1]["entries"] < rows[0]["entries"]
        assert rows[-1]["mean_abs_premium"] > rows[0]["mean_abs_premium"]


def test_no_entry_ever_happens_above_the_analytic_threshold(res):
    assert all(r["entry_gas_max"] <= BOUND for r in res.rows)
    for d in (1, -1):    # gas permanently above the bound, both premium signs
        cfg = FundingArbConfig(direction=d, calm_gas=BOUND + 1)
        always = run_funding_arb(cfg, [BOUND + 1, BOUND + 40], [0.0, 1.0], master_seed=34)
        assert all(r["entries"] == 0 and r["gross_funding"] == 0 for r in always.rows)
    mirror = run_funding_arb(FundingArbConfig(direction=-1), [CFG.calm_gas], [0.0], master_seed=34)
    assert mirror.rows[0]["entries"] > 0


def test_bisection_threshold_is_exact_and_within_the_analytic_bound():
    t = find_gas_threshold(CFG, shock_probability=0.6, fraction=0.5, master_seed=34)
    assert t.threshold is not None and t.within_bound
    assert t.target == 0.5 * t.baseline_entries
    check = run_funding_arb(CFG, [t.threshold - 1, t.threshold], [0.6], master_seed=34)
    below, at = check.column("entries")
    assert at < t.target <= below
    assert t.probes[t.threshold] == at and t.probes[t.threshold - 1] == below


def test_serial_and_parallel_rows_are_identical():
    kw = dict(levels=[1, 58], probs=[0.0, 0.6], replicates=2, master_seed=7)
    assert run_funding_arb(CFG, **kw).rows == run_funding_arb(CFG, n_workers=2, **kw).rows


def test_reproducible_from_master_seed(res):
    again = run_funding_arb(CFG, LEVELS, PROBS, master_seed=34)
    assert again.rows == res.rows and again.findings == res.findings
    other = run_funding_arb(CFG, LEVELS[:1], PROBS[:1], master_seed=35)
    assert other.rows[0]["seed"] != res.rows[0]["seed"]


def test_validation():
    with pytest.raises(ValueError):
        find_gas_threshold(CFG, fraction=0)
    with pytest.raises(ExperimentError, match="below calm gas"):
        run_funding_arb(FundingArbConfig(calm_gas=5), [1], [0.0])
