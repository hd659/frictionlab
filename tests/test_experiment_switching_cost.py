"""fl-062: switching-cost experiment over mean_switching_cost x target_share."""
import pytest

from frictionlab.experiments import (
    SwitchingCostConfig, acquisition_discount, analytic_lockin_threshold, find_lockin_threshold,
    klemperer_harvest_markup, run_switching_cost, run_switching_grid,
)
from frictionlab.experiments.switching_cost import COSTS, MEAN, TARGET, TARGETS, build, cost_schedule

CFG, SEED = SwitchingCostConfig(), 62
HOMO = SwitchingCostConfig(spread=0)


@pytest.fixture(scope="module")
def grid():
    return run_switching_grid(CFG, master_seed=SEED)


def _cell(res, m, target):
    (row,) = [r for r in res.rows if r[MEAN] == m and r[TARGET] == target]
    return row


def test_ledger_invariants_and_accounting_in_every_cell(grid):
    assert len(grid.rows) == len(COSTS) * len(TARGETS) and not grid.errors
    for r in grid.rows:
        assert r["violations"] == 0
        assert r["acquisition_trades"] + r["harvest_trades"] == r["trades"] > 0
        assert r["share_1"] + r["share_2"] <= 10_000 and 0 <= r["churn_rate"] <= 1
        # lock-in is a transfer: welfare moves only by the switching cost paid
        assert r["welfare_change_vs_bertrand"] == -r["switching_paid"]
        assert r["cs_change_vs_bertrand"] == -(r["seller_profit"] + r["switching_paid"])
        assert 0 <= r["surplus_share"] <= 1


def test_markup_non_decreasing_and_churn_non_increasing_in_switching_cost(grid):
    for tgt in TARGETS:
        mk = [_cell(grid, m, tgt)["harvest_markup"] for m in COSTS]
        churn = [_cell(grid, m, tgt)["churn_rate"] for m in COSTS]
        assert mk == sorted(mk), (tgt, mk)
        assert churn == sorted(churn, reverse=True), (tgt, churn)
    assert grid.findings["max_harvest_markup"] > 0 and grid.findings["lockin_cells"] > 0


def test_zero_switching_cost_has_zero_markup_every_round(grid):
    for tgt in TARGETS:
        cell = build(CFG, {MEAN: 0, TARGET: tgt}, 7).run()
        c = CFG.marginal_cost
        assert all(p is None or p == c for r in cell.market.log for p in r.prices)
        row = _cell(grid, 0, tgt)
        assert row["mean_markup"] == row["harvest_markup"] == 0 and row["seller_profit"] == 0
        assert row["cs_change_vs_bertrand"] == 0


def test_harvest_markup_equals_klemperer_when_costs_homogeneous():
    res = run_switching_grid(HOMO, COSTS, TARGETS, master_seed=SEED)
    floor = min(v for v in HOMO.valuations if v >= HOMO.marginal_cost)
    for r in res.rows:
        if r["harvest_trades"]:
            want = klemperer_harvest_markup(r[MEAN], floor, HOMO.marginal_cost)
            assert r["harvest_markup"] == r["analytic_harvest_markup"] == want, r
    assert any(r["harvest_trades"] for r in res.rows)


def test_acquisition_price_never_above_harvest_price(grid):
    for r in grid.rows:
        if r["acquisition_trades"] and r["harvest_trades"]:
            assert r["acquisition_price"] <= r["harvest_price"], r
    for m in COSTS:
        cell = build(CFG, {MEAN: m, TARGET: 6000}, 11).run()
        for s in cell.market.sellers.values():
            acq = [p for p, ph in zip(s.prices, s.phases) if ph == "acquire"]
            har = [p for p, ph in zip(s.prices, s.phases) if ph == "harvest"]
            if acq and har:
                assert max(acq) <= min(har)


def test_closed_forms():
    assert klemperer_harvest_markup(0, 100, 40) == 0
    assert klemperer_harvest_markup(8, 100, 40) == 8
    assert klemperer_harvest_markup(99, 70, 40) == 30           # valuation cap binds
    assert klemperer_harvest_markup(5, 30, 40) == 0
    assert acquisition_discount(10, 8, 6000) == 43              # 9 * 8 * 6000 // 10000
    assert acquisition_discount(1, 8, 10_000) == 0 and acquisition_discount(5, 0, 9000) == 0
    assert cost_schedule(1, 2) == (0, 1, 2) and cost_schedule(8, 2) == (6, 8, 10)
    assert sum(cost_schedule(16, 3)) == 3 * 16


@pytest.mark.parametrize("k", [0, 4, 10])
def test_bisected_threshold_is_exact_and_matches_analytic(k):
    th = find_lockin_threshold(CFG, 6000, k, master_seed=SEED)
    markup = lambda m: run_switching_grid(CFG, [m], [6000], master_seed=SEED).rows[0]["harvest_markup"]
    assert th.analytic == analytic_lockin_threshold(CFG, k) == k + 1 + CFG.spread
    assert th.threshold == th.analytic and markup(th.threshold) > k
    if th.threshold > 0:
        assert markup(th.threshold - 1) <= k


def test_threshold_errors_when_lockin_impossible():
    with pytest.raises(ValueError, match="valuation cap"):
        find_lockin_threshold(CFG, 6000, 30, master_seed=SEED)
    with pytest.raises(ValueError, match="never reaches"):
        find_lockin_threshold(CFG, 9000, 4, master_seed=SEED)


def test_findings_and_reproducibility_from_master_seed():
    small = SwitchingCostConfig(n_consumers=16, horizon=6)
    a, b = (run_switching_cost(small, (0, 8), (6000,), master_seed=SEED) for _ in range(2))
    assert a.rows == b.rows and a.findings == b.findings
    assert a.findings["lockin_threshold"] == a.findings["analytic_lockin_threshold"]
    other = run_switching_cost(small, (0, 8), (6000,), master_seed=SEED + 1)
    assert [r["seed"] for r in other.rows] != [r["seed"] for r in a.rows]


def test_serial_and_parallel_rows_identical():
    ser = run_switching_grid(CFG, (0, 4, 16), (6000, 9000), master_seed=SEED)
    par = run_switching_grid(CFG, (0, 4, 16), (6000, 9000), master_seed=SEED, n_workers=2)
    assert ser.rows == par.rows
