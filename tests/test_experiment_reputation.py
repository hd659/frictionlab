"""fl-059: reputation experiment over rating_accuracy_bps x whitewash_cost."""
import pytest

from frictionlab.experiments import (
    ReputationConfig, find_reputation_threshold, milking_switch_round, min_deterrent_whitewash_cost,
    predicted_milking_round, reputation_qualities, run_reputation, run_reputation_grid,
)
from frictionlab.experiments.reputation import ACC, ACCURACIES, WASH, WASH_COSTS, build, opportunist_ids

CFG, SEED = ReputationConfig(), 59


@pytest.fixture(scope="module")
def grid():
    return run_reputation_grid(CFG, master_seed=SEED)


def _cell(res, acc, cost):
    (row,) = [r for r in res.rows if r[ACC] == acc and r[WASH] == cost]
    return row


def test_ledger_invariants_and_accounting_in_every_cell(grid):
    assert len(grid.rows) == len(ACCURACIES) * len(WASH_COSTS) and not grid.errors
    for r in grid.rows:
        assert r["violations"] == 0
        assert r["whitewash_paid"] == r["whitewashes"] * r[WASH]
        assert 0 <= r["cheating_rate"] <= 1 and 0 <= r["true_quality_share"] <= 1
        assert r["cheats"] == round((1 - r["true_quality_share"]) * r["trades"])
        assert 0 <= r["mean_milking_round"] <= CFG.horizon and r["surplus_share"] <= 1
        assert r["surplus_gain_vs_lemons"] == pytest.approx(r["surplus_share"] - r["lemons_surplus_share"])
    assert len({r["lemons_surplus_share"] for r in grid.rows}) == 1        # one CRN twin


def test_cheating_and_surplus_monotone_in_accuracy(grid):
    for c in WASH_COSTS:
        cheat = [_cell(grid, a, c)["cheating_rate"] for a in ACCURACIES]
        share = [_cell(grid, a, c)["surplus_share"] for a in ACCURACIES]
        assert cheat == sorted(cheat, reverse=True), (c, cheat)
        assert share == sorted(share), (c, share)


def test_whitewash_count_non_increasing_in_cost(grid):
    for a in ACCURACIES:
        w = [_cell(grid, a, c)["whitewashes"] for c in WASH_COSTS]
        assert w == sorted(w, reverse=True), (a, w)


def test_no_opportunists_reproduces_the_pooled_lemons_surplus(grid):
    honest = run_reputation_grid(ReputationConfig(opportunist_bps=0), ACCURACIES, (0, 64), master_seed=SEED)
    for r in honest.rows:
        assert r["cheats"] == r["whitewashes"] == 0 and r["true_quality_share"] == 1.0
        assert r["price_quality_gap"] > 0                    # all ids price at the pooled mean
        assert r["surplus_gain_vs_lemons"] == pytest.approx(0.0)
    # opportunists drag the fresh-id price below the pooled price: reputation pays a discount
    assert grid.findings["max_surplus_gain_vs_lemons"] == max(r["surplus_gain_vs_lemons"] for r in grid.rows)


def test_milking_switch_round_closed_form():
    assert milking_switch_round(10, 0, 5) == 10            # no saving: never cheats in-horizon
    assert milking_switch_round(10, 50, 0) == 0            # no premium: cheats at once
    assert milking_switch_round(10, 50, 10) == 5           # R * 10 < 50 <=> R <= 4
    assert milking_switch_round(10, 51, 10) == 4
    assert milking_switch_round(3, 1000, 1) == 0


@pytest.mark.parametrize("cost", [0, 64])
def test_simulated_milking_round_equals_analytic_at_perfect_accuracy(cost):
    cell = build(CFG, {ACC: 10_000, WASH: cost}, 123).run()
    m = cell.market
    opp = opportunist_ids(CFG.n_sellers, CFG.opportunist_bps)
    assert len(opp) == CFG.n_sellers * CFG.opportunist_bps // 10_000
    for sid in opp:
        s = m.sellers[sid]
        got = m.rounds if s.milking_round is None else s.milking_round
        assert got == predicted_milking_round(m, sid), sid


def test_deterrent_bound_stops_every_whitewash():
    qs = reputation_qualities(CFG, SEED)
    bound = min_deterrent_whitewash_cost(10_000 - CFG.opportunist_bps, CFG.premium, qs, min(CFG.schedule))
    assert bound > 0
    res = run_reputation_grid(CFG, ACCURACIES, [bound, bound + 7], master_seed=SEED)
    assert all(r["whitewashes"] == 0 and r["whitewash_paid"] == 0 for r in res.rows)
    assert min_deterrent_whitewash_cost(0, CFG.premium, qs) == 0         # prior 0: fresh id = floor


@pytest.mark.parametrize("acc", [6000, 8500])
def test_bisected_threshold_is_exact_and_within_bound(acc):
    th = find_reputation_threshold(CFG, acc, master_seed=SEED)
    washes = lambda c: run_reputation_grid(CFG, [acc], [c], master_seed=SEED).rows[0]["whitewashes"]
    assert 0 <= th.threshold <= th.analytic and th.accuracy == acc
    assert washes(th.threshold) == 0
    if th.threshold > 0:
        assert washes(th.threshold - 1) > 0


def test_findings_and_reproducibility_from_master_seed():
    small = ReputationConfig(n_sellers=20, horizon=6)
    a, b = (run_reputation(small, (6000, 10000), (0, 16), master_seed=SEED) for _ in range(2))
    assert a.rows == b.rows and a.findings == b.findings
    assert a.findings["whitewash_threshold"] <= a.findings["deterrent_bound"]
    other = run_reputation(small, (6000, 10000), (0, 16), master_seed=SEED + 1)
    assert [r["seed"] for r in other.rows] != [r["seed"] for r in a.rows]


def test_serial_and_parallel_rows_identical():
    ser = run_reputation_grid(CFG, (6000, 10000), (0, 8), master_seed=SEED)
    par = run_reputation_grid(CFG, (6000, 10000), (0, 8), master_seed=SEED, n_workers=2)
    assert ser.rows == par.rows
