"""fl-056: lemons and certification experiment over certification_cost x buyer_premium."""
import pytest

from frictionlab.experiments import (
    LemonsConfig, akerlof_pooled_price, certification_bound, certification_cutoff,
    find_certification_threshold, lemons_qualities, run_lemons, run_lemons_grid,
)
from frictionlab.experiments.lemons import COSTS, PREMIUMS

CFG, SEED = LemonsConfig(), 56
INF = 10**9


@pytest.fixture(scope="module")
def grid():
    return run_lemons_grid(CFG, master_seed=SEED)


@pytest.fixture(scope="module")
def pooled():
    return run_lemons_grid(LemonsConfig(certification=False), COSTS[:1], PREMIUMS, master_seed=SEED)


def _cell(res, cost, prem):
    (row,) = [r for r in res.rows if r["certification_cost"] == cost and r["buyer_premium"] == prem]
    return row


def test_ledger_invariants_hold_in_every_cell(grid, pooled):
    assert len(grid.rows) == len(COSTS) * len(PREMIUMS) and not grid.errors
    assert all(r["violations"] == 0 for r in grid.rows + pooled.rows)
    for r in grid.rows:
        assert 0 <= r["certification_uptake"] <= r["traded_share"] <= 1
        assert r["certification_paid"] == r["certified"] * r["certification_cost"]


def test_pooled_price_equals_akerlof_fixed_point_without_certification(pooled):
    qs = lemons_qualities(CFG, SEED)
    for r in pooled.rows:
        price, top = akerlof_pooled_price(qs, r["buyer_premium"])
        assert r["pooled_price"] == price and r["certified"] == 0
        assert top <= max(qs) and r["traded_share"] == sum(q <= price for q in qs) / len(qs)


def test_akerlof_helper_unravels_and_pools_at_the_extremes():
    qs = list(CFG.schedule)
    assert akerlof_pooled_price(qs, 0) == (min(qs), min(qs))
    assert akerlof_pooled_price(qs, max(qs) - min(qs))[1] == max(qs)


def test_traded_and_surplus_share_non_decreasing_in_premium(grid):
    for c in COSTS:
        for key in ("traded_share", "surplus_share"):
            xs = [_cell(grid, c, p)[key] for p in PREMIUMS]
            assert xs == sorted(xs), (c, key, xs)


def test_uptake_non_increasing_and_cutoff_non_decreasing_in_cost(grid):
    for p in PREMIUMS:
        up = [_cell(grid, c, p)["certification_uptake"] for c in COSTS]
        cut = [_cell(grid, c, p)["cutoff"] for c in COSTS]
        assert up == sorted(up, reverse=True) and cut == sorted(cut), (p, up, cut)


def test_analytic_cutoff_matches_simulation_and_is_monotone(grid):
    qs = lemons_qualities(CFG, SEED)
    for r in grid.rows:
        cut = certification_cutoff(qs, r["buyer_premium"], r["certification_cost"])
        assert r["cutoff"] == (max(qs) + 1 if cut is None else cut)
    for p in PREMIUMS:
        cuts = [certification_cutoff(qs, p, c) for c in range(certification_bound(qs, p) + 3)]
        vals = [INF if x is None else x for x in cuts]
        assert vals == sorted(vals), (p, cuts)


def test_no_certification_beyond_the_analytic_bound():
    qs = lemons_qualities(CFG, SEED)
    for p in PREMIUMS:
        b = certification_bound(qs, p)
        res = run_lemons_grid(CFG, [b, b + 5], [p], master_seed=SEED)
        assert all(r["certified"] == 0 and r["certification_paid"] == 0 for r in res.rows)
        assert certification_cutoff(qs, p, b) is None


@pytest.mark.parametrize("prem,frac", [(20, 0.05), (20, 0.5), (40, 0.3), (0, 0.05)])
def test_bisected_threshold_is_exact_and_within_bound(prem, frac):
    th = find_certification_threshold(CFG, prem, frac, master_seed=SEED)
    up = lambda c: run_lemons_grid(CFG, [c], [prem], master_seed=SEED).rows[0]["certification_uptake"]
    assert 0 <= th.threshold <= th.analytic == certification_bound(lemons_qualities(CFG, SEED), prem)
    assert up(th.threshold) < frac
    if th.threshold > 0:
        assert up(th.threshold - 1) >= frac
    with pytest.raises(ValueError):
        find_certification_threshold(CFG, prem, 0.0)


def test_findings_and_reproducibility_from_master_seed():
    a, b = (run_lemons(CFG, (0, 8, 32), (0, 20, 90), master_seed=SEED) for _ in range(2))
    assert a.rows == b.rows and a.findings == b.findings
    assert a.findings["certification_threshold"] <= a.findings["certification_bound"]
    assert a.findings["akerlof_pooled_price"][0] == min(lemons_qualities(CFG, SEED))
    other = run_lemons(CFG, (0, 8, 32), (0, 20, 90), master_seed=SEED + 1)
    assert [r["seed"] for r in other.rows] != [r["seed"] for r in a.rows]


def test_serial_and_parallel_rows_identical():
    ser = run_lemons_grid(CFG, (0, 4, 16), (5, 40), master_seed=SEED)
    par = run_lemons_grid(CFG, (0, 4, 16), (5, 40), master_seed=SEED, n_workers=2)
    assert ser.rows == par.rows
