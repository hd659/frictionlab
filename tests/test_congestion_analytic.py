"""fl-068: Naor individual / social thresholds, optimal toll and the bisected congestion toll."""
from fractions import Fraction

import pytest

from frictionlab.experiments import (
    CongestionConfig, find_congestion_toll, naor_individual_threshold, naor_optimal_toll,
    naor_social_threshold, run_queue_grid, welfare_per_step,
)
from frictionlab.experiments.congestion import SMALL, build

MASTER = 68
SMALL_CFG = CongestionConfig(**{k: SMALL[k] for k in CongestionConfig.__dataclass_fields__})
CASES = [(r, c, p, q) for r in (10, 24, 40) for c in (1, 2, 4) for p in (3000, 7000)
         for q in (2500, 5000, 9000)]


def test_individual_threshold_closed_form():
    # R=24, c=2, q=1/2: k = 12 steps affordable, 12 * 1/2 - 1 = 5 customers ahead tolerated
    assert naor_individual_threshold(24, 2, 5000) == 5
    assert naor_individual_threshold(24, 2, 5000, toll=17) == 0
    assert naor_individual_threshold(24, 2, 5000, toll=25) == -1
    assert naor_individual_threshold(24, 0, 5000) is None


def test_individual_threshold_matches_queue_market_rule():
    for r, c, _, q in CASES:
        for toll in (0, 3, 9):
            m = build(CongestionConfig(reward=r, service_prob_bps=q), {"toll": toll, "waiting_cost": c}, 0).market
            n = naor_individual_threshold(r, c, q, toll)
            assert all(m.expected_net(k) >= 0 for k in range(n + 1))
            assert m.expected_net(n + 1) < 0


@pytest.mark.parametrize("r,c,p,q", CASES)
def test_social_threshold_at_most_individual(r, c, p, q):
    s, n = naor_social_threshold(r, c, p, q), naor_individual_threshold(r, c, q)
    assert -1 <= s <= n                                    # Naor: self-interest over-joins
    w = [welfare_per_step(r, c, p, q, k) for k in range(-1, n + 1)]
    assert w[s + 1] == max(w) and all(x < max(w) for x in w[:s + 1])


@pytest.mark.parametrize("r,c,p,q", CASES[::3])
def test_individual_threshold_non_increasing_in_toll(r, c, p, q):
    ns = [naor_individual_threshold(r, c, q, t) for t in range(r + 2)]
    assert ns == sorted(ns, reverse=True) and ns[-1] == -1
    t = naor_optimal_toll(r, c, p, q)
    assert ns[t] == naor_social_threshold(r, c, p, q) and (t == 0 or ns[t - 1] > ns[t])


def test_welfare_per_step_exact_mm1_check():
    # threshold 0: X in {0, 1}, pi(1) = a / (a + q) with a = p(1 - q)
    p, q = Fraction(7, 10), Fraction(1, 2)
    a = p * (1 - q)
    pi1 = a / (a + q)
    expect = 24 * p * (1 - pi1) - 2 * (pi1 + p * (1 - pi1))
    assert welfare_per_step(24, 2, 7000, 5000, 0) == expect
    assert welfare_per_step(24, 2, 7000, 5000, -1) == 0


def test_optimal_toll_raises_welfare_on_small_preset():
    for c in SMALL["waiting_costs"]:
        t = naor_optimal_toll(SMALL_CFG.reward, c, SMALL_CFG.arrival_prob_bps, SMALL_CFG.service_prob_bps)
        zero, opt = run_queue_grid(SMALL_CFG, [0, t], [c], master_seed=MASTER).rows
        assert opt["welfare"] >= zero["welfare"] and opt["welfare_change_vs_zero_toll"] >= 0


def test_bisected_toll_is_tight_and_matches_analytic():
    res = find_congestion_toll(SMALL_CFG, 2, master_seed=MASTER)
    s, t = res.social_threshold, res.toll
    assert res.probes[t] <= s
    lower = res.probes.get(t - 1)
    if lower is None:
        lower = run_queue_grid(SMALL_CFG, [t - 1], [res.waiting_cost], master_seed=MASTER).rows[0][
            "max_queue_length"]
    assert lower > s
    assert t == res.analytic == naor_optimal_toll(SMALL_CFG.reward, res.waiting_cost,
                                                  SMALL_CFG.arrival_prob_bps, SMALL_CFG.service_prob_bps)
    assert find_congestion_toll(SMALL_CFG, 2, master_seed=MASTER) == res
