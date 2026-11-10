"""fl-074: exact hold-up analytics and the bisected contracting threshold."""
import pytest

from frictionlab.agents import SpecificInvestor
from frictionlab.agents.specific_investor import seller_share
from frictionlab.experiments import (
    HoldupConfig, find_contracting_threshold, first_best_investment, holdup_investment,
    holdup_surplus_loss, max_contracting_cost, run_holdup_grid,
)

CFG, MASTER = HoldupConfig(), 74
SCHEDS = {
    "default": (CFG.schedule, CFG.outside_value),
    "steep": (((0, 20), (4, 30), (10, 40), (18, 48), (30, 55)), 15),
    "flat": (((0, 50), (5, 52), (9, 53)), 10),
    "linear": (((0, 10), (2, 14), (4, 18), (6, 22)), 10),
}
WS = list(range(0, 10_001, 250))
# Monotonicity holds for the real-valued split; the fl-040 floor can flip the argmax by one
# level between weights 250 bps apart, so it is asserted on the 1000 bps and experiment grids.
MONO = sorted(set(range(0, 10_001, 1_000)) | {5_000, 6_000, 7_500, 9_000})


@pytest.mark.parametrize("name", sorted(SCHEDS))
def test_holdup_investment_monotone_in_weight_and_first_best_at_full(name):
    s, o = SCHEDS[name]
    xs = [holdup_investment(s, o, w) for w in MONO]
    fb = first_best_investment(s, o)
    assert xs == sorted(xs) and max(xs) <= fb
    assert all(holdup_investment(s, o, w) <= fb for w in WS)
    assert holdup_investment(s, o, 10_000) == fb


@pytest.mark.parametrize("name", sorted(SCHEDS))
def test_first_best_is_smallest_surplus_maximizer(name):
    s, o = SCHEDS[name]
    surplus = [v - o - c for c, v in s]
    i = first_best_investment(s, o)
    assert surplus[i] == max(surplus) and all(x < surplus[i] for x in surplus[:i])


@pytest.mark.parametrize("name", sorted(SCHEDS))
def test_max_contracting_cost_zero_when_holdup_reaches_first_best(name):
    s, o = SCHEDS[name]
    fb = first_best_investment(s, o)
    for w in WS:
        at_fb = holdup_investment(s, o, w) == fb
        assert (holdup_surplus_loss(s, o, w) == 0) == at_fb
        if at_fb:      # no efficiency loss left: only the buyer's share of the sunk cost remains
            c, v = s[fb]
            shifted = c - (seller_share(v - o, w) - seller_share(v - o - c, w))
            assert max_contracting_cost(s, o, w) == max(shifted - 1, 0)
    assert max_contracting_cost(s, o, 10_000) == 0 and max_contracting_cost(s, o, 0) == 0


@pytest.mark.parametrize("name", sorted(SCHEDS))
def test_max_contracting_cost_is_largest_cost_the_seller_still_contracts_at(name):
    s, o = SCHEDS[name]
    agent = SpecificInvestor(1)
    for w in WS:
        m = max_contracting_cost(s, o, w)
        assert isinstance(m, int) and m >= 0
        assert not agent.wants_contract(s, o, w, m + 1)
        if m > 0 or agent.wants_contract(s, o, w, 0):
            assert agent.wants_contract(s, o, w, m)


@pytest.mark.parametrize("name", sorted(SCHEDS))
def test_max_contracting_cost_is_recovered_loss_identity(name):
    s, o = SCHEDS[name]
    fb = first_best_investment(s, o)
    s_star = s[fb][1] - o - s[fb][0]
    for w in WS:
        c, v = s[holdup_investment(s, o, w)]
        buyer = lambda x: x - seller_share(x, w)                          # noqa: E731
        gain = holdup_surplus_loss(s, o, w) + buyer(v - o) - buyer(s_star)
        assert gain == seller_share(s_star, w) - (seller_share(v - o, w) - c)   # exact ticks
        assert max_contracting_cost(s, o, w) == max(gain - 1, 0)
    s0 = ((0, 10), (1, 10))                        # degenerate: costly step adds no value
    assert holdup_surplus_loss(s0, 0, 5_000) == max_contracting_cost(s0, 0, 5_000) == 0


@pytest.mark.parametrize("w", [5_000, 7_500, 9_000, 10_000])
def test_simulated_rate_zero_once_cost_exceeds_bound(w):
    bound = max(max_contracting_cost(tuple((c, v + e) for c, v in CFG.schedule),
                                     CFG.outside_value, w) for e in CFG.value_noise)
    res = run_holdup_grid(CFG, [bound + 1, bound + 5], [w], master_seed=MASTER)
    assert all(r["contracting_rate"] == 0.0 and r["surplus_share"] <= 1.0 for r in res.rows)


@pytest.mark.parametrize("w", [5_000, 6_000, 7_500, 9_000])
def test_bisected_threshold_collapses_at_level_not_below(w):
    thr = find_contracting_threshold(CFG, w, collapse_bps=5_000, master_seed=MASTER)
    rate = lambda c: run_holdup_grid(CFG, [c], [w], master_seed=MASTER).rows[0]["contracting_rate"]  # noqa: E731
    c = thr.contracting_cost
    assert rate(c) < 0.5 and c >= 1 and rate(c - 1) >= 0.5
    assert c <= thr.analytic + 1 and thr.analytic == max(
        max_contracting_cost(tuple((a, v + e) for a, v in CFG.schedule), CFG.outside_value, w)
        for e in CFG.value_noise)
    assert c <= max_contracting_cost(CFG.schedule, CFG.outside_value, w) + 1
    assert thr.probes[c] == rate(c)


def test_threshold_zero_when_contracting_never_pays():
    thr = find_contracting_threshold(CFG, 10_000, master_seed=MASTER)
    assert thr.contracting_cost == 0 and thr.analytic == 0


def test_bisection_rejects_bad_fraction():
    with pytest.raises(ValueError):
        find_contracting_threshold(CFG, 5_000, collapse_bps=0)
