"""fl-065: analytic critical mass, stable fixed points and the bisected tipping seed."""
import pytest

from frictionlab.experiments import (
    CriticalMassConfig, critical_mass, empty_is_fixed_point, find_tipping_seed, run_platform_grid,
    tipping_fixed_points,
)
from frictionlab.experiments.critical_mass import COSTS, build, side_values

MASTER = 65
# homogeneous small preset: every member has u = -6, b = 1, f = 0 on both sides of 8
HOMO = CriticalMassConfig(n_a=8, n_b=8, standalone_a=(-6,), standalone_b=(-6,), benefit_a=1,
                          benefit_b=1, fee_a=0, fee_b=0, horizon=20)
U8 = [-6] * 8


def _hetero_args():
    m = build(CriticalMassConfig(), {"adoption_cost": 0, "seed_members": 0}, 1).market
    return side_values(m), m.benefit, m.fee


def test_homogeneous_closed_form():
    # stay needs n_other >= 6, join needs n_other >= 6 + c: from k seeded, k >= 6 + c tips
    assert critical_mass(U8, U8, 1, 1, 0, 0, 0) == 6
    assert critical_mass(U8, U8, 1, 1, 0, 0, 2) == 8         # tips via the stay rule at k = 8
    assert tipping_fixed_points(U8, U8, 1, 1, 0, 0, 0) == [(0, 0), (8, 8)]
    assert tipping_fixed_points(U8, U8, 1, 1, 0, 0, 2) == [(0, 0), (6, 6), (7, 7), (8, 8)]


def test_critical_mass_non_decreasing_in_adoption_cost():
    (ua, ub), b, f = _hetero_args()
    for args in ((ua, ub, *b, *f), (U8, U8, 1, 1, 0, 0), ([-3] * 10, [-3] * 10, 1, 1, 2, -1)):
        ks = [critical_mass(*args, c) for c in range(0, 50, 2)]
        assert ks == sorted(ks), (args, ks)


def test_zero_when_empty_is_not_a_fixed_point():
    (ua, ub), b, f = _hetero_args()
    for c in range(0, 50):
        empty = empty_is_fixed_point(ua, ub, *f, c)
        assert empty == ((0, 0) in tipping_fixed_points(ua, ub, *b, *f, c))
        if not empty:
            assert critical_mass(ua, ub, *b, *f, c) == 0
    assert critical_mass([5] * 4, [-9] * 4, 3, 3, 0, 0, 4) == 0   # side A joins an empty platform
    assert not empty_is_fixed_point([5] * 4, [-9] * 4, 0, 0, 4)


def test_fixed_points_are_increasing_and_stable():
    (ua, ub), b, f = _hetero_args()
    for c in COSTS:
        pts = tipping_fixed_points(ua, ub, *b, *f, c)
        assert pts == sorted(pts) and len(set(pts)) == len(pts)
        assert pts[-1] == tipping_fixed_points(ua, ub, *b, *f, 0)[-1]   # top point is c-free
        na, nb = pts[-1]                                  # top point = stay-rule cutoff at itself
        assert na == sum(u + b[0] * nb - f[0] >= 0 for u in ua)
        assert nb == sum(u + b[1] * na - f[1] >= 0 for u in ub)
        assert ((0, 0) in pts) == empty_is_fixed_point(ua, ub, *f, c)


def test_cycle_is_never_reported_as_fixed_point():
    u = [-3] * 10
    pts = tipping_fixed_points(u, u, 1, 1, 2, -1, 0)
    res = run_platform_grid(CriticalMassConfig(n_a=10, n_b=10, standalone_a=(-3,), standalone_b=(-3,),
                                               benefit_a=1, benefit_b=1, fee_a=2, fee_b=-1, horizon=20),
                            [0], [2], master_seed=MASTER)
    (row,) = res.rows
    assert row["outcome"] == "cycle" and row["cycle_length"] == 2 and not row["on_fixed_point"]
    assert (row["members_a"], row["members_b"]) not in pts


def _members(cfg, c, k):
    return run_platform_grid(cfg, [c], [k], master_seed=MASTER).rows[0]["members"]


@pytest.mark.parametrize("c", [0, 2, 8, 40])
def test_bisected_seed_is_exact_and_matches_critical_mass(c):
    tip = find_tipping_seed(HOMO, c, 0.9, master_seed=MASTER)
    target = 0.9 * (HOMO.n_a + HOMO.n_b)
    assert _members(HOMO, c, tip.seed_members) > target
    if tip.seed_members > 0:
        assert _members(HOMO, c, tip.seed_members - 1) <= target
    assert tip.seed_members == tip.analytic == critical_mass(U8, U8, 1, 1, 0, 0, c)
    assert list(tip.probes) == sorted(tip.probes)


@pytest.mark.parametrize("c, frac", [(8, 0.9), (16, 0.5), (40, 0.8)])
def test_bisected_seed_condition_on_heterogeneous_default(c, frac):
    cfg = CriticalMassConfig()
    tip = find_tipping_seed(cfg, c, frac, master_seed=MASTER)
    target = frac * (cfg.n_a + cfg.n_b)
    assert _members(cfg, c, tip.seed_members) > target
    if tip.seed_members > 0:
        assert _members(cfg, c, tip.seed_members - 1) <= target


def test_tipping_seed_errors_when_unreachable():
    with pytest.raises(ValueError, match="never exceeds"):
        find_tipping_seed(HOMO, 0, 0.999 + 0.001, master_seed=MASTER)
