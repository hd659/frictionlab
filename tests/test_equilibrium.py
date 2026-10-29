"""Tests for frictionlab.analysis.equilibrium: replicator map, rounding, detector."""
import numpy as np
import pytest

from frictionlab.analysis import (
    StabilityDetector, as_simplex, evolve, largest_remainder, matrix_fitness, replicator_step,
)

PD = [[3.0, 0.0], [5.0, 1.0]]          # prisoner's dilemma: defect (row 1) strictly dominates
V, C = 2.0, 4.0
HAWK_DOVE = [[(V - C) / 2, V], [0.0, V / 2]]   # interior ESS: hawk share = V / C


def on_simplex(x, tol=1e-12):
    x = np.asarray(x)
    return bool(np.all(x >= 0) and abs(x.sum() - 1.0) < tol)


def test_as_simplex_clips_and_validates():
    assert np.allclose(as_simplex([2, 2, -1]), [0.5, 0.5, 0.0])
    for bad in ([], [0, 0], [np.nan, 1], [[0.5, 0.5]]):
        with pytest.raises(ValueError):
            as_simplex(bad)


@pytest.mark.parametrize("seed", range(5))
def test_shares_stay_on_simplex_every_step(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 5, size=(4, 4))
    res = evolve(rng.dirichlet(np.ones(4)), matrix_fitness(a), eta=0.05, generations=300,
                 detector=StabilityDetector(epsilon=1e-12, window=5))
    assert all(on_simplex(x) for x in res.shares)
    assert res.fitness.shape == (len(res.shares) - 1, 4)


def test_large_eta_never_produces_negative_shares():
    x = replicator_step([0.5, 0.5], [0.0, 100.0], eta=1.0)
    assert on_simplex(x) and x[0] == 0.0
    assert on_simplex(replicator_step([0.3, 0.7], [-50.0, 9.0], eta=0.5, normalize=True))


def test_equal_fitness_leaves_shares_unchanged():
    x0 = np.array([0.2, 0.3, 0.5])
    assert np.array_equal(replicator_step(x0, [7.0, 7.0, 7.0], eta=0.9), x0)
    res = evolve(x0, lambda x, g: np.full(3, -4.0), eta=0.5, generations=50,
                 detector=StabilityDetector(epsilon=1e-9, window=3))
    assert res.reason == "stable" and np.allclose(res.shares, x0, atol=0, rtol=0)
    assert np.array_equal(res.fixed_point, x0)


def test_zero_share_types_are_absorbing():
    x = replicator_step([0.0, 0.4, 0.6], [100.0, 1.0, 0.0], eta=0.5)
    assert x[0] == 0.0 and x[1] > 0.4


def test_largest_remainder_preserves_total():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 7))
        total = int(rng.integers(0, 60))
        counts = largest_remainder(rng.dirichlet(np.ones(n)), total)
        assert counts.sum() == total and np.all(counts >= 0)
        assert counts.dtype.kind == "i"


def test_largest_remainder_known_cases():
    assert largest_remainder([1 / 3, 1 / 3, 1 / 3], 10).tolist() == [4, 3, 3]
    assert largest_remainder([0.45, 0.35, 0.2], 7).tolist() == [3, 3, 1]
    assert largest_remainder([0.5, 0.5], 0).tolist() == [0, 0]
    with pytest.raises(ValueError):
        largest_remainder([0.5, 0.5], 2.5)


def test_strictly_dominant_strategy_goes_to_fixation():
    res = evolve([0.99, 0.01], matrix_fitness(PD), eta=0.2, generations=5000,
                 detector=StabilityDetector(epsilon=1e-9, window=10, floor=1e-6),
                 labels=("cooperate", "defect"))
    assert res.converged and res.reason == "extinction"
    assert res.final["defect"] > 1 - 1e-6
    assert np.all(np.diff(res.shares[:, 1]) > 0)     # monotone takeover


def test_hawk_dove_converges_to_interior_mixed_equilibrium():
    for h0 in (0.05, 0.3, 0.9):
        res = evolve([h0, 1 - h0], matrix_fitness(HAWK_DOVE), eta=0.1, generations=5000,
                     detector=StabilityDetector(epsilon=1e-10, window=20),
                     labels=("hawk", "dove"))
        assert res.reason == "stable"
        assert res.fixed_point[0] == pytest.approx(V / C, abs=1e-6)
        f = res.fitness[-1]
        assert f[0] == pytest.approx(f[1], abs=1e-5)   # indifference at the mix


def test_detector_fires_on_constant_trajectory():
    det = StabilityDetector(epsilon=1e-8, window=5)
    traj = np.tile([0.25, 0.75], (6, 1))
    assert det.check(traj[:5]) is None                # needs window + 1 rows
    assert det.check(traj) == "stable"
    assert det.scan(np.tile([0.25, 0.75], (20, 1))) == (5, "stable")


def test_detector_does_not_fire_on_oscillation():
    det = StabilityDetector(epsilon=1e-3, window=5)
    traj = np.array([[0.4, 0.6], [0.6, 0.4]] * 50)
    assert det.check(traj) is None and det.scan(traj) is None
    damped = np.array([[0.5 + 0.1 * (-0.5) ** t, 0.5 - 0.1 * (-0.5) ** t] for t in range(60)])
    hit = det.scan(damped)
    assert hit is not None and hit[1] == "stable" and hit[0] > 5


def test_detector_extinction_floor():
    det = StabilityDetector(epsilon=1e-12, window=50, floor=0.01)
    assert det.check([[0.5, 0.5], [0.2, 0.8]]) is None
    assert det.check([[0.5, 0.5], [0.005, 0.995]]) == "extinction"
    assert det.check([[0.005, 0.995], [0.004, 0.996]]) is None   # never alive above floor
    with pytest.raises(ValueError):
        StabilityDetector(window=0)


def test_rock_paper_scissors_cycles_without_convergence():
    rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    res = evolve([0.5, 0.3, 0.2], matrix_fitness(rps), eta=0.1, generations=400,
                 detector=StabilityDetector(epsilon=1e-6, window=10))
    assert not res.converged and res.fixed_point is None
    assert len(res.shares) == 401 and all(on_simplex(x) for x in res.shares)


def test_input_validation():
    with pytest.raises(ValueError):
        matrix_fitness([[1, 2, 3]])
    with pytest.raises(ValueError):
        replicator_step([0.5, 0.5], [1.0], eta=0.1)
    with pytest.raises(ValueError):
        replicator_step([0.5, 0.5], [1.0, 2.0], eta=-0.1)
    with pytest.raises(ValueError):
        evolve([0.5, 0.5], matrix_fitness(PD), labels=("a",))
