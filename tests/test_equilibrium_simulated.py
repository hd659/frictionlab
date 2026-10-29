"""Simulated-fitness replicator runs: satisficer vs optimizer under decision cost."""
import numpy as np
import pytest

from frictionlab.agents import BoundedRationalAgent, ZICTrader
from frictionlab.analysis import (
    SimulatedFitness, StabilityDetector, evolve, largest_remainder, mean_net_by_type,
)
from frictionlab.core.runner import MarketRunner
from frictionlab.core.types import Side

LABELS = ("satisficing", "optimizing")
N_AGENTS = 8
SEEN = []                                       # (generation, counts) seen by the factory


def br_market(params, seed):
    """Bounded-rational agents in the given mix trading against a ZI-C crowd."""
    counts = params["counts"]
    SEEN.append((params["generation"], dict(counts)))
    ss = np.random.SeedSequence(seed).spawn(N_AGENTS + 6)
    agents, aid = [], 1
    for mode in LABELS:
        for k in range(counts[mode]):
            agents.append(BoundedRationalAgent(
                aid, Side.BUY if k % 2 == 0 else Side.SELL, np.random.default_rng(ss[aid - 1]),
                mode=mode, max_steps=20, cost_per_step=params["cost_per_step"],
                aspiration=1.0, value_range=(95, 105), search_width=10, kappa=0.3,
                wake_rate=0.5))
            aid += 1
    for i in range(6):
        rng = np.random.default_rng(ss[N_AGENTS + i])
        v = int(rng.integers(80, 121))
        agents.append(ZICTrader(100 + i, Side.BUY if i < 3 else Side.SELL, rng,
                                schedule=[v] * 200, price_range=(60, 140)))
    return MarketRunner(agents, None, seed=seed, cash=10**9)


def br_metrics(res):
    return mean_net_by_type(res, lambda a: a.valuation, key=lambda a: getattr(a, "mode", None),
                            labels=LABELS)


def fitness(cost, **kw):
    kw.setdefault("master_seed", 0)
    return SimulatedFitness(br_market, br_metrics, LABELS, n_agents=N_AGENTS,
                            params={"cost_per_step": cost}, replicates=4,
                            run_kwargs={"until": 30.0}, **kw)


def run(cost, generations=4, **kw):
    return evolve([0.5, 0.5], fitness(cost, **kw), eta=0.5, generations=generations,
                  normalize=True, labels=LABELS,
                  detector=StabilityDetector(epsilon=1e-9, window=50, floor=0.01))


@pytest.fixture(scope="module")
def high_cost():
    return run(cost=8)


def test_high_decision_cost_moves_share_toward_satisficers(high_cost):
    sat = high_cost.shares[:, 0]
    assert sat[-1] > 0.5 and sat[-1] == sat.max()
    first = high_cost.fitness[0]
    assert first[0] > first[1]                  # satisficers out-earn optimizers net of cost
    assert all(abs(x.sum() - 1) < 1e-12 and np.all(x >= 0) for x in high_cost.shares)


def test_free_thinking_does_not_favour_satisficers(high_cost):
    free = run(cost=0)
    assert free.shares[-1, 0] < 0.5 < high_cost.shares[-1, 0]


def test_factory_receives_integer_counts_summing_to_population():
    SEEN.clear()
    res = run(cost=4, generations=3)
    assert len(SEEN) == 3 * 4                   # generations x replicates
    for g, counts in SEEN:
        assert sum(counts.values()) == N_AGENTS
        expect = largest_remainder(res.shares[g], N_AGENTS).tolist()
        assert [counts[lab] for lab in LABELS] == expect


def test_runs_are_deterministic_under_fixed_seed(high_cost):
    again = run(cost=8)
    assert np.array_equal(again.shares, high_cost.shares)
    assert np.array_equal(again.fitness, high_cost.fitness, equal_nan=True)
    other = run(cost=8, master_seed=4)
    assert not np.array_equal(other.fitness, high_cost.fitness, equal_nan=True)


def test_per_generation_seeds_are_fixed_and_distinct():
    f = fitness(4)
    assert f.seeds(0) == fitness(4).seeds(0)
    assert f.seeds(0) != f.seeds(1) and len(set(f.seeds(0))) == 4


def test_parallel_fitness_matches_serial():
    x = np.array([0.5, 0.5])
    assert np.array_equal(fitness(4)(x, 0), fitness(4, n_workers=2)(x, 0))


def test_unrepresented_archetype_gets_nan_and_is_neutral():
    f = fitness(4)(np.array([1.0, 0.0]), 0)
    assert np.isfinite(f[0]) and np.isnan(f[1])
    res = evolve([1.0, 0.0], fitness(4), eta=0.5, generations=2, normalize=True)
    assert np.array_equal(res.shares[-1], [1.0, 0.0])


def test_failed_generation_raises():
    bad = SimulatedFitness(br_market, br_metrics, LABELS, n_agents=N_AGENTS, params={})
    with pytest.raises(RuntimeError, match="generation 0"):
        bad(np.array([0.5, 0.5]), 0)
