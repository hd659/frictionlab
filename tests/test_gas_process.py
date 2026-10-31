"""GasProcess: seeded two-regime gas shocks as a time-varying fixed per-trade cost."""
import numpy as np
import pytest

from frictionlab.agents import make_zi_population
from frictionlab.core import Side
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import (
    CALM, SHOCK, TAKER_ONLY, Fill, FixedCost, GasCost, GasProcess, MakerTakerFee, Role,
)


def fills(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [Fill(1, Side(int(rng.choice([-1, 1]))), Role(rng.choice(["maker", "taker"])),
                 int(rng.integers(1, 500)), int(rng.integers(1, 50)), float(rng.uniform(0, 300)))
            for _ in range(n)]


def test_zero_shock_matches_fixed_cost_per_fill():
    gas = GasProcess(7, 900, p_shock=0.0, p_calm=0.4, seed=5).cost()
    fixed = FixedCost(7)
    for f in fills():
        assert gas.charge(f) == fixed.charge(f)
        assert gas.raw_cost(f) == fixed.raw_cost(f)


def _ledger_state(stack, seed=4):
    agents = make_zi_population("zi-c", 4, 4, seed=seed, n_units=6)
    res = MarketRunner(agents, stack, seed=seed, cash=10**6, position=10,
                       check_every=True).run(until=60.0)
    return ([(t.price, t.qty, t.buyer_id, t.seller_id) for t in res.trades],
            [(a.agent_id, a.cash, a.position, a.fees_paid) for a in res.ledger],
            res.fees)


def test_zero_shock_runner_bit_for_bit_with_constant_gas():
    proc = GasProcess(3, 500, p_shock=0.0, p_calm=0.2, dt=0.5, seed=9)
    a = _ledger_state(MakerTakerFee(taker_bps=10) + proc.cost())
    b = _ledger_state(MakerTakerFee(taker_bps=10) + FixedCost(3))
    assert a == b and a[2] > 0


def test_constant_constructor_and_shock_start():
    c = GasProcess.constant(4, seed=1)
    assert set(c.path(500)) == {4} and c.stationary == (1.0, 0.0)
    absorbed = GasProcess(1, 60, p_shock=0.0, p_calm=0.0, start=SHOCK)
    assert set(absorbed.path(100)) == {60} and absorbed.stationary == (0.0, 1.0)


@pytest.mark.parametrize("p_shock,p_calm", [(0.05, 0.2), (0.3, 0.3), (0.1, 0.9)])
def test_regime_frequency_matches_stationary(p_shock, p_calm):
    proc = GasProcess(2, 40, p_shock, p_calm, seed=17)
    pi_shock = proc.stationary[1]
    assert pi_shock == pytest.approx(p_shock / (p_shock + p_calm))
    assert proc.shock_frequency(200_000) == pytest.approx(pi_shock, abs=0.01)
    lv = proc.path(200_000)
    assert lv.mean() == pytest.approx(proc.mean_gas, rel=0.03)


def test_transition_frequencies_match_probabilities():
    proc = GasProcess(2, 40, 0.15, 0.35, seed=3)
    r = proc.regimes(100_001)
    prev, nxt = r[:-1], r[1:]
    assert nxt[prev == CALM].mean() == pytest.approx(0.15, abs=0.01)
    assert (nxt[prev == SHOCK] == CALM).mean() == pytest.approx(0.35, abs=0.015)


def test_level_is_query_order_independent_and_seeded():
    a = GasProcess(2, 30, 0.2, 0.4, dt=0.25, seed=11)
    b = GasProcess(2, 30, 0.2, 0.4, dt=0.25, seed=11)
    ts = np.linspace(0, 700, 901)
    forward = [a.level(t) for t in ts]
    backward = [b.level(t) for t in ts[::-1]][::-1]
    assert forward == backward
    assert a.level(1.2) == a.level_at_step(4) == a(1.2) == a.value(1.2)
    assert not np.array_equal(GasProcess(2, 30, 0.2, 0.4, seed=12).path(2000), a.path(2000))


def test_gas_cost_reads_time_or_clock_and_roles():
    proc = GasProcess(1, 99, 0.5, 0.5, seed=2)
    n = next(i for i in range(1000) if proc.regime_at_step(i) == SHOCK)
    cost = GasCost(proc, TAKER_ONLY)
    taker = Fill(1, Side.BUY, Role.TAKER, 100, 5, float(n))
    maker = Fill(2, Side.SELL, Role.MAKER, 100, 5, float(n))
    assert cost.charge(taker) == 99 and cost.charge(maker) == 0

    class Clock:
        now = 0.0
    bound = proc.cost(clock=Clock())
    assert bound.charge(taker) == proc.level(0.0)
    Clock.now = float(n)
    assert bound.charge(Fill(1, Side.BUY, Role.TAKER, 100, 5, 0.0)) == 99


def test_validation():
    with pytest.raises(ValueError):
        GasProcess(-1, 5)
    with pytest.raises(ValueError):
        GasProcess(1, 2.5)
    with pytest.raises(ValueError):
        GasProcess(1, 5, p_shock=1.5)
    with pytest.raises(ValueError):
        GasProcess(1, 5, dt=0)
    with pytest.raises(ValueError):
        GasProcess(1, 5).level(-1.0)
    with pytest.raises(TypeError):
        GasCost(FixedCost(1))
