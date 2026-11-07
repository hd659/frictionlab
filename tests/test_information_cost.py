"""fl-052: SignalSource (costly two-point signals) and the expected-edge arithmetic."""
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import frictionlab.costs as costs
from frictionlab.agents import FundamentalProcess
from frictionlab.core.ledger import Ledger
from frictionlab.costs import (
    BPS_ONE, HIGH, INFO_SINK_ID, LOW, FixedCost, MakerTakerFee, SignalSource,
    breakeven_accuracy, expected_edge, two_point,
)

ROOT = Path(__file__).resolve().parents[1]


def source(acc=7_500, price=0, v=110, center=100, jump=10, seed=3):
    return SignalSource(FundamentalProcess.constant(v), acc, price=price,
                        center=center, jump=jump, seed=seed)


def exact_edge(acc, jump, spread):
    import math
    return math.floor(Fraction((2 * acc - BPS_ONE) * jump, BPS_ONE) - Fraction(spread, 2))


def test_reexported_and_sink_id_unique():
    for name in ("SignalSource", "INFO_SINK_ID", "expected_edge", "two_point"):
        assert name in costs.__all__
    assert INFO_SINK_ID not in (-1, -2, -3, -4, -5, 0)


@pytest.mark.parametrize("acc", [0, 1, 4_999, 5_000, 5_001, 6_250, 7_777, 9_999, 10_000])
@pytest.mark.parametrize("jump,spread", [(1, 0), (10, 0), (10, 1), (10, 3), (7, 5), (40, 2)])
def test_expected_edge_matches_exact_rational_floor(acc, jump, spread):
    assert expected_edge(acc, jump, spread) == exact_edge(acc, jump, spread)


def test_uninformative_signal_has_no_positive_edge():
    for jump in (1, 5, 50):
        for spread in range(0, 6):
            assert expected_edge(5_000, jump, spread) <= 0
    assert expected_edge(10_000, 10, 0) == 10 and expected_edge(10_000, 10, 2) == 9


@pytest.mark.parametrize("jump,spread,hurdle", [(10, 0, 0), (10, 2, 3), (25, 1, 7), (3, 0, 1)])
def test_breakeven_accuracy_is_the_tight_threshold(jump, spread, hurdle):
    a = breakeven_accuracy(jump, spread, hurdle)
    assert a is not None and expected_edge(a, jump, spread) > hurdle
    assert a == 0 or expected_edge(a - 1, jump, spread) <= hurdle
    assert breakeven_accuracy(10, 0, 10) is None   # even a perfect signal earns only 10


def test_perfect_signal_always_matches_state():
    for v, state in ((110, HIGH), (90, LOW), (100, HIGH)):
        s = source(10_000, v=v)
        assert all(s.draw(aid, 0.0) == state for aid in range(1, 6) for _ in range(200))


@pytest.mark.parametrize("acc", [5_000, 6_000, 8_500, 9_500])
def test_empirical_hit_rate_matches_accuracy(acc):
    s, n = source(acc, seed=11), 20_000
    hits = sum(s.draw(7, 0.0) == HIGH for _ in range(n))
    p = acc / BPS_ONE
    assert abs(hits / n - p) < 4 * np.sqrt(p * (1 - p) / n) + 1e-9   # 4-sigma band


def test_streams_are_per_agent_and_seeded():
    a, b = source(6_000, seed=5), source(6_000, seed=5)
    seq_a = [a.draw(1, 0.0) for _ in range(300)]
    for _ in range(500):                               # agent 2 buys a lot on `b` first
        b.draw(2, 0.0)
    assert [b.draw(1, 0.0) for _ in range(300)] == seq_a
    assert [source(6_000, seed=6).draw(1, 0.0) for _ in range(300)] != seq_a


def test_sell_charges_price_into_sink_and_reconciles():
    s = source(8_000, price=FixedCost(3) + MakerTakerFee(taker_bps=25))  # 3 + 0.25 -> 4
    led = Ledger()
    led.open_account(1, 100)
    led.open_account(2, 100)
    assert s.cost(1) == 4
    for t in range(5):
        assert s.sell(led, 1, float(t)) in (HIGH, LOW)
    assert s.sell(led, 2, 0.0) in (HIGH, LOW)
    assert s.purchases == {1: 5, 2: 1} and s.revenue == 6 * 4
    assert led[INFO_SINK_ID].cash == s.revenue == sum(s.purchases.values()) * 4
    assert led[1].cash == 80 and led[2].cash == 96
    led.check()


def test_sell_refuses_when_agent_cannot_pay():
    s = source(9_000, price=5)
    led = Ledger(allow_overdraft=False)
    led.open_account(1, 7)
    assert s.sell(led, 1, 0.0) is not None and s.sell(led, 1, 0.0) is None
    assert led[1].cash == 2 and s.purchases == {1: 1} and len(s.log) == 1
    led.check()


def test_hit_rate_log_and_state_follow_fundamental():
    walk = FundamentalProcess.random_walk(100, jump_rate=1.0, jump_size=10, seed=4)
    s = SignalSource(walk, 10_000, seed=1)
    led = Ledger()
    led.open_account(1)
    for t in range(30):
        s.sell(led, 1, float(t))
    assert s.hit_rate() == 1.0
    assert all(st == (HIGH if walk.value(t) >= 100 else LOW) for t, _, _, st in s.log)


def test_two_point_is_seeded_and_symmetric():
    vals = [two_point(100, 10, seed=k).value(0.0) for k in range(200)]
    assert set(vals) == {90, 110} and 60 < vals.count(110) < 140
    assert two_point(100, 10, seed=9).value(5.0) == two_point(100, 10, seed=9).value(0.0)


def test_validation():
    f = FundamentalProcess.constant(100)
    with pytest.raises(ValueError):
        SignalSource(f, 10_001)
    with pytest.raises(TypeError):
        SignalSource(object())
    with pytest.raises(ValueError):
        two_point(5, 5)
    with pytest.raises(ValueError):
        SignalSource(f, price=-1)


def test_module_under_hygiene_ceiling():
    for rel in ("frictionlab/costs/information.py", "frictionlab/agents/info_acquirer.py"):
        assert len((ROOT / rel).read_text("utf-8").splitlines()) <= 200, rel
