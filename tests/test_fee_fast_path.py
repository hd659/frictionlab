"""fl-040: the integer-bps fee fast path returns exactly the frozen Fraction oracle.

Seeded random (notional, maker_bps, taker_bps, fixed cost, side) tuples go
through `FrictionStack` and through tests/reference_fees.py; every buyer/seller
charge must be the same integer, including rebates and rounding boundaries.
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from frictionlab.core.types import Side, Trade
from frictionlab.costs import (CostModel, Fill, FixedCost, FrictionStack, GasProcess,
                               HalfSpreadCost, LatencyCost, MakerTakerFee, Role, fills_of)
from frictionlab.costs.fees import div_round
from frictionlab.markets.fee_tiers import FeeComponent, FeeTier, FeeTierSchedule

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_fees as ref  # noqa: E402


def trade(price, qty, side):
    return Trade(1, 0.0, price, qty, 10, 11, 1, 2, side)


def pair(maker, taker, fixed=0, spread=None, latency=None):
    new = [MakerTakerFee(maker, taker), FixedCost(fixed)]
    old = [ref.ReferenceMakerTakerFee(maker, taker), ref.ReferenceFixedCost(fixed)]
    if spread is not None:
        new.append(HalfSpreadCost(spread))
        old.append(ref.ReferenceHalfSpreadCost(spread))
    if latency is not None:
        new.append(LatencyCost(*latency))
        old.append(ref.ReferenceLatencyCost(*latency))
    return FrictionStack(new), ref.ReferenceStack(old)


def boundary_price(rng, qty, bps):
    """A price whose notional sits within +-1 of an exact multiple of 10_000/bps."""
    g = 10_000 // math.gcd(10_000, abs(bps) or 1)
    target = g * int(rng.integers(1, 50))
    return max(1, -(-target // qty) + int(rng.integers(-1, 2)))


@pytest.mark.parametrize("seed", range(40))
def test_integer_bps_fast_path_matches_oracle(seed):
    rng = np.random.default_rng(seed)
    taker = int(rng.integers(0, 60))
    maker = int(rng.integers(-taker, 30))           # negative = rebate, never > taker
    fixed = int(rng.integers(0, 5))
    spread = int(rng.integers(0, 7)) if seed % 3 == 0 else None
    latency = (int(rng.integers(0, 3)), int(rng.integers(0, 4))) if seed % 4 == 0 else None
    fast, oracle = pair(maker, taker, fixed, spread, latency)
    assert not fast._plan(Role.TAKER)[1] and not fast._plan(Role.MAKER)[1]
    for i in range(250):
        qty = int(rng.integers(1, 40)) | (1 if i % 2 else 0)   # odd qtys half the time
        bps = maker if i % 2 else taker
        price = boundary_price(rng, qty, bps) if i % 3 == 0 else int(rng.integers(1, 5000))
        t = trade(price, qty, Side.BUY if rng.random() < 0.5 else Side.SELL)
        assert fast.trade_fees(t) == oracle.trade_fees(t), (price, qty, maker, taker)
        for f in fills_of(t):
            assert fast.charge(f) == oracle.charge(f) == fast.charge_fraction(f)
        assert fast.round_trip(price, qty) == oracle.round_trip(price, qty)


def test_rebate_rounds_toward_zero_and_fee_rounds_up():
    fast, oracle = pair(-1, 5)
    for price, qty in ((100, 10), (1, 1), (9_999, 1), (10_001, 1), (3, 3333), (7, 1429)):
        for side in Side:
            t = trade(price, qty, side)
            assert fast.trade_fees(t) == oracle.trade_fees(t)
    b, s = fast.trade_fees(trade(100, 10, Side.BUY))   # +0.5 -> 1, -0.1 -> 0
    assert (b, s) == (1, 0)


@pytest.mark.parametrize("maker,taker", [(Fraction(1, 2), 3), (0.1, 2.5), (-0.7, 1.3),
                                         (2, Fraction(7, 3))])
def test_non_integer_rates_fall_back_with_identical_results(maker, taker):
    fast, oracle = pair(maker, taker, 1, spread=Fraction(3, 2), latency=(0.25, 3))
    assert fast._plan(Role.TAKER)[1] or fast._plan(Role.MAKER)[1]
    rng = np.random.default_rng(11)
    for _ in range(400):
        t = trade(int(rng.integers(1, 20_000)), int(rng.integers(1, 30)),
                  Side.BUY if rng.random() < 0.5 else Side.SELL)
        assert fast.trade_fees(t) == oracle.trade_fees(t)


def test_mixed_slow_components_keep_exact_single_rounding():
    gas = GasProcess(1, 9, 0.3, 0.4, seed=3).cost()
    tiers = FeeComponent(FeeTierSchedule([FeeTier(0, -1, 7), FeeTier(50, -2, 4)]))
    fast = FrictionStack([MakerTakerFee(-1, 3), FixedCost(1), gas])
    oracle = ref.ReferenceStack([ref.ReferenceMakerTakerFee(-1, 3), ref.ReferenceFixedCost(1), gas])
    lone = FrictionStack([tiers, HalfSpreadCost(1)])
    lone_ref = ref.ReferenceStack([tiers, ref.ReferenceHalfSpreadCost(1)])
    rng = np.random.default_rng(5)
    for k in range(300):
        t = Trade(1, float(k), int(rng.integers(1, 9_000)), int(rng.integers(1, 9)),
                  10, 11, 1, 2, Side.SELL if k % 2 else Side.BUY)
        assert fast.trade_fees(t) == oracle.trade_fees(t)
        assert lone.trade_fees(t) == lone_ref.trade_fees(t)


def test_custom_subclass_and_direct_mutation_stay_correct():
    class Doubled(MakerTakerFee):            # overrides raw_cost: must not be fast-pathed
        def raw_cost(self, fill):
            return 2 * super().raw_cost(fill)

    fill = Fill(1, Side.BUY, Role.TAKER, 1_234, 7)
    st = FrictionStack([Doubled(0, 10)])
    assert st.charge(fill) == math.ceil(Fraction(2 * 10 * 1_234 * 7, 10_000))
    st = FrictionStack([MakerTakerFee(0, 10)])
    st.models.append(FixedCost(5))           # legacy direct mutation recompiles the plan
    assert st.charge(fill) == ref.ReferenceStack(st.models).charge(fill)


def test_div_round_matches_fraction_ceil_and_floor():
    rng = np.random.default_rng(2)
    for _ in range(2000):
        num, den = int(rng.integers(-10**9, 10**9)), int(rng.integers(1, 10**5))
        assert div_round(num, den) == math.ceil(Fraction(num, den))
        assert div_round(num, den, "floor") == math.floor(Fraction(num, den))
    with pytest.raises(ValueError):
        div_round(1, 0)
    with pytest.raises(ValueError):
        div_round(1, 2, "half_even")


def test_validation_errors_unchanged():
    for cls in (MakerTakerFee, ref.ReferenceMakerTakerFee):
        with pytest.raises(ValueError, match="taker_bps must be >= 0"):
            cls(0, -1)
        with pytest.raises(ValueError, match="maker rebate exceeds"):
            cls(-3, 2)
        with pytest.raises(TypeError):
            cls(True, 1)
    assert isinstance(MakerTakerFee(), CostModel)
