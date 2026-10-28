from fractions import Fraction

import numpy as np
import pytest

from frictionlab.core.order_book import OrderBook
from frictionlab.core.types import Account, IdGen, Order, OrderType, Side, Trade
from frictionlab.costs import (
    BOTH, CostModel, Fill, FixedCost, FrictionStack, HalfSpreadCost, LatencyCost,
    MakerTakerFee, Role, fills_of, to_frac, venue_round, zero_friction,
)


def trade(price=1000, qty=20, aggressor=Side.BUY, buyer=1, seller=2):
    return Trade(1, 0.0, price, qty, 10, 11, buyer, seller, aggressor)


class Const(CostModel):
    def __init__(self, name, amount, roles=BOTH):
        super().__init__(roles)
        self.name, self.amount = name, Fraction(amount)

    def raw_cost(self, fill):
        return self.amount


def test_fills_of_assigns_roles_by_aggressor():
    b, s = fills_of(trade(aggressor=Side.BUY))
    assert (b.role, s.role) == (Role.TAKER, Role.MAKER)
    assert (b.side, s.side, b.agent_id, s.agent_id) == (Side.BUY, Side.SELL, 1, 2)
    b, s = fills_of(trade(aggressor=Side.SELL))
    assert (b.role, s.role) == (Role.MAKER, Role.TAKER)
    assert b.notional == s.notional == 20_000


def test_to_frac_and_venue_round():
    assert to_frac(0.1) == Fraction(1, 10)
    assert to_frac(3) == 3 and to_frac(Fraction(1, 3)) == Fraction(1, 3)
    for bad in (True, "1", None):
        with pytest.raises(TypeError):
            to_frac(bad)
    with pytest.raises(ValueError):
        to_frac(float("nan"))
    assert venue_round(Fraction(1, 2)) == 1
    assert venue_round(Fraction(-1, 2)) == 0
    assert venue_round(Fraction(-5, 2)) == -2


def test_maker_taker_fee_with_rebate():
    fee = MakerTakerFee(maker_bps=-1, taker_bps=5)
    b, s = fills_of(trade())
    assert fee.charge(b) == 10       # 5 bps of 20000
    assert fee.charge(s) == -2       # 1 bps rebate
    tiny = fills_of(trade(price=100, qty=10))  # notional 1000
    assert fee.charge(tiny[0]) == 1  # 0.5 -> ceil 1 (venue favourable)
    assert fee.charge(tiny[1]) == 0  # -0.1 -> 0 (rebate rounds toward zero)


def test_maker_taker_validation():
    with pytest.raises(ValueError):
        MakerTakerFee(taker_bps=-1)
    with pytest.raises(ValueError):
        MakerTakerFee(maker_bps=-3, taker_bps=2)
    assert MakerTakerFee(taker_bps=2, roles=Role.MAKER).charge(fills_of(trade())[0]) == 0


def test_half_spread_and_fixed_cost():
    b, s = fills_of(trade())
    hs = HalfSpreadCost(spread_ticks=3)
    assert hs.half_spread == Fraction(3, 2)
    assert hs.charge(b) == 30 and hs.charge(s) == 0  # taker-only by default
    fx = FixedCost(7)
    assert fx.charge(b) == fx.charge(s) == 7
    with pytest.raises(ValueError):
        HalfSpreadCost(-1)
    with pytest.raises(ValueError):
        FixedCost(-1)


def test_latency_cost_and_delay():
    lat = LatencyCost(delay=0.5, drift_ticks_per_sec=2)
    b, s = fills_of(trade())
    assert lat.delay == 0.5
    assert lat.charge(b) == 20 and lat.charge(s) == 0
    assert LatencyCost(delay=0.25).charge(b) == 0
    with pytest.raises(ValueError):
        LatencyCost(delay=-1)


def test_stack_composition_and_breakdown():
    stack = MakerTakerFee(-1, 5) + HalfSpreadCost(2) + FixedCost(3) + LatencyCost(0.5, 2)
    assert isinstance(stack, FrictionStack) and len(stack) == 4
    assert "latency" in stack and stack["fixed"].per_trade == 3
    b, s = fills_of(trade())
    bd = stack.breakdown(b)
    assert bd == {"maker_taker": 10, "half_spread": 20, "fixed": 3, "latency": 20}
    assert stack.trade_fees(trade()) == (53, 1)
    assert stack.delay == 0.5
    lean = stack.without("half_spread")
    assert len(lean) == 3 and lean.trade_fees(trade()) == (33, 1)
    with pytest.raises(ValueError):
        stack + FixedCost(1)
    with pytest.raises(KeyError):
        stack.without("nope")
    with pytest.raises(TypeError):
        FrictionStack([object()])


def test_stack_rounds_once_per_fill():
    stack = FrictionStack([Const("a", Fraction(3, 10)), Const("b", Fraction(3, 10)),
                           Const("c", Fraction(3, 10))])
    b, _ = fills_of(trade())
    assert sum(m.charge(b) for m in stack) == 3  # per-component rounding inflates
    assert stack.charge(b) == 1                   # single rounding: ceil(0.9)


def test_zero_friction_is_frictionless():
    z = zero_friction()
    assert z.trade_fees(trade()) == (0, 0) and z.delay == 0.0
    assert z.round_trip(1000, 5) == 0


def test_settle_conserves_cash_and_inventory():
    stack = MakerTakerFee(-1, 5) + FixedCost(2)
    accts = {1: Account(1, cash=1_000_000), 2: Account(2, cash=1_000_000)}
    bf, sf = stack.settle(trade(), accts)
    assert (bf, sf) == (12, 0)
    assert accts[1].position == 20 and accts[2].position == -20
    assert accts[1].cash == 1_000_000 - 20_000 - 12
    assert sum(a.cash for a in accts.values()) == 2_000_000 - (bf + sf)
    assert accts[1].fees_paid == 12 and accts[2].fees_paid == 0


def test_round_trip_and_breakeven():
    stack = MakerTakerFee(0, 5) + HalfSpreadCost(2) + FixedCost(4)
    # enter: 10 + 20 + 4, exit: 10 + 20 + 4
    assert stack.round_trip(1000, 20) == 68
    assert stack.breakeven_ticks(1000, 20) == Fraction(68, 20)
    assert stack.round_trip(1000, 20, role=Role.MAKER) == 8


def test_integration_with_order_book_seeded():
    rng = np.random.default_rng(7)
    ids, book = IdGen(), OrderBook()
    stack = MakerTakerFee(-1, 4) + HalfSpreadCost(1) + FixedCost(1)
    accts = {a: Account(a, cash=10**9) for a in range(1, 6)}
    trades = []
    for _ in range(300):
        agent = int(rng.integers(1, 6))
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        qty = int(rng.integers(1, 10))
        if rng.random() < 0.3:
            o = Order(ids(), agent, side, qty, otype=OrderType.MARKET)
        else:
            o = Order(ids(), agent, side, qty, price=int(rng.integers(95, 106)))
        trades += book.submit(o)
    assert trades
    collected = stack.settle_all(trades, accts)
    assert collected > 0
    assert sum(a.position for a in accts.values()) == 0
    assert sum(a.cash for a in accts.values()) == 5 * 10**9 - collected
    assert sum(a.fees_paid for a in accts.values()) == collected
