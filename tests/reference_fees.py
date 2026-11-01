"""FROZEN ORACLE: the pre-fl-040 Fraction fee computation, verbatim.

Do not optimise or "fix" this file. tests/test_fee_fast_path.py replays
seeded random fills through these classes and through
`frictionlab.costs.FrictionStack`; any integer that differs is a regression
in the fast path. Only the class names differ from the original
(`Reference*` prefix). Every charge is an exact `Fraction`, summed per fill
and rounded once with `venue_round` (ceil: costs up, rebates toward zero).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Tuple

from frictionlab.core.types import Side, Trade
from frictionlab.costs.base import (BOTH, TAKER_ONLY, CostModel, Fill, Number, Role,
                                    fills_of, to_frac, venue_round)

BPS = Fraction(1, 10_000)


class ReferenceMakerTakerFee(CostModel):
    name = "maker_taker"

    def __init__(self, maker_bps: Number = 0, taker_bps: Number = 0, roles=BOTH) -> None:
        super().__init__(roles)
        self.maker_bps = to_frac(maker_bps, "maker_bps")
        self.taker_bps = to_frac(taker_bps, "taker_bps")
        if self.taker_bps < 0:
            raise ValueError("taker_bps must be >= 0")
        if self.maker_bps + self.taker_bps < 0:
            raise ValueError("maker rebate exceeds taker fee: venue would lose money per trade")

    def rate(self, role: Role) -> Fraction:
        return (self.maker_bps if role is Role.MAKER else self.taker_bps) * BPS

    def raw_cost(self, fill: Fill) -> Fraction:
        return self.rate(fill.role) * fill.notional


class ReferenceHalfSpreadCost(CostModel):
    name = "half_spread"

    def __init__(self, spread_ticks: Number, roles=TAKER_ONLY) -> None:
        super().__init__(roles)
        self.spread_ticks = to_frac(spread_ticks, "spread_ticks")
        if self.spread_ticks < 0:
            raise ValueError("spread_ticks must be >= 0")

    @property
    def half_spread(self) -> Fraction:
        return self.spread_ticks / 2

    def raw_cost(self, fill: Fill) -> Fraction:
        return self.half_spread * fill.qty


class ReferenceLatencyCost(CostModel):
    name = "latency"

    def __init__(self, delay: Number, drift_ticks_per_sec: Number = 0, roles=TAKER_ONLY) -> None:
        super().__init__(roles)
        self._delay = to_frac(delay, "delay")
        self.drift = to_frac(drift_ticks_per_sec, "drift_ticks_per_sec")
        if self._delay < 0 or self.drift < 0:
            raise ValueError("delay and drift_ticks_per_sec must be >= 0")

    @property
    def delay(self) -> float:
        return float(self._delay)

    def raw_cost(self, fill: Fill) -> Fraction:
        return self.drift * self._delay * fill.qty


class ReferenceFixedCost(CostModel):
    name = "fixed"

    def __init__(self, per_trade: int, roles=BOTH) -> None:
        super().__init__(roles)
        self.per_trade = to_frac(per_trade, "per_trade")
        if self.per_trade < 0:
            raise ValueError("per_trade must be >= 0")

    def raw_cost(self, fill: Fill) -> Fraction:
        return self.per_trade


class ReferenceStack:
    """The pre-fl-040 `FrictionStack` evaluation path (no composition sugar)."""

    def __init__(self, models: Iterable[CostModel] = ()) -> None:
        self.models: List[CostModel] = list(models)

    def breakdown(self, fill: Fill) -> Dict[str, Fraction]:
        return {m.name: (m.raw_cost(fill) if m.applies(fill) else Fraction(0))
                for m in self.models}

    def charge(self, fill: Fill) -> int:
        return venue_round(sum(self.breakdown(fill).values(), Fraction(0)))

    def trade_fees(self, trade: Trade) -> Tuple[int, int]:
        buy, sell = fills_of(trade)
        return self.charge(buy), self.charge(sell)

    def round_trip(self, price: int, qty: int, role: Role = Role.TAKER,
                   exit_price: Optional[int] = None) -> int:
        exit_price = price if exit_price is None else exit_price
        enter = Fill(0, Side.BUY, role, price, qty)
        leave = Fill(0, Side.SELL, role, exit_price, qty)
        return self.charge(enter) + self.charge(leave)
