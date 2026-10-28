"""Concrete friction knobs.

All charges are in tick-lots (price_ticks * lots), matching `Account.cash`.

- MakerTakerFee : proportional fee in basis points of notional, per role.
                  A negative maker rate is a rebate (inverted / rebate venues).
- HalfSpreadCost: implicit cost of crossing a spread of `spread_ticks`;
                  charges spread/2 ticks per lot (taker-only by default).
- LatencyCost   : `delay` seconds before an order is live, plus an expected
                  adverse move of `drift_ticks_per_sec * delay` per lot.
- FixedCost     : flat charge per fill (gas, ticket fee, clearing fee).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Union

from .base import BOTH, TAKER_ONLY, CostModel, Fill, Number, Role, to_frac

BPS = Fraction(1, 10_000)
Roles = Union[Role, Iterable[Role], str]


class MakerTakerFee(CostModel):
    name = "maker_taker"

    def __init__(self, maker_bps: Number = 0, taker_bps: Number = 0,
                 roles: Roles = BOTH) -> None:
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

    def __repr__(self) -> str:
        return f"MakerTakerFee(maker_bps={self.maker_bps}, taker_bps={self.taker_bps})"


class HalfSpreadCost(CostModel):
    name = "half_spread"

    def __init__(self, spread_ticks: Number, roles: Roles = TAKER_ONLY) -> None:
        super().__init__(roles)
        self.spread_ticks = to_frac(spread_ticks, "spread_ticks")
        if self.spread_ticks < 0:
            raise ValueError("spread_ticks must be >= 0")

    @property
    def half_spread(self) -> Fraction:
        return self.spread_ticks / 2

    def raw_cost(self, fill: Fill) -> Fraction:
        return self.half_spread * fill.qty

    def __repr__(self) -> str:
        return f"HalfSpreadCost(spread_ticks={self.spread_ticks})"


class LatencyCost(CostModel):
    name = "latency"

    def __init__(self, delay: Number, drift_ticks_per_sec: Number = 0,
                 roles: Roles = TAKER_ONLY) -> None:
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

    def __repr__(self) -> str:
        return f"LatencyCost(delay={float(self._delay)}, drift_ticks_per_sec={self.drift})"


class FixedCost(CostModel):
    name = "fixed"

    def __init__(self, per_trade: int, roles: Roles = BOTH) -> None:
        super().__init__(roles)
        self.per_trade = to_frac(per_trade, "per_trade")
        if self.per_trade < 0:
            raise ValueError("per_trade must be >= 0")

    def raw_cost(self, fill: Fill) -> Fraction:
        return self.per_trade

    def __repr__(self) -> str:
        return f"FixedCost(per_trade={self.per_trade})"
