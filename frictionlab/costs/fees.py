"""Concrete friction knobs.

All charges are in tick-lots (price_ticks * lots), matching `Account.cash`.

- MakerTakerFee : proportional fee in basis points of notional, per role.
                  A negative maker rate is a rebate (inverted / rebate venues).
- HalfSpreadCost: implicit cost of crossing a spread of `spread_ticks`;
                  charges spread/2 ticks per lot (taker-only by default).
- LatencyCost   : `delay` seconds before an order is live, plus an expected
                  adverse move of `drift_ticks_per_sec * delay` per lot.
- FixedCost     : flat charge per fill (gas, ticket fee, clearing fee).

Integer fast path (fl-040)
Every knob here is linear in the fill: raw_cost = (a*notional + b*qty + c) / den.
When its parameters are integers (integer bps, integer spread ticks, ...),
`int_terms(role)` returns that `(a, b, c, den)` tuple of ints and the charge is
one exact integer division (`div_round`, same ceil as `venue_round`). With a
non-integer parameter (e.g. 0.5 bps) `int_terms` returns None and the charge
falls back to the exact `Fraction` path. Both paths give identical integers;
tests/reference_fees.py freezes the pre-fl-040 Fraction code as the oracle.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Optional, Tuple, Union

from .base import BOTH, TAKER_ONLY, CostModel, Fill, Number, Role, to_frac, venue_round

BPS = Fraction(1, 10_000)
BPS_DEN = 10_000
Roles = Union[Role, Iterable[Role], str]
Terms = Tuple[int, int, int, int]          # (per_notional, per_lot, per_fill, den)
ZERO_TERMS: Terms = (0, 0, 0, 1)
ROUNDING = "ceil"                          # must match base.venue_round


def div_round(num: int, den: int, mode: str = ROUNDING) -> int:
    """Exact integer num/den rounded like `math.ceil` / `math.floor` of a Fraction."""
    if den <= 0:
        raise ValueError(f"den must be > 0, got {den}")
    if mode == "ceil":
        return -(-num // den)
    if mode == "floor":
        return num // den
    raise ValueError(f"unknown rounding mode {mode!r}")


def eval_terms(t: Terms, notional: int, qty: int) -> int:
    """Rounded charge of linear terms `t` for one fill."""
    a, b, c, d = t
    return -(-(a * notional + b * qty + c) // d)


def _int_or_none(x: Fraction) -> Optional[int]:
    return x.numerator if x.denominator == 1 else None


class LinearCost(CostModel):
    """A knob whose charge is linear in (notional, qty, 1) per role."""

    def _terms(self, role: Role) -> Optional[Terms]:
        raise NotImplementedError

    def int_terms(self, role: Role) -> Optional[Terms]:
        """Integer terms for `role`, ZERO_TERMS if not covered, None if not exact-int.

        Subclasses outside this module may override `raw_cost`, so only the
        exact built-in types take the fast path; anything else stays on Fraction.
        """
        if type(self) not in FAST_TYPES:
            return None
        return self._terms(role) if role in self.roles else ZERO_TERMS

    def charge(self, fill: Fill) -> int:
        t = self.int_terms(fill.role)
        if t is None:
            return venue_round(self.raw_cost(fill)) if self.applies(fill) else 0
        return eval_terms(t, int(fill.price) * int(fill.qty), int(fill.qty))


class MakerTakerFee(LinearCost):
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

    def bps(self, role: Role) -> Fraction:
        return self.maker_bps if role is Role.MAKER else self.taker_bps

    def rate(self, role: Role) -> Fraction:
        return self.bps(role) * BPS

    def raw_cost(self, fill: Fill) -> Fraction:
        return self.rate(fill.role) * fill.notional

    def _terms(self, role: Role) -> Optional[Terms]:
        k = _int_or_none(self.bps(role))
        return None if k is None else (k, 0, 0, BPS_DEN)

    def __repr__(self) -> str:
        return f"MakerTakerFee(maker_bps={self.maker_bps}, taker_bps={self.taker_bps})"


class HalfSpreadCost(LinearCost):
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

    def _terms(self, role: Role) -> Optional[Terms]:
        k = _int_or_none(self.spread_ticks)
        return None if k is None else (0, k, 0, 2)

    def __repr__(self) -> str:
        return f"HalfSpreadCost(spread_ticks={self.spread_ticks})"


class LatencyCost(LinearCost):
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

    def _terms(self, role: Role) -> Optional[Terms]:
        k = _int_or_none(self.drift * self._delay)
        return None if k is None else (0, k, 0, 1)

    def __repr__(self) -> str:
        return f"LatencyCost(delay={float(self._delay)}, drift_ticks_per_sec={self.drift})"


class FixedCost(LinearCost):
    name = "fixed"

    def __init__(self, per_trade: int, roles: Roles = BOTH) -> None:
        super().__init__(roles)
        self.per_trade = to_frac(per_trade, "per_trade")
        if self.per_trade < 0:
            raise ValueError("per_trade must be >= 0")

    def raw_cost(self, fill: Fill) -> Fraction:
        return self.per_trade

    def _terms(self, role: Role) -> Optional[Terms]:
        k = _int_or_none(self.per_trade)
        return None if k is None else (0, 0, k, 1)

    def __repr__(self) -> str:
        return f"FixedCost(per_trade={self.per_trade})"


FAST_TYPES = frozenset({MakerTakerFee, HalfSpreadCost, LatencyCost, FixedCost})
