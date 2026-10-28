"""Friction interface: per-fill cost models in integer tick-lots.

Contract
- A `Trade` has two legs. The aggressor leg is the TAKER, the resting leg the
  MAKER. `fills_of(trade)` splits a trade into these two `Fill` views.
- A `CostModel` maps one `Fill` to an integer charge in tick-lots (the same
  unit as `Account.cash`). Positive = agent pays, negative = rebate.
- Rounding is venue-favourable: `venue_round` is ceil, so a fractional cost
  rounds up and a fractional rebate rounds toward zero. Arithmetic uses
  `Fraction` so `0.1 bps` means exactly 1/100000, not a binary float.
- A model may also declare a `delay` (seconds) that the simulator should add
  before the order reaches the book; it is the time dimension of friction.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from numbers import Integral, Real
from typing import FrozenSet, Iterable, Tuple, Union

from ..core.types import Side, Trade

Number = Union[int, float, Fraction]


class Role(Enum):
    MAKER = "maker"
    TAKER = "taker"


BOTH: FrozenSet[Role] = frozenset({Role.MAKER, Role.TAKER})
TAKER_ONLY: FrozenSet[Role] = frozenset({Role.TAKER})


def to_frac(x: Number, name: str = "value") -> Fraction:
    """Exact rational from int / Fraction / float (floats via their repr)."""
    if isinstance(x, bool) or not isinstance(x, (Real, Fraction)):
        raise TypeError(f"{name} must be a real number, got {type(x).__name__}")
    if isinstance(x, (Integral, Fraction)):
        return Fraction(x)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}")
    return Fraction(repr(float(x)))


def venue_round(x: Fraction) -> int:
    """Round a charge in the venue's favour (ceil): costs up, rebates down."""
    return math.ceil(x)


def as_roles(roles: Union[Role, Iterable[Role], str]) -> FrozenSet[Role]:
    if isinstance(roles, (Role, str)):
        roles = [roles]
    out = frozenset(Role(r) for r in roles)
    if not out:
        raise ValueError("roles must not be empty")
    return out


@dataclass(frozen=True)
class Fill:
    """One counterparty's view of a trade."""

    agent_id: int
    side: Side
    role: Role
    price: int
    qty: int
    time: float = 0.0

    @property
    def notional(self) -> int:
        return self.price * self.qty


def fills_of(trade: Trade) -> Tuple[Fill, Fill]:
    """Return (buyer_fill, seller_fill) with maker/taker roles assigned."""
    buy_role = Role.TAKER if trade.aggressor is Side.BUY else Role.MAKER
    sell_role = Role.MAKER if buy_role is Role.TAKER else Role.TAKER
    mk = lambda aid, side, role: Fill(aid, side, role, trade.price, trade.qty, trade.time)
    return (mk(trade.buyer_id, Side.BUY, buy_role),
            mk(trade.seller_id, Side.SELL, sell_role))


class CostModel(ABC):
    """One friction knob. Subclasses implement `raw_cost` in exact rationals."""

    name: str = "cost"

    def __init__(self, roles: Union[Role, Iterable[Role], str] = BOTH) -> None:
        self.roles = as_roles(roles)

    @property
    def delay(self) -> float:
        """Seconds of latency this friction adds before an order is live."""
        return 0.0

    def applies(self, fill: Fill) -> bool:
        return fill.role in self.roles

    @abstractmethod
    def raw_cost(self, fill: Fill) -> Fraction:
        """Unrounded charge in tick-lots for a fill this model applies to."""

    def charge(self, fill: Fill) -> int:
        """Integer charge in tick-lots (0 if the role is not covered)."""
        return venue_round(self.raw_cost(fill)) if self.applies(fill) else 0

    def __add__(self, other):
        from .stack import FrictionStack
        return FrictionStack([self]) + other

    def __repr__(self) -> str:
        roles = ",".join(sorted(r.value for r in self.roles))
        return f"{type(self).__name__}(name={self.name!r}, roles={roles})"
