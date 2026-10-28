"""Order / Trade / Account primitives with integer-tick prices.

Units contract (exact arithmetic, no float drift):
- price: integer number of ticks (tick size lives at the venue, see `to_ticks`).
- qty:   integer number of lots.
- cash / fees: integer tick-lots (price * qty), so settlement is exactly conserved.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from numbers import Integral
from typing import Optional


def _check_int(name: str, v, *, positive: bool = False, nonneg: bool = False) -> int:
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise TypeError(f"{name} must be an integer, got {type(v).__name__}")
    v = int(v)
    if positive and v <= 0:
        raise ValueError(f"{name} must be > 0, got {v}")
    if nonneg and v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


def to_ticks(price: float, tick_size: float) -> int:
    """Snap a float price onto the tick grid (round half to even)."""
    if tick_size <= 0:
        raise ValueError(f"tick_size must be > 0, got {tick_size}")
    return int(round(price / tick_size))


def from_ticks(ticks: int, tick_size: float) -> float:
    return _check_int("ticks", ticks) * float(tick_size)


class Side(IntEnum):
    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> "Side":
        return Side(-self.value)


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


class IdGen:
    """Monotone id source; one per simulation keeps ids reproducible."""

    def __init__(self, start: int = 1) -> None:
        self._it = itertools.count(start)

    def __call__(self) -> int:
        return next(self._it)


@dataclass(eq=False)
class Order:
    order_id: int
    agent_id: int
    side: Side
    qty: int
    price: Optional[int] = None  # ticks; None for market orders
    time: float = 0.0
    otype: OrderType = OrderType.LIMIT
    filled: int = 0
    cancelled: bool = False

    def __post_init__(self) -> None:
        self.side = Side(self.side)
        self.otype = OrderType(self.otype)
        self.qty = _check_int("qty", self.qty, positive=True)
        if self.otype is OrderType.LIMIT:
            if self.price is None:
                raise ValueError("limit order requires a price")
            self.price = _check_int("price", self.price, positive=True)
        elif self.price is not None:
            raise ValueError("market order must not carry a price")
        self.filled = _check_int("filled", self.filled, nonneg=True)
        if self.filled > self.qty:
            raise ValueError("filled exceeds qty")

    @property
    def remaining(self) -> int:
        return 0 if self.cancelled else self.qty - self.filled

    @property
    def is_active(self) -> bool:
        return self.remaining > 0

    def fill(self, qty: int) -> int:
        """Consume `qty` lots; returns lots still open."""
        qty = _check_int("fill qty", qty, positive=True)
        if not self.is_active:
            raise ValueError(f"order {self.order_id} is not active")
        if qty > self.remaining:
            raise ValueError(f"overfill: {qty} > remaining {self.remaining}")
        self.filled += qty
        return self.remaining

    def cancel(self) -> int:
        """Cancel the open remainder; returns lots cancelled (0 if none)."""
        left = self.remaining
        if left:
            self.cancelled = True
        return left

    def crosses(self, price: int) -> bool:
        """Would this order trade against a resting quote at `price` ticks?"""
        if self.price is None:
            return True
        return price <= self.price if self.side is Side.BUY else price >= self.price


@dataclass(frozen=True)
class Trade:
    trade_id: int
    time: float
    price: int
    qty: int
    buy_order_id: int
    sell_order_id: int
    buyer_id: int
    seller_id: int
    aggressor: Side

    def __post_init__(self) -> None:
        _check_int("price", self.price, positive=True)
        _check_int("qty", self.qty, positive=True)
        if self.buyer_id == self.seller_id:
            raise ValueError("self-trade: buyer and seller are the same agent")

    @property
    def notional(self) -> int:
        return self.price * self.qty


@dataclass
class Account:
    agent_id: int
    cash: int = 0
    position: int = 0
    fees_paid: int = 0
    volume: int = 0
    n_trades: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.cash = _check_int("cash", self.cash)
        self.position = _check_int("position", self.position)

    def apply_fill(self, side: Side, price: int, qty: int, fee: int = 0) -> None:
        """Book one fill. `fee` may be negative (maker rebate)."""
        side = Side(side)
        price = _check_int("price", price, positive=True)
        qty = _check_int("qty", qty, positive=True)
        fee = _check_int("fee", fee)
        self.position += side.value * qty
        self.cash -= side.value * price * qty + fee
        self.fees_paid += fee
        self.volume += qty
        self.n_trades += 1

    def equity(self, mark: int) -> int:
        """Mark-to-market value in tick-lots at `mark` ticks."""
        return self.cash + self.position * _check_int("mark", mark, nonneg=True)


def settle(trade: Trade, buyer: Account, seller: Account,
           buyer_fee: int = 0, seller_fee: int = 0) -> None:
    """Apply a trade to both counterparties.

    Conservation: sum(cash) changes by exactly -(buyer_fee + seller_fee) and
    sum(position) is unchanged.
    """
    if buyer.agent_id != trade.buyer_id or seller.agent_id != trade.seller_id:
        raise ValueError("accounts do not match trade counterparties")
    buyer.apply_fill(Side.BUY, trade.price, trade.qty, buyer_fee)
    seller.apply_fill(Side.SELL, trade.price, trade.qty, seller_fee)
