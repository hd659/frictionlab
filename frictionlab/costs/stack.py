"""FrictionStack: compose friction knobs and apply them to trades.

Rounding policy: component charges are summed as exact rationals and rounded
once per fill (venue-favourable ceil). Rounding each component separately
would let N tiny fees inflate a fill cost by up to N-1 tick-lots.

Latency composes additively: `stack.delay` is the sum of component delays.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from ..core.types import Account, Side, Trade, settle
from .base import CostModel, Fill, Role, fills_of, venue_round


class FrictionStack:
    def __init__(self, models: Iterable[CostModel] = ()) -> None:
        self.models: List[CostModel] = []
        for m in models:
            self.add(m)

    # ---- composition -------------------------------------------------
    def add(self, model: CostModel) -> "FrictionStack":
        if not isinstance(model, CostModel):
            raise TypeError(f"expected CostModel, got {type(model).__name__}")
        if any(m.name == model.name for m in self.models):
            raise ValueError(f"duplicate friction component {model.name!r}")
        self.models.append(model)
        return self

    def __add__(self, other) -> "FrictionStack":
        out = FrictionStack(self.models)
        others = other.models if isinstance(other, FrictionStack) else [other]
        for m in others:
            out.add(m)
        return out

    def without(self, name: str) -> "FrictionStack":
        """Copy with one knob removed (for counterfactual friction runs)."""
        if name not in self:
            raise KeyError(name)
        return FrictionStack(m for m in self.models if m.name != name)

    def __contains__(self, name: str) -> bool:
        return any(m.name == name for m in self.models)

    def __getitem__(self, name: str) -> CostModel:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)

    def __iter__(self) -> Iterator[CostModel]:
        return iter(self.models)

    def __len__(self) -> int:
        return len(self.models)

    def __repr__(self) -> str:
        return f"FrictionStack({self.models!r})"

    # ---- evaluation --------------------------------------------------
    @property
    def delay(self) -> float:
        return float(sum(Fraction(m.delay) for m in self.models))

    def breakdown(self, fill: Fill) -> Dict[str, Fraction]:
        """Exact (unrounded) charge per component for one fill."""
        return {m.name: (m.raw_cost(fill) if m.applies(fill) else Fraction(0))
                for m in self.models}

    def charge(self, fill: Fill) -> int:
        return venue_round(sum(self.breakdown(fill).values(), Fraction(0)))

    def trade_fees(self, trade: Trade) -> Tuple[int, int]:
        """(buyer_fee, seller_fee) in tick-lots for one trade."""
        buy, sell = fills_of(trade)
        return self.charge(buy), self.charge(sell)

    def settle(self, trade: Trade, accounts: Mapping[int, Account]) -> Tuple[int, int]:
        """Book a trade with frictions into `accounts` (keyed by agent_id)."""
        bf, sf = self.trade_fees(trade)
        settle(trade, accounts[trade.buyer_id], accounts[trade.seller_id], bf, sf)
        return bf, sf

    def settle_all(self, trades: Iterable[Trade], accounts: Mapping[int, Account]) -> int:
        """Settle a batch; returns total friction collected (venue revenue)."""
        return sum(sum(self.settle(t, accounts)) for t in trades)

    def round_trip(self, price: int, qty: int, role: Role = Role.TAKER,
                   exit_price: Optional[int] = None) -> int:
        """Cost of entering and exiting `qty` lots in the same role.

        This is the minimum gross edge (tick-lots) a strategy must clear
        before trading is worth it -- the Coase threshold for one agent.
        """
        exit_price = price if exit_price is None else exit_price
        enter = Fill(0, Side.BUY, role, price, qty)
        leave = Fill(0, Side.SELL, role, exit_price, qty)
        return self.charge(enter) + self.charge(leave)

    def breakeven_ticks(self, price: int, qty: int = 1, role: Role = Role.TAKER) -> Fraction:
        """Round-trip friction expressed as ticks of price move per lot."""
        return Fraction(self.round_trip(price, qty, role), qty)


def zero_friction() -> FrictionStack:
    """Frictionless baseline: every charge is 0 and delay is 0."""
    return FrictionStack()
