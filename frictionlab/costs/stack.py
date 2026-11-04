"""FrictionStack: compose friction knobs and apply them to trades.

Rounding policy: component charges are summed as exact rationals and rounded
once per fill (venue-favourable ceil). Rounding each component separately
would let N tiny fees inflate a fill cost by up to N-1 tick-lots.

Fast path (fl-040): at construction (and on `add`) the stack folds every
integer-parameter `LinearCost` into one integer term tuple per role,
`(A, B, C, D)`, so a fill costs `ceil((A*notional + B*qty + C) / D)` in pure
int arithmetic: the same exact sum, rounded once. Components without integer
terms (non-integer bps, gas, tiered fees, custom models) stay on the Fraction
path and are added to the integer part as an exact rational before rounding.
Knob parameters are treated as immutable after construction.

Latency composes additively: `stack.delay` is the sum of component delays.
"""
from __future__ import annotations

from fractions import Fraction
from math import lcm
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from ..core.types import Account, Side, Trade, settle
from .base import CostModel, Fill, Role, fills_of, venue_round
from .fees import LinearCost, Terms, eval_terms

Plan = Tuple[Terms, Tuple[CostModel, ...]]   # (folded int terms, Fraction-path models)


def fold_terms(terms: Iterable[Terms]) -> Terms:
    """Sum linear terms exactly over a common denominator."""
    terms = list(terms)
    den = lcm(*(t[3] for t in terms)) if terms else 1
    a = b = c = 0
    for ta, tb, tc, td in terms:
        k = den // td
        a, b, c = a + ta * k, b + tb * k, c + tc * k
    return a, b, c, den


class FrictionStack:
    """An ordered set of `CostModel`s summed per fill and rounded once per trade.

    Implements the ledger's `FeeSource` protocol; integer-bps components use
    an exact integer fast path folded once per stack.
    """
    def __init__(self, models: Iterable[CostModel] = ()) -> None:
        self.models: List[CostModel] = []
        self._plans: Dict[Role, Plan] = {}
        self._planned = -1
        for m in models:
            self.add(m)
        self._compile()

    # ---- composition -------------------------------------------------
    def add(self, model: CostModel) -> "FrictionStack":
        if not isinstance(model, CostModel):
            raise TypeError(f"expected CostModel, got {type(model).__name__}")
        if any(m.name == model.name for m in self.models):
            raise ValueError(f"duplicate friction component {model.name!r}")
        self.models.append(model)
        self._planned = -1
        return self

    def _compile(self) -> None:
        """Precompute per-role (int terms, slow models) once, not per trade."""
        for role in Role:
            fast, slow = [], []
            for m in self.models:
                t = m.int_terms(role) if isinstance(m, LinearCost) else None
                (slow if t is None else fast).append(m if t is None else t)
            self._plans[role] = (fold_terms(fast), tuple(slow))
        self._planned = len(self.models)

    def _plan(self, role: Role) -> Plan:
        if self._planned != len(self.models):   # `models` mutated directly
            self._compile()
        return self._plans[role]

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
        (a, b, c, d), slow = self._plan(fill.role)
        q = int(fill.qty)
        num = a * int(fill.price) * q + b * q + c
        if not slow:
            return -(-num // d)
        extra = sum((m.raw_cost(fill) for m in slow if m.applies(fill)), Fraction(0))
        return venue_round(Fraction(num, d) + extra)

    def charge_fraction(self, fill: Fill) -> int:
        """Pre-fl-040 reference: round the exact Fraction sum of `breakdown`."""
        return venue_round(sum(self.breakdown(fill).values(), Fraction(0)))

    def trade_fees(self, trade: Trade) -> Tuple[int, int]:
        """(buyer_fee, seller_fee) in tick-lots for one trade."""
        taker_buys = trade.aggressor is Side.BUY
        bt, bslow = self._plan(Role.TAKER if taker_buys else Role.MAKER)
        st, sslow = self._plans[Role.MAKER if taker_buys else Role.TAKER]
        if bslow or sslow:
            buy, sell = fills_of(trade)
            return self.charge(buy), self.charge(sell)
        q = int(trade.qty)
        n = int(trade.price) * q
        return eval_terms(bt, n, q), eval_terms(st, n, q)

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
