"""Zero-intelligence traders (Gode & Sunder 1993 baseline).

Each trader holds a private *unit schedule*: a buyer has redemption values
(sorted high -> low), a seller has unit costs (sorted low -> high), drawn once
from a configurable integer tick range with the agent's own seeded RNG. Every
decide() replaces the trader's resting quote with a fresh single-lot limit
order for its next untraded unit:

- ZI-U (unconstrained): price ~ U[price_min, price_max], blind to its schedule.
- ZI-C (budget-constrained): a buyer bids ~ U[price_min, value], a seller asks
  ~ U[cost, price_max]; it can never trade a unit at a loss.

Units traded are inferred from the account position seen in `MarketView`
(fills settle through the ledger), so the agent never touches book or ledger.
Default `decision_cost` is 0: ZI agents do not think, so thinking is free.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.types import Side, _check_int
from .base import Action, Agent, Cancel, MarketView, Submit

RngLike = Union[np.random.Generator, int, None]
Range = Tuple[int, int]


def _check_range(name: str, rng_range: Range) -> Range:
    lo, hi = (_check_int(f"{name}[0]", rng_range[0], positive=True),
              _check_int(f"{name}[1]", rng_range[1], positive=True))
    if lo > hi:
        raise ValueError(f"{name} must satisfy lo <= hi, got {rng_range}")
    return lo, hi


class ZITrader(Agent):
    """Shared machinery; subclasses only choose the quote's price support."""

    constrained: bool = False

    def __init__(self, agent_id: int, side: Side, rng: RngLike = None, *,
                 n_units: int = 1, value_range: Range = (50, 150),
                 price_range: Range = (1, 200),
                 schedule: Optional[Sequence[int]] = None,
                 decision_cost: int = 0) -> None:
        super().__init__(agent_id, decision_cost)
        self.side = Side(side)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self.price_range = _check_range("price_range", price_range)
        self.value_range = _check_range("value_range", value_range)
        if schedule is None:
            lo, hi = self.value_range
            n = _check_int("n_units", n_units, positive=True)
            schedule = self.rng.integers(lo, hi + 1, size=n).tolist()
        vals = [_check_int("schedule value", v, positive=True) for v in schedule]
        if not vals:
            raise ValueError("schedule must hold at least one unit")
        # Buyers trade their most valuable unit first, sellers their cheapest.
        self.schedule: Tuple[int, ...] = tuple(sorted(vals, reverse=self.side is Side.BUY))
        self._pos0: Optional[int] = None
        self.units_traded = 0
        self.quotes: List[int] = []  # every price this trader ever posted

    # -- schedule bookkeeping ---------------------------------------------
    @property
    def n_units(self) -> int:
        return len(self.schedule)

    @property
    def exhausted(self) -> bool:
        return self.units_traded >= self.n_units

    @property
    def unit_value(self) -> Optional[int]:
        """Value (buyer) or cost (seller) of the next untraded unit."""
        return None if self.exhausted else self.schedule[self.units_traded]

    def observe(self, view: MarketView) -> None:
        super().observe(view)
        if self._pos0 is None:
            self._pos0 = view.position
        traded = self.side.value * (view.position - self._pos0)
        self.units_traded = max(0, min(traded, self.n_units))

    # -- pricing ------------------------------------------------------------
    def price_bounds(self, value: int) -> Range:
        return self.price_range

    def draw_price(self) -> int:
        lo, hi = self.price_bounds(self.unit_value)
        return int(self.rng.integers(lo, hi + 1))

    def decide(self, view: MarketView) -> List[Action]:
        actions: List[Action] = [Cancel(oid) for oid in view.open_orders]
        if self.exhausted:
            return actions
        price = self.draw_price()
        self.quotes.append(price)
        actions.append(Submit(self.side, 1, price))
        return actions

    def surplus(self, prices: Sequence[int]) -> int:
        """Realized surplus in ticks given the fill prices of traded units."""
        units = self.schedule[:len(prices)]
        return sum(self.side.value * (v - p) for v, p in zip(units, prices))

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(id={self.agent_id}, side={self.side.name}, "
                f"traded={self.units_traded}/{self.n_units})")


class ZIUTrader(ZITrader):
    """Unconstrained: quotes uniformly over the full price range."""

    constrained = False


class ZICTrader(ZITrader):
    """Budget-constrained: never bids above value nor asks below cost."""

    constrained = True

    def __init__(self, agent_id: int, side: Side, rng: RngLike = None, **kw) -> None:
        super().__init__(agent_id, side, rng, **kw)
        lo, hi = self.price_range
        bad = [v for v in self.schedule if not lo <= v <= hi]
        if bad:
            raise ValueError(f"unit values {bad} fall outside price_range {self.price_range}")

    def price_bounds(self, value: int) -> Range:
        lo, hi = self.price_range
        return (lo, value) if self.side is Side.BUY else (value, hi)


def make_zi_population(kind: str, n_buyers: int, n_sellers: int,
                       seed: RngLike = 0, first_id: int = 1, **kw) -> List[ZITrader]:
    """Build buyers then sellers, each with an independent child RNG stream."""
    cls = {"zi-u": ZIUTrader, "zi-c": ZICTrader}.get(kind.lower())
    if cls is None:
        raise ValueError(f"kind must be 'zi-u' or 'zi-c', got {kind!r}")
    sides = [Side.BUY] * n_buyers + [Side.SELL] * n_sellers
    seq = np.random.SeedSequence(seed if not isinstance(seed, np.random.Generator)
                                 else int(seed.integers(2**63)))
    streams = [np.random.default_rng(s) for s in seq.spawn(len(sides))]
    return [cls(first_id + i, s, r, **kw) for i, (s, r) in enumerate(zip(sides, streams))]
