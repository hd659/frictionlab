"""(S, s) pricer (fl-070): deterministic menu-cost pricing with an inaction band.

An `SsPricer` posts one integer-tick price in a fl-070 `MenuCostMarket`. Each
step it observes its frictionless target p* and its posted price p and applies
the Sheshinski-Weiss (S, s) rule:

    keep    if |p - p*| <= band                    (ties: no adjustment)
    reset   to p* + overshoot                       otherwise

The overshoot anticipates the common drift d (ticks per step) over the band
half-life h, the number of whole steps drift needs to cover half the band:

    h = (band // 2) // |d|   (0 if d == 0),   overshoot = d * h

so |overshoot| <= band // 2 and a fresh reset always lands inside the band;
with positive inflation the seller resets above target and lets drift erode
the gap. A flexible pricer (band = 0) has overshoot 0 and re-prices to p*
whenever the gap is non-zero. The market drives every choice and charges the
menu cost; `decide` returns no book intents because the venue is posted-price.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..core.types import _check_int
from .base import Action, Agent, MarketView


def half_life(band: int, drift: int) -> int:
    """Whole steps for drift `drift` to cover half of `band` (0 without drift)."""
    band = _check_int("band", band, nonneg=True)
    drift = _check_int("drift", drift)
    return 0 if drift == 0 else (band // 2) // abs(drift)


def overshoot(band: int, drift: int) -> int:
    """Reset overshoot in ticks: expected drift over the band half-life."""
    return _check_int("drift", drift) * half_life(band, drift)


class SsPricer(Agent):
    """Seller with an (S, s) inaction band around its frictionless target price."""

    def __init__(self, agent_id: int, band: int = 0, *, drift: int = 0,
                 decision_cost: int = 0) -> None:
        super().__init__(agent_id, decision_cost)
        self.band = _check_int("band", band, nonneg=True)
        self.drift = _check_int("drift", drift)
        self.overshoot = overshoot(self.band, self.drift)
        self.price: Optional[int] = None
        self.prices: List[int] = []           # posted price per step
        self.adjusted: List[bool] = []        # adjustment flag per step

    @classmethod
    def flexible(cls, agent_id: int, drift: int = 0) -> "SsPricer":
        """Pricer with band 0: re-prices to its target whenever the gap is non-zero."""
        return cls(agent_id, 0, drift=drift)

    @property
    def is_flexible(self) -> bool:
        return self.band == 0

    # ---- policy --------------------------------------------------------
    def start(self, target: int) -> int:
        """Open at the initial target (no menu cost: the first price is not a change)."""
        self.price = _check_int("target", target)
        return self.price

    def should_adjust(self, price: int, target: int) -> bool:
        """True iff |price - target| > band (ties keep the price)."""
        return abs(_check_int("price", price) - _check_int("target", target)) > self.band

    def reset_price(self, target: int) -> int:
        """Price posted on adjustment: target plus the drift overshoot."""
        return _check_int("target", target) + self.overshoot

    def reprice(self, target: int) -> bool:
        """Apply the rule against `target`; returns True iff the price changed."""
        if self.price is None:
            raise RuntimeError("call start(target) before reprice")
        new = self.reset_price(target) if self.should_adjust(self.price, target) else self.price
        changed = new != self.price
        self.price = new
        self.prices.append(new)
        self.adjusted.append(changed)
        return changed

    def decide(self, view: MarketView) -> Sequence[Action]:
        """Posted-price venue: the market calls `reprice`, no book intents."""
        return ()

    def __repr__(self) -> str:
        return (f"SsPricer(id={self.agent_id}, band={self.band}, drift={self.drift}, "
                f"overshoot={self.overshoot}, price={self.price})")
