"""Repeated pricing market with menu costs and nominal rigidity (fl-070).

Sellers post integer-tick prices over steps t = 1 .. T. Seller i has a
frictionless target price p*_i that moves every step by a common drift d
(`inflation_ticks_per_step`) plus an idiosyncratic shock e_{i,t} drawn from a
seeded discrete schedule on the per-seller stream seeded by (seed, i), one draw
per step whatever the seller does, so no choice moves another agent's RNG:

    p*_{i,t} = p*_{i,t-1} + d + e_{i,t}

Each step runs: (1) targets move; (2) every `SsPricer` applies its (S, s) rule
and pays the `MenuCost` to the menu sink iff its price changed; (3) every seller
sells at its posted price p and earns, in exact integers,

    profit = base_profit - curvature * (p - p*)^2

from the demand account (`DEMAND_ID`), the quadratic loss being the
misalignment cost of a stale price. Seller net PnL therefore splits exactly as

    net = T * base_profit - sum curvature * gap^2 - menu cost paid

Sales reduce to fl-022 `TradeRecord`s tagged "menu" (price = posted price,
notional = profit moved); cash moves only by `Ledger.transfer`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np

from ..agents.ss_pricer import SsPricer
from ..core.ledger import Ledger
from ..core.types import Side, _check_int
from ..costs.menu import MenuCost

MENU = "menu"
DEMAND_ID = -14


@dataclass(frozen=True)
class MenuStep:
    """One seller-step: target, posted price, gap, adjustment and profit (ticks)."""
    t: int
    seller: int
    target: int
    price: int
    gap_before: int     # posted(t-1) - target(t), the gap the rule saw
    adjusted: bool
    loss: int           # curvature * (price - target)^2
    profit: int         # base_profit - loss

    @property
    def gap(self) -> int:
        return self.price - self.target


class MenuCostMarket:
    """Posted-price market where changing a price costs a menu cost."""

    def __init__(self, sellers: Sequence[SsPricer], targets: Sequence[int], *, steps: int,
                 inflation_ticks_per_step: int = 0, shocks: Sequence[int] = (0,),
                 menu: Union[MenuCost, int] = 0, curvature: int = 1, base_profit: int = 0,
                 seed: int = 0, cash: int = 0) -> None:
        if not sellers or len(sellers) != len(targets):
            raise ValueError("need one initial target per seller (at least one seller)")
        ids = [s.agent_id for s in sellers]
        if len(set(ids)) != len(ids) or min(ids) < 1:
            raise ValueError("seller ids must be distinct positive ints")
        self.sellers: Dict[int, SsPricer] = {s.agent_id: s for s in sellers}
        self.steps = _check_int("steps", steps, positive=True)
        self.drift = _check_int("inflation_ticks_per_step", inflation_ticks_per_step)
        self.schedule = [_check_int("shock", e) for e in shocks]
        if not self.schedule:
            raise ValueError("shock schedule must be non-empty")
        self.menu = menu if isinstance(menu, MenuCost) else MenuCost(menu)
        self.curvature = _check_int("curvature", curvature, nonneg=True)
        self.base_profit = _check_int("base_profit", base_profit)
        self.targets: Dict[int, int] = {
            i: _check_int("target", p, positive=True) for i, p in zip(ids, targets)}
        self._rngs = {i: np.random.default_rng([seed, i]) for i in ids}
        self.ledger = Ledger()
        self.ledger.open_account(DEMAND_ID)
        for i in ids:
            self.ledger.open_account(i, _check_int("cash", cash))
        self.menu.attach(self.ledger)
        self.shock_draws: Dict[int, List[int]] = {i: [] for i in ids}
        self.log: List[MenuStep] = []
        self._ran = False

    # ---- model ---------------------------------------------------------
    def loss(self, gap: int) -> int:
        """Misalignment loss curvature * gap^2 (>= 0, zero iff gap == 0 or curvature == 0)."""
        return self.curvature * _check_int("gap", gap) ** 2

    def _settle(self, seller: int, profit: int) -> None:
        if profit >= 0:
            self.ledger.transfer(DEMAND_ID, seller, profit)
        else:
            self.ledger.transfer(seller, DEMAND_ID, -profit)

    def _step(self, t: int) -> None:
        for i, s in self.sellers.items():
            e = self.schedule[int(self._rngs[i].integers(len(self.schedule)))]
            self.shock_draws[i].append(e)
            self.targets[i] += self.drift + e
        for i, s in self.sellers.items():
            target, before = self.targets[i], s.price - self.targets[i]
            changed = s.reprice(target)
            if changed:
                self.menu.charge(self.ledger, i)
            loss = self.loss(s.price - target)
            profit = self.base_profit - loss
            self._settle(i, profit)
            self.log.append(MenuStep(t, i, target, s.price, before, changed, loss, profit))

    def run(self) -> int:
        """Play steps 1 .. T once (idempotent); returns the number of price changes."""
        if not self._ran:
            self._ran = True
            for i, s in self.sellers.items():
                s.start(self.targets[i])
            for t in range(1, self.steps + 1):
                self._step(t)
        return self.adjustments

    # ---- views ---------------------------------------------------------
    @property
    def adjustments(self) -> int:
        """Price changes over the run."""
        return sum(1 for r in self.log if r.adjusted)

    def path(self, seller: int) -> List[MenuStep]:
        """Per-step log of one seller."""
        return [r for r in self.log if r.seller == seller]

    def adjustment_rate(self) -> float:
        """Share of seller-steps with a price change."""
        return self.adjustments / len(self.log) if self.log else 0.0

    def records(self) -> list:
        """Sales as fl-022 `TradeRecord`s, venue tag "menu" (one lot per seller-step)."""
        from ..analysis.metrics import TradeRecord  # lazy: analysis sits above markets
        return [TradeRecord(k + 1, float(r.t), r.price, 1, Side.BUY, DEMAND_ID, r.seller,
                            MENU, r.profit) for k, r in enumerate(self.log)]

    def __repr__(self) -> str:
        return (f"MenuCostMarket(sellers={len(self.sellers)}, steps={self.steps}, "
                f"drift={self.drift}, adjustments={self.adjustments})")
