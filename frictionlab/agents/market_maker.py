"""Inventory-risk market maker (Avellaneda & Stoikov 2008 style).

With mid price s, signed inventory q (lots, relative to `target_inventory`),
risk aversion gamma, volatility sigma (ticks per sqrt(time unit)), order
arrival decay k and remaining horizon tau = max(T - t, 0):

    reservation  r     = s - q * gamma * sigma^2 * tau
    total spread delta = gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / k)
    bid = floor(r - delta / 2),  ask = ceil(r + delta / 2),  ask - bid >= min_spread

A long maker shades both quotes down (sell more, buy less), a short maker
shades them up. Quotes live on the integer tick grid, never cross each other,
and a side is withdrawn once filling it could push |q| past `inventory_limit`.

Each decide() keeps a resting quote whose price is still optimal (preserving
time priority) and emits `Cancel` + `Submit` intents for every stale one.
Order ids are learned in `act()`; any open order the maker cannot map (e.g.
submitted through a deferring venue) is treated as stale and cancelled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.types import Side, Trade, _check_int
from .base import Action, Agent, Cancel, MarketView, Submit, Venue

RngLike = Union[np.random.Generator, int, None]


@dataclass(frozen=True)
class ASQuote:
    mid: float
    inventory: int
    reservation: float
    spread: float
    bid: Optional[int]  # None -> side withdrawn (inventory cap)
    ask: Optional[int]
    bid_qty: int = 0
    ask_qty: int = 0


def _positive(name: str, v: float) -> float:
    v = float(v)
    if not v > 0 or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number > 0, got {v}")
    return v


class AvellanedaStoikovMM(Agent):
    def __init__(self, agent_id: int, *, gamma: float = 0.1, sigma: float = 2.0,
                 k: float = 1.5, horizon: float = 100.0, start_time: float = 0.0,
                 order_size: int = 1, inventory_limit: int = 10,
                 target_inventory: int = 0, fair_value: int = 100,
                 min_spread: int = 1, jitter: int = 0, rng: RngLike = None,
                 decision_cost: int = 0) -> None:
        super().__init__(agent_id, decision_cost)
        self.gamma = _positive("gamma", gamma)
        self.sigma = _positive("sigma", sigma)
        self.k = _positive("k", k)
        self.horizon = _positive("horizon", horizon)
        self.start_time = float(start_time)
        self.order_size = _check_int("order_size", order_size, positive=True)
        self.inventory_limit = _check_int("inventory_limit", inventory_limit, positive=True)
        self.target_inventory = _check_int("target_inventory", target_inventory)
        self.fair_value = _check_int("fair_value", fair_value, positive=True)
        self.min_spread = _check_int("min_spread", min_spread, positive=True)
        self.jitter = _check_int("jitter", jitter, nonneg=True)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self._mid: float = float(self.fair_value)
        self._quotes: Dict[int, Tuple[Side, int]] = {}  # live oid -> (side, price)
        self.history: List[ASQuote] = []

    # -- model --------------------------------------------------------------
    def remaining(self, t: float) -> float:
        return max(self.start_time + self.horizon - float(t), 0.0)

    def estimate_mid(self, view: MarketView) -> float:
        """Book mid; else one-sided touch or last trade; else last known mid."""
        if view.mid is not None:
            self._mid = view.mid
        elif view.best_bid is not None or view.best_ask is not None:
            self._mid = float(view.best_bid if view.best_ask is None else view.best_ask)
        elif view.last_trade is not None:
            self._mid = float(view.last_trade.price)
        return self._mid

    def quote(self, view: MarketView) -> ASQuote:
        s = self.estimate_mid(view)
        q = view.position - self.target_inventory
        risk = self.gamma * self.sigma ** 2 * self.remaining(view.time)
        r = s - q * risk
        if self.jitter:
            r += int(self.rng.integers(-self.jitter, self.jitter + 1))
        delta = risk + (2.0 / self.gamma) * math.log1p(self.gamma / self.k)
        bid = math.floor(r - delta / 2)
        ask = math.ceil(r + delta / 2)
        if ask - bid < self.min_spread:
            ask = bid + self.min_spread
        if bid < 1:  # keep both quotes on the positive tick grid
            bid, ask = 1, max(ask, 1 + self.min_spread)
        cap = self.inventory_limit
        bid_qty = max(0, min(self.order_size, cap - q))
        ask_qty = max(0, min(self.order_size, cap + q))
        return ASQuote(s, q, r, delta, bid if bid_qty else None,
                       ask if ask_qty else None, bid_qty, ask_qty)

    # -- lifecycle ----------------------------------------------------------
    def decide(self, view: MarketView) -> List[Action]:
        qt = self.quote(view)
        self.history.append(qt)
        want = {Side.BUY: (qt.bid, qt.bid_qty), Side.SELL: (qt.ask, qt.ask_qty)}
        live = set(view.open_orders)
        self._quotes = {o: sp for o, sp in self._quotes.items() if o in live}
        actions: List[Action] = []
        kept = set()
        for oid in sorted(live):
            side_price = self._quotes.get(oid)
            if (side_price is not None and side_price[0] not in kept
                    and want[side_price[0]][0] == side_price[1]):
                kept.add(side_price[0])
            else:
                actions.append(Cancel(oid))
                self._quotes.pop(oid, None)
        for side in (Side.BUY, Side.SELL):
            price, qty = want[side]
            if price is not None and side not in kept:
                actions.append(Submit(side, qty, price))
        return actions

    def act(self, venue: Venue, actions: Sequence[Action], time: float) -> List[Trade]:
        trades: List[Trade] = []
        for a in actions:
            before = set(self.open_orders)
            trades += venue.execute(self, a, time)
            if isinstance(a, Submit):
                for oid in set(self.open_orders) - before:
                    self._quotes[oid] = (a.side, a.price)
        return trades

    @property
    def quotes(self) -> Dict[int, Tuple[Side, int]]:
        return dict(self._quotes)

    def __repr__(self) -> str:
        return (f"AvellanedaStoikovMM(id={self.agent_id}, gamma={self.gamma}, "
                f"sigma={self.sigma}, k={self.k}, cap={self.inventory_limit})")
