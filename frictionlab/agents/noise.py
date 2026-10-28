"""Noise (liquidity) traders: value-blind marketable order flow.

In Kyle (1985) and Glosten-Milgrom the noise trader is the reason a market
maker can survive informed flow: its orders carry no information, so the
maker earns the spread from them and loses it to the informed. Here:

- arrival: the runner wakes the agent as a Poisson process with rate
  `arrival_rate` (exposed as `wake_rate`); on each wake-up the agent trades
  with probability `participation`, so the effective order arrival rate is
  arrival_rate * participation.
- side:    BUY with probability `buy_prob` (0.5 = balanced), drawn from the
  agent's own seeded RNG, independent of the fundamental and of the book.
- size:    uniform integer in [min_size, max_size].
- order:   a market order (IOC): it takes what is at the touch and any
  unfilled remainder is cancelled, so the noise trader never makes liquidity.

The trader pays friction like anyone else but never conditions on it: that is
what makes it noise. `decision_cost` defaults to 0 (it does not think).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import numpy as np

from ..core.types import Side, _check_int
from .base import Action, Agent, MarketView, Submit

RngLike = Union[np.random.Generator, int, None]


def _prob(name: str, p: float) -> float:
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {p}")
    return p


class NoiseTrader(Agent):
    def __init__(self, agent_id: int, rng: RngLike = None, *,
                 arrival_rate: float = 1.0, participation: float = 1.0,
                 buy_prob: float = 0.5, min_size: int = 1, max_size: int = 1,
                 decision_cost: int = 0) -> None:
        super().__init__(agent_id, decision_cost)
        rate = float(arrival_rate)
        if not (rate > 0 and math.isfinite(rate)):
            raise ValueError(f"arrival_rate must be a finite number > 0, got {arrival_rate}")
        self.arrival_rate = rate
        self.participation = _prob("participation", participation)
        self.buy_prob = _prob("buy_prob", buy_prob)
        self.min_size = _check_int("min_size", min_size, positive=True)
        self.max_size = _check_int("max_size", max_size, positive=True)
        if self.min_size > self.max_size:
            raise ValueError("min_size must be <= max_size")
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self.orders: List[Submit] = []  # every intent ever emitted

    @property
    def wake_rate(self) -> float:
        """Read by `MarketRunner` to schedule this agent's Poisson wake-ups."""
        return self.arrival_rate

    @property
    def effective_rate(self) -> float:
        return self.arrival_rate * self.participation

    def draw(self) -> Optional[Submit]:
        """One wake-up's intent; RNG draw order is fixed for determinism."""
        u_part, u_side = self.rng.random(), self.rng.random()
        size = int(self.rng.integers(self.min_size, self.max_size + 1))
        if u_part >= self.participation:
            return None
        side = Side.BUY if u_side < self.buy_prob else Side.SELL
        return Submit(side, size)  # price=None -> marketable (IOC) order

    def decide(self, view: MarketView) -> List[Action]:
        order = self.draw()
        if order is None:
            return []
        self.orders.append(order)
        return [order]

    def imbalance(self) -> float:
        """(buy lots - sell lots) / total lots of emitted orders; 0 if none."""
        signed = sum(o.side.value * o.qty for o in self.orders)
        total = sum(o.qty for o in self.orders)
        return signed / total if total else 0.0

    def __repr__(self) -> str:
        return (f"NoiseTrader(id={self.agent_id}, rate={self.arrival_rate}, "
                f"participation={self.participation}, buy_prob={self.buy_prob})")


def make_noise_population(n: int, seed: RngLike = 0, first_id: int = 1,
                          **kw) -> List[NoiseTrader]:
    """n noise traders with independent child RNG streams from one seed."""
    n = _check_int("n", n, positive=True)
    root = seed if not isinstance(seed, np.random.Generator) else int(seed.integers(2**63))
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(root).spawn(n)]
    return [NoiseTrader(first_id + i, r, **kw) for i, r in enumerate(streams)]


def order_flow(traders: Sequence[NoiseTrader]) -> np.ndarray:
    """Signed lots of every emitted order across traders (BUY > 0)."""
    return np.array([o.side.value * o.qty for t in traders for o in t.orders],
                    dtype=np.int64)
