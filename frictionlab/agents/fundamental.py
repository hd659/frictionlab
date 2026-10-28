"""Latent fundamental value V(t) shared by informed traders (and analysis).

Two regimes, both on the integer tick grid:
- constant:    V(t) = v0 for all t (jump_rate == 0).
- random walk: jumps arrive as a Poisson process with `jump_rate` per time
  unit; each jump moves V by +jump_size or -jump_size with equal probability
  (a seeded compound-Poisson random walk, the Glosten-Milgrom value shock).
  V is floored at `floor` so it stays a valid positive price.

The path is generated lazily but strictly in time order from the process's
own RNG, so `value(t)` is a pure function of (seed, t): querying times out of
order, repeatedly, or from several agents never changes the realized path.
"""
from __future__ import annotations

import bisect
import math
from typing import List, Sequence, Tuple, Union

import numpy as np

from ..core.types import _check_int

RngLike = Union[np.random.Generator, int, None]


class FundamentalProcess:
    def __init__(self, initial: int = 100, *, jump_rate: float = 0.0,
                 jump_size: int = 1, floor: int = 1, start_time: float = 0.0,
                 rng: RngLike = None) -> None:
        self.initial = _check_int("initial", initial, positive=True)
        self.jump_size = _check_int("jump_size", jump_size, positive=True)
        self.floor = _check_int("floor", floor, positive=True)
        if self.initial < self.floor:
            raise ValueError(f"initial {initial} is below floor {floor}")
        rate = float(jump_rate)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError(f"jump_rate must be a finite number >= 0, got {jump_rate}")
        self.jump_rate = rate
        self.start_time = float(start_time)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self._times: List[float] = [self.start_time]
        self._values: List[int] = [self.initial]
        self._next = self._draw_gap(self.start_time)

    # -- constructors ---------------------------------------------------------
    @classmethod
    def constant(cls, value: int) -> "FundamentalProcess":
        return cls(value)

    @classmethod
    def random_walk(cls, initial: int, jump_rate: float, jump_size: int = 1,
                    seed: RngLike = 0, **kw) -> "FundamentalProcess":
        return cls(initial, jump_rate=jump_rate, jump_size=jump_size, rng=seed, **kw)

    # -- path generation ------------------------------------------------------
    @property
    def is_constant(self) -> bool:
        return self.jump_rate == 0.0

    def _draw_gap(self, t: float) -> float:
        if self.is_constant:
            return math.inf
        return t + float(self.rng.exponential(1.0 / self.jump_rate))

    def _extend(self, t: float) -> None:
        while self._next <= t:
            step = self.jump_size if self.rng.random() < 0.5 else -self.jump_size
            self._times.append(self._next)
            self._values.append(max(self.floor, self._values[-1] + step))
            self._next = self._draw_gap(self._next)

    def value(self, t: float) -> int:
        """V(t): right-continuous, i.e. a jump at time s is visible at t == s."""
        t = float(t)
        if t < self.start_time:
            return self.initial
        self._extend(t)
        return self._values[bisect.bisect_right(self._times, t) - 1]

    __call__ = value

    def path(self, times: Sequence[float]) -> np.ndarray:
        return np.array([self.value(t) for t in times], dtype=np.int64)

    def jumps(self, until: float) -> List[Tuple[float, int]]:
        """(time, new value) for every realized jump in (start_time, until]."""
        self._extend(float(until))
        hi = bisect.bisect_right(self._times, float(until))
        return list(zip(self._times[1:hi], self._values[1:hi]))

    def __repr__(self) -> str:
        kind = "constant" if self.is_constant else f"jump_rate={self.jump_rate}"
        return (f"FundamentalProcess(initial={self.initial}, {kind}, "
                f"jump_size={self.jump_size})")
