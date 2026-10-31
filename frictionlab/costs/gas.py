"""Seeded gas-shock process and a time-varying fixed per-trade gas cost.

GasProcess is a two-regime Markov chain on a step grid of width `dt` sim
seconds. Regime 0 is CALM (gas = `calm_gas` ticks per trade) and regime 1 is
SHOCK (gas = `shock_gas`). At every step boundary one uniform draw u decides
the transition:

    CALM  -> SHOCK  iff u < p_shock
    SHOCK -> CALM   iff u < p_calm

Stationary distribution: pi_shock = p_shock / (p_shock + p_calm). Exactly one
draw is consumed per step, in fixed-size chunks, so the path is a pure
function of the seed and never depends on the order of queries. With
p_shock = 0 and a CALM start the level is `calm_gas` forever, and GasCost then
charges bit-for-bit what `FixedCost(calm_gas)` charges.

GasCost plugs into a FrictionStack (component name "gas"). It reads the level
at the fill time, or at `clock.now` when bound to a SimClock.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np

from .base import BOTH, CostModel, Fill

CALM, SHOCK = 0, 1
_CHUNK = 1024


def _prob(name: str, p: float) -> float:
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {p}")
    return p


def _level(name: str, g: int) -> int:
    if isinstance(g, bool) or not isinstance(g, (int, np.integer)) or g < 0:
        raise ValueError(f"{name} must be a non-negative integer tick count, got {g!r}")
    return int(g)


class GasProcess:
    def __init__(self, calm_gas: int, shock_gas: int, p_shock: float = 0.0,
                 p_calm: float = 0.5, *, dt: float = 1.0, seed: int = 0,
                 start: int = CALM) -> None:
        self.calm_gas = _level("calm_gas", calm_gas)
        self.shock_gas = _level("shock_gas", shock_gas)
        self.p_shock = _prob("p_shock", p_shock)
        self.p_calm = _prob("p_calm", p_calm)
        if not float(dt) > 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if start not in (CALM, SHOCK):
            raise ValueError("start must be CALM (0) or SHOCK (1)")
        self.dt, self.seed, self.start = float(dt), int(seed), start
        self._rng = np.random.default_rng(self.seed)
        self._regimes: List[int] = [start]

    @classmethod
    def constant(cls, gas: int, **kw) -> "GasProcess":
        """Degenerate process: gas never shocks (constant-gas baseline)."""
        return cls(gas, gas, 0.0, 1.0, **kw)

    # ---- path ----------------------------------------------------------
    def _extend(self, n: int) -> None:
        reg = self._regimes
        while len(reg) <= n:
            r = reg[-1]
            for u in self._rng.random(_CHUNK):
                r = (SHOCK if u < self.p_shock else CALM) if r == CALM else \
                    (CALM if u < self.p_calm else SHOCK)
                reg.append(r)

    def step_of(self, t: float) -> int:
        if t < 0:
            raise ValueError(f"time must be >= 0, got {t}")
        return int(float(t) // self.dt)

    def regime_at_step(self, n: int) -> int:
        self._extend(n)
        return self._regimes[n]

    def level_at_step(self, n: int) -> int:
        return self.shock_gas if self.regime_at_step(n) == SHOCK else self.calm_gas

    def regime(self, t: float) -> int:
        return self.regime_at_step(self.step_of(t))

    def level(self, t: float) -> int:
        """Gas in integer ticks per trade at sim time `t`."""
        return self.level_at_step(self.step_of(t))

    __call__ = level
    value = level  # duck-types FundamentalProcess-style `value(t)`

    def regimes(self, n_steps: int) -> np.ndarray:
        self._extend(n_steps - 1)
        return np.asarray(self._regimes[:n_steps], dtype=np.int8)

    def path(self, n_steps: int) -> np.ndarray:
        """Gas level per step for the first `n_steps` steps."""
        return np.where(self.regimes(n_steps) == SHOCK, self.shock_gas, self.calm_gas)

    # ---- statistics ----------------------------------------------------
    @property
    def stationary(self) -> Tuple[float, float]:
        """(pi_calm, pi_shock); an absorbing chain keeps its start regime."""
        s = self.p_shock + self.p_calm
        if s == 0:
            return (1.0, 0.0) if self.start == CALM else (0.0, 1.0)
        return self.p_calm / s, self.p_shock / s

    @property
    def mean_gas(self) -> float:
        pc, ps = self.stationary
        return pc * self.calm_gas + ps * self.shock_gas

    def shock_frequency(self, n_steps: int) -> float:
        return float(self.regimes(n_steps).mean())

    def cost(self, roles=BOTH, clock=None) -> "GasCost":
        return GasCost(self, roles, clock)

    def __repr__(self) -> str:
        return (f"GasProcess(calm_gas={self.calm_gas}, shock_gas={self.shock_gas}, "
                f"p_shock={self.p_shock}, p_calm={self.p_calm}, dt={self.dt}, seed={self.seed})")


class GasCost(CostModel):
    """Fixed per-fill charge equal to the current gas level (ticks)."""
    name = "gas"

    def __init__(self, process: GasProcess, roles=BOTH, clock=None) -> None:
        super().__init__(roles)
        if not isinstance(process, GasProcess):
            raise TypeError(f"expected GasProcess, got {type(process).__name__}")
        self.process = process
        self.clock = clock

    def level(self, time: Optional[float] = None) -> int:
        t = self.clock.now if self.clock is not None else (0.0 if time is None else time)
        return self.process.level(t)

    def raw_cost(self, fill: Fill) -> Fraction:
        return Fraction(self.level(fill.time))

    def __repr__(self) -> str:
        return f"GasCost({self.process!r})"
