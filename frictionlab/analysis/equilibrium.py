"""Equilibrium detection for strategy populations (discrete replicator dynamics).

A population is a share vector x on the simplex over agent archetypes
(satisficer vs optimizer, ZI-C vs informed vs noise, ...). One generation is

    x_i' = x_i * (1 + eta * (f_i - f_bar)),   f_bar = sum_i x_i f_i

renormalized onto the simplex (negative shares are clipped first). The update
is zero-sum, so sum(x') = 1 before clipping. With `normalize=True` the
deviations are divided by max |f_i - f_bar|, which bounds the step at eta and
makes eta unit-free when fitness is a PnL in ticks. Fitness is either

- analytic: `matrix_fitness(A)` gives f = A @ x (pure payoff-matrix mode), or
- simulated: `SimulatedFitness` turns shares into integer agent counts by
  largest-remainder rounding, runs the market through the fl-023 sweep
  engine with a fixed seed per generation, and reads mean net PnL per agent
  of each type (`mean_net_by_type`).

A `StabilityDetector` declares convergence when the max absolute share
change stays below epsilon for `window` generations ("stable") or when an
archetype that started above `floor` drops below it ("extinction").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .pnl import breakdown_by
from .sweep import SweepSpec, run_sweep

Fitness = Callable[[np.ndarray, int], np.ndarray]


def as_simplex(x) -> np.ndarray:
    """Clip to >= 0 and renormalize to sum 1."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError(f"shares must be a non-empty finite 1-d vector, got {x!r}")
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s <= 0:
        raise ValueError("shares must have positive total mass")
    return x / s


def replicator_step(x, f, eta: float, normalize: bool = False) -> np.ndarray:
    """One generation. Types with zero share or non-finite fitness are neutral."""
    x = as_simplex(x)
    f = np.asarray(f, dtype=float)
    if f.shape != x.shape:
        raise ValueError(f"fitness shape {f.shape} != shares shape {x.shape}")
    if not eta >= 0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    act = (x > 0) & np.isfinite(f)
    if not act.any() or f[act].max() == f[act].min():
        return x.copy()                           # equal fitness: no selection
    fbar = float(x[act] @ f[act]) / float(x[act].sum())
    dev = np.zeros_like(x)
    dev[act] = f[act] - fbar
    if normalize:
        dev /= np.abs(dev).max()
    return as_simplex(x * (1.0 + eta * dev))


def largest_remainder(shares, total: int) -> np.ndarray:
    """Integer counts summing exactly to `total` (Hamilton apportionment).

    Floors first, then hands the leftover units to the largest fractional
    remainders; ties go to the lower index, so the result is deterministic."""
    if int(total) != total or total < 0:
        raise ValueError(f"total must be a non-negative integer, got {total}")
    raw = as_simplex(shares) * int(total)
    base = np.floor(raw + 1e-12).astype(np.int64)
    frac = raw - base
    short = int(total) - int(base.sum())
    idx = np.arange(raw.size)
    if short > 0:
        base[np.lexsort((idx, -frac))[:short]] += 1
    elif short < 0:                               # float guard: take from smallest
        for i in np.lexsort((-idx, frac)):
            if short == 0:
                break
            if base[i] > 0:
                base[i] -= 1
                short += 1
    return base


@dataclass(frozen=True)
class StabilityDetector:
    epsilon: float = 1e-6
    window: int = 10
    floor: float = 0.0                            # extinction threshold, 0 disables

    def __post_init__(self) -> None:
        if self.window < 1 or not self.epsilon > 0 or self.floor < 0:
            raise ValueError("need window >= 1, epsilon > 0, floor >= 0")

    def check(self, traj) -> Optional[str]:
        """Reason the trajectory has settled at its last row, else None."""
        traj = np.atleast_2d(np.asarray(traj, dtype=float))
        if self.floor > 0 and len(traj) > 1:
            if np.any((traj[0] >= self.floor) & (traj[-1] < self.floor)):
                return "extinction"
        if len(traj) > self.window:
            if np.abs(np.diff(traj[-self.window - 1:], axis=0)).max() < self.epsilon:
                return "stable"
        return None

    def scan(self, traj) -> Optional[Tuple[int, str]]:
        """First generation at which the detector fires on a stored trajectory."""
        traj = np.atleast_2d(np.asarray(traj, dtype=float))
        for t in range(1, len(traj)):
            reason = self.check(traj[:t + 1])
            if reason:
                return t, reason
        return None


@dataclass(frozen=True)
class EquilibriumResult:
    labels: Tuple[str, ...]
    shares: np.ndarray                            # (G + 1, n) share trajectory
    fitness: np.ndarray                           # (G, n) fitness history
    reason: Optional[str]                         # "stable" | "extinction" | None
    generation: Optional[int]                     # generation the detector fired
    fixed_point: Optional[np.ndarray]             # final shares if converged

    @property
    def converged(self) -> bool:
        return self.reason is not None

    @property
    def final(self) -> Dict[str, float]:
        return dict(zip(self.labels, map(float, self.shares[-1])))


def evolve(x0, fitness: Fitness, *, eta: float = 0.1, generations: int = 500,
           detector: Optional[StabilityDetector] = None, normalize: bool = False,
           labels: Optional[Sequence[str]] = None) -> EquilibriumResult:
    """Iterate the replicator map until the detector fires or `generations` ends."""
    detector = detector or StabilityDetector()
    x = as_simplex(x0)
    labels = tuple(labels) if labels is not None else tuple(f"s{i}" for i in range(x.size))
    if len(labels) != x.size:
        raise ValueError(f"{len(labels)} labels for {x.size} archetypes")
    traj: List[np.ndarray] = [x]
    fits: List[np.ndarray] = []
    reason, gen = None, None
    for g in range(generations):
        f = np.asarray(fitness(x, g), dtype=float)
        fits.append(f)
        x = replicator_step(x, f, eta, normalize)
        traj.append(x)
        reason = detector.check(np.vstack(traj))
        if reason:
            gen = g + 1
            break
    shares = np.vstack(traj)
    fit = np.vstack(fits) if fits else np.empty((0, x.size))
    return EquilibriumResult(labels, shares, fit, reason, gen,
                             shares[-1].copy() if reason else None)


def matrix_fitness(payoff) -> Fitness:
    """Pure payoff-matrix mode: f = A @ x (row player's expected payoff)."""
    a = np.asarray(payoff, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError(f"payoff must be a finite square matrix, got shape {a.shape}")
    return lambda x, g: a @ x


def mean_net_by_type(result, mark, key: Callable[[Any], Hashable],
                     labels: Sequence[str]) -> Dict[str, float]:
    """Mean net PnL per agent for each archetype label present in the run."""
    ids = [aid for aid, a in result.agents.items() if key(a) in labels]
    groups = breakdown_by(result, mark, key=key, agents=ids)
    return {str(k): b.net / b.n_agents for k, b in groups.items() if b.n_agents}


@dataclass(frozen=True)
class SimulatedFitness:
    """Fitness from simulated markets; `factory(params, seed)` reads
    params["counts"] = {label: n_agents} and builds a runner for that mix."""
    factory: Callable[[Dict[str, Any], int], Any]
    metrics: Callable[[Any], Mapping[str, float]]  # label -> mean net PnL
    labels: Sequence[str]
    n_agents: int
    params: Mapping[str, Any] = field(default_factory=dict)
    master_seed: int = 0
    replicates: int = 1
    run_kwargs: Mapping[str, Any] = field(default_factory=dict)
    n_workers: int = 1

    def seeds(self, generation: int) -> List[int]:
        kids = np.random.SeedSequence([self.master_seed, generation]).spawn(self.replicates)
        return [int(k.generate_state(1, np.uint64)[0]) for k in kids]

    def __call__(self, x: np.ndarray, generation: int) -> np.ndarray:
        counts = largest_remainder(x, self.n_agents)
        cell = {**self.params, "counts": dict(zip(self.labels, map(int, counts))),
                "generation": generation}
        spec = SweepSpec(self.factory, self.metrics, cells=[cell], seeds=self.seeds(generation),
                         run_kwargs=dict(self.run_kwargs))
        rows = run_sweep(spec, n_workers=self.n_workers)
        errors = [r["error"] for r in rows if r["error"] is not None]
        if errors:
            raise RuntimeError(f"generation {generation} failed: {errors[0]}")
        out = np.full(len(self.labels), np.nan)
        for i, lab in enumerate(self.labels):
            vals = [float(r[lab]) for r in rows if r.get(lab) is not None]
            if vals and counts[i] > 0:
                out[i] = float(np.mean(vals))
        return out
