"""Reproducible parameter sweeps over friction levels.

A `SweepSpec` couples a scenario factory `factory(params, seed) -> runner`
with a grid (or explicit list) of parameter cells, a replicate count and a
metrics function `metrics(result) -> dict`. Every (cell, replicate) pair gets
a child seed spawned from one master `numpy.random.SeedSequence`, indexed by
its position in the sweep, so results never depend on execution order: a
serial run and a `ProcessPoolExecutor` run produce identical rows.

    spec = SweepSpec(factory, metrics, grid={"taker_bps": [0, 50]}, replicates=4)
    rows = run_sweep(spec, n_workers=2)
    table = aggregate(rows, by=spec.param_keys)

With `crn=True` (common random numbers) replicate r reuses one seed in every
cell, so friction levels are compared on the same draws; explicit `seeds`
pin those per-replicate seeds (and imply CRN). A cell that raises becomes an
error row (`error="Type: message"`) instead of aborting the sweep.
"""
from __future__ import annotations

import csv
import io
import itertools
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..costs import FixedCost, FrictionStack, HalfSpreadCost, LatencyCost, MakerTakerFee

Params = Dict[str, Any]
Row = Dict[str, Any]
RESERVED = ("cell", "replicate", "seed", "error")
FRICTION_KEYS = ("maker_bps", "taker_bps", "fixed_cost", "half_spread", "latency", "drift",
                 "decision_cost", "cost_per_step")


def friction_stack(params: Mapping[str, Any]) -> FrictionStack:
    """Venue frictions from a params dict; agent-side keys (decision_cost,
    cost_per_step) are left for the factory to pass to its agents."""
    models = []
    if "maker_bps" in params or "taker_bps" in params:
        models.append(MakerTakerFee(params.get("maker_bps", 0), params.get("taker_bps", 0)))
    if params.get("fixed_cost"):
        models.append(FixedCost(params["fixed_cost"]))
    if params.get("half_spread"):
        models.append(HalfSpreadCost(params["half_spread"]))
    if params.get("latency"):
        models.append(LatencyCost(params["latency"], params.get("drift", 0)))
    return FrictionStack(models)


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> List[Params]:
    """Cartesian product in a stable order: keys as given, last key fastest."""
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


@dataclass(frozen=True)
class SweepSpec:
    factory: Callable[[Params, int], Any]
    metrics: Callable[[Any], Mapping[str, Any]]
    grid: Optional[Mapping[str, Sequence[Any]]] = None
    cells: Optional[Sequence[Mapping[str, Any]]] = None
    base: Mapping[str, Any] = field(default_factory=dict)
    replicates: int = 1
    master_seed: int = 0
    crn: bool = False
    seeds: Optional[Sequence[int]] = None
    run_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.grid is None) == (self.cells is None):
            raise ValueError("give exactly one of grid= or cells=")
        if self.seeds is not None:
            object.__setattr__(self, "replicates", len(self.seeds))
        if self.replicates < 1:
            raise ValueError(f"replicates must be >= 1, got {self.replicates}")

    @property
    def points(self) -> List[Params]:
        raw = expand_grid(self.grid) if self.grid is not None else [dict(c) for c in self.cells]
        return [{**self.base, **c} for c in raw]

    @property
    def param_keys(self) -> Tuple[str, ...]:
        keys: Dict[str, None] = {}
        for p in self.points:
            keys.update(dict.fromkeys(p))
        return tuple(keys)

    def child_seeds(self) -> List[List[int]]:
        """seeds[cell][replicate] as ints, a pure function of the spec."""
        n, r = len(self.points), self.replicates
        if self.seeds is not None:
            return [[int(s) for s in self.seeds] for _ in range(n)]
        kids = np.random.SeedSequence(self.master_seed).spawn(r if self.crn else n * r)
        ints = [int(k.generate_state(1, np.uint64)[0]) for k in kids]
        return [ints if self.crn else ints[c * r:(c + 1) * r] for c in range(n)]

    def tasks(self) -> List[Tuple]:
        seeds = self.child_seeds()
        return [(self.factory, self.metrics, dict(self.run_kwargs), c, k, seeds[c][k], p)
                for c, p in enumerate(self.points) for k in range(self.replicates)]


def _run_cell(task: Tuple) -> Row:
    factory, metrics, run_kwargs, cell, rep, seed, params = task
    row: Row = {"cell": cell, "replicate": rep, "seed": seed, **params}
    try:
        obj = factory(dict(params), seed)
        result = obj.run(**run_kwargs) if callable(getattr(obj, "run", None)) else obj
        out = dict(metrics(result))
        clash = set(out) & (set(RESERVED) | set(params))
        if clash:
            raise KeyError(f"metric names shadow sweep columns: {sorted(clash)}")
        row.update(out)
        row["error"] = None
    except Exception as exc:  # one bad cell must not abort the sweep
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_sweep(spec: SweepSpec, n_workers: int = 1) -> List[Row]:
    """Tidy rows in (cell, replicate) order; parallel when n_workers > 1."""
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    tasks = spec.tasks()
    if n_workers == 1 or len(tasks) <= 1:
        return [_run_cell(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_run_cell, tasks))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def aggregate(rows: Sequence[Row], by: Sequence[str],
              metrics: Optional[Sequence[str]] = None, ddof: int = 0) -> List[Row]:
    """Per-cell mean / std / count over replicates, skipping error rows."""
    groups: Dict[Tuple, List[Row]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in by), []).append(r)
    out = []
    for key, grp in groups.items():
        ok = [r for r in grp if r.get("error") is None]
        names = metrics if metrics is not None else [
            k for k in dict.fromkeys(k for r in ok for k in r)
            if k not in RESERVED and k not in by and all(_is_number(r.get(k)) for r in ok)]
        agg: Row = {**dict(zip(by, key)), "n": len(ok), "errors": len(grp) - len(ok)}
        for m in names:
            xs = np.array([float(r[m]) for r in ok], dtype=float)
            agg[f"{m}_mean"] = float(xs.mean()) if xs.size else math.nan
            agg[f"{m}_std"] = float(xs.std(ddof=ddof)) if xs.size > ddof else math.nan
        out.append(agg)
    return out


def write_csv(rows: Sequence[Row], dest: Union[str, Path, io.TextIOBase, None] = None) -> str:
    """Pandas-free CSV (union of columns, first-seen order); returns the text."""
    cols = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, restval="", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: "" if v is None else v for k, v in r.items()})
    text = buf.getvalue()
    if isinstance(dest, (str, Path)):
        Path(dest).write_text(text)
    elif dest is not None:
        dest.write(text)
    return text
