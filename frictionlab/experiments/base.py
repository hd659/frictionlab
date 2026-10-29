"""Shared shape for Phase 5 experiments.

Every experiment is one fl-023 sweep plus a few headline numbers. Wrapping
both in an `ExperimentResult` lets each experiment ship the same artifacts:
tidy rows (one per cell x replicate), a per-cell aggregate, a findings dict
and fl-026 report sections, so a Phase 5 article is a function of
`(name, master_seed, params)` alone.

    res = run_experiment("coase", factory, metrics, {"friction": [0, 10, 20]},
                         replicates=4, master_seed=28)
    res.aggregate()                     # mean / std / n per friction level
    res.report().write_report("coase.html")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..analysis.report import Report
from ..analysis.report_sections import MetricsSection, Section, SweepSection
from ..analysis.sweep import RESERVED, SweepSpec, aggregate, run_sweep

Row = Dict[str, Any]


class ExperimentError(RuntimeError):
    """A sweep cell failed and the experiment was asked to be strict."""


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    master_seed: int
    params: Dict[str, List[Any]]          # swept grid, key -> levels (stable order)
    rows: List[Row] = field(repr=False)   # tidy fl-023 rows, (cell, replicate) order
    findings: Dict[str, Any] = field(default_factory=dict)

    @property
    def param_keys(self) -> List[str]:
        return list(self.params)

    @property
    def errors(self) -> List[Row]:
        return [r for r in self.rows if r.get("error") is not None]

    @property
    def ok_rows(self) -> List[Row]:
        return [r for r in self.rows if r.get("error") is None]

    def aggregate(self, metrics: Optional[Sequence[str]] = None) -> List[Row]:
        return aggregate(self.rows, self.param_keys, metrics)

    def column(self, metric: str, stat: str = "mean") -> List[float]:
        """One aggregate column in grid order, e.g. `column("volume")`."""
        return [r[f"{metric}_{stat}"] for r in self.aggregate()]

    def metric_names(self) -> List[str]:
        names: Dict[str, None] = {}
        for r in self.ok_rows:
            names.update(dict.fromkeys(k for k in r if k not in RESERVED and k not in self.params))
        return list(names)

    def raise_on_error(self) -> "ExperimentResult":
        if self.errors:
            e = self.errors[0]
            raise ExperimentError(f"{self.name}: {len(self.errors)} failed cell(s); first "
                                  f"cell={e.get('cell')} replicate={e.get('replicate')}: "
                                  f"{e['error']}")
        return self

    # ---- fl-026 report ---------------------------------------------------
    def sections(self, metric: Optional[str] = None) -> List[Section]:
        title = self.name.replace("_", " ").capitalize()
        return [MetricsSection(extra=self.findings, title=f"{title}: findings"),
                SweepSection(self.rows, self.param_keys, metric, title=f"{title}: sweep")]

    def report(self, metric: Optional[str] = None, title: Optional[str] = None) -> Report:
        return Report(title or f"frictionlab experiment: {self.name}",
                      master_seed=self.master_seed, grid=self.params,
                      meta={"experiment": self.name}, sections=self.sections(metric))


def run_experiment(name: str, factory: Callable[[Dict[str, Any], int], Any],
                   metrics: Callable[[Any], Mapping[str, Any]],
                   grid: Mapping[str, Sequence[Any]], *, replicates: int = 1,
                   master_seed: int = 0, crn: bool = True,
                   seeds: Optional[Sequence[int]] = None,
                   base: Optional[Mapping[str, Any]] = None,
                   run_kwargs: Optional[Mapping[str, Any]] = None, n_workers: int = 1,
                   findings: Optional[Callable[[List[Row]], Mapping[str, Any]]] = None,
                   strict: bool = True) -> ExperimentResult:
    """Run one fl-023 sweep and package it as an `ExperimentResult`.

    Common random numbers (`crn=True`) are the default: replicate r sees the
    same seed at every level, so differences across the grid are caused by
    the swept parameter, not by different draws. `findings(rows)` turns the
    rows into headline numbers. With `strict=True` any error row raises.
    """
    params = {k: list(v) for k, v in grid.items()}
    if not params or any(not v for v in params.values()):
        raise ValueError(f"grid must have at least one non-empty axis, got {params}")
    spec = SweepSpec(factory, metrics, grid=params, base=dict(base or {}),
                     replicates=replicates, master_seed=master_seed, crn=crn,
                     seeds=seeds, run_kwargs=dict(run_kwargs or {}))
    rows = run_sweep(spec, n_workers=n_workers)
    res = ExperimentResult(name, int(master_seed), params, rows)
    if strict:
        res.raise_on_error()
    if findings is not None:
        res.findings.update(findings(res.ok_rows))
    return res
