"""Phase 5 experiments: each one a seeded sweep, a findings dict and a report.

- `base`: `ExperimentResult` (name, master seed, params grid, tidy fl-023
  rows, headline findings) and `run_experiment`, the shared wrapper around
  the fl-023 sweep engine that every experiment uses, with fl-026 report
  sections built from the result.
- `coase`: the Coase-threshold experiment. A ZI-C market with a fixed
  valuation/cost schedule under one total per-trade friction level (fixed
  cost on both legs plus a taker fee, integer ticks); `find_coase_threshold`
  locates the collapse level by grid scan plus integer bisection and
  reports the analytic bound (largest valuation-cost gap).
"""
from .base import ExperimentError, ExperimentResult, run_experiment
from .coase import (
    CoaseThreshold, CoaseZIC, Config as CoaseConfig, coase_metrics, default_levels,
    find_coase_threshold, friction_stack as coase_friction_stack, max_volume, run_coase,
    split_friction,
)

__all__ = [
    "ExperimentError", "ExperimentResult", "run_experiment",
    "CoaseConfig", "CoaseThreshold", "CoaseZIC", "coase_friction_stack", "coase_metrics",
    "default_levels", "find_coase_threshold", "max_volume", "run_coase", "split_friction",
]
