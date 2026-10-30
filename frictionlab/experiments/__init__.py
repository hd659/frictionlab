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
- `decision_cost`: satisficing vs optimizing `BoundedRationalAgent`s over a
  cost_per_step x aspiration grid; `find_break_even_cost` bisects for the
  smallest step cost at which the satisficer's mean net PnL wins, and
  `population` runs the fl-024 replicator at a given cost.
"""
from .base import ExperimentError, ExperimentResult, run_experiment
from .coase import (
    CoaseThreshold, CoaseZIC, Config as CoaseConfig, coase_metrics, default_levels,
    find_coase_threshold, friction_stack as coase_friction_stack, max_volume, run_coase,
    split_friction,
)
from .decision_cost import (
    BreakEven, Config as DecisionCostConfig, find_break_even_cost, population,
    run_decision_cost, run_study as run_decision_cost_study, settled_share,
)

__all__ = [
    "ExperimentError", "ExperimentResult", "run_experiment",
    "CoaseConfig", "CoaseThreshold", "CoaseZIC", "coase_friction_stack", "coase_metrics",
    "default_levels", "find_coase_threshold", "max_volume", "run_coase", "split_friction",
    "BreakEven", "DecisionCostConfig", "find_break_even_cost", "population",
    "run_decision_cost", "run_decision_cost_study", "settled_share",
]
