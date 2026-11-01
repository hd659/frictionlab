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
- `adverse_selection`: `gm_zero_profit_spread` (analytic Glosten-Milgrom
  break-even quotes) plus a maker / informed / noise market swept over
  informed share x jump size, with maker PnL split into spread capture and
  adverse selection, and a continuous-vs-batch mitigation run.
- `two_venue`: two fl-031 continuous books sharing one population; takers
  routed by `VenueRouter`, makers choosing a venue each epoch by rebate net
  of fl-030 adverse-selection loss, per-lot integer-tick `LotFee`s.
- `fee_wars`: alternating venue best responses over a (maker, taker) fee
  grid through the fl-023 engine, with `detect` reporting a pure fixed point
  or an Edgeworth-style cycle (and its length).
- `funding_arb`: fl-033 `FundingArbitrageur`s against an inventory-skewed perp
  maker and one-sided retail flow, swept over shock_gas_level x
  shock_probability of a `GasProcess`; per cell entries, holding time, gross
  funding, friction breakdown, net PnL and mean |premium|.
  `entry_threshold_gas` is the analytic per-leg gas bound and
  `find_gas_threshold` bisects for the entry-collapse shock level.
"""
from .adverse_selection import (
    Config as AdverseSelectionConfig, GMMaker, GMQuote, gm_zero_profit_spread, maker_split,
    mitigation, run_grid as run_adverse_selection, run_study as run_adverse_selection_study,
    study_report as adverse_selection_report,
)
from .base import ExperimentError, ExperimentResult, run_experiment
from .coase import (
    CoaseThreshold, CoaseZIC, Config as CoaseConfig, coase_metrics, default_levels,
    find_coase_threshold, friction_stack as coase_friction_stack, max_volume, run_coase,
    split_friction,
)
from .fee_wars import (
    Config as FeeWarsConfig, Detection, best_response, detect, fee_grid, probe_seeds,
    run_fee_wars, study_report as fee_wars_report,
)
from .funding_arb import (
    Config as FundingArbConfig, GasThreshold, InventoryMaker, RetailFlow, analytic_bound as funding_arb_bound,
    arb_metrics as funding_arb_metrics, entry_threshold_gas, find_gas_threshold, run_funding_arb,
)
from .two_venue import LotFee, MarketConfig as TwoVenueConfig, TwoVenueMarket, simulate as simulate_two_venue
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
    "AdverseSelectionConfig", "GMMaker", "GMQuote", "adverse_selection_report",
    "gm_zero_profit_spread", "maker_split", "mitigation", "run_adverse_selection",
    "run_adverse_selection_study",
    "Detection", "FeeWarsConfig", "LotFee", "TwoVenueConfig", "TwoVenueMarket", "best_response",
    "detect", "fee_grid", "fee_wars_report", "probe_seeds", "run_fee_wars", "simulate_two_venue",
    "FundingArbConfig", "GasThreshold", "InventoryMaker", "RetailFlow", "entry_threshold_gas",
    "find_gas_threshold", "funding_arb_bound", "funding_arb_metrics", "run_funding_arb",
]
