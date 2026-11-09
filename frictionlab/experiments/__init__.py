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
- `search_cost` (Phase 7): the fl-046 `SearchMarket` swept over search_cost x
  decay_per_step; volume, time to match, surplus share of the CE maximum,
  search cost paid, decay loss and seller share. `reservation_price` is the
  analytic stopping-rule reservation and `find_search_collapse` bisects for
  the volume-collapse search cost against `search_cost_bound`.
- `intermediation` (Phase 7): the fl-049 `Middleman` in the fl-046
  `SearchMarket`, swept over search_cost x middleman_half_spread; intermediated
  share, volume, time to match, surplus share, middleman PnL split, search cost
  paid and the surplus change against the no-middleman twin.
  `middleman_indifference_spread` is the analytic bound and
  `find_intermediation_threshold` bisects for the intermediation search cost.
- `grossman_stiglitz` (Phase 8): a GM dealer, noise traders and fl-052
  `InfoAcquirer`s swept over signal_cost x noise_intensity; buyer fraction,
  spread, price efficiency, buyer vs non-buyer PnL and sink revenue.
  `gs_equilibrium_informed_share` is the analytic share, `gs_population` the
  replicator and `find_information_collapse` bisects for the collapse cost.
- `lemons` (Phase 9): the fl-055 `LemonsMarket` swept over certification_cost x
  buyer_premium; traded share (overall and by quality quintile), pooled price and
  rounds, certification uptake and cutoff, surplus share and certification paid.
  `akerlof_pooled_price` / `certification_cutoff` are the analytic fixed point and
  cutoff, and `find_certification_threshold` bisects for the uptake-collapse cost.
- `reputation` (Phase 10): the fl-058 `ReputationMarket` swept over
  rating_accuracy_bps x whitewash_cost; true-quality share, cheating rate and
  milking round, whitewashes, price-quality gap, surplus share and its gain over
  the fl-056 pooled lemons twin. `milking_switch_round` /
  `min_deterrent_whitewash_cost` are the analytic end-game round and re-entry
  deterrent, and `find_reputation_threshold` bisects for the no-whitewash cost.
- `switching_cost` (Phase 11): the fl-061 `SwitchingMarket` with two `LockinSeller`s
  swept over mean_switching_cost x target_share; acquisition vs harvest prices and
  markups, churn, final shares, switching cost paid, consumer surplus, seller
  profit and the welfare change against the zero-cost Bertrand twin.
  `klemperer_harvest_markup` / `acquisition_discount` are the analytic lock-in
  markup and round-0 investment, and `find_lockin_threshold` bisects for the
  smallest mean switching cost whose harvest markup exceeds a tick floor.
- `critical_mass` (Phase 12): the fl-064 `PlatformMarket` swept over adoption_cost x
  seed_members; final membership, convergence (fixed point or 2-cycle), adoption
  paid, seed subsidy, platform PnL, member surplus and the welfare change against
  the zero-adoption-cost twin. `critical_mass` / `tipping_fixed_points` iterate the
  best-response map on counts, and `find_tipping_seed` bisects for the tipping seed.
- `congestion` (Phase 13): the fl-067 `QueueMarket` swept over toll x waiting_cost;
  join / balk rates, mean and max queue length, steps in system, toll revenue,
  waiting paid, customer surplus, welfare and its change against the zero-toll twin.
  `naor_individual_threshold` / `naor_social_threshold` / `naor_optimal_toll` are the
  exact Naor thresholds and toll, and `find_congestion_toll` bisects for the toll.
- `registry` (Phase 6): name -> runner, param schema and defaults for every
  experiment; `run_from_config` turns a JSON/YAML config into a result.

Several modules share short names (`Config`, `run_grid`, `default_levels`), so
the package re-exports them under prefixed aliases such as `CoaseConfig`.
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
    Config as FundingArbConfig, GasThreshold, InventoryMaker, RetailFlow,
    analytic_bound as funding_arb_bound, arb_metrics as funding_arb_metrics,
    default_levels as funding_arb_levels, entry_threshold_gas, find_gas_threshold,
    run_funding_arb,
)
from .grossman_stiglitz import (
    CommittedBuyer, Config as GrossmanStiglitzConfig, GSCell, GSMaker, InformationCollapse,
    analytic_collapse_cost, find_information_collapse, gs_equilibrium_informed_share, gs_metrics,
    population as gs_population, run_grossman_stiglitz, run_gs_grid,
)
from .intermediation import (
    Config as IntermediationConfig, IntermediationCell, IntermediationThreshold,
    analytic_bound as intermediation_bound, find_intermediation_threshold,
    intermediation_metrics, middleman_indifference_spread, run_intermediation,
)
from .lemons import (
    CertificationThreshold, Config as LemonsConfig, LemonsCell, akerlof_pooled_price,
    cell_qualities as lemons_qualities, certification_bound, certification_cutoff,
    find_certification_threshold, lemons_metrics, run_lemons, run_lemons_grid,
)
from .reputation import (
    Config as ReputationConfig, ReputationCell, ReputationThreshold,
    cell_qualities as reputation_qualities, find_reputation_threshold, milking_switch_round,
    min_deterrent_whitewash_cost, predicted_milking_round, reputation_metrics, run_reputation,
    run_reputation_grid,
)
from .switching_cost import (
    Config as SwitchingCostConfig, LockinThreshold, SwitchingCell, acquisition_discount,
    analytic_lockin_threshold, find_lockin_threshold, klemperer_harvest_markup,
    run_switching_cost, run_switching_grid, switching_metrics,
)
from .critical_mass import (
    Config as CriticalMassConfig, PlatformCell, TippingSeed, critical_mass, empty_is_fixed_point,
    find_tipping_seed, platform_metrics, run_critical_mass, run_platform_grid, tipping_fixed_points,
)
from .congestion import (
    CongestionToll, Config as CongestionConfig, QueueCell, find_congestion_toll,
    naor_individual_threshold, naor_optimal_toll, naor_social_threshold, queue_metrics,
    run_congestion, run_queue_grid, welfare_per_step,
)
from .registry import (
    Experiment, UnknownExperimentError, experiment, get_experiment, list_experiments,
    register_experiment, resolve_config, run_from_config, unregister_experiment,
)
from .search_cost import (
    Config as SearchCostConfig, SearchCell, SearchCollapse, ShadedPolicy,
    analytic_bound as search_cost_bound, find_search_collapse, reservation_price,
    run_search_cost, search_metrics,
)
from .two_venue import (
    LotFee, MarketConfig as TwoVenueConfig, TwoVenueMarket, simulate as simulate_two_venue,
)
from .decision_cost import (
    BreakEven, Config as DecisionCostConfig, find_break_even_cost, population,
    run_decision_cost, run_study as run_decision_cost_study, settled_share,
)

__all__ = [
    # shared wrapper
    "ExperimentError", "ExperimentResult", "run_experiment",
    # 5.1 Coase threshold
    "CoaseConfig", "CoaseThreshold", "CoaseZIC", "coase_friction_stack", "coase_metrics",
    "default_levels", "find_coase_threshold", "max_volume", "run_coase", "split_friction",
    # 5.2 decision cost
    "BreakEven", "DecisionCostConfig", "find_break_even_cost", "population",
    "run_decision_cost", "run_decision_cost_study", "settled_share",
    # 5.3 adverse selection
    "AdverseSelectionConfig", "GMMaker", "GMQuote", "adverse_selection_report",
    "gm_zero_profit_spread", "maker_split", "mitigation", "run_adverse_selection",
    "run_adverse_selection_study",
    # 5.4 fee wars on two venues
    "Detection", "FeeWarsConfig", "LotFee", "TwoVenueConfig", "TwoVenueMarket", "best_response",
    "detect", "fee_grid", "fee_wars_report", "probe_seeds", "run_fee_wars", "simulate_two_venue",
    # 5.5 funding-rate arbitrage
    "FundingArbConfig", "GasThreshold", "InventoryMaker", "RetailFlow", "entry_threshold_gas",
    "find_gas_threshold", "funding_arb_bound", "funding_arb_levels", "funding_arb_metrics",
    "run_funding_arb",
    # 7.2 search cost and perishable value
    "SearchCell", "SearchCollapse", "SearchCostConfig", "ShadedPolicy", "find_search_collapse",
    "reservation_price", "run_search_cost", "search_cost_bound", "search_metrics",
    # 7.3 intermediation
    "IntermediationCell", "IntermediationConfig", "IntermediationThreshold",
    "find_intermediation_threshold", "intermediation_bound", "intermediation_metrics",
    "middleman_indifference_spread", "run_intermediation",
    # 8.2 Grossman-Stiglitz costly information
    "CommittedBuyer", "GSCell", "GSMaker", "GrossmanStiglitzConfig", "InformationCollapse",
    "analytic_collapse_cost", "find_information_collapse", "gs_equilibrium_informed_share",
    "gs_metrics", "gs_population", "run_grossman_stiglitz", "run_gs_grid",
    # 9.2 lemons and certification
    "CertificationThreshold", "LemonsCell", "LemonsConfig", "akerlof_pooled_price",
    "certification_bound", "certification_cutoff", "find_certification_threshold",
    "lemons_metrics", "lemons_qualities", "run_lemons", "run_lemons_grid",
    # 10.2 reputation
    "ReputationCell", "ReputationConfig", "ReputationThreshold", "find_reputation_threshold",
    "milking_switch_round", "min_deterrent_whitewash_cost", "predicted_milking_round",
    "reputation_metrics", "reputation_qualities", "run_reputation", "run_reputation_grid",
    # 11.2 switching costs and lock-in
    "LockinThreshold", "SwitchingCell", "SwitchingCostConfig", "acquisition_discount",
    "analytic_lockin_threshold", "find_lockin_threshold", "klemperer_harvest_markup",
    "run_switching_cost", "run_switching_grid", "switching_metrics",
    # 12.2 critical mass and cold start
    "CriticalMassConfig", "PlatformCell", "TippingSeed", "critical_mass", "empty_is_fixed_point",
    "find_tipping_seed", "platform_metrics", "run_critical_mass", "run_platform_grid",
    "tipping_fixed_points",
    # 13.2 congestion and optimal tolls
    "CongestionConfig", "CongestionToll", "QueueCell", "find_congestion_toll",
    "naor_individual_threshold", "naor_optimal_toll", "naor_social_threshold", "queue_metrics",
    "run_congestion", "run_queue_grid", "welfare_per_step",
    # registry
    "Experiment", "UnknownExperimentError", "experiment", "get_experiment", "list_experiments",
    "register_experiment", "resolve_config", "run_from_config", "unregister_experiment",
]
