"""Phase 6 config layer: typed experiment configs, loaded from JSON or YAML.

- `ExperimentConfig`: experiment name, master_seed, n_workers, replicates,
  small flag, output dir and a `params` mapping validated per experiment.
- `Param`: one schema entry (int / float / bool / str, scalar or list, bounds,
  `friction=True` for non-negative integer-tick frictions).
- `ConfigError`: every problem with its dotted key path, raised together.
- `load_config` / `loads_config`: stdlib JSON; YAML only when PyYAML exists.

The registry (`frictionlab.experiments.registry`) maps names to runners and
param schemas and turns a validated config into an `ExperimentResult`.
"""
from .schema import (
    TOP, ConfigError, ExperimentConfig, Param, Schema, check_mapping, load_config, loads_config,
    validate_params,
)

__all__ = ["TOP", "ConfigError", "ExperimentConfig", "Param", "Schema", "check_mapping",
           "load_config", "loads_config", "validate_params"]
