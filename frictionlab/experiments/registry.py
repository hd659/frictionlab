"""Experiment registry: name -> (runner, param schema, defaults), driven by configs.

    from frictionlab.experiments.registry import list_experiments, run_from_config
    list_experiments()        # ['adverse_selection', 'coase', 'decision_cost', ...]
    res = run_from_config("examples/configs/coase_small.json")
    res.config                # resolved config, also shown in res.report() header

A runner has signature `runner(params, master_seed, replicates, n_workers)` and
returns an `ExperimentResult`. Resolution is deterministic: registered defaults,
then the experiment's `small` overrides (if `small: true`), then user params,
validated and emitted in schema order.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from ..config.schema import ConfigError, ExperimentConfig, Param, load_config, validate_params
from . import adverse_selection, coase, decision_cost, fee_wars, funding_arb
from .base import ExperimentResult
from .two_venue import MarketConfig

Runner = Callable[[Dict[str, Any], int, int, int], ExperimentResult]
ConfigLike = Union[str, Path, Mapping[str, Any], ExperimentConfig]


class UnknownExperimentError(ConfigError, LookupError):
    def __init__(self, name: str, suggestion: Optional[str]) -> None:
        hint = f"; did you mean {suggestion!r}?" if suggestion else ""
        super().__init__([("experiment", f"unknown experiment {name!r}{hint} "
                                          f"(registered: {', '.join(list_experiments())})")])
        self.name, self.suggestion = name, suggestion


@dataclass(frozen=True)
class Experiment:
    name: str
    runner: Runner = field(repr=False)
    param_schema: Dict[str, Param] = field(repr=False)
    defaults: Dict[str, Any] = field(repr=False)
    small: Dict[str, Any] = field(default_factory=dict, repr=False)
    master_seed: int = 0
    replicates: int = 1
    small_replicates: Optional[int] = None
    doc: str = ""

    def resolve_params(self, params: Mapping[str, Any] = (), small: bool = False) -> Dict[str, Any]:
        merged = {**self.defaults, **(self.small if small else {}), **dict(params)}
        return validate_params(merged, self.param_schema)


_REGISTRY: Dict[str, Experiment] = {}


def register_experiment(name: str, runner: Runner, param_schema: Mapping[str, Param],
                        defaults: Mapping[str, Any], *, small: Optional[Mapping[str, Any]] = None,
                        master_seed: int = 0, replicates: int = 1,
                        small_replicates: Optional[int] = None, doc: str = "") -> Experiment:
    """Register `runner` under `name`; defaults must cover and satisfy the schema."""
    if name in _REGISTRY:
        raise ValueError(f"experiment {name!r} is already registered")
    schema = dict(param_schema)
    if set(defaults) != set(schema):
        raise ValueError(f"{name}: defaults keys {sorted(defaults)} != schema keys {sorted(schema)}")
    exp = Experiment(name, runner, schema, validate_params(defaults, schema),
                     dict(small or {}), int(master_seed), int(replicates), small_replicates, doc)
    exp.resolve_params(small=True)                    # small overrides must validate too
    _REGISTRY[name] = exp
    return exp


def experiment(name: str, param_schema: Mapping[str, Param], defaults: Mapping[str, Any],
               **kw: Any) -> Callable[[Runner], Runner]:
    """Decorator form of `register_experiment`; returns the runner unchanged."""
    def deco(fn: Runner) -> Runner:
        register_experiment(name, fn, param_schema, defaults, **kw)
        return fn
    return deco


def unregister_experiment(name: str) -> None:
    _REGISTRY.pop(name, None)


def list_experiments() -> List[str]:
    return sorted(_REGISTRY)


def get_experiment(name: str) -> Experiment:
    try:
        return _REGISTRY[name]
    except KeyError:
        close = difflib.get_close_matches(str(name), list(_REGISTRY), n=1, cutoff=0.4)
        raise UnknownExperimentError(str(name), close[0] if close else None) from None


def resolve_config(config: ConfigLike) -> ExperimentConfig:
    """Load (if a path), validate and fill every default; the result is self-contained."""
    if isinstance(config, (str, Path)):
        config = load_config(config)
    if isinstance(config, ExperimentConfig):
        config = config.to_dict()
    if not isinstance(config, Mapping):
        raise ConfigError([("<root>", f"expected a mapping, got {type(config).__name__}")])
    name = config.get("experiment")
    exp = get_experiment(name) if isinstance(name, str) else None
    cfg = ExperimentConfig.from_dict(config, exp.param_schema if exp else None)
    reps = cfg.replicates or (exp.small_replicates if cfg.small and exp.small_replicates
                              else exp.replicates)
    seed = exp.master_seed if cfg.master_seed is None else cfg.master_seed
    return replace(cfg, params=exp.resolve_params(cfg.params, cfg.small),
                   master_seed=seed, replicates=reps)


def run_from_config(config: ConfigLike) -> ExperimentResult:
    """Resolve `config`, run its experiment, and attach the resolved config to the result."""
    cfg = resolve_config(config)
    res = get_experiment(cfg.experiment).runner(dict(cfg.params), cfg.master_seed,
                                                cfg.replicates, cfg.n_workers)
    return replace(res, config=cfg.to_dict())


# ---- the five Phase 5 experiments --------------------------------------------
def _build(cls: type, p: Mapping[str, Any], **extra: Any) -> Any:
    names = {f.name for f in fields(cls)}
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in p.items() if k in names}
    return cls(**{**kw, **extra})


def _pick(obj: Any, *keys: str, **extra: Any) -> Dict[str, Any]:
    return {**{k: getattr(obj, k) for k in keys}, **extra}


INTS, FLOATS = Param("int", seq=True), Param("float", seq=True)
TICKS = Param("int", seq=True, friction=True)
POS, NONNEG, PROB = Param("int", min=1), Param("float", min=0), Param("float", min=0, max=1)

register_experiment(
    "coase", lambda p, s, r, w: coase.run_coase(_build(coase.Config, p), p["levels"], replicates=r,
                                                master_seed=s, n_workers=w),
    {"values": INTS, "costs": INTS, "levels": Param("int", seq=True, friction=True, nullable=True,
                                                    doc="F grid; null -> 0..analytic bound"),
     "taker_share": PROB, "max_wakeups": POS},
    _pick(coase.Config(), "values", "costs", "taker_share", "max_wakeups", levels=None),
    small={"values": [150, 140, 130, 120], "costs": [70, 80, 90, 100], "max_wakeups": 1_500},
    master_seed=28, replicates=4, small_replicates=3, doc="Coase threshold: volume vs total friction")

register_experiment(
    "decision_cost", lambda p, s, r, w: decision_cost.run_decision_cost(
        _build(decision_cost.Config, p), p["costs"], p["aspirations"], replicates=r,
        master_seed=s, n_workers=w),
    {"costs": TICKS, "aspirations": Param("float", seq=True, min=0), "max_steps": POS,
     "horizon": NONNEG, "n_br": POS, "n_zi": POS, "kappa": NONNEG},
    _pick(decision_cost.Config(), "max_steps", "horizon", "n_br", "n_zi", "kappa",
          costs=[0, 1, 2, 4, 8], aspirations=[100.0, 500.0, 2000.0]),
    small={"max_steps": 20, "horizon": 60.0, "costs": [0, 4, 16], "aspirations": [100.0, 500.0]},
    master_seed=29, replicates=4, small_replicates=2, doc="satisficer vs optimizer under step cost")

register_experiment(
    "adverse_selection", lambda p, s, r, w: adverse_selection.run_grid(
        _build(adverse_selection.Config, p), p["shares"], p["jumps"], replicates=r,
        master_seed=s, n_workers=w),
    {"shares": Param("float", seq=True, min=0, max=1), "jumps": Param("int", seq=True, min=1),
     "horizon": NONNEG, "taker_bps": Param("int", friction=True), "alpha": POS,
     "batch_interval": NONNEG},
    _pick(adverse_selection.Config(), "horizon", "taker_bps", "alpha", "batch_interval",
          shares=list(adverse_selection.SHARES), jumps=list(adverse_selection.JUMPS)),
    small={"shares": [0.0, 0.25, 0.5], "jumps": [2, 8]},
    master_seed=30, replicates=3, small_replicates=2, doc="informed share x jump vs GM spread")

_FW = fee_wars.Config()
register_experiment(
    "fee_wars", lambda p, s, r, w: fee_wars.run_fee_wars(
        _build(fee_wars.Config, p, market=MarketConfig(steps=p["steps"], epoch=p["epoch"]),
               start=(tuple(p["start_A"]), tuple(p["start_B"])), replicates=r),
        master_seed=s, n_workers=w),
    {"makers": INTS, "takers": TICKS, "start_A": Param("int", seq=True, length=2),
     "start_B": Param("int", seq=True, length=2), "max_rounds": POS, "window": POS,
     "steps": POS, "epoch": POS},
    _pick(_FW, "makers", "takers", "max_rounds", "window", start_A=list(_FW.start[0]),
          start_B=list(_FW.start[1]), steps=_FW.market.steps, epoch=_FW.market.epoch),
    small={"steps": 120, "epoch": 20, "makers": [-2, -1, 0], "takers": [1, 2, 3, 4],
           "max_rounds": 6},
    master_seed=32, replicates=_FW.replicates, small_replicates=1,
    doc="two-venue Bertrand fee undercutting")

register_experiment(
    "funding_arb", lambda p, s, r, w: funding_arb.run_funding_arb(
        _build(funding_arb.Config, p), p["levels"], p["probs"], replicates=r, master_seed=s,
        n_workers=w),
    {"levels": Param("int", seq=True, friction=True, nullable=True,
                     doc="shock gas grid; null -> default_levels"),
     "probs": Param("float", seq=True, min=0, max=1), "calm_gas": Param("int", friction=True),
     "p_calm": PROB, "decision_cost": Param("int", friction=True),
     "taker_bps": Param("int", friction=True), "lp_fee_bps": Param("int", friction=True),
     "horizon": POS, "until": NONNEG},
    _pick(funding_arb.Config(), "calm_gas", "p_calm", "decision_cost", "taker_bps", "lp_fee_bps",
          "horizon", "until", levels=None, probs=[0.0, 0.1, 0.3, 0.6]),
    small={"probs": [0.0, 0.3, 0.6]},
    master_seed=34, replicates=1, small_replicates=1, doc="funding-rate arbitrage under gas shocks")
