"""Experiment registry: name -> (runner, param schema, defaults), driven by configs.

`run_from_config(path)` runs one config via `runner(params, master_seed, replicates, n_workers)`.
"""
from __future__ import annotations

import difflib, importlib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from ..config.schema import ConfigError, ExperimentConfig, Param, load_config, validate_params
from . import (adverse_selection, coase, decision_cost, fee_wars, funding_arb, grossman_stiglitz,
               intermediation, lemons, reputation, search_cost, switching_cost)
from .base import ExperimentResult
from .two_venue import MarketConfig
Runner = Callable[[Dict[str, Any], int, int, int], ExperimentResult]; ConfigLike = Union[str, Path, Mapping[str, Any], ExperimentConfig]


class UnknownExperimentError(ConfigError, LookupError):
    """Config error for an unregistered experiment name, with a close-match hint."""

    def __init__(self, name: str, suggestion: Optional[str]) -> None:
        hint = f"; did you mean {suggestion!r}?" if suggestion else ""
        super().__init__([("experiment", f"unknown experiment {name!r}{hint} "
                                          f"(registered: {', '.join(list_experiments())})")])
        self.name, self.suggestion = name, suggestion


@dataclass(frozen=True)
class Experiment:
    """A registered experiment: runner, param schema, defaults, small preset and seeds."""
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
    return lambda fn: register_experiment(name, fn, param_schema, defaults, **kw).runner


def unregister_experiment(name: str) -> None:
    """Remove `name` from the registry (no-op if absent); used by tests and plugins."""
    _REGISTRY.pop(name, None)


def list_experiments() -> List[str]:
    """Registered experiment names, sorted."""
    return sorted(_REGISTRY)


def get_experiment(name: str) -> Experiment:
    """Look up a registered experiment; raises `UnknownExperimentError` with a hint."""
    try:
        return _REGISTRY[name]
    except KeyError:
        close = difflib.get_close_matches(str(name), list(_REGISTRY), n=1, cutoff=0.4)
        raise UnknownExperimentError(str(name), close[0] if close else None) from None


def resolve_config(config: ConfigLike) -> ExperimentConfig:
    """Load (if a path), validate and fill every default; the result is self-contained."""
    if isinstance(config, (str, Path)):
        config = load_config(config)
    config = config.to_dict() if isinstance(config, ExperimentConfig) else config
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


# ---- Phase 5 experiments; Phase 7 search; Phase 8 information; Phase 9-14
def _build(cls: type, p: Mapping[str, Any], **extra: Any) -> Any:
    names = {f.name for f in fields(cls)}
    return cls(**{**{k: tuple(v) if isinstance(v, list) else v for k, v in p.items() if k in names}, **extra})


def _pick(obj: Any, *keys: str, **extra: Any) -> Dict[str, Any]:
    return {**{k: getattr(obj, k) for k in keys}, **extra}

INTS, RATE, TICKS = Param("int", seq=True), Param("float", min=1e-9), Param("int", seq=True, friction=True)
POS, NONNEG, PROB = Param("int", min=1), Param("float", min=0), Param("float", min=0, max=1)
BPS, FRIC, SCHED = Param("int", min=0, max=10_000), Param("int", friction=True), Param("int", seq=True, min=1)
register_experiment(
    "coase", lambda p, s, r, w: coase.run_coase(_build(coase.Config, p), p["levels"], replicates=r, master_seed=s, n_workers=w),
    {"values": INTS, "costs": INTS, "levels": Param("int", seq=True, friction=True, nullable=True,
                                                    doc="F grid; null -> 0..analytic bound"),
     "taker_share": PROB, "max_wakeups": POS},
    _pick(coase.Config(), "values", "costs", "taker_share", "max_wakeups", levels=None),
    small={"values": [150, 140, 130, 120], "costs": [70, 80, 90, 100], "max_wakeups": 1_500},
    master_seed=28, replicates=4, small_replicates=3, doc="Coase threshold: volume vs total friction")
register_experiment(
    "decision_cost", lambda p, s, r, w: decision_cost.run_decision_cost(
        _build(decision_cost.Config, p), p["costs"], p["aspirations"], replicates=r, master_seed=s, n_workers=w),
    {"costs": TICKS, "aspirations": Param("float", seq=True, min=0), "max_steps": POS,
     "horizon": NONNEG, "n_br": POS, "n_zi": POS, "kappa": NONNEG},
    _pick(decision_cost.Config(), "max_steps", "horizon", "n_br", "n_zi", "kappa",
          costs=[0, 1, 2, 4, 8], aspirations=[100.0, 500.0, 2000.0]),
    small={"max_steps": 20, "horizon": 60.0, "costs": [0, 4, 16], "aspirations": [100.0, 500.0]},
    master_seed=29, replicates=4, small_replicates=2, doc="satisficer vs optimizer under step cost")
register_experiment(
    "adverse_selection", lambda p, s, r, w: adverse_selection.run_grid(
        _build(adverse_selection.Config, p), p["shares"], p["jumps"], replicates=r, master_seed=s, n_workers=w),
    {"shares": Param("float", seq=True, min=0, max=1), "jumps": Param("int", seq=True, min=1),
     "horizon": NONNEG, "taker_bps": FRIC, "alpha": POS, "batch_interval": NONNEG},
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
     "start_B": Param("int", seq=True, length=2), "max_rounds": POS, "window": POS, "steps": POS, "epoch": POS},
    _pick(_FW, "makers", "takers", "max_rounds", "window", start_A=list(_FW.start[0]),
          start_B=list(_FW.start[1]), steps=_FW.market.steps, epoch=_FW.market.epoch),
    small={"steps": 120, "epoch": 20, "makers": [-2, -1, 0], "takers": [1, 2, 3, 4], "max_rounds": 6},
    master_seed=32, replicates=_FW.replicates, small_replicates=1, doc="two-venue Bertrand fee undercutting")
register_experiment(
    "funding_arb", lambda p, s, r, w: funding_arb.run_funding_arb(
        _build(funding_arb.Config, p), p["levels"], p["probs"], replicates=r, master_seed=s, n_workers=w),
    {"levels": Param("int", seq=True, friction=True, nullable=True, doc="shock gas grid; null -> default_levels"),
     "probs": Param("float", seq=True, min=0, max=1), "calm_gas": FRIC, "p_calm": PROB,
     "decision_cost": FRIC, "taker_bps": FRIC, "lp_fee_bps": FRIC, "horizon": POS, "until": NONNEG},
    _pick(funding_arb.Config(), "calm_gas", "p_calm", "decision_cost", "taker_bps", "lp_fee_bps",
          "horizon", "until", levels=None, probs=[0.0, 0.1, 0.3, 0.6]),
    small={"probs": [0.0, 0.3, 0.6]}, master_seed=34, doc="funding-rate arbitrage under gas shocks")
register_experiment(
    "search_cost", lambda p, s, r, w: search_cost.run_search_cost(
        _build(search_cost.Config, p), p["search_costs"], p["decays"], replicates=r, master_seed=s, n_workers=w),
    {"search_costs": TICKS, "decays": TICKS, "values": INTS, "costs": INTS, "meeting_rate": Param("float", min=1e-9),
     "buyer_weight_bps": BPS, "floor": POS, "until": NONNEG},
    _pick(search_cost.Config(), "values", "costs", "meeting_rate", "buyer_weight_bps", "floor",
          "until", search_costs=[0, 5, 10, 20, 30, 40, 52], decays=[0, 1, 2, 4]),
    small={"search_costs": [0, 10, 30, 52], "decays": [0, 2], "until": 100.0},
    master_seed=47, replicates=2, small_replicates=1, doc="search cost x perishability in a bilateral search market")
register_experiment(
    "intermediation", lambda p, s, r, w: intermediation.run_intermediation(
        _build(intermediation.Config, p), p["search_costs"], p["half_spreads"], replicates=r, master_seed=s,
        n_workers=w),
    {"search_costs": TICKS, "half_spreads": TICKS, "values": INTS, "costs": INTS,
     "meeting_rate": RATE, "middleman_meeting_rate": RATE, "capacity": POS, "holding_cost": FRIC,
     "decay": FRIC, "floor": POS, "buyer_weight_bps": BPS, "until": NONNEG},
    _pick(intermediation.Config(), "values", "costs", "meeting_rate", "middleman_meeting_rate",
          "capacity", "holding_cost", "decay", "floor", "buyer_weight_bps", "until",
          search_costs=[0, 1, 2, 4, 8, 16, 32], half_spreads=[0, 2, 4, 8, 16, 24]),
    small={"search_costs": [0, 2, 8, 32], "half_spreads": [0, 4, 16], "until": 100.0},
    master_seed=50, replicates=2, small_replicates=1, doc="middleman half-spread x search cost in a search market")
_GS = grossman_stiglitz
register_experiment(
    "grossman_stiglitz", lambda p, s, r, w: _GS.run_grossman_stiglitz(
        _build(_GS.Config, p), p["signal_costs"], p["noise_intensities"], replicates=r, master_seed=s, n_workers=w,
        **{k: p[k] for k in ("population_costs", "generations", "collapse_floor")}),
    {"signal_costs": TICKS, "noise_intensities": Param("float", seq=True, min=0), "center": POS,
     "jump": POS, "accuracy_bps": Param("int", min=5_000, max=10_000), "n_acquirers": POS,
     "acquirer_rate": RATE, "n_noise": POS, "noise_buy_prob": PROB, "maker_rate": RATE,
     "quote_size": POS, "horizon": NONNEG, "population_costs": TICKS, "generations": POS,
     "collapse_floor": Param("float", min=1e-9, max=1)},
    _pick(_GS.Config(), *(f.name for f in fields(_GS.Config) if f.name != "cash"),
          signal_costs=list(_GS.COSTS), noise_intensities=list(_GS.NOISES),
          population_costs=[0, 4, 9], generations=8, collapse_floor=0.05),
    small={"signal_costs": [0, 2, 4, 8], "noise_intensities": [1.0, 4.0], "horizon": 40.0},
    master_seed=53, replicates=2, small_replicates=1, doc="signal cost x noise: who buys costly information")
_LM, LM_POS = lemons.Config(), {"threshold_premium": 20, "uptake_fraction": 0.05}
register_experiment(
    "lemons", lambda p, s, r, w: lemons.run_lemons(_build(lemons.Config, p), p["certification_costs"],
                                                   p["buyer_premiums"], replicates=r, master_seed=s,
                                                   n_workers=w, **{k: p[k] for k in LM_POS}),
    {"certification_costs": TICKS, "buyer_premiums": TICKS, "schedule": SCHED, "n_sellers": POS, "certification":
     Param("bool"), "threshold_premium": FRIC, "uptake_fraction": Param("float", min=1e-9, max=1)},
    _pick(_LM, "n_sellers", "certification", schedule=list(_LM.schedule), **LM_POS,
          certification_costs=list(lemons.COSTS), buyer_premiums=list(lemons.PREMIUMS)),
    small={"certification_costs": [0, 4, 16, 32], "buyer_premiums": [0, 10, 40], "n_sellers": 20},
    master_seed=56, doc="certification cost x buyer premium in an Akerlof lemons market")
_RP, RP_SMALL = reputation.Config(), {"rating_accuracies": [6000, 8500, 10000], "whitewash_costs": [0, 4, 16, 64], "n_sellers": 20, "horizon": 6}
register_experiment(
    "reputation", lambda p, s, r, w: reputation.run_reputation(
        _build(reputation.Config, p), p["rating_accuracies"], p["whitewash_costs"],
        threshold_accuracy=p["threshold_accuracy"], replicates=r, master_seed=s, n_workers=w),
    {"rating_accuracies": Param("int", seq=True, min=0, max=10_000), "whitewash_costs": TICKS, "schedule": SCHED,
     "n_sellers": POS, "horizon": POS, "premium": FRIC, "opportunist_bps": BPS, "threshold_accuracy": BPS},
    _pick(_RP, "n_sellers", "horizon", "premium", "opportunist_bps", schedule=list(_RP.schedule), threshold_accuracy=8500,
          rating_accuracies=list(reputation.ACCURACIES), whitewash_costs=list(reputation.WASH_COSTS)),
    small=RP_SMALL, master_seed=59, doc="rating accuracy x whitewash cost: reputation vs cheap pseudonyms")
_SC = switching_cost.Config()
register_experiment(
    "switching_cost", lambda p, s, r, w: switching_cost.run_switching_cost(
        _build(_SC.__class__, p), p["mean_switching_costs"], p["target_shares"], replicates=r, master_seed=s,
        n_workers=w, **{k: p[k] for k in ("threshold_target", "min_markup")}),
    {"mean_switching_costs": TICKS, "target_shares": Param("int", seq=True, min=5_001, max=10_000), "valuations": SCHED,
     "n_consumers": POS, "horizon": POS, "marginal_cost": POS, "spread": FRIC,
     "threshold_target": Param("int", min=5_001, max=10_000), "min_markup": FRIC},
    _pick(_SC, "n_consumers", "horizon", "marginal_cost", "spread", valuations=list(_SC.valuations), threshold_target=6000,
          min_markup=4, mean_switching_costs=list(switching_cost.COSTS), target_shares=list(switching_cost.TARGETS)),
    small={"mean_switching_costs": [0, 4, 16, 32], "target_shares": [6000, 9000], "n_consumers": 16, "horizon": 6},
    master_seed=62, doc="mean switching cost x target share: Klemperer lock-in and harvest markup")
for _n, _seed, _doc in (("critical_mass", 65, "adoption cost x seeded members: critical mass and cold start"),
                        ("congestion", 68, "toll x waiting cost: Naor congestion and the optimal toll"),
                        ("menu_cost", 71, "menu cost x inflation: (S, s) bands and price rigidity")):
    _m = importlib.import_module(f".{_n}", __package__)   # the package attr may be a same-named function
    register_experiment(_n, _m.registry_runner, _m.PARAM_SCHEMA, _m.defaults(), small=_m.SMALL, master_seed=_seed, doc=_doc)
