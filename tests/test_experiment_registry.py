"""fl-036: experiment registry, run_from_config parity, determinism, provenance."""
import json
from pathlib import Path

import pytest

from frictionlab.config import ConfigError, Param
from frictionlab.experiments import (
    CoaseConfig, ExperimentResult, FundingArbConfig, UnknownExperimentError, experiment,
    get_experiment, list_experiments, register_experiment, resolve_config, run_coase,
    run_from_config, run_funding_arb,
)
from frictionlab.experiments.registry import unregister_experiment

CONFIGS = Path(__file__).resolve().parents[1] / "examples" / "configs"
COASE, FUNDING = CONFIGS / "coase_small.json", CONFIGS / "funding_arb_small.json"


def strip(rows):
    return json.dumps(rows, sort_keys=True, default=str)


@pytest.fixture(scope="module")
def coase_res():
    return run_from_config(COASE)


@pytest.fixture(scope="module")
def funding_res():
    return run_from_config(FUNDING)


def test_shipped_configs_validate():
    for p in (COASE, FUNDING):
        cfg = resolve_config(p)
        assert cfg.experiment in list_experiments() and cfg.small and cfg.output is None


def test_coase_config_matches_direct_call(coase_res):
    raw = json.loads(COASE.read_text())
    p = raw["params"]
    direct = run_coase(CoaseConfig(values=tuple(p["values"]), costs=tuple(p["costs"]),
                                   taker_share=p["taker_share"], max_wakeups=p["max_wakeups"]),
                       p["levels"], replicates=raw["replicates"], master_seed=raw["master_seed"])
    assert strip(coase_res.rows) == strip(direct.rows)
    assert coase_res.params == direct.params and coase_res.findings == direct.findings


def test_funding_config_matches_direct_call(funding_res):
    raw = json.loads(FUNDING.read_text())
    p = raw["params"]
    direct = run_funding_arb(FundingArbConfig(calm_gas=p["calm_gas"], decision_cost=p["decision_cost"]),
                             p["levels"], p["probs"], replicates=raw["replicates"],
                             master_seed=raw["master_seed"])
    assert strip(funding_res.rows) == strip(direct.rows)


def test_deterministic_under_fixed_master_seed(coase_res):
    assert strip(run_from_config(COASE).rows) == strip(coase_res.rows)
    other = run_from_config({**json.loads(COASE.read_text()), "master_seed": 29})
    assert other.master_seed == 29 and other.config["master_seed"] == 29


def test_resolved_config_stored_and_in_report_header(coase_res):
    cfg = coase_res.config
    assert cfg == resolve_config(COASE).to_dict()
    assert json.loads(json.dumps(cfg)) == cfg
    html = coase_res.report().to_html()
    assert "config" in html and "max_wakeups" in html


def test_unknown_experiment_suggests_close_match():
    with pytest.raises(UnknownExperimentError) as ei:
        get_experiment("coaes")
    assert ei.value.suggestion == "coase" and "did you mean 'coase'" in str(ei.value)
    assert isinstance(ei.value, ConfigError)
    with pytest.raises(UnknownExperimentError) as ei:
        run_from_config({"experiment": "funding_arbs"})
    assert ei.value.suggestion == "funding_arb"
    with pytest.raises(UnknownExperimentError) as ei:
        get_experiment("zzzzzz")
    assert ei.value.suggestion is None


def test_defaults_merged_under_user_params():
    small = resolve_config({"experiment": "coase", "small": True})
    assert small.params["values"] == [150, 140, 130, 120] and small.replicates == 3
    assert small.master_seed == 28
    user = resolve_config({"experiment": "coase", "small": True, "params": {"max_wakeups": 99}})
    assert user.params == {**small.params, "max_wakeups": 99}
    full = resolve_config({"experiment": "coase"})
    assert full.params["values"] == list(CoaseConfig().values) and full.replicates == 4
    assert list(user.params) == list(get_experiment("coase").param_schema)


def _toy(p, seed, reps, workers):
    rows = [{"cell": 0, "replicate": i, "seed": seed + i, "error": None, "x": p["k"] * i}
            for i in range(reps)]
    return ExperimentResult("toy", seed, {"k": [p["k"]]}, rows)


def test_register_duplicate_raises_and_decorator_form():
    schema, defaults = {"k": Param("int", friction=True)}, {"k": 2}
    try:
        register_experiment("toy_fl036", _toy, schema, defaults, master_seed=5, replicates=2)
        with pytest.raises(ValueError, match="already registered"):
            register_experiment("toy_fl036", _toy, schema, defaults)
        res = run_from_config({"experiment": "toy_fl036", "params": {"k": 3}})
        assert [r["x"] for r in res.rows] == [0, 3] and res.master_seed == 5
        with pytest.raises(ConfigError) as ei:
            run_from_config({"experiment": "toy_fl036", "params": {"k": -1}})
        assert ei.value.paths == ["params.k"]

        @experiment("toy_fl036_deco", schema, defaults)
        def runner(p, s, r, w):
            return _toy(p, s, r, w)

        assert runner is not None and "toy_fl036_deco" in list_experiments()
        with pytest.raises(ValueError):
            experiment("toy_fl036_deco", schema, defaults)(runner)
    finally:
        unregister_experiment("toy_fl036")
        unregister_experiment("toy_fl036_deco")
    assert "toy_fl036" not in list_experiments()


def test_bad_registration_rejected():
    with pytest.raises(ValueError):
        register_experiment("bad_fl036", _toy, {"k": Param("int")}, {})
    with pytest.raises(ConfigError):
        register_experiment("bad_fl036", _toy, {"k": Param("int", friction=True)}, {"k": -4})
    assert "bad_fl036" not in list_experiments()
