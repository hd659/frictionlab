"""fl-036: typed config schema, all-errors-at-once validation, JSON/YAML loading."""
import builtins
import json

import pytest

from frictionlab.config import ConfigError, ExperimentConfig, Param, load_config, loads_config
from frictionlab.config.schema import validate_params
from frictionlab.experiments.registry import get_experiment, list_experiments, resolve_config

SCHEMA = {"levels": Param("int", seq=True, friction=True), "share": Param("float", min=0, max=1),
          "fee": Param("int", friction=True), "flag": Param("bool"),
          "pair": Param("int", seq=True, length=2)}


def errors_of(fn, *a, **kw):
    with pytest.raises(ConfigError) as ei:
        fn(*a, **kw)
    return dict(ei.value.errors)


@pytest.mark.parametrize("name", ["coase", "decision_cost", "adverse_selection", "fee_wars",
                                  "funding_arb"])
def test_every_registered_default_validates(name):
    exp = get_experiment(name)
    assert set(exp.defaults) == set(exp.param_schema)
    assert validate_params(exp.defaults, exp.param_schema) == exp.defaults
    assert set(exp.resolve_params(small=True)) == set(exp.param_schema)


def test_all_five_phase5_experiments_registered():
    assert {"coase", "decision_cost", "adverse_selection", "fee_wars",
            "funding_arb"} <= set(list_experiments())


def test_unknown_key_named_with_path():
    errs = errors_of(validate_params, {"fee": 1, "bogus": 2}, SCHEMA)
    assert list(errs) == ["params.bogus"] and "unknown key" in errs["params.bogus"]


@pytest.mark.parametrize("value,path", [("3", "params.fee"), (1.5, "params.fee"),
                                        (True, "params.fee"), ([1, "x"], "params.levels[1]"),
                                        (7, "params.levels"), ("yes", "params.flag")])
def test_bad_types_named_with_path(value, path):
    key = path.split(".")[1].split("[")[0]
    errs = errors_of(validate_params, {key: value}, SCHEMA)
    assert path in errs and "expected" in errs[path]


@pytest.mark.parametrize("data,path", [({"fee": -1}, "params.fee"),
                                       ({"levels": [0, 5, -2]}, "params.levels[2]")])
def test_negative_friction_rejected(data, path):
    errs = errors_of(validate_params, data, SCHEMA)
    assert "negative friction" in errs[path]


def test_bounds_length_and_empty():
    errs = errors_of(validate_params, {"share": 1.5, "pair": [1], "levels": []}, SCHEMA)
    assert set(errs) == {"params.share", "params.pair", "params.levels"}


def test_multiple_errors_reported_together():
    raw = {"experiment": "coase", "bogus": 1, "master_seed": "x", "n_workers": 0,
           "params": {"levels": [0, -3], "foo": 2, "values": "a"}}
    with pytest.raises(ConfigError) as ei:
        resolve_config(raw)
    paths = set(ei.value.paths)
    assert paths == {"bogus", "master_seed", "n_workers", "params.levels[1]", "params.foo",
                     "params.values"}
    msg = str(ei.value)
    assert "6 config error(s)" in msg and all(p in msg for p in paths)


def test_missing_experiment_and_non_mapping():
    assert "experiment" in errors_of(ExperimentConfig.from_dict, {"master_seed": 1})
    assert "<root>" in errors_of(ExperimentConfig.from_dict, [1, 2])
    assert "params" in errors_of(ExperimentConfig.from_dict, {"experiment": "x", "params": [1]})


def test_floats_coerced_and_order_follows_schema():
    out = validate_params({"fee": 2, "share": 1}, SCHEMA)
    assert out == {"fee": 2, "share": 1.0} and list(out) == ["share", "fee"]  # schema order
    assert isinstance(out["share"], float)


def test_param_definition_guards():
    with pytest.raises(ValueError):
        Param("complex")
    with pytest.raises(ValueError):
        Param("float", friction=True)


def test_json_round_trip_of_resolved_config_is_lossless():
    for name in list_experiments():
        cfg = resolve_config({"experiment": name, "small": True})
        again = resolve_config(json.loads(cfg.to_json()))
        assert again == cfg and again.to_json() == cfg.to_json()
        assert ExperimentConfig(**{**json.loads(cfg.to_json())}) == cfg


def test_load_json_file_and_bad_extension(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"experiment": "coase"}))
    assert load_config(p) == {"experiment": "coase"}
    bad = tmp_path / "c.toml"
    bad.write_text("")
    assert "<file>" in errors_of(load_config, bad)
    assert "invalid JSON" in errors_of(loads_config, "{nope")["<file>"]


def test_yaml_loads_when_available(tmp_path):
    pytest.importorskip("yaml")
    p = tmp_path / "c.yaml"
    p.write_text("experiment: coase\nmaster_seed: 3\nparams:\n  levels: [0, 10]\n")
    assert resolve_config(p).params["levels"] == [0, 10]


def test_yaml_without_pyyaml_gives_clear_error(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == "yaml":
            raise ImportError("no yaml")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    msg = errors_of(loads_config, "experiment: coase", "yaml")["<file>"]
    assert "PyYAML" in msg and ".json" in msg
