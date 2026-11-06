"""fl-050: the intermediation experiment in the fl-036 registry and the fl-037 CLI."""
import csv
import json
from pathlib import Path

import pytest

from frictionlab import cli
from frictionlab.config import ConfigError
from frictionlab.config.schema import load_config
from frictionlab.experiments import (
    IntermediationConfig, get_experiment, list_experiments, resolve_config, run_from_config,
    run_intermediation,
)

CONFIGS = Path(__file__).resolve().parents[1] / "examples" / "configs"
SMALL = CONFIGS / "intermediation_small.json"
PHASE7 = {"search_cost", "intermediation"}


@pytest.fixture(scope="module")
def small_res():
    return run_from_config(SMALL)


def test_intermediation_is_registered_next_to_the_other_experiments():
    names = list_experiments()
    assert PHASE7 | {"coase", "fee_wars", "funding_arb"} <= set(names) and names == sorted(names)
    exp = get_experiment("intermediation")
    assert exp.replicates == 2 and exp.small_replicates == 1 and "middleman" in exp.doc
    assert exp.runner is not get_experiment("search_cost").runner


def test_close_name_suggests_intermediation():
    with pytest.raises(ConfigError) as e:
        get_experiment("intermediaton")
    assert "intermediation" in str(e.value)


def test_shipped_small_config_equals_resolved_small_preset():
    raw, cfg = load_config(SMALL), resolve_config(SMALL)
    preset = resolve_config({"experiment": "intermediation", "small": True})
    assert cfg.experiment == "intermediation" and cfg.small and cfg.output is None
    assert raw["params"] == cfg.params == preset.params
    assert cfg.master_seed == preset.master_seed == 50 and cfg.replicates == preset.replicates == 1
    assert list(cfg.params) == list(get_experiment("intermediation").param_schema)


def test_defaults_map_onto_the_experiment_config():
    d = get_experiment("intermediation").defaults
    base = IntermediationConfig()
    for k in ("meeting_rate", "middleman_meeting_rate", "capacity", "holding_cost", "decay",
              "floor", "buyer_weight_bps", "until"):
        assert d[k] == getattr(base, k), k
    assert tuple(d["values"]) == base.values and tuple(d["costs"]) == base.costs


def test_run_from_config_rows_equal_direct_call(small_res):
    p = small_res.config["params"]
    cfg = IntermediationConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in p.items()
                                  if k not in ("search_costs", "half_spreads")})
    direct = run_intermediation(cfg, p["search_costs"], p["half_spreads"], replicates=1,
                                master_seed=50)
    assert small_res.rows == direct.rows and small_res.findings == direct.findings
    assert small_res.config == resolve_config(SMALL).to_dict()


def test_user_params_override_the_preset_and_bad_params_are_rejected():
    res = run_from_config({"experiment": "intermediation", "small": True,
                           "params": {"search_costs": [4], "half_spreads": [2]}})
    assert len(res.rows) == 1 and res.rows[0]["search_cost"] == 4
    for bad in ({"half_spreads": [-1]}, {"capacity": 0}, {"middleman_meeting_rate": 0.0},
                {"holding_cost": -2}, {"spread": [1]}):
        with pytest.raises(ConfigError):
            resolve_config({"experiment": "intermediation", "params": bad})


def test_cli_list_validate_and_run(tmp_path, capsys, small_res):
    assert cli.main(["list"]) == 0
    assert any(line.startswith("intermediation ") for line in capsys.readouterr().out.splitlines())
    assert cli.main(["validate", str(SMALL)]) == 0
    assert "-> intermediation (master_seed=50, replicates=1, small=true)" in capsys.readouterr().out
    out = tmp_path / "run"
    assert cli.main(["run", str(SMALL), "--out", str(out)]) == 0
    with open(out / "rows.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(small_res.rows)
    assert [float(r["intermediated_share"]) for r in rows] == \
        [r["intermediated_share"] for r in small_res.rows]
    assert json.loads((out / "resolved_config.json").read_text()) == small_res.config


def test_cli_run_is_byte_reproducible(tmp_path):
    dirs = [tmp_path / f"r{i}" for i in range(2)]
    assert [cli.main(["run", str(SMALL), "--out", str(d)]) for d in dirs] == [0, 0]
    for name in ("rows.csv", "resolved_config.json"):
        assert (dirs[0] / name).read_bytes() == (dirs[1] / name).read_bytes()


def test_small_config_parallel_matches_serial(small_res):
    raw = json.loads(SMALL.read_text())
    par = run_from_config({**raw, "n_workers": 2})
    assert par.rows == small_res.rows
