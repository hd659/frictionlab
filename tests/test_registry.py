"""fl-050 / fl-053 / fl-056: intermediation, Grossman-Stiglitz and lemons in the fl-036 registry and fl-037 CLI."""
import csv
import json
from pathlib import Path

import pytest

from frictionlab import cli
from frictionlab.config import ConfigError
from frictionlab.config.schema import load_config
from frictionlab.experiments import (
    GrossmanStiglitzConfig, IntermediationConfig, get_experiment, list_experiments, resolve_config,
    run_from_config, run_grossman_stiglitz, run_intermediation,
)

CONFIGS = Path(__file__).resolve().parents[1] / "examples" / "configs"
SMALL = CONFIGS / "intermediation_small.json"
GS_SMALL = CONFIGS / "grossman_stiglitz_small.json"
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


# ---- fl-053: Grossman-Stiglitz -------------------------------------------------
@pytest.fixture(scope="module")
def gs_res():
    return run_from_config(GS_SMALL)


def test_grossman_stiglitz_registered_with_preset_and_suggestion():
    assert "grossman_stiglitz" in list_experiments()
    exp = get_experiment("grossman_stiglitz")
    assert exp.master_seed == 53 and exp.replicates == 2 and exp.small_replicates == 1
    assert "information" in exp.doc and exp.runner is not get_experiment("intermediation").runner
    with pytest.raises(ConfigError) as e:
        get_experiment("grossman_stiglits")
    assert "grossman_stiglitz" in str(e.value)


def test_gs_shipped_config_equals_small_preset_and_maps_onto_config():
    raw, cfg = load_config(GS_SMALL), resolve_config(GS_SMALL)
    preset = resolve_config({"experiment": "grossman_stiglitz", "small": True})
    assert raw["params"] == cfg.params == preset.params and cfg.small and cfg.output is None
    assert list(cfg.params) == list(get_experiment("grossman_stiglitz").param_schema)
    d, base = get_experiment("grossman_stiglitz").defaults, GrossmanStiglitzConfig()
    for k in ("center", "jump", "accuracy_bps", "n_acquirers", "n_noise", "maker_rate", "horizon"):
        assert d[k] == getattr(base, k), k


def test_gs_run_from_config_rows_equal_direct_call(gs_res):
    p = gs_res.config["params"]
    keep = {f for f in GrossmanStiglitzConfig.__dataclass_fields__}
    direct = run_grossman_stiglitz(GrossmanStiglitzConfig(**{k: v for k, v in p.items() if k in keep}),
                                   p["signal_costs"], p["noise_intensities"],
                                   population_costs=p["population_costs"],
                                   generations=p["generations"], collapse_floor=p["collapse_floor"],
                                   replicates=1, master_seed=53)
    assert gs_res.rows == direct.rows and gs_res.findings == direct.findings
    assert gs_res.config == resolve_config(GS_SMALL).to_dict()


def test_gs_bad_params_rejected_and_overrides_apply():
    res = run_from_config({"experiment": "grossman_stiglitz", "small": True,
                           "params": {"signal_costs": [3], "noise_intensities": [2.0],
                                      "population_costs": [0]}})
    assert len(res.rows) == 1 and res.rows[0]["signal_cost"] == 3
    for bad in ({"signal_costs": [-1]}, {"accuracy_bps": 4_999}, {"accuracy_bps": 10_001},
                {"collapse_floor": 0.0}, {"n_acquirers": 0}, {"signal_cost": [1]}):
        with pytest.raises(ConfigError):
            resolve_config({"experiment": "grossman_stiglitz", "params": bad})


def test_gs_cli_run_and_parallel_match(tmp_path, capsys, gs_res):
    assert cli.main(["validate", str(GS_SMALL)]) == 0
    assert "-> grossman_stiglitz (master_seed=53, replicates=1, small=true)" in capsys.readouterr().out
    out = tmp_path / "gs"
    assert cli.main(["run", str(GS_SMALL), "--out", str(out)]) == 0
    with open(out / "rows.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [float(r["buyer_fraction"]) for r in rows] == [r["buyer_fraction"] for r in gs_res.rows]
    par = run_from_config({**json.loads(GS_SMALL.read_text()), "n_workers": 2})
    assert par.rows == gs_res.rows


# ---- fl-056: lemons and certification ------------------------------------------
LM_SMALL = CONFIGS / "lemons_small.json"


@pytest.fixture(scope="module")
def lm_res():
    return run_from_config(LM_SMALL)


def test_lemons_registered_defaults_validate_against_schema():
    from frictionlab.config.schema import validate_params
    from frictionlab.experiments import LemonsConfig
    exp = get_experiment("lemons")
    assert "lemons" in list_experiments() and exp.master_seed == 56 and "lemons" in exp.doc
    assert validate_params(exp.defaults, exp.param_schema) == exp.defaults
    assert list(exp.defaults) == list(exp.param_schema)
    base = LemonsConfig()
    assert tuple(exp.defaults["schedule"]) == base.schedule and exp.defaults["n_sellers"] == base.n_sellers
    with pytest.raises(ConfigError) as e:
        get_experiment("lemmons")
    assert "lemons" in str(e.value)


def test_lemons_shipped_config_equals_small_preset():
    raw, cfg = load_config(LM_SMALL), resolve_config(LM_SMALL)
    preset = resolve_config({"experiment": "lemons", "small": True})
    assert raw["params"] == cfg.params == preset.params and cfg.small and cfg.output is None
    assert cfg.master_seed == 56 and cfg.replicates == 1


def test_lemons_run_from_config_rows_equal_direct_call(lm_res):
    from frictionlab.experiments import LemonsConfig, run_lemons
    p = lm_res.config["params"]
    cfg = LemonsConfig(schedule=tuple(p["schedule"]), n_sellers=p["n_sellers"],
                       certification=p["certification"])
    direct = run_lemons(cfg, p["certification_costs"], p["buyer_premiums"],
                        threshold_premium=p["threshold_premium"],
                        uptake_fraction=p["uptake_fraction"], replicates=1, master_seed=56)
    assert lm_res.rows == direct.rows and lm_res.findings == direct.findings
    assert lm_res.config == resolve_config(LM_SMALL).to_dict()


def test_lemons_bad_params_rejected_and_cli_parallel_match(tmp_path, capsys, lm_res):
    for bad in ({"certification_costs": [-1]}, {"n_sellers": 0}, {"uptake_fraction": 0.0},
                {"schedule": [0, 10]}, {"premium": [1]}):
        with pytest.raises(ConfigError):
            resolve_config({"experiment": "lemons", "params": bad})
    assert cli.main(["validate", str(LM_SMALL)]) == 0
    assert "-> lemons (master_seed=56, replicates=1, small=true)" in capsys.readouterr().out
    out = tmp_path / "lm"
    assert cli.main(["run", str(LM_SMALL), "--out", str(out)]) == 0
    with open(out / "rows.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [float(r["traded_share"]) for r in rows] == [r["traded_share"] for r in lm_res.rows]
    par = run_from_config({**json.loads(LM_SMALL.read_text()), "n_workers": 2})
    assert par.rows == lm_res.rows
