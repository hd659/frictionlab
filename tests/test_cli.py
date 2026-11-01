"""fl-037: argparse CLI over the fl-036 registry; main(argv) is called in-process."""
import json
import runpy
import sys
from pathlib import Path

import pytest

from frictionlab import cli
from frictionlab.analysis.sweep import write_csv
from frictionlab.experiments import resolve_config, run_from_config

CONFIGS = Path(__file__).resolve().parents[1] / "examples" / "configs"
COASE, FEE_WARS = CONFIGS / "coase_small.json", CONFIGS / "fee_wars_small.json"
FIVE = {"coase", "decision_cost", "adverse_selection", "fee_wars", "funding_arb"}


def call(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


def write(tmp_path, data, name="c.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    dirs = [tmp_path_factory.mktemp(f"run{i}") for i in range(2)]
    codes = [cli.main(["run", str(COASE), "--out", str(d)]) for d in dirs]
    return codes, dirs


def test_run_writes_three_artifacts_matching_run_from_config(two_runs):
    codes, (d, _) = two_runs
    assert codes == [0, 0]
    assert all((d / name).is_file() for name in cli.ARTIFACTS)
    res = run_from_config(COASE)
    assert (d / "rows.csv").read_text() == write_csv(res.rows)
    assert json.loads((d / "resolved_config.json").read_text()) == res.config
    html = (d / "report.html").read_text()
    assert html.startswith("<!DOCTYPE html>") and "max_wakeups" in html


def test_same_seed_runs_are_byte_identical(two_runs):
    _, (a, b) = two_runs
    for name in cli.ARTIFACTS:
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_precedence_file_then_flags_then_set(capsys):
    code, out, _ = call(capsys, "show-config", COASE)
    assert code == 0 and json.loads(out)["master_seed"] == 28
    _, out, _ = call(capsys, "show-config", COASE, "--seed", 5, "--workers", 2)
    cfg = json.loads(out)
    assert cfg["master_seed"] == 5 and cfg["n_workers"] == 2
    _, out, _ = call(capsys, "show-config", COASE, "--seed", 5, "--set", "master_seed=9",
                     "--set", "params.levels=[0,10]", "--set", "params.max_wakeups=99",
                     "--set", "params.max_wakeups=77")
    cfg = json.loads(out)
    assert cfg["master_seed"] == 9 and cfg["params"]["levels"] == [0, 10]
    assert cfg["params"]["max_wakeups"] == 77                    # later --set wins


def test_small_flag_and_set_override_it(capsys, tmp_path):
    p = write(tmp_path, {"experiment": "coase"})
    _, out, _ = call(capsys, "show-config", p, "--small")
    assert json.loads(out)["small"] is True and json.loads(out)["replicates"] == 3
    _, out, _ = call(capsys, "show-config", p, "--small", "--set", "small=false")
    assert json.loads(out)["small"] is False and json.loads(out)["replicates"] == 4


def test_validate_reports_every_bad_path_and_exits_2(capsys, tmp_path):
    p = write(tmp_path, {"experiment": "coase", "bogus": 1, "params": {"levels": [0, -3]}})
    code, out, err = call(capsys, "validate", p)
    assert code == 2 and out == ""
    assert "2 config error(s)" in err and "bogus" in err and "params.levels[1]" in err
    assert "Traceback" not in err


def test_validate_ok_and_does_not_run(capsys, monkeypatch):
    monkeypatch.setattr(cli, "run_from_config", lambda *_: pytest.fail("validate ran"))
    code, out, _ = call(capsys, "validate", FEE_WARS)
    assert code == 0 and out.startswith("ok:") and "fee_wars" in out


def test_unknown_experiment_exits_2_with_suggestion(capsys, tmp_path):
    code, _, err = call(capsys, "run", write(tmp_path, {"experiment": "coaes"}),
                        "--out", tmp_path / "out")
    assert code == 2 and "did you mean 'coase'" in err and "Traceback" not in err
    assert not (tmp_path / "out").exists()


def test_show_config_round_trips_through_loader(capsys, tmp_path):
    for src in (COASE, FEE_WARS):
        _, out, _ = call(capsys, "show-config", src)
        again = resolve_config(write(tmp_path, json.loads(out)))
        assert again == resolve_config(src) and again.to_json() + "\n" == out
    assert resolve_config(FEE_WARS) == resolve_config({"experiment": "fee_wars", "small": True})


def test_list_names_all_five(capsys):
    code, out, _ = call(capsys, "list")
    names = {line.split()[0] for line in out.splitlines()}
    assert code == 0 and FIVE <= names
    assert all(len(line.split(None, 1)) == 2 for line in out.splitlines())   # has a description


def test_runtime_error_exits_1_traceback_only_with_debug(capsys, monkeypatch, tmp_path):
    def boom(cfg):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(cli, "run_from_config", boom)
    code, _, err = call(capsys, "run", COASE, "--out", tmp_path)
    assert code == 1 and "engine exploded" in err and "Traceback" not in err
    code, _, err = call(capsys, "run", COASE, "--out", tmp_path, "--debug")
    assert code == 1 and "Traceback" in err


@pytest.mark.parametrize("argv", [[], ["run"], ["frobnicate"], ["run", "x.json", "--seed", "a"]])
def test_usage_errors_exit_2(capsys, argv):
    assert call(capsys, *argv)[0] == 2


def test_bad_set_missing_file_and_extension_exit_2(capsys, tmp_path):
    assert call(capsys, "validate", COASE, "--set", "novalue")[0] == 2
    assert call(capsys, "validate", COASE, "--set", "master_seed.x=1")[0] == 2
    code, _, err = call(capsys, "validate", tmp_path / "missing.json")
    assert code == 2 and "cannot read" in err
    assert call(capsys, "validate", write(tmp_path, {}, "c.toml"))[0] == 2


def test_parse_set_values():
    assert cli.parse_set("params.levels=[1, 2]") == (["params", "levels"], [1, 2])
    assert cli.parse_set("output=runs/x") == (["output"], "runs/x")
    assert cli.parse_set("output=null") == (["output"], None)


def test_python_dash_m_help_exits_0(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["frictionlab", "--help"])
    with pytest.raises(SystemExit) as ei:
        runpy.run_module("frictionlab", run_name="__main__")
    assert ei.value.code == 0 and "run" in capsys.readouterr().out
    assert call(capsys, "--help")[0] == 0
