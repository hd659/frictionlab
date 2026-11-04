"""fl-045: examples/phase6_ecosystem.py (config -> run -> bench gate -> docs site, one dir)."""
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
BUDGET_S = 60.0
RUN_FILES = ("rows.csv", "resolved_config.json", "report.html")


def load():
    if "phase6_ecosystem" in sys.modules:
        return sys.modules["phase6_ecosystem"]
    spec = importlib.util.spec_from_file_location("phase6_ecosystem",
                                                  EXAMPLES / "phase6_ecosystem.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase6_ecosystem"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ex():
    return load()


@pytest.fixture(scope="module")
def two_runs(ex, tmp_path_factory):
    outs, times = [], []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp("phase6") / name
        t0 = time.perf_counter()
        assert ex.main(["--small", str(out)]) == 0
        times.append(time.perf_counter() - t0)
        outs.append(out)
    return outs, times


def _files(root: Path, sub: str):
    base = root / sub
    return sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())


def test_runs_within_budget(two_runs):
    assert max(two_runs[1]) < BUDGET_S


def test_writes_run_artifacts_and_site_index(two_runs):
    out = two_runs[0][0]
    for name in RUN_FILES:
        assert (out / "run" / name).is_file(), name
    assert (out / "site" / "index.html").is_file()
    assert (out / "site" / "phase6-ecosystem.html").is_file()
    assert (out / "site" / "api" / "index.html").is_file()
    assert (out / "bench_small.json").is_file() and (out / "MANIFEST.txt").is_file()


def test_run_artifacts_match_config(two_runs):
    run = two_runs[0][0] / "run"
    cfg = json.loads((run / "resolved_config.json").read_text("utf-8"))
    assert cfg["experiment"] == "coase" and cfg["master_seed"] == 28 and cfg["small"] is True
    with open(run / "rows.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    levels = cfg["params"]["levels"]
    assert len(rows) == len(levels) * cfg["replicates"]
    assert sorted({int(r["friction"]) for r in rows}) == sorted(levels)


def test_bench_file_matches_baseline_checksums(two_runs):
    got = json.loads((two_runs[0][0] / "bench_small.json").read_text("utf-8"))
    base = json.loads((ROOT / "benchmarks" / "baseline_small.json").read_text("utf-8"))
    assert {k: v["checksum"] for k, v in got["scenarios"].items()} == \
        {k: v["checksum"] for k, v in base["scenarios"].items()}


@pytest.mark.parametrize("sub", ["run", "site"])
def test_byte_reproducible_across_runs(two_runs, sub):
    (a, b), _ = two_runs
    assert _files(a, sub) == _files(b, sub) and _files(a, sub)
    for rel in _files(a, sub):
        assert (a / sub / rel).read_bytes() == (b / sub / rel).read_bytes(), rel


def test_manifest_lists_every_artifact_and_is_identical(two_runs):
    (a, b), _ = two_runs
    lines = (a / "MANIFEST.txt").read_text("utf-8").splitlines()
    assert lines == (b / "MANIFEST.txt").read_text("utf-8").splitlines()
    listed = [ln.split()[-1] for ln in lines]
    assert listed == sorted(listed)
    assert {f"run/{n}" for n in RUN_FILES} | {"site/index.html", "bench_small.json"} <= set(listed)
    assert str(a) not in "\n".join(lines)                  # no absolute paths


def test_prints_short_manifest(ex, tmp_path, capsys):
    assert ex.main(["--small", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for label in ("validate config", "run experiment", "bench checksum gate",
                  "build docs site", "check docs links"):
        assert f"[ok] {label}" in out
    assert "compare: ok" in out and "0 broken" in out
    assert "run/rows.csv" in out and "site/ (" in out


def test_default_output_is_a_temp_dir(ex, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ex.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "tmpout"))
    assert ex.main(["--small"]) == 0
    assert (tmp_path / "tmpout" / "run" / "rows.csv").is_file()
    assert f"artifacts in {tmp_path / 'tmpout'}" in capsys.readouterr().out


def test_failing_step_returns_its_exit_code(ex, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ex, "DOCS", tmp_path / "no_docs")
    assert ex.main(["--small", str(tmp_path / "o")]) == 2
    assert "docs build" in capsys.readouterr().err
