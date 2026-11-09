"""fl-069: examples/phase13_congestion.py, its write-up, docs index and ROADMAP status."""
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab import bench, cli
from frictionlab.docs_site import build_site, check_site
from frictionlab.experiments import run_queue_grid

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES, DOCS = ROOT / "examples", ROOT / "docs"
CONFIG = "examples/configs/congestion_small.json"
BASELINE = ROOT / "benchmarks" / "baseline_small.json"
BUDGET_S = 60.0


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ex():
    return load("phase13_congestion")


@pytest.fixture(scope="module")
def two_runs(ex, tmp_path_factory):
    outs, times = [], []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp("phase13") / name
        t0 = time.perf_counter()
        assert ex.main(["--small", str(out)]) == 0
        times.append(time.perf_counter() - t0)
        outs.append(out)
    return outs, times


@pytest.fixture(scope="module")
def p(ex, two_runs):
    return ex.run(two_runs[0][0] / "again", small=True)


def parse(doc: str) -> ET.Element:
    assert doc.startswith("<!DOCTYPE html>\n")
    return ET.fromstring(doc.split("\n", 1)[1])


def test_runs_within_budget(two_runs):
    assert max(two_runs[1]) < BUDGET_S


@pytest.mark.parametrize("rel", ["run/rows.csv", "report.html", "run/report.html"])
def test_byte_reproducible_across_runs(two_runs, rel):
    (a, b), _ = two_runs
    assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_join_rate_non_increasing_in_toll(p):
    tolls = sorted(p.params["tolls"])
    for c in p.params["waiting_costs"]:
        s = [p.cell(t, c)["join_rate"] for t in tolls]
        assert all(y <= x for x, y in zip(s, s[1:])), (c, s)


def test_rows_match_the_experiment_called_directly(p):
    direct = run_queue_grid(p.cfg, p.params["tolls"], p.params["waiting_costs"],
                            master_seed=p.master_seed)
    keys = ("joins", "served", "max_queue_length", "toll_revenue", "waiting_paid",
            "customer_surplus", "welfare", "welfare_change_vs_zero_toll", "violations")
    assert [[r[k] for k in keys] for r in direct.rows] == [[r[k] for k in keys] for r in p.rows]
    assert all(r["violations"] == 0 for r in p.rows)


def test_thresholds_and_tolls(p):
    assert (p.individual, p.tol.social_threshold) == (5, 0)
    assert p.revenue == {"threshold": 0, "toll": 20}
    assert p.tol.toll == p.tol.analytic == 17 and p.tol.waiting_cost == 2
    assert p.tol.probes[p.tol.toll] <= p.tol.social_threshold < p.tol.probes[p.tol.toll - 1]


def test_queue_paths_and_welfare(p):
    assert p.zero_path[:13] == [0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5]
    assert max(p.zero_path) == 5 and max(p.opt_path) == 0
    assert len(p.zero_path) == len(p.opt_path) == p.cfg.horizon
    assert (p.welfare_zero, p.welfare_opt) == (158, 1590)
    assert p.welfare_zero == p.cell(0, 2)["welfare"]


def test_welfare_split_reconciles(ex, p):
    s = p.split
    assert tuple(s) == ex.SPLIT_KEYS
    assert s == {"customer reward": 26232, "waiting-cost sink": -17123,
                 "toll paid by customers": -7796, "toll revenue": 7796}
    assert sum(s.values()) == sum(r["welfare"] for r in p.rows)
    assert -s["waiting-cost sink"] == sum(r["waiting_paid"] for r in p.rows)


def test_report_is_offline_xml_with_heatmap_paths_and_split(two_runs):
    doc = (two_runs[0][0] / "report.html").read_text(encoding="utf-8")
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    kinds = [s.get("class") for s in root.iter("section")]
    assert {"sweep", "path", "pnl"} <= set(kinds)
    text = "".join(root.itertext())
    for needle in ("welfare: toll x waiting_cost", "Queue-length paths", "customer reward",
                   "waiting-cost sink", "toll revenue", "bisected optimal toll"):
        assert needle in text, needle


def test_prints_tables(ex, tmp_path, capsys):
    assert ex.main(["--small", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for needle in ("congestion sweep (seed 68, R 24",
                   "thresholds at waiting cost 2: individual 5, social 0, revenue-maximizing 0 (toll 20)",
                   "optimal toll: analytic 17, bisected 17",
                   "queue path, toll 0: 0 0 1 2 2 3 4 4 5", "toll 0 -> 158, toll 17 -> 1590",
                   "welfare split: customer reward 26232"):
        assert needle in out, needle


# ---------------------------------------------------------------- docs, CI, roadmap, checksums
def test_writeup_numbers_match_the_run(p):
    text = " ".join((DOCS / "phase13-congestion.md").read_text("utf-8").split())
    for needle in ("seed 68", "individual threshold of 5", "is 0", "that toll is 17",
                   "bisects to 17", "from 158 to 1590", "0 0 1 2 2 3 4 4 5 5 5 5 5",
                   f"takes {p.split['waiting-cost sink']} ticks", "picks toll 20",
                   "phase5-coase-threshold.md", "phase12-critical-mass.md"):
        assert needle in text, needle


def test_index_links_writeup_and_site_passes_check(tmp_path):
    assert "(phase13-congestion.md)" in (DOCS / "index.md").read_text("utf-8")
    build_site(DOCS, tmp_path / "_site")
    assert (tmp_path / "_site" / "phase13-congestion.html").is_file()
    assert check_site(tmp_path / "_site", src=DOCS).ok
    assert cli.main(["docs", "check", str(tmp_path / "_site"), "--src", str(DOCS)]) == 0


def test_ci_validate_step_covers_config():
    wf = (ROOT / "ci" / "ci.yml").read_text("utf-8")
    assert "for cfg in examples/configs/*.json" in wf and "frictionlab validate" in wf
    assert (ROOT / CONFIG).is_file() and cli.main(["validate", str(ROOT / CONFIG)]) == 0


def test_roadmap_ticks_phase13_and_reports_status():
    text = (ROOT / "ROADMAP.md").read_text("utf-8")
    phase13 = text.split("## Phase 13", 1)[1].split("\n## ", 1)[0]
    assert "- [ ]" not in phase13 and phase13.count("- [x]") == 3
    assert "docs/phase13-congestion.md" in phase13 and "examples/phase13_congestion.py" in phase13
    assert "Phase 13 status: complete (3 of 3)" in phase13
    assert "the analytic optimal toll is 17 and the bisected toll is 17" in phase13


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
