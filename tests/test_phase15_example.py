"""fl-075: examples/phase15_holdup.py, its write-up, docs index and ROADMAP status."""
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab import bench, cli
from frictionlab.docs_site import build_site, check_site
from frictionlab.experiments.holdup import run_holdup_grid

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES, DOCS = ROOT / "examples", ROOT / "docs"
CONFIG = "examples/configs/holdup_small.json"
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
    return load("phase15_holdup")


@pytest.fixture(scope="module")
def two_runs(ex, tmp_path_factory):
    outs, times = [], []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp("phase15") / name
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


def test_contracting_rate_non_increasing_in_contracting_cost(p):
    costs = sorted(p.params["contracting_costs"])
    for w in p.params["seller_weights"]:
        s = [p.cell(c, w)["contracting_rate"] for c in costs]
        assert all(y <= x for x, y in zip(s, s[1:])), (w, s)
    assert [p.cell(c, 5000)["contracting_rate"] for c in costs] == [1.0, 1.0, 1.0, 0.0]
    assert all(p.cell(c, 10000)["contracting_rate"] == 0.0 for c in costs)


def test_rows_match_the_experiment_called_directly(p):
    direct = run_holdup_grid(p.cfg, p.params["contracting_costs"], p.params["seller_weights"],
                             master_seed=p.master_seed)
    keys = ("contracts", "contracting_cost_paid", "seller_surplus", "buyer_surplus", "welfare",
            "max_surplus", "welfare_change_vs_free_contracting", "max_contracting_cost", "violations")
    assert [[r[k] for k in keys] for r in direct.rows] == [[r[k] for k in keys] for r in p.rows]
    assert all(r["violations"] == 0 for r in p.rows)


def test_levels_curve_and_threshold(p):
    assert (p.weight, p.cost, p.holdup_level, p.first_best) == (5000, 10, 1, 3)
    assert [h for _, h, _ in p.curve] == [5, 8, 8, 5, -2, -13]
    assert [k for _, _, k in p.curve] == [-5, 1, 5, 7, 6, 2]
    assert max(h for _, h, _ in p.curve) == p.curve[p.holdup_level][1]
    assert p.thr.analytic == 8 and p.thr.contracting_cost == 9
    assert p.thr.contracting_cost <= p.thr.analytic + 1
    assert p.cell(10, 5000)["underinvestment_gap"] == 12.0


def test_surplus_split_reconciles(ex, p):
    s = p.split
    assert tuple(s) == ex.SPLIT_KEYS
    assert s == {"buyer gain": 2616, "seller net gain": 6648, "sunk investment": 6480,
                 "contract sink": 240, "underinvestment deadweight loss": 288}
    total = sum(r["max_surplus"] for r in p.rows)
    assert (s["buyer gain"] + s["seller net gain"] + s["contract sink"]
            + s["underinvestment deadweight loss"]) == total == 9792
    assert s["contract sink"] == sum(r["contracting_cost_paid"] for r in p.rows)


def test_report_is_offline_xml_with_heatmap_curve_and_split(two_runs):
    doc = (two_runs[0][0] / "report.html").read_text(encoding="utf-8")
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    kinds = [s.get("class") for s in root.iter("section")]
    assert {"sweep", "payoff", "pnl"} <= set(kinds)
    text = "".join(root.itertext())
    for needle in ("Surplus share: contracting cost x seller weight",
                   "Seller payoff by investment level", "buyer gain", "seller net gain",
                   "sunk investment", "contract sink", "underinvestment deadweight loss",
                   "bisected contracting threshold"):
        assert needle in text, needle


def test_prints_tables(ex, tmp_path, capsys):
    assert ex.main(["--small", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for needle in ("hold-up sweep (seed 74, 2 pairs, outside value 30, 12 rounds):",
                   "seller payoff at weight 5000, contracting cost 10:",
                   "level 1 hold-up    8 contract    1  <- hold-up",
                   "level 3 hold-up    5 contract    7  <- first best",
                   "max contracting cost: analytic 8, bisected threshold 9 (weight 5000)",
                   "surplus split: buyer gain 2616"):
        assert needle in out, needle


# ---------------------------------------------------------------- docs, CI, roadmap, checksums
def test_writeup_numbers_match_the_run(p):
    text = " ".join((DOCS / "phase15-holdup.md").read_text("utf-8").split())
    for needle in ("seed 74", "first-best level is 3", "level 1", "analytic max contracting cost is 8",
                   "bisected threshold", "is 9", "1, 1, 1, 0", "0.647",
                   f"buyer gain {p.split['buyer gain']}", f"contract sink {p.split['contract sink']}",
                   "phase5-coase-threshold.md", "phase11-switching-cost.md", "phase14-menu-cost.md"):
        assert needle in text, needle


def test_index_links_writeup_and_site_passes_check(tmp_path):
    assert "(phase15-holdup.md)" in (DOCS / "index.md").read_text("utf-8")
    build_site(DOCS, tmp_path / "_site")
    assert (tmp_path / "_site" / "phase15-holdup.html").is_file()
    assert check_site(tmp_path / "_site", src=DOCS).ok
    assert cli.main(["docs", "check", str(tmp_path / "_site"), "--src", str(DOCS)]) == 0


def test_ci_validate_step_covers_config():
    wf = (ROOT / "ci" / "ci.yml").read_text("utf-8")
    assert "for cfg in examples/configs/*.json" in wf and "frictionlab validate" in wf
    assert (ROOT / CONFIG).is_file() and cli.main(["validate", str(ROOT / CONFIG)]) == 0


def test_roadmap_ticks_phase15_and_reports_status():
    text = (ROOT / "ROADMAP.md").read_text("utf-8")
    phase15 = text.split("## Phase 15", 1)[1].split("\n## ", 1)[0]
    assert "- [ ]" not in phase15 and phase15.count("- [x]") == 3
    assert "docs/phase15-holdup.md" in phase15 and "examples/phase15_holdup.py" in phase15
    assert "Phase 15 status: complete (3 of 3)" in phase15
    assert "the analytic max contracting cost is 8 and the bisected contracting threshold is 9" in phase15


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
