"""fl-072: examples/phase14_menu_cost.py, its write-up, docs index and ROADMAP status."""
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import pytest

from frictionlab import bench, cli
from frictionlab.docs_site import build_site, check_site
from frictionlab.experiments.menu_cost import run_menu_grid

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES, DOCS = ROOT / "examples", ROOT / "docs"
CONFIG = "examples/configs/menu_cost_small.json"
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
    return load("phase14_menu_cost")


@pytest.fixture(scope="module")
def two_runs(ex, tmp_path_factory):
    outs, times = [], []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp("phase14") / name
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


def test_adjustment_frequency_non_increasing_in_menu_cost(p):
    costs = sorted(p.params["menu_costs"])
    for d in p.params["inflations"]:
        s = [p.cell(f, d)["adjustment_frequency"] for f in costs]
        assert all(y <= x for x, y in zip(s, s[1:])), (d, s)
    assert [p.cell(f, 1)["adjustment_frequency"] for f in costs] == [1.0, 0.5, 0.25, 0.2]


def test_rows_match_the_experiment_called_directly(p):
    direct = run_menu_grid(p.cfg, p.params["menu_costs"], p.params["inflations"],
                           master_seed=p.master_seed)
    keys = ("band", "adjustments", "seller_steps", "misalignment_loss", "menu_cost_paid",
            "seller_net_pnl", "welfare_change_vs_flexible", "violations")
    assert [[r[k] for k in keys] for r in direct.rows] == [[r[k] for k in keys] for r in p.rows]
    assert all(r["violations"] == 0 for r in p.rows)


def test_band_frequency_and_threshold(p):
    assert (p.traced_cost, p.drift, p.traced_band) == (32, 2, 4)
    assert p.traced_frequency == Fraction(1, 4)
    assert p.thr.menu_cost == p.thr.analytic == 17 and p.thr.band == 4


def test_traced_sawtooth_path(p):
    assert len(p.path) == p.cfg.horizon
    assert [r.gap for r in p.path[:8]] == [0, -2, -4, 2, 0, -2, -4, 2]
    assert [r.t for r in p.path if r.adjusted][:6] == [4, 8, 12, 16, 20, 24]
    assert all(abs(r.gap) <= p.traced_band for r in p.path)
    assert sum(r.adjusted for r in p.path) * 4 == p.cfg.horizon


def test_surplus_split_reconciles(ex, p):
    s = p.split
    assert tuple(s) == ex.SPLIT_KEYS
    assert s == {"frictionless profit": 576000, "misalignment loss": -4200, "menu sink": -11856}
    assert sum(s.values()) == sum(r["seller_net_pnl"] for r in p.rows)
    assert -s["menu sink"] == sum(r["menu_cost_paid"] for r in p.rows)


def test_report_is_offline_xml_with_heatmap_path_and_split(two_runs):
    doc = (two_runs[0][0] / "report.html").read_text(encoding="utf-8")
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    kinds = [s.get("class") for s in root.iter("section")]
    assert {"sweep", "path", "pnl"} <= set(kinds)
    text = "".join(root.itertext())
    for needle in ("Adjustment frequency: menu cost x inflation", "Sawtooth price path",
                   "frictionless profit", "misalignment loss", "menu sink",
                   "bisected rigidity threshold"):
        assert needle in text, needle


def test_prints_tables(ex, tmp_path, capsys):
    assert ex.main(["--small", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for needle in ("menu-cost sweep (seed 71, 2 sellers",
                   "seller 1 at menu cost 32, inflation 2 (band 4):",
                   "t   4 target 1009 posted 1011 gap +2  <- adjust",
                   "analytic band 4, analytic frequency 1/4; rigidity threshold: "
                   "analytic 17, bisected 17 (band 4)",
                   "surplus split: frictionless profit 576000"):
        assert needle in out, needle


# ---------------------------------------------------------------- docs, CI, roadmap, checksums
def test_writeup_numbers_match_the_run(p):
    text = " ".join((DOCS / "phase14-menu-cost.md").read_text("utf-8").split())
    for needle in ("seed 71", "analytic band is 4", "analytic frequency of 1/4",
                   "1, 0.5, 0.25, 0.2", "analytic threshold is 17", "bisects to 17",
                   f"lose {p.split['misalignment loss']}", f"pay {p.split['menu sink']}",
                   "phase5-coase-threshold.md", "phase5-decision-cost.md"):
        assert needle in text, needle


def test_index_links_writeup_and_site_passes_check(tmp_path):
    assert "(phase14-menu-cost.md)" in (DOCS / "index.md").read_text("utf-8")
    build_site(DOCS, tmp_path / "_site")
    assert (tmp_path / "_site" / "phase14-menu-cost.html").is_file()
    assert check_site(tmp_path / "_site", src=DOCS).ok
    assert cli.main(["docs", "check", str(tmp_path / "_site"), "--src", str(DOCS)]) == 0


def test_ci_validate_step_covers_config():
    wf = (ROOT / "ci" / "ci.yml").read_text("utf-8")
    assert "for cfg in examples/configs/*.json" in wf and "frictionlab validate" in wf
    assert (ROOT / CONFIG).is_file() and cli.main(["validate", str(ROOT / CONFIG)]) == 0


def test_roadmap_ticks_phase14_and_reports_status():
    text = (ROOT / "ROADMAP.md").read_text("utf-8")
    phase14 = text.split("## Phase 14", 1)[1].split("\n## ", 1)[0]
    assert "- [ ]" not in phase14 and phase14.count("- [x]") == 3
    assert "docs/phase14-menu-cost.md" in phase14 and "examples/phase14_menu_cost.py" in phase14
    assert "Phase 14 status: complete (3 of 3)" in phase14
    assert "the analytic rigidity threshold is 17 and the bisected threshold is 17" in phase14


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
