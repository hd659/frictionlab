"""fl-063: examples/phase11_switching_cost.py, its write-up, docs index and ROADMAP status."""
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab import bench, cli
from frictionlab.docs_site import build_site, check_site
from frictionlab.experiments import switching_cost as sc

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES, DOCS = ROOT / "examples", ROOT / "docs"
CONFIG = "examples/configs/switching_cost_small.json"
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
    return load("phase11_switching_cost")


@pytest.fixture(scope="module")
def two_runs(ex, tmp_path_factory):
    outs, times = [], []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp("phase11") / name
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


def test_harvest_markup_non_decreasing_in_switching_cost(ex, p):
    costs = sorted(p.params["mean_switching_costs"])
    for t in p.params["target_shares"]:
        s = [ex.mean_by(p.rows, "harvest_markup", **{ex.MEAN: m, ex.TARGET: t}) for m in costs]
        assert all(y >= x for x, y in zip(s, s[1:])), (t, s)
    avg = [p.harvest_markup(m) for m in costs]
    assert all(y >= x for x, y in zip(avg, avg[1:])) and avg[-1] > avg[0]


def test_rows_match_the_experiment_called_directly(p):
    direct = sc.run_switching_grid(p.cfg, p.params["mean_switching_costs"], p.params["target_shares"],
                                   master_seed=p.master_seed)
    keys = ("trades", "switches", "switching_paid", "consumer_surplus", "seller_profit", "violations")
    assert [[r[k] for k in keys] for r in direct.rows] == [[r[k] for k in keys] for r in p.rows]
    assert all(r["violations"] == 0 for r in p.rows)


def test_markup_table_matches_klemperer(ex, p):
    got = {(r[ex.MEAN], r[ex.TARGET]): r["harvest_markup"] for r in p.rows}
    assert [got[(m, 6000)] for m in (0, 4, 16, 32)] == [0, 2, 14, 30]
    assert all(got[(m, 9000)] == 0 for m in (0, 4, 16, 32))      # 8750 bps max share
    assert all(r["harvest_markup"] == r["analytic_harvest_markup"] for r in p.rows)
    assert all(r["churn_rate"] == 0 and r["welfare_change_vs_bertrand"] == 0 for r in p.rows)


def test_traced_seller_switches_from_acquisition_to_harvest(p):
    assert (p.mean, p.target, p.cfg.horizon) == (16, 6000, 6)
    assert [s[1] for s in p.steps] == [40, 54, 54, 54, 54, 54]
    assert [s[2] for s in p.steps] == [40] * 6
    assert [s[6] for s in p.steps].index("acquire -> harvest") == p.harvest_round == 1
    assert all(s[1] <= s2[1] for s, s2 in zip(p.steps, p.steps[1:]))


def test_lockin_threshold(p):
    th = p.threshold
    assert (th.threshold, th.analytic, th.target, th.min_markup) == (7, 7, 6000, 4)
    assert th.probes[th.threshold] > th.min_markup >= th.probes[th.threshold - 1]


def test_surplus_split_reconciles(ex, p):
    s = p.split
    assert tuple(s) == ex.SPLIT_KEYS
    assert s == {"consumer gain": 37100, "acquisition loss": 0, "harvest gain": 3220, "switching sink": 0}
    assert -s["switching sink"] == sum(r["switching_paid"] for r in p.rows)
    assert s["harvest gain"] + s["acquisition loss"] == sum(r["seller_profit"] for r in p.rows)
    assert s["consumer gain"] + s["switching sink"] == sum(r["consumer_surplus"] for r in p.rows)


def test_report_is_offline_xml_with_heatmap_paths_and_split(two_runs):
    doc = (two_runs[0][0] / "report.html").read_text(encoding="utf-8")
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    kinds = [s.get("class") for s in root.iter("section")]
    assert {"sweep", "path", "pnl"} <= set(kinds)
    text = "".join(root.itertext())
    for needle in ("harvest_markup: mean_switching_cost x target_share", "Posted price paths",
                   "consumer gain", "acquisition loss", "harvest gain", "switching sink",
                   "bisected lock-in threshold"):
        assert needle in text, needle


def test_prints_tables(ex, tmp_path, capsys):
    assert ex.main(["--small", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for needle in ("switching-cost sweep (seed 62", "seller 1 (mean cost 16, target 6000 bps)",
                   "round 1: prices 54/40 harvest sold 14 switches 0 acquire -> harvest",
                   "bisected 7, analytic 7", "surplus split: consumer gain 37100"):
        assert needle in out, needle


# ---------------------------------------------------------------- docs, CI, roadmap, checksums
def test_writeup_numbers_match_the_run(p):
    text = " ".join((DOCS / "phase11-switching-cost.md").read_text("utf-8").split())
    for needle in ("seed 62", "40 -> 54 -> 54 -> 54 -> 54 -> 54", "markups of 2, 14 and 30 ticks",
                   "finds 7 ticks, against the analytic 7", "8750 bps",
                   f"consumer gain {p.split['consumer gain']}", "phase5-coase-threshold.md",
                   "phase7-search-cost.md", "phase10-reputation.md"):
        assert needle in text, needle


def test_index_links_writeup_and_site_passes_check(tmp_path):
    assert "(phase11-switching-cost.md)" in (DOCS / "index.md").read_text("utf-8")
    build_site(DOCS, tmp_path / "_site")
    assert (tmp_path / "_site" / "phase11-switching-cost.html").is_file()
    assert check_site(tmp_path / "_site", src=DOCS).ok
    assert cli.main(["docs", "check", str(tmp_path / "_site"), "--src", str(DOCS)]) == 0


def test_ci_validate_step_covers_config():
    wf = (ROOT / "ci" / "ci.yml").read_text("utf-8")
    assert "for cfg in examples/configs/*.json" in wf and "frictionlab validate" in wf
    assert (ROOT / CONFIG).is_file() and cli.main(["validate", str(ROOT / CONFIG)]) == 0


def test_roadmap_ticks_phase11_and_reports_status():
    text = (ROOT / "ROADMAP.md").read_text("utf-8")
    phase11 = text.split("## Phase 11", 1)[1].split("\n## ", 1)[0]
    assert "- [ ]" not in phase11
    assert "docs/phase11-switching-cost.md" in phase11 and "examples/phase11_switching_cost.py" in phase11
    assert "Phase 11 status: complete (3 of 3)" in phase11
    assert "lock-in threshold (markup > 4) is 7 ticks against an analytic 7" in phase11


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
