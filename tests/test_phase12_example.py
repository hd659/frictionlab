"""fl-066: examples/phase12_critical_mass.py, its write-up, docs index and ROADMAP status."""
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab import bench, cli
from frictionlab.docs_site import build_site, check_site
from frictionlab.experiments import run_platform_grid

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES, DOCS = ROOT / "examples", ROOT / "docs"
CONFIG = "examples/configs/critical_mass_small.json"
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
    return load("phase12_critical_mass")


@pytest.fixture(scope="module")
def two_runs(ex, tmp_path_factory):
    outs, times = [], []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp("phase12") / name
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


def test_final_membership_non_decreasing_in_seed_members(ex, p):
    seeds = sorted(p.params["seed_members"])
    for c in p.params["adoption_costs"]:
        s = [p.members(c, k) for k in seeds]
        assert all(y >= x for x, y in zip(s, s[1:])), (c, s)
        assert s[0] == 0 and s[-1] == p.cfg.n_a + p.cfg.n_b


def test_rows_match_the_experiment_called_directly(ex, p):
    direct = run_platform_grid(p.cfg, p.params["adoption_costs"], p.params["seed_members"],
                               master_seed=p.master_seed)
    keys = ("members_a", "members_b", "adoption_paid", "seed_subsidy", "platform_pnl",
            "member_surplus", "welfare_change_vs_free_adoption", "critical_mass", "violations")
    assert [[r[k] for k in keys] for r in direct.rows] == [[r[k] for k in keys] for r in p.rows]
    assert all(r["violations"] == 0 and r["outcome"] == "fixed_point" for r in p.rows)


def test_membership_table_and_critical_mass(p):
    assert [p.members(c, 6) for c in (0, 2, 8, 40)] == [16, 12, 12, 12]
    assert all(p.members(c, 4) == 0 and p.members(c, 8) == 16 for c in (0, 2, 8, 40))
    assert p.analytic == {0: 6, 2: 8, 8: 8, 40: 8}
    a = list(p.analytic.values())
    assert all(y >= x for x, y in zip(a, a[1:]))


def test_paths_just_below_and_at_the_tipping_seed(p):
    assert p.trace_tip.seed_members == 6 == p.trace_tip.analytic
    assert [(s[1], s[2]) for s in p.below] == [(5, 5), (0, 0), (0, 0)]
    assert [(s[1], s[2]) for s in p.above] == [(6, 6), (8, 8), (8, 8)]


def test_bisected_tipping_seed(p):
    tip = p.tip
    assert (tip.seed_members, tip.analytic, tip.adoption_cost, tip.fraction) == (8, 8, 2, 0.9)
    target = tip.fraction * (p.cfg.n_a + p.cfg.n_b)
    assert tip.probes[tip.seed_members] > target >= tip.probes[tip.seed_members - 1]


def test_surplus_split_reconciles(ex, p):
    s = p.split
    assert tuple(s) == ex.SPLIT_KEYS
    assert s == {"member standalone value": -792, "cross-side gain": 952, "platform fee revenue": 0,
                 "subsidy outlay": -1800, "adoption sink": -1800}
    assert s["platform fee revenue"] + s["subsidy outlay"] == sum(r["platform_pnl"] for r in p.rows)
    assert -s["adoption sink"] == sum(r["adoption_paid"] for r in p.rows)


def test_report_is_offline_xml_with_heatmap_paths_and_split(two_runs):
    doc = (two_runs[0][0] / "report.html").read_text(encoding="utf-8")
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    kinds = [s.get("class") for s in root.iter("section")]
    assert {"sweep", "path", "pnl"} <= set(kinds)
    text = "".join(root.itertext())
    for needle in ("share: adoption_cost x seed_members", "Membership paths", "member standalone value",
                   "cross-side gain", "platform fee revenue", "subsidy outlay", "adoption sink",
                   "bisected tipping seed"):
        assert needle in text, needle


def test_prints_tables(ex, tmp_path, capsys):
    assert ex.main(["--small", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for needle in ("critical-mass sweep (seed 65, 8 + 8 members",
                   "seed 5 (below tipping seed, adoption cost 0): 5/5 -> 0/0 -> 0/0",
                   "seed 6 (at tipping seed, adoption cost 0): 6/6 -> 8/8 -> 8/8",
                   "cost 0 -> 6, cost 2 -> 8", "bisected 8, analytic 8",
                   "surplus split: member standalone value -792"):
        assert needle in out, needle


# ---------------------------------------------------------------- docs, CI, roadmap, checksums
def test_writeup_numbers_match_the_run(p):
    text = " ".join((DOCS / "phase12-critical-mass.md").read_text("utf-8").split())
    for needle in ("seed 65", "6/6 -> 8/8 -> 8/8", "5/5 -> 0/0 -> 0/0", "6, 8, 8 and 8",
                   "bisects to seed 8, against an analytic critical mass of 8",
                   f"subsidy outlay is {p.split['subsidy outlay']}", "phase5-coase-threshold.md",
                   "phase11-switching-cost.md"):
        assert needle in text, needle


def test_index_links_writeup_and_site_passes_check(tmp_path):
    assert "(phase12-critical-mass.md)" in (DOCS / "index.md").read_text("utf-8")
    build_site(DOCS, tmp_path / "_site")
    assert (tmp_path / "_site" / "phase12-critical-mass.html").is_file()
    assert check_site(tmp_path / "_site", src=DOCS).ok
    assert cli.main(["docs", "check", str(tmp_path / "_site"), "--src", str(DOCS)]) == 0


def test_ci_validate_step_covers_config():
    wf = (ROOT / "ci" / "ci.yml").read_text("utf-8")
    assert "for cfg in examples/configs/*.json" in wf and "frictionlab validate" in wf
    assert (ROOT / CONFIG).is_file() and cli.main(["validate", str(ROOT / CONFIG)]) == 0


def test_roadmap_ticks_phase12_and_reports_status():
    text = (ROOT / "ROADMAP.md").read_text("utf-8")
    phase12 = text.split("## Phase 12", 1)[1].split("\n## ", 1)[0]
    assert "- [ ]" not in phase12
    assert "docs/phase12-critical-mass.md" in phase12 and "examples/phase12_critical_mass.py" in phase12
    assert "Phase 12 status: complete (3 of 3)" in phase12
    assert "bisected tipping seed at adoption cost 2 (fraction 0.9) is 8 against an analytic" in phase12


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
