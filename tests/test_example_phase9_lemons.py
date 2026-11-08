"""fl-057: examples/phase9_lemons.py, its write-up, docs index and ROADMAP status."""
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab import bench, cli
from frictionlab.docs_site import build_site, check_site
from frictionlab.experiments import lemons as lm

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES, DOCS = ROOT / "examples", ROOT / "docs"
CONFIG = "examples/configs/lemons_small.json"
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
    return load("phase9_lemons")


@pytest.fixture(scope="module")
def two_runs(ex, tmp_path_factory):
    outs, times = [], []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp("phase9") / name
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


def test_surplus_share_non_decreasing_in_buyer_premium(ex, p):
    prems = sorted(p.params["buyer_premiums"])
    for c in p.params["certification_costs"]:
        s = [ex.mean_by(p.rows, "surplus_share", **{ex.COST: c, ex.PREMIUM: b}) for b in prems]
        assert all(y >= x for x, y in zip(s, s[1:])), (c, s)
    avg = [p.surplus_share(b) for b in prems]
    assert all(y >= x for x, y in zip(avg, avg[1:])) and avg[-1] > avg[0]


def test_rows_match_the_experiment_called_directly(p):
    direct = lm.run_lemons_grid(p.cfg, p.params["certification_costs"], p.params["buyer_premiums"],
                                master_seed=p.master_seed)
    keys = ("traded_share", "pooled_price", "certified", "cutoff", "surplus_share",
            "certification_paid")
    assert [[r[k] for k in keys] for r in direct.rows] == [[r[k] for k in keys] for r in p.rows]


def test_unraveling_path_and_thresholds(p):
    assert [pr for _, pr, _ in p.path] == [57, 30, 20, 18, 10, 10]
    assert [n for *_, n in p.path] == [20, 9, 6, 5, 1, 1]
    assert p.akerlof == (10, 10) and p.path[-1][1] == p.akerlof[0]
    th = p.threshold
    assert (th.threshold, th.analytic) == (21, 21) and th.threshold <= th.analytic
    assert th.probes[th.threshold - 1] >= th.fraction    # uptake 0.7 one tick below
    at = lm.run_lemons_grid(p.cfg, [th.threshold], [p.params["threshold_premium"]],
                            master_seed=p.master_seed).rows[0]["certification_uptake"]
    assert at < th.fraction                                # the bound itself is not probed


def test_surplus_split_reconciles(ex, p):
    s = p.split
    assert tuple(s) == ex.SPLIT_KEYS
    assert -s["certification sink"] == sum(r["certification_paid"] for r in p.rows) == 568
    assert (s["buyer gain"], s["pooled-seller gain"], s["certified-seller gain"]) == (7, 1123, 2590)


def test_report_is_offline_xml_with_heatmap_path_and_split(two_runs):
    doc = (two_runs[0][0] / "report.html").read_text(encoding="utf-8")
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    kinds = [s.get("class") for s in root.iter("section")]
    assert {"sweep", "path", "pnl"} <= set(kinds)
    text = "".join(root.itertext())
    for needle in ("surplus_share: certification_cost x buyer_premium", "Pooled price by round",
                   "buyer gain", "pooled-seller gain", "certified-seller gain",
                   "certification sink", "bisected certification threshold"):
        assert needle in text, needle


def test_prints_tables(ex, tmp_path, capsys):
    assert ex.main(["--small", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for needle in ("lemons sweep (seed 56", "unraveling (premium 0, no certification): 57",
                   "akerlof fixed point: price 10", "bisected 21, analytic 21", "surplus split"):
        assert needle in out, needle


# ---------------------------------------------------------------- docs, CI, roadmap, checksums
def test_writeup_numbers_match_the_run(p):
    text = " ".join((DOCS / "phase9-lemons.md").read_text("utf-8").split())
    for needle in ("seed 56", "57 -> 30 -> 20 -> 18 -> 10 -> 10", "10, 30 and 95",
                   "21 ticks against the analytic bound of 21",
                   f"buyer gain {p.split['buyer gain']}", "certification sink -568",
                   "phase5-coase-threshold.md", "phase8-grossman-stiglitz.md"):
        assert needle in text, needle


def test_index_links_writeup_and_site_passes_check(tmp_path):
    assert "(phase9-lemons.md)" in (DOCS / "index.md").read_text("utf-8")
    build_site(DOCS, tmp_path / "_site")
    assert (tmp_path / "_site" / "phase9-lemons.html").is_file()
    assert check_site(tmp_path / "_site", src=DOCS).ok
    assert cli.main(["docs", "check", str(tmp_path / "_site"), "--src", str(DOCS)]) == 0


def test_ci_validate_step_covers_config():
    wf = (ROOT / "ci" / "ci.yml").read_text("utf-8")
    assert "for cfg in examples/configs/*.json" in wf and "frictionlab validate" in wf
    assert (ROOT / CONFIG).is_file() and cli.main(["validate", str(ROOT / CONFIG)]) == 0


def test_roadmap_ticks_phase9_and_reports_status():
    text = (ROOT / "ROADMAP.md").read_text("utf-8")
    phase9 = text.split("## Phase 9", 1)[1].split("\n## ", 1)[0]
    assert "- [ ]" not in phase9
    assert "docs/phase9-lemons.md" in phase9 and "examples/phase9_lemons.py" in phase9
    assert "Phase 9 status: complete (3 of 3)" in phase9
    assert "21 ticks against an analytic bound of 21" in phase9
    assert "57 -> 30 -> 20 -> 18 -> 10" in phase9


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
