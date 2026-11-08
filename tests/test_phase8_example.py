"""fl-054: examples/phase8_grossman_stiglitz.py, its write-up, docs index and ROADMAP status."""
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab import bench, cli
from frictionlab.docs_site import build_site, check_site
from frictionlab.experiments import grossman_stiglitz as gs

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES, DOCS = ROOT / "examples", ROOT / "docs"
CONFIG = "examples/configs/grossman_stiglitz_small.json"
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
    return load("phase8_grossman_stiglitz")


@pytest.fixture(scope="module")
def two_runs(ex, tmp_path_factory):
    outs, times = [], []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp("phase8gs") / name
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


def test_buyer_fraction_non_increasing_in_signal_cost(ex, p):
    costs, noises = p.params["signal_costs"], p.params["noise_intensities"]
    for lam in noises:
        fr = [ex.mean_by(p.rows, "buyer_fraction", **{ex.COST: c, ex.NOISE: lam}) for c in costs]
        assert all(b <= a for a, b in zip(fr, fr[1:])), (lam, fr)
        assert fr[0] > 0.5 and fr[-1] == 0.0
    avg = [p.buyer_fraction(c) for c in costs]
    assert all(b <= a for a, b in zip(avg, avg[1:]))


def test_rows_match_the_experiment_called_directly(ex, p):
    direct = gs.run_gs_grid(p.cfg, p.params["signal_costs"], p.params["noise_intensities"],
                            master_seed=p.master_seed)
    keys = ("buyer_fraction", "quoted_spread", "efficiency_error", "sink_revenue", "buyer_net_pnl")
    assert [[r[k] for k in keys] for r in direct.rows] == [[r[k] for k in keys] for r in p.rows]


def test_fixed_points_and_collapse(p):
    got = {d["cost"]: d["analytic"] for d in p.population}
    assert got == {0: 0.8, 4: 0.4, 9: 0.0}
    for lo, hi in zip(p.population, p.population[1:]):
        assert hi["buyer_share"] <= lo["buyer_share"]
    assert p.eq.reason == "extinction" and p.eq_cost == 4
    c = p.collapse
    assert (c.threshold, c.analytic) == (7, 8) and c.threshold <= c.analytic
    assert c.probes[c.threshold] < c.floor <= c.probes[c.threshold - 1]


def test_buyer_pnl_split_reconciles(ex, p):
    s = p.split
    assert tuple(s) == ex.SPLIT_KEYS
    assert s["trading gain"] + s["signal cost"] + s["decision cost"] == s["net"]
    assert -s["signal cost"] == sum(r["sink_revenue"] for r in p.rows)
    assert s["decision cost"] == 0 and s["signal cost"] < 0


def test_report_is_offline_xml_with_heatmap_equilibrium_and_split(two_runs):
    doc = (two_runs[0][0] / "report.html").read_text(encoding="utf-8")
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    kinds = [s.get("class") for s in root.iter("section")]
    assert {"sweep", "equilibrium", "pnl"} <= set(kinds)
    assert root.find(".//*[@class='fixed-point']") is not None
    text = "".join(root.itertext())
    for needle in ("buyer_fraction: signal_cost x noise_intensity", "trading gain", "signal cost",
                   "decision cost", "bisected collapse cost"):
        assert needle in text, needle


def test_prints_tables(ex, tmp_path, capsys):
    assert ex.main(["--small", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for needle in ("grossman-stiglitz sweep (seed 53", "fixed point (cost 4)",
                   "information collapse: bisected 7, analytic 8", "buyer PnL split"):
        assert needle in out, needle


# ---------------------------------------------------------------- docs, CI, roadmap, checksums
def test_writeup_numbers_match_the_run(p):
    text = " ".join((DOCS / "phase8-grossman-stiglitz.md").read_text("utf-8").split())
    for needle in ("seed 53", "collapse is at 7 ticks against the analytic 8",
                   f"trading gain is {p.split['trading gain']}", f"net is {p.split['net']}",
                   "0.80, 0.60, 0.40 and 0.00"):
        assert needle in text, needle


def test_index_links_writeup_and_site_passes_check(tmp_path):
    assert "(phase8-grossman-stiglitz.md)" in (DOCS / "index.md").read_text("utf-8")
    build_site(DOCS, tmp_path / "_site")
    assert (tmp_path / "_site" / "phase8-grossman-stiglitz.html").is_file()
    assert check_site(tmp_path / "_site", src=DOCS).ok
    assert cli.main(["docs", "check", str(tmp_path / "_site"), "--src", str(DOCS)]) == 0


def test_ci_validate_step_covers_config():
    wf = (ROOT / "ci" / "ci.yml").read_text("utf-8")
    assert "for cfg in examples/configs/*.json" in wf and "frictionlab validate" in wf
    assert (ROOT / CONFIG).is_file() and cli.main(["validate", str(ROOT / CONFIG)]) == 0


def test_roadmap_ticks_phase8_and_reports_status():
    text = (ROOT / "ROADMAP.md").read_text("utf-8")
    phase8 = text.split("## Phase 8", 1)[1].split("\n## ", 1)[0]
    assert "- [ ]" not in phase8
    assert "docs/phase8-grossman-stiglitz.md" in phase8
    assert "examples/phase8_grossman_stiglitz.py" in phase8
    assert "Phase 8 status: complete (3 of 3)" in phase8
    assert "7 ticks against an analytic 8" in phase8 and "0.99 -> 0.74 -> 0.48 -> 0.00" in phase8


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
