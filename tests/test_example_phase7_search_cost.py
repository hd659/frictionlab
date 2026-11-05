"""fl-048: examples/phase7_search_cost.py, the Phase 7 write-up and the summary-report flag."""
import hashlib
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab import bench, cli
from frictionlab.docs_site import build_site, check_site

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOCS = ROOT / "docs"
CONFIG = "examples/configs/search_cost_small.json"
BASELINE = ROOT / "benchmarks" / "baseline_small.json"
BUDGET_S = 60.0
# sha256 of the default five-section summary report before --include-phase7 existed
SUMMARY_SHA256 = "473013de2e67abb669812cab1c4e1a33c5a5c95259032327d537feafcab2a7a9"


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
    return load("phase7_search_cost")


@pytest.fixture(scope="module")
def two_runs(ex, tmp_path_factory):
    outs, times = [], []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp("phase7") / name
        t0 = time.perf_counter()
        assert ex.main(["--small", str(out)]) == 0
        times.append(time.perf_counter() - t0)
        outs.append(out)
    return outs, times


def parse(doc: str) -> ET.Element:
    assert doc.startswith("<!DOCTYPE html>\n")
    return ET.fromstring(doc.split("\n", 1)[1])


def test_runs_within_budget(two_runs):
    assert max(two_runs[1]) < BUDGET_S


@pytest.mark.parametrize("rel", ["run/rows.csv", "report.html", "run/report.html"])
def test_byte_reproducible_across_runs(two_runs, rel):
    (a, b), _ = two_runs
    assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_surplus_share_non_increasing_in_search_cost(ex, two_runs):
    rows = ex.read_rows(two_runs[0][0] / "run" / "rows.csv")
    for d in sorted({r[ex.DECAY] for r in rows}):
        shares = [ex.mean_by(rows, "surplus_share", **{ex.COST: c, ex.DECAY: d})
                  for c in sorted({r[ex.COST] for r in rows})]
        assert all(x >= y for x, y in zip(shares, shares[1:])), (d, shares)
        assert shares[0] == 1.0 and shares[-1] == 0.0


def test_thresholds_and_decay_view(ex, two_runs):
    p = ex.run(two_runs[0][0] / "again", small=True)
    for c in p.collapses:
        assert c.threshold is not None and c.within_bound and c.analytic_bound == 51
    prices = [r["mean_price"] for r in p.decay]
    assert all(x >= y for x, y in zip(prices, prices[1:])) and prices[0] > prices[-1]
    assert p.decay[-1]["mean_time_to_match"] <= p.decay[0]["mean_time_to_match"]
    assert p.split["search-cost sink"] <= 0 and p.split["decay loss"] <= 0
    total = sum(r["surplus"] for r in p.rows)
    assert p.split["buyer gain"] + p.split["seller gain"] == total


def test_report_is_offline_xml_with_heatmap_and_split(two_runs):
    doc = (two_runs[0][0] / "report.html").read_text(encoding="utf-8")
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    kinds = [s.get("class") for s in root.iter("section")]
    assert "sweep" in kinds and "pnl" in kinds
    text = "".join(root.itertext())
    for needle in ("surplus_share: search_cost x decay_per_step", "buyer gain", "seller gain",
                   "search-cost sink", "decay loss"):
        assert needle in text, needle


def test_prints_tables(ex, tmp_path, capsys):
    assert ex.main(["--small", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for needle in ("search cost sweep", "collapse (decay 0): bisected", "analytic bound 51",
                   "decay view", "surplus split"):
        assert needle in out, needle


# ---------------------------------------------------------------- Phase 5 summary flag
@pytest.fixture(scope="module")
def summaries(tmp_path_factory):
    mod = load("phase5_summary_report")
    out = tmp_path_factory.mktemp("summary")
    assert mod.main(["--out", str(out / "plain")]) == 0
    assert mod.main(["--out", str(out / "p7"), "--include-phase7"]) == 0
    return [(out / d / "phase5_summary.html").read_bytes() for d in ("plain", "p7")]


def _experiments(doc: bytes):
    root = parse(doc.decode("utf-8"))
    return [s for s in root.iter("section") if s.get("class") == "experiment"]


def test_summary_without_flag_is_unchanged(summaries):
    assert hashlib.sha256(summaries[0]).hexdigest() == SUMMARY_SHA256
    assert len(_experiments(summaries[0])) == 5


def test_summary_with_flag_has_six_sections(summaries):
    secs = _experiments(summaries[1])
    assert len(secs) == 6 and secs[-1].find("h2").text == "search_cost (seed 47)"
    assert not re.search(rb"https?://", summaries[1], re.I)


# ---------------------------------------------------------------- docs, CI, checksums
def test_index_links_writeup_and_site_passes_check(tmp_path, capsys):
    assert "(phase7-search-cost.md)" in (DOCS / "index.md").read_text("utf-8")
    res = build_site(DOCS, tmp_path / "_site")
    assert (tmp_path / "_site" / "phase7-search-cost.html").is_file() and res
    assert check_site(tmp_path / "_site", src=DOCS).ok
    assert cli.main(["docs", "check", str(tmp_path / "_site"), "--src", str(DOCS)]) == 0


def test_ci_validate_step_covers_config():
    wf = (ROOT / "ci" / "ci.yml").read_text("utf-8")
    assert "for cfg in examples/configs/*.json" in wf and "frictionlab validate" in wf
    assert (ROOT / CONFIG).is_file()
    assert cli.main(["validate", str(ROOT / CONFIG)]) == 0


def test_roadmap_ticks_phase7_example_and_writeup():
    text = (ROOT / "ROADMAP.md").read_text("utf-8")
    assert "- [x] Search-cost experiment" in text
    assert "docs/phase7-search-cost.md" in text and "examples/phase7_search_cost.py" in text
    assert "Phase 7 status: in progress (2 of 3)" in text


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
