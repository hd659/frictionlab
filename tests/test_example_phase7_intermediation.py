"""fl-051: examples/phase7_intermediation.py, its write-up and the summary-report seventh section."""
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab import bench, cli
from frictionlab.docs_site import build_site, check_site
from frictionlab.experiments import intermediation as im

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOCS = ROOT / "docs"
CONFIG = "examples/configs/intermediation_small.json"
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
    return load("phase7_intermediation")


@pytest.fixture(scope="module")
def two_runs(ex, tmp_path_factory):
    outs, times = [], []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp("phase7im") / name
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


def test_intermediated_share_non_decreasing_in_search_cost(ex, two_runs):
    rows = ex.read_rows(two_runs[0][0] / "run" / "rows.csv")
    costs, spreads = sorted({r[ex.COST] for r in rows}), sorted({r[ex.SPREAD] for r in rows})
    for h in spreads:
        shares = [ex.mean_by(rows, "intermediated_share", **{ex.COST: c, ex.SPREAD: h}) for c in costs]
        assert all(x <= y for x, y in zip(shares, shares[1:])), (h, shares)
        assert shares[-1] > 0.5
    for c in costs:                                           # and non-increasing in the spread
        shares = [ex.mean_by(rows, "intermediated_share", **{ex.COST: c, ex.SPREAD: h}) for h in spreads]
        assert all(x >= y for x, y in zip(shares, shares[1:])), (c, shares)


def test_rows_match_the_experiment_called_directly(ex, p):
    direct = im.run_intermediation(p.cfg, sorted({r[ex.COST] for r in p.rows}),
                                   sorted({r[ex.SPREAD] for r in p.rows}), master_seed=p.master_seed)
    keys = ("intermediated_share", "volume", "mm_net_pnl", "search_cost_paid", "surplus")
    assert [[r[k] for k in keys] for r in direct.rows] == [[r[k] for k in keys] for r in p.rows]


def test_thresholds_match_analytic_bound(p):
    got = {t.half_spread: (t.threshold, t.analytic_bound, t.within_bound) for t in p.thresholds}
    assert got == {0: (0, 0, True), 4: (3, 3, True), 16: (11, 11, True)}
    for t in p.thresholds:
        if t.threshold:
            assert t.probes[t.threshold] > 0.5 >= t.probes[t.threshold - 1]


def test_pnl_view_carry_eats_the_spread(p):
    nets = [r["net"] for r in p.carry]
    assert all(x > y for x, y in zip(nets, nets[1:])) and nets[0] > 0 > nets[-1]
    for r in p.carry:
        assert r["spread"] - r["holding"] - r["decay"] == r["net"]
        assert 0 <= r["peak"] <= p.cfg.capacity
    assert len({r["spread"] for r in p.carry}) == 1 and p.break_even_carry == 4


def test_surplus_split_reconciles(ex, p):
    s = p.split
    assert tuple(s) == ex.SPLIT_KEYS
    assert s["buyer gain"] + s["seller gain"] + s["middleman net"] - s["holding-cost sink"] \
        - s["decay loss"] == sum(r["surplus"] for r in p.rows)
    assert s["middleman net"] == sum(r["mm_net_pnl"] for r in p.rows)
    assert s["search-cost sink"] == -sum(r["search_cost_paid"] for r in p.rows)
    assert all(s[k] <= 0 for k in ("search-cost sink", "holding-cost sink", "decay loss"))


def test_report_is_offline_xml_with_heatmap_and_split(two_runs):
    doc = (two_runs[0][0] / "report.html").read_text(encoding="utf-8")
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    kinds = [s.get("class") for s in root.iter("section")]
    assert "sweep" in kinds and "pnl" in kinds
    text = "".join(root.itertext())
    for needle in ("intermediated_share: search_cost x middleman_half_spread", "buyer gain",
                   "seller gain", "middleman net", "search-cost sink", "holding-cost sink",
                   "decay loss"):
        assert needle in text, needle


def test_prints_tables(ex, tmp_path, capsys):
    assert ex.main(["--small", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for needle in ("intermediation sweep (seed 50", "threshold (h 4): bisected 3, analytic bound 3",
                   "middleman PnL view", "break-even carry: 4", "surplus split"):
        assert needle in out, needle


# ---------------------------------------------------------------- docs, CI, roadmap, checksums
def test_writeup_numbers_match_the_run(p):
    text = " ".join((DOCS / "phase7-intermediation.md").read_text("utf-8").split())
    for needle in ("0, 3 and 11 ticks", "8.22 to 2.77", f"buyer gain {p.split['buyer gain']}",
                   f"middleman net {p.split['middleman net']}", "seed 50"):
        assert needle in text, needle


def test_index_links_writeup_and_site_passes_check(tmp_path):
    assert "(phase7-intermediation.md)" in (DOCS / "index.md").read_text("utf-8")
    build_site(DOCS, tmp_path / "_site")
    assert (tmp_path / "_site" / "phase7-intermediation.html").is_file()
    assert check_site(tmp_path / "_site", src=DOCS).ok
    assert cli.main(["docs", "check", str(tmp_path / "_site"), "--src", str(DOCS)]) == 0


def test_ci_validate_step_covers_config():
    wf = (ROOT / "ci" / "ci.yml").read_text("utf-8")
    assert "for cfg in examples/configs/*.json" in wf and "frictionlab validate" in wf
    assert (ROOT / CONFIG).is_file() and cli.main(["validate", str(ROOT / CONFIG)]) == 0


def test_roadmap_ticks_remaining_phase7_intermediation_items():
    text = (ROOT / "ROADMAP.md").read_text("utf-8")
    phase7 = text.split("## Phase 7", 1)[1].split("\n## ", 1)[0]
    assert "- [x] Intermediation experiment and write-up" in phase7
    assert "- [ ] Intermediation" not in phase7
    assert "docs/phase7-intermediation.md" in phase7 and "examples/phase7_intermediation.py" in phase7
    assert "thresholds 0 / 3 / 11 ticks" in phase7 and "8.22 -> 2.77" in phase7


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
