"""fl-060: examples/phase10_reputation.py, its write-up, docs index and ROADMAP status."""
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab import bench, cli
from frictionlab.docs_site import build_site, check_site
from frictionlab.experiments import reputation as rp

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES, DOCS = ROOT / "examples", ROOT / "docs"
CONFIG = "examples/configs/reputation_small.json"
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
    return load("phase10_reputation")


@pytest.fixture(scope="module")
def two_runs(ex, tmp_path_factory):
    outs, times = [], []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp("phase10") / name
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


def test_surplus_share_non_decreasing_in_rating_accuracy(ex, p):
    accs = sorted(p.params["rating_accuracies"])
    for c in p.params["whitewash_costs"]:
        s = [ex.mean_by(p.rows, "surplus_share", **{ex.ACC: a, ex.WASH: c}) for a in accs]
        assert all(y >= x for x, y in zip(s, s[1:])), (c, s)
    avg = [p.surplus_share(a) for a in accs]
    assert all(y >= x for x, y in zip(avg, avg[1:])) and avg[-1] > avg[0]


def test_rows_match_the_experiment_called_directly(p):
    direct = rp.run_reputation_grid(p.cfg, p.params["rating_accuracies"], p.params["whitewash_costs"],
                                    master_seed=p.master_seed)
    keys = ("trades", "cheats", "whitewashes", "whitewash_paid", "violations")
    assert [[r[k] for k in keys] for r in direct.rows] == [[r[k] for k in keys] for r in p.rows]
    assert all(r["violations"] == 0 for r in p.rows)


def test_traced_opportunist_milks_in_the_end_game(p):
    assert p.seller == 8 and p.cfg.horizon == 6
    assert [s[2] for s in p.steps] == [93, 103, 93, 103, 105, 105]
    assert [s[3] for s in p.steps] == [30, 30, 30, 30, 30, 10]
    assert [s[6] for s in p.steps].index("milk") == p.milking_round == 5
    assert p.analytic_round == 5 and p.milking_round >= p.analytic_round


def test_thresholds_and_whitewash_counts(ex, p):
    th = p.threshold
    assert (th.threshold, th.analytic, th.accuracy) == (21, 38, 8500)
    assert th.threshold <= th.analytic
    assert th.probes[th.threshold] == 0 and th.probes[th.threshold - 1] > 0
    washes = {(r[ex.ACC], r[ex.WASH]): r["whitewashes"] for r in p.rows}
    assert washes[(6000, 0)] == 15 and all(washes[(10000, c)] == 0 for c in p.params["whitewash_costs"])


def test_surplus_split_reconciles(ex, p):
    s = p.split
    assert tuple(s) == ex.SPLIT_KEYS
    assert -s["identity sink"] == sum(r["whitewash_paid"] for r in p.rows) == 80
    assert (s["buyer gain"], s["honest-seller gain"], s["opportunist cheating gain"]) == (-6850, 34546, 23244)
    premium = p.cfg.premium                 # every trade creates exactly `premium` of surplus
    assert sum(s.values()) == premium * sum(r["trades"] for r in p.rows) - 80


def test_report_is_offline_xml_with_heatmap_path_and_split(two_runs):
    doc = (two_runs[0][0] / "report.html").read_text(encoding="utf-8")
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    kinds = [s.get("class") for s in root.iter("section")]
    assert {"sweep", "path", "pnl"} <= set(kinds)
    text = "".join(root.itertext())
    for needle in ("surplus_share: rating_accuracy_bps x whitewash_cost",
                   "Price vs delivered-quality value", "buyer gain", "honest-seller gain",
                   "opportunist cheating gain", "identity sink", "bisected whitewash threshold"):
        assert needle in text, needle


def test_prints_tables(ex, tmp_path, capsys):
    assert ex.main(["--small", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    for needle in ("reputation sweep (seed 59", "seller 8 (accuracy 8500 bps",
                   "round 5: score 4/1 price 105 delivered 10 (bad) milk",
                   "milking round: simulated 5, analytic 5",
                   "bisected 21, analytic deterrent bound 38", "surplus split"):
        assert needle in out, needle


# ---------------------------------------------------------------- docs, CI, roadmap, checksums
def test_writeup_numbers_match_the_run(p):
    text = " ".join((DOCS / "phase10-reputation.md").read_text("utf-8").split())
    for needle in ("seed 59", "93 -> 103 -> 93 -> 103 -> 105 -> 105", "milks in round 5",
                   "21 ticks against the analytic deterrent bound of 38", "93 - 55 = 38",
                   "15 whitewashes", f"buyer gain {p.split['buyer gain']}", "identity sink -80",
                   "phase9-lemons.md", "phase5-coase-threshold.md"):
        assert needle in text, needle


def test_index_links_writeup_and_site_passes_check(tmp_path):
    assert "(phase10-reputation.md)" in (DOCS / "index.md").read_text("utf-8")
    build_site(DOCS, tmp_path / "_site")
    assert (tmp_path / "_site" / "phase10-reputation.html").is_file()
    assert check_site(tmp_path / "_site", src=DOCS).ok
    assert cli.main(["docs", "check", str(tmp_path / "_site"), "--src", str(DOCS)]) == 0


def test_ci_validate_step_covers_config():
    wf = (ROOT / "ci" / "ci.yml").read_text("utf-8")
    assert "for cfg in examples/configs/*.json" in wf and "frictionlab validate" in wf
    assert (ROOT / CONFIG).is_file() and cli.main(["validate", str(ROOT / CONFIG)]) == 0


def test_roadmap_ticks_phase10_and_reports_status():
    text = (ROOT / "ROADMAP.md").read_text("utf-8")
    phase10 = text.split("## Phase 10", 1)[1].split("\n## ", 1)[0]
    assert "- [ ]" not in phase10
    assert "docs/phase10-reputation.md" in phase10 and "examples/phase10_reputation.py" in phase10
    assert "Phase 10 status: complete (3 of 3)" in phase10
    assert "21 ticks against an analytic deterrent bound of 38" in phase10


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
