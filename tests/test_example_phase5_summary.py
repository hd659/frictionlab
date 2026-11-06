"""fl-035: examples/phase5_summary_report.py (all five Phase 5 experiments, one HTML).

fl-051: `--include-phase7` appends search cost and intermediation (seven sections);
the default five-section output stays byte-identical.
"""
import hashlib
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NAMES = ("coase_threshold", "decision_cost", "adverse_selection", "fee_wars", "funding_arb")
SEEDS = {"coase_threshold": 28, "decision_cost": 29, "adverse_selection": 30, "fee_wars": 32,
         "funding_arb": 34}
BUDGET_S = 120.0
# sha256 of the default five-section report, frozen before --include-phase7 existed (fl-048)
SUMMARY_SHA256 = "473013de2e67abb669812cab1c4e1a33c5a5c95259032327d537feafcab2a7a9"
PHASE7 = {"search_cost": 47, "intermediation": 50}


def load():
    if "phase5_summary_report" in sys.modules:
        return sys.modules["phase5_summary_report"]
    spec = importlib.util.spec_from_file_location("phase5_summary_report",
                                                  EXAMPLES / "phase5_summary_report.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase5_summary_report"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ex():
    return load()


@pytest.fixture(scope="module")
def run_once(ex, tmp_path_factory):
    out = tmp_path_factory.mktemp("summary")
    t0 = time.perf_counter()
    assert ex.main(["--out", str(out)]) == 0
    elapsed = time.perf_counter() - t0
    return out / "phase5_summary.html", elapsed


def parse(doc: str) -> ET.Element:
    assert doc.startswith("<!DOCTYPE html>\n")
    root = ET.fromstring(doc.split("\n", 1)[1])
    assert root.tag == "html"
    return root


def experiment_sections(root):
    return [s for s in root.iter("section") if s.get("class") == "experiment"]


def test_main_runs_within_budget(run_once):
    path, elapsed = run_once
    assert path.is_file() and elapsed < BUDGET_S


def test_report_is_wellformed_offline_html(run_once):
    doc = run_once[0].read_text(encoding="utf-8")
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    assert not list(root.iter("script")) and not list(root.iter("link"))
    assert all(a.get("href", "").startswith("#") for a in root.iter("a"))


def test_exactly_five_experiment_sections_in_order(run_once):
    root = parse(run_once[0].read_text(encoding="utf-8"))
    secs = experiment_sections(root)
    assert len(secs) == 5
    titles = [s.find("h2").text for s in secs]
    assert titles == [f"{n} (seed {SEEDS[n]})" for n in NAMES]
    for s in secs:
        assert s.find("p").get("class") == "headline" and s.find("p").text
        assert [t.get("class") for t in s.iter("table")][:2] == ["data", "aggregate"]
        assert s.find(".//figure") is not None or s.find(".//svg") is not None


def test_header_carries_headline_findings(run_once):
    root = parse(run_once[0].read_text(encoding="utf-8"))
    header = "".join(root.find("body/header").itertext())
    for n in NAMES:
        assert n in header and f"seed {n}" in header
    for needle in ("F* =", "c*=", "batch auction", "fixed_point", "gas threshold"):
        assert needle in header


def test_byte_identical_across_two_runs(ex, run_once, tmp_path):
    assert ex.main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "phase5_summary.html").read_bytes() == run_once[0].read_bytes()


def test_sections_match_the_standalone_examples(ex):
    res, headline, *_ = ex.funding_arb()
    st = ex.load("phase5_funding_arb").run(34, small=True)
    assert res.rows == st.res.rows and res.findings == st.res.findings
    assert str(st.res.findings["gas_threshold_bisected"]) in headline


def test_default_report_is_frozen_and_has_five_sections(run_once):
    data = run_once[0].read_bytes()
    assert hashlib.sha256(data).hexdigest() == SUMMARY_SHA256
    assert len(experiment_sections(parse(data.decode("utf-8")))) == 5


@pytest.fixture(scope="module")
def with_phase7(ex, tmp_path_factory):
    out = tmp_path_factory.mktemp("summary_p7")
    assert ex.main(["--out", str(out), "--include-phase7"]) == 0
    return (out / "phase5_summary.html").read_text(encoding="utf-8")


def test_phase7_flag_appends_search_cost_and_intermediation(with_phase7):
    root = parse(with_phase7)
    secs = experiment_sections(root)
    titles = [s.find("h2").text for s in secs]
    assert len(secs) == 7
    assert titles[:5] == [f"{n} (seed {SEEDS[n]})" for n in NAMES]
    assert titles[5:] == [f"{n} (seed {s})" for n, s in PHASE7.items()]
    assert not re.search(r"https?://", with_phase7, re.I)
    header = "".join(root.find("body/header").itertext())
    assert "intermediated share" in header and "search cost 3 (analytic bound 3)" in header
    assert "Sweep: intermediated_share by search_cost, middleman_half_spread" in "".join(
        secs[-1].itertext())


def test_scalars_drops_nested_findings(ex):
    flat = ex.scalars({"a": 1, "b": 2.5, "c": "x", "d": (1, 2), "e": [1], "f": {"g": 1}, "h": None})
    assert flat == {"a": 1, "b": 2.5, "c": "x", "d": "(1, 2)", "h": None}
