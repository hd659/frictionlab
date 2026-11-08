"""fl-054 / fl-057 / fl-060: `phase5_summary_report.py --include-phase8/9/10`.

The default five-section report and the `--include-phase7/8/9` reports are frozen
by sha256, so no later flag can perturb any of them.
"""
import hashlib
import importlib.util
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
# sha256 of the default report (fl-048) and of the --include-phase7 report (fl-051)
DEFAULT_SHA256 = "473013de2e67abb669812cab1c4e1a33c5a5c95259032327d537feafcab2a7a9"
PHASE7_SHA256 = "c980e163fc4ca28f714d192040d8e2f1f7db847f9d52a3ba608063def4e7ba35"
# sha256 of the --include-phase8 report and of --include-phase7 --include-phase8 (fl-054)
PHASE8_SHA256 = "2a4e29b010c6607fe52169194a1d07ae2cc150f79633b7f2c5d21b295695fbde"
PHASE78_SHA256 = "54929b57560292671bf7eecabd865fea0c857cc4ec86e3bcd6795f8b7a1764b7"
# sha256 of --include-phase9 and of --include-phase7 --include-phase8 --include-phase9 (fl-057)
PHASE9_SHA256 = "032e95141efa935f2fe455eef65e33fbba5872250721bf05689b3b29a51777d5"
PHASE789_SHA256 = "2242f03058075f8a0224273d6b4b30e5ac3595c0b8bc152b6eedc43dbe7ef1cb"
GS_TITLE = "grossman_stiglitz (seed 53)"
LM_TITLE = "lemons (seed 56)"
RP_TITLE = "reputation (seed 59)"
ALL9 = ("--include-phase7", "--include-phase8", "--include-phase9")


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
def reports(ex, tmp_path_factory):
    """Bytes of the report for each flag combination, keyed by the flags."""
    out = {}
    for flags in ((), ("--include-phase7",), ("--include-phase8",),
                  ("--include-phase7", "--include-phase8"), ("--include-phase9",),
                  ALL9, ("--include-phase10",), ALL9 + ("--include-phase10",)):
        d = tmp_path_factory.mktemp("summary")
        assert ex.main(["--out", str(d), *flags]) == 0
        out[flags] = (d / "phase5_summary.html").read_bytes()
    return out


def parse(data: bytes) -> ET.Element:
    doc = data.decode("utf-8")
    assert doc.startswith("<!DOCTYPE html>\n")
    return ET.fromstring(doc.split("\n", 1)[1])


def titles(data: bytes):
    return [s.find("h2").text for s in parse(data).iter("section") if s.get("class") == "experiment"]


def test_default_report_is_byte_identical_with_five_sections(reports):
    data = reports[()]
    assert hashlib.sha256(data).hexdigest() == DEFAULT_SHA256
    assert len(titles(data)) == 5


def test_phase7_report_is_unchanged(reports):
    data = reports[("--include-phase7",)]
    assert hashlib.sha256(data).hexdigest() == PHASE7_SHA256
    assert len(titles(data)) == 7 and GS_TITLE not in titles(data)


def test_phase8_flag_adds_exactly_one_section(reports):
    base, got = titles(reports[()]), titles(reports[("--include-phase8",)])
    assert got == base + [GS_TITLE]
    both = titles(reports[("--include-phase7", "--include-phase8")])
    assert both == titles(reports[("--include-phase7",)]) + [GS_TITLE]


def test_phase8_section_is_offline_and_carries_headline(reports):
    data = reports[("--include-phase8",)]
    doc = data.decode("utf-8")
    assert not re.search(r"https?://", doc, re.I)
    root = parse(data)
    header = "".join(root.find("body/header").itertext())
    assert "information collapse at 7 ticks (analytic 8)" in header
    assert "Phase 8 Grossman-Stiglitz" in "".join(root.find("head/title").itertext())
    sec = [s for s in root.iter("section") if s.get("class") == "experiment"][-1]
    text = "".join(sec.itertext())
    assert "Sweep: buyer_fraction by signal_cost, noise_intensity" in text
    assert sec.find("p").get("class") == "headline"


def test_phase8_section_matches_run_from_config(ex):
    from frictionlab.experiments import run_from_config
    res, headline, rows, by, metric = ex.grossman_stiglitz()
    direct = run_from_config(EXAMPLES / "configs" / "grossman_stiglitz_small.json")
    assert rows == direct.rows and res.findings == direct.findings
    assert by == ["signal_cost", "noise_intensity"] and metric == "buyer_fraction"
    assert f"collapse at {direct.findings['collapse_threshold']} ticks" in headline


def test_phase8_report_is_byte_reproducible(ex, reports, tmp_path):
    assert ex.main(["--out", str(tmp_path), "--include-phase8"]) == 0
    assert (tmp_path / "phase5_summary.html").read_bytes() == reports[("--include-phase8",)]


# ---------------------------------------------------------------- fl-057: --include-phase9
def test_phase8_reports_are_unchanged(reports):
    assert hashlib.sha256(reports[("--include-phase8",)]).hexdigest() == PHASE8_SHA256
    both = reports[("--include-phase7", "--include-phase8")]
    assert hashlib.sha256(both).hexdigest() == PHASE78_SHA256
    assert LM_TITLE not in titles(both)


def test_phase9_flag_adds_exactly_one_section(reports):
    assert titles(reports[("--include-phase9",)]) == titles(reports[()]) + [LM_TITLE]
    every = titles(reports[("--include-phase7", "--include-phase8", "--include-phase9")])
    assert every == titles(reports[("--include-phase7", "--include-phase8")]) + [LM_TITLE]


def test_phase9_section_is_offline_and_carries_headline(reports):
    data = reports[("--include-phase9",)]
    assert not re.search(r"https?://", data.decode("utf-8"), re.I)
    root = parse(data)
    header = "".join(root.find("body/header").itertext())
    assert "certification stops at 21 ticks (analytic bound 21) at premium 20" in header
    assert "Phase 9 lemons" in "".join(root.find("head/title").itertext())
    sec = [s for s in root.iter("section") if s.get("class") == "experiment"][-1]
    assert "Sweep: surplus_share by certification_cost, buyer_premium" in "".join(sec.itertext())
    assert sec.find("p").get("class") == "headline"


def test_phase9_section_matches_run_from_config(ex):
    from frictionlab.experiments import run_from_config
    res, headline, rows, by, metric = ex.lemons()
    direct = run_from_config(EXAMPLES / "configs" / "lemons_small.json")
    assert rows == direct.rows and res.findings == direct.findings
    assert by == ["certification_cost", "buyer_premium"] and metric == "surplus_share"
    assert f"stops at {direct.findings['certification_threshold']} ticks" in headline


def test_phase9_report_is_byte_reproducible(ex, reports, tmp_path):
    assert ex.main(["--out", str(tmp_path), "--include-phase9"]) == 0
    assert (tmp_path / "phase5_summary.html").read_bytes() == reports[("--include-phase9",)]


# ---------------------------------------------------------------- fl-060: --include-phase10
def test_phase9_reports_are_unchanged(reports):
    assert hashlib.sha256(reports[("--include-phase9",)]).hexdigest() == PHASE9_SHA256
    assert hashlib.sha256(reports[ALL9]).hexdigest() == PHASE789_SHA256
    assert RP_TITLE not in titles(reports[ALL9])


def test_phase10_flag_adds_exactly_one_section(reports):
    assert titles(reports[("--include-phase10",)]) == titles(reports[()]) + [RP_TITLE]
    assert titles(reports[ALL9 + ("--include-phase10",)]) == titles(reports[ALL9]) + [RP_TITLE]


def test_phase10_section_is_offline_and_carries_headline(reports):
    data = reports[("--include-phase10",)]
    assert not re.search(r"https?://", data.decode("utf-8"), re.I)
    root = parse(data)
    header = "".join(root.find("body/header").itertext())
    assert "whitewashing stops at 21 ticks (deterrent bound 38) at accuracy 8500 bps" in header
    assert "Phase 10 reputation" in "".join(root.find("head/title").itertext())
    sec = [s for s in root.iter("section") if s.get("class") == "experiment"][-1]
    assert "Sweep: surplus_share by rating_accuracy_bps, whitewash_cost" in "".join(sec.itertext())
    assert sec.find("p").get("class") == "headline"


def test_phase10_section_matches_run_from_config(ex):
    from frictionlab.experiments import run_from_config
    res, headline, rows, by, metric = ex.reputation()
    direct = run_from_config(EXAMPLES / "configs" / "reputation_small.json")
    assert rows == direct.rows and res.findings == direct.findings
    assert by == ["rating_accuracy_bps", "whitewash_cost"] and metric == "surplus_share"
    assert f"stops at {direct.findings['whitewash_threshold']} ticks" in headline


def test_phase10_report_is_byte_reproducible(ex, reports, tmp_path):
    assert ex.main(["--out", str(tmp_path), "--include-phase10"]) == 0
    assert (tmp_path / "phase5_summary.html").read_bytes() == reports[("--include-phase10",)]
