"""fl-063: `phase5_summary_report.py --include-phase11`.

The default five-section report and the `--include-phase10` reports are frozen by
sha256 (the phase 7/8/9 hashes are pinned in tests/test_phase5_summary_report.py),
so the new flag cannot perturb any earlier output.
"""
import hashlib
import importlib.util
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DEFAULT_SHA256 = "473013de2e67abb669812cab1c4e1a33c5a5c95259032327d537feafcab2a7a9"
# sha256 of --include-phase10 and of --include-phase7 ... --include-phase10 (fl-060)
PHASE10_SHA256 = "c074169a434e19e276ebf324ed508b5a2a10cedba81d5e18dc1a21e13802973f"
PHASE78910_SHA256 = "d8467f4264d3e33156bb7437b943bf960df6233e73574dc7cd371ce3d28b05e7"
SC_TITLE = "switching_cost (seed 62)"
ALL10 = ("--include-phase7", "--include-phase8", "--include-phase9", "--include-phase10")
P11 = ("--include-phase11",)


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
    for flags in ((), ("--include-phase10",), ALL10, P11, ALL10 + P11):
        d = tmp_path_factory.mktemp("summary11")
        assert ex.main(["--out", str(d), *flags]) == 0
        out[flags] = (d / "phase5_summary.html").read_bytes()
    return out


def parse(data: bytes) -> ET.Element:
    doc = data.decode("utf-8")
    assert doc.startswith("<!DOCTYPE html>\n")
    return ET.fromstring(doc.split("\n", 1)[1])


def titles(data: bytes):
    return [s.find("h2").text for s in parse(data).iter("section") if s.get("class") == "experiment"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_default_report_is_byte_identical_with_five_sections(reports):
    assert sha(reports[()]) == DEFAULT_SHA256
    assert len(titles(reports[()])) == 5 and SC_TITLE not in titles(reports[()])


def test_phase10_reports_are_unchanged(reports):
    assert sha(reports[("--include-phase10",)]) == PHASE10_SHA256
    assert sha(reports[ALL10]) == PHASE78910_SHA256
    assert SC_TITLE not in titles(reports[ALL10])


def test_phase11_flag_adds_exactly_one_section(reports):
    assert titles(reports[P11]) == titles(reports[()]) + [SC_TITLE]
    assert titles(reports[ALL10 + P11]) == titles(reports[ALL10]) + [SC_TITLE]


def test_phase11_section_is_offline_and_carries_headline(reports):
    data = reports[P11]
    assert not re.search(r"https?://", data.decode("utf-8"), re.I)
    root = parse(data)
    header = "".join(root.find("body/header").itertext())
    assert "markup exceeds 4 ticks from mean switching cost 7 (analytic 7) at target 6000 bps" in header
    assert "harvest markup up to 30 ticks" in header
    assert "Phase 11 switching costs" in "".join(root.find("head/title").itertext())
    sec = [s for s in root.iter("section") if s.get("class") == "experiment"][-1]
    assert "Sweep: harvest_markup by mean_switching_cost, target_share" in "".join(sec.itertext())
    assert sec.find("p").get("class") == "headline"


def test_phase11_section_matches_run_from_config(ex):
    from frictionlab.experiments import run_from_config
    res, headline, rows, by, metric = ex.switching_cost()
    direct = run_from_config(EXAMPLES / "configs" / "switching_cost_small.json")
    assert rows == direct.rows and res.findings == direct.findings
    assert by == ["mean_switching_cost", "target_share"] and metric == "harvest_markup"
    assert f"switching cost {direct.findings['lockin_threshold']} " in headline


def test_phase11_report_is_byte_reproducible(ex, reports, tmp_path):
    assert ex.main(["--out", str(tmp_path), *P11]) == 0
    assert (tmp_path / "phase5_summary.html").read_bytes() == reports[P11]
