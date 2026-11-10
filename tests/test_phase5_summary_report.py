"""fl-054 / fl-057 / fl-060 / fl-066 / fl-069 / fl-072: `phase5_summary_report.py --include-phase8..14`.

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


# ---------------------------------------------------------------- fl-066: --include-phase12
# sha256 of --include-phase11 and of --include-phase7 ... --include-phase11 (fl-063)
PHASE11_SHA256 = "803a7b61b0139e0efe5f32ee4fd4331b13b5eaf6034c1c1e4c840086ea8ea7ab"
PHASE7_11_SHA256 = "e93c05c4e042aa517ed8799989674daf4169cd7a42a1765abb8ad6d6d0fede3d"
CM_TITLE = "critical_mass (seed 65)"
ALL11 = ALL9 + ("--include-phase10", "--include-phase11")
P12 = ("--include-phase12",)


@pytest.fixture(scope="module")
def reports12(ex, reports, tmp_path_factory):
    out = {(): reports[()]}
    for flags in (("--include-phase11",), ALL11, P12, ALL11 + P12):
        d = tmp_path_factory.mktemp("summary12")
        assert ex.main(["--out", str(d), *flags]) == 0
        out[flags] = (d / "phase5_summary.html").read_bytes()
    return out


def test_phase11_reports_are_unchanged(reports12):
    assert hashlib.sha256(reports12[()]).hexdigest() == DEFAULT_SHA256
    assert hashlib.sha256(reports12[("--include-phase11",)]).hexdigest() == PHASE11_SHA256
    assert hashlib.sha256(reports12[ALL11]).hexdigest() == PHASE7_11_SHA256
    assert CM_TITLE not in titles(reports12[ALL11])


def test_phase12_flag_adds_exactly_one_section(reports12):
    assert titles(reports12[P12]) == titles(reports12[()]) + [CM_TITLE]
    assert titles(reports12[ALL11 + P12]) == titles(reports12[ALL11]) + [CM_TITLE]


def test_phase12_section_is_offline_and_carries_headline(reports12):
    data = reports12[P12]
    assert not re.search(r"https?://", data.decode("utf-8"), re.I)
    root = parse(data)
    header = "".join(root.find("body/header").itertext())
    assert "4 cold-start traps without seeding" in header
    assert "from seed 8 (analytic critical mass 8) at adoption cost 2" in header
    assert "Phase 12 critical mass" in "".join(root.find("head/title").itertext())
    sec = [s for s in root.iter("section") if s.get("class") == "experiment"][-1]
    assert "Sweep: share by adoption_cost, seed_members" in "".join(sec.itertext())
    assert sec.find("p").get("class") == "headline"


def test_phase12_section_matches_run_from_config(ex, reports12, tmp_path):
    from frictionlab.experiments import run_from_config
    res, headline, rows, by, metric = ex.critical_mass()
    direct = run_from_config(EXAMPLES / "configs" / "critical_mass_small.json")
    assert rows == direct.rows and res.findings == direct.findings
    assert by == ["adoption_cost", "seed_members"] and metric == "share"
    assert f"from seed {direct.findings['tipping_seed']} " in headline
    assert ex.main(["--out", str(tmp_path), *P12]) == 0
    assert (tmp_path / "phase5_summary.html").read_bytes() == reports12[P12]


# ---------------------------------------------------------------- fl-069: --include-phase13
# sha256 of --include-phase12 and of --include-phase7 ... --include-phase12 (fl-066)
PHASE12_SHA256 = "82ed9961472dedb8568536455410e23e4767a7018ff58d3f649a41a13ee3976c"
PHASE7_12_SHA256 = "cf80baea7cb31353d9b70502eae35b97fd87dc20845ba377e6c4fdafc5686a3c"
CG_TITLE = "congestion (seed 68)"
ALL12 = ALL11 + P12
P13 = ("--include-phase13",)


@pytest.fixture(scope="module")
def reports13(ex, reports12, tmp_path_factory):
    out = dict(reports12)
    for flags in (P13, ALL12 + P13):
        d = tmp_path_factory.mktemp("summary13")
        assert ex.main(["--out", str(d), *flags]) == 0
        out[flags] = (d / "phase5_summary.html").read_bytes()
    return out


def test_phase12_reports_are_unchanged(reports13):
    assert hashlib.sha256(reports13[()]).hexdigest() == DEFAULT_SHA256
    assert len(titles(reports13[()])) == 5
    assert hashlib.sha256(reports13[P12]).hexdigest() == PHASE12_SHA256
    assert hashlib.sha256(reports13[ALL12]).hexdigest() == PHASE7_12_SHA256
    assert CG_TITLE not in titles(reports13[ALL12])


def test_phase13_flag_adds_exactly_one_section(reports13):
    assert titles(reports13[P13]) == titles(reports13[()]) + [CG_TITLE]
    assert titles(reports13[ALL12 + P13]) == titles(reports13[ALL12]) + [CG_TITLE]


def test_phase13_section_is_offline_and_carries_headline(reports13):
    data = reports13[P13]
    assert not re.search(r"https?://", data.decode("utf-8"), re.I)
    root = parse(data)
    header = "".join(root.find("body/header").itertext())
    assert "individual threshold 5 vs social 0 at waiting cost 2" in header
    assert "optimal toll 17 (analytic 17)" in header
    assert "Phase 13 congestion" in "".join(root.find("head/title").itertext())
    sec = [s for s in root.iter("section") if s.get("class") == "experiment"][-1]
    assert "Sweep: welfare by toll, waiting_cost" in "".join(sec.itertext())
    assert sec.find("p").get("class") == "headline"


def test_phase13_section_matches_run_from_config(ex, reports13, tmp_path):
    from frictionlab.experiments import run_from_config
    res, headline, rows, by, metric = ex.congestion()
    direct = run_from_config(EXAMPLES / "configs" / "congestion_small.json")
    assert rows == direct.rows and res.findings == direct.findings
    assert by == ["toll", "waiting_cost"] and metric == "welfare"
    assert f"optimal toll {direct.findings['bisected_toll']} " in headline
    assert ex.main(["--out", str(tmp_path), *P13]) == 0
    assert (tmp_path / "phase5_summary.html").read_bytes() == reports13[P13]


# ---------------------------------------------------------------- fl-072: --include-phase14
# sha256 of --include-phase13 and of --include-phase7 ... --include-phase13 (fl-069)
PHASE13_SHA256 = "9286f57234711868a6f288604651251c5f58f793ebaa43677880288cd784b9d5"
PHASE7_13_SHA256 = "6dd363f5031e1d058258ec6a9e9cb49fe429e67edb51a8ca5987dd59fee1a7af"
MC_TITLE = "menu_cost (seed 71)"
ALL13 = ALL12 + P13
P14 = ("--include-phase14",)


@pytest.fixture(scope="module")
def reports14(ex, reports13, tmp_path_factory):
    out = dict(reports13)
    for flags in (P14, ALL13 + P14):
        d = tmp_path_factory.mktemp("summary14")
        assert ex.main(["--out", str(d), *flags]) == 0
        out[flags] = (d / "phase5_summary.html").read_bytes()
    return out


def test_phase13_reports_are_unchanged(reports14):
    assert hashlib.sha256(reports14[()]).hexdigest() == DEFAULT_SHA256
    assert len(titles(reports14[()])) == 5
    assert hashlib.sha256(reports14[P13]).hexdigest() == PHASE13_SHA256
    assert hashlib.sha256(reports14[ALL13]).hexdigest() == PHASE7_13_SHA256
    assert MC_TITLE not in titles(reports14[ALL13])


def test_phase14_flag_adds_exactly_one_section(reports14):
    assert titles(reports14[P14]) == titles(reports14[()]) + [MC_TITLE]
    assert titles(reports14[ALL13 + P14]) == titles(reports14[ALL13]) + [MC_TITLE]


def test_phase14_section_is_offline_and_carries_headline(reports14):
    data = reports14[P14]
    assert not re.search(r"https?://", data.decode("utf-8"), re.I)
    root = parse(data)
    header = "".join(root.find("body/header").itertext())
    assert "(S,s) bands up to 4 ticks cut adjustment frequency to 0.2" in header
    assert "rigidity threshold 17 ticks (analytic 17, band 4) at inflation 2" in header
    assert "Phase 14 menu costs" in "".join(root.find("head/title").itertext())
    sec = [s for s in root.iter("section") if s.get("class") == "experiment"][-1]
    assert ("Sweep: adjustment_frequency by menu_cost, inflation_ticks_per_step"
            in "".join(sec.itertext()))
    assert sec.find("p").get("class") == "headline"


def test_phase14_section_matches_run_from_config(ex, reports14, tmp_path):
    from frictionlab.experiments import run_from_config
    res, headline, rows, by, metric = ex.menu_cost()
    direct = run_from_config(EXAMPLES / "configs" / "menu_cost_small.json")
    assert rows == direct.rows and res.findings == direct.findings
    assert by == ["menu_cost", "inflation_ticks_per_step"] and metric == "adjustment_frequency"
    assert f"rigidity threshold {direct.findings['bisected_threshold']} ticks" in headline
    assert ex.main(["--out", str(tmp_path), *P14]) == 0
    assert (tmp_path / "phase5_summary.html").read_bytes() == reports14[P14]
