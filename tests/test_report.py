import re
import xml.etree.ElementTree as ET

import pytest

from frictionlab.analysis import (
    EquilibriumSection, MetricsSection, PnLBreakdown, PnLSection, Report, SweepSection,
    SweepSpec, evolve, frictionlab_version, matrix_fitness, run_sweep, write_report,
)
from frictionlab.analysis.metrics import EfficiencyStats, SpreadDecomposition
from frictionlab.analysis.welfare import Equilibrium, WelfareReport

GRID = {"taker_bps": [0, 10], "latency": [0.0, 2.0]}
SEED = 20261030


def _factory(params, seed):
    if params["taker_bps"] == 10 and params["latency"] == 2.0:
        raise RuntimeError("venue halted <latency> & \"fee\"")
    return {"net": 100.0 - params["taker_bps"] - 5 * params["latency"] + seed % 3}


def sweep_rows():
    spec = SweepSpec(factory=_factory, metrics=lambda r: r, grid=GRID,
                     replicates=2, master_seed=SEED)
    return run_sweep(spec)


def build(title="Fee x latency <draft> & \"v1\"") -> Report:
    w = WelfareReport(Equilibrium(4, 80, 99, 101), 72, {1: PnLBreakdown(72, 4, 2, 4)})
    eq = evolve([0.3, 0.7], matrix_fitness([[0, 3], [1, 2]]), eta=0.5,
                generations=2000, labels=("satisficer", "optimizer"))
    rep = Report(title, master_seed=SEED, grid=GRID, meta={"scenario": "zi <C> & 'book'"})
    rep.add(MetricsSection(spread=SpreadDecomposition(5.0, 2.0, 0.75, 1.25, 30, 9, 0),
                           efficiency=EfficiencyStats(0.5, 1.0, 0.98), welfare=w))
    rep.add(SweepSection(sweep_rows(), by=("taker_bps", "latency"), metric="net"))
    rep.add(EquilibriumSection(eq))
    rep.add(PnLSection({"ZI-C": PnLBreakdown(30, 2, 0, 6), "informed": PnLBreakdown(-12, 1, 3, 2)}))
    return rep


def parse(doc: str) -> ET.Element:
    assert doc.startswith("<!DOCTYPE html>\n")
    root = ET.fromstring(doc.split("\n", 1)[1])
    assert root.tag == "html"
    return root


def by_class(root, tag, cls):
    return [e for e in root.iter(tag) if e.get("class") == cls]


def test_wellformed_offline_no_urls_no_script():
    doc = build().to_html()
    root = parse(doc)
    assert root.find("head/style") is not None and root.find("head/title") is not None
    assert not re.search(r"https?://|//cdn|@import|url\(http", doc, re.I)
    assert not list(root.iter("script")) and not list(root.iter("link"))
    assert all(a.get("href", "").startswith("#") for a in root.iter("a"))
    assert all(not (e.get("src") or "") for e in root.iter())


def test_every_section_renders_title_and_rows():
    root = parse(build().to_html())
    secs = root.find("body/main").findall("section")
    assert [s.get("class") for s in secs] == ["metrics", "sweep", "equilibrium", "pnl"]
    assert [s.find("h2").text for s in secs] == [
        "Market quality", "Parameter sweep", "Strategy-mix equilibrium", "PnL by agent type"]
    counts = [len(s.find("table/tbody")) for s in secs]
    assert counts == [12, 4, 2, 2]
    assert len(list(secs[1].iter("svg"))) == 1  # heatmap for two swept keys
    assert len(root.find("body/nav/ol")) == 4
    assert [a.get("href") for a in root.iter("a")] == ["#" + s.get("id") for s in secs]


def test_user_strings_are_escaped():
    doc = build().to_html()
    assert "<draft>" not in doc and "&lt;draft&gt; &amp; &quot;v1&quot;" in doc
    assert "zi &lt;C&gt; &amp; &#x27;book&#x27;" in doc
    assert "&lt;latency&gt; &amp; &quot;fee&quot;" in doc
    root = parse(doc)
    assert root.find("head/title").text == 'Fee x latency <draft> & "v1"'


def test_sweep_error_rows_listed_not_dropped():
    rows = sweep_rows()
    bad = [r for r in rows if r["error"] is not None]
    assert len(bad) == 2
    root = parse(build().to_html())
    errs = by_class(root, "table", "errors")[0].find("tbody")
    assert len(errs) == 2
    assert [tr[2].text for tr in errs] == [str(r["seed"]) for r in bad]
    assert all(tr[-1].text == 'RuntimeError: venue halted <latency> & "fee"' for tr in errs)
    agg = by_class(root, "table", "aggregate")[0].find("tbody")
    assert [tr[3].text for tr in agg] == ["0", "0", "0", "2"]


def test_header_records_seed_grid_and_version():
    root = parse(build().to_html())
    header = root.find("body/header")
    prov = {tr[0].text: tr[1].text for tr in by_class(header, "table", "provenance")[0].find("tbody")}
    assert prov["master seed"] == str(SEED)
    assert prov["frictionlab version"] == frictionlab_version() != "unknown"
    grid = {tr[0].text: tr[1].text for tr in by_class(header, "table", "grid")[0].find("tbody")}
    assert grid == {"taker_bps": "0, 10", "latency": "0, 2"}
    cells = Report(grid=[{"taker_bps": 5}], version="9.9").to_html()
    assert "taker_bps=5" in cells and "not recorded" in cells and ">9.9<" in cells


def test_same_inputs_byte_identical():
    assert build().to_html() == build().to_html()
    assert build("a").to_html() != build("b").to_html()


def test_write_report_to_tmp_path(tmp_path):
    out = build().write_report(tmp_path / "nested" / "report.html")
    data = out.read_bytes()
    assert out.is_file() and len(data) > 2000
    assert data.decode("utf-8") == build().to_html()
    one = write_report(tmp_path / "pnl.html", PnLSection({"noise": -3}), master_seed=1)
    root = parse(one.read_text(encoding="utf-8"))
    assert root.find("body/nav") is None  # single section: no table of contents
    assert len(root.find("body/main").findall("section")) == 1


def test_add_rejects_non_sections():
    with pytest.raises(TypeError):
        Report().add("<p>raw</p>")
    rep = Report()
    assert rep.add(PnLSection({"x": 1})) is rep
