import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from frictionlab.analysis import (
    EquilibriumSection, MetricsSection, PnLBreakdown, PnLSection, SweepSection,
    evolve, matrix_fitness,
)
from frictionlab.analysis.metrics import EfficiencyStats, SpreadDecomposition
from frictionlab.analysis.report_sections import cell, esc, inline_svg, table
from frictionlab.analysis.welfare import Equilibrium, WelfareReport


def parse(fragment: str) -> ET.Element:
    root = ET.fromstring(fragment)
    assert root.tag == "section"
    return root


def body_rows(root, cls=None):
    tables = [t for t in root.iter("table") if cls is None or t.get("class") == cls]
    return [tr for t in tables for tr in t.find("tbody").iter("tr")]


def raw_rows():
    rows = []
    for c, (fee, lat) in enumerate([(f, l) for f in (0, 5) for l in (0.0, 1.0)]):
        for k in range(3):
            r = {"cell": c, "replicate": k, "seed": 100 * c + k, "taker_bps": fee, "latency": lat}
            if c == 3 and k == 1:
                r["error"] = "ValueError: book <crossed> & 'bad'"
            else:
                r.update(net=float(10 - fee - lat + k), volume=20 + k, error=None)
            rows.append(r)
    return rows


def test_helpers_escape_and_format():
    assert esc('<a href="x">&\'') == "&lt;a href=&quot;x&quot;&gt;&amp;&#x27;"
    assert cell(3) == "3" and cell(np.int64(4)) == "4" and cell(None) == ""
    assert cell(0.125) == "0.125" and cell(-0.0) == "0" and cell(math.nan) == "nan"
    assert cell(True) == "True" and cell("<b>") == "&lt;b&gt;"
    t = ET.fromstring(table(("a", "b"), [(1, "<x>"), (2.5, None)]))
    assert len(t.find("tbody")) == 2 and t.find("tbody")[0][1].text == "<x>"
    svg = inline_svg('<svg xmlns="http://www.w3.org/2000/svg"><g/></svg>\n')
    assert "http" not in svg and svg.startswith('<figure class="chart"><svg>')


def test_metrics_section_rows():
    sd = SpreadDecomposition(5.0, 2.0, 0.5, 1.5, 40, 12, 1)
    eff = EfficiencyStats(0.75, 0.9, 1.02)
    w = WelfareReport(Equilibrium(6, 120, 98, 102), 108,
                      {1: PnLBreakdown(60, 3, 1, 4), 2: PnLBreakdown(48, 2, 0, 4)})
    sec = MetricsSection(spread=sd, efficiency=eff, welfare=w, extra={"note <1>": "ok & done"})
    root = parse(sec.render("m"))
    assert root.get("id") == "m" and root.find("h2").text == "Market quality"
    rows = body_rows(root)
    assert len(rows) == 4 + 3 + 5 + 1 == len(sec.rows())
    vals = {tr[0].text: tr[1].text for tr in rows}
    assert vals["Realized spread"] == "0.5" and vals["Price impact"] == "1.5"
    assert vals["CE maximum surplus"] == "120" and vals["Allocative efficiency"] == "0.9"
    assert vals["Net surplus after frictions"] == "102" and vals["note <1>"] == "ok & done"
    assert len(body_rows(parse(MetricsSection(spread=sd).render()))) == 4


def test_sweep_section_raw_rows_heatmap_and_errors():
    sec = SweepSection(raw_rows(), by=("taker_bps", "latency"), metric="net")
    root = parse(sec.render())
    assert root.find("h2").text == "Parameter sweep"
    agg = body_rows(root, "aggregate")
    assert len(agg) == 4
    first = [td.text for td in agg[0]]
    assert first[:4] == ["0", "0", "3", "0"] and first[4] == "11"  # mean of 10, 11, 12
    assert math.isclose(float(first[5]), np.std([10, 11, 12]), rel_tol=1e-5)
    assert [td.text for td in agg[3]][3] == "1"  # one error in the last cell
    svg = root.find("figure").find("svg")
    assert len([r for r in svg.iter("rect") if (r.get("class") or "").startswith("cell")]) == 4
    errs = body_rows(root, "errors")
    assert len(errs) == 1 and errs[0][-1].text == "ValueError: book <crossed> & 'bad'"
    assert [td.text for td in errs[0][:5]] == ["3", "1", "301", "5", "1"]
    assert "&lt;crossed&gt; &amp; &#x27;bad&#x27;" in sec.render()


def test_sweep_section_aggregates_line_chart_no_error_table():
    agg = [{"taker_bps": f, "n": 2, "errors": 0, "net_mean": 10.0 - f, "net_std": 0.5}
           for f in (10, 0, 5)]
    root = parse(SweepSection(agg, by=("taker_bps",), title="Fee sweep").render())
    assert root.find("h2").text == "Fee sweep"
    assert len(body_rows(root, "aggregate")) == 3 and not body_rows(root, "errors")
    (line,) = [p for p in root.iter("polyline") if p.get("class") == "series"]
    assert len(line.get("points").split()) == 3
    cats = [{"venue": v, "n": 1, "errors": 0, "net_mean": 1.0} for v in ("book", "batch")]
    bars = [r for r in parse(SweepSection(cats, by=("venue",)).render()).iter("rect")
            if r.get("class") == "bar"]
    assert len(bars) == 2


def test_equilibrium_section_marks_fixed_point():
    res = evolve([0.5, 0.5], matrix_fitness([[0, 3], [1, 2]]), eta=0.5,
                 generations=2000, labels=("hawk", "dove"))
    assert res.converged
    root = parse(EquilibriumSection(res).render())
    rows = body_rows(root)
    assert [tr[0].text for tr in rows] == ["hawk", "dove"]
    assert math.isclose(float(rows[0][3].text), 0.5, abs_tol=1e-3)
    (fp,) = [g for g in root.iter("g") if g.get("class") == "fixed-point"]
    assert len(list(fp.iter("circle"))) == 2
    assert "stable @ gen" in "".join(fp.itertext())
    assert len([p for p in root.iter("polyline") if p.get("class") == "series"]) == 2


def test_equilibrium_section_without_fixed_point():
    res = evolve([0.2, 0.8], matrix_fitness([[1, 0], [0, 1]]), eta=0.01,
                 generations=3, labels=("a", "b"))
    assert not res.converged
    root = parse(EquilibriumSection(res).render())
    assert not [g for g in root.iter("g") if g.get("class") == "fixed-point"]
    assert "No fixed point" in root.find("p").text
    assert body_rows(root)[0][3].text is None


@pytest.mark.parametrize("detailed", [False, True])
def test_pnl_section_bars_and_rows(detailed):
    if detailed:
        vals = {"ZI<C>": PnLBreakdown(50, 5, 2, 10), "informed": PnLBreakdown(-20, 1, 0, 3)}
    else:
        vals = {"ZI<C>": 43, "informed": -21, "noise": 0}
    root = parse(PnLSection(vals).render())
    assert root.find("h2").text == "PnL by agent type"
    assert len(body_rows(root)) == len(vals)
    bars = [r for r in root.iter("rect") if r.get("class") == "bar"]
    assert [b.get("data-label") for b in bars] == list(vals)
    assert body_rows(root)[0][0].text == "ZI<C>"
    if detailed:
        assert body_rows(root)[0][4].text == "43"
