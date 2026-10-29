import math
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from frictionlab.analysis import bar_chart, heatmap, line_chart, multi_line_chart
from frictionlab.analysis.scales import luminance

NS = "{http://www.w3.org/2000/svg}"
NAN = float("nan")


def parse(svg: str) -> ET.Element:
    root = ET.fromstring(svg)
    assert root.tag == NS + "svg"
    return root


def polylines(root):
    return [p for p in root.iter(NS + "polyline") if p.get("class") == "series"]


def bars(root):
    return [r for r in root.iter(NS + "rect") if r.get("class") == "bar"]


def cells(root):
    return [r for r in root.iter(NS + "rect") if (r.get("class") or "").startswith("cell")]


def baseline_y(root) -> float:
    (line,) = [l for l in root.iter(NS + "line") if l.get("class") == "baseline"]
    return float(line.get("y1"))


def test_line_chart_wellformed_and_point_count():
    rng = np.random.default_rng(7)
    ys = np.cumsum(rng.normal(size=137)) + 100.0
    root = parse(line_chart(ys, title="mid - V <&>", xlabel="t", ylabel="ticks"))
    (pl,) = polylines(root)
    assert len(pl.get("points").split()) == 137
    assert root.find(NS + "title").text == "mid - V <&>"


def test_multi_line_chart_legend_and_counts():
    xs = list(range(0, 50, 2))
    series = {"mid": [100 + 0.1 * x for x in xs], "fundamental": [100.0] * len(xs),
              "share": [x / 50 for x in xs]}
    root = parse(multi_line_chart(series, xs=xs))
    pls = polylines(root)
    assert [p.get("data-name") for p in pls] == list(series)
    assert all(len(p.get("points").split()) == len(xs) for p in pls)
    legend = [g for g in root.iter(NS + "g") if g.get("class") == "legend"]
    assert len(legend) == 1 and len(list(legend[0].iter(NS + "text"))) == 3


def test_line_chart_length_mismatch_raises():
    with pytest.raises(ValueError):
        line_chart([1, 2, 3], xs=[0, 1])


@pytest.mark.parametrize("ys", [[], [5.0], [3.0] * 10, [0.0] * 4, [NAN, NAN]])
def test_empty_or_constant_series_does_not_raise(ys):
    root = parse(line_chart(ys))
    (pl,) = polylines(root)
    finite = [y for y in ys if math.isfinite(y)]
    assert len(pl.get("points").split()) == len(finite)


def test_nan_points_skipped_in_line():
    (pl,) = polylines(parse(line_chart([1.0, NAN, 3.0, 4.0])))
    assert len(pl.get("points").split()) == 3


def test_bar_heights_proportional_and_negative_below_baseline():
    vals = {"mm": 40.0, "informed": 20.0, "noise": -30.0, "zi": 10.0}
    root = parse(bar_chart(vals, title="PnL by type", ylabel="ticks"))
    rs = bars(root)
    assert [r.get("data-label") for r in rs] == list(vals)
    base = baseline_y(root)
    heights = {r.get("data-label"): float(r.get("height")) for r in rs}
    for k, v in vals.items():
        assert heights[k] / heights["mm"] == pytest.approx(abs(v) / 40.0, abs=0.01)
    for r, v in zip(rs, vals.values()):
        y, h = float(r.get("y")), float(r.get("height"))
        if v >= 0:
            assert y + h == pytest.approx(base, abs=0.01)
        else:
            assert y == pytest.approx(base, abs=0.01) and y + h > base


def test_bar_chart_edge_cases():
    root = parse(bar_chart({}))
    assert bars(root) == []
    rs = bars(parse(bar_chart({"a": 0.0, "b": 0.0})))
    assert all(float(r.get("height")) == 0.0 for r in rs)


def test_heatmap_cell_count_and_monotone_ramp():
    fees = [0, 5, 10, 20]
    lat = [0, 1, 2]
    grid = [[f * 0.5 + l * 3.0 for l in lat] for f in fees]
    root = parse(heatmap(grid, [f"fee={f}" for f in fees], [f"lat={l}" for l in lat],
                         xlabel="latency", ylabel="taker fee"))
    cs = cells(root)
    assert len(cs) == len(fees) * len(lat)
    pairs = sorted((float(c.get("data-value")), luminance(c.get("fill"))) for c in cs)
    for (v0, l0), (v1, l1) in zip(pairs, pairs[1:]):
        if v1 > v0:
            assert l1 < l0


def test_heatmap_nan_cell_hatched_and_constant_grid():
    root = parse(heatmap([[1.0, NAN], [2.0, 3.0]]))
    cs = cells(root)
    assert len(cs) == 4
    nod = [c for c in cs if "nodata" in c.get("class")]
    assert len(nod) == 1 and nod[0].get("fill") == "url(#nodata)"
    assert any(p.get("id") == "nodata" for p in root.iter(NS + "pattern"))
    assert len(cells(parse(heatmap([[2.0, 2.0], [2.0, 2.0]])))) == 4
    assert cells(parse(heatmap([]))) == []
    with pytest.raises(ValueError):
        heatmap([[1.0, 2.0], [3.0]])


def test_byte_identical_output():
    rng = np.random.default_rng(42)
    ys = rng.normal(size=60).tolist()
    grid = rng.normal(size=(3, 5)).tolist()
    grid[1][2] = NAN
    for make in (lambda: line_chart(ys), lambda: multi_line_chart({"a": ys, "b": ys[::-1]}),
                 lambda: bar_chart({"x": 1.5, "y": -0.25}), lambda: heatmap(grid)):
        assert make() == make()
    assert line_chart(ys) == line_chart(np.array(ys))
    assert not re.search(r"\de[-+]?\d", line_chart([1e-9, 2e-9]))
    assert not re.search(r"[\s\",]-0[\s\",]", line_chart(ys))
