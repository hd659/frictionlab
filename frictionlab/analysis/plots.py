"""Dependency-free SVG charts for analysis output (no matplotlib, no network).

Every function is pure and returns an SVG document string:

- `line_chart(ys, xs=None)`: one time series, e.g. mid vs fundamental gap,
  a funding-rate path or one replicator share trajectory.
- `multi_line_chart({name: ys}, xs=None)`: several series plus a legend,
  e.g. mid and fundamental, or every archetype share from `evolve`.
- `bar_chart({label: value})`: PnL by agent type or one metric per sweep
  cell; negative bars hang below the zero baseline.
- `heatmap(grid, row_labels, col_labels)`: a two-parameter sweep surface,
  e.g. taker fee x latency; NaN cells are drawn as hatched no-data cells.

All coordinates go through `scales.fmt`, so output is byte-identical for the
same input. NaN/inf points in line series are skipped (not plotted).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence
from xml.sax.saxutils import escape

from .scales import LinearScale, extent, fmt, format_tick, ramp

W, H = 640, 360
M = {"l": 60, "r": 20, "t": 30, "b": 40}
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf")
NODATA = "#cccccc"


def _a(**kw) -> str:
    out = []
    for k, v in kw.items():
        val = fmt(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
        out.append(f'{k.rstrip("_").replace("_", "-")}="{escape(val, {chr(34): "&quot;"})}"')
    return " ".join(out)


def _doc(body: list[str], title: str, w: int = W, h: int = H, defs: str = "") -> str:
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" {_a(width=w, height=h)} '
            f'viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="11">')
    parts = [head, f"<title>{escape(title)}</title>"]
    if defs:
        parts.append(f"<defs>{defs}</defs>")
    parts.append(f'<rect {_a(x=0, y=0, width=w, height=h, fill="#ffffff")}/>')
    if title:
        parts.append(f'<text {_a(x=w / 2, y=18, text_anchor="middle", font_size=13)}>'
                     f"{escape(title)}</text>")
    return "\n".join(parts + body + ["</svg>"]) + "\n"


def _axes(sx: LinearScale, sy: LinearScale, xlabel: str, ylabel: str) -> list[str]:
    x0, x1, y0, y1 = sx.r0, sx.r1, sy.r0, sy.r1
    out = [f'<g class="axes" stroke="#333333" fill="none">',
           f'<line {_a(x1=x0, y1=y0, x2=x1, y2=y0)}/>',
           f'<line {_a(x1=x0, y1=y0, x2=x0, y2=y1)}/>', "</g>"]
    for scale, horiz in ((sx, True), (sy, False)):
        ticks = scale.ticks()
        step = ticks[1] - ticks[0] if len(ticks) > 1 else None
        for t in ticks:
            p = scale(t)
            lab = escape(format_tick(t, step))
            if horiz:
                out.append(f'<text {_a(x=p, y=y0 + 15, text_anchor="middle")}>{lab}</text>')
            else:
                out.append(f'<line {_a(x1=x0, y1=p, x2=x1, y2=p, stroke="#eeeeee")}/>')
                out.append(f'<text {_a(x=x0 - 5, y=p + 4, text_anchor="end")}>{lab}</text>')
    if xlabel:
        out.append(f'<text {_a(x=(x0 + x1) / 2, y=H - 6, text_anchor="middle")}>'
                   f"{escape(xlabel)}</text>")
    if ylabel:
        out.append(f'<text {_a(x=14, y=(y0 + y1) / 2, text_anchor="middle")} '
                   f'transform="rotate(-90 14 {fmt((y0 + y1) / 2)})">{escape(ylabel)}</text>')
    return out


def _frame(xs_all, ys_all) -> tuple[LinearScale, LinearScale]:
    xlo, xhi = extent(*xs_all)
    ylo, yhi = extent(*ys_all)
    sx = LinearScale.nice(xlo, xhi, M["l"], W - M["r"])
    sy = LinearScale.nice(ylo, yhi, H - M["b"], M["t"])
    return sx, sy


def _polyline(xs, ys, sx, sy, color: str, name: str) -> str:
    pts = [f"{fmt(sx(x))},{fmt(sy(y))}" for x, y in zip(xs, ys)
           if math.isfinite(float(x)) and math.isfinite(float(y))]
    return (f'<polyline {_a(class_="series", data_name=name, fill="none", stroke=color, stroke_width=1.5)}'
            f' points="{" ".join(pts)}"/>')


def multi_line_chart(series: Mapping[str, Sequence[float]], xs: Sequence[float] | None = None,
                     title: str = "", xlabel: str = "", ylabel: str = "") -> str:
    names = list(series)
    ys_list = [[float(v) for v in series[n]] for n in names]
    xs_list = [[float(v) for v in xs] if xs is not None else [float(i) for i in range(len(y))]
               for y in ys_list]
    for xv, yv in zip(xs_list, ys_list):
        if len(xv) != len(yv):
            raise ValueError("xs and series must have equal length")
    sx, sy = _frame(xs_list, ys_list)
    body = _axes(sx, sy, xlabel, ylabel)
    for i, n in enumerate(names):
        body.append(_polyline(xs_list[i], ys_list[i], sx, sy, PALETTE[i % len(PALETTE)], n))
    if len(names) > 1:
        body.append('<g class="legend">')
        for i, n in enumerate(names):
            y = M["t"] + 4 + 14 * i
            c = PALETTE[i % len(PALETTE)]
            body.append(f'<rect {_a(x=W - M["r"] - 110, y=y, width=10, height=10, fill=c)}/>')
            body.append(f'<text {_a(x=W - M["r"] - 95, y=y + 9)}>{escape(n)}</text>')
        body.append("</g>")
    return _doc(body, title)


def line_chart(ys: Sequence[float], xs: Sequence[float] | None = None, title: str = "",
               xlabel: str = "", ylabel: str = "", name: str = "y") -> str:
    return multi_line_chart({name: ys}, xs, title, xlabel, ylabel)


def bar_chart(values: Mapping[str, float], title: str = "", ylabel: str = "") -> str:
    labels = list(values)
    vals = [float(values[k]) for k in labels]
    finite = [v for v in vals if math.isfinite(v)]
    lo, hi = extent(finite + [0.0])
    sy = LinearScale.nice(lo, hi, H - M["b"], M["t"])
    sx = LinearScale(0, max(len(labels), 1), M["l"], W - M["r"])
    base = sy(0.0)
    yt = sy.ticks()
    step = yt[1] - yt[0] if len(yt) > 1 else None
    body = [f'<line {_a(class_="baseline", x1=sx.r0, y1=base, x2=sx.r1, y2=base, stroke="#333333")}/>']
    for t in yt:
        p = sy(t)
        body.append(f'<text {_a(x=M["l"] - 5, y=p + 4, text_anchor="end")}>'
                    f"{escape(format_tick(t, step))}</text>")
    slot = (sx.r1 - sx.r0) / max(len(labels), 1)
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = sx(i) + slot * 0.15
        v = v if math.isfinite(v) else 0.0
        top, h = (sy(v), base - sy(v)) if v >= 0 else (base, sy(v) - base)
        c = PALETTE[0] if v >= 0 else PALETTE[1]
        body.append(f'<rect {_a(class_="bar", data_label=lab, x=x, y=top, width=slot * 0.7, height=h, fill=c)}/>')
        body.append(f'<text {_a(x=x + slot * 0.35, y=H - M["b"] + 15, text_anchor="middle")}>'
                    f"{escape(lab)}</text>")
    if ylabel:
        body.append(f'<text {_a(x=14, y=H / 2, text_anchor="middle")} '
                    f'transform="rotate(-90 14 {fmt(H / 2)})">{escape(ylabel)}</text>')
    return _doc(body, title)


def heatmap(grid: Sequence[Sequence[float]], row_labels: Sequence[str] | None = None,
            col_labels: Sequence[str] | None = None, title: str = "",
            xlabel: str = "", ylabel: str = "") -> str:
    rows = [[float(v) for v in r] for r in grid]
    nr, nc = len(rows), max((len(r) for r in rows), default=0)
    if any(len(r) != nc for r in rows):
        raise ValueError("heatmap grid must be rectangular")
    rl = list(row_labels) if row_labels is not None else [str(i) for i in range(nr)]
    cl = list(col_labels) if col_labels is not None else [str(j) for j in range(nc)]
    lo, hi = extent(*rows)
    span = hi - lo
    cw = (W - M["l"] - M["r"]) / max(nc, 1)
    ch = (H - M["t"] - M["b"]) / max(nr, 1)
    defs = ('<pattern id="nodata" patternUnits="userSpaceOnUse" width="6" height="6">'
            f'<rect width="6" height="6" fill="{NODATA}"/>'
            '<path d="M0,6 L6,0" stroke="#888888" stroke-width="1"/></pattern>')
    body = []
    for i, r in enumerate(rows):
        y = M["t"] + i * ch
        body.append(f'<text {_a(x=M["l"] - 5, y=y + ch / 2 + 4, text_anchor="end")}>{escape(str(rl[i]))}</text>')
        for j, v in enumerate(r):
            x = M["l"] + j * cw
            if math.isfinite(v):
                fill, cls, val = ramp((v - lo) / span), "cell", fmt(v, 6)
            else:
                fill, cls, val = "url(#nodata)", "cell nodata", "nan"
            body.append(f'<rect {_a(class_=cls, data_value=val, x=x, y=y, width=cw, height=ch, fill=fill)}/>')
    for j, lab in enumerate(cl):
        body.append(f'<text {_a(x=M["l"] + (j + 0.5) * cw, y=H - M["b"] + 15, text_anchor="middle")}>'
                    f"{escape(str(lab))}</text>")
    if xlabel:
        body.append(f'<text {_a(x=W / 2, y=H - 6, text_anchor="middle")}>{escape(xlabel)}</text>')
    if ylabel:
        body.append(f'<text {_a(x=14, y=H / 2, text_anchor="middle")} '
                    f'transform="rotate(-90 14 {fmt(H / 2)})">{escape(ylabel)}</text>')
    return _doc(body, title, defs=defs)
