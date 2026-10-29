"""Typed sections for the offline HTML report (`report.Report`).

Each section's `render()` is pure and returns a `<section>` fragment (h2
title, tables, inlined fl-025 SVG). Strings go through `esc`, numbers
through `cell`, so identical inputs give byte-identical markup.

- `MetricsSection`: fl-022 market quality (spread decomposition, price
  efficiency, welfare vs the competitive-equilibrium maximum, extras).
- `SweepSection`: fl-023 raw rows or `aggregate` output -> mean/std table,
  a line chart (1 swept key) or heatmap (2 keys), and an error-row table.
- `EquilibriumSection`: fl-024 `EquilibriumResult` -> share chart with the
  detected fixed point marked, plus initial / final / fitness table.
- `PnLSection`: PnL by agent type (ints or `PnLBreakdown`) -> bar chart.
"""
from __future__ import annotations

import html
import math
import numbers
from typing import Any, Mapping, Optional, Sequence

from .plots import _a, _frame, bar_chart, heatmap, line_chart, multi_line_chart
from .sweep import RESERVED, aggregate

SVG_NS = ' xmlns="http://www.w3.org/2000/svg"'


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool) or not isinstance(v, numbers.Number):
        return esc(v)
    if isinstance(v, numbers.Integral):
        return str(int(v))
    f = float(v)
    return "nan" if math.isnan(f) else ("0" if f == 0 else f"{f:.6g}")


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], cls: str = "data") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in r) + "</tr>" for r in rows)
    return f'<table class="{esc(cls)}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def inline_svg(svg: str) -> str:
    """Inline an fl-025 SVG document; HTML5 needs no namespace URL."""
    return '<figure class="chart">' + svg.replace(SVG_NS, "", 1).strip() + "</figure>"


class Section:
    kind = "section"

    def __init__(self, title: str) -> None:
        self.title = title

    def body(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def render(self, anchor: str = "") -> str:
        ident = f' id="{esc(anchor)}"' if anchor else ""
        return (f'<section class="{self.kind}"{ident}><h2>{esc(self.title)}</h2>'
                f"{self.body()}</section>")


class MetricsSection(Section):
    kind = "metrics"

    def __init__(self, spread=None, efficiency=None, welfare=None,
                 extra: Optional[Mapping[str, Any]] = None, title: str = "Market quality") -> None:
        super().__init__(title)
        self.spread, self.efficiency, self.welfare = spread, efficiency, welfare
        self.extra = dict(extra or {})

    def rows(self) -> list:
        out = []
        if (s := self.spread) is not None:
            out += [(f"Effective spread (tau={cell(s.tau)})", s.effective, "ticks"),
                    ("Realized spread", s.realized, "ticks"),
                    ("Price impact", s.impact, "ticks"),
                    ("Trades decomposed", s.n_trades, "count")]
        if (e := self.efficiency) is not None:
            out += [("Price-efficiency error |mid - V|", e.mad, "ticks"),
                    ("Mid coverage", e.coverage, "share"),
                    ("Variance ratio", e.variance_ratio, "ratio")]
        if (w := self.welfare) is not None:
            out += [("Realized gross gains", w.gross_gains, "ticks"),
                    ("CE maximum surplus", w.equilibrium.max_surplus, "ticks"),
                    ("Allocative efficiency", w.efficiency, "share of CE max"),
                    ("Net surplus after frictions", w.net_surplus, "ticks"),
                    ("Net efficiency", w.net_efficiency, "share of CE max")]
        out += [(k, v, "") for k, v in self.extra.items()]
        return out

    def body(self) -> str:
        return table(("Metric", "Value", "Unit"), self.rows())


class SweepSection(Section):
    kind = "sweep"

    def __init__(self, rows: Sequence[Mapping[str, Any]], by: Sequence[str],
                 metric: Optional[str] = None, title: str = "Parameter sweep") -> None:
        super().__init__(title)
        self.by = tuple(by)
        raw = [dict(r) for r in rows]
        self.is_raw = any("replicate" in r for r in raw)
        self.errors = [r for r in raw if self.is_raw and r.get("error") is not None]
        self.agg = aggregate(raw, self.by) if self.is_raw else raw
        self.metrics = list(dict.fromkeys(k[:-5] for r in self.agg for k in r if k.endswith("_mean")))
        self.metric = metric if metric is not None else next(iter(self.metrics), None)

    def _chart(self) -> str:
        m, by = self.metric, self.by
        if m is None or not self.agg or len(by) not in (1, 2):
            return ""
        key = f"{m}_mean"
        if len(by) == 1:
            pts = [(r.get(by[0]), r.get(key, math.nan)) for r in self.agg]
            if all(isinstance(x, numbers.Real) and not isinstance(x, bool) for x, _ in pts):
                pts.sort(key=lambda p: p[0])
                return inline_svg(line_chart([y for _, y in pts], [x for x, _ in pts],
                                             title=f"{m} vs {by[0]}", xlabel=by[0], ylabel=m, name=m))
            return inline_svg(bar_chart({str(x): y for x, y in pts}, title=f"{m} by {by[0]}", ylabel=m))
        rv = list(dict.fromkeys(r.get(by[0]) for r in self.agg))
        cv = list(dict.fromkeys(r.get(by[1]) for r in self.agg))
        grid = [[math.nan] * len(cv) for _ in rv]
        for r in self.agg:
            grid[rv.index(r.get(by[0]))][cv.index(r.get(by[1]))] = float(r.get(key, math.nan))
        return inline_svg(heatmap(grid, [cell(v) for v in rv], [cell(v) for v in cv],
                                  title=f"{m}: {by[0]} x {by[1]}", xlabel=by[1], ylabel=by[0]))

    def error_table(self) -> str:
        params = [k for k in self.by if k not in RESERVED]
        rows = [[r.get("cell"), r.get("replicate"), r.get("seed"), *[r.get(k) for k in params],
                 r.get("error")] for r in self.errors]
        hdr = ("cell", "replicate", "seed", *params, "error")
        return f"<h3>Error rows ({len(rows)})</h3>" + table(hdr, rows, "errors")

    def body(self) -> str:
        hdr = [*self.by, "n", "errors"]
        for m in self.metrics:
            hdr += [f"{m} mean", f"{m} std"]
        rows = [[r.get(k) for k in self.by] + [r.get("n"), r.get("errors")]
                + [r.get(f"{m}_{s}") for m in self.metrics for s in ("mean", "std")]
                for r in self.agg]
        parts = [table(hdr, rows, "aggregate"), self._chart()]
        if self.is_raw:
            parts.append(self.error_table())
        return "".join(parts)


class EquilibriumSection(Section):
    kind = "equilibrium"

    def __init__(self, result, title: str = "Strategy-mix equilibrium") -> None:
        super().__init__(title)
        self.result = result

    def _chart(self) -> str:
        r = self.result
        shares = [[float(v) for v in col] for col in r.shares.T]
        svg = multi_line_chart(dict(zip(r.labels, shares)), title="Population shares",
                               xlabel="generation", ylabel="share")
        if r.fixed_point is None or r.generation is None:
            return inline_svg(svg)
        sx, sy = _frame([list(map(float, range(len(s)))) for s in shares], shares)
        g = float(len(r.shares) - 1)
        mark = ['<g class="fixed-point">',
                f'<line {_a(x1=sx(g), y1=sy.r0, x2=sx(g), y2=sy.r1, stroke="#555555", stroke_dasharray="4 3")}/>']
        mark += [f'<circle {_a(cx=sx(g), cy=sy(float(v)), r=4, fill="none", stroke="#000000")}/>'
                 for v in r.fixed_point]
        mark.append(f'<text {_a(x=sx(g) - 4, y=sy.r1 + 12, text_anchor="end")}>'
                    f"{esc(r.reason)} @ gen {int(r.generation)}</text></g>")
        return inline_svg(svg.replace("</svg>", "\n".join(mark) + "\n</svg>"))

    def body(self) -> str:
        r, fp = self.result, self.result.fixed_point
        last = r.fitness[-1] if len(r.fitness) else [math.nan] * len(r.labels)
        rows = [[lab, float(r.shares[0][i]), float(r.shares[-1][i]),
                 float(fp[i]) if fp is not None else None, float(last[i])]
                for i, lab in enumerate(r.labels)]
        note = (f"Detector fired ({esc(r.reason)}) at generation {int(r.generation)}."
                if fp is not None else "No fixed point detected within the generation budget.")
        return (f'<p class="fixed-point">{note}</p>' + self._chart()
                + table(("archetype", "initial share", "final share", "fixed point",
                         "last fitness"), rows))


class PnLSection(Section):
    kind = "pnl"

    def __init__(self, values: Mapping[str, Any], title: str = "PnL by agent type") -> None:
        super().__init__(title)
        self.values = dict(values)

    def body(self) -> str:
        detailed = all(hasattr(v, "net") for v in self.values.values()) and self.values
        if detailed:
            hdr = ("agent type", "trading", "fees", "decision costs", "net", "volume")
            rows = [[k, v.trading, v.fees, v.decision_costs, v.net, v.volume]
                    for k, v in self.values.items()]
            nets = {str(k): float(v.net) for k, v in self.values.items()}
        else:
            hdr, rows = ("agent type", "net PnL"), [[k, v] for k, v in self.values.items()]
            nets = {str(k): float(v) for k, v in self.values.items()}
        chart = inline_svg(bar_chart(nets, title="Net PnL by agent type", ylabel="ticks"))
        return chart + table(hdr, rows)
