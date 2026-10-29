"""Self-contained, offline HTML report assembled from typed analysis sections.

    rep = Report("Taker fee x latency", master_seed=spec.master_seed, grid=spec.grid)
    rep.add(MetricsSection(spread=sd, efficiency=eff, welfare=w))
    rep.add(SweepSection(rows, by=("taker_bps", "latency"), metric="net_surplus"))
    rep.add(EquilibriumSection(evolve(...)))
    rep.add(PnLSection(pnl_by_agent_type(result, mark)))
    rep.write_report("report.html")

The output is one HTML5 file: CSS is embedded, charts are inlined fl-025
SVG, and there is no JavaScript, no external asset and no URL of any kind,
so it opens offline and can be archived next to the CSV it came from. The
header records the master seed, parameter grid and frictionlab version for
reproducibility. No timestamp is written: identical inputs give
byte-identical HTML, which makes reports diffable and testable. Apart from
the DOCTYPE line the markup is also well-formed XML (void tags self-closed).
"""
from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union

from .report_sections import Section, cell, esc, table

CSS = """
body{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;margin:2rem auto;
max-width:980px;color:#222;line-height:1.4;padding:0 1rem}
h1{font-size:1.6rem;margin-bottom:.2rem}h2{border-bottom:1px solid #ddd;padding-bottom:.2rem}
header{border-bottom:2px solid #333;margin-bottom:1.5rem;padding-bottom:.8rem}
table{border-collapse:collapse;margin:.6rem 0 1rem;font-size:.85rem}
th,td{border:1px solid #ccc;padding:.25rem .55rem;text-align:right}
th{background:#f3f5f8}td:first-child,th:first-child{text-align:left}
table.errors td{background:#fff4f4}table.errors td:last-child{text-align:left}
figure.chart{margin:.5rem 0}figure.chart svg{max-width:100%;height:auto}
nav ol{columns:2;font-size:.9rem}p.fixed-point{font-style:italic}
footer{margin-top:2rem;color:#777;font-size:.8rem}
""".strip()


def frictionlab_version() -> str:
    """Installed distribution version, else the source tree's pyproject.toml."""
    try:
        return importlib.metadata.version("frictionlab")
    except importlib.metadata.PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject.is_file():
            m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
            if m:
                return m.group(1)
        return "unknown"


def _grid_rows(grid: Union[Mapping[str, Sequence[Any]], Sequence[Mapping[str, Any]], None]) -> list:
    if grid is None:
        return []
    if isinstance(grid, Mapping):
        return [[k, ", ".join(cell(v) for v in vals), len(vals)] for k, vals in grid.items()]
    return [[f"cell {i}", ", ".join(f"{k}={cell(v)}" for k, v in c.items()), 1]
            for i, c in enumerate(grid)]


def _slug(text: str, used: set) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"
    slug, i = base, 2
    while slug in used:
        slug, i = f"{base}-{i}", i + 1
    used.add(slug)
    return slug


class Report:
    """Ordered collection of `Section`s rendered into one offline HTML page."""

    def __init__(self, title: str = "frictionlab report", *, master_seed: Optional[int] = None,
                 grid=None, version: Optional[str] = None,
                 meta: Optional[Mapping[str, Any]] = None,
                 sections: Iterable[Section] = ()) -> None:
        self.title = title
        self.master_seed = master_seed
        self.grid = grid
        self.version = version if version is not None else frictionlab_version()
        self.meta = dict(meta or {})
        self.sections: List[Section] = []
        for s in sections:
            self.add(s)

    def add(self, section: Section) -> "Report":
        if not isinstance(section, Section):
            raise TypeError(f"expected a Section, got {type(section).__name__}")
        self.sections.append(section)
        return self

    def header(self) -> str:
        info = [("frictionlab version", self.version),
                ("master seed", "not recorded" if self.master_seed is None else self.master_seed)]
        info += list(self.meta.items())
        parts = [f'<header><h1>{esc(self.title)}</h1>',
                 table(("field", "value"), info, "provenance")]
        if self.grid is not None:
            parts.append('<h3>Parameter grid</h3>'
                         + table(("parameter", "values", "count"), _grid_rows(self.grid), "grid"))
        parts.append("</header>")
        return "".join(parts)

    def to_html(self) -> str:
        used: set = set()
        anchors = [_slug(s.title, used) for s in self.sections]
        nav = ""
        if len(self.sections) > 1:
            items = "".join(f'<li><a href="#{a}">{esc(s.title)}</a></li>'
                            for a, s in zip(anchors, self.sections))
            nav = f"<nav><ol>{items}</ol></nav>"
        body = "\n".join(s.render(a) for a, s in zip(anchors, self.sections))
        return ("<!DOCTYPE html>\n"
                '<html lang="en"><head><meta charset="utf-8"/>'
                f"<title>{esc(self.title)}</title><style>{CSS}</style></head>\n<body>\n"
                f"{self.header()}\n{nav}\n<main>\n{body}\n</main>\n"
                f"<footer>Generated offline by frictionlab {esc(self.version)}.</footer>\n"
                "</body></html>\n")

    def write_report(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_html(), encoding="utf-8", newline="\n")
        return p


def write_report(path: Union[str, Path], *sections: Section, **kwargs) -> Path:
    """One-shot helper: `write_report("out.html", s1, s2, master_seed=7)`."""
    return Report(sections=sections, **kwargs).write_report(path)
