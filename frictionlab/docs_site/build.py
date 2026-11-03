"""Build the static docs site from `docs/index.md`.

`docs/index.md` fixes the page order: the site is the index itself followed by every
`docs/*.md` page it links, in first-link order (duplicates and `#fragments` folded).
Each page is rendered into `OUT/<name>.html` with the shared layout: a sidebar nav
in index order, prev/next links and an h2/h3 table of contents.

Links are resolved against the source file. A relative link to a site page becomes
`<name>.html`; links to `.md` files outside the site (README, ROADMAP) and every
link with a scheme are left exactly as written, so nothing points at a missing page.

Output is byte-reproducible: pages are written in sorted order with `\\n` newlines,
UTF-8 and no timestamps, so two builds of the same tree are identical.

    res = build_site("docs", "site")    # -> BuildResult(out, pages)
"""
from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

from . import layout
from .markdown import SCHEME, render

PathLike = Union[str, Path]
INDEX = "index.md"


class DocsError(Exception):
    """A usage-level problem with the docs tree (missing index or linked page)."""

    def __init__(self, path: PathLike, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path, self.message = str(path), message


@dataclass
class BuildResult:
    out: Path
    pages: List[Path] = field(default_factory=list)     # in nav order

    def names(self) -> List[str]:
        return [p.name for p in self.pages]


def _site_target(url: str) -> Union[str, None]:
    """`phase1.md#x` -> `phase1.md` when the link points at a sibling .md, else None."""
    if not url or url.startswith(("#", "/")) or SCHEME.match(url):
        return None
    path = posixpath.normpath(url.partition("#")[0])
    if not path.endswith(".md") or "/" in path or path.startswith("."):
        return None
    return path


def html_name(md_name: str) -> str:
    return md_name[:-3] + ".html"


def page_order(src: Path) -> List[str]:
    """[index.md, *linked sibling pages in first-link order]. Raises DocsError."""
    index = src / INDEX
    if not index.is_file():
        raise DocsError(index, "docs index not found")
    order: List[str] = [INDEX]

    def collect(url: str) -> str:
        name = _site_target(url)
        if name and name not in order:
            order.append(name)
        return url
    render(index.read_text(encoding="utf-8"), rewrite=collect)
    missing = [n for n in order if not (src / n).is_file()]
    if missing:
        raise DocsError(src / missing[0], "linked from docs/index.md but not found")
    return order


def _rewriter(pages: List[str]):
    known = set(pages)

    def rewrite(url: str) -> str:
        name = _site_target(url)
        if name is None or name not in known:
            return url
        _, sep, frag = url.partition("#")
        return html_name(name) + sep + frag
    return rewrite


def render_site(src: PathLike) -> Dict[str, str]:
    """{output file name: full HTML}. Pure: reads `src`, writes nothing."""
    src = Path(src)
    order = page_order(src)
    rewrite = _rewriter(order)
    docs: List[Tuple[str, object]] = []
    for name in order:
        docs.append((name, render((src / name).read_text(encoding="utf-8"), rewrite=rewrite)))
    nav = [layout.NavItem(html_name(n), "Overview" if n == INDEX else (d.title or n[:-3]))
           for n, d in docs]
    out: Dict[str, str] = {}
    for (name, d), item in zip(docs, nav):
        out[item.href] = layout.page(d.title or item.label, d.html, nav, item.href, d.toc())
    return out


def build_site(src: PathLike = "docs", out: PathLike = "site") -> BuildResult:
    """Render every page and write it under `out`. Returns the written paths in nav order."""
    pages = render_site(src)
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    for name in sorted(pages):
        with open(dest / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(pages[name])
    return BuildResult(out=dest, pages=[dest / n for n in pages])
