"""Build the static docs site from `docs/index.md`, plus the generated API reference.

`docs/index.md` fixes the page order: the site is the index itself followed by every
`docs/*.md` page it links, in first-link order (duplicates and `#fragments` folded).
Each page is rendered into `OUT/<name>.html` with the shared layout: a sidebar nav
in index order, prev/next links and an h2/h3 table of contents.

With `api=True` (the default) the nav gains an API reference section after the phase
write-ups: `OUT/api/index.html` lists every package in `api_ref.PACKAGES` with its
symbol count, and `OUT/api/<package>.html` documents each public symbol (see
`api_ref`). Nav links on API pages are rewritten relative to `api/`.

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
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from . import api_ref, layout
from .markdown import SCHEME, render

PathLike = Union[str, Path]
INDEX = "index.md"
API_DIR = "api"
Entry = Tuple[str, str, List]       # (title, body html, headings) of one page


class DocsError(Exception):
    """A usage-level problem with the docs tree (missing index or linked page)."""

    def __init__(self, path: PathLike, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path, self.message = str(path), message


@dataclass
class BuildResult:
    """Where a build wrote its pages: `out` and every page path, in nav order."""
    out: Path
    pages: List[Path] = field(default_factory=list)     # in nav order

    def names(self) -> List[str]:
        """Page paths relative to `out`, e.g. `index.html`, `api/core.html`."""
        return [p.relative_to(self.out).as_posix() for p in self.pages]


def _site_target(url: str) -> Union[str, None]:
    """`phase1.md#x` -> `phase1.md` when the link points at a sibling .md, else None."""
    if not url or url.startswith(("#", "/")) or SCHEME.match(url):
        return None
    path = posixpath.normpath(url.partition("#")[0])
    if not path.endswith(".md") or "/" in path or path.startswith("."):
        return None
    return path


def html_name(md_name: str) -> str:
    """`phase1.md` -> `phase1.html`."""
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


def render_api(packages: Sequence[str] = api_ref.PACKAGES) -> Dict[str, Tuple[str, Entry]]:
    """{`api/<page>`: (nav label, page entry)}: the index first, then one page per package."""
    tables = {api_ref.module_of(p).__name__: api_ref.collect(p) for p in packages}
    idx = api_ref.xref_index(tables)
    pages: Dict[str, Tuple[str, Entry]] = {}
    rows = []
    for pkg, syms in tables.items():
        page = api_ref.page_name(pkg)
        title, body, heads = api_ref.render_package(pkg, syms, idx)
        pages[f"{API_DIR}/{page}"] = (pkg, (title, body, heads))
        doc = api_ref.inline(api_ref.summary(api_ref.own_doc(api_ref.module_of(pkg))), idx)
        rows.append(f'<tr><td><a href="{escape(page, quote=True)}"><code>{escape(pkg)}'
                    f"</code></a></td><td>{len(syms)}</td><td>{doc}</td></tr>")
    body = ('<h1 id="api-reference">API reference</h1>\n'
            "<p>Every public class, function and constant, per package, with normalized "
            "signatures and docstrings. Generated from the source by "
            "<code>frictionlab docs build</code>.</p>\n<table>\n<thead>\n<tr><th>Package</th>"
            "<th>Symbols</th><th>Summary</th></tr>\n</thead>\n<tbody>\n"
            + "\n".join(rows) + "\n</tbody>\n</table>\n")
    return {f"{API_DIR}/index.html": ("API reference", ("API reference", body, [])), **pages}


def _relative(nav: List[layout.NavItem], here: str) -> List[layout.NavItem]:
    """Nav hrefs (site-root relative) as seen from page `here`."""
    if "/" not in here:
        return nav
    pre = here.rsplit("/", 1)[0] + "/"
    return [layout.NavItem(n.href[len(pre):] if n.href.startswith(pre) else "../" + n.href,
                           n.label) for n in nav]


def render_site(src: PathLike, api: bool = True) -> Dict[str, str]:
    """{output path relative to the site root: full HTML}. Pure: reads `src`, writes nothing."""
    src = Path(src)
    order = page_order(src)
    rewrite = _rewriter(order)
    entries: Dict[str, Entry] = {}
    nav: List[layout.NavItem] = []
    for name in order:
        d = render((src / name).read_text(encoding="utf-8"), rewrite=rewrite)
        label = "Overview" if name == INDEX else (d.title or name[:-3])
        nav.append(layout.NavItem(html_name(name), label))
        entries[html_name(name)] = (d.title or label, d.html, d.toc())
    for href, (label, entry) in (render_api() if api else {}).items():
        nav.append(layout.NavItem(href, label))
        entries[href] = entry
    out: Dict[str, str] = {}
    for item in nav:
        title, body, heads = entries[item.href]
        rel = _relative(nav, item.href)
        out[item.href] = layout.page(title, body, rel, rel[nav.index(item)].href, heads)
    return out


def build_site(src: PathLike = "docs", out: PathLike = "site", api: bool = True) -> BuildResult:
    """Render every page and write it under `out`. Returns the written paths in nav order."""
    pages = render_site(src, api=api)
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    for name in sorted(pages):
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(pages[name])
    return BuildResult(out=dest, pages=[dest / n for n in pages])
