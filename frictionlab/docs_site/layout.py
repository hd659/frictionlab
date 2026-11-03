"""Shared page shell for the static docs site: embedded CSS, sidebar nav, TOC, prev/next.

The output is well-formed XHTML-style markup (every element closed, void elements
self-closed, only XML-safe entities), so tests can parse each page with
`xml.etree`. There is no JavaScript, no external asset and no http(s) reference:
the site works from a file:// open or any static host.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import List, Optional, Sequence

from .markdown import Heading

SITE_NAME = "frictionlab docs"

CSS = """\
*{box-sizing:border-box}
body{margin:0;font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:#1d232b;background:#fff}
.wrap{display:flex;min-height:100vh}
nav.side{flex:0 0 16rem;padding:1.2rem 1rem;background:#f4f6f8;border-right:1px solid #dde2e7}
nav.side .brand{display:block;font-weight:700;margin-bottom:.8rem;color:#1d232b;text-decoration:none}
nav.side ol{list-style:none;margin:0;padding:0}
nav.side li{margin:.15rem 0}
nav.side a{color:#2f5d8a;text-decoration:none;font-size:.92rem}
nav.side a.current{font-weight:700;color:#1d232b}
main{flex:1 1 auto;max-width:52rem;padding:1.5rem 2rem 3rem}
.toc{border:1px solid #dde2e7;border-radius:6px;padding:.6rem 1rem;margin:1rem 0;font-size:.92rem}
.toc ul{margin:.2rem 0;padding-left:1.1rem}
.toc .lvl3{margin-left:1rem}
h1,h2,h3{line-height:1.25}
h2{border-bottom:1px solid #e6e9ed;padding-bottom:.2rem;margin-top:2rem}
a{color:#2f5d8a}
code{font:.88em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f1f3f5;padding:.1em .3em;border-radius:3px}
pre{background:#f6f8fa;border:1px solid #e1e4e8;border-radius:6px;padding:.8rem 1rem;overflow-x:auto}
pre code{background:none;padding:0;font-size:.85em}
table{border-collapse:collapse;margin:1rem 0;font-size:.92rem}
th,td{border:1px solid #d6dbe0;padding:.3rem .6rem}
th{background:#f4f6f8}
.pager{display:flex;justify-content:space-between;margin-top:2.5rem;padding-top:1rem;border-top:1px solid #e6e9ed}
@media (max-width:48rem){.wrap{display:block}nav.side{border-right:0;border-bottom:1px solid #dde2e7}}
"""


@dataclass(frozen=True)
class NavItem:
    href: str           # output file name, e.g. "phase1-core-engine.html"
    label: str          # plain text (escaped on render)


def _a(href: str, label: str, cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<a href="{escape(href, quote=True)}"{c}>{escape(label)}</a>'


def sidebar(nav: Sequence[NavItem], current: str) -> str:
    items = []
    for item in nav:
        cls = "current" if item.href == current else ""
        items.append(f"<li>{_a(item.href, item.label, cls)}</li>")
    brand = _a(nav[0].href, SITE_NAME, "brand") if nav else escape(SITE_NAME)
    return (f'<nav class="side" aria-label="Pages">\n{brand}\n<ol>\n'
            + "\n".join(items) + "\n</ol>\n</nav>")


def toc(headings: Sequence[Heading]) -> str:
    """Per-page contents from h2/h3; empty when the page has fewer than two."""
    if len(headings) < 2:
        return ""
    lis = [f'<li class="lvl{lvl}"><a href="#{escape(hid, quote=True)}">{body}</a></li>'
           for lvl, hid, body, _ in headings]
    return ('<div class="toc" role="navigation"><strong>On this page</strong>\n<ul>\n'
            + "\n".join(lis) + "\n</ul>\n</div>")


def pager(prev: Optional[NavItem], nxt: Optional[NavItem]) -> str:
    left = _a(prev.href, "<- " + prev.label, "prev") if prev else "<span></span>"
    right = _a(nxt.href, nxt.label + " ->", "next") if nxt else "<span></span>"
    return f'<div class="pager">\n{left}\n{right}\n</div>'


def page(title: str, body: str, nav: Sequence[NavItem], current: str,
         headings: Sequence[Heading] = ()) -> str:
    """Full HTML document for one page. `body` is trusted HTML from `markdown.render`."""
    hrefs: List[str] = [n.href for n in nav]
    i = hrefs.index(current) if current in hrefs else -1
    prev = nav[i - 1] if i > 0 else None
    nxt = nav[i + 1] if 0 <= i < len(nav) - 1 else None
    full_title = f"{title} - {SITE_NAME}" if title else SITE_NAME
    return "\n".join([
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        f"<title>{escape(full_title)}</title>",
        f"<style>\n{CSS}</style>",
        "</head>",
        "<body>",
        '<div class="wrap">',
        sidebar(nav, current),
        "<main>",
        toc(headings),
        body.rstrip("\n"),
        pager(prev, nxt),
        "</main>",
        "</div>",
        "</body>",
        "</html>",
        "",
    ])
