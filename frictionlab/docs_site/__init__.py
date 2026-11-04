"""Stdlib-only static documentation site for frictionlab (fl-043).

    from frictionlab.docs_site import build_site
    build_site("docs", "site")          # or: frictionlab docs build --out site

`markdown` renders the Markdown subset the docs use (block parse in `blocks`),
`layout` provides the shared page shell, and `build` follows `docs/index.md` for the
page order and writes a byte-reproducible site with no JavaScript and no external
assets. `linkcheck` verifies a built site: every internal link and anchor resolves
and no external URL slips in (`frictionlab docs check DIR`).
"""
from .build import BuildResult, DocsError, build_site, page_order, render_site
from .linkcheck import CheckResult, LinkError, check_site
from .markdown import Doc, render, rewrite_link, slugify

__all__ = ["BuildResult", "CheckResult", "Doc", "DocsError", "LinkError", "build_site",
           "check_site", "page_order", "render", "render_site", "rewrite_link", "slugify"]
