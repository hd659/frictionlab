"""Link checker for a built docs site: every internal link must land, nothing external.

    res = check_site("_site", src="docs")      # or: frictionlab docs check _site
    print(res.format()); res.ok

Every `*.html` page under the site root is parsed with `html.parser`. The parser
collects element ids (`id`, plus `name` on `<a>`) and every `href` / `src` value.
Each link is then checked against the output tree:

- a relative link must resolve to an existing output file (a directory link means
  its `index.html`), and a `#fragment` must name an existing id on the target page;
  a bare `#fragment` is checked against the page itself;
- a relative link that leaves the site root (`../README.md`, which the builder keeps
  as written) is resolved against `src`, the docs source directory. The file must
  exist there, and a fragment on a `.md` target must match one of its heading slugs.
  Without `src` such a link is reported;
- an `http(s)` link is reported unless it starts with an entry of `allow`, which is
  empty by default. Protocol-relative links, absolute paths and other schemes are
  always reported.

All pages are checked in one pass. Errors come back in page order (sorted names) and
document order, each one carrying the page, the target as written and a reason.
No network access, and the site is never modified.
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import unquote

from .build import DocsError
from .markdown import SCHEME, render

PathLike = Union[str, Path]
EXTERNAL = re.compile(r"^https?:", re.I)
LINK_ATTRS = ("href", "src")
Parsed = Tuple[Set[str], List[str]]                    # (element ids, link targets)


@dataclass(frozen=True)
class LinkError:
    """One broken link: the page it is on, the target as written, and why."""
    page: str
    target: str
    reason: str

    def __str__(self) -> str:
        return f"{self.page}: {self.target}: {self.reason}"


@dataclass
class CheckResult:
    """Outcome of `check_site`: page and link counts plus every broken link."""
    pages: int
    links: int
    errors: List[LinkError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no link is broken."""
        return not self.errors

    def format(self) -> str:
        """One line per error, then a summary line."""
        tail = (f"linkcheck: {self.pages} pages, {self.links} links, "
                f"{len(self.errors)} broken")
        return "\n".join([*(f"  {e}" for e in self.errors), tail])


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: Set[str] = set()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        for key, value in attrs:
            if value is None:
                continue
            if key == "id" or (key == "name" and tag == "a"):
                self.ids.add(value)
            elif key in LINK_ATTRS:
                self.links.append(value.strip())


def parse_page(html: str) -> Parsed:
    """(ids, links) of one HTML page, links in document order."""
    p = _PageParser()
    p.feed(html)
    p.close()
    return p.ids, p.links


def site_pages(out: Path) -> List[str]:
    """Sorted site-relative names of every `.html` page under `out`."""
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*.html") if p.is_file())


class _Checker:
    def __init__(self, out: Path, src: Optional[Path], allow: Sequence[str]) -> None:
        self.out, self.src, self.allow = out, src, tuple(a for a in allow if a)
        self.pages: Dict[str, Parsed] = {n: parse_page((out / n).read_text("utf-8"))
                                         for n in site_pages(out)}
        self._md_ids: Dict[Path, Set[str]] = {}

    def md_ids(self, path: Path) -> Set[str]:
        if path not in self._md_ids:
            doc = render(path.read_text(encoding="utf-8"))
            self._md_ids[path] = {h[1] for h in doc.headings}
        return self._md_ids[path]

    def outside(self, target: str, frag: str) -> Optional[str]:
        if self.src is None:
            return "outside the site root (no source directory given)"
        path = self.src / target
        if not path.exists():
            return f"missing source file {posixpath.normpath(target)}"
        if frag and path.suffix == ".md" and frag not in self.md_ids(path):
            return f"missing anchor #{frag} in {path.name}"
        return None

    def check(self, page: str, href: str) -> Optional[str]:
        if EXTERNAL.match(href):
            ok = any(href.startswith(a) for a in self.allow)
            return None if ok else "external link not in allowlist"
        if href.startswith("//"):
            return "protocol-relative external link"
        if SCHEME.match(href):
            return f"unsupported scheme {href.split(':', 1)[0]}:"
        raw, _, frag = href.partition("#")
        path, frag = unquote(raw.partition("?")[0]), unquote(frag)
        if path.startswith("/"):
            return "absolute path"
        target = posixpath.normpath(posixpath.join(posixpath.dirname(page), path)) \
            if path else page
        if target == ".." or target.startswith("../"):
            return self.outside(target, frag)
        if (self.out / target).is_dir():
            target = posixpath.join(target, "index.html")
        if target not in self.pages:
            if not (self.out / target).is_file():
                return "missing file"
            return None                                # non-HTML asset: no ids to check
        if frag and frag not in self.pages[target][0]:
            return f"missing anchor #{frag}"
        return None


def check_site(out: PathLike, src: Optional[PathLike] = None,
               allow: Sequence[str] = ()) -> CheckResult:
    """Check every page under `out`. Raises DocsError when there is no site to check."""
    out = Path(out)
    if not out.is_dir():
        raise DocsError(out, "site directory not found")
    checker = _Checker(out, Path(src) if src is not None else None, allow)
    if not checker.pages:
        raise DocsError(out, "no .html pages found")
    res = CheckResult(pages=len(checker.pages), links=0)
    for name, (_, links) in checker.pages.items():
        for href in links:
            res.links += 1
            reason = checker.check(name, href)
            if reason:
                res.errors.append(LinkError(name, href, reason))
    return res
