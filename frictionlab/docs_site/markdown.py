"""Deterministic Markdown -> HTML for the subset the repo's docs actually use.

Supported: ATX headings with stable slug ids (a repeated slug gets `-2`, `-3`, ...),
paragraphs, ordered and unordered lists with nesting, fenced code blocks (with a
`language-*` class), four-space indented code, inline code, `*em*`, `**strong**`,
`[text](url)` links and pipe tables with `:--` / `:-:` / `--:` alignment. The block
parse lives in `blocks.py`. All text is HTML-escaped and raw HTML is never passed
through. Output is a pure function of the input text and the link rewriter.

    doc = render(text)                  # -> Doc(html, title, headings)
    doc = render(text, rewrite=fn)      # custom link rewriting (default: .md -> .html)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from typing import Callable, Dict, List, Optional, Tuple

from .blocks import parse

Rewriter = Callable[[str], str]
Heading = Tuple[int, str, str, str]             # (level, id, inline html, plain text)

SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
CODE = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)", re.S)
LINK = re.compile(r"\[((?:[^\[\]]|\[[^\]]*\])*)\]\(\s*<?([^)\s>]*)>?(?:\s+\"[^\"]*\")?\s*\)")
STRONG = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.S)
EM = re.compile(r"\*(?=[^\s*])(.+?)(?<=[^\s*])\*", re.S)
ESC = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>])")
INLINE = (CODE, LINK, STRONG, EM, ESC)


def rewrite_link(url: str) -> str:
    """Relative `x.md` / `x.md#frag` -> `x.html[#frag]`; schemes and `#frag` untouched."""
    if not url or url.startswith("#") or SCHEME.match(url) or url.startswith("//"):
        return url
    path, sep, frag = url.partition("#")
    return (path[:-3] + ".html" if path.endswith(".md") else path) + sep + frag


def plain_text(raw: str) -> str:
    """Inline markup stripped: what a heading reads as (for titles and slugs)."""
    s = LINK.sub(lambda m: m.group(1), raw)
    s = re.sub(r"(?<!\\)[`*]", "", s)
    return ESC.sub(r"\1", s).strip()


def slugify(raw: str) -> str:
    s = re.sub(r"[^a-z0-9 _-]", "", plain_text(raw).lower())
    return re.sub(r"-{2,}", "-", re.sub(r"\s+", "-", s.strip())).strip("-") or "section"


def inline(s: str, rewrite: Rewriter = rewrite_link) -> str:
    """Inline spans, left to right; code spans win, so their content is literal."""
    out: List[str] = []
    i = 0
    while i < len(s):
        m: Optional[re.Match] = None
        for rx in INLINE:
            m = rx.match(s, i) if s[i] in "`[*\\" else None
            if m:
                break
        if m is None:
            out.append(escape(s[i]))
            i += 1
            continue
        if rx is CODE:
            out.append(f"<code>{escape(m.group(2).strip() or m.group(2))}</code>")
        elif rx is LINK:
            href = escape(rewrite(m.group(2)), quote=True)
            out.append(f'<a href="{href}">{inline(m.group(1), rewrite)}</a>')
        elif rx is ESC:
            out.append(escape(m.group(1)))
        else:
            tag = "strong" if rx is STRONG else "em"
            out.append(f"<{tag}>{inline(m.group(1), rewrite)}</{tag}>")
        i = m.end()
    return "".join(out)


def _align(delim: str) -> str:
    left, right = delim.startswith(":"), delim.endswith(":")
    a = "center" if left and right else "right" if right else "left" if left else ""
    return f' style="text-align:{a}"' if a else ""


@dataclass
class Doc:
    html: str
    title: str
    headings: List[Heading] = field(default_factory=list)

    def toc(self, levels: Tuple[int, ...] = (2, 3)) -> List[Heading]:
        return [h for h in self.headings if h[0] in levels]


class _Renderer:
    def __init__(self, rewrite: Rewriter) -> None:
        self.rewrite = rewrite
        self.seen: Dict[str, int] = {}
        self.headings: List[Heading] = []

    def slug(self, raw: str) -> str:
        base = slugify(raw)
        n = self.seen.get(base, 0) + 1
        self.seen[base] = n
        return base if n == 1 else f"{base}-{n}"

    def blocks(self, blocks: List[tuple], tight: bool = False) -> List[str]:
        out: List[str] = []
        for b in blocks:
            kind = b[0]
            if kind == "h":
                _, level, raw = b
                hid, body = self.slug(raw), inline(raw, self.rewrite)
                self.headings.append((level, hid, body, plain_text(raw)))
                out.append(f'<h{level} id="{hid}">{body}</h{level}>')
            elif kind == "p":
                body = inline(b[1], self.rewrite)
                out.append(body if tight else f"<p>{body}</p>")
            elif kind == "code":
                cls = f' class="language-{escape(b[1], quote=True)}"' if b[1] else ""
                out.append(f"<pre><code{cls}>{escape(b[2])}\n</code></pre>")
            elif kind == "list":
                out.append(self.list(b))
            elif kind == "table":
                out.append(self.table(b))
        return out

    def list(self, b: tuple) -> str:
        _, ordered, start, items = b
        tag = "ol" if ordered else "ul"
        attr = f' start="{start}"' if ordered and start != 1 else ""
        tight = all(sum(1 for x in it if x[0] == "p") <= 1 for it in items)
        lis = ["<li>" + "\n".join(self.blocks(it, tight)) + "</li>" for it in items]
        return f"<{tag}{attr}>\n" + "\n".join(lis) + f"\n</{tag}>"

    def table(self, b: tuple) -> str:
        _, head, delims, rows = b
        aligns = [_align(d) for d in delims]
        width = len(aligns)

        def row(cells: List[str], tag: str) -> str:
            cells = (cells + [""] * width)[:width]
            return "<tr>" + "".join(f"<{tag}{a}>{inline(c, self.rewrite)}</{tag}>"
                                    for c, a in zip(cells, aligns)) + "</tr>"
        body = "".join("\n" + row(r, "td") for r in rows)
        return (f"<table>\n<thead>\n{row(head, 'th')}\n</thead>\n"
                f"<tbody>{body}\n</tbody>\n</table>")


def render(text: str, rewrite: Rewriter = rewrite_link) -> Doc:
    """Markdown text -> Doc. The title is the first h1 (else first heading, else '')."""
    r = _Renderer(rewrite)
    html = "\n".join(r.blocks(parse(text))) + "\n"
    h1 = [h for h in r.headings if h[0] == 1] or r.headings
    return Doc(html=html, title=h1[0][3] if h1 else "", headings=r.headings)
