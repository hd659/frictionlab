"""Block-level Markdown parse: lines -> a list of plain tuples (no HTML here).

    ("h", level, raw)                    ATX heading
    ("p", raw)                           paragraph (soft breaks kept as newlines)
    ("code", lang, text)                 fenced or four-space indented code
    ("list", ordered, start, [blocks])   one parsed block list per item (nesting)
    ("table", head, delims, rows)        pipe table; delims carry the alignment

`markdown.render` turns these into HTML; keeping the parse separate keeps each
module to one concept and lets tests inspect structure without string matching.
"""
from __future__ import annotations

import re
from typing import List, Tuple

HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*([\w+#.-]*)[^`]*$")
ITEM = re.compile(r"^( {0,3})([-*+]|\d{1,9}[.)])( +|$)(.*)$")
DELIM_CELL = re.compile(r"^:?-+:?$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def split_row(line: str) -> List[str]:
    """Pipe-table cells; a `|` inside backticks or escaped as `\\|` does not split."""
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") and not s.endswith("\\|") else s
    cells: List[str] = []
    cur: List[str] = []
    tick, i = False, 0
    while i < len(s):
        c = s[i]
        if c == "\\" and s[i + 1:i + 2] == "|":
            cur.append("\\|")
            i += 2
            continue
        if c == "`":
            tick = not tick
        if c == "|" and not tick:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    return cells + ["".join(cur).strip()]


def is_table(lines: List[str], i: int) -> bool:
    if "|" not in lines[i] or i + 1 >= len(lines) or "|" not in lines[i + 1]:
        return False
    return all(DELIM_CELL.match(c) for c in split_row(lines[i + 1]))


def starts_block(lines: List[str], i: int) -> bool:
    ln = lines[i]
    return bool(HEADING.match(ln) or FENCE.match(ln) or ITEM.match(ln)) or is_table(lines, i)


def _skip_blank(lines: List[str], i: int) -> int:
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i


def _item_body(lines: List[str], i: int, off: int, first: str) -> Tuple[List[str], int]:
    """Collect one item's lines (de-indented by `off`) starting after its marker line."""
    body, n = [first], len(lines)
    while i < n:
        ln = lines[i]
        if not ln.strip():
            j = _skip_blank(lines, i)
            if j < n and _indent(lines[j]) >= off:       # blank inside the item
                body.extend([""] * (j - i))
                i = j
                continue
            break
        if _indent(ln) >= off:
            body.append(ln[off:])
        elif body[-1] == "" or starts_block(lines, i):
            break
        else:
            body.append(ln.strip())                      # lazy paragraph continuation
        i += 1
    return body, i


def _list(lines: List[str], i: int) -> Tuple[tuple, int]:
    m = ITEM.match(lines[i])
    ordered = m.group(2)[0].isdigit()
    start = int(m.group(2)[:-1]) if ordered else 1
    items: List[List[tuple]] = []
    while True:
        off = len(m.group(1)) + len(m.group(2)) + max(len(m.group(3)), 1)
        body, i = _item_body(lines, i + 1, off, m.group(4))
        items.append(parse_blocks(body))
        j = _skip_blank(lines, i)
        m = ITEM.match(lines[j]) if j < len(lines) else None
        if not m or m.group(2)[0].isdigit() != ordered or _indent(lines[j]) > 3:
            return ("list", ordered, start, items), i
        i = j


def _fence(lines: List[str], i: int) -> Tuple[tuple, int]:
    m = FENCE.match(lines[i])
    mark, ind, body = m.group(1), _indent(lines[i]), []
    i += 1
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith(mark) and set(s) == {mark[0]}:
            return ("code", m.group(2), "\n".join(body)), i + 1
        body.append(lines[i][min(ind, _indent(lines[i])):])
        i += 1
    return ("code", m.group(2), "\n".join(body)), i      # unclosed: runs to the end


def parse_blocks(lines: List[str]) -> List[tuple]:
    out: List[tuple] = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif HEADING.match(ln):
            m = HEADING.match(ln)
            out.append(("h", len(m.group(1)), m.group(2)))
            i += 1
        elif FENCE.match(ln):
            block, i = _fence(lines, i)
            out.append(block)
        elif ln.startswith("    "):
            body = []
            while i < n and (not lines[i].strip() or lines[i].startswith("    ")):
                body.append(lines[i][4:])
                i += 1
            out.append(("code", "", "\n".join(body).rstrip("\n")))
        elif ITEM.match(ln):
            block, i = _list(lines, i)
            out.append(block)
        elif is_table(lines, i):
            head, delims, rows = split_row(ln), split_row(lines[i + 1]), []
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i]:
                rows.append(split_row(lines[i]))
                i += 1
            out.append(("table", head, delims, rows))
        else:
            para = [ln.strip()]
            i += 1
            while i < n and lines[i].strip() and not starts_block(lines, i):
                para.append(lines[i].strip())
                i += 1
            out.append(("p", "\n".join(para)))
    return out


def parse(text: str) -> List[tuple]:
    """Normalize newlines and tabs, then parse."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4).split("\n")
    return parse_blocks(lines)
