"""fl-043: the stdlib Markdown subset renders to exact, escaped, deterministic HTML."""
import xml.etree.ElementTree as ET

import pytest

from frictionlab.docs_site.blocks import parse, split_row
from frictionlab.docs_site.markdown import inline, plain_text, render, rewrite_link, slugify


def html(text):
    return render(text).html


def test_atx_headings_get_slug_ids_and_title():
    doc = render("# Phase 1: the *core* engine\n\n## Before / after ##\n")
    assert doc.html == ('<h1 id="phase-1-the-core-engine">Phase 1: the <em>core</em> engine</h1>\n'
                        '<h2 id="before-after">Before / after</h2>\n')
    assert doc.title == "Phase 1: the core engine"
    assert [h[:2] for h in doc.headings] == [(1, "phase-1-the-core-engine"), (2, "before-after")]


def test_duplicate_slugs_are_suffixed_and_stable():
    text = "## Results\n\n## Results\n\n### Results\n\n## `odd` *!!*\n"
    ids = [h[1] for h in render(text).headings]
    assert ids == ["results", "results-2", "results-3", "odd"]
    assert len(set(ids)) == len(ids)
    assert [h[1] for h in render(text).headings] == ids        # stable across renders
    assert slugify("!!!") == "section"


def test_paragraph_joins_lines_and_escapes_text():
    assert html("a < b & c\n\"q\" 'x'\n") == "<p>a &lt; b &amp; c\n&quot;q&quot; &#x27;x&#x27;</p>\n"


def test_raw_html_is_never_passed_through():
    out = html('<script>alert(1)</script>\n\n<b onclick="x">hi</b>\n')
    assert "<script" not in out and "<b " not in out
    assert "&lt;script&gt;" in out


def test_inline_code_emphasis_strong():
    assert inline("`a<b>` **bold** *em* and `x*y*`") == (
        "<code>a&lt;b&gt;</code> <strong>bold</strong> <em>em</em> and <code>x*y*</code>")
    assert inline("``a ` b``") == "<code>a ` b</code>"
    assert inline(r"\*not em\*") == "*not em*"
    assert inline("2 * 3 * 4") == "2 * 3 * 4"


def test_links_rewrite_md_and_leave_external_untouched():
    assert inline("[p1](phase1.md)") == '<a href="phase1.html">p1</a>'
    assert inline("[s](phase1.md#setup)") == '<a href="phase1.html#setup">s</a>'
    assert inline("[gh](https://github.com/x.md)") == '<a href="https://github.com/x.md">gh</a>'
    assert inline("[top](#intro)") == '<a href="#intro">top</a>'
    assert inline("[**b** `c`](a.md)") == '<a href="a.html"><strong>b</strong> <code>c</code></a>'
    assert rewrite_link("mailto:x") == "mailto:x" and rewrite_link("data.csv") == "data.csv"
    assert inline('[q](a.md?x="1")').count("&quot;") == 2


def test_unordered_list_with_nesting():
    text = "- a\n  - b\n  - c\n    1. deep\n- d\n  continued\n"
    assert html(text) == ("<ul>\n<li>a\n<ul>\n<li>b</li>\n<li>c\n<ol>\n<li>deep</li>\n</ol></li>\n"
                          "</ul></li>\n<li>d\ncontinued</li>\n</ul>\n")


def test_ordered_list_start_and_loose_items():
    assert html("3. three\n4. four\n") == '<ol start="3">\n<li>three</li>\n<li>four</li>\n</ol>\n'
    loose = html("1. one\n\n   more\n2. two\n")
    assert loose == "<ol>\n<li><p>one</p>\n<p>more</p></li>\n<li><p>two</p></li>\n</ol>\n"


def test_list_ends_at_a_different_marker_kind():
    blocks = parse("- a\n1. b\n")
    assert [(b[0], b[1]) for b in blocks] == [("list", False), ("list", True)]


def test_fenced_code_keeps_content_literal_and_escaped():
    text = '```python\nif a < b && c > "d":\n    s = \'x\'  # **not bold**\n```\n'
    assert html(text) == ('<pre><code class="language-python">if a &lt; b &amp;&amp; c &gt; '
                          '&quot;d&quot;:\n    s = &#x27;x&#x27;  # **not bold**\n</code></pre>\n')
    assert html("~~~\n# not a heading\n~~~\n") == "<pre><code># not a heading\n</code></pre>\n"


def test_indented_code_block():
    assert html("text\n\n    x = 1\n    -> y\n") == "<p>text</p>\n<pre><code>x = 1\n-&gt; y\n</code></pre>\n"


def test_table_with_alignment_and_pipe_in_code():
    text = "| name | qty | price |\n|:-----|:---:|------:|\n| `a|b` | 2 | 10 |\n| short |\n"
    assert html(text) == (
        "<table>\n<thead>\n<tr><th style=\"text-align:left\">name</th>"
        "<th style=\"text-align:center\">qty</th><th style=\"text-align:right\">price</th></tr>\n"
        "</thead>\n<tbody>\n<tr><td style=\"text-align:left\"><code>a|b</code></td>"
        "<td style=\"text-align:center\">2</td><td style=\"text-align:right\">10</td></tr>\n"
        "<tr><td style=\"text-align:left\">short</td><td style=\"text-align:center\"></td>"
        "<td style=\"text-align:right\"></td></tr>\n</tbody>\n</table>\n")
    assert split_row(r"| a \| b | c |") == [r"a \| b", "c"]
    assert inline(r"a \| b") == "a | b"


def test_pipe_without_delimiter_row_is_a_paragraph():
    assert html("a | b\nc | d\n") == "<p>a | b\nc | d</p>\n"


def test_plain_text_strips_markup():
    assert plain_text("Use [`run`](cli.md) **now**") == "Use run now"


@pytest.mark.parametrize("text", [
    "# T\n\n- a\n  - b\n\n| x | y |\n|---|--:|\n| 1 | 2 |\n\n```\n<&>\n```\n",
    "1. a *b* **c**\n2. [d](e.md)\n\n    code <\n",
    "```\nunclosed fence <\n",
])
def test_output_is_well_formed_and_deterministic(text):
    a, b = html(text), html(text.replace("\n", "\r\n"))
    assert a == b
    ET.fromstring("<div>" + a + "</div>")
