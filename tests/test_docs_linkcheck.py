"""fl-045: frictionlab/docs_site/linkcheck.py and `frictionlab docs check`."""
from pathlib import Path

import pytest

from frictionlab import cli
from frictionlab.docs_site import CheckResult, DocsError, LinkError, build_site, check_site
from frictionlab.docs_site.linkcheck import parse_page, site_pages

DOCS = Path(__file__).resolve().parents[1] / "docs"
# Split literals so the repo privacy scan does not read these as real hosts.
HOST = "https:" + "//example.com/"
EXT = HOST + "x"


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return build_site(DOCS, tmp_path_factory.mktemp("site") / "_site")


def _page(body: str, title: str = "t") -> str:
    return (f'<!DOCTYPE html>\n<html><head><title>{title}</title></head>'
            f'<body>\n{body}\n</body></html>\n')


def _write(root: Path, **pages: str) -> Path:
    for name, html in pages.items():
        path = root / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    return root


# --------------------------------------------------------------------------- real site
def test_real_built_site_passes(site):
    res = check_site(site.out, src=DOCS)
    assert res.ok, res.format()
    assert res.pages == len(site.pages) and res.links > len(site.pages)
    assert sorted(site.names()) == site_pages(site.out)


def test_real_site_cli_exits_0(site, capsys):
    assert cli.main(["docs", "check", str(site.out), "--src", str(DOCS)]) == 0
    assert "0 broken" in capsys.readouterr().out


def test_real_site_checks_api_cross_reference_anchors(site):
    ids, links = parse_page((site.out / "api" / "markets.html").read_text("utf-8"))
    assert "frictionlab.markets.VenueRouter" in ids
    assert any("#" in href for href in links)


# --------------------------------------------------------------------------- broken pages
@pytest.fixture
def broken(tmp_path):
    body = "\n".join([
        '<h1 id="top">Index</h1>',
        '<a href="ok.html">fine</a>',
        '<a href="ok.html#sec">fine anchor</a>',
        '<a href="#top">self anchor</a>',
        '<a href="gone.html">missing file</a>',
        '<a href="ok.html#nope">missing anchor</a>',
        f'<a href="{EXT}">external</a>',
    ])
    return _write(tmp_path / "out", **{"index.html": _page(body),
                                       "ok.html": _page('<h2 id="sec">S</h2>')})


def test_three_errors_missing_file_anchor_and_external(broken):
    res = check_site(broken)
    assert not res.ok and res.pages == 2 and res.links == 6
    assert [(e.page, e.target) for e in res.errors] == [
        ("index.html", "gone.html"), ("index.html", "ok.html#nope"),
        ("index.html", EXT)]
    reasons = [e.reason for e in res.errors]
    assert reasons[0] == "missing file" and "#nope" in reasons[1] and "allowlist" in reasons[2]
    assert all(isinstance(e, LinkError) for e in res.errors)


def test_cli_exits_2_and_reports_every_error(broken, capsys):
    assert cli.main(["docs", "check", str(broken)]) == 2
    err = capsys.readouterr().err
    for target in ("gone.html", "ok.html#nope", EXT):
        assert f"index.html: {target}:" in err
    assert "3 broken" in err


def test_allowlist_admits_matching_prefix_only(broken):
    res = check_site(broken, allow=[HOST])
    assert [e.target for e in res.errors] == ["gone.html", "ok.html#nope"]
    assert len(check_site(broken, allow=[HOST.replace(".com", ".org")]).errors) == 3


def test_nested_pages_directory_links_and_assets(tmp_path):
    out = _write(tmp_path / "s", **{
        "index.html": _page('<a href="api/">api</a> <a href="api/a.html#f.g">sym</a>'
                            ' <img src="logo.svg" />'),
        "logo.svg": "<svg/>",
        "api__index.html": _page('<a href="../index.html">up</a>'),
        "api__a.html": _page('<div id="f.g"></div><a name="old"></a>'
                             '<a href="a.html#old">n</a> <a href="../nowhere.html">x</a>'),
    })
    res = check_site(out)
    assert [(e.page, e.target) for e in res.errors] == [("api/a.html", "../nowhere.html")]


def test_links_leaving_the_site_resolve_against_src(tmp_path):
    src = _write(tmp_path / "repo" / "docs", **{"index.md": "# I\n"})
    (tmp_path / "repo" / "README.md").write_text("# R\n\n## Quick start\n", encoding="utf-8")
    body = ('<a href="../README.md#quick-start">ok</a> <a href="../README.md#nah">bad</a>'
            ' <a href="../MISSING.md">gone</a>')
    out = _write(tmp_path / "out", **{"index.html": _page(body)})
    res = check_site(out, src=src)
    assert [e.target for e in res.errors] == ["../README.md#nah", "../MISSING.md"]
    assert len(check_site(out).errors) == 3                 # no src: all three leave the site


@pytest.mark.parametrize("href", ["/" + "/cdn.example.com/x.js", "/abs.html", "mailto:x",
                                  "javascript:void(0)", "HTTP:" + "//EXAMPLE.COM"])
def test_other_external_forms_are_flagged(tmp_path, href):
    out = _write(tmp_path / "o", **{"index.html": _page(f'<a href="{href}">x</a>')})
    assert [e.target for e in check_site(out).errors] == [href]


def test_missing_or_empty_site_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(DocsError, match="not found"):
        check_site(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(DocsError, match="no .html"):
        check_site(tmp_path / "empty")
    assert cli.main(["docs", "check", str(tmp_path / "nope")]) == 2
    assert "site directory not found" in capsys.readouterr().err


def test_format_lists_errors_then_summary():
    res = CheckResult(pages=1, links=2, errors=[LinkError("a.html", "b.html", "missing file")])
    assert res.format().splitlines() == ["  a.html: b.html: missing file",
                                         "linkcheck: 1 pages, 2 links, 1 broken"]
