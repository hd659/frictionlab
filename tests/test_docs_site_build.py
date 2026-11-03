"""fl-043: the static docs site built from docs/index.md, and `frictionlab docs build`."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab import cli
from frictionlab.docs_site import DocsError, build_site, page_order, render_site

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
NEW_MODULES = ["frictionlab/docs_site/__init__.py", "frictionlab/docs_site/blocks.py",
               "frictionlab/docs_site/markdown.py", "frictionlab/docs_site/build.py",
               "frictionlab/docs_site/layout.py"]
MAX_LINES = 250                                     # tests/test_repo_hygiene.py default


def parse_page(text):
    assert text.startswith("<!DOCTYPE html>\n")
    return ET.fromstring(text.split("\n", 1)[1])


def index_links():
    return list(dict.fromkeys(re.findall(r"\]\(([a-z0-9-]+\.md)(?:#[^)]*)?\)",
                                         (DOCS / "index.md").read_text("utf-8"))))


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return build_site(DOCS, tmp_path_factory.mktemp("site"))


def test_one_page_per_index_entry_in_nav_order(site):
    expected = ["index.html"] + [n[:-3] + ".html" for n in index_links()]
    assert site.names() == expected
    assert sorted(p.name for p in site.out.iterdir()) == sorted(expected)
    assert all(f"{p.stem}.html" in site.names() for p in DOCS.glob("phase*.md"))


def test_every_page_is_well_formed_with_nav_in_index_order(site):
    for i, path in enumerate(site.pages):
        root = parse_page(path.read_text("utf-8"))
        nav = root.find(".//nav")
        hrefs = [a.get("href") for a in nav.iter("a") if a.get("class") != "brand"]
        assert hrefs == site.names()
        current = [a.get("href") for a in nav.iter("a") if a.get("class") == "current"]
        assert current == [path.name]
        pager = {a.get("class"): a.get("href") for a in root.iter("a")
                 if a.get("class") in ("prev", "next")}
        assert pager.get("prev") == (site.names()[i - 1] if i else None)
        assert pager.get("next") == (site.names()[i + 1] if i + 1 < len(site.pages) else None)


def test_no_external_references_or_scripts(site):
    for path in site.pages:
        text = path.read_text("utf-8")
        assert not re.search(r"https?://", text), path.name
        assert "<script" not in text and "<link" not in text and "src=" not in text


def test_toc_lists_h2_h3_and_anchors_exist(site):
    root = parse_page((site.out / "phase6-performance.html").read_text("utf-8"))
    ids = {el.get("id") for el in root.iter() if el.get("id")}
    toc = [a.get("href") for div in root.iter("div") if div.get("class") == "toc"
           for a in div.iter("a")]
    assert len(toc) >= 5 and all(h[1:] in ids for h in toc)
    assert len(ids) == len([el for el in root.iter() if el.get("id")])   # unique ids


def test_internal_links_resolve_and_external_md_links_are_kept(site):
    names = set(site.names())
    for path in site.pages:
        for a in parse_page(path.read_text("utf-8")).iter("a"):
            href = a.get("href")
            target = href.partition("#")[0]
            if target.endswith(".html"):
                assert target in names, (path.name, href)
            assert not re.fullmatch(r"[a-z0-9-]+\.md(#.*)?", href), (path.name, href)
    index = (site.out / "index.html").read_text("utf-8")
    assert 'href="phase1-core-engine.html"' in index and 'href="../README.md"' in index


def test_two_builds_are_byte_identical(tmp_path):
    a, b = build_site(DOCS, tmp_path / "a"), build_site(DOCS, tmp_path / "b")
    assert a.names() == b.names()
    for pa, pb in zip(a.pages, b.pages):
        assert pa.read_bytes() == pb.read_bytes()
        assert b"\r\n" not in pa.read_bytes()


def test_render_site_is_pure_and_matches_files(site):
    pages = render_site(DOCS)
    assert list(pages) == site.names()
    assert all((site.out / n).read_text("utf-8") == html for n, html in pages.items())


def _tree(tmp_path, index, **pages):
    src = tmp_path / "docs"
    src.mkdir()
    (src / "index.md").write_text(index, encoding="utf-8")
    for name, text in pages.items():
        (src / f"{name}.md").write_text(text, encoding="utf-8")
    return src


def test_page_order_follows_first_link_and_folds_duplicates(tmp_path):
    src = _tree(tmp_path, "# I\n\n[b](b.md) [a](a.md#x) [b again](b.md) [r](../README.md)\n",
                a="# A\n", b="# B\n[back](index.md)\n")
    assert page_order(src) == ["index.md", "b.md", "a.md"]
    out = build_site(src, tmp_path / "out")
    assert out.names() == ["index.html", "b.html", "a.html"]
    assert 'href="index.html">back' in (tmp_path / "out" / "b.html").read_text("utf-8")


def test_missing_linked_page_raises_and_cli_exits_2(tmp_path, capsys):
    src = _tree(tmp_path, "# I\n\n- [gone](gone.md)\n")
    with pytest.raises(DocsError, match="gone.md"):
        page_order(src)
    code = cli.main(["docs", "build", "--src", str(src), "--out", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert code == 2 and str(src / "gone.md") in err and "not found" in err
    assert not (tmp_path / "o").exists()


def test_missing_index_exits_2(tmp_path, capsys):
    code = cli.main(["docs", "build", "--src", str(tmp_path), "--out", str(tmp_path / "o")])
    assert code == 2 and "index.md" in capsys.readouterr().err


def test_cli_build_writes_site(tmp_path, capsys):
    code = cli.main(["docs", "build", "--src", str(DOCS), "--out", str(tmp_path / "s")])
    out = capsys.readouterr().out
    assert code == 0 and (tmp_path / "s" / "index.html").is_file()
    assert out.startswith(f"docs: {len(index_links()) + 1} pages")
    assert cli.main(["docs"]) == 2


def test_runtime_error_exits_1(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    code = cli.main(["docs", "build", "--src", str(DOCS), "--out", str(blocker / "sub")])
    assert code == 1 and "error:" in capsys.readouterr().err


@pytest.mark.parametrize("rel", NEW_MODULES)
def test_new_modules_stay_under_line_ceiling(rel):
    assert len((ROOT / rel).read_text("utf-8").splitlines()) <= MAX_LINES
