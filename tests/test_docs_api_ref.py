"""fl-044: the generated API reference pages under OUT/api/ and the --api build flag."""
import enum
import importlib
import re
import xml.etree.ElementTree as ET
from fractions import Fraction

import pytest

from frictionlab import cli
from frictionlab.docs_site import api_ref, build_site
from frictionlab.docs_site.build import render_api
from tests.test_docs_site_build import DOCS, index_links

PACKAGES = api_ref.PACKAGES
ADDRESS = re.compile(r"0x[0-9a-fA-F]+|object at|<[A-Za-z_][\w.]* object|<function |<bound ")


def parse(path):
    text = path.read_text("utf-8")
    assert text.startswith("<!DOCTYPE html>\n")
    return ET.fromstring(text.split("\n", 1)[1])


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return build_site(DOCS, tmp_path_factory.mktemp("site"))


def symbol_ids(root):
    return [h.get("id") for h in root.iter("h3")]


def test_nav_has_api_section_after_phase_writeups(site):
    names = site.names()
    md = ["index.html"] + [n[:-3] + ".html" for n in index_links()]
    api = ["api/index.html"] + [f"api/{api_ref.page_name(p)}" for p in PACKAGES]
    assert names == md + api                         # write-ups first, then the API section
    assert [p.name for p in (site.out / "api").iterdir()] and \
        sorted(n.name for n in (site.out / "api").iterdir()) == sorted(a[4:] for a in api)


@pytest.mark.parametrize("pkg", PACKAGES)
def test_every_public_name_appears_exactly_once(site, pkg):
    mod = importlib.import_module(f"frictionlab.{pkg}")
    root = parse(site.out / "api" / api_ref.page_name(pkg))
    ids = symbol_ids(root)
    expected = sorted(f"{mod.__name__}.{n}" for n in api_ref.public_names(mod))
    if hasattr(mod, "__all__"):
        assert len(mod.__all__) == len(set(mod.__all__)), pkg
    kinds = {s.qualname: s.kind for s in api_ref.collect(pkg)}
    order = [k for k, _ in api_ref.KINDS]
    assert ids == sorted(ids, key=lambda i: (order.index(kinds[i]), i))   # by kind, then name
    assert sorted(ids) == expected and len(ids) == len(set(ids))


@pytest.mark.parametrize("pkg", PACKAGES)
def test_rendered_signatures_have_no_addresses_or_object_reprs(site, pkg):
    root = parse(site.out / "api" / api_ref.page_name(pkg))
    sigs = ["".join(p.itertext()) for p in root.iter("pre") if p.get("class") == "sig"]
    sigs += ["".join(c.itertext()) for li in root.iter("li") if li.get("id") for c in li.iter("code")]
    assert sigs
    for s in sigs:
        assert not ADDRESS.search(s), s


def test_signature_normalization():
    class Color(enum.Enum):
        RED = 1

    class Thing:
        pass

    def f(a: int, b=Color.RED, c=frozenset({"z", "a"}), d=Thing(), e=(1,), g=Fraction(1, 3),
          h=len, *, k: "Optional[int]" = None) -> "List[int]":
        """Doc."""
    sig = api_ref.signature(f)
    assert sig == ("(a: int, b=Color.RED, c=frozenset({'a', 'z'}), d=Thing(...), e=(1,), "
                   "g=Fraction(1, 3), h=len, *, k: Optional[int] = None) -> List[int]")
    assert api_ref.norm(set()) == "set()" and api_ref.norm({"k": [Color.RED]}) == "{'k': [Color.RED]}"


def test_anchors_unique_per_page_and_every_xref_resolves(site):
    anchors = {}
    for path in (site.out / "api").iterdir():
        root = parse(path)
        ids = [el.get("id") for el in root.iter() if el.get("id")]
        assert len(ids) == len(set(ids)), path.name
        anchors[path.name] = set(ids)
    n_xref = 0
    for path in (site.out / "api").iterdir():
        for a in parse(path).iter("a"):
            href = a.get("href")
            page, _, frag = href.partition("#")
            if page.startswith("../"):
                assert (site.out / page[3:]).is_file(), href
                continue
            target = page or path.name
            assert target in anchors, (path.name, href)
            assert not frag or frag in anchors[target], (path.name, href)
            n_xref += a.get("class") == "xref"
    assert n_xref > 20


def test_qualified_cross_reference_links_stable_anchor():
    idx = {"frictionlab.markets.VenueRouter": "markets.html#frictionlab.markets.VenueRouter"}
    html = api_ref.inline("see `frictionlab.markets.VenueRouter` and `x < y`", idx)
    assert ('<a class="xref" href="markets.html#frictionlab.markets.VenueRouter">'
            "<code>frictionlab.markets.VenueRouter</code></a>") in html
    assert "<code>x &lt; y</code>" in html and "<a" not in html.split("and")[1]
    assert api_ref.inline("`VenueRouter`", idx, "frictionlab.markets").count('class="xref"') == 1


def test_api_index_lists_every_package_with_counts(site):
    root = parse(site.out / "api" / "index.html")
    rows = [tr for tr in root.iter("tr") if tr.find("td") is not None]
    got = {"".join(tr[0].itertext()): int(tr[1].text) for tr in rows}
    assert list(got) == [f"frictionlab.{p}" for p in PACKAGES]
    assert got == {f"frictionlab.{p}": len(api_ref.collect(p)) for p in PACKAGES}


def test_symbol_page_shows_module_summary_and_members(site):
    root = parse(site.out / "api" / "markets.html")
    div = next(d for d in root.iter("div") if d.get("class") == "api-symbol"
               and d.find("h3").get("id") == "frictionlab.markets.VenueRouter")
    text = "".join(div.itertext())
    assert "Defined in frictionlab.markets.router" in text and "class VenueRouter(" in text
    assert any(li.get("id") == "frictionlab.markets.VenueRouter.plan" for li in div.iter("li"))
    exp = "".join(parse(site.out / "api" / "experiments.html").itertext())
    assert "frictionlab.experiments.coase.Config" in exp          # aliases name their origin


def test_two_builds_are_byte_identical(tmp_path):
    a, b = build_site(DOCS, tmp_path / "a"), build_site(DOCS, tmp_path / "b")
    assert a.names() == b.names()
    for pa, pb in zip(a.pages, b.pages):
        assert pa.read_bytes() == pb.read_bytes(), pa.name
        assert not re.search(rb"https?://", pa.read_bytes())


def test_render_api_is_pure_and_matches_build(site):
    pages = render_api()
    assert list(pages) == [n for n in site.names() if n.startswith("api/")]


def test_cli_api_flag(tmp_path, capsys):
    assert cli.main(["docs", "build", "--out", str(tmp_path / "on")]) == 0
    assert f"{len(PACKAGES) + 1} API pages" in capsys.readouterr().out
    assert cli.main(["docs", "build", "--no-api", "--out", str(tmp_path / "off")]) == 0
    assert ", 0 API pages" in capsys.readouterr().out
    assert not (tmp_path / "off" / "api").exists() and (tmp_path / "on" / "api").is_dir()
