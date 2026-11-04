"""API reference: public symbols, normalized signatures and docstrings, as HTML (fl-044).

`collect(pkg)` walks a package's `__all__` (or, when absent, the classes and functions
it defines or re-exports from its own submodules). Signature defaults go through
`norm`: enum members print by name, sets are sorted, other objects print as
`Type(...)`, so no memory address or hash-ordered repr reaches a page. Symbols and
members are anchored by qualified name; a backticked name in a docstring links to its
anchor, qualified (`frictionlab.markets.VenueRouter`) or local to the page's package.
`api_coverage(packages)` lists public classes / functions without their own docstring
(an auto-generated dataclass `Name(...)` doc does not count).
"""
from __future__ import annotations

import enum
import importlib
import inspect
import re
import types
from dataclasses import dataclass, field
from fractions import Fraction
from html import escape, unescape
from typing import Any, Dict, Iterable, List, Tuple, Union

ROOT = "frictionlab"
PACKAGES = ("core", "costs", "markets", "agents", "analysis", "experiments", "config",
            "bench", "cli")
KINDS = (("class", "Classes"), ("function", "Functions"), ("data", "Data"))
SCALARS = (bool, int, float, str, bytes, Fraction, type(None))
UNSAFE = re.compile(r"\s+at 0x[0-9A-Fa-f]+|<([^<>]*)>")    # last-line guard on any text
TICKS = re.compile(r"`([^`\n]+)`")
ModLike = Union[str, types.ModuleType]
Heading = Tuple[int, str, str, str]


class _Text(str):
    """Pre-rendered text that `inspect.Signature` prints verbatim."""

    def __repr__(self) -> str:
        return str(self)


def _safe(text: str) -> str:
    return UNSAFE.sub(lambda m: m.group(1) or "", text)


def norm(value: Any) -> str:
    """Deterministic, address-free repr of a default or constant value."""
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, SCALARS):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(map(norm, value)) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(map(norm, value)) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, (set, frozenset)):
        inner = "{" + ", ".join(sorted(map(norm, value))) + "}" if value else ""
        return f"{type(value).__name__}({inner})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{norm(k)}: {norm(v)}" for k, v in value.items()) + "}"
    if getattr(value, "__module__", None) == "typing":
        return _safe(inspect.formatannotation(value))
    if inspect.isclass(value) or inspect.isroutine(value):
        return getattr(value, "__qualname__", type(value).__name__)
    return f"{type(value).__name__}(...)"


def _ann(a: Any) -> Any:
    return a if a is inspect.Parameter.empty else _Text(
        a if isinstance(a, str) else inspect.formatannotation(a))


def signature(obj: Any, drop_first: bool = False) -> str:
    """`(params) -> ret` with normalized defaults and unquoted annotations, or ''."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return ""
    params = [p.replace(annotation=_ann(p.annotation),
                        default=p.default if p.default is p.empty else _Text(norm(p.default)))
              for p in list(sig.parameters.values())[1 if drop_first else 0:]]
    return _safe(str(sig.replace(parameters=params,
                                 return_annotation=_ann(sig.return_annotation))))


def own_doc(obj: Any) -> str:
    """The object's own cleaned docstring; '' when missing or auto-generated."""
    raw = obj.__dict__.get("__doc__") if inspect.isclass(obj) else getattr(obj, "__doc__", None)
    if not isinstance(raw, str) or not raw.strip() or (
            inspect.isclass(obj) and raw.startswith(obj.__name__ + "(")):
        return ""
    return inspect.cleandoc(raw)


def summary(doc: str) -> str:
    """First paragraph, joined onto one line."""
    return " ".join(doc.strip().split("\n\n", 1)[0].split()) if doc else ""


@dataclass(frozen=True)
class Member:
    """A public method, property or enum member of a documented class."""
    name: str
    kind: str           # method | classmethod | staticmethod | property | member
    signature: str
    doc: str


@dataclass(frozen=True)
class Symbol:
    """One public name of a package, ready to render."""
    package: str        # e.g. "frictionlab.markets"
    name: str           # public (possibly aliased) name
    kind: str           # class | function | data
    module: str         # defining module ('' for data)
    origin: str         # defining qualname (type name for data), differs for aliases
    signature: str
    doc: str
    members: Tuple[Member, ...] = field(default=())

    @property
    def qualname(self) -> str:
        return f"{self.package}.{self.name}"


def module_of(pkg: ModLike) -> types.ModuleType:
    """`"markets"`, `"frictionlab.markets"` or the module itself -> the module."""
    if isinstance(pkg, types.ModuleType):
        return pkg
    return importlib.import_module(pkg if pkg.startswith(ROOT + ".") else f"{ROOT}.{pkg}")


def page_name(pkg: ModLike) -> str:
    """`frictionlab.markets` -> `markets.html`."""
    return module_of(pkg).__name__.removeprefix(ROOT + ".").replace(".", "-") + ".html"


def public_names(mod: types.ModuleType) -> List[str]:
    """`__all__`, else the classes / functions the module defines or re-exports."""
    if getattr(mod, "__all__", None) is not None:
        return list(mod.__all__)
    pre = mod.__name__
    return [n for n, v in vars(mod).items() if not n.startswith("_")
            and (inspect.isclass(v) or inspect.isfunction(v))
            and (v.__module__ == pre or v.__module__.startswith(pre + "."))]


def members(cls: type) -> Tuple[Member, ...]:
    """Public methods, properties and enum members defined on `cls`, by name."""
    out = [Member(m.name, "member", "= " + norm(m.value), "") for m in cls] \
        if issubclass(cls, enum.Enum) else []
    for name, raw in vars(cls).items():
        if name.startswith("_"):
            continue
        if isinstance(raw, (property, types.DynamicClassAttribute)):
            out.append(Member(name, "property", "", own_doc(raw)))
        elif isinstance(raw, (staticmethod, classmethod)):
            kind, f = type(raw).__name__, raw.__func__
            out.append(Member(name, kind, signature(f, kind == "classmethod"), own_doc(f)))
        elif inspect.isfunction(raw):
            out.append(Member(name, "method", signature(raw, True), own_doc(raw)))
    return tuple(sorted(out, key=lambda m: m.name))


def _symbol(mod: types.ModuleType, name: str) -> Symbol:
    v, pkg = getattr(mod, name), mod.__name__
    if inspect.isclass(v):
        sig = "" if issubclass(v, enum.Enum) else signature(v).removesuffix(" -> None")
        return Symbol(pkg, name, "class", v.__module__, v.__qualname__, sig, own_doc(v), members(v))
    if inspect.isfunction(v) or inspect.isbuiltin(v):
        return Symbol(pkg, name, "function", v.__module__, v.__qualname__, signature(v), own_doc(v))
    text = norm(v)
    text = text if len(text) <= 160 else text[:157] + "..."
    return Symbol(pkg, name, "data", "", type(v).__name__, " = " + text, "")


def collect(pkg: ModLike) -> List[Symbol]:
    """Every public symbol of `pkg`, sorted by name."""
    mod = module_of(pkg)
    return sorted((_symbol(mod, n) for n in dict.fromkeys(public_names(mod))), key=lambda s: s.name)


def api_coverage(packages: Iterable[ModLike] = PACKAGES) -> List[str]:
    """Qualified names of public classes / functions without their own docstring."""
    return sorted(s.qualname for p in packages for s in collect(p) if s.kind != "data" and not s.doc)


def xref_index(tables: Dict[str, List[Symbol]]) -> Dict[str, str]:
    """{qualified name: 'page.html#anchor'} for every symbol and member."""
    idx: Dict[str, str] = {}
    for pkg, syms in tables.items():
        page = page_name(pkg)
        for s in syms:
            idx[s.qualname] = f"{page}#{s.qualname}"
            idx.update({f"{s.qualname}.{m.name}": f"{page}#{s.qualname}.{m.name}" for m in s.members})
    return idx


def inline(text: str, idx: Dict[str, str], pkg: str = "") -> str:
    """Escape `text`; backticked spans become code, linked when they name a symbol."""
    def sub(m: re.Match) -> str:
        code, key = f"<code>{m.group(1)}</code>", unescape(m.group(1))
        href = idx.get(key) or (idx.get(f"{pkg}.{key}") if pkg else None)
        return f'<a class="xref" href="{escape(href, quote=True)}">{code}</a>' if href else code
    return TICKS.sub(sub, escape(text, quote=False))


def _doc_html(doc: str, idx: Dict[str, str], pkg: str) -> List[str]:
    out = [f"<p>{inline(summary(doc), idx, pkg)}</p>"] if doc else []
    if "\n\n" in doc.strip():
        out.append(f'<pre class="doc">{inline(doc, idx, pkg)}</pre>')
    return out


def _render_symbol(s: Symbol, idx: Dict[str, str]) -> str:
    head = {"class": "class ", "function": "def ", "data": ""}[s.kind]
    sig = f"{head}{s.name}{s.signature}" + (f": {s.origin}" if s.kind == "data" else "")
    parts = ['<div class="api-symbol">',
             f'<h3 id="{escape(s.qualname, quote=True)}"><code>{escape(s.name)}</code></h3>',
             f'<pre class="sig"><code>{escape(sig)}</code></pre>']
    if s.module:
        what = s.module if s.origin == s.name else f"{s.module}.{s.origin}"
        parts.append(f'<p class="defined">Defined in <code>{escape(what)}</code></p>')
    parts += _doc_html(s.doc, idx, s.package)
    if s.members:
        lis = [f'<li id="{escape(f"{s.qualname}.{m.name}", quote=True)}"><code>'
               f"{escape(m.name + m.signature)}</code> <em>{m.kind}</em>"
               + (f" - {inline(summary(m.doc), idx, s.package)}" if m.doc else "") + "</li>"
               for m in s.members]
        parts.append('<ul class="members">\n' + "\n".join(lis) + "\n</ul>")
    return "\n".join(parts + ["</div>"])


def render_package(pkg: ModLike, syms: List[Symbol],
                   idx: Dict[str, str]) -> Tuple[str, str, List[Heading]]:
    """(title, body html, h2/h3 headings) for one package page."""
    name = module_of(pkg).__name__
    parts = [f'<h1 id="{escape(name, quote=True)}">{escape(name)}</h1>']
    parts += _doc_html(own_doc(module_of(pkg)), idx, name)
    heads: List[Heading] = []
    for kind, label in KINDS:
        group = [s for s in syms if s.kind == kind]
        if group:
            heads.append((2, label.lower(), label, label))
            parts.append(f'<h2 id="{label.lower()}">{label}</h2>')
            for s in group:
                heads.append((3, s.qualname, escape(s.name), s.name))
                parts.append(_render_symbol(s, idx))
    return name, "\n".join(parts) + "\n", heads
