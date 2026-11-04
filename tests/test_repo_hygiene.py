"""Repository hygiene: CI wiring, doc links, example configs, module size, privacy.

These tests guard the repo as a whole rather than one module. They read files only,
never run the network, and skip the YAML parse when PyYAML is absent.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List

import pytest

from frictionlab.config.schema import load_config
from frictionlab.experiments.registry import resolve_config

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "ci" / "ci.yml"
BASELINE = "benchmarks/baseline_small.json"

# Module line ceiling (design principle: one concept per 100-250 line module).
MAX_MODULE_LINES = int(os.environ.get("FRICTIONLAB_MAX_MODULE_LINES", "250"))
# Known exceptions, each capped at its current size so it cannot keep growing.
LINE_ALLOWLIST: Dict[str, int] = {
    "frictionlab/markets/batch_auction.py": 330,
    "frictionlab/agents/arbitrageur.py": 280,
    "frictionlab/experiments/adverse_selection.py": 270,
    "frictionlab/core/runner.py": 260,
    "frictionlab/core/ledger.py": 260,
}

# Public hosts that may appear in tracked files; anything else counts as a personal host.
PUBLIC_HOSTS = ("www.w3.org", "github.com", "docs.github.com", "pypi.org", "python.org",
                "docs.python.org", "numpy.org", "doi.org", "arxiv.org", "en.wikipedia.org")
URL_RE = re.compile(r"https?://([A-Za-z0-9][A-Za-z0-9.-]*)")
# Split literals so this file does not match its own patterns.
HOME_RE = re.compile("|".join([r"/" + r"Users/[A-Za-z0-9_.-]+", r"/" + r"home/[a-z_][a-z0-9_-]*/",
                               r"[A-Za-z]:\\" + r"Users\\"]))
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".yml", ".yaml",
                 ".ref", ".html", ".csv", ""}


def _repo_files() -> List[Path]:
    """Tracked plus untracked-but-not-ignored files; falls back to a walk without git."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [ROOT / line for line in out.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        skip = {".git", "__pycache__", ".pytest_cache"}
        files = [p for p in ROOT.rglob("*")
                 if p.is_file() and not skip.intersection(p.relative_to(ROOT).parts)]
    return sorted(p for p in files if p.is_file())


def _text_files() -> List[Path]:
    return [p for p in _repo_files() if p.suffix.lower() in TEXT_SUFFIXES
            or p.name in {".gitignore", "LICENSE"}]


def _md_links(path: Path) -> List[str]:
    return [m.split("#")[0] for m in re.findall(r"\]\(([^)\s]+)\)", path.read_text("utf-8"))
            if not re.match(r"[a-z]+:", m) and not m.startswith("#")]


# --------------------------------------------------------------------------- CI workflow
def _workflow_runs(doc: dict) -> str:
    return "\n".join(str(step.get("run", "")) for job in doc["jobs"].values()
                     for step in job.get("steps", []))


def test_workflow_yaml_references_tests_bench_and_baseline():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(WORKFLOW.read_text("utf-8"))
    triggers = doc.get("on", doc.get(True))           # PyYAML reads bare `on` as True
    assert {"push", "pull_request"} <= set(triggers)
    runs = _workflow_runs(doc)
    assert "pytest" in runs
    assert re.search(r"frictionlab bench --size small\b.*--compare " + re.escape(BASELINE), runs)
    assert "frictionlab validate" in runs and "examples/configs/" in runs
    matrix = doc["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    assert [str(v) for v in matrix] == ["3.10", "3.11", "3.12"]
    numba = doc["jobs"]["numba"]
    assert str(numba["env"]["FRICTIONLAB_NUMBA"]) == "1"
    assert "numba" in _workflow_runs({"jobs": {"n": numba}})
    assert (ROOT / BASELINE).is_file()


def test_workflow_docs_job_builds_then_checks_without_deploy():
    yaml = pytest.importorskip("yaml")
    jobs = yaml.safe_load(WORKFLOW.read_text("utf-8"))["jobs"]
    runs = _workflow_runs({"jobs": {"docs": jobs["docs"]}})
    build = re.search(r"frictionlab docs build\b[^\n]*--out (\S+)", runs)
    check = re.search(r"frictionlab docs check (\S+)", runs)
    assert build and check, "docs job must run `docs build --out DIR` and `docs check DIR`"
    assert build.group(1) == check.group(1) == "_site" and build.start() < check.start()
    uses = [str(s.get("uses", "")) for s in jobs["docs"]["steps"]]
    assert not any("pages" in u or "deploy" in u for u in uses)
    assert not re.search(r"--allow\b|curl |wget ", runs)


def test_workflow_text_references_docs_build_and_check():
    text = WORKFLOW.read_text("utf-8")
    assert "frictionlab docs build --out _site" in text
    assert "frictionlab docs check _site" in text


def test_workflow_text_has_no_timing_threshold():
    text = WORKFLOW.read_text("utf-8")
    assert BASELINE in text and "pytest" in text
    assert not re.search(r"--(max|min)-(time|seconds)|speedup\s*[<>]", text)


# --------------------------------------------------------------------------- docs
def test_roadmap_doc_references_exist():
    refs = set(re.findall(r"docs/[\w./-]+\.md", (ROOT / "ROADMAP.md").read_text("utf-8")))
    assert refs, "ROADMAP.md should reference its write-ups"
    missing = sorted(r for r in refs if not (ROOT / r).is_file())
    assert not missing, f"ROADMAP.md references missing docs: {missing}"


def test_docs_index_links_every_phase_doc():
    index = ROOT / "docs" / "index.md"
    linked = {(index.parent / link).resolve() for link in _md_links(index)}
    phases = sorted((ROOT / "docs").glob("phase*.md"))
    assert phases
    unlinked = [p.name for p in phases if p.resolve() not in linked]
    assert not unlinked, f"docs/index.md does not link: {unlinked}"
    assert (ROOT / "docs" / "phase6-ecosystem.md").resolve() in linked


def test_markdown_relative_links_resolve():
    for md in [ROOT / "README.md", ROOT / "ROADMAP.md", ROOT / "CONTRIBUTING.md",
               *sorted((ROOT / "docs").glob("*.md"))]:
        bad = [l for l in _md_links(md) if l and not (md.parent / l).exists()]
        assert not bad, f"{md.relative_to(ROOT)}: dangling links {bad}"


# --------------------------------------------------------------------------- configs
CONFIGS = sorted((ROOT / "examples" / "configs").glob("*.json"))


def test_example_configs_exist():
    assert CONFIGS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_example_config_validates(path):
    cfg = resolve_config(load_config(path))
    assert cfg.experiment and isinstance(cfg.master_seed, int)


# --------------------------------------------------------------------------- module size
def _module_lines() -> Dict[str, int]:
    return {p.relative_to(ROOT).as_posix(): len(p.read_text("utf-8").splitlines())
            for p in sorted((ROOT / "frictionlab").rglob("*.py"))
            if "__pycache__" not in p.parts}


def test_modules_stay_under_line_ceiling():
    over = {name: n for name, n in _module_lines().items()
            if n > LINE_ALLOWLIST.get(name, MAX_MODULE_LINES)}
    assert not over, (f"modules over the {MAX_MODULE_LINES}-line ceiling (split them, or "
                      f"justify an allowlist entry): {over}")


def test_line_allowlist_is_not_stale():
    lines = _module_lines()
    unknown = sorted(set(LINE_ALLOWLIST) - set(lines))
    assert not unknown, f"allowlist names missing modules: {unknown}"
    shrunk = sorted(n for n in LINE_ALLOWLIST if lines[n] <= MAX_MODULE_LINES)
    if MAX_MODULE_LINES == 250:                         # only meaningful at the default
        assert not shrunk, f"now under the ceiling, drop from allowlist: {shrunk}"


# --------------------------------------------------------------------------- privacy
def _host_ok(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(host == h or host.endswith("." + h) for h in PUBLIC_HOSTS)


def test_no_personal_urls_or_home_paths():
    problems = []
    for path in _text_files():
        try:
            text = path.read_text("utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT).as_posix()
        for i, line in enumerate(text.splitlines(), 1):
            hosts = [h for h in URL_RE.findall(line) if not _host_ok(h)]
            if hosts:
                problems.append(f"{rel}:{i}: url host {hosts}")
            if HOME_RE.search(line):
                problems.append(f"{rel}:{i}: absolute home-directory path")
    assert not problems, "\n".join(problems)


def test_privacy_patterns_catch_samples():
    assert not _host_ok("my-blog.example.net") and _host_ok("www.w3.org")
    assert URL_RE.findall("see https://" + "private.example.org/x") == ["private.example.org"]
    assert HOME_RE.search("/" + "Users/someone/project") and HOME_RE.search("/" + "home/dev/x")
    assert not HOME_RE.search("/tmp/after.json")
