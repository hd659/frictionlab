"""Phase 6 ecosystem walkthrough: config -> run -> bench gate -> docs site, in one directory.

Everything goes through the public CLI entry point `frictionlab.cli.main(argv)`, the
same code path as the `frictionlab` console script and as CI:

1. `validate` then `run` `examples/configs/coase_small.json` into OUT/run/
   (rows.csv, resolved_config.json, report.html);
2. `bench --size small --compare benchmarks/baseline_small.json`, writing the
   checksum-only result to OUT/bench_small.json (the gate fails on checksum drift,
   never on timing);
3. `docs build --out OUT/site` and `docs check OUT/site` (internal links and anchors
   resolve, no external URL).

It then prints a manifest of every artifact with its size and a short sha256 and
writes the same lines to OUT/MANIFEST.txt. Nothing time- or path-dependent is
written, so two runs give byte-identical run, bench and docs artifacts.

    python examples/phase6_ecosystem.py --small [OUT]     # OUT defaults to a temp dir

`--small` uses the experiment's small preset and one bench repeat with no warmup;
without it the bench runs 3 repeats after 1 warmup. Bench timings never reach disk
(`--checksums-only`), so the bench file is reproducible too.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))  # run from a checkout

from frictionlab.cli import main as cli_main  # noqa: E402  (after sys.path tweak)

CONFIG = HERE / "configs" / "coase_small.json"
BASELINE = ROOT / "benchmarks" / "baseline_small.json"
DOCS = ROOT / "docs"


class StepFailed(RuntimeError):
    """A CLI step returned a non-zero exit code."""

    def __init__(self, argv: Sequence[str], code: int, output: str) -> None:
        super().__init__(f"`frictionlab {' '.join(argv)}` exited {code}\n{output}")
        self.code = code


def cli(*argv: str) -> str:
    """Run `frictionlab ARGV` in-process; return its output, raise StepFailed on error."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli_main(list(argv))
    if code != 0:
        raise StepFailed(argv, code, buf.getvalue())
    return buf.getvalue()


def steps(out: Path, small: bool) -> List[Tuple[str, Tuple[str, ...]]]:
    """(label, argv) of every CLI step, in order."""
    run_flags = ("--small",) if small else ()
    bench_flags = ("--repeat", "1", "--warmup", "0") if small else ("--repeat", "3")
    return [
        ("validate config", ("validate", str(CONFIG), *run_flags)),
        ("run experiment", ("run", str(CONFIG), *run_flags, "--out", str(out / "run"))),
        ("bench checksum gate", ("bench", "--size", "small", *bench_flags,
                                 "--json", str(out / "bench_small.json"), "--checksums-only",
                                 "--compare", str(BASELINE))),
        ("build docs site", ("docs", "build", "--src", str(DOCS), "--out", str(out / "site"))),
        ("check docs links", ("docs", "check", str(out / "site"), "--src", str(DOCS))),
    ]


def manifest(out: Path) -> List[str]:
    """`<sha256[:12]>  <bytes>  <relative path>` for every artifact, sorted by path."""
    lines = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.txt":
            data = p.read_bytes()
            rel = p.relative_to(out).as_posix()
            lines.append(f"{hashlib.sha256(data).hexdigest()[:12]}  {len(data):>8}  {rel}")
    return lines


def summarize(lines: List[str]) -> List[str]:
    """Short console view: run and bench files in full, the site folded into one line."""
    site = [ln for ln in lines if ln.split()[-1].startswith("site/")]
    rest = [ln for ln in lines if ln not in site]
    n_bytes = sum(int(ln.split()[1]) for ln in site)
    return rest + [f"{'':12}  {n_bytes:>8}  site/ ({len(site)} pages, index.html)"]


def run(out: Path, small: bool = True) -> List[str]:
    """Run every step into `out`; return the manifest lines."""
    out.mkdir(parents=True, exist_ok=True)
    for label, argv in steps(out, small):
        text = cli(*argv)
        last = text.strip().splitlines()[-1] if text.strip() else ""
        print(f"[ok] {label}: {last.strip()}")
    lines = manifest(out)
    (out / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", help="output directory (default: a new temp dir)")
    ap.add_argument("--small", action="store_true", help="small preset, one bench repeat")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab_phase6_"))
    try:
        lines = run(out, small=args.small)
    except StepFailed as e:
        print(f"error: {e}", file=sys.stderr)
        return e.code
    print(f"artifacts in {out}:")
    for line in summarize(lines):
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
