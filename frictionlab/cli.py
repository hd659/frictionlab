"""`frictionlab` command line: a stdlib-only argparse front end to the fl-036 registry.

    frictionlab list
    frictionlab validate examples/configs/coase_small.json
    frictionlab show-config examples/configs/coase_small.json --seed 7
    frictionlab run examples/configs/fee_wars_small.json --set params.max_rounds=4
    frictionlab bench --size small --compare benchmarks/baseline_small.json
    frictionlab docs build --src docs --out site && frictionlab docs check site

Overrides are applied in a fixed order: the config file, then flags (`--small`,
`--seed`, `--workers`), then each `--set key.path=value` left to right. A `--set`
value is parsed as JSON when it parses (`5`, `[0,10]`, `true`, `null`), else kept as
a string. `--out` only picks the destination directory; it is not written into the
resolved config, so artifacts are byte-identical wherever they are written.

`bench --compare` exits 1 on any checksum mismatch against the baseline, never on
timing. `docs build` renders the pages linked from `docs/index.md` plus the API
reference (`--no-api` skips it); `docs check DIR` exits 2 on any broken internal link
or anchor, or any http(s) link not matched by `--allow`.

Exit codes: 0 success, 2 config or usage error (including a missing docs page or a
broken link), 1 runtime error. Errors go to stderr; tracebacks only with `--debug`.
`main(argv)` returns the code instead of exiting, so tests call it in-process.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .analysis.sweep import write_csv
from .config.schema import ConfigError, load_config
from .experiments.registry import get_experiment, list_experiments, resolve_config, run_from_config

OK, RUNTIME, USAGE = 0, 1, 2
ARTIFACTS = ("rows.csv", "resolved_config.json", "report.html")


def parse_set(item: str) -> tuple:
    """`'params.levels=[0,10]'` -> (['params', 'levels'], [0, 10])."""
    key, sep, raw = item.partition("=")
    parts = key.strip().split(".")
    if not sep or not all(parts):
        raise ConfigError([("--set", f"expected key.path=value, got {item!r}")])
    try:
        return parts, json.loads(raw)
    except ValueError:
        return parts, raw


def apply_overrides(raw: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Config file < flags < --set (left to right). Returns a new raw mapping."""
    if not isinstance(raw, dict):
        raise ConfigError([("<root>", f"expected a mapping, got {type(raw).__name__}")])
    out = json.loads(json.dumps(raw))                  # deep copy, JSON-normalized
    for key, val in (("small", args.small or None), ("master_seed", args.seed),
                     ("n_workers", args.workers)):
        if val is not None:
            out[key] = val
    for item in args.set or ():
        parts, value = parse_set(item)
        node = out
        for i, k in enumerate(parts[:-1]):
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ConfigError([("--set", f"{'.'.join(parts[:i + 1])} is not a mapping")])
        node[parts[-1]] = value
    return out


def _load(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        raw = load_config(args.config)
    except OSError as e:
        raise ConfigError([("<file>", f"cannot read {args.config}: {e.strerror or e}")]) from None
    return apply_overrides(raw, args)


def cmd_list(args: argparse.Namespace) -> int:
    """`frictionlab list`: print each registered experiment and its summary."""
    width = max(map(len, list_experiments()), default=0)
    for n in list_experiments():
        print(f"{n:<{width}}  {get_experiment(n).doc}")
    return OK


def cmd_validate(args: argparse.Namespace) -> int:
    """`frictionlab validate`: resolve a config and report it without running."""
    cfg = resolve_config(_load(args))
    print(f"ok: {args.config} -> {cfg.experiment} (master_seed={cfg.master_seed}, "
          f"replicates={cfg.replicates}, small={str(cfg.small).lower()})")
    return OK


def cmd_show_config(args: argparse.Namespace) -> int:
    """`frictionlab show-config`: print the resolved config as canonical JSON."""
    print(resolve_config(_load(args)).to_json())
    return OK


def cmd_run(args: argparse.Namespace) -> int:
    """`frictionlab run`: run a config and write rows.csv, resolved_config.json and report.html."""
    cfg = resolve_config(_load(args))                  # config errors surface before any work
    out = Path(args.out or cfg.output or Path("frictionlab_runs") / cfg.experiment)
    res = run_from_config(cfg)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(res.rows, out / "rows.csv")
    (out / "resolved_config.json").write_text(json.dumps(res.config, sort_keys=True, indent=2)
                                              + "\n", encoding="utf-8")
    res.report().write_report(out / "report.html")
    print(f"{cfg.experiment}: {len(res.rows)} rows, master_seed={cfg.master_seed}, "
          f"replicates={cfg.replicates}")
    print("\n".join(f"  wrote {out / name}" for name in ARTIFACTS))
    return OK


def cmd_bench(args: argparse.Namespace) -> int:
    """`frictionlab bench`: time scenarios, optionally write JSON and compare to a baseline."""
    from . import bench                                # lazy: keeps `run`/`list` startup lean
    try:
        names = bench.scenario_names(args.scenario or ())
        baseline = bench.load_json(args.compare) if args.compare else None
    except KeyError as e:
        raise ConfigError([("--scenario", str(e.args[0]))]) from None
    except (OSError, ValueError) as e:
        raise ConfigError([("--compare", str(e))]) from None
    res = bench.run_bench(names, args.size, warmup=args.warmup, repeat=args.repeat)
    print(bench.format_table(res))
    if args.json:
        out = bench.checksums_only(res) if args.checksums_only else res
        print(f"  wrote {bench.write_json(out, args.json)}")
    if baseline is None:
        return OK
    cmp = bench.compare(res, baseline)
    print(cmp.format())
    return OK if cmp.ok else RUNTIME


def _docs(fn, *a, **kw):
    """Call a docs_site function (imported lazily), mapping DocsError to a usage error."""
    from .docs_site import DocsError
    try:
        return fn(*a, **kw)
    except DocsError as e:
        raise ConfigError([(e.path, e.message)]) from None


def cmd_docs_build(args: argparse.Namespace) -> int:
    """`frictionlab docs build`: build the static docs site and API reference."""
    from .docs_site import build_site                  # lazy, like bench
    res = _docs(build_site, args.src, args.out, api=args.api)
    n_api = sum(n.startswith("api/") for n in res.names())
    print(f"docs: {len(res.pages) - n_api} pages from {args.src}, {n_api} API pages\n"
          + "\n".join(f"  wrote {p}" for p in res.pages))
    return OK


def cmd_docs_check(args: argparse.Namespace) -> int:
    """`frictionlab docs check`: link-check a built site; exit 2 on any broken link."""
    from .docs_site import check_site
    res = _docs(check_site, args.dir, src=args.src, allow=args.allow or ())
    print(res.format(), file=sys.stdout if res.ok else sys.stderr)
    return OK if res.ok else USAGE


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for every `frictionlab` subcommand."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", help="print tracebacks on errors")
    over = argparse.ArgumentParser(add_help=False)
    over.add_argument("config", help="experiment config (.json, or .yaml/.yml with PyYAML)")
    over.add_argument("--small", action="store_true", default=None,
                      help="use the experiment's small (fast) preset")
    over.add_argument("--seed", type=int, help="override master_seed")
    over.add_argument("--workers", type=int, help="override n_workers")
    over.add_argument("--set", action="append", metavar="KEY.PATH=VALUE",
                      help="override any config key; applied last, left to right")
    p = argparse.ArgumentParser(prog="frictionlab",
                                description="Agent-based simulator for markets with friction.")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True
    run = sub.add_parser("run", parents=[common, over], help="run an experiment from a config")
    run.add_argument("--out", metavar="DIR", help="output dir (default: config 'output', "
                     "else frictionlab_runs/NAME)")
    run.set_defaults(func=cmd_run)
    for name, parents, fn, text in (
            ("list", [common], cmd_list, "list registered experiments"),
            ("validate", [common, over], cmd_validate, "check a config without running"),
            ("show-config", [common, over], cmd_show_config,
             "print the fully resolved config as canonical JSON")):
        sub.add_parser(name, parents=parents, help=text).set_defaults(func=fn)
    b = sub.add_parser("bench", parents=[common], help="time the benchmark scenarios")
    b.add_argument("--scenario", action="append", metavar="NAME",
                   help="run only this scenario (repeatable; default: all)")
    b.add_argument("--size", choices=("small", "medium"), default="small")
    b.add_argument("--repeat", type=int, default=3, help="timed repeats (default 3)")
    b.add_argument("--warmup", type=int, default=1, help="untimed warmup runs (default 1)")
    b.add_argument("--json", metavar="PATH", help="write results as sorted-key JSON")
    b.add_argument("--checksums-only", action="store_true", help="with --json: no timings")
    b.add_argument("--compare", metavar="BASELINE", help="exit 1 on any checksum mismatch")
    b.set_defaults(func=cmd_bench)
    d = sub.add_parser("docs", help="docs tools").add_subparsers(dest="docs", metavar="ACTION")
    d.required = True
    db = d.add_parser("build", parents=[common], help="build the static docs site")
    dc = d.add_parser("check", parents=[common], help="link-check a built docs site")
    for q in (db, dc):
        q.add_argument("--src", default="docs", help="docs source directory (default docs)")
    db.add_argument("--out", default="site", metavar="DIR", help="output dir (default site)")
    db.add_argument("--api", action=argparse.BooleanOptionalAction, default=True,
                    help="also generate the API reference under api/ (default on)")
    db.set_defaults(func=cmd_docs_build)
    dc.add_argument("dir", help="built site directory")
    dc.add_argument("--allow", action="append", metavar="PREFIX",
                    help="allow external links starting with PREFIX (default: none)")
    dc.set_defaults(func=cmd_docs_check)
    return p


def _report(err: BaseException, debug: bool, header: str) -> None:
    if debug:
        traceback.print_exc(file=sys.stderr)
    print(header, file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: parse `argv`, dispatch and return an exit code (0, 1 or 2)."""
    parser = build_parser()
    try:
        args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    except SystemExit as e:                            # --help -> 0, usage error -> 2
        return int(e.code or 0) if isinstance(e.code, int) else USAGE
    debug = getattr(args, "debug", False)
    try:
        return int(args.func(args))
    except ConfigError as e:
        where = getattr(args, "config", None)
        lines: List[str] = [f"  {p}: {m}" for p, m in e.errors]
        _report(e, debug, f"error: {len(e.errors)} config error(s)"
                          f"{f' in {where}' if where else ''}:\n" + "\n".join(lines))
        return USAGE
    except Exception as e:                             # noqa: BLE001 - CLI boundary
        _report(e, debug, f"error: {type(e).__name__}: {e}")
        return RUNTIME


if __name__ == "__main__":                             # pragma: no cover
    sys.exit(main())
