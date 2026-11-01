"""`frictionlab` command line: a stdlib-only argparse front end to the fl-036 registry.

    frictionlab list
    frictionlab validate examples/configs/coase_small.json
    frictionlab show-config examples/configs/coase_small.json --seed 7
    frictionlab run examples/configs/fee_wars_small.json --out runs/fee_wars \
        --set params.max_rounds=4
    frictionlab bench --size small --json bench.json \
        --compare benchmarks/baseline_small.json

Overrides are applied in a fixed order: the config file, then flags (`--small`,
`--seed`, `--workers`), then each `--set key.path=value` left to right. A `--set`
value is parsed as JSON when it parses (`5`, `[0,10]`, `true`, `null`), else kept as
a string. `--out` only picks the destination directory; it is not written into the
resolved config, so artifacts are byte-identical wherever they are written.

`bench` times the fl-038 scenarios and prints a table; `--compare` exits 1 on any
checksum mismatch against the baseline (results changed), never on timing.

Exit codes: 0 success, 2 config or usage error, 1 runtime error. Errors go to
stderr with their dotted paths; tracebacks only with `--debug`. `main(argv)`
returns the code instead of exiting, so tests call it in-process.
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
        value = json.loads(raw)
    except ValueError:
        value = raw
    return parts, value


def apply_overrides(raw: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Config file < flags < --set (left to right). Returns a new raw mapping."""
    if not isinstance(raw, dict):
        raise ConfigError([("<root>", f"expected a mapping, got {type(raw).__name__}")])
    out = json.loads(json.dumps(raw))                  # deep copy, JSON-normalized
    if args.small:
        out["small"] = True
    if args.seed is not None:
        out["master_seed"] = args.seed
    if args.workers is not None:
        out["n_workers"] = args.workers
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
    names = list_experiments()
    width = max(map(len, names), default=0)
    for n in names:
        print(f"{n:<{width}}  {get_experiment(n).doc}")
    return OK


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = resolve_config(_load(args))
    print(f"ok: {args.config} -> {cfg.experiment} (master_seed={cfg.master_seed}, "
          f"replicates={cfg.replicates}, small={str(cfg.small).lower()})")
    return OK


def cmd_show_config(args: argparse.Namespace) -> int:
    print(resolve_config(_load(args)).to_json())
    return OK


def cmd_run(args: argparse.Namespace) -> int:
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
    for name in ARTIFACTS:
        print(f"  wrote {out / name}")
    return OK


def cmd_bench(args: argparse.Namespace) -> int:
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


def build_parser() -> argparse.ArgumentParser:
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
    run.add_argument("--out", metavar="DIR",
                     help="output directory (default: config 'output', else frictionlab_runs/NAME)")
    run.set_defaults(func=cmd_run)
    sub.add_parser("list", parents=[common], help="list registered experiments"
                   ).set_defaults(func=cmd_list)
    sub.add_parser("validate", parents=[common, over], help="check a config without running"
                   ).set_defaults(func=cmd_validate)
    sub.add_parser("show-config", parents=[common, over],
                   help="print the fully resolved config as canonical JSON"
                   ).set_defaults(func=cmd_show_config)
    b = sub.add_parser("bench", parents=[common], help="time the benchmark scenarios")
    b.add_argument("--scenario", action="append", metavar="NAME",
                   help="run only this scenario (repeatable; default: all)")
    b.add_argument("--size", choices=("small", "medium"), default="small")
    b.add_argument("--repeat", type=int, default=3, help="timed repeats (default 3)")
    b.add_argument("--warmup", type=int, default=1, help="untimed warmup runs (default 1)")
    b.add_argument("--json", metavar="PATH", help="write results as sorted-key JSON")
    b.add_argument("--checksums-only", action="store_true",
                   help="with --json: drop timings (baseline format)")
    b.add_argument("--compare", metavar="BASELINE",
                   help="diff against a result file; exit 1 on checksum mismatch")
    b.set_defaults(func=cmd_bench)
    return p


def _report(err: BaseException, debug: bool, header: str) -> None:
    if debug:
        traceback.print_exc(file=sys.stderr)
    print(header, file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
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
