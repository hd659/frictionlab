"""Benchmark harness: warmup + repeat timing, deterministic checksums, compare.

    results = run_bench(size="small", repeat=5)
    print(format_table(results)); write_json(results, "bench.json")
    report = compare(load_json("bench.json"), load_json("benchmarks/baseline_small.json"))

Timing (`min_s`, `median_s`, `max_s`, `ops_per_s`) is machine-specific and
never asserted against a threshold. The checksum is machine-independent: the
harness raises if two repeats of one scenario disagree, and `compare` flags any
mismatch against a baseline as an error, so a speedup can prove it did not
change results. JSON uses sorted keys so files diff cleanly.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .scenarios import get_scenario, scenario_names

SCHEMA = 1
PathLike = Union[str, Path]


class ChecksumDrift(RuntimeError):
    """A scenario produced different results on two repeats (nondeterminism)."""


def run_scenario(name: str, size: str = "small", *, warmup: int = 1,
                 repeat: int = 3) -> Dict[str, Any]:
    if repeat < 1 or warmup < 0:
        raise ValueError(f"need repeat >= 1 and warmup >= 0, got {repeat}, {warmup}")
    sc = get_scenario(name)
    for _ in range(warmup):
        sc.run(size)
    times: List[float] = []
    sums = set()
    work = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        work = sc.run(size)
        times.append(max(time.perf_counter() - t0, 1e-9))   # clock floor: never 0
        sums.add(work.checksum)
    if len(sums) != 1:
        raise ChecksumDrift(f"{name}/{size}: checksums differ across repeats: {sorted(sums)}")
    med = statistics.median(times)
    return {"checksum": work.checksum, "ops": work.ops, "trades": work.trades,
            "volume": work.volume, "repeat": repeat, "warmup": warmup,
            "min_s": min(times), "median_s": med, "max_s": max(times),
            "ops_per_s": work.ops / med}


def run_bench(names: Sequence[str] = (), size: str = "small", *, warmup: int = 1,
              repeat: int = 3) -> Dict[str, Any]:
    return {"schema": SCHEMA, "size": size,
            "scenarios": {n: run_scenario(n, size, warmup=warmup, repeat=repeat)
                          for n in scenario_names(names)}}


def checksums_only(results: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip machine-specific timing: the form committed as a baseline."""
    keep = ("checksum", "ops", "trades", "volume")
    return {"schema": results.get("schema", SCHEMA), "size": results["size"],
            "scenarios": {n: {k: r[k] for k in keep if k in r}
                          for n, r in results["scenarios"].items()}}


def to_json(results: Mapping[str, Any]) -> str:
    return json.dumps(results, sort_keys=True, indent=2) + "\n"


def write_json(results: Mapping[str, Any], path: PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_json(results), encoding="utf-8")
    return p


def load_json(path: PathLike) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), dict):
        raise ValueError(f"{path}: not a frictionlab bench result (missing 'scenarios')")
    return data


def format_table(results: Mapping[str, Any]) -> str:
    rows = [("scenario", "ops", "min_s", "median_s", "max_s", "ops/s", "checksum")]
    for n, r in sorted(results["scenarios"].items()):
        rows.append((n, str(r["ops"]), f"{r['min_s']:.4f}", f"{r['median_s']:.4f}",
                     f"{r['max_s']:.4f}", f"{r['ops_per_s']:,.0f}", r["checksum"]))
    w = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    fmt = lambda row: "  ".join(c.ljust(w[i]) if i in (0, 6) else c.rjust(w[i])
                                for i, c in enumerate(row)).rstrip()
    head = f"frictionlab bench (size={results['size']})"
    return "\n".join([head, fmt(rows[0]), fmt(tuple("-" * x for x in w))]
                     + [fmt(r) for r in rows[1:]])


@dataclass
class Comparison:
    errors: List[str] = field(default_factory=list)
    speedups: Dict[str, Optional[float]] = field(default_factory=dict)  # baseline / current

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = [f"{n}: speedup {'n/a (no timing)' if s is None else f'{s:.2f}x'}"
                 for n, s in sorted(self.speedups.items())]
        return "\n".join(lines + [f"ERROR: {e}" for e in self.errors]
                         + [f"compare: {'ok' if self.ok else f'{len(self.errors)} error(s)'}"])


def compare(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> Comparison:
    """Diff `current` against `baseline`; speedup = baseline median / current median.

    Checksum mismatches and size mismatches are errors. Scenarios missing from
    either side are errors only when they are in `current` but not `baseline`
    (a subset run with `--scenario` is fine).
    """
    out = Comparison()
    if current.get("size") != baseline.get("size"):
        out.errors.append(f"size mismatch: current {current.get('size')!r} "
                          f"vs baseline {baseline.get('size')!r}")
    base = baseline["scenarios"]
    for n, cur in sorted(current["scenarios"].items()):
        ref = base.get(n)
        if ref is None:
            out.errors.append(f"{n}: not in baseline")
            continue
        if cur.get("checksum") != ref.get("checksum"):
            out.errors.append(f"{n}: checksum mismatch: current {cur.get('checksum')} "
                              f"vs baseline {ref.get('checksum')}")
        bt, ct = ref.get("median_s"), cur.get("median_s")
        out.speedups[n] = bt / ct if bt and ct else None
    return out
