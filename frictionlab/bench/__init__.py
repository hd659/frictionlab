"""Reproducible benchmark harness (Phase 6 performance baseline).

`scenarios` defines named, seeded workloads at small/medium sizes; `harness`
times them (warmup + repeats, min/median/max, ops/sec) next to a deterministic
work checksum, and `compare` diffs two result files. The CLI front end is
`frictionlab bench`; `benchmarks/baseline_small.json` holds checksums only.
"""
from .harness import (
    SCHEMA, ChecksumDrift, Comparison, checksums_only, compare, format_table, load_json,
    run_bench, run_scenario, to_json, write_json,
)
from .scenarios import SCENARIOS, SIZES, Scenario, Work, get_scenario, scenario_names

__all__ = [
    "SCHEMA", "ChecksumDrift", "Comparison", "checksums_only", "compare", "format_table",
    "load_json", "run_bench", "run_scenario", "to_json", "write_json",
    "SCENARIOS", "SIZES", "Scenario", "Work", "get_scenario", "scenario_names",
]
