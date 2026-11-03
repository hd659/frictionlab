# frictionlab

An agent-based simulator for markets with friction: fees, spreads, latency, gas, funding,
information asymmetry and the cost of deciding.

Status: Phases 1-5 complete (core engine, agents, market structures, analysis toolkit, five
seeded experiments). Phase 6 (ecosystem) is in progress: config schema, experiment registry,
CLI, benchmark harness, performance work, CI and the contribution guide are in. See
[ROADMAP.md](ROADMAP.md), the write-ups in [docs/index.md](docs/index.md) and
[CONTRIBUTING.md](CONTRIBUTING.md).

```bash
pip install -e ".[test]"     # add yaml for YAML configs: ".[test,yaml]"
python -m pytest -q
```

## Continuous integration

`ci/ci.yml` runs on every push and pull request, on Python 3.10, 3.11 and 3.12:

1. `python -m pytest -q`, which includes `tests/test_repo_hygiene.py` (workflow wiring,
   doc links, example configs, module line ceiling, no personal URLs or home paths);
2. `frictionlab bench --size small --compare benchmarks/baseline_small.json`, a gate on
   checksums only (no timing thresholds);
3. `python -m frictionlab validate` on every `examples/configs/*.json`.

An optional job installs `numba`, sets `FRICTIONLAB_NUMBA=1` and reruns the order book
oracle tests. Run the same steps locally before you open a PR (see CONTRIBUTING).

## CLI quickstart

Every Phase 5 experiment can be run from one JSON (or YAML, with PyYAML) config:

```bash
frictionlab list                                         # registered experiments
frictionlab validate examples/configs/coase_small.json   # check only, runs nothing
frictionlab show-config examples/configs/fee_wars_small.json --seed 7
frictionlab run examples/configs/fee_wars_small.json --out runs/fee_wars
python -m frictionlab run examples/configs/coase_small.json --set params.levels=[0,28,56]
```

`run` writes three artifacts into the output directory:

| file | content |
|------|---------|
| `rows.csv` | tidy rows, one per (cell, replicate), identical to `run_from_config(...).rows` |
| `resolved_config.json` | the fully resolved config (defaults, small preset, overrides) |
| `report.html` | offline fl-026 report; the header records the resolved config |

The same config and seed give byte-identical artifacts. The output directory is
`--out`, else the config's `output`, else `frictionlab_runs/<experiment>`.

Override precedence, lowest to highest:

1. the config file;
2. flags: `--small`, `--seed N`, `--workers N`;
3. `--set key.path=value`, applied left to right. Values are parsed as JSON when
   possible (`5`, `[0,10]`, `true`, `null`), otherwise kept as strings.

Exit codes: `0` success, `2` config or usage error (every bad key is listed with its dotted
path, and an unknown experiment gets a close-match suggestion), `1` runtime error. Add
`--debug` to print a traceback.

A config looks like this (`examples/configs/coase_small.json`):

```json
{"experiment": "coase", "master_seed": 28, "replicates": 3, "small": true,
 "params": {"values": [150, 140, 130, 120], "costs": [70, 80, 90, 100],
            "levels": [0, 14, 28, 42, 56, 70, 84]}}
```

Keys left out fall back to the experiment's defaults, or its small preset when
`small` is true. `frictionlab show-config` prints exactly what will run.

## Python API

```python
from frictionlab.experiments import run_from_config
res = run_from_config("examples/configs/coase_small.json")
res.findings                       # headline numbers
res.report().write_report("coase.html")
```

From Python, `frictionlab.cli.main(argv)` returns the exit code instead of exiting.

## Layout

- `frictionlab/core`, `costs`, `markets`, `agents`: engine, friction stack, venues, agents
- `frictionlab/analysis`: metrics, sweeps, welfare, equilibrium, reports
- `frictionlab/experiments`: the five Phase 5 experiments and the registry
- `frictionlab/config`: typed config schema and loader
- `examples/`: one script per phase/experiment, plus `examples/configs/*.json`
- `docs/`: one write-up per phase and experiment, indexed in `docs/index.md`
- `benchmarks/`: checksum-only baseline for `frictionlab bench --compare`
- `tests/reference_*.py`: frozen pre-optimisation oracles for the performance work
