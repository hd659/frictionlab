# Contributing to frictionlab

frictionlab is built to answer one question: how do transaction costs, information
asymmetry and decision costs change who trades and what prices form. Any contribution
that makes that question easier to answer, or the answers easier to trust, is welcome.
[ROADMAP.md](ROADMAP.md) gives the plan and [docs/index.md](docs/index.md) lists the write-ups.

## Design principles

1. **Seeded reproducibility.** Each run is a pure function of its config and
   `master_seed`. Draw randomness from the seeded RNG passed in (`core/clock.py`,
   `numpy.random.default_rng(seed)`). Never use global `random` state, wall-clock time or
   iteration over unordered sets in a result path. Same seed, same bytes: `rows.csv`,
   `report.html` and the bench checksums are compared byte for byte.
2. **Integer ticks.** Prices, cash, fees and frictions are Python `int` ticks. Rates are
   integer basis points, and rounding goes through one explicit, documented helper.
   `Fraction` is acceptable as a fallback. A float must never reach the ledger. The
   ledger rejects `bool`, `float` and numpy floats, and that rejection is part of the API.
3. **One concept per module, 100-250 lines.** When a file grows past that, split it on a
   concept boundary. `tests/test_repo_hygiene.py` enforces a line ceiling and keeps a
   short, explicit allowlist of known exceptions. Adding to that allowlist needs a
   reason in the PR.
4. **Tests with every module.** A new `frictionlab/x/y.py` comes with `tests/test_y.py`
   in the same PR. Every phase ends with a runnable `examples/` script, a regression
   test that pins its seeded headline numbers, and a `docs/` write-up linked from
   `docs/index.md`.
5. **Public and generic.** Do not commit personal data, credentials, private hostnames or
   absolute home-directory paths. The hygiene test scans every tracked text file.

## Running tests and benchmarks locally

```bash
python -m pip install -e ".[test,yaml]"     # yaml is optional (YAML configs)
python -m pytest -q                         # full suite, no network
python -m pytest -q tests/test_order_book_equivalence.py   # one module

frictionlab bench --size small --compare benchmarks/baseline_small.json
frictionlab bench --size medium --scenario batch_auction_clear --json /tmp/after.json
frictionlab bench --size medium --compare /tmp/before.json          # speedup ratios
for c in examples/configs/*.json; do python -m frictionlab validate "$c"; done
```

CI (`ci/ci.yml`) runs exactly these steps on Python 3.10-3.12. It also
runs an optional job with `numba` installed and `FRICTIONLAB_NUMBA=1` that reruns the
order book oracle tests. The bench step is a **checksum gate only**: CI never asserts a
timing, because timings depend on the machine and checksums do not.

## Performance PRs

The fl-038 harness hashes results (trade count, volume, final ledger balances), not
timings. A performance PR must:

1. keep every checksum in `benchmarks/baseline_small.json` unchanged. A PR that moves a
   checksum is a behaviour change and has to be reviewed as one;
2. freeze the pre-change implementation verbatim as an oracle under
   `tests/reference_*.py` (see `reference_order_book.py`, `reference_fees.py`,
   `reference_batch_auction.py`), and add a seeded randomized equivalence test that
   compares every output against it, edge cases included (empty, one-sided, ties,
   rounding boundaries);
3. put a before/after `frictionlab bench` table (small and medium) in
   `docs/phase6-performance.md`;
4. keep optional accelerators optional. When `numba` is missing, the code falls back
   silently and gives identical results.

You may add a new bench scenario with a new checksum. Existing checksums must never move.

## Adding an experiment (fl-036 registry)

1. Write `frictionlab/experiments/<name>.py` with a frozen `Config` dataclass and a
   `run_<name>(cfg, ..., master_seed=, n_workers=) -> ExperimentResult` built on
   `experiments/base.py` (`run_experiment` over the sweep engine).
2. Register it in `frictionlab/experiments/registry.py`:

   ```python
   register_experiment(
       "my_exp", lambda p, s, r, w: my_exp.run_my_exp(_build(my_exp.Config, p),
                                                      replicates=r, master_seed=s, n_workers=w),
       {"levels": Param("int", seq=True, friction=True), "horizon": POS},  # param schema
       _pick(my_exp.Config(), "horizon", levels=[0, 10, 20]),            # defaults
       small={"levels": [0, 10]}, master_seed=40, replicates=3, small_replicates=1,
       doc="one-line description shown by `frictionlab list`")
   ```

   The defaults must cover the schema exactly, and the small preset must validate.
   Registration checks both.
3. Ship `examples/configs/<name>_small.json`, `examples/phase<N>_<name>.py` with a
   `--small` flag, and `docs/phase<N>-<name>.md`. Link the doc from `docs/index.md` and
   `ROADMAP.md`. The hygiene test fails if a ROADMAP doc link is dangling or a phase doc
   is missing from the index.
4. Add tests: `run_from_config` rows equal the direct call, seeded headline numbers are
   pinned, and the small config validates through the CLI.

## Commit messages

Use `type(scope): summary`, written in the imperative and focused on what changed and why:

- types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `ci`, `chore`
- scopes: a package or area, e.g. `core`, `costs`, `markets`, `agents`, `analysis`,
  `experiments`, `config`, `cli`, `bench`, `repo`

```text
perf(markets): Replace the per-price volume scan with a sorted cumulative sweep
feat(experiments): Add a latency-arbitrage experiment to the registry
ci(repo): Add CI and a contribution guide
```

The body names the tests that prove the change and, for `perf`, confirms that the
checksums are unchanged. Each commit covers one milestone or concern, with the
ROADMAP tick in the same commit.
