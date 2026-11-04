# Phase 6: the ecosystem around the simulator

Phase 6 added no new market mechanism. It is the tooling that makes the Phase 1-5 results
easy to rerun, hard to break and easy to read. That means a config schema with an experiment
registry, a CLI, a checksum-gated benchmark harness, CI, and a static documentation site
with a generated API reference and a link checker. This page explains how those pieces fit
together. The performance work they guarded is covered in
[Phase 6: performance work](phase6-performance.md).

Example: `python examples/phase6_ecosystem.py --small [OUT]` runs the whole chain
(validate, run, bench gate, docs build, docs check) through `frictionlab.cli.main(argv)`
into one directory and prints an artifact manifest. Two runs give byte-identical outputs.

## Config precedence

An experiment config is JSON (or YAML, when PyYAML is installed). It is resolved
deterministically. Each layer overrides the one before it:

1. the experiment's registered defaults;
2. its `small` preset, when `small` is true;
3. the config file's own `params` and top-level keys;
4. CLI flags: `--small`, `--seed N`, `--workers N`;
5. `--set key.path=value`, applied left to right. Values are parsed as JSON when they
   parse, and kept as strings otherwise.

`--out` only chooses where artifacts go. It is never written into `resolved_config.json`,
so the same config and seed give identical bytes in any directory.
`frictionlab show-config` prints exactly what will run. Validation reports every bad key
in one pass, each with its dotted path (`params.levels[2]`), and an unknown experiment name
gets a close-match suggestion.

## The experiment registry

`frictionlab/experiments/registry.py` maps a name to a runner, a typed `Param` schema,
defaults, a small preset, a master seed and replicate counts. Registration checks that the
defaults cover the schema exactly and that the small preset validates, so a broken
experiment fails at import time and not halfway through a run. `run_from_config` returns
the same `ExperimentResult` as the direct call, with identical rows, and records the
resolved config in the report header. The steps for adding an experiment are in
[CONTRIBUTING](../CONTRIBUTING.md#adding-an-experiment-fl-036-registry).

## CLI exit codes

Every subcommand follows one convention, so CI and scripts can branch on the code alone:

| code | meaning | examples |
|-----:|---------|----------|
| 0 | success | `run` wrote its artifacts; `bench --compare` matched; `docs check` found no broken link |
| 1 | runtime error | an exception inside a run; a benchmark checksum mismatch |
| 2 | config or usage error | bad config key, unknown experiment, bad `--set`, missing docs page, broken docs link |

Errors go to stderr. Tracebacks are printed only with `--debug`. `main(argv)` returns
the code instead of exiting, which is how the tests and the Phase 6 example call it.

## The checksum-only bench gate

`frictionlab bench` times five seeded scenarios. For each one it also hashes the work done
(trades, volume, final ledger balances). `benchmarks/baseline_small.json` stores only those
checksums, never timings. `bench --compare` exits 1 if any checksum differs, and it never
fails on speed. Timings depend on the machine and checksums do not, so the gate gives the
same answer on a laptop and on a CI runner. Any change that moves a checksum is a behaviour
change, even if it is labelled as a speedup.

## The oracle rule for performance PRs

A performance PR must leave every baseline checksum unchanged. It must also freeze the
pre-change implementation verbatim under `tests/reference_*.py` and add a seeded randomized
equivalence test against it, edge cases included. Phase 6 followed this rule three times:
`reference_order_book.py`, `reference_fees.py` and `reference_batch_auction.py`. The
checksums show that end-to-end results did not move. The oracles go further and check
every intermediate output, so a compensating pair of bugs cannot hide behind an unchanged
total. Optional accelerators such as numba fall back silently when missing, with
identical results.

## How the site and API reference are built

`frictionlab docs build [--src docs] [--out DIR]` is stdlib only:

- `docs/index.md` fixes the page order. The site is the index plus every sibling `.md`
  page it links, in first-link order. A linked page that does not exist is exit 2.
- A small deterministic Markdown subset is rendered with all text HTML-escaped and no raw
  HTML. Headings get stable slug ids, and the subset covers lists, fenced code and pipe
  tables. Links to site pages become `.html`, and links outside the site, such as
  `../README.md`, are kept as written.
- The shared layout embeds its CSS and adds a sidebar nav, prev/next links and an h2/h3
  table of contents. It has no JavaScript, no external assets and no http(s) references.
- The API reference (`api/`, on by default and turned off with `--no-api`) documents each
  package's `__all__`. It shows normalized signatures with no memory addresses, docstrings
  and class members, all under qualified-name anchors. `api_coverage` keeps public
  docstrings complete.
- Pages are written in sorted order with `\n` newlines and no timestamps, so two builds
  are byte-identical.

`frictionlab docs check DIR [--src docs] [--allow PREFIX]` parses every built page with
`html.parser` and reports all broken links in one pass, each with its page and target:

- an internal `href` must resolve to an existing output file, and a `#fragment` must
  name an existing element id on that page;
- a link that leaves the site root (`../README.md#cli-quickstart`) is resolved against the
  docs source directory, and its fragment is checked against the target's heading slugs;
- any http(s) link is reported unless it matches an `--allow` prefix. The allowlist is
  empty by default.

Any broken link exits 2. The `docs` job in `ci/ci.yml` runs
`frictionlab docs build --out _site` and then `frictionlab docs check _site`, with no
deploy step and no network access. `tests/test_repo_hygiene.py` fails if either command
is dropped from the workflow.

## Phase 6 in one paragraph

Any Phase 5 result is now one command away (`frictionlab run CONFIG`), and its artifacts
are byte-reproducible. Performance work is only accepted when the checksums hold and an
oracle agrees. CI runs the tests, the checksum gate, config validation and a link-checked
docs build on Python 3.10-3.12. The documentation, including the API reference, is built
from the repository itself and cannot silently link to a page or anchor that does not exist.
