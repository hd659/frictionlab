# frictionlab ROADMAP

frictionlab is an agent-based simulator for markets with friction. Its question: how do transaction costs, information asymmetry and decision costs change who trades, what prices form, and which strategies survive?

Most toy market simulators assume frictionless matching. Real markets are defined by their frictions: fees, spreads, latency, gas, funding, search costs, the cost of thinking. frictionlab treats friction as the first-class object and lets you watch equilibria shift as you turn each friction knob.

Design principles
- Pure Python core, numpy allowed, no network on the hot path; every experiment is reproducible from a seed.
- Every module ships with pytest tests; every phase ends with a runnable example in `examples/` and a short write-up in `docs/`.
- Small files (100-250 lines), one concept per module.
- Public, generic, no personal data.

## Phase 1: Core engine (foundation)
- [x] Event-driven simulation clock, seeded RNG, event queue
- [x] Order / Trade / Account primitives with integer-tick prices
- [x] Continuous limit order book (price-time priority, partial fills, cancels)
- [x] Friction stack interface: maker/taker fees, half-spread, latency, fixed per-trade cost
- [x] Ledger with invariant checks (cash + inventory conservation)

## Phase 2: Agents
- [x] Agent base class: observe -> decide -> act, with an explicit `decision_cost`
- [x] Zero-intelligence traders (Gode-Sunder baseline)
- [x] Market maker with inventory risk (Avellaneda-Stoikov style)
- [x] Informed trader + noise traders (Kyle 1985 / Glosten-Milgrom)
- [x] Bayesian learner that updates beliefs from order flow
- [x] Bounded-rational agent: pays a cost per deliberation step (satisficing vs optimizing)
- [x] Phase 2 examples: ZI-C efficiency under fees, adverse selection, decision cost (`examples/phase2_*.py`)
- [x] Phase 2 write-up (`docs/phase2-agents.md`) and PnL breakdown helper (`frictionlab/analysis/pnl.py`)

Phase 2 status: complete. Seeded examples show informed flow widening spreads and a
step-cost threshold above which satisficers out-earn optimizers net of deliberation cost.

## Phase 3: Market structures
- [x] Constant-product AMM and concentrated-liquidity AMM (port `amm_v3_legacy.py` -> `markets/amm_cl.py`)
- [x] Perpetual futures with funding rate
- [x] Cross-venue arbitrageur that only trades when net edge > total friction (`agents/arbitrageur.py`)
- [x] Batch auction vs continuous book comparison (`markets/batch_auction.py`, `analysis/structure_compare.py`)
- [x] Phase 3 example: book vs batch under latency, arbitrage band vs gas, perp funding (`examples/phase3_market_structures.py`)
- [x] Phase 3 write-up (`docs/phase3-market-structures.md`)

Phase 3 status: complete. Seeded runs show the batch auction cutting the sniping share
at high latency, the no-arbitrage band widening with gas (more slowly for concentrated
liquidity), and funding holding a perp's mark near index against one-sided flow.

## Phase 4: Analysis toolkit
- [x] Metrics: realized spread, price impact, price efficiency, welfare / surplus, PnL by agent type (`analysis/metrics.py`, `analysis/welfare.py`)
- [x] Parameter sweeps over friction levels with parallel runs (`analysis/sweep.py`)
- [x] Equilibrium detection (strategy mix stabilization, replicator dynamics) (`analysis/equilibrium.py`)
- [x] Plots and HTML report generator (`analysis/plots.py`, `analysis/report.py`)
- [x] Phase 4 example: fee x latency sweep, satisficer-vs-optimizer replicator, one HTML report (`examples/phase4_analysis_toolkit.py`)
- [x] Phase 4 write-up (`docs/phase4-analysis-toolkit.md`)

Phase 4 status: complete. One seeded run sweeps taker fee x latency over a ZI-C + maker +
informed + noise market (effective = realized + impact on every trade, books close to zero,
welfare non-increasing in the fee), finds the satisficer fixed point at high cost_per_step,
and writes a byte-reproducible offline HTML report.

## Phase 5: Experiments (each one a docs/ article + example)
- [x] Experiment scaffold: `ExperimentResult` + `run_experiment` over the fl-023 sweep engine, report-ready (`experiments/base.py`)
- [x] Coase threshold: at what transaction cost does trading collapse? (`experiments/coase.py`, `examples/phase5_coase_threshold.py`, `docs/phase5-coase-threshold.md`)
- [x] Decision cost and over-analysis: when does a satisficer beat an optimizer? (`experiments/decision_cost.py`, `examples/phase5_decision_cost.py`, `docs/phase5-decision-cost.md`; Phase 2 seed: `examples/phase2_decision_cost.py`)
- [x] Adverse selection: how informed flow widens spreads, vs the Glosten-Milgrom benchmark, and batch-auction mitigation (`experiments/adverse_selection.py`, `examples/phase5_adverse_selection.py`, `docs/phase5-adverse-selection.md`; Phase 2 seed: `examples/phase2_adverse_selection.py`)
- [x] Fee tier wars between two venues: Bertrand undercutting under a shared router, makers priced by rebate net of adverse selection (`experiments/fee_wars.py`, `experiments/two_venue.py`, `examples/phase5_fee_wars.py`, `docs/phase5-fee-wars.md`; building blocks: `markets/fee_tiers.py`, `markets/router.py`)
- [x] Funding-rate arbitrage under gas shocks: basis band vs gas, two-regime shocks (`experiments/funding_arb.py`, `examples/phase5_funding_arb.py`, `docs/phase5-funding-arb.md`; building blocks: `costs/gas.py`, `agents/funding_arb.py`)
- [x] Phase 5 summary report: all five experiments in one offline HTML (`examples/phase5_summary_report.py`)

Phase 5 status: complete. Five seeded experiments, each one an `ExperimentResult`,
an example with `--small` and a write-up; `examples/phase5_summary_report.py` reruns all
five into one byte-reproducible HTML.
- Coase threshold (seed 28): volume falls one unit at a time as F = 2*fixed + taker passes
  each binding pair gap; half-volume collapse at F* = 66 ticks, zero above the 100-tick bound.
- Decision cost (seed 29): net PnL is linear in cost_per_step; satisficer wins iff
  dG < dS * c, break-even c* = 9 / 5 / 5 at aspiration 100 / 500 / 2000, matching bisection.
- Adverse selection (seed 30): GM spread 2 * mu * L; simulated spread tracks it plus a
  ~3-tick inventory term; at mu = 0.5, J = 8 the batch auction cuts AS loss per unit 7.40 -> 3.38.
- Fee tier wars (seed 32): undercutting from (0, 4) to a pure fixed point at (-2, 3) on both
  venues by round 6; trader friction 4 -> 1 tick per lot, share_A = 0.47.
- Funding arbitrage (seed 34): entries 12 -> 4.5 and mean |premium| 27.3 -> 43.3 bps from
  constant gas to shock gas 116 at p = 0.6 (pi_shock = 0.92); premium non-decreasing in shock
  gas at every p; bisected collapse at 52 ticks vs the 115-tick analytic per-leg bound;
  myopic calm-state entries held through shocks turn net arb PnL from +914 to -699 ticks.

## Phase 6: Ecosystem
- [x] CLI (`frictionlab run config.yaml`): stdlib argparse, `run` / `list` / `validate` / `show-config`; overrides file < `--small`/`--seed`/`--workers` < `--set key.path=value`; `run` writes `rows.csv`, `resolved_config.json`, `report.html`; exit 0 / 2 config-usage / 1 runtime; console script and `python -m frictionlab` (`cli.py`, `__main__.py`, `examples/configs/fee_wars_small.json`)
- [x] Config schema and experiment registry: typed `ExperimentConfig` + per-experiment `Param` schemas, JSON (YAML with PyYAML), all errors reported at once with dotted paths; registry of the five Phase 5 experiments with defaults, small presets and `run_from_config` (`config/schema.py`, `experiments/registry.py`, `examples/configs/*.json`)
- [x] Benchmarks and performance work (vectorized book, optional numba)
  - [x] Benchmark harness: five seeded scenarios (book flow, runner market, batch auction, AMM swaps, Coase sweep) at small/medium, warmup + repeats, min/median/max and ops/sec, deterministic work checksum, `compare` with speedup ratio; `frictionlab bench`; checksum-only `benchmarks/baseline_small.json` (`bench/`)
  - [x] Performance work measured against the baseline checksums
    - [x] Vectorized book: integer-tick dict-of-deque levels, heapq best-price index with lazy level deletion, O(1) cancel by order id, `depth_array` int64 snapshots, optional numba kernel behind `FRICTIONLAB_NUMBA` with numpy fallback; pre-change book frozen as the oracle, all fl-038 checksums unchanged (`core/order_book.py`, `core/book_levels.py`, `tests/reference_order_book.py`, `docs/phase6-performance.md`)
    - [x] Ledger / fee-stack hot path: integer linear-term fast path folded once per stack (exact ceil division, `Fraction` fallback for non-integer rates), `type(x) is int` guard with identical errors, per-trade conservation check with per-leg strict mode behind `FRICTIONLAB_STRICT_LEDGER`; pre-change fees frozen as `tests/reference_fees.py`, all fl-038 checksums unchanged (`costs/fees.py`, `costs/stack.py`, `core/ledger.py`)
    - [x] Batch auction clear: one aggregation pass into integer-tick demand / supply maps, single sorted running-sum sweep over change points with the same tie-break order (max volume, min imbalance, closest to previous price, lower price), priority sort only for executable orders, unchanged pro-rata rounding; pre-change clear frozen as `tests/reference_batch_auction.py`, all fl-038 checksums unchanged, medium `batch_auction_clear` 3.9x (`markets/batch_auction.py`, `bench/scenarios.py` `DEEP_BATCH`)
- [ ] Documentation site, contribution guide, CI
  - [x] CI: GitHub Actions on push / pull_request over Python 3.10-3.12 with pip caching; pytest, `frictionlab bench --size small --compare benchmarks/baseline_small.json` as a checksum-only gate (no timing thresholds), `python -m frictionlab validate` over every `examples/configs/*.json`; optional numba job (`FRICTIONLAB_NUMBA=1`) rerunning the order book oracle tests (`ci/ci.yml`)
  - [x] Contribution guide: design principles (seeded reproducibility, integer ticks, one concept per 100-250 line module, tests with every module), local test / bench workflow, performance PR rule (checksums unchanged, oracle frozen under `tests/reference_*.py`), adding an experiment through the fl-036 registry, `type(scope): summary` commits (`CONTRIBUTING.md`)
  - [x] Repo hygiene tests: workflow wiring (YAML parsed when PyYAML is present), ROADMAP doc links exist, `docs/index.md` links every `docs/phase*.md`, example configs validate, module line ceiling with an explicit capped allowlist, no personal URL hosts or absolute home paths in tracked text (`tests/test_repo_hygiene.py`)
  - [x] Documentation site: seeded by `docs/index.md`, which links every phase write-up in order; static site build and API reference
    - [x] Static site build: stdlib Markdown subset (slugged headings, nested lists, fenced code, pipe tables, all text escaped), page order from `docs/index.md`, embedded-CSS layout with sidebar nav, prev/next and h2/h3 TOC, `.md` -> `.html` link rewriting, byte-reproducible output, `frictionlab docs build [--src docs] [--out DIR]` (`frictionlab/docs_site/`)
    - [x] API reference: per-package pages under `api/` from `__all__` (or own re-exports), address-free normalized signatures, docstrings, class members, qualified-name anchors and cross-references, `api/index.html` with symbol counts, `--api/--no-api` build flag, `api_coverage` reporting no undocumented public symbol (`frictionlab/docs_site/api_ref.py`)

Phase 6 status: in progress (benchmarks and performance item done; CI, contribution guide
and repo hygiene tests are in; the documentation site, including the generated API reference, is done). The config schema, registry and CLI are in: a JSON config
(`examples/configs/coase_small.json`) resolves deterministically (defaults, then small
preset, then user params), runs any of the five Phase 5 experiments with rows identical to
the direct call, and records the resolved config in the fl-026 report header.
`frictionlab run CONFIG --out DIR` writes byte-reproducible artifacts for a fixed seed;
`validate` lists every bad dotted path in one pass. The benchmark harness is in:
`frictionlab bench --compare benchmarks/baseline_small.json` fails on any checksum drift, so
optimizations must keep results identical. Performance work is closed out against those
checksums, each step checked against a frozen pre-change oracle: the vectorized book
(1.20-1.32x book-only), the ledger / fee-stack fast path (fees 10-16x, settlement 5-6x,
`book_random_flow` / `amm_swap_stream` about 1.4x end to end) and the batch auction clear
(a sorted sweep over aggregated ticks, `batch_auction_clear` 2.6x small / 3.9x medium).
End-to-end time is now dominated by agent decision code and `MarketView` construction in the
runner scenarios, and by order construction / validation in the book and AMM streams
(`docs/phase6-performance.md`). CI now runs the test suite, the checksum gate and config
validation on Python 3.10-3.12 for every push, and `CONTRIBUTING.md` writes down the rules
the earlier phases followed. `frictionlab docs build` now renders the pages linked from `docs/index.md` into a
byte-reproducible static site with no JavaScript or external assets, plus an API reference
for every public symbol; `api_coverage` keeps public docstrings complete, and the fl-038
checksums confirm the docstring pass changed no behavior.

## Ongoing maintenance (after Phase 6, loop forever)
- Refactor hot spots, raise coverage, add new agent archetypes and experiments, fix issues found by running examples.
