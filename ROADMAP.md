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
- [x] Documentation site, contribution guide, CI
  - [x] CI: GitHub Actions on push / pull_request over Python 3.10-3.12 with pip caching; pytest, `frictionlab bench --size small --compare benchmarks/baseline_small.json` as a checksum-only gate (no timing thresholds), `python -m frictionlab validate` over every `examples/configs/*.json`; optional numba job (`FRICTIONLAB_NUMBA=1`) rerunning the order book oracle tests (`ci/ci.yml`)
  - [x] Contribution guide: design principles (seeded reproducibility, integer ticks, one concept per 100-250 line module, tests with every module), local test / bench workflow, performance PR rule (checksums unchanged, oracle frozen under `tests/reference_*.py`), adding an experiment through the fl-036 registry, `type(scope): summary` commits (`CONTRIBUTING.md`)
  - [x] Repo hygiene tests: workflow wiring (YAML parsed when PyYAML is present), ROADMAP doc links exist, `docs/index.md` links every `docs/phase*.md`, example configs validate, module line ceiling with an explicit capped allowlist, no personal URL hosts or absolute home paths in tracked text (`tests/test_repo_hygiene.py`)
  - [x] Documentation site: seeded by `docs/index.md`, which links every phase write-up in order; static site build and API reference
    - [x] Static site build: stdlib Markdown subset (slugged headings, nested lists, fenced code, pipe tables, all text escaped), page order from `docs/index.md`, embedded-CSS layout with sidebar nav, prev/next and h2/h3 TOC, `.md` -> `.html` link rewriting, byte-reproducible output, `frictionlab docs build [--src docs] [--out DIR]` (`frictionlab/docs_site/`)
    - [x] API reference: per-package pages under `api/` from `__all__` (or own re-exports), address-free normalized signatures, docstrings, class members, qualified-name anchors and cross-references, `api/index.html` with symbol counts, `--api/--no-api` build flag, `api_coverage` reporting no undocumented public symbol (`frictionlab/docs_site/api_ref.py`)
    - [x] Link check and wrap-up: `frictionlab docs check DIR` parses every built page with `html.parser`, resolves each internal href to an output file and each `#fragment` to an element id, reports all broken links in one pass, flags any http(s) link outside an empty-by-default allowlist, exits 2 on any break; CI `docs` job builds and checks `_site` (no deploy, no network); `examples/phase6_ecosystem.py` runs validate / run / bench gate / docs build / docs check into one directory with a manifest (`frictionlab/docs_site/linkcheck.py`, `docs/phase6-ecosystem.md`)

Phase 6 status: complete. The simulator now has an ecosystem around it, and every piece
is held in place by a test or a CI gate (`docs/phase6-ecosystem.md`).
- Config and registry: JSON / YAML configs resolve deterministically (defaults < small
  preset < file < flags < `--set`), and every bad dotted path is reported in one pass.
  The five Phase 5 experiments run from config with rows identical to the direct call.
- CLI: `frictionlab run / list / validate / show-config / bench / docs build / docs check`
  with exit codes 0 / 1 runtime / 2 config-usage; `run` artifacts are byte-reproducible.
- Benchmarks and performance: five seeded scenarios and a checksum-only baseline; vectorized
  book (1.20-1.32x book-only), ledger / fee fast path (fees 10-16x, settlement 5-6x) and
  batch auction sweep (3.9x medium), each against a frozen `tests/reference_*.py` oracle
  with every checksum unchanged (`docs/phase6-performance.md`).
- CI and contribution guide: tests, checksum gate and config validation on Python
  3.10-3.12, an optional numba oracle job, and a `docs` job that builds the site and
  link-checks it; `CONTRIBUTING.md` records the rules; repo hygiene tests guard the wiring.
- Documentation: a stdlib, byte-reproducible static site from `docs/index.md` with a
  generated API reference (no undocumented public symbol) and a link checker that fails on
  any missing page, missing anchor or external URL. `examples/phase6_ecosystem.py --small`
  runs the whole chain into one directory.

## Phase 7: Search and perishability frictions
- [x] Decentralized bilateral search venue: unit buyers / sellers with integer-tick values meet pairwise on the clock via a seeded Poisson matching process (`meeting_rate`); each meeting charges both sides a per-search cost priced by the friction stack and moved to a search-cost sink (`SEARCH_SINK_ID`); a pair trades one lot iff reservations cross, at an integer Nash split (buyer weight in bps, remainder to the seller); optional perishability decays an unsold seller's value by fixed ticks per interval down to a floor; `ReservationPolicy` is `fixed` or the myopic optimal stopping rule against a known uniform counterparty; fills reduce to `TradeRecord` (venue "search") and settle through the ledger; frictionless runs reach the competitive-equilibrium quantity (`markets/search_market.py`)
- [x] Search-cost experiment: volume, realized surplus and time-to-trade vs per-search cost and meeting rate, against the Diamond-paradox and frictionless CE benchmarks, with an example and write-up (`experiments/search_cost.py`, registered as `search_cost`, config `examples/configs/search_cost_small.json`)
  - [x] Seeded example `examples/phase7_search_cost.py --small`: runs the config through the CLI, prints the search-cost table, collapse thresholds and a decay view, and renders one offline report (surplus-share heatmap plus surplus split); `examples/phase5_summary_report.py --include-phase7` adds a sixth section
  - [x] Write-up `docs/phase7-search-cost.md`: reservation-price stopping rule, search cost vs decay, collapse bound vs the fl-028 Coase threshold; linked from `docs/index.md` and link-checked in the docs site
- [x] Middleman venue (Rubinstein-Wolinsky intermediation): an optional `Middleman` agent (`agents/middleman.py`) plugs into `SearchMarket` and meets traders at `middleman_meeting_rate` from the same seeded matching process; it posts integer-tick bid / ask quotes, holds at most `capacity` units, and pays a per-unit holding cost and perishable decay (down to a floor) through friction stacks to hold / decay sinks; a trader fills only when the quote meets its reservation, else keeps searching; fills reduce to `TradeRecord` with venue "middleman" and settle through the ledger; `analysis.middleman_pnl` splits middleman PnL into spread capture, holding cost and decay loss; `middleman=None` reproduces fl-046 runs bit-for-bit
- [x] Intermediation experiment and write-up: middleman share of volume, spread and welfare vs bilateral search cost, meeting rates, capacity and decay, with a seeded example and `docs/` write-up (`experiments/intermediation.py`, registered as `intermediation`, config `examples/configs/intermediation_small.json`)
  - [x] Seeded example `examples/phase7_intermediation.py --small`: runs the config through the CLI, prints the intermediated-share and surplus table over search cost x half-spread, the analytic and bisected thresholds and a middleman PnL view (spread vs holding cost and decay), and renders one offline report (intermediated-share heatmap plus a six-way surplus split); `examples/phase5_summary_report.py --include-phase7` adds it as a seventh section
  - [x] Write-up `docs/phase7-intermediation.md`: rent only when the middleman shortens search by more than its spread, perishable carry caps inventory, relation to the fl-047 collapse bound and the fl-028 Coase threshold; linked from `docs/index.md` and link-checked in the docs site
- [ ] Perishability and endogenous reservation dynamics: reservation prices that respond to elapsed time and decay (sequential optimal stopping with a deadline), welfare loss from spoilage vs search cost, registered in the experiment registry with a small preset

Phase 7 status: in progress (2 of 3) core items done, plus the middleman extension (4 of 5 items ticked). On seed 47 (small preset) volume falls 6 -> 3 -> 0 and surplus share 1.00 -> 0.80 -> 0.00 as search cost goes 0 -> 30 -> 52; the bisected half-volume collapse is 33 ticks against an analytic bound of 51; at search cost 10, decay 0 -> 8 lowers the mean price 70.00 -> 61.33 and the time to match 21.89 -> 4.67. The intermediation extension (fl-049 to fl-051) is complete: on seed 50 (small preset) the intermediated share goes 0.00 -> 0.83 once search cost covers the half-spread, with bisected thresholds 0 / 3 / 11 ticks for half-spreads 0 / 4 / 16, each equal to its analytic bound; the middleman nets 32 ticks at half-spread 4 and 152 at 16 (search cost 32), cuts mean time to match 8.22 -> 2.77 at no change in realized surplus, and at half-spread 4 per-step carry 0 -> 8 takes its net 40 -> -24 (break-even carry 4 on the view grid). Endogenous reservation dynamics come next.

## Phase 8: Costly information and the Grossman-Stiglitz paradox
- [x] Priced signal source and acquisition agent: `SignalSource` (`costs/information.py`) sells noisy signals of a two-point state (`V >= center` from the existing `FundamentalProcess`; `two_point` builds the seeded center +- jump fundamental) with integer `accuracy_bps` (5000 = uninformative, 10000 = perfect), drawn from a per-agent stream seeded from (seed, agent_id) so purchases never move other agents' draws; the integer-tick price is a friction-stack charge moved by `Ledger.transfer` to an information sink (`INFO_SINK_ID`); `InfoAcquirer` (`agents/info_acquirer.py`) buys iff the floored expected edge `(2 * accuracy_bps - 10000) * jump / 10000 - spread / 2` exceeds signal cost + decision cost (ties do not buy), then trades the signal like the fl-018 informed trader, else behaves as an uninformed noise trader (bit-identical to a `NoiseTrader` at 5000 bps); `analysis.info_pnl` splits PnL into trading gain, signal cost and decision cost, exact in ticks; runs without acquirers keep every fl-038 checksum
- [x] Grossman-Stiglitz experiment: equilibrium informed fraction and price informativeness vs signal cost, accuracy and noise-trader volume, with an analytic indifference cost checked by integer bisection, registered in the experiment registry with a small preset (`experiments/grossman_stiglitz.py`: `GSMaker` quoting posterior mid +- half the fl-030 GM spread at the live informed share, `gs_equilibrium_informed_share`, fl-024 replicator over committed buyers vs non-buyers, `find_information_collapse`; registered as `grossman_stiglitz`, config `examples/configs/grossman_stiglitz_small.json`)
- [x] Seeded example and write-up: `examples/phase8_grossman_stiglitz.py --small` plus a `docs/` page on the paradox (if prices were fully revealing nobody would pay for information), linked from `docs/index.md` and link-checked in the docs site
  - [x] Seeded example `examples/phase8_grossman_stiglitz.py --small`: runs the config through the CLI, prints the buyer-fraction / spread / price-efficiency table over signal cost x noise intensity, the replicator vs analytic informed shares and the bisected collapse cost, and renders one offline report (buyer-fraction heatmap, replicator buyer-share trajectory with its fixed point, buyer PnL split into trading gain, signal cost and decision cost); `examples/phase5_summary_report.py --include-phase8` adds it as one more section
  - [x] Write-up `docs/phase8-grossman-stiglitz.md`: why fully revealing prices are not an equilibrium under costly information, noise trading pays for information, the GM spread closes the loop, and the collapse vs the fl-028 Coase threshold and the fl-047 search collapse bound; linked from `docs/index.md` and link-checked in the docs site

Phase 8 status: complete (3 of 3). On seed 53 (small preset) the buyer fraction falls 0.99 -> 0.74 -> 0.48 -> 0.00 at noise 4 as signal cost goes 0 -> 2 -> 4 -> 8, against analytic informed shares 0.80 / 0.60 / 0.40 / 0.00; the quoted spread narrows 13.48 -> 2.00 and the price-efficiency error jumps 0.22 -> 10.00 (the full jump) once nobody buys; the bisected information collapse is 7 ticks against an analytic 8; the replicator ends at a buyer share of 0.72 at cost 0 (analytic 0.80) and goes extinct by generation 3 at costs 4 and 9; over the sweep buyers gain -9936 ticks trading and pay 1854 for signals (net -11790). On the seeded 99 x 101 test market (jump 10) the expected edge per lot is 9 / 7 / -1 ticks at accuracy 10000 / 9000 / 5000: a 9000 bps signal priced at 2 ticks with decision cost 1 is bought 156 times (hit rate 0.88), at 6 ticks it is never bought, and at 5000 bps it is never bought at any positive price.

## Phase 9: Quality uncertainty (lemons, certification and reputation)
- [x] Hidden-quality market and priced certification friction: `LemonsMarket` (`markets/lemons.py`) gives each seller one unit with hidden integer-tick quality `q` drawn from a seeded discrete schedule (`r(q) = q`); buyers value a unit at `q + premium` and observe only the offered pool, posting a uniform price equal to `floor(mean(q)) + premium` until the fixed point repeats or the pool is empty; `CertificationPolicy` (`costs/certification.py`) lets a seller pay an integer-tick cost through the friction stack (routed to `CERT_SINK_ID`) to reveal `q` and clear in a separate certified segment at `q + premium` iff `q + premium - c(q) >= max(p, q)`; fills reduce to `TradeRecord` tagged `"pooled"` or `"certified"` and settle through the ledger; `analysis.lemons_pnl` splits seller PnL into trade surplus and certification cost, exact in ticks
- [x] Lemons and certification experiment (`experiments/lemons.py`): traded share (overall and by quality quintile), pooled price and rounds, certification uptake and cutoff, surplus share of the full-information maximum and certification paid over certification cost x buyer premium; `akerlof_pooled_price` / `certification_cutoff` give the analytic fixed point and cutoff, `find_certification_threshold` bisects the uptake-collapse cost against the bound `min(premium, q_top - q_min) + 1`; registered as `lemons` with `examples/configs/lemons_small.json`
- [x] Seeded example and write-up: `examples/phase9_lemons.py --small` plus a `docs/` page on Akerlof unraveling and separating certification, linked from `docs/index.md` and link-checked in the docs site
  - [x] Seeded example `examples/phase9_lemons.py --small`: runs the config through the CLI, prints the traded-share / pooled-price / surplus-share table over certification cost x buyer premium, the zero-premium unraveling path and the analytic and bisected certification thresholds, and renders one offline report (surplus-share heatmap, pooled-price fixed-point line chart, surplus split into buyer gain, pooled-seller gain, certified-seller gain and the certification sink); `examples/phase5_summary_report.py --include-phase9` adds it as one more section
  - [x] Write-up `docs/phase9-lemons.md`: unraveling from the top, certification as a separating cutoff whose cost is deadweight friction, and the certification bound vs the fl-053 information collapse and the fl-028 Coase threshold; linked from `docs/index.md` and link-checked in the docs site

Phase 9 status: complete (3 of 3). On seed 56 (small preset, 20 sellers) the zero-premium pooled price unravels 57 -> 30 -> 20 -> 18 -> 10 while the pool shrinks 20 -> 9 -> 6 -> 5 -> 1, so only the lowest-quality unit trades (traded share 0.05); the Akerlof fixed points are 10 / 30 / 95 at premiums 0 / 10 / 40; at premium 40 the certification cutoff rises 30 -> 50 -> 80 as cost goes 4 -> 16 -> 32; surplus share at cost 32 is 0.00 / 0.10 / 0.59 over the premiums; the bisected certification threshold at premium 20 is 21 ticks against an analytic bound of 21; over the sweep buyers gain 7 ticks, pooled sellers 1123, certified sellers 2590 and 568 go to the certification sink.

## Ongoing maintenance (after Phase 9, loop forever)
- Refactor hot spots, raise coverage, add new agent archetypes and experiments, fix issues found by running examples.
