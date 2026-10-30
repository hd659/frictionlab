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
- [ ] Adverse selection: how informed flow widens spreads
- [ ] Fee tier wars between two venues
- [ ] Funding-rate arbitrage under gas shocks

Phase 5 status: in progress. Coase threshold done: on a fixed 6x6 ZI-C schedule volume
falls one unit at a time as F = 2*fixed + taker passes each binding pair gap, half-volume
collapse at F* = 66 ticks (seed 28), and zero volume above the 100-tick analytic bound.
Decision cost done: net PnL is linear in cost_per_step, so the satisficer wins iff
dG < dS * c; bisection matches floor(dG / dS) + 1 exactly, with break-even c* = 9 / 5 / 5
at aspiration 100 / 500 / 2000 (seed 29, max_steps 40).

## Phase 6: Ecosystem
- [ ] CLI (`frictionlab run config.yaml`)
- [ ] Config schema and experiment registry
- [ ] Benchmarks and performance work (vectorized book, optional numba)
- [ ] Documentation site, contribution guide, CI

## Ongoing maintenance (after Phase 6, loop forever)
- Refactor hot spots, raise coverage, add new agent archetypes and experiments, fix issues found by running examples.
