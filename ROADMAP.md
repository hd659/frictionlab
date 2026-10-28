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
- [ ] Bounded-rational agent: pays a cost per deliberation step (satisficing vs optimizing)

## Phase 3: Market structures
- [ ] Constant-product AMM and concentrated-liquidity AMM (port `amm_v3_legacy.py`)
- [ ] Perpetual futures with funding rate
- [ ] Cross-venue arbitrageur that only trades when net edge > total friction
- [ ] Batch auction vs continuous book comparison

## Phase 4: Analysis toolkit
- [ ] Metrics: realized spread, price impact, price efficiency, welfare / surplus, PnL by agent type
- [ ] Parameter sweeps over friction levels with parallel runs
- [ ] Equilibrium detection (strategy mix stabilization, replicator dynamics)
- [ ] Plots and HTML report generator

## Phase 5: Experiments (each one a docs/ article + example)
- [ ] Coase threshold: at what transaction cost does trading collapse?
- [ ] Decision cost and over-analysis: when does a satisficer beat an optimizer?
- [ ] Adverse selection: how informed flow widens spreads
- [ ] Fee tier wars between two venues
- [ ] Funding-rate arbitrage under gas shocks

## Phase 6: Ecosystem
- [ ] CLI (`frictionlab run config.yaml`)
- [ ] Config schema and experiment registry
- [ ] Benchmarks and performance work (vectorized book, optional numba)
- [ ] Documentation site, contribution guide, CI

## Ongoing maintenance (after Phase 6, loop forever)
- Refactor hot spots, raise coverage, add new agent archetypes and experiments, fix issues found by running examples.
