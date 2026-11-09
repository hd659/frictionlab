# frictionlab documentation

frictionlab is an agent-based simulator for markets with friction: fees, spreads,
latency, gas, funding, information asymmetry and the cost of deciding. This page seeds
the documentation site. It links every phase write-up in roadmap order, and each
write-up comes with a seeded example under `examples/` that reproduces its numbers.

Start with the [README](../README.md) for installation and the CLI quickstart. The
[ROADMAP](../ROADMAP.md) has the plan and status, and [CONTRIBUTING](../CONTRIBUTING.md)
covers design principles, the test and benchmark workflow and commit conventions.

## Phase 1: core engine

- [Phase 1: the core engine, and how per-trade friction shifts realized PnL](phase1-core-engine.md)
  covers the event clock, integer-tick orders, the continuous limit order book, the
  friction stack and the conservation-checked ledger.
  Example: `examples/phase1_friction_walkthrough.py`.

## Phase 2: agents

- [Phase 2: agents, adverse selection and the cost of thinking](phase2-agents.md)
  covers zero-intelligence traders, an inventory-aware market maker, informed and noise
  flow, a Bayesian learner and bounded-rational satisficers.
  Examples: `examples/phase2_zi_efficiency.py`, `examples/phase2_adverse_selection.py`,
  `examples/phase2_decision_cost.py`.

## Phase 3: market structures

- [Phase 3: market structures](phase3-market-structures.md) covers constant-product and
  concentrated-liquidity AMMs, perpetual funding, the cross-venue arbitrageur, and the
  batch auction compared with the continuous book.
  Example: `examples/phase3_market_structures.py`.

## Phase 4: analysis toolkit

- [Phase 4: the analysis toolkit](phase4-analysis-toolkit.md) covers spread, impact and
  welfare metrics, parallel parameter sweeps, equilibrium detection, and the offline
  HTML report.
  Example: `examples/phase4_analysis_toolkit.py`.

## Phase 5: experiments

Each experiment is registered in `frictionlab/experiments/registry.py` and can be run
with `frictionlab run CONFIG`.

1. [Phase 5.1: the Coase threshold](phase5-coase-threshold.md): the transaction
   cost at which trading collapses. Config: `examples/configs/coase_small.json`.
2. [Phase 5.2: decision cost and over-analysis](phase5-decision-cost.md): when a
   satisficer beats an optimizer net of deliberation cost.
3. [Phase 5.3: adverse selection and the Glosten-Milgrom benchmark](phase5-adverse-selection.md):
   how informed flow widens spreads, and how the batch auction mitigates it.
4. [Phase 5.4: fee-tier wars between two venues](phase5-fee-wars.md): Bertrand
   undercutting under a shared router. Config: `examples/configs/fee_wars_small.json`.
5. [Phase 5.5: funding-rate arbitrage under gas shocks](phase5-funding-arb.md): how the
   basis band responds to gas shocks. Config: `examples/configs/funding_arb_small.json`.

`examples/phase5_summary_report.py` reruns all five experiments into a single
byte-reproducible HTML report.

## Phase 6: ecosystem

- [Phase 6: performance work](phase6-performance.md) covers the benchmark harness, the
  vectorized order book, the ledger and fee-stack fast path, and the batch auction sweep.
  Each step was checked against a frozen oracle, and every checksum stayed unchanged.
- [Phase 6: the ecosystem around the simulator](phase6-ecosystem.md) covers config
  precedence, the experiment registry, CLI exit codes, the checksum-only bench gate, the
  oracle rule for performance PRs, and how the docs site, API reference and link check
  are built.
  Example: `examples/phase6_ecosystem.py`.
- Tooling: the CLI quickstart is in the [README](../README.md#cli-quickstart).
  CI and the contribution workflow are in [CONTRIBUTING](../CONTRIBUTING.md).

## Phase 7: search and perishability

- [Phase 7: search costs and perishable value](phase7-search-cost.md) covers the
  reservation-price stopping rule in the bilateral search market, why search cost and
  decay push the market in opposite directions, and how the collapse bound relates to
  the Coase threshold. Config: `examples/configs/search_cost_small.json`.
  Example: `examples/phase7_search_cost.py`.
- [Phase 7: intermediation under search costs](phase7-intermediation.md) covers why a
  middleman earns rent only when it shortens search by more than its spread, how
  perishable carry caps the inventory it will hold, and how this relates to the search
  collapse bound and the Coase threshold.
  Config: `examples/configs/intermediation_small.json`.
  Example: `examples/phase7_intermediation.py`.

`examples/phase5_summary_report.py --include-phase7` adds both Phase 7 experiments to
the summary report, as sections six and seven.

## Phase 8: costly information

- [Phase 8: the Grossman-Stiglitz paradox](phase8-grossman-stiglitz.md) covers why
  perfectly efficient prices cannot be an equilibrium when information is costly, how
  noise trading pays for information production, how the Glosten-Milgrom spread closes
  the loop, and how the information collapse relates to the Coase threshold and the
  search collapse bound.
  Config: `examples/configs/grossman_stiglitz_small.json`.
  Example: `examples/phase8_grossman_stiglitz.py`.

`examples/phase5_summary_report.py --include-phase8` adds the Grossman-Stiglitz
experiment to the summary report as one more section.

## Phase 9: quality uncertainty

- [Phase 9: lemons, unraveling and certification](phase9-lemons.md) covers why hidden
  quality unravels a pooled market from the top, how priced certification creates a
  separating cutoff whose cost is pure deadweight friction, and how the certification
  bound relates to the information collapse and the Coase threshold.
  Config: `examples/configs/lemons_small.json`.
  Example: `examples/phase9_lemons.py`.

`examples/phase5_summary_report.py --include-phase9` adds the lemons experiment to the
summary report as one more section.

## Phase 10: reputation

- [Phase 10: reputation, end-game milking and cheap identities](phase10-reputation.md)
  covers why a finite horizon produces end-game milking, why cheap identities let
  opportunists escape their score, and how the deterrent whitewash cost plays the role
  of the certification bound.
  Config: `examples/configs/reputation_small.json`.
  Example: `examples/phase10_reputation.py`.

`examples/phase5_summary_report.py --include-phase10` adds the reputation experiment to
the summary report as one more section.

## Phase 11: switching costs

- [Phase 11: switching costs, bargain-then-ripoff pricing and lock-in](phase11-switching-cost.md)
  covers why switching costs move competition ex ante, why the harvest markup is bounded
  by the minimum switching cost among locked-in customers, and how the lock-in threshold
  relates to the Coase threshold, the search collapse bound and the whitewash deterrent
  cost.
  Config: `examples/configs/switching_cost_small.json`.
  Example: `examples/phase11_switching_cost.py`.

`examples/phase5_summary_report.py --include-phase11` adds the switching-cost experiment
to the summary report as one more section.

## Phase 12: network effects and critical mass

- [Phase 12: network effects, critical mass and cold start](phase12-critical-mass.md)
  covers why cross-side benefits create multiple equilibria and a cold-start trap, why a
  one-time adoption friction raises the critical mass so a seeding subsidy is the price
  of crossing it, and how the tipping seed relates to the Coase threshold and the
  lock-in threshold.
  Config: `examples/configs/critical_mass_small.json`.
  Example: `examples/phase12_critical_mass.py`.

`examples/phase5_summary_report.py --include-phase12` adds the critical-mass experiment
to the summary report as one more section.

## Phase 13: congestion and queueing costs

- [Phase 13: congestion and queueing costs](phase13-congestion.md) covers why
  self-interested arrivals over-join an observable queue, why a toll equal to the
  waiting cost they impose on later arrivals turns a congestion friction into a transfer
  and restores the social threshold, and how the optimal toll relates to the Coase
  threshold and the adoption cost.
  Config: `examples/configs/congestion_small.json`.
  Example: `examples/phase13_congestion.py`.

`examples/phase5_summary_report.py --include-phase13` adds the congestion experiment to
the summary report as one more section.

## Conventions used across the write-ups

- All prices, cash and frictions are integer ticks, and fee rates are integer basis points.
- Every headline number cites its seed. A regression test pins it, so a doc and its
  example cannot drift apart without a failing test.
- `tests/test_repo_hygiene.py` checks that this index links every `docs/phase*.md`, and
  `frictionlab docs check` checks that every link and anchor in the built site resolves.
