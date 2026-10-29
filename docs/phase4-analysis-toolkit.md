# Phase 4: the analysis toolkit

Phase 4 turns raw `MarketRunner` output into numbers, experiments and a
report. It has five pieces, and `examples/phase4_analysis_toolkit.py` runs all
of them on one seed:

| milestone | module | job |
|-----------|--------|-----|
| fl-022 | `analysis/metrics.py`, `analysis/welfare.py` | realized spread, price impact, price efficiency, surplus vs the competitive equilibrium (CE), zero-sum PnL by account |
| fl-023 | `analysis/sweep.py` | seeded parameter sweeps over friction levels, run serially or in parallel |
| fl-024 | `analysis/equilibrium.py` | replicator dynamics over agent archetypes, with a stabilization detector |
| fl-025 | `analysis/plots.py`, `analysis/scales.py` | SVG charts with no dependencies and byte-identical output |
| fl-026 | `analysis/report.py`, `analysis/report_sections.py` | one offline HTML report |

```
python examples/phase4_analysis_toolkit.py                  # 4x3 grid, report in a temp dir
python examples/phase4_analysis_toolkit.py --out out/ --seed 7 --workers 2
python examples/phase4_analysis_toolkit.py --small          # 2x2 grid, test-sized
```

## Part 1: taker fee x latency over a mixed market

Each sweep cell builds the same population: an Avellaneda-Stoikov market
maker, an informed trader who sees the jump fundamental V(t), four noise
traders and a ZI-C crowd of 4 buyers and 4 sellers with 3 units each. The
grid sets `taker_bps` and `latency`. Latency adds two costs. Intents arrive
`latency` seconds after the agent saw the book, and taker fills also pay
`drift * latency` ticks per lot. Each cell reports:

- `effective_spread`, `realized_spread`, `price_impact`: volume-weighted, at
  horizon `tau`, in doubled-mid ticks (see below).
- `welfare_surplus`: total surplus. Each ZI unit traded is valued at its
  schedule value, every other account is marked at the final fundamental
  V_T, and fees and decision costs are subtracted.
- `welfare_loss`: the ZI-only CE maximum minus `welfare_surplus`. Outside
  liquidity (the maker and the noise flow) can give the ZI crowd trades that
  are not in its own CE, so this number is a benchmark, not a bound.
- `violations`, `zero_sum_residual`, `identity_residual`: health checks. All
  three must be 0.

### Metric identities

With d = +1 for buyer-initiated fills and -1 for seller-initiated ones, and
the doubled mid M = bid + ask (always an integer):

```
effective = d * (2p - M(t-))
realized  = d * (2p - M(t + tau))
impact    = d * (M(t + tau) - M(t-))
effective = realized + impact          (exact integer identity, per trade)
```

Realized spread is what the liquidity supplier keeps after the price has
moved. Impact is the permanent, informational part of the move. In the seeded
run, at zero latency the maker's realized spread is negative (informed flow
picks it off) and impact is large. As latency rises, both the informed
trader's edge and the impact shrink.

Welfare closes with no remainder. Prices are transfers, and the fee and
decision-cost sinks are accounts too, so the PnL of all accounts marked at
V_T sums to 0. That gives

```
welfare_surplus = sum_ZI side * (value_k - V_T) - fees - decision_costs
```

Each fill moves a unit from its holder's valuation to the buyer's valuation,
and the friction leaves the market. Higher fees remove marginal trades and
take a cut of the trades that still happen. The tests pin this down:
for every latency, mean surplus does not increase as `taker_bps` goes up.

## Why child seeds make sweeps order-independent

`SweepSpec.child_seeds()` is a pure function of the spec. A
`numpy.random.SeedSequence(master_seed)` is spawned into one child per
(cell, replicate) position, or one per replicate when `crn=True`. Each task
carries its own integer seed, and the factory builds every RNG stream from
that seed. No cell reads state that another cell wrote, so the order in which
cells run, or which process runs them, cannot change any row. A serial run
and `n_workers=2` return identical rows, and re-running with the same master
seed is bit-identical. The example uses common random numbers (`crn=True`):
replicate r sees the same fundamental path, schedules and arrivals in every
cell, so differences between cells come from friction and not from sampling
noise.

The replicator uses the same scheme per generation:
`SeedSequence([master_seed, generation])`. Fitness at generation g therefore
depends only on the current mix and g.

## Part 2: satisficer vs optimizer

`BoundedRationalAgent`s in the current mix trade against a ZI-C crowd. Shares
become integer counts by largest-remainder rounding, fitness is mean net PnL
per agent of each type, and the update is the normalized replicator step
with eta = 0.5. The run stops when the detector fires: either `stable` (max
share change below epsilon for 3 generations) or `extinction` (a type falls
below 5%). Deliberation is free at `cost_per_step = 0`, so neither type has a
consistent advantage and the drift depends on the seed. At `cost_per_step = 8`,
optimizers pay for up to 20 search steps per wake-up, and satisficers stabilize
near 0.94. The tests check that the high-cost satisficer share is at least
the low-cost share.

## Part 3: reading the report

`phase4_report.html` is a single file. It has no JavaScript and no URLs, and
it is byte-identical across reruns of the same seed and grid.

1. Header: frictionlab version, master seed, the parameter grid, replicates,
   horizon, tau and the cost levels. These are enough to reproduce the run.
2. Part 1 sweep section: per-cell mean and std of every metric, a
   `welfare_surplus` heatmap (rows are taker_bps, columns are latency; read
   it top to bottom for the fee effect), and an error-row table. That table
   should be empty, because a failing cell shows up there instead of being
   dropped.
3. Part 2 replicator section: one share chart per cost level, with the
   detected fixed point circled and labelled with its reason and generation,
   plus an initial/final/fixed-point/fitness table.

If the sweep and the replicator disagree with the printed summary, the report
was generated from a different seed or grid. Check the header first.
