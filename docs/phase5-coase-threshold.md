# Phase 5.1: the Coase threshold

Coase's point was that trade happens only when its gains exceed the cost of
doing it. This experiment measures the friction level at which a double
auction stops trading, and checks the measurement against a closed-form bound.

```
python examples/phase5_coase_threshold.py                 # 6x6 schedule, 4 replicates
python examples/phase5_coase_threshold.py --small         # 4x4 schedule, 3 replicates
python examples/phase5_coase_threshold.py --workers 2 --report out/coase.html
```

## Setup

| piece | where | what |
|-------|-------|------|
| `ExperimentResult`, `run_experiment` | `experiments/base.py` | the Phase 5 shape: name, master seed, params grid, tidy fl-023 rows, findings dict, fl-026 report sections |
| `CoaseZIC`, `build`, `coase_metrics` | `experiments/coase.py` | friction-aware ZI-C market and per-cell metrics |
| `find_coase_threshold` | `experiments/coase.py` | grid scan plus integer bisection |

The schedule is fixed and has no seed. Six buyers hold one unit each, valued
160, 150, ..., 110. Six sellers hold one unit each, costing 60, 70, ..., 110.
The CE trades q* = 6 units for a maximum surplus of 300 ticks. The seed only
drives quote draws and wake-up times.

There is one friction knob, the total per-trade friction `F` in integer
ticks. `split_friction(F)` divides it into a `FixedCost` charged to both legs
and a flat taker fee (`TakerTickFee`), with `2 * fixed + taker == F` exactly.
Each trader knows its own role cost. A resting quote budgets for `fixed`. A
quote that would cross the touch budgets for `fixed + taker`, and if the
crossing trade would lose money the trader rests one tick behind the touch
instead. So every fill is individually rational net of friction, and a fill
between buyer v and seller c needs `v - c >= F`.

Each level reports:

- `volume`: units traded. `max_volume`: the cap, the largest k for which k
  buyers and k sellers can be paired with every gap >= F.
- `surplus_share`: net surplus (gains minus friction) as a share of the CE
  maximum, from fl-022 `welfare`. `gross_share` is the same before friction.
- `friction_collected`: the venue sink. It equals `F * volume` exactly
  (`friction_residual == 0`), since each fill pays `fixed` per leg plus one
  taker fee.
- `violations` and `zero_sum_residual`: ledger health. Both must be 0.

Levels run under common random numbers: replicate r uses the same seed at
every F, so differences across the grid come from friction and not from the
draws.

## Why the collapse starts at the marginal pair

Sort buyers high to low and sellers low to high. The CE pairs the k-th buyer
with the k-th seller, and the gaps shrink down the list: 100, 80, 60, 40, 20,
0. The last pair in the CE, the marginal pair (buyer 110, seller 110), has the
smallest gap. With any F > 0 that pair can no longer trade with each other.
The marginal traders still have a way back in: buyer 110 can trade with the
cheapest seller (gap 50), and seller 110 with the richest buyer (gap 50). To
keep k units trading, the market must pair the top k buyers with the bottom k
sellers in crossed order, so the binding gap is

    min over i < k of (value[i] - cost[k - 1 - i])

That is `max_volume(F)`. On this schedule the binding gaps are 50, 60, ..., 100
for k = 6, 5, ..., 1, so the cap drops by one unit each time F passes the next
one. Each time, the marginal trader on the long side drops out,
because it has the least surplus left to cover F. The intramarginal pairs
follow in order. Volume falls one unit at a time, not all at once.

Surplus falls faster than volume, because every unit that still trades also
pays F:

    net surplus(F) = gross gains - F * volume

ZI-C traders do not choose the crossed matching. Random quotes can pair a
wide-gap seller with a narrow-gap buyer, which strands the other marginal
trader. So realized volume falls below `max_volume` at middle frictions, and
`gross_share` falls before the cap does. The threshold therefore depends on
the matching process as well as the gaps: the gaps give the ceiling, and
zero intelligence shows how much of the ceiling a market with no strategy
reaches.

The analytic bound is `max(values) - min(costs)` = 100 ticks, the widest
possible pair. Above it no pair clears, so volume is exactly 0. At F equal to
the bound only that one pair can trade, so volume is at most 1.

## Locating the threshold

`find_coase_threshold(cfg, fraction)` defines F* as the smallest F at which
mean volume falls below `fraction` times the frictionless volume (or reaches
zero). It works in two steps:

1. A coarse scan over the grid, always including 0 and bound + 1, gives a
   bracket (last level above target, first level below).
2. Integer bisection inside the bracket narrows it to adjacent integers. Each
   probe reuses the same master seed, so a probe is a pure function of F, and
   no level is simulated twice.

Mean volume is non-increasing in F on this schedule, so bisection returns the
first crossing that the grid scan brackets. F* lies in [0, bound] whenever
`fraction * baseline > 1`, and is never above bound + 1.

## Seeded result (seed 28, 4 replicates)

| F | cap | volume | surplus share | friction |
|---|-----|--------|---------------|----------|
| 0 | 6 | 5.50 | 0.992 | 0 |
| 34 | 6 | 4.25 | 0.460 | 144.5 |
| 68 | 4 | 2.50 | 0.125 | 170.0 |
| 102 | 0 | 0.00 | 0.000 | 0 |

The half-volume threshold is F* = 66 ticks, about two thirds of the bound.
Friction collected peaks at middle F: past that point the venue charges more
per trade but loses more trades than it gains in fees. That is the venue
side of the Coase story, and it sets up the fee-tier experiment later in
Phase 5.
