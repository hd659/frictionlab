# Phase 14: menu costs and nominal price rigidity

Before Phase 14 every seller re-priced for free every step. The fl-070 `MenuCostMarket`
changes that. Each seller has an integer-tick frictionless target `p*` that moves by a
common drift `d` (inflation, in ticks per step) plus a seeded idiosyncratic shock. A seller
posting `p` earns `base_profit - k * (p - p*)^2` ticks that step, and every price change
costs a menu cost `F`, paid to the menu sink. The fl-071 `menu_cost` experiment sweeps
`F x d` on common random numbers, with every seller an `SsPricer` at the analytic band.

```
python examples/phase14_menu_cost.py --small [OUT]            # seed 71, small preset
frictionlab run examples/configs/menu_cost_small.json --small --out OUT/run
python examples/phase5_summary_report.py --include-phase14    # adds one section
```

## Setup

| piece | where | what |
|-------|-------|------|
| `MenuCostMarket` | `markets/menu_cost.py` | drifting targets, per-seller shock streams, quadratic misalignment loss |
| `MenuCost` | `costs/menu.py` | integer-tick charge per price change, routed to the menu sink |
| `SsPricer` | `agents/ss_pricer.py` | keep while `abs(p - p*) <= band`, else reset to `p*` plus an overshoot |
| `menu_pnl` | `analysis/pnl.py` | frictionless profit, misalignment loss and menu cost per seller |
| `optimal_ss_band`, `adjustment_frequency` | `experiments/menu_cost.py` | exact sawtooth loss in `Fraction`s |
| `run_menu_grid`, `find_rigidity_threshold` | `experiments/menu_cost.py` | `F x d` sweep, menu-cost bisection |
| `menu_cost_small.json` | `examples/configs/` | menu costs 0, 2, 8, 32; inflation 1, 2, 3; 2 sellers, `k = 1`, 120 steps, no shocks |

## Why a fixed cost gives an inaction band, not partial adjustment

A convex adjustment cost, say `c * (dp)^2`, makes the first tick of a change almost free.
A seller facing it moves a little every step and closes part of the gap each time. That
is partial adjustment. A menu cost is different: it is the same `F` for a one-tick change
as for a fifty-tick change. Once a seller decides to pay it, nothing argues for stopping
short of the best price, so every adjustment is a full reset. The only question is when to
pay. The answer is a threshold on the gap: hold while `abs(gap) <= band`, reset once it is
crossed. That is the (S,s) rule of Sheshinski and Weiss (1977). Prices move in lumps,
and between lumps they do not move at all.

Under steady drift the gap traces a sawtooth. After a reset to `p* + o` it falls by `d`
per step until it leaves the band, so one cycle has `L` steps and the long-run loss per
step is `V(b) = (k * sum gap^2 + F) / L`. `optimal_ss_band` evaluates `V` exactly for
each integer band and returns the smallest minimizer. On seed 71 at `F = 32`, `d = 2` the
flexible band 0 loses 32 ticks per step (the whole menu cost, every step), band 2 loses 18
and band 4 loses 14, so the analytic band is 4 with an analytic frequency of 1/4. The traced
seller's path shows the pattern: target 1003, 1005, 1007, 1009 against posted
1003, 1003, 1003, 1011, so the gap runs `+0 -2 -4 +2`, and the seller adjusts at steps 4,
8, 12, 16, 20 and 24.

## Why small frictions buy large rigidity

The misalignment loss is second order in the gap: a gap of `x` costs `k * x^2`. The menu
cost is first order: it is `F` whatever the size of the change. Near the target the loss
from not adjusting is tiny, so even a small `F` buys a band of real width. Balancing the
two under drift `d` (mean loss about `k * b^2 / 3` per step against `F * d / (2 * b)` of menu
cost per step) gives a band of order `(F * d / k)^(1/3)`: the lost profit from waiting stays
second order while the saving on menu costs is first order. This is Mankiw's
(1985) point: a menu cost that is small next to profit can produce large nominal rigidity.

The sweep shows it. Adjustment frequency is non-increasing in menu cost in every inflation
column: at `d = 1` it falls 1, 0.5, 0.25, 0.2 as `F` goes 0, 2, 8, 32, and the mean
absolute gap rises 0, 0.5, 1, 1.4. Higher inflation pushes the other way. At `F = 32`
frequency is 0.2, 0.25 and 0.5 at `d = 1, 2, 3`, because a faster-moving target leaves the
band sooner. Over the sweep sellers keep 576000 ticks of frictionless profit, lose -4200 to
misalignment and pay -11856 to the menu sink. Misalignment is the smaller of the two even
though the bands are what keep it small. The welfare change against the flexible
zero-cost twin is at worst -4920 ticks, at `F = 32`, `d = 3`.

The rigidity threshold is the smallest `F` at which simulated adjustment frequency drops
below half of the flexible baseline (`rigidity_bps = 5000`). At `d = 2` the analytic band
jumps from 2 to 4 between `F = 16` and `F = 17`, and frequency falls from 1/2 to 1/4. The
analytic threshold is 17 and `find_rigidity_threshold` bisects to 17, with band 4 at that
level.

## Frictions on acting rather than on trading

The rigidity threshold belongs with two earlier thresholds:

- the [Coase threshold](phase5-coase-threshold.md) (fl-028) is the per-trade transaction
  cost above which bilateral volume collapses. It taxes each trade, so it shrinks how
  much is traded;
- the [decision-cost break-even](phase5-decision-cost.md) (fl-029) is the per-step
  deliberation cost above which a satisficer beats an optimizer. It taxes thinking, so
  agents stop optimizing every step.

A menu cost taxes acting. Every sale still happens and nobody stops thinking; the seller
just stops changing its price. All three are fixed costs that produce a threshold and an
inaction region, not a smooth response. The difference is in what goes still. Under a Coase
cost volume vanishes, under a decision cost optimization gives way to rules of thumb,
and under a menu cost the price freezes while the target keeps drifting. Only the menu cost
creates a nominal friction that inflation interacts with: more drift shrinks the inaction
spell, which is why frequency rises with `d` in every row.

## Reproduce and check

`tests/test_phase14_example.py` pins every number above to seed 71, checks that two runs
give byte-identical `rows.csv` and `report.html`, and checks that the report is valid XML
with no network references. `tests/test_experiment_menu_cost.py` covers the analytic band
and the bisection on their own.
