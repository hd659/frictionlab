# Phase 7: intermediation under search costs

The fl-046 `SearchMarket` makes every buyer and seller find a counterparty one
meeting at a time. The fl-049 `Middleman` is a second way to trade. It quotes a bid
below the mid and an ask above it, holds up to `capacity` units, and pays a holding
cost and perishable decay on every unit it carries. The fl-050 `intermediation`
experiment asks when traders route through it, what it earns, and what it costs the
market.

```
python examples/phase7_intermediation.py --small [OUT]       # seed 50, small preset
frictionlab run examples/configs/intermediation_small.json --small --out OUT/run
python examples/phase5_summary_report.py --include-phase7    # adds sections six and seven
```

## Setup

| piece | where | what |
|-------|-------|------|
| `Middleman`, `middleman_pnl` | `agents/middleman.py`, `analysis/pnl.py` | integer bid / ask, capacity, carry to hold / decay sinks, PnL split |
| `middleman_indifference_spread`, `analytic_bound` | `experiments/intermediation.py` | the trader's break-even spread and its inverse in search cost |
| `run_intermediation`, `find_intermediation_threshold` | `experiments/intermediation.py` | search_cost x middleman_half_spread sweep, integer bisection |
| `intermediation_small.json` | `examples/configs/` | costs 0, 2, 8, 32; half-spreads 0, 4, 16; horizon 100 |

Six buyers are valued 130, 125, ..., 105 and six sellers cost 40, 45, ..., 65, so the
competitive equilibrium trades all 6 units for 390 ticks of surplus. The middleman
quotes `85 - h` and `85 + h` around the integer mid 85 of the CE price band. Traders
meet each other at rate 0.5 and the middleman at rate 2.0. It holds at most 2 units
and pays 1 tick of holding cost and 1 tick of decay per unit per step. Every cell also
runs a no-middleman twin on the same seed, which gives the surplus change.

## The middleman earns rent only if it shortens search by more than its spread

A trader who needs `mpm` direct meetings for a match expects to wait `mpm / lambda_d`
for it, against `1 / lambda_m` for the middleman. At a search cost of `c` per unit of
waiting time, the middleman saves

```
saving = c * (mpm / lambda_d - 1 / lambda_m)
```

and the trader takes a half-spread `h` only if `h <= floor(saving)`. That floor is
`middleman_indifference_spread`. It is 0 at zero cost and grows with `c`. On this
schedule every pair can trade (`mpm = 1`), so the gap is `1/0.5 - 1/2 = 1.5` and the
spread a trader accepts is `floor(1.5 * c)`: 0, 3, 12, 48 on the small grid. Inverting
it gives `analytic_bound = ceil(h / 1.5)`, the smallest search cost at which a spread
`h` is worth paying. On seed 50:

| search cost | accepted spread | share at h = 0 | share at h = 4 | share at h = 16 | middleman net at h = 4 / 16 |
|------------:|----------------:|---------------:|---------------:|----------------:|----------------------------:|
| 0  | 0  | 0.83 | 0.00 | 0.00 | 0 / 0    |
| 2  | 3  | 0.83 | 0.00 | 0.00 | 0 / 0    |
| 8  | 12 | 0.83 | 0.83 | 0.00 | 32 / 0   |
| 32 | 48 | 0.83 | 0.83 | 0.83 | 32 / 152 |

The intermediated share does not decrease with search cost, and it does not increase
with the spread (tests pin both). `find_intermediation_threshold` bisects for the
smallest search cost at which the share passes 0.5, reusing seed 50 at every probe.
It finds 0, 3 and 11 ticks for half-spreads 0, 4 and 16, equal to the analytic
bound each time. Once traders route through the middleman, 5 of the 6 units go
through it, and the remaining unit is matched directly.

A zero-spread middleman never earns rent. At `h = 0` it carries units for nothing
and ends at -8 ticks (4 holding, 4 decay). Rent appears only where the accepted spread
exceeds the quote: 8 ticks a round trip at `h = 4`, 32 at `h = 16`.

What the market gains is time. Every cell realizes the full 390 ticks with or without
the middleman, so the surplus change is 0. Mean time to match falls from 8.22 to 2.77
when traders use it. Because each intermediated unit needs two charged meetings
(seller to middleman, middleman to buyer) in place of one meeting charged to both
sides, realized search cost does not fall. It rises by `2c` per cell, which leaves
the net surplus change at `-2c`: -4, -16 and -64 ticks. In this model search cost is a
per-meeting charge, and the analytic saving above is a cost of time. The sweep shows
the faster match. The sinks show that the middleman does not reduce the number of
meetings that get charged.

## Perishability caps the inventory it will hold

A held unit costs `holding + decay` ticks per step, and the middleman recovers at
most `2h` when it sells. A unit that waits `T` steps for a buyer is worth holding only
if `(holding + decay) * T <= 2h`. Buyers reach the middleman at a finite rate, so the
k-th unit in inventory waits about k buyer meetings, and the expected `T` grows with
inventory. The largest inventory worth carrying is about
`2h * (buyer meetings per step) / (holding + decay)`. A wider spread raises it, and
a faster decay lowers it. The capacity `K` is the hard cap on top of this.

The example's PnL view holds the costliest cell (search cost 32, `h = 4`, capacity 2)
fixed and raises the carry, holding cost = decay = k per step:

| carry k | spread capture | holding | decay | net |
|--------:|---------------:|--------:|------:|----:|
| 0 | 40 | 0  | 0  | 40  |
| 1 | 40 | 4  | 4  | 32  |
| 2 | 40 | 8  | 8  | 24  |
| 4 | 40 | 16 | 16 | 8   |
| 8 | 40 | 32 | 32 | -24 |

Five round trips earn 40 ticks, and 4 unit-steps of carry cost `8k`, so the
middleman breaks even at `k = 5`. The last carry level in the view with `net >= 0` is
4. Peak inventory never goes above 1 unit here. With a 2.0 meeting rate a unit is
sold almost as soon as it is bought, so the capacity of 2 is never used. Over the
whole small sweep the report's PnL section splits the realized surplus into buyer
gain 2220, seller gain 2220 and middleman net 184. Against these are a search-cost
sink of -1740, a holding-cost sink of -28 and a decay loss of -28 ticks. The
middleman net is its spread capture of 240 minus its 56 ticks of carry.

## Relation to the search collapse and the Coase threshold

In [Phase 7 search costs](phase7-search-cost.md) volume collapses once no meeting can
cover its cost (an analytic bound of 51 on the fl-047 schedule). The indifference
spread runs the other way. It grows linearly in search cost, so the closer a market
is to that collapse, the larger the spread traders will pay to skip direct search.
High search cost is where a middleman's rent comes from, and it is also the regime
where direct trade is in danger of collapsing.

[Phase 5.1, the Coase threshold](phase5-coase-threshold.md), has a fill need
`v - c >= F` for a per-trade friction `F`. A direct search trade pays
`F = 2 * c * mpm / lambda_d` in expected search time. A trade through the middleman
pays the round-trip spread plus one middleman meeting per side,
`F = 2h + 2c / lambda_m`. Intermediation lowers the effective Coase friction exactly
when the second is smaller. That is the indifference condition above, applied to both
sides of the trade. Perishable carry adds a third term on the middleman's side. Its
rent `2h` has to cover `(holding + decay) * T`, so the spread cannot fall to zero
even when the meeting-rate advantage is large.

## Reproducibility

`tests/test_example_phase7_intermediation.py` runs the example with `--small` twice
and requires byte-identical `rows.csv` and `report.html`. It also checks that the
report parses as XML with no external references, and that the thresholds and the
PnL view match this page. Every number here comes from master seed 50 and the small
preset.
