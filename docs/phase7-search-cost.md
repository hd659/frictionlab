# Phase 7: search costs and perishable value

Every venue before Phase 7 is centralized: an order meets the whole book at once.
In the fl-046 `SearchMarket` a buyer and a seller find each other one meeting at a
time, and every meeting costs both of them. The fl-047 `search_cost` experiment asks
how much trade survives that cost, and what changes when an unsold good loses value
while it waits.

```
python examples/phase7_search_cost.py --small [OUT]         # seed 47, small preset
frictionlab run examples/configs/search_cost_small.json --small --out OUT/run
python examples/phase5_summary_report.py --include-phase7    # adds a sixth section
```

## Setup

| piece | where | what |
|-------|-------|------|
| `SearchMarket`, `SearchAgent`, `ReservationPolicy` | `markets/search_market.py` | seeded pairwise meetings, per-meeting cost to a sink, integer Nash split, decay to a floor |
| `reservation_price`, `analytic_bound` | `experiments/search_cost.py` | the stopping rule and the collapse bound |
| `run_search_cost`, `find_search_collapse` | `experiments/search_cost.py` | search_cost x decay_per_step sweep and integer bisection |
| `search_cost_small.json` | `examples/configs/` | costs 0, 10, 30, 52; decays 0, 2; horizon 100 |

The schedule has six buyers valued 120, 110, ..., 70 and six sellers costing 20, 30,
..., 70. The competitive equilibrium trades 6 units for a maximum surplus of 300 ticks.
The meeting rate is 1, the bargaining weight is 5000 bps (an even split, remainder to
the seller) and the decay floor is 20 ticks.

## The reservation-price stopping rule

A meeting costs `k = search_cost / meeting_rate` in expectation per unit of time spent
looking. Suppose counterparties are uniform on `lo..hi` (`n = hi - lo + 1` prices).
A buyer with value `v` who accepts any seller at or below `p` finds one with
probability `m(p) / n`, where `m(p) = #{x <= p}`. So it expects `n / m(p)` meetings,
and a search bill of `k * n / m(p)`, before it stops. The buyer's reservation price is
the largest `p <= v` that still pays for the search:

```
(v - p) * m(p) >= k * n
```

A seller mirrors this with the smallest ask `a >= value` such that
`(a - value) * #{x >= a} >= k * n`. All of it is exact integer arithmetic on
`Fraction`s. With zero cost the reservation equals the value. If no price covers the
expected bill, the agent stays out and pays nothing. The markup is fixed at entry.
A perishable seller's ask then tracks its current value, so decay lowers it.

## Search cost and decay pull in opposite directions

Search cost makes both sides pickier. Each side shades its reservation away from its
value by at least `k`, so a pair `(v, s)` trades only if `v - s >= 2k`. Low-gap
pairs drop out and fewer agents enter. On seed 47 with no decay:

| search cost | volume | mean time to match | surplus share | search cost paid |
|------------:|-------:|-------------------:|--------------:|-----------------:|
| 0  | 6 | 2.87   | 1.00 | 0   |
| 10 | 6 | 2.87   | 1.00 | 120 |
| 30 | 3 | 1.73   | 0.80 | 180 |
| 52 | 0 | 100.00 | 0.00 | 0   |

Surplus share does not increase with search cost (a test pins this). The time to
match falls at cost 30 only because the agents left are the high-gap pairs, which
clear on almost any meeting. Mean time to match counts an unmatched seller at the
horizon, so a cost of 52 shows the full 100.

Decay pushes the other way. A seller whose good spoils loses by waiting, so its ask
falls each step and it accepts lower prices. Matching speeds up, and decayed value
is destroyed, not transferred. At search cost 10 in the decay view of the example:

| decay per step | volume | mean price | mean time to match |
|---------------:|-------:|-----------:|-------------------:|
| 0 | 5 | 70.00 | 21.89 |
| 1 | 6 | 68.33 | 4.67  |
| 2 | 6 | 66.67 | 4.67  |
| 4 | 6 | 64.17 | 4.67  |
| 8 | 6 | 61.33 | 4.67  |

The seller's share of realized surplus falls with decay (0.50 at decay 0, 0.45 at
decay 2 on the small grid) because perishability weakens its outside option. At high
search cost decay can also keep sellers in the market who would otherwise exit. They
search, pay and spoil for longer, which is why net surplus can go negative at cost 30
with decay 2.

Over the whole small sweep the report's PnL section splits the gains into buyer gain
883, seller gain 797, a search-cost sink of -960 and a decay loss of -356 ticks.

## The collapse bound and the Coase threshold

In Phase 5.1 ([the Coase threshold](phase5-coase-threshold.md)) a fill between
buyer `v` and seller `c` needs `v - c >= F`, so trade stops once the per-trade
friction `F` passes the largest gap. Search has the same structure with
`F = 2k = 2 * search_cost / meeting_rate`, so `analytic_bound` is the smallest integer
cost with `2k > max(values) - min(costs)`:

```
analytic_bound = floor((120 - 20) * meeting_rate / 2) + 1 = 51
```

With decay the lowest seller value is `min(costs, floor)`. The two differences from
Coase are where the predictions part ways. First, search cost is paid per meeting,
not per trade, so agents pay it even on meetings that end with no trade. Second,
the stopping rule adds a markup beyond `k` when acceptable counterparties are rare.
Volume therefore collapses well before the bound. `find_search_collapse` bisects for
the smallest cost at which volume falls below half its zero-cost level, reusing seed
47 at every probe. It finds 33 ticks at decay 0 and at decay 2, inside the bound of 51
in both cases. At 52 no meeting can cover its cost and nothing trades, which is the
search version of the Diamond paradox: a small per-meeting cost is enough to stop
trade that would be worth doing.

## Reproducibility

`tests/test_example_phase7_search_cost.py` reruns the example with `--small` twice
and requires byte-identical `rows.csv` and `report.html`. It also checks that the
report parses as XML with no external references. Every number on this page comes from
master seed 47 and the small preset.
