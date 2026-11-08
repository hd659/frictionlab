# Phase 9: lemons, unraveling and certification

Every information friction before Phase 9 concerned a common fundamental. The fl-055
`LemonsMarket` moves the asymmetry onto the unit itself: each seller knows the quality
of the one unit it holds, buyers see only the pool on offer. The fl-056 `lemons`
experiment asks how far that pool unravels, how much surplus a priced certification
recovers, and at what certification cost sellers stop paying to escape the pool.

```
python examples/phase9_lemons.py --small [OUT]               # seed 56, small preset
frictionlab run examples/configs/lemons_small.json --small --out OUT/run
python examples/phase5_summary_report.py --include-phase9    # adds one section
```

## Setup

| piece | where | what |
|-------|-------|------|
| `LemonsMarket` | `markets/lemons.py` | hidden integer-tick quality, uniform pooled price, certified segment |
| `CertificationPolicy`, `lemons_pnl` | `costs/certification.py`, `analysis/pnl.py` | priced reveal routed to `CERT_SINK_ID`, seller PnL split |
| `akerlof_pooled_price`, `certification_cutoff` | `experiments/lemons.py` | analytic fixed point and separating cutoff |
| `run_lemons_grid`, `find_certification_threshold` | `experiments/lemons.py` | certification cost x buyer premium sweep, bisection |
| `lemons_small.json` | `examples/configs/` | costs 0, 4, 16, 32; premiums 0, 10, 40; 20 sellers |

Seller `i` holds quality `q` drawn from the schedule 10, 20, ..., 100 and values its
unit at `r(q) = q`. A buyer values a unit of known quality at `q + premium`, so every
trade is efficient and the full-information market clears everything. Every cell of
the sweep sees the same seeded draw (common random numbers).

## Why the pool unravels from the top

Buyers cannot tell units apart, so they pay for the average. The market posts

```
p_0     = floor(mean q over all sellers) + premium
p_(k+1) = floor(mean q over sellers with q <= p_k) + premium
```

and stops when a price repeats. A seller whose quality is above the posted price
would sell at a loss and withdraws. The withdrawals are always the best units, so the
average of what remains falls, the price falls with it, and the next-best units leave.
With a zero premium nothing stops the slide. On seed 56 the pooled price runs
57 -> 30 -> 20 -> 18 -> 10 -> 10 while the pool shrinks 20 -> 9 -> 6 -> 5 -> 1 units,
and only the single lowest-quality unit trades (traded share 0.05). The premium
cushions the fall: `akerlof_pooled_price` gives fixed points of 10, 30 and 95 at
premiums 0, 10 and 40, and at premium 40 the pool keeps every unit up to quality 90.

## Certification as a separating cutoff

A seller can pay `c` ticks to reveal `q` and sell at `q + premium` in the certified
segment. It does so iff `q + premium - c >= max(p, q)`, comparing the certified sale
with its best outside option, the pooled sale or keeping the unit. The left side rises
one-for-one in `q` and the right side is flat in `q` below `p`, so certification is a
cutoff rule: every certifier is at least as good as every pooled seller, and the
cutoff rises with `c`. At premium 40 the small sweep moves the cutoff 30 -> 50 -> 80
as cost goes 4 -> 16 -> 32, and the certifier count 15 -> 14 -> 7.

The cost buys no real resource. It moves ticks from a seller into `CERT_SINK_ID`, so
it is pure deadweight friction whenever the certifying unit would have traded anyway.
At premium 40 every unit trades in every cell, and realized surplus is exactly the
zero-cost surplus minus the certification sink: 800 ticks at cost 0, then 576 at
costs 16 and 32, where 224 ticks go to certification. Where the pool has unraveled,
certification instead rescues trade. At premium 10 and cost 4 all 20 units trade and
60 ticks are burned. At cost 16 nobody certifies, only 6 units trade, and 60 ticks
of surplus are left. The surplus share is non-decreasing in the premium at every cost
(0.00, 0.34, 0.82 at cost 0, and 0.00, 0.10, 0.59 at cost 32). The share is measured
against the fl-022 competitive benchmark, which treats buyer values and reservations as
two anonymous schedules. That benchmark counts cross-unit gains a unit-specific market
cannot reach, so even the frictionless cell reaches 0.82, not 1, at premium 40.

Over the whole sweep the surplus split is: buyer gain 7, pooled-seller gain 1123,
certified-seller gain 2590 and certification sink -568 ticks. Buyers get almost
nothing because both posted prices sit at the buyer value of what they deliver.

## The certification bound, the information collapse and the Coase threshold

A certifier gains `premium` over keeping its unit, and any pooled price is at least
`q_min + premium`, so its gain over the pooled sale is at most `q_top - q_min`. Nobody
certifies once

```
c >= certification_bound = min(premium, q_top - q_min) + 1
```

`find_certification_threshold` bisects for the smallest cost at which uptake falls
below 0.05, reusing seed 56 at every probe. At premium 20 it lands on 21 ticks
against the analytic bound of 21. Here the bound is attained exactly, because at the
fixed point some seller is exactly `premium` ticks better off certifying than keeping.

This is the fourth collapse bound in the project.
[Phase 5.1, the Coase threshold](phase5-coase-threshold.md), collapses volume at
F* = 61 ticks against a bound of 80 (seed 28), and
[Phase 8](phase8-grossman-stiglitz.md) collapses information production at 7 ticks
against 8 (seed 53). Each bound is the break-even cost of the best-placed participant.
The frictions remove different things. A Coase friction removes trades, and the
fl-053 signal cost removes information from the price while trade goes on. A
certification cost above the bound removes the escape route from the pool, so the
market falls back to the Akerlof fixed point. Below the bound it removes surplus
instead, tick for tick, from units that cross into the certified segment.

## Reproducibility

`tests/test_example_phase9_lemons.py` runs the example with `--small` twice and
requires byte-identical `rows.csv` and `report.html`. It also checks that the report
parses as XML with no external references, that the surplus share is non-decreasing
in the buyer premium, and that the headline numbers above match the run. Every number
on this page comes from master seed 56 and the small preset.
