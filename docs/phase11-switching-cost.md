# Phase 11: switching costs, bargain-then-ripoff pricing and lock-in

Phases 7 to 10 priced frictions that hit a single trade: search, intermediation,
information and quality checks. A switching cost is different, because it is paid only
when a consumer leaves its current supplier. Past purchases therefore become a stock that
later prices can draw on. The fl-061 `SwitchingMarket` runs two `LockinSeller`s for `T`
rounds over consumers who each carry a seeded switching cost. The fl-062
`switching_cost` experiment asks how large a markup lock-in supports, who pays it, and
how much switching cost it takes before the markup becomes material.

```
python examples/phase11_switching_cost.py --small [OUT]       # seed 62, small preset
frictionlab run examples/configs/switching_cost_small.json --small --out OUT/run
python examples/phase5_summary_report.py --include-phase11    # adds one section
```

## Setup

| piece | where | what |
|-------|-------|------|
| `SwitchingMarket` | `markets/switching.py` | T rounds, persistent suppliers, per-consumer switching cost, switching sink |
| `LockinSeller`, `switching_pnl`, `consumer_surplus` | `agents/lockin_seller.py`, `analysis/pnl.py` | acquire / harvest policy, PnL split |
| `klemperer_harvest_markup`, `acquisition_discount` | `experiments/switching_cost.py` | analytic markup and round-0 price cut |
| `run_switching_grid`, `find_lockin_threshold` | `experiments/switching_cost.py` | mean cost x target share sweep, bisection |
| `switching_cost_small.json` | `examples/configs/` | mean cost 0, 4, 16, 32; target share 6000, 9000 bps; 16 consumers, T = 6 |

Marginal cost is `c = 40` ticks. Valuations cycle through 20, 70, 80, ..., 130, so the
two consumers valued at 20 never buy and the most any seller can hold is 14 of 16
consumers, or 8750 bps. Each consumer draws its switching cost from the mean-preserving
schedule `(m - k, m, m + k)` with `k = min(2, m)`, on its own stream, so every cell sees
the same draws (common random numbers).

## Why competition happens ex ante

In round 0 nobody is attached yet, so both sellers price at `c` and the tie goes to
seller 1. That is the bargain. Seller 1 now holds 8750 bps, which is above a 6000 bps
target. From round 1 on it harvests: it posts the rival's price plus the smallest
switching cost among its own customers. At that price staying is weakly cheaper than
switching for every customer, so nobody leaves. That is the ripoff. On seed 62 with
mean cost 16 and target 6000 the traced price path of seller 1 is
`40 -> 54 -> 54 -> 54 -> 54 -> 54`. It switches from acquisition to harvest in round 1,
while seller 2 stays at 40 throughout and sells nothing.

Competition is still present, but it is all spent on the first sale. Over a horizon `T`
the seller would rationally give away up to

    d = (T - 1) * markup * target // 10000      (acquisition_discount)

ticks in round 0 to win the locked-in share. In this preset the acquisition price is
already `c`, so the acquisition loss is 0 and every rent shows up as harvest gain. A
target of 9000 bps is out of reach, because only 8750 bps of consumers ever buy, so those
cells never harvest and stay at Bertrand pricing.

## Why the markup is bounded by the minimum switching cost

The harvest price has to hold every locked-in customer at once. The weakest link is the
customer with the smallest switching cost `s_min`, and the price can never exceed the
lowest valuation `v_min` among the seller's customers. Against a rival at `c` this gives

    markup = max(0, min(s_min, v_min - c))      (klemperer_harvest_markup)

With spread 2, `s_min = m - 2`, so seed 62 gives harvest markups of 2, 14 and 30 ticks at
mean costs 4, 16 and 32 for target 6000. The simulated markup equals the analytic one in
every cell, and it is non-decreasing in the switching cost. At mean cost 32 the valuation
cap `70 - 40 = 30` is reached exactly, so a higher switching cost would buy no more. Churn is 0 everywhere, because a harvest price that
nobody wants to escape is exactly what the policy targets.

The welfare change against the zero-switching-cost Bertrand twin is 0 in every cell.
Nobody stops buying and nobody switches, so no switching cost is ever paid. Lock-in only
moves surplus: consumers lose 140, 980 and 2100 ticks at mean costs 4, 16 and 32, and the
harvesting seller gains the same amounts. Over the whole sweep the split is consumer gain
37100, acquisition loss 0, harvest gain 3220 and switching sink 0. Deadweight loss would
appear only if consumers actually switched, and a switch-proof harvest price is designed
to prevent that.

## The lock-in threshold

`find_lockin_threshold` bisects on the mean switching cost for the smallest level at which
the harvest markup exceeds `min_markup = 4` ticks. Each probe uses the same master seed.
The analytic answer is `min_markup + 1 + spread = 7`, because the markup is `m - 2` and it
must reach 5. On seed 62 at target 6000 the bisection also finds 7 ticks, against the
analytic 7.

## Frictions that convert history into rent

The lock-in threshold belongs with the other critical frictions in frictionlab. Each of
them marks the point where a cost tied to the past, or to the way a trade happens,
starts to change who earns what:

- the [Coase threshold](phase5-coase-threshold.md) (fl-028) is the per-trade cost at
  which volume collapses. A switching cost is a transaction cost charged only on the
  trade that changes counterparty;
- the [search collapse bound](phase7-search-cost.md) (fl-047) prices the cost of
  finding a new counterparty. A buyer who has already searched is effectively locked in
  by the search cost it would have to pay again;
- the [whitewash deterrent cost](phase10-reputation.md) (fl-059) is the price of
  discarding a history. Reputation makes a seller's past a liability it pays to escape,
  and switching costs make a consumer's past purchase a liability it pays to escape.

In each case the friction means the history of a match has value. A seller holding that
history can charge for it, up to the cost the counterparty would pay to walk away.

## Reproducing the numbers

All numbers above come from seed 62 with the small preset and are pinned by
`tests/test_example_phase11_switching_cost.py`. Two runs of the example give
byte-identical `rows.csv` and `report.html`.
