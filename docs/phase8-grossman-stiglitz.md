# Phase 8: the Grossman-Stiglitz paradox

Every informed trader before Phase 8 got its information for free. The fl-052
`SignalSource` puts a price on it, and the `InfoAcquirer` buys a signal only when the
expected edge clears that price. The fl-053 `grossman_stiglitz` experiment puts a
population of acquirers in front of a dealer who prices their flow, and asks how many
of them pay, how much the price then reveals, and at what signal cost information
production stops.

```
python examples/phase8_grossman_stiglitz.py --small [OUT]    # seed 53, small preset
frictionlab run examples/configs/grossman_stiglitz_small.json --small --out OUT/run
python examples/phase5_summary_report.py --include-phase8    # adds one section
```

## Setup

| piece | where | what |
|-------|-------|------|
| `SignalSource`, `InfoAcquirer`, `info_pnl` | `costs/information.py`, `agents/info_acquirer.py`, `analysis/pnl.py` | priced signal, buy-or-not decision, PnL split |
| `GSMaker` | `experiments/grossman_stiglitz.py` | posterior mid +- half the GM spread at the live informed share |
| `gs_equilibrium_informed_share`, `analytic_collapse_cost` | `experiments/grossman_stiglitz.py` | analytic fixed point and its zero |
| `run_gs_grid`, `population`, `find_information_collapse` | `experiments/grossman_stiglitz.py` | signal_cost x noise_intensity sweep, replicator, bisection |
| `grossman_stiglitz_small.json` | `examples/configs/` | costs 0, 2, 4, 8; noise 1 and 4; horizon 40 |

The fundamental is `V = 100 +- 10`, drawn once per seed. Eight acquirers wake at rate
1, each signal is right with probability 9000 bps, and a decision costs nothing extra
(decision cost 0), so the whole hurdle is the signal price. Four noise traders share an
arrival rate equal to the noise intensity. The dealer quotes size 50 at rate 4.

## Why fully revealing prices cannot be an equilibrium

Suppose the price already equals `V`. A signal then tells a buyer nothing the price
has not told it, its edge is zero, and at any positive cost nobody buys. With nobody
informed the dealer has no reason to move the mid off the prior, so the price no longer
reveals `V`, and a signal is worth buying again. Neither end is a resting point. The
equilibrium has to sit in between, with just enough informed traders that the marginal
buyer breaks even.

The small sweep on seed 53 (buyer fraction is the share of acquirer decisions that buy;
the efficiency error is the time-averaged `|mid - V|`):

| signal cost | analytic share | buyers, noise 1 | buyers, noise 4 | spread, noise 4 | `|mid - V|`, noise 4 |
|------------:|---------------:|----------------:|----------------:|----------------:|---------------------:|
| 0 | 0.80 | 0.70 | 0.99 | 13.48 | 0.22  |
| 2 | 0.60 | 0.48 | 0.74 | 10.23 | 0.22  |
| 4 | 0.40 | 0.33 | 0.48 | 6.91  | 0.23  |
| 8 | 0.00 | 0.00 | 0.00 | 2.00  | 10.00 |

While anyone buys, the mid stays within about a quarter tick of `V`. At cost 8 nobody
buys, the dealer quotes the minimum spread of 2 around the prior, and the error is the
full jump of 10 ticks. That is the Grossman-Stiglitz point: the price stops carrying
information because it no longer pays anyone to put information into it.

## Noise trading pays for information

An informed trader earns its edge from someone. The dealer breaks even by
construction, so what informed buyers gain comes out of the noise traders. More noise
volume means more uninformed flow per informed lot, so the dealer's adverse-selection
loss per trade is smaller and more acquirers clear the hurdle. In the table the buyer
fraction rises with noise intensity at every cost below the collapse: 0.70 to 0.99 at
cost 0, 0.48 to 0.74 at cost 2, 0.33 to 0.48 at cost 4. Tests pin it as non-decreasing
in noise intensity and non-increasing in signal cost on the fixed seed.

Being informed does not mean being profitable here. Over the whole sweep the buyers'
trading gain is -9936 ticks, they pay -1854 ticks for signals (exactly the information
sink's revenue), decision cost is 0, and their net is -11790. The dealer's
posterior moves after each informed print, so later buyers trade at prices that have
already absorbed most of the signal. This is the paradox in realized PnL: the more
informed the flow, the less any one signal is worth.

## The GM spread closes the loop

The fl-030 Glosten-Milgrom spread at informed share `mu` is `2 * mu * L` with
`L = jump`, so on this market it is `0.2 * pct` ticks for `mu = pct / 100`. A buyer's
expected edge is

```
edge(mu) = floor((2 * 9000 - 10000) * 10 / 10000 - spread(mu) / 2) = floor(8 - 0.1 * pct)
```

and `gs_equilibrium_informed_share` returns the largest integer-percent `mu` with
`edge(mu) >= signal_cost`. That gives `mu* = 0.8 - 0.1 * cost`: 0.80, 0.60, 0.40 and
0.00 at costs 0, 2, 4 and 8. The loop is closed in both directions. More buyers widen
the spread, a wider spread lowers every buyer's edge, and the share settles where the
edge equals the price. At the smallest share (1%) the edge is 7 ticks, so
`analytic_collapse_cost` is 8.

The fl-024 replicator runs committed buyers against non-buyers at noise 4. At cost 0
it ends at a buyer share of 0.72 without the detector firing in 8 generations, against
an analytic 0.80. At cost 4 and cost 9 committed buyers go extinct at generation 3,
ending at the 1/8 floor of 0.0625, against analytic shares of 0.40 and 0.00. The
report's equilibrium section draws the cost-4 trajectory (0.50, 0.25, 0.125, 0.0625)
with its fixed point annotated. Committed buyers pay on every wake-up, even when the
spread has already eaten the edge, so they do worse than the conditional
`InfoAcquirer` and the replicator share sits at or below the analytic one.

## Collapse, the Coase threshold and the search bound

`find_information_collapse` bisects for the smallest signal cost at which the buyer
fraction at noise 4 falls below 0.05, reusing seed 53 at every probe. The probes give
0.99, 0.48, 0.24 and 0.00 at costs 0, 4, 6 and 7, so the collapse is at 7 ticks
against the analytic 8.

This is the third collapse bound in the project, and all three share a shape.
[Phase 5.1, the Coase threshold](phase5-coase-threshold.md), bisects a per-trade
friction F to F* = 61 ticks against an analytic bound of 80 (seed 28).
[Phase 7 search costs](phase7-search-cost.md) collapses half the volume at 33 ticks
against 51 (seed 47). Here a per-decision information cost collapses information
production at 7 against 8 (seed 53). Each analytic bound is the cost at which the
best-placed participant just breaks even, and the simulated market collapses at or
below it because real participants are not all best-placed. In this case the
one-tick gap comes from two integer conventions. The analytic share counts a buyer
whose edge equals the cost (`>=`), while the fl-052 `InfoAcquirer` buys only when the
edge strictly exceeds it, and the dealer never quotes a half-spread below one tick. At
cost 7 the 7-tick edge at the smallest share therefore no longer buys.
The friction is different, too. A Coase friction removes trades, and a search cost
removes meetings. A signal cost removes information from the price even while trading
goes on, because noise traders and the dealer keep trading at the uninformed prior.

## Reproducibility

`tests/test_phase8_example.py` runs the example with `--small` twice and requires
byte-identical `rows.csv` and `report.html`. It also checks that the report parses as
XML with no external references, that the buyer fraction is non-increasing in signal
cost, and that the headline numbers above match the run. Every number on this page
comes from master seed 53 and the small preset.
