# Phase 5.4: fee-tier wars between two venues

Two exchanges list the same asset. Takers send their orders to whichever
venue is cheaper all in, and market makers quote wherever the rebate is
best once adverse selection is netted off. Each venue sets a
`(maker_fee, taker_fee)` pair. This experiment has the venues best-respond
to each other round by round and reports where fees end up: either a pure
fixed point or an Edgeworth-style cycle.

```
python examples/phase5_fee_wars.py                  # 17-pair grid, 200 steps, 2 replicates
python examples/phase5_fee_wars.py --small          # 11-pair grid, 120 steps, 1 replicate
python examples/phase5_fee_wars.py --workers 2 --report out/fee_wars.html
```

## Setup

| piece | where | what |
|-------|-------|------|
| `LotFee` | `experiments/two_venue.py` | per-lot integer-tick maker/taker fee; a negative maker fee is a rebate paid from the venue sink |
| `TwoVenueMarket` | `experiments/two_venue.py` | two fl-031 `RouteVenue` books with one population, a `VenueRouter` for takers, and maker venue choice |
| `fee_grid`, `best_response` | `experiments/fee_wars.py` | strategy space and the tie-broken argmax of venue revenue |
| `detect` | `experiments/fee_wars.py` | fixed-point / cycle detector on the `(fee_A, fee_B)` path |
| `run_fee_wars`, `study_report` | `experiments/fee_wars.py` | the best-response loop as one `ExperimentResult`, plus the fl-026 HTML |

The market, step by step:

1. With probability `jump_prob` the fundamental V moves by `+-jump`. Makers
   only see last step's value. Informed takers see the current one.
2. Each maker cancels and requotes `quote_size` lots at
   `V_pub -+ half_spread`, skewed by inventory. Quotes are post-only, so a
   maker never crosses a rival on the same venue. More makers on a venue
   means more depth there and a touch that is at least as tight.
3. Zero to `max_arrivals` takers arrive: ZI-C (private value
   `V_pub + U[-6, 6]`), noise (always trades) or informed (trades in the
   direction of the jump). The router quotes the all-in cost on both venues:
   price walked through depth, plus taker fee, plus fixed cost. It picks the
   cheaper venue. On an exact tie a pre-drawn coin decides. ZI-C and informed
   takers trade only when their value beats the all-in price.
4. Every `epoch` steps each maker re-scores both venues as
   `-maker_fee - AS_loss_per_unit` and moves if the other venue is better by
   at least `switch_cost`. `AS_loss_per_unit` is the fl-030 adverse
   component, `-sum s_j (V_t - ref_j) q_j`, over all maker fills on that
   venue, shrunk toward 0 by `as_prior` lots.

All random draws happen in the same order whatever the fees are. So two
probes with the same seed see the same arrivals and the same V path, and
differ only through the fee schedule. That is what makes the unilateral-cut
check meaningful: on the same path, a taker-fee cut keeps the orders the
venue already won and can pull marginal orders from the rival. Maker
relocation reacts to flow through the measured adverse-selection loss, so
this is checked on a grid of fees, rivals and seeds rather than assumed.

## The best-response loop

A round has two moves, A then B. The venue that moves evaluates every grid
pair against the rival's current pair through `run_experiment` (fl-023).
It uses explicit per-replicate seeds, which are the same at every probe and
in every round. It picks the pair with the highest summed sink revenue.
Revenue is an integer number of ticks, so ties are exact and are broken
toward the lower taker fee, then the lower maker fee. Because the probe
seeds are fixed, the map from state to best response is a deterministic
function, and the state path must either settle or repeat.

`detect(path, window)` fires with:

- `fixed_point` once the state has not changed for `window` rounds;
- `cycle` with length `k` as soon as a state reappears after a different one.
  `cycle_states` holds one period of the cycle.

## Bertrand undercutting

Takers treat the two venues as perfect substitutes up to the all-in cost.
A venue charging one tick more per lot loses almost all routed flow, so each
venue's best response is to undercut the rival's taker fee by one tick
whenever the volume it gains is worth more than the margin it gives up.
This is the Bertrand logic, played out on an integer grid. On seed 32 the
path is

    (0,4)/(0,4) -> (0,3)/(-1,4) -> (-1,3)/(-2,4) -> (-2,4)/(-2,3) -> (-2,3)/(-2,3)

A cuts the taker fee. B answers with a deeper maker rebate, which pulls
every maker, and so all the depth, onto B (share_A = 0). A matches the
rebate. The war ends at `(-2, 3)` on both venues, with share_A = 0.47.

## Why the rebate is bounded

A venue earns `maker_fee + taker_fee` per lot, so a rebate deeper than the
taker fee loses money on every trade. The grid enforces `m + t >= 0`. That
is a hard bound, but it is not the binding one here. The binding one comes
from the makers. A rebate is only worth paying if it brings makers, and
makers weigh the rebate against what they lose to informed flow on that
venue. Once both venues pay the same rebate, makers spread across them, and
a further cut would give up `1` tick on every lot to win flow that costs the
same as before. The outcome net take is `1` tick per lot, down from `4`.
Adverse selection caps how much a rebate can buy, because each extra lot of
maker depth on a venue also takes on that venue's informed flow.

## Reading the share and revenue paths

`round_rows` (and the example table) give, per round:

- `share_A`: volume share by lots. `share_A + share_B = 1` in every round
  that has volume. A swing to 0 or 1 means the makers have all moved.
- `revenue_V`: net sink balance. It equals `gross_fees_V - rebates_V` and
  `(maker + taker) * volume_V` exactly in ticks (`fee_residual = 0`).
- `rebates_V`: total maker rebates paid. Large rebates with low revenue mean
  the venue is buying depth.
- `spread_V`: mean quoted touch per step. With no makers on a venue its
  book is empty and the spread is recorded as 0.
- `makers_V`: average number of makers on the venue per epoch.
- `friction_per_lot`: total sink revenue over volume. This is what traders
  pay as a group, and it falls to the war's floor.
- `net_surplus`: gross gains at private values minus friction (fl-022
  style). It rises as fees fall.

## Invariants (tested in `tests/test_experiment_fee_wars.py`)

- Venue shares sum to 1 in every round; ledger violations are 0 in every cell.
- `revenue + rebates == gross fees` and `revenue == (m + t) * volume` exactly.
- Identical fee schedules give `|share_A - 0.5| <= 0.5` on the fixed seed.
- A unilateral taker-fee cut never lowers the cutter's volume share.
- Friction per lot at the detected outcome is `<=` the starting fee level.
- `detect` fires on a constant path and reports a constructed 2-cycle.
- Serial and `n_workers=2` rows are identical, and reruns from the master
  seed reproduce them.
