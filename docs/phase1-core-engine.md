# Phase 1: the core engine, and how per-trade friction shifts realized PnL

Phase 1 delivers five modules plus one example that wires them together:

| Module | Role |
| --- | --- |
| `frictionlab/core/clock.py` | `SimClock`: seeded RNG, event queue ordered by `(time, priority, seq)` |
| `frictionlab/core/types.py` | `Order`, `Trade`, `Account`; prices in integer ticks, cash in tick-lots |
| `frictionlab/core/order_book.py` | price-time priority book: partial fills, cancels, self-trade prevention |
| `frictionlab/costs/` | `CostModel` knobs (`MakerTakerFee`, `HalfSpreadCost`, `LatencyCost`, `FixedCost`) composed into a `FrictionStack` |
| `frictionlab/core/ledger.py` | `Ledger`: atomic settlement net of friction, fees sent to a venue sink, conservation checks |
| `examples/phase1_friction_walkthrough.py` | one seeded market replayed under four friction regimes |

```bash
python examples/phase1_friction_walkthrough.py        # seed 7
python examples/phase1_friction_walkthrough.py 11     # any other seed
```

## 1. Pipeline

```
SimClock --fair/quote/flow events--> agents build Orders
         --"arrive" event after stack.delay--> OrderBook.submit -> [Trade]
         --> Ledger.settle(trade): FrictionStack.trade_fees -> (buyer_fee, seller_fee)
         --> Account cash/position update, venue sink += fees, invariants checked
```

The example has one market maker (agent 1) that quotes `fair +- 2` ticks and
re-quotes with Poisson rate 2/s. Four takers (agents 2-5) send Poisson flow:
70% market orders, 30% passive limits. Fair value moves +-3 ticks as a Poisson
jump process, and inventory is marked at the final fair value.

## 2. The controlled-experiment design

In Phase 1 the order flow does not react to costs. So for a fixed seed:

- every regime produces the same tape (price, qty, buyer, seller, aggressor).
  `main()` checks this and returns exit code 1 if any tape differs;
- latency moves every order by the same `stack.delay`, so timestamps shift
  but the relative event order, and therefore the matching, stays the same;
- for each agent, `gross` PnL is identical across regimes, and
  `net = gross - fees_paid` exactly, in integer tick-lots.

PnL per agent, marked at fair value `m`:

```
net_i   = equity_i(m) - (cash0 + pos0 * m)
gross_i = net_i + fees_i
sum_i gross_i = 0                       (trading is zero-sum before friction)
sum_i net_i   = -venue_fees             (friction is a transfer to the venue)
```

## 3. Results (seed 7, 400 taker decisions, 321 trades)

| Regime | Stack | Maker net | Takers net (sum) | Venue |
| --- | --- | ---: | ---: | ---: |
| frictionless | none | 2828 | -2828 | 0 |
| rebate_venue | maker -1 bps, taker 5 bps | 3118 | -5977 | 2859 |
| retail | taker 10 bps + half-spread(2) + 5 per fill | 1405 | -11903 | 10498 |
| slow_retail | retail + 50 ms latency at 20 ticks/s drift | 1378 | -12511 | 11133 |

How to read the table:

1. **Frictionless.** The maker earns the spread: takers cross 2 ticks from
   fair, and fair-value jumps only partly offset that gain. All four takers
   lose money gross. This is the price of immediacy even with zero fees.
2. **Rebate venue.** The maker's net PnL is *above* its gross PnL
   (fees = -290), because the venue pays for resting liquidity out of taker
   fees. Takers pay about 2x their frictionless loss.
3. **Retail.** The flat `FixedCost(5)` charges both legs, so the maker pays
   about 5 * 222 fills, which cuts its net roughly in half. The taker-only
   half-spread and the 10 bps fee fall on the aggressors. Venue revenue
   (10498) is now larger than the maker's whole gross edge (2828).
4. **Slow retail.** `LatencyCost` adds `0.05 s * 20 ticks/s = 1 tick` per
   taker lot on top of retail. The tape is unchanged. Only the bill grows.

Friction does not change who wins gross in this phase. It changes how much
of the gross surplus stays with the traders. Once fees exceed the maker's
edge (`FrictionStack.round_trip` > spread captured), the maker's net goes
negative, and an adaptive maker would widen its quotes or exit. Modelling that
threshold is the job of Phase 2 agents and the Phase 5 Coase experiment.

## 4. Invariants the run enforces

`Ledger(check_every=True)` checks the following after every settlement:

- `sum(agent cash) + venue cash == cash deposited` (fees and rebates included);
- `sum(agent position) == inventory deposited`, and the venue holds no inventory;
- `venue cash == sum(fees_paid) == sum(journal fees)`.

Fees are rounded once per fill in the venue's favour (ceil over an exact
`Fraction` sum), so totals are exact integers and reproducible across
platforms.

## 5. Known limitations (Phase 2 inputs)

- Flow does not react to cost, so friction cannot yet push volume down.
  The Coase-threshold question needs agents that observe, decide and act.
- `LatencyCost` charges expected adverse drift as a fee instead of
  simulating the move itself. A latency-aware book (stale quotes get
  picked off) is future work.
- The maker never manages inventory risk. Avellaneda-Stoikov quoting is a
  Phase 2 milestone.

## 6. Tests

`tests/test_example_phase1.py` pins the example as a regression:
same seed gives a bit-identical tape and PnL, a different seed gives a different
tape, the tape is the same across regimes, `net == gross - fees` for every
agent, gross PnL sums to zero, venue revenue equals the agent fee total, and
fees rise monotonically from frictionless to slow retail.
