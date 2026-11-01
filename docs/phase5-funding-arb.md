# Phase 5.5: funding-rate arbitrage under gas shocks

A perpetual future stays near its index only because someone is paid to
hold the other side of one-sided flow. That someone is the basis trader: short
perp plus long spot when funding is positive, the mirror when it is negative.
Every leg of that trade pays gas. This experiment asks how much basis survives
once gas can spike, and how the answer depends on how often and how long the
spikes last.

```
python examples/phase5_funding_arb.py --small          # 5 levels x 3 probabilities, 1 replicate
python examples/phase5_funding_arb.py --out out/       # 8 levels x 4 probabilities, 2 replicates
python examples/phase5_summary_report.py --out out/    # all five Phase 5 experiments, one HTML
```

## Setup

| piece | where | what |
|-------|-------|------|
| `GasProcess` | `costs/gas.py` | two-regime Markov chain (calm / shock), integer-tick gas per fill |
| `FundingArbitrageur` | `agents/funding_arb.py` | delta-neutral perp/spot basis trade, enters only when edge > friction |
| `InventoryMaker`, `RetailFlow` | `experiments/funding_arb.py` | Ho-Stoll skewed perp ladder; book-blind one-sided retail flow |
| `run_funding_arb` | `experiments/funding_arb.py` | shock_gas_level x shock_probability sweep as one `ExperimentResult` |
| `entry_threshold_gas`, `find_gas_threshold` | `experiments/funding_arb.py` | analytic per-leg gas bound; bisected entry-collapse level |

Retail buys 80% of the time, so the maker ends up short and skews its ladder
up. Mark sits above index and funding is positive, so longs pay shorts. The
arbitrageur sells perp, buys spot on a constant-product pool, and collects
funding until the horizon expires or the carry turns adverse. While it holds,
it absorbs the maker's inventory and the skew shrinks. When it stays out, the
premium stays wide. Retail flow is book-blind, so the only way gas reaches the
premium is through the arbitrageur's entry decision.

## The entry condition

At size q lots, horizon h funding periods and expected funding rate E per
period (in ticks for the whole position), entry is individually rational iff

```
E * h  >  fees + impact + 4 * gas + decision_cost
```

- `fees`: perp taker fee on the entry and the exit (2 legs).
- `impact`: AMM LP fee plus constant-product price impact, both ways, plus the
  cost of crossing the perp spread.
- `4 * gas`: one fill per leg, four legs per round trip (perp in, spot in,
  perp out, spot out). Gas is a fixed charge per fill, independent of size.
- `decision_cost`: the arbitrageur's own deliberation charge.

Solving for gas gives `entry_threshold_gas`, the largest integer g with
`E * h - fees - impact - decision_cost - 4 g > 0`:

```
g_max = ceil(slack / 4) - 1,   slack = E * h - fees - impact - decision_cost
```

`analytic_bound(cfg)` evaluates it at the friendliest state the market allows:
the funding rate at its cap, mark at the top of the ladder, the fee at the
bottom, and no impact. With the default config that is
`12 * 10.11 * 4 - 23.74 = 461.5` ticks of slack, so `g_max = 115`. No entry can
happen at any gas above 115, in any regime. The experiment tests this.

## Why gas shocks widen the tolerated basis

The condition is a band. Gas is a fixed cost, so it does not scale with q and
it sits in the denominator of a hurdle rate:

```
required premium per period  ~  (fees + impact + 4 * gas + dc) / (q * h)
```

When gas rises, the smallest premium worth arbitraging rises with it, and
any basis inside the new band is left alone. The maker's skew is no longer
absorbed, so the mean absolute premium (mark over index, bps) goes up. On the
fixed seed it never goes down as shock gas rises, at every shock probability.
That is the monotonicity the tests assert. It is the perp analogue of the
Phase 3 no-arbitrage band that widens with gas: friction does not change
where the price should be, it changes how far the price can drift before
anyone is paid to correct it.

The bisected collapse level is well below the analytic bound: 52 vs 115. The
analytic bound is evaluated at the capped rate and the top of the ladder. The
realized funding rate is usually smaller and the AMM leg pays impact, so
the slack in typical states is much smaller than in the extreme state. The
bound is an upper limit, not a prediction.

## Two regimes: the effective threshold

The shipped arbitrageur is myopic. It prices all four legs at current gas.
Gas, however, follows a two-state chain:

```
CALM -> SHOCK with p_shock,   SHOCK -> CALM with p_calm,   per gas step
pi_shock = p_shock / (p_shock + p_calm)
```

A position opened in calm pays its two exit legs at whatever gas prevails
when it closes. When the holding horizon (here 4 periods x 4 s = 16 gas steps)
is long relative to the chain's mixing time `1 / (p_shock + p_calm)`, the exit
regime is close to a draw from the stationary distribution. A forward-looking
trader would then require

```
slack  >  2 * g_now + 2 * (pi_calm * g_calm + pi_shock * g_shock)
```

and the shock level at which calm-state entry stops is

```
g_shock*  =  (slack / 2 - g_calm * (1 + pi_calm)) / pi_shock
```

The effective threshold therefore falls as pi_shock rises. At pi_shock -> 0 it
goes to infinity, because shocks are irrelevant. At pi_shock -> 1 it goes to
slack / 2 - g_calm: a calm-state entry is then certain to exit in shock, so
only the two exit legs pay shock gas. Entries made during a shock face all
four legs at shock gas and the myopic bound slack / 4 applies. With the default
`p_calm = 0.05` shocks are persistent: p_shock = 0.6 gives pi_shock = 0.923,
the chain is in shock for 190 of 200 steps, and every entry happens in one of
the 10 calm steps.

The regime-conditioned table in the example shows what myopia costs. At shock
gas 116 and p = 0.6, entries are 0.40 per calm step and 0 per shock step. Of
the 4 positions that were open when a shock hit, 3 were held through the
spell rather than unwound, because exiting would itself cost 2 x shock gas.
Net arbitrageur PnL is -699 ticks, compared with +914 at constant gas: gas of
1059 ticks against 480 of gross funding. The loss comes from exit legs paid
at shock gas on calm-state entries, which is the cost the myopic rule leaves
out. A trader that priced exits at stationary gas would demand more slack
per entry and would trade less in calm spells.

## Seeded headline numbers (seed 34, full grid, 2 replicates)

| quantity | value |
|----------|-------|
| analytic per-leg gas bound | 115 ticks |
| bisected collapse level at p = 0.6 (entries < 50% of baseline) | 52 ticks |
| entries, constant gas -> shock gas 116 at p = 0.6 | 12.0 -> 4.5 |
| mean abs premium, same cells | 27.3 -> 43.3 bps |
| stationary shock share at p = 0.6 | 0.923 |
| net arb PnL (replicate 0), gas 1 -> gas 116 | +914 -> -699 ticks |

With p = 0 every shock level reproduces the constant-gas baseline exactly.
Funding plus fees plus gas reconciles with ledger balances in integer ticks in
every cell, and serial and `n_workers=2` rows are identical.

## Reading the report

`examples/phase5_funding_arb.py` writes `funding_arb.html` into `--out`, or
into a fresh temp directory if `--out` is not given. The report has the
findings table, a heatmap of mean |premium| over shock gas x shock probability
(the first column is flat because p = 0 means no shocks), the arbitrageur
PnL, and a friction breakdown into gross funding, perp fee, AMM LP fee plus
impact, gas and decision cost for the baseline cell and the worst shock cell.
`examples/phase5_summary_report.py` reruns all five Phase 5 experiments in
`--small` mode and writes one byte-reproducible HTML with a section per
experiment.
