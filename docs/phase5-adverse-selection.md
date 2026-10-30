# Phase 5.3: adverse selection and the Glosten-Milgrom benchmark

A market maker who trades with someone better informed loses on those trades
and has to make the loss back from everyone else through the spread. This
experiment puts the simulated spread next to the Glosten-Milgrom (1985)
zero-profit spread, splits the maker's PnL into what it earns from the spread
and what it loses to informed flow, and checks whether a frequent batch
auction reduces that loss.

```
python examples/phase5_adverse_selection.py                 # 6 shares x 3 jumps, 3 replicates
python examples/phase5_adverse_selection.py --small         # 3 shares x 2 jumps, 2 replicates
python examples/phase5_adverse_selection.py --workers 2 --report out/as.html
```

## Setup

| piece | where | what |
|-------|-------|------|
| `gm_zero_profit_spread` | `experiments/adverse_selection.py` | analytic GM bid/ask in exact Fractions, put on the integer tick grid |
| `GMMaker`, `build`, `metrics` | `experiments/adverse_selection.py` | maker / informed / noise market and per-cell metrics |
| `maker_split` | `experiments/adverse_selection.py` | maker trading PnL = spread capture + adverse-selection part |
| `mitigation`, `run_study` | `experiments/adverse_selection.py` | continuous vs batch rerun; one `ExperimentResult` |
| `study_report` | `experiments/adverse_selection.py` | fl-026 HTML: findings, share x jump heatmap, structure table |

## The analytic benchmark

V is `v_low` or `v_high` with prior 1/2. A share `mu` of takers is informed
and buys iff V is high. The rest is noise that buys with probability `q`.
A risk-neutral maker facing competition quotes the conditional expectation
of V given the side of the next order:

    P(H | buy)  = (mu + (1-mu) q) / (mu + 2 (1-mu) q)
    P(H | sell) = ((1-mu)(1-q)) / (mu + 2 (1-mu)(1-q))
    ask = v_low + (v_high - v_low) * P(H | buy)
    bid = v_low + (v_high - v_low) * P(H | sell)

With balanced noise (`q = 1/2`) this reduces to `ask = mean + mu * L` and
`bid = mean - mu * L`, where `L = (v_high - v_low) / 2`. So the GM spread is
`2 * mu * L`. The function computes both quotes exactly and returns
`(floor(bid), ceil(ask))`. Rounding outward means the integer quote never
loses money in expectation. Three properties follow, and
`tests/test_gm_benchmark.py` checks each of them:

- `mu = 0`: both quotes equal the prior mean and the spread is zero. Noise
  flow carries no information.
- Symmetry: with `q = 1/2`, `ask - mean == mean - bid` exactly. Swapping
  `q` and `1 - q` mirrors the quotes around the mean.
- Monotonicity: the spread is non-decreasing in `mu`, and at `mu = 1` it
  reaches `v_high - v_low`.

## The simulated market

`build` sets up a seeded jump random walk V(t) with jumps of size `J`, plus:

- a `GMMaker`: Avellaneda-Stoikov quotes around a belief mid that moves
  toward each print by `learn * (p - mid)`, widened on each side by the GM
  premium for a two-point value `v0 +- L`, where `L = alpha * J`. The maker
  knows `mu`, which is the GM rational-expectations assumption.
- an `InformedTrader` arriving at rate `mu * lam`. It trades only when its
  edge beats the taker fee.
- noise traders with total rate `(1 - mu) * lam`.

The grid is `share x jump`, run on the fl-023 engine with common random
numbers, so replicate r uses the same seed in every cell. Per cell:

- `tw_spread` (time-weighted quoted spread), `gm_spread`, `excess_spread`.
  The excess is the Avellaneda-Stoikov inventory component.
- `effective_spread`, `realized_spread` and `price_impact` at `tau = 5`
  (fl-022). `identity_residual` is the largest per-trade
  `|effective - realized - impact|`. It is 0.
- `mm_spread_capture`, `mm_adverse_selection`, `mm_as_loss`,
  `mm_as_per_unit`, and `split_residual`, which is 0.
- `violations` and `zero_sum_residual`: ledger health. Both are 0.

### The PnL split

For each maker fill j, with maker side `s_j`, price `p_j`, size `q_j`, the
integer reference `r_j = floor(pre-trade book mid)` and the final value `V_T`:

    trading = sum s_j (V_T - p_j) q_j
            = sum s_j (r_j - p_j) q_j  +  sum s_j (V_T - r_j) q_j
            = capture                  +  adverse

The split is exact in integer ticks because it only adds and subtracts
`r_j`. Capture is the half-spread the maker earned against the mid it was
quoting around. Adverse is how far the value moved against the maker's
position after the fill, and `mm_as_loss = -adverse`. Using the doubled mid
`M = bid + ask` keeps `r_j = M // 2` on the tick grid.

## Results (seed 30, horizon 200, lam 4, alpha 5, 3 replicates)

| mu | J | quoted | GM | impact | AS loss / unit |
|----|---|--------|----|--------|----------------|
| 0.0 | 8 | 2.61 | 0 | 6.42 | 3.38 |
| 0.1 | 8 | 10.32 | 8 | 11.21 | 6.50 |
| 0.2 | 8 | 18.36 | 16 | 18.98 | 10.24 |
| 0.3 | 8 | 26.87 | 24 | 16.76 | 6.82 |
| 0.4 | 8 | 35.08 | 32 | 15.95 | 6.99 |
| 0.5 | 8 | 42.95 | 40 | 17.99 | 7.40 |

- The quoted spread follows the GM benchmark plus a roughly constant
  inventory component of about 2.6 to 3 ticks. It is non-decreasing in `mu`
  at every jump size.
- Mean price impact per trade, averaged over cells with informed flow, rises
  with the jump size: about 13.2, 13.5 and 16.2 ticks at J = 2, 4 and 8.
  Larger jumps mean every informed fill moves the mid further.
- Realized spread becomes positive once the GM premium is large enough:
  from `mu = 0.3` at J = 8, `mu = 0.4` at J = 4 and `mu = 0.5` at J = 2. The premium is then covering the impact, which is the
  GM break-even logic seen through the fl-022 decomposition.

## Mitigation: batch auction

`mitigation` reruns the `(max share, max jump)` cell twice on the same seeds,
once on the continuous book and once through `batch_runner` (fl-020,
interval 1). At `mu = 0.5`, `J = 8`:

| structure | quoted | maker volume | AS loss / unit |
|-----------|--------|--------------|----------------|
| continuous | 42.95 | 521.0 | 7.40 |
| batch | 31.16 | 224.7 | 3.38 |

The batch auction roughly halves the maker's adverse-selection loss per unit.
Informed orders that arrive within one interval of a jump now clear together
with noise orders at one uniform price. They no longer pick off a stale quote
one at a time, so less informed volume reaches the maker at stale prices.
The cost is lower maker volume. The headline number is
`findings["as_per_unit_batch_minus_continuous"]`.

## Reproducibility

Every number above is a function of `(master_seed, grid, Config)`. Child
seeds come from one `SeedSequence`, so `n_workers=2` gives the same rows as a
serial run, and the HTML report is byte-identical across runs.
