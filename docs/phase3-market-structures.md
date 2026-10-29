# Phase 3: market structures

Phase 3 adds venues beyond the continuous limit order book. Each one settles
through the same friction stack and ledger, so cash and inventory
conservation hold in every structure.

| Module | Role |
| --- | --- |
| `frictionlab/markets/amm_cp.py` | constant-product pool (x * y = k), integer reserves, LP fee in bps |
| `frictionlab/markets/amm_cl.py` | concentrated-liquidity pool, ranges `[tick_lower, tick_upper)` |
| `frictionlab/markets/perp.py` | perpetual futures, clamped periodic funding, funding sink |
| `frictionlab/markets/batch_auction.py` | frequent batch auction, uniform volume-maximizing price, pro-rata |
| `frictionlab/agents/arbitrageur.py` | book <-> AMM arbitrageur, trades only when net edge > total friction |
| `frictionlab/analysis/structure_compare.py` | volume, effective spread, snipe share per structure |
| `examples/phase3_market_structures.py` | the three experiments below |

```bash
python examples/phase3_market_structures.py     # seed 7
```

All numbers below are from seed 7. `tests/test_example_phase3_structures.py`
checks the qualitative claims on several seeds.

## 1. Continuous book vs batch auction under latency

Agent mix: 6 + 6 value-anchored ZI-C traders, one informed trader and three
noise traders. A ZI-C buyer quotes uniformly in `[V - 3, V]` and a seller in
`[V, V + 3]`, where `V` is the value it saw when it woke up. `LatencyCost(d)`
delays every order by `d` seconds, so after a jump of V (size 5) the resting
quotes are stale until their owners wake up again and their replacements
arrive. The batch auction clears once per second.

A fill counts as a snipe if it prints within `interval + d` of a jump and the
aggressor gains against the new V.

| latency | cont. volume | batch volume | cont. snipe% | batch snipe% | informed PnL (cont / batch) |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 529 | 321 | 21.6 | 17.8 | -44 / 0 |
| 0.5 | 588 | 395 | 29.8 | 25.1 | -25 / -16 |
| 1.0 | 641 | 467 | 34.5 | 28.7 | -66 / 22 |
| 2.0 | 730 | 567 | 40.7 | 31.7 | -97 / -49 |

What the table shows:

- Snipe share rises with latency in both structures, because stale quotes
  live longer. The batch auction's share is lower at every latency, and the
  gap is widest at `d = 2` (9 points). In a continuous book the first taker
  to arrive hits a stale quote at its own price. In a batch every response
  inside the interval competes for the same quote, and the uniform price
  moves toward the new V. This is the Budish-Cramton-Shim argument for
  frequent batch auctions: the race becomes a price auction.
- The batch prints less volume. Orders that would cross one at a time in a
  continuous book get netted against each other at one clearing price.
- Effective spreads are negative here, so takers gain against V on average.
  That is the stale-quote rent. It grows with latency in both structures.
  The batch auction cuts the rent's frequency (fewer snipes) more than its
  size per snipe, so this metric does not separate the two structures
  cleanly on a single seed.
- Informed PnL is marked at V(T), net of fees and decision costs. It is
  small and noisy because the informed trader has a capped position and
  competes with the ZI-C traders, who also track V. It is reported, not
  asserted.

## 2. Cross-venue arbitrage and the no-arbitrage band

The pools start at spot 100 and charge a 30 bps LP fee. The book charges a
10 bps taker fee. The concentrated-liquidity pool puts roughly the same
capital on `[80, 125)`. Gas is a `FixedCost` charged on each swap.

The band width is the smallest buy-side gap plus the smallest sell-side gap
between book touch and pool spot at which `CrossVenueArbitrageur` finds a
trade with positive net edge. The residual gap is pool spot minus book ask
after 300 wake-ups against a deep static ask at 92.

| pool | gas | band (ticks) | residual gap | arb trades | arb net PnL |
| --- | --- | --- | --- | --- | --- |
| cp | 0 | 2 | 0.50 | 1 | 1513 |
| cp | 50 | 4 | 0.50 | 1 | 1463 |
| cp | 200 | 8 | 0.50 | 1 | 1313 |
| cp | 800 | 13 | 0.50 | 1 | 713 |
| cl | 0 | 2 | 0.41 | 11 | 15134 |
| cl | 50 | 2 | 0.48 | 10 | 14631 |
| cl | 200 | 4 | 1.19 | 9 | 13146 |
| cl | 800 | 6 | 2.64 | 7 | 8116 |

The arbitrageur trades only if

    gross edge(q) > book fee(q) + LP fee(q) + price impact(q) + gas + decision_cost

Gas is a fixed cost, so it has to be spread over the trade size. The
smallest gap that pays is roughly where `q * gap - impact(q)` first exceeds
`gas`, so the band widens with gas. Tests assert that it never narrows.
The CL pool's band grows more slowly: price impact is lower for the same
capital, so a given gap supports a larger `q` and gas is spread over more
lots.

Residual gap: the cp pool is shallow, so one optimally sized swap moves its
spot almost all the way to the book and leaves the half-tick that fees and
rounding cannot cover. The CL pool is deep near spot, so it takes several
trades to close. Each further trade needs a larger gap as gas grows, and the
pool settles further from the book: 0.41 ticks at gas 0 vs 2.64 at gas 800.
The arbitrageur's net PnL per trade is non-negative throughout.

## 3. Perpetual funding pulls mark toward index

The book belongs to a maker that quotes +/- 1 tick around a belief that
follows prints. The maker does not know the index. Noise flow buys with
probability 0.6, so without any counterweight the mark drifts up without
bound. The index is a mild jump random walk around 100. Funding is
`clamp(premium_bps * k, -100, 100)` every 4 seconds.

The carry trader reads the last funding rate and takes the receiving side
whenever `|rate| >= 2 bps`: it sells when longs pay and buys when shorts pay.
With `k = 0` the rate is always 0, so it never trades.

| funding | mean abs premium (bps) | carry position |
| --- | --- | --- |
| on (k = 1) | 193.0 | -422 |
| off (k = 0) | 2068.5 | 0 |

The rate path spends most periods at the +100 bps cap and occasionally flips
sign when the carry trader overshoots. Funding is the only force tying the
perp to its index. Once it is switched off, the one-sided noise flow sets the
mark, and the premium is roughly ten times larger. Funding transfers net to
zero across accounts plus the sink, and the sink keeps only rounding
remainders, so the ledger invariants hold in both runs.

## Phase 3 status

Phase 3 is complete. Every structure settles through one ledger. The
example reproduces from a seed, and its claims are regression-tested:

- ledger invariants hold in every scenario;
- results are identical across reruns with the same seed;
- the no-arbitrage band width is non-decreasing in gas cost, for both pools;
- at the highest latency, the batch auction's snipe share is <= the
  continuous book's;
- mean abs premium with funding on is <= the same run with funding off.

Phase 4 turns these one-off sweeps into a toolkit with metrics, parallel
parameter sweeps and equilibrium detection.
