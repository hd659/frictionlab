# Phase 2: agents, and why informed flow widens spreads

Phase 2 adds agents that observe a `MarketView`, pay an explicit
`decision_cost`, and emit typed `Submit` / `Cancel` intents that the `Venue`
executes and settles through the friction stack and ledger.

| Module | Role |
| --- | --- |
| `frictionlab/agents/base.py` | `Agent` (observe -> decide -> act), `Venue`, `MarketView`, decision-cost sink |
| `frictionlab/agents/zero_intelligence.py` | Gode-Sunder ZI-U / ZI-C traders |
| `frictionlab/agents/market_maker.py` | Avellaneda-Stoikov inventory-risk maker |
| `frictionlab/agents/fundamental.py` | seeded latent value V(t): constant or compound-Poisson jump walk |
| `frictionlab/agents/informed.py` | informed trader that trades only when edge > total friction |
| `frictionlab/agents/noise.py` | value-blind Poisson marketable flow |
| `frictionlab/core/runner.py` | `MarketRunner`: Poisson wake-ups, latency, settlement, invariant checks |
| `frictionlab/analysis/spread.py` | time-weighted quoted spread, PnL by agent / type, markouts vs V(t) |
| `examples/phase2_zi_efficiency.py` | ZI-C efficiency vs the competitive equilibrium as the taker fee rises |
| `examples/phase2_adverse_selection.py` | quoted spread and PnL as the informed share rises |

```bash
python examples/phase2_zi_efficiency.py        # seed 11
python examples/phase2_adverse_selection.py    # seed 7
```

## 1. The adverse-selection problem

A market maker posts a bid and an ask and has to trade with whoever arrives.
Two kinds of takers arrive:

- noise traders trade for reasons unrelated to value. Their buys and sells
  are balanced on average, so each fill earns the maker about one half-spread;
- the informed trader knows V(t). It buys only when `V - ask` beats its
  friction and sells only when `bid - V` does, so every informed fill is one
  the maker will regret once V becomes public.

Glosten and Milgrom (1985) show that a competitive, risk-neutral maker sets
quotes so that its expected profit on each side is zero:

    ask = E[V | a buy arrives],   bid = E[V | a sell arrives]

With informed share `pi` and an expected value move `L` per informed fill,
the zero-profit condition per side is roughly

    (1 - pi) * half_spread  =  pi * (L - half_spread)
    -> half_spread = pi * L

So the spread scales with `pi * L`. When the maker knows how much informed
flow it faces, the spread is the price of information. Kyle (1985) gets the
same result in a batch auction: price impact `lambda` grows with the ratio of
value uncertainty to noise volume.

## 2. What the example implements

`GMMarketMaker` extends `AvellanedaStoikovMM` in two ways:

1. Premium. Each side is widened by `ceil(pi * adverse_ticks)` ticks on top
   of the A-S inventory spread. This is the reduced-form zero-profit
   condition, with `adverse_ticks` standing in for L.
2. Belief. Its mid is a belief, not the book mid, since the maker's own
   quotes make up most of the book. Each new print moves the belief by
   `learn * (price - mid)`. Informed buys lift it, informed sells lower it,
   so quotes follow V(t) through order flow, the way Glosten-Milgrom quotes
   update after each trade.

Total taker intensity stays at `lam`. A share `pi` goes to the informed
trader and the rest is split across the noise traders. Changing `pi`
therefore changes who trades, while the amount of flow stays the same.

The metrics come from `frictionlab.analysis`:

| Column | Definition |
| --- | --- |
| `spread` | time-weighted `ask - bid` over the time the book is two-sided (`QuoteRecorder` samples the touch after every wake-up) |
| `cover` | fraction of the horizon with a two-sided book |
| `mm_pnl`, `inf_pnl`, `noise_pnl` | `sum side * (V_end - price) * qty - fees - decision costs` |
| `mm_vs_inf` | maker markout against informed fills, valued at V at trade time (gross of fees) |
| `mm_vs_noise` | the same markout against noise fills |

Seed 7, `lam = 4`, horizon 300, `adverse_ticks = 10`, taker fee 10 bps:

```
   pi  spread  cover  volume   mm_pnl mm_vs_inf mm_vs_noise  inf_pnl noise_pnl
 0.00    2.88   0.90    1159   -26179         0      -25975        0     25020
 0.10    5.13   0.95    1133   -14715    -10899       -4216    10625      2957
 0.20    6.36   0.95    1029    -6250     -9085        2879     8571     -3350
 0.30    9.68   0.97    1035    -3650     -6966        3148     6232     -3617
 0.40   11.79   0.99     809       93     -3041        3214     2667     -3569
 0.50   13.77   0.99     819      555     -4226        4485     3694     -5068
```

## 3. Reading the table

- The quoted spread rises with `pi`, from under 3 ticks to almost 14. Most of
  that is the premium. A wider spread also makes noise prints scatter
  further, so the belief moves around more.
- `mm_vs_inf` is never positive. The informed trader only trades when it has
  an edge, and the edge has to be at least the fee, so each of its fills is a
  loss for the maker when valued at V. `inf_pnl` is that same transfer minus
  taker fees.
- The spread is how the maker pays for this. As `pi` rises, the maker's
  markout against noise moves from negative to clearly positive. Noise
  traders pay for the information through the spread (`noise_pnl < 0`),
  which is the Glosten-Milgrom transfer from the uninformed to the informed.
- The informed trader's PnL mostly falls as the premium grows (not strictly:
  0.40 -> 0.50 ticks up on this seed). The edge it needs to
  cover the half-spread and fee gets bigger, and the maker's belief follows
  V more quickly, so fewer opportunities clear its friction test.
- Volume drops when `pi` is high. Wider quotes push some marginal trades out
  of the market. This is the Phase 5 "Coase threshold" question seen from
  the side of information.

The noisiest number is `mm_pnl`. It marks the maker's inventory at the final
V, and a single fundamental jump on a large position can move it by thousands
of tick-lots. The tests therefore check the markout against informed flow,
which does not depend on how inventory is marked at the end.

## 4. Invariants the tests enforce

`tests/test_spread_metrics.py` and `tests/test_example_phase2_adverse_selection.py` check that:

- ledger conservation holds after every run (cash, inventory, and the
  fee and decision-cost sinks);
- the same seed gives identical rows, and different seeds give different rows;
- across the `pi` sweep, the average quoted spread never goes down;
- the maker's aggregate markout against informed flow is `<= 0`;
- PnL by agent adds up to minus the fees and decision costs collected, because
  trading between agents is zero-sum before friction.
