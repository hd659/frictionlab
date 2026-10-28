# Phase 2: agents, adverse selection and the cost of thinking

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
| `frictionlab/agents/bayesian.py` | Glosten-Milgrom posterior over V from trade signs |
| `frictionlab/agents/bounded_rational.py` | satisficing vs optimizing search, metered per step |
| `frictionlab/analysis/pnl.py` | per-agent PnL split into trading, fees and decision costs |
| `examples/phase2_decision_cost.py` | satisficer vs optimizer net PnL as `cost_per_step` rises |

```bash
python examples/phase2_zi_efficiency.py        # seed 11
python examples/phase2_adverse_selection.py    # seed 7
python examples/phase2_decision_cost.py        # seed 15
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

## 5. Decision cost: when over-analysis loses to satisficing

`examples/phase2_decision_cost.py` closes Phase 2. Four
`BoundedRationalAgent`s (two buyers, two sellers, private valuations drawn
from `[95_000, 105_000]` ticks) trade against a seeded ZI-C crowd. Each
market is run twice on the same seed, once with the four agents satisficing and
once with them optimizing. Valuations, the ZI crowd and the candidate price
streams are the same in both runs, so the only thing that changes is how long
the agents search. Each deliberation step costs `cost_per_step` ticks and is
charged through the ledger's decision-cost sink.

The accounting is done by `frictionlab.analysis.pnl`:

    trading  = sum side * (valuation - price) * qty     (gross surplus)
    net      = trading - fees - decision_costs

`pnl_breakdown(result, mark)` takes a common mark (then `net` matches
`pnl_by_agent`), a per-agent mapping, or a callable such as
`lambda a: a.valuation`. `total` and `breakdown_by` sum the parts by group.

Seed 15, 12 paired markets, `max_steps = 40`, aspiration 500 ticks:

```
 cost  sat_gross  sat_dec   sat_net  opt_gross  opt_dec   opt_net      winner
    0    2730214        0   2730214    2738040        0   2738040  optimizing
    1    2730214    79003   2651211    2738040    95800   2642240 satisficing
    2    2730214   158006   2572208    2738040   191600   2546440 satisficing
    4    2730214   316012   2414202    2738040   383200   2354840 satisficing
    8    2730214   632024   2098190    2738040   766400   1971640 satisficing
   16    2730214  1264048   1466166    2738040  1532800   1205240 satisficing
```

### The threshold

Let `G_o, G_s` be gross surplus and `N_o, N_s` the step counts. Prices do not
depend on the step cost (thinking is charged after the plan is formed), so

    net_o - net_s = (G_o - G_s) - c * (N_o - N_s)

and the satisficer wins once

    c > c* = (G_o - G_s) / (N_o - N_s)

i.e. once a step costs more than the marginal surplus that extra search
produces. On seed 15 the optimizer spends 16_797 extra steps to earn 7_826
extra ticks, about 0.47 ticks per step, so a cost of 1 tick per step already
flips the ranking. Across seeds 1 to 30, the first swept cost at which the
satisficer wins ranges from 0 to 8 ticks per step. The marginal return is small because:

- the satisficer stops at a candidate that already clears the aspiration,
  and the remaining candidates are random draws from the same window, so the
  expected improvement from the maximum of a further sample is small (order
  statistics flatten quickly);
- the fill-probability model `exp(-kappa * d)` penalises the deep quotes that
  would carry the largest surplus, so the best candidate is usually only a
  little better than a good-enough one;
- the ZI-C crowd sets the prices. Searching harder changes the agent's limit
  price but not the prices that come back, so a lot of the extra effort goes
  into small price changes that do not change whether an order fills.

This is Simon's (1955) argument stated in the simulator's units. Optimizing
is only rational if the cost of searching is ignored. Once search has a
price, the right stopping rule is an aspiration level, and the right
aspiration rises as the step cost rises.

### Caveats

- Zero-cost dominance holds in expectation, not on every seed. In 4 of 30
  seeds the satisficer earns more gross surplus even at `c = 0`, because
  quoting different prices changes the order of fills against the ZI crowd.
  The per-decision guarantee in `test_bounded_rational.py` (the optimizer's
  chosen expected surplus is at least the satisficer's) still holds. The
  market outcome is a separate quantity and can differ from it.
- `net_o - net_s` is linear in `c` with slope `-(N_o - N_s) < 0`, so after the
  crossover the satisficer's lead only grows.

## 6. Learning from order flow

The same example ends with a `BayesianLearner` that watches a market it
never trades in (its friction is prohibitive). A scanning liquidity
provider quotes a 2-tick market at a random centre on `[80, 130]`, an
informed trader with share `mu = 0.4` hits whichever side is mispriced against
`V = 108`, and noise traders fill in the rest. Each print moves the posterior
by the Glosten-Milgrom likelihood. On seed 15 the posterior mean moves
98.6 -> 104.9 -> 108.4 and ends at 108.22 with sd 0.57 and MAP 108.

Convergence needs prints on both sides of V. A maker that quotes only near
its own belief generates trades that never test values far from that belief,
and the posterior then settles in the wrong place. The scanning provider is
there to supply prices across the whole grid.

## 7. Tests for the decision-cost example

`tests/test_pnl_breakdown.py` and `tests/test_example_phase2_decision_cost.py` check:

- ledger invariants hold, and decision costs equal `cost * steps`;
- the same seed reproduces identical sweeps and tapes;
- at zero step cost the optimizer's net PnL is `>=` the satisficer's
  (pinned seeds);
- every sweep contains a step cost at which the satisficer's net PnL is
  higher, and the gap grows monotonically with cost;
- the learner's posterior ends within 1.5 ticks of V with sd below 1.5.
