# Phase 5.2: decision cost and over-analysis

Simon's point was that thinking costs something. An agent that searches every
option finds better prices, but it pays for each step of the search. This
experiment measures the step cost at which a satisficer, which stops at the
first good-enough option, out-earns an optimizer net of what each one spent on
deliberation. It also checks that measurement against a closed-form condition.

The Phase 2 seed (`examples/phase2_decision_cost.py`) showed that such a
crossover exists. This article moves it onto the fl-028 experiment base, sweeps
it over aspiration, and adds a population view.

```
python examples/phase5_decision_cost.py                   # 3 aspirations x 6 costs, 4 replicates
python examples/phase5_decision_cost.py --small           # 2 x 3 grid, 2 replicates
python examples/phase5_decision_cost.py --max-steps 20 --report out/decision_cost.html
```

## Setup

| piece | where | what |
|-------|-------|------|
| `build`, `metrics` | `experiments/decision_cost.py` | four `BoundedRationalAgent`s plus a 6+6 ZI-C crowd, with per-cell PnL from `analysis/pnl.py` |
| `run_decision_cost` | `experiments/decision_cost.py` | aspiration x cost_per_step x mode grid through `run_experiment` |
| `find_break_even_cost` | `experiments/decision_cost.py` | doubling, then integer bisection on cost_per_step |
| `population`, `run_study` | `experiments/decision_cost.py` | fl-024 replicator at three cost levels, and the combined result |

Each cell fixes `(aspiration, cost_per_step, mode)`. Every bounded-rational
slot runs in the cell's mode. Common random numbers are on: replicate r gets the
same ZI-C crowd, the same private valuations and the same candidate stream in
every cell. So a satisficing cell and an optimizing cell differ only in when
the agent stops evaluating candidates. Per cell, summed over the
bounded-rational accounts:

- `gross`: fills valued at each agent's private valuation (Gode-Sunder surplus).
- `decision`: deliberation charges routed to the ledger's decision-cost sink.
  This always equals `cost_per_step * steps` (`decision_residual == 0`).
- `net = gross - fees - decision`. The venue charges no fees here, so the only
  friction is the cost of thinking.
- `violations` and `zero_sum_residual`: ledger health. Both must be 0.

## The break-even condition

The step cost is paid after deliberation, and it never enters the
expected-surplus calculation that picks the price. Within one aspiration,
`gross` and `steps` are therefore flat in `c = cost_per_step`, and net PnL is
linear in `c` with slope `-steps`:

```
net_mode(c) = gross_mode - c * steps_mode
```

Write `dG = gross_opt - gross_sat` for the optimizer's surplus gain and
`dS = steps_opt - steps_sat` for its extra steps. The satisficer wins at `c` iff

```
net_sat(c) > net_opt(c)   <=>   dG < dS * c
```

The optimizer evaluates a superset of the satisficer's candidates, so `dS > 0`
whenever aspiration ever stops the satisficer early. That gives three cases:

- `dG < 0`: the satisficer wins even at `c = 0`. This happens when the
  optimizer's argmax of expected surplus fills less often in realized trades,
  so `c* = 0`.
- `dG >= 0, dS > 0`: `c* = floor(dG / dS) + 1`, the smallest integer above the
  optimizer's mean surplus gain per extra step.
- `dS = 0`: the two modes behave identically, so no cost separates them.

`find_break_even_cost` does not use the formula. It simulates each probe on the
same seeds: first `c = 0`, then doubling `1, 2, 4, ...` until the satisficer
wins, then integer bisection inside the last bracket. Means are exact
`Fraction`s, so `c* == BreakEven.analytic` is an equality test and not a float
tolerance. The test suite checks that the satisficer-wins condition holds at
`c*` and fails at `c* - 1`.

## How aspiration shifts it

Aspiration controls how early the satisficer stops. On the default seed
(29, 4 replicates, max_steps 40):

| aspiration | dG | dS | dG/dS | c* |
|-----------:|---:|---:|------:|---:|
| 100 | 13325.5 | 1580.8 | 8.43 | 9 |
| 500 | 5891.8 | 1365.2 | 4.32 | 5 |
| 2000 | 3903.2 | 866.2 | 4.51 | 5 |

A low aspiration accepts the first barely profitable candidate. That saves the
most steps, but it also gives up the most surplus, and `dG` grows faster than
`dS`, so the optimizer stays ahead up to a higher step cost. Raising aspiration
shrinks both gaps. The satisficer then behaves more like the optimizer, so
there is less surplus to lose by stopping early, and `c*` falls from 9 to 5,
then stays flat because `dG` and `dS` shrink at about the same rate. In the limit
of an aspiration that is never met, the satisficer searches the whole batch,
`dS = dG = 0`, and the comparison is void.

`max_steps` works the other way at first. With 2 or 5 candidates the two modes
rarely differ (`c* = 0` on this seed). For aspiration 100 and 500, `c*` does
not decrease from 2 to 20 steps (500: 0, 0, 5, 9 at 2, 5, 10, 20 steps). At
aspiration 2000 the satisficer stops so rarely that `c*` is noise around 0-5. Past that point returns diminish: the 40th candidate seldom beats the
best of the first 20, `dS` keeps growing roughly linearly while `dG` levels
off, and `c*` falls again (9 at 20 steps, 5 at 40 for aspiration 500). Over-
analysis gets cheaper to beat as the search gets longer.

## Population view

`population` gives the replicator (fl-024) a mixed population of 8 slots with
fixed per-side valuations (`pop_values`), so differences in fitness come from
the decision rule and not from which slots a given mix fills. Fitness is mean
net PnL per agent of each type, on fixed per-generation seeds. An archetype
counts as extinct below one agent's share. `run_study` runs it at three cost
levels: zero, the break-even cost for the default aspiration, and a high cost.
At high cost the satisficer share at the fixed point is at least its share at
zero cost. At zero cost, selection runs on small-sample noise between nearly
identical rules, so that share should not be read as a prediction.

## Reading the output

The grid table prints both modes' gross, decision cost and net per
`(aspiration, cost)` cell, plus the winner. Within one aspiration the winner
switches exactly once, at the first swept cost `>= c*`. The break-even table
shows `dG`, `dS`, their ratio, the bisected `c*` and the closed form side by
side. The population table shows the detector reason, the generation where it
fired, and the settled satisficer share. `--report` writes the findings and
the sweep into an offline fl-026 HTML file.
