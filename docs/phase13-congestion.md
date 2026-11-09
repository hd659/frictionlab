# Phase 13: congestion and queueing costs

Before Phase 13 a trader never paid for the time spent behind other traders. The fl-067
`QueueMarket` adds that cost. One server works through `T` seeded steps. Each step an
arrival shows up with probability `p` and the head customer is served with probability
`q`. An arriving customer with reward `R` and waiting cost `c` per step sees the `n`
customers already in the system and joins iff

    R - toll - c * ceil((n + 1) * 10000 / service_prob_bps) >= 0      (ties join)

The fl-068 `congestion` experiment sweeps toll x waiting cost on common random numbers and
compares three thresholds: the individual (Naor) one, the social one and the one a
revenue-maximizing operator would pick. Queue length below means the customers waiting
behind the one in service, so an arrival that sees `n` in the system leaves a queue of
`n`. A threshold is the largest such `n` at which an arrival still joins.

```
python examples/phase13_congestion.py --small [OUT]           # seed 68, small preset
frictionlab run examples/configs/congestion_small.json --small --out OUT/run
python examples/phase5_summary_report.py --include-phase13    # adds one section
```

## Setup

| piece | where | what |
|-------|-------|------|
| `QueueMarket` | `markets/queue_market.py` | one server, Bernoulli arrivals and service on separate streams |
| `CongestionCost` | `costs/congestion.py` | admission toll to the toll sink, per-step waiting cost to the waiting sink |
| `queue_surplus`, `operator_pnl` | `analysis/pnl.py` | reward, toll and waiting per customer; toll revenue |
| `naor_individual_threshold`, `naor_social_threshold`, `naor_optimal_toll` | `experiments/congestion.py` | exact birth-death steady state in `Fraction`s |
| `run_queue_grid`, `find_congestion_toll` | `experiments/congestion.py` | toll x waiting cost sweep, toll bisection |
| `congestion_small.json` | `examples/configs/` | tolls 0, 4, 8, 16; waiting costs 1, 2, 4; `R = 24`, `p = 0.7`, `q = 0.5`, 200 steps |

## Why self-interested arrivals over-join

An arrival that sees `n` ahead expects to spend `(n + 1) / q` steps in the system and
compares that delay with its own reward. What it leaves out is the cost it imposes on
everyone who arrives while it is still in the system. Each of them waits one service time
longer, roughly `c / q` ticks each. The private join condition is therefore too loose
whenever later arrivals are likely, so the individual threshold is at least the social
one. That is Naor's (1969) result, and `tests/test_congestion_analytic.py` checks it on a
grid of presets.

On seed 68 at waiting cost 2 the gap is large. Arrivals join up to an individual threshold
of 5, while the social optimum, which maximizes `R * joins - c * customer-steps` per step
exactly, is 0: join only when nobody is waiting. With no toll the queue fills and stays
near the cap. The zero-toll path starts `0 0 1 2 2 3 4 4 5 5 5 5 5`, the mean queue length
is 4.25 and welfare is 158 ticks over 200 steps. Most of the reward goes into waiting. Over
the sweep the waiting-cost sink takes -17123 ticks of the 26232 ticks of customer reward.

## Why a toll equal to the externality restores the social threshold

A toll changes the join condition without consuming anything. It moves ticks from the
customer to the operator, so it drops out of welfare, defined as customer surplus plus
toll revenue (fl-022). The waiting cost is different: it goes to a sink and is lost. A
Pigouvian toll charges each arrival roughly the delay cost it imposes on later arrivals.
The private calculation then matches the social one, and arrivals stop joining a queue
the social optimum would not let them join.

`naor_optimal_toll` returns the smallest integer toll whose individual threshold equals
the social one. At `R = 24`, `c = 2`, `q = 0.5` that toll is 17, and `find_congestion_toll`
bisects to 17, the first toll at which the simulated max queue length is at most the
social threshold. At toll 17 the queue stays at 0 throughout and welfare rises from 158 to
1590 ticks on the same draws. Join rate is non-increasing in toll in every column of the
sweep (0.671, 0.664, 0.658, 0.604 at `c = 2`). Over the sweep customers pay
-7796 ticks of toll and the operator receives 7796. The toll is a transfer and nets to zero
in welfare.

A revenue-maximizing operator sets its toll by a different rule. It maximizes
`toll * joins` per step and picks toll 20 at `c = 2`, which also induces threshold 0 on
this preset. With a single server and a small `R`, the revenue and social thresholds
coincide. On larger presets the monopolist can price above the Pigouvian level, which
leaves too few joiners, the opposite of Naor's over-joining.

## Frictions that destroy or allocate surplus

The optimal toll belongs with two earlier frictions:

- the [Coase threshold](phase5-coase-threshold.md) (fl-028) is the per-trade transaction
  cost above which bilateral volume collapses. That cost is paid to a sink, so every tick
  of it is deadweight loss, and trades that do not happen lose their surplus as well;
- the [adoption cost](phase12-critical-mass.md) (fl-064) is a one-time onboarding friction,
  also paid to a sink. It raises the critical mass, and the platform's seeding subsidy is
  the price of getting past it.

The congestion toll has the same form, an integer-tick charge in the friction stack. The
difference is who receives it. Paid to a sink, as with waiting time, Coase costs and
adoption costs, a friction destroys surplus. Paid to another account, as a toll is, it is
a transfer that still changes behaviour, and set at the externality it allocates the
scarce resource (here a place in the queue) to those who value it most. Waiting is the
congestion friction that destroys surplus, and the toll replaces it with one that
allocates surplus.

## Reproducing the numbers

All numbers above come from seed 68 with the small preset and are pinned by
`tests/test_phase13_example.py`. Two runs of the example give byte-identical `rows.csv`
and `report.html`.
