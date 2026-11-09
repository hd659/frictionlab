# Phase 12: network effects, critical mass and cold start

Every venue before Phase 12 was worth the same to a trader whether one other trader or a
hundred had shown up. A platform is different: what a member gets depends on how many
members the other side has. The fl-064 `PlatformMarket` runs two sides, A and B, for `T`
rounds of myopic best responses. Each member has an integer-tick standalone value `u`,
gets a cross-side benefit `b` for each member on the other side, pays a fee `f`, and pays
a one-time `AdoptionCost` `c` when it joins. The fl-065 `critical_mass` experiment asks
how many members the platform has to seed before membership tips, and what an adoption
friction does to that number.

```
python examples/phase12_critical_mass.py --small [OUT]        # seed 65, small preset
frictionlab run examples/configs/critical_mass_small.json --small --out OUT/run
python examples/phase5_summary_report.py --include-phase12    # adds one section
```

## Setup

| piece | where | what |
|-------|-------|------|
| `PlatformMarket` | `markets/platform.py` | two sides, cross-side benefits, fees, seeded members, T rounds |
| `AdoptionCost` | `costs/adoption.py` | one-time onboarding friction routed to the adoption sink |
| `platform_pnl`, `member_surplus` | `analysis/pnl.py` | fee revenue vs subsidy outlay; standalone, cross-side, fees, adoption |
| `critical_mass`, `tipping_fixed_points` | `experiments/critical_mass.py` | best-response map on integer counts |
| `run_platform_grid`, `find_tipping_seed` | `experiments/critical_mass.py` | adoption cost x seed members sweep, bisection |
| `critical_mass_small.json` | `examples/configs/` | adoption cost 0, 2, 8, 40; seed members 0, 4, 6, 8; 8 + 8 members |

The small preset is homogeneous on purpose. Every member has `u = -6`, `b = 1` on both
sides and both fees are 0. Nobody would join an empty platform, and every member values
a full one at `-6 + 8 = 2` ticks per round. Seeded members are pre-enrolled from the
lowest ids, and the platform pays their adoption cost.

## Why cross-side benefits create multiple equilibria

The update rule acts only on the counts of the previous round:

    stay iff u + b * n_other - f >= 0,      join iff u + b * n_other - f >= c

With `b >= 0` the map is monotone: more members on one side never make anyone on the
other side want to leave. A monotone map can have several fixed points, and here it does.
The empty platform is one, because `-6 < 0`. The full platform `(8, 8)` is another,
because `2 >= 0`. For `c = 0`, `tipping_fixed_points` returns exactly `(0, 0)` and
`(8, 8)`. Which one the market reaches depends only on where it starts.

That is the cold-start trap. In all four seed-0 cells on seed 65 membership stays at 0,
even though full membership is better for every member. No single member can move the
market alone, so the good outcome needs coordination, and the platform is the party that
can pay for it.

## Why adoption friction raises the critical mass

The critical mass is the smallest symmetric seed whose path ends at the high fixed point.
At `c = 0` a seed of 6 per side is enough. Each unseeded member sees 6 members on the
other side, gets `-6 + 6 = 0 >= 0` and joins. The path is `6/6 -> 8/8 -> 8/8`. One member
fewer and the seeded members leave in round 1: `5/5 -> 0/0 -> 0/0`.

An adoption cost only tightens the join condition, because a member already on the
platform does not pay `c` again to stay. At `c = 2` an outsider needs
`-6 + n_other >= 2`, so `n_other >= 8`, and a seed of 6 now freezes at `6/6`. That
partial fixed point is stable, since insiders still get `0 >= 0`, but no outsider
crosses. `critical_mass` returns 6, 8, 8 and 8 at adoption costs 0, 2, 8 and 40, which
is non-decreasing in `c` as the monotone map requires. It saturates at the side size,
where the only way to tip is to seed everyone.

The seeding subsidy is what the platform pays to get past the threshold. It covers the
adoption cost of every seeded member: 16, 24 and 32 ticks at `c = 2` for seeds 4, 6 and 8,
and 640 ticks at `c = 40` with seed 8. Bisection agrees with the analytic count:
`find_tipping_seed` at `c = 2` with fraction 0.9 bisects to seed 8, against an analytic
critical mass of 8. Seeding below the critical mass is the worst case. The subsidy is
spent and the platform still ends empty, which costs 320 ticks at `c = 40` with seed 4.

Against the zero-adoption-cost twin on the same draws, welfare falls by the adoption sink
plus any membership lost to the higher threshold. At `c = 2` and seed 6 the twin tips to
`8/8`, while the costly market stops at `6/6`, so the loss is 56 ticks, more than the
24 ticks of subsidy. Over the whole sweep, members collect standalone value -792 and
cross-side gain 952, platform fee revenue is 0, subsidy outlay is -1800 and the adoption
sink takes -1800.

## Frictions that make arrival order decide the outcome

The tipping seed belongs with two earlier thresholds:

- the [Coase threshold](phase5-coase-threshold.md) (fl-028) is the per-trade cost above
  which bilateral volume collapses. An adoption cost is a transaction cost paid once, on
  the first trade with a platform, and it raises the number of other traders a newcomer
  must already see before joining is worth it;
- the [lock-in threshold](phase11-switching-cost.md) (fl-062) is the switching cost at
  which an incumbent can harvest. Switching costs make a consumer's first purchase decide
  later prices. Adoption costs make the platform's first members decide whether anyone
  else ever joins.

In all three cases the friction separates outcomes that would coincide without it. With
`c = 0` and no cross-side benefit, any arrival order ends in the same place. With a
friction, history matters: who arrived first, who is already locked in, who the platform
has already paid to seed. The threshold is the smallest initial condition that changes
the outcome.

## Reproducing the numbers

All numbers above come from seed 65 with the small preset and are pinned by
`tests/test_phase12_example.py`. Two runs of the example give byte-identical `rows.csv`
and `report.html`.
