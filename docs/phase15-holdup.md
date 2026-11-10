# Phase 15: hold-up, underinvestment and costly contracting

The Coase threshold of Phase 5 treats a transaction cost as a per-trade tax. Coase,
Williamson and Grossman-Hart had a narrower cost in mind: the cost of writing down, before
anyone invests, what a relationship will pay. Phase 15 models that cost. In the fl-073
`HoldupMarket` each `SpecificInvestor` seller picks an integer investment level `i` from a
schedule of `(investment_cost, buyer_value)` pairs. The investment is specific: outside
the match the good is worth `outside_value` whatever `i` is. Without a contract the price
is bargained after the investment is sunk. With a contract, paid for through a
`ContractingCost` to the contract sink, both sides commit up front to the first-best level.
The fl-074 `holdup` experiment sweeps contracting cost x seller weight on common random
numbers.

```
python examples/phase15_holdup.py --small [OUT]               # seed 74, small preset
frictionlab run examples/configs/holdup_small.json --small --out OUT/run
python examples/phase5_summary_report.py --include-phase15    # adds one section
```

## Setup

| piece | where | what |
|-------|-------|------|
| `HoldupMarket` | `markets/holdup.py` | seeded buyer-seller pairs, ex-post Nash bargaining over the quasi-rent |
| `ContractingCost` | `costs/contracting.py` | integer-tick ex-ante contract-writing cost, routed to the contract sink |
| `SpecificInvestor` | `agents/specific_investor.py` | hold-up level, first-best level, contract iff strictly better |
| `holdup_pnl`, `holdup_buyer_surplus` | `analysis/pnl.py` | revenue, sunk investment, contracting cost; value and price |
| `holdup_investment`, `first_best_investment`, `max_contracting_cost` | `experiments/holdup.py` | exact integer analytics |
| `run_holdup_grid`, `find_contracting_threshold` | `experiments/holdup.py` | cost x weight sweep, contracting-cost bisection |
| `holdup_small.json` | `examples/configs/` | costs 0, 2, 6, 10; weights 5000, 7500, 10000 bps; 2 pairs, 12 rounds, outside value 30, no noise |

The small schedule has investment costs 0, 6, 14, 24, 36, 50 and buyer values 40, 58,
74, 88, 98, 104. The joint surplus over the outside option, `S(i) = v(i) - 30 - c(i)`, is
10, 22, 30, 34, 32, 24, so the first-best level is 3 with `S = 34`.

## Why ex-post bargaining makes the investor underinvest

Once the seller has sunk `c(i)`, the cost is gone and plays no part in the bargain. What
is left to split is the quasi-rent `Q(i) = v(i) - outside_value`. With seller weight `w`
the seller gets `w / 10000` of it (fl-040 floor convention, remainder to the seller). The
seller still pays all of the investment, so its hold-up payoff is
`seller_share(Q(i), w) - c(i)`. A planner sets the full marginal return `dQ` against `dc`.
The seller sets only its share `w * dQ / 10000` against the same `dc`. Each step of
investment is cheaper to the relationship than the seller's private return suggests, so the
seller stops early.

On seed 74 at `w = 5000` the marginal returns in `Q` are 18, 16, 14, 10, 6 and the
marginal costs are 6, 8, 10, 12, 14. The planner climbs while `dQ > dc`, to level 3. The
seller sees halves, 9, 8, 7, 5, 3. Step one pays (9 > 6), step two ties (8 = 8) and
step three loses (7 < 10). The smallest maximizer is level 1, where the hold-up payoff
curve printed by the example reads 5, 8, 8, 5, -2, -13. Investing at 1 instead of 3 costs
`S(3) - S(1) = 12` ticks of surplus per pair-round. The experiment reports this as an
underinvestment gap of 12.0 in the cell where no contract is written. At `w = 10000` the
seller owns the whole return and invests at the first best, so the gap and the loss are 0.

## Why a contract is worth exactly the surplus it recovers

A contract fixes level `i*` and a price that gives the seller `seller_share(S*, w)` of the
net first-best surplus. The seller writes one iff that, minus the contracting cost `k`, is
strictly greater than its hold-up payoff. The largest `k` it will pay is the gain minus
one tick, because ties go to no contract:

```
G = seller_share(S*, w) - (seller_share(Q(h), w) - c(h))
max_contracting_cost = max(G - 1, 0)
```

At `w = 5000`, `G = 17 - 8 = 9`, so the analytic max contracting cost is 8. The
contracting rate in that column is 1, 1, 1, 0 at costs 0, 2, 6, 10. The bisected threshold
(the smallest cost at which fewer than half the pairs contract) is 9, one tick above the
analytic maximum, as it must be. `G` is the 12-tick hold-up loss the contract recovers,
plus the buyer's hold-up share of the quasi-rent (14), minus the buyer's contract share of
the net surplus (17): `12 + 14 - 17 = 9`. The contract is a
technology that buys back surplus. It is worth writing up to the value it buys back and
no further. That is the Coasean transaction cost in its original form: what decides
whether to govern a relationship by a contract is how much surplus the contract restores
against what it costs to write.

The sweep shows both sides of the trade-off. While contracts are written, every tick of
contracting cost is a deadweight loss: at `w = 5000` the surplus share falls 1, 0.941,
0.824 at costs 0, 2, 6. Once the cost passes the maximum, the pair falls back to hold-up
and the share drops to 0.647, the 12-tick underinvestment loss. At `w = 7500` the gain is
6 and the analytic maximum 5, so contracting stops by cost 6. At `w = 10000` nobody ever
contracts and the share stays 1. Over the sweep the 9792 ticks of first-best surplus split
into buyer gain 2616, seller net gain 6648, contract sink 240 and underinvestment
deadweight loss 288, exact in ticks; sellers sink 6480 ticks of investment on top of
their net gain.

## Markets, contracts or nothing

The contracting threshold is the fourth friction threshold in frictionlab that decides how
a relationship is governed:

- the [Coase threshold](phase5-coase-threshold.md) (fl-028) is the per-trade cost above
  which spot-market volume collapses: above it, the market is replaced by nothing;
- the [lock-in threshold](phase11-switching-cost.md) (fl-062) is the switching cost above
  which an incumbent can price above marginal cost: the market still clears, but ex-post
  rents go to whoever holds the customer, the same ex-post appropriation that drives hold-up;
- the [rigidity threshold](phase14-menu-cost.md) (fl-071) is the menu cost above which
  prices stop tracking their target: a fixed cost of acting, not of trading;
- the contracting threshold is the cost above which the parties stop writing contracts and
  fall back on ex-post bargaining.

All four are fixed costs that produce a switch, not a smooth response. Below the Coase
threshold a spot market works; below the contracting threshold, a contract governs an
investment that a spot market would hold up; above both, specific investment shrinks to
what a share of the return can pay for, or trade stops. This is Williamson's discriminating
alignment: the cheapest available governance mode wins, and the friction that prices each
mode decides which one.

## Reproduce and check

`tests/test_phase15_example.py` pins every number above to seed 74, checks that two runs
give byte-identical `rows.csv` and `report.html`, and checks that the report is valid XML
with no network references. `tests/test_experiment_holdup.py` and
`tests/test_holdup_analytic.py` cover the analytic levels, `max_contracting_cost` and the
bisection on their own.
