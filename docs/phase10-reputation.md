# Phase 10: reputation, end-game milking and cheap identities

[Phase 9](phase9-lemons.md) fixed hidden quality with a one-shot, priced reveal. The
fl-058 `ReputationMarket` tries the dynamic alternative: sellers trade for `T` rounds
under a public id, buyers rate each delivery, and the price follows the rating history.
The fl-059 `reputation` experiment asks how much cheating survives, when sellers shed a
bad score by re-entering under a new id, and what that escape route costs.

```
python examples/phase10_reputation.py --small [OUT]           # seed 59, small preset
frictionlab run examples/configs/reputation_small.json --small --out OUT/run
python examples/phase5_summary_report.py --include-phase10    # adds one section
```

## Setup

| piece | where | what |
|-------|-------|------|
| `ReputationMarket` | `markets/reputation.py` | T rounds, integer (good, bad) scores, integer posterior price |
| `ReputationSeller`, `reputation_pnl` | `agents/reputation_seller.py`, `analysis/pnl.py` | cheat / whitewash policy, seller PnL split |
| `milking_switch_round`, `min_deterrent_whitewash_cost` | `experiments/reputation.py` | analytic milking round and deterrent bound |
| `run_reputation_grid`, `find_reputation_threshold` | `experiments/reputation.py` | rating accuracy x whitewash cost sweep, bisection |
| `reputation_small.json` | `examples/configs/` | accuracy 6000, 8500, 10000 bps; whitewash cost 0, 4, 16, 64; 20 sellers, T = 6 |

Qualities come from the fl-055 schedule 10, 20, ..., 100, with buyer premium 45. One
seller in four (ids 4, 8, 12, 16, 20) is an opportunist that may deliver the floor
quality 10 instead of its own `q`, saving `q - 10`. Buyers start from a prior of 7500
bps that a fresh id is honest. On seed 59 a fresh id is priced at 93 ticks and a seller
known to deliver the floor at 55. At 8500 bps one good rating lifts the price to 103,
one bad rating drops it to 72, and a long clean record tops out at 105.

## Why a finite horizon produces end-game milking

An opportunist compares two numbers each round. Cheating saves `q - f` once. Staying
honest keeps the score premium `p_now - p_floor` for each of the `R` rounds left. It
cheats once `R * (p_now - p_floor) < q - f`. `R` shrinks to zero in the last round, so
every opportunist with a positive saving cheats there at the latest: reputation cannot
discipline the last trade, because there is no future price left to lose. With a
constant gap `g` the first cheating round is

```
t* = max(0, T - 1 - (saving - 1) // g)        (milking_switch_round)
```

The example traces seller 8 (`q = 30`, saving 20) at 8500 bps. Its score runs
0/0 -> 1/0 -> 1/1 -> 2/1 -> 3/1 -> 4/1 and its price 93 -> 103 -> 93 -> 103 -> 105 -> 105,
honest for five rounds, including one unlucky bad rating at round 1. It milks in
round 5, the last one, which matches the analytic `t*` of 5 at perfect accuracy (gap
106 - 55 = 51). Better ratings cut cheating, but only so far. The cheating rate is 0.20
to 0.23 at 6000 bps and 0.13 at 8500 and 10000 bps. The remaining 0.13 is pure
end-game milking: four opportunists cheat once each in the final round, and the
fifth (`q = 10`) has nothing to save.

## Why cheap identities let opportunists escape their score

A score is attached to an id, not to the seller. Any seller re-enters under a fresh id
iff `p_fresh - c > p_now`, so a bad record is worth only as much as it costs to drop.
At 6000 bps and zero cost there are 15 whitewashes in six rounds. Opportunists use them
to wipe out a detected cheat and cheat again (seller 12 milks twice, from round 4),
but honest sellers also use them to dodge noise: an honest seller with a wrong bad
rating has every reason to start over. Noisy ratings and free identities therefore
feed each other. With perfect ratings no honest seller is ever rated bad, and nobody
whitewashes at any cost.

## The deterrent whitewash cost as the certification bound

Since any score is priced at least `p_floor`, re-entry never pays once

```
c >= min_deterrent_whitewash_cost = p_fresh - p_floor = 93 - 55 = 38 ticks
```

This plays the role of the fl-056 certification bound: it is the break-even cost of
the seller with the most to gain from the friction. `find_reputation_threshold` bisects
for the smallest cost with zero whitewashes at 8500 bps, reusing seed 59 at every
probe. It lands on 21 ticks against the analytic deterrent bound of 38, because the
worst score anyone reaches at 8500 bps on this seed is one bad rating (price 72), and
93 - 72 = 21. The bound is the worst case and the bisection measures the case that
actually occurs.

The trade-off is the reverse of certification's. Certification charges a deadweight
cost up front and reveals quality exactly. Reputation is free per trade but runs on a
stream of rating noise, and a whitewash cost above the threshold buys deterrence with
an up-front cost of its own. On this seed the deterrence is not free: at 6000 bps the
surplus share drops from 0.68 at cost 0 to 0.64 at every positive cost. An honest
`q = 90` seller with one wrong bad rating is priced at 89, below its cost, and once
re-entry costs anything it no longer pays to start over, so it stops trading.
Over the sweep 80 ticks go to the identity sink.

## Welfare, and what ratings cannot fix

Every trade creates exactly 45 ticks of surplus (buyer value of the delivered quality
minus its cost), so surplus share tracks the trade count. It is non-decreasing in
rating accuracy at every whitewash cost (0.64, 0.67, 0.68 at cost 16). It never
exceeds 0.68, below the 0.85 of the matching fl-056 pooled lemons cell, which comes
from a cold start. A fresh id is priced at 93, below the cost of the four `q = 100`
sellers, so they never make a first sale, never earn a rating, and sit out all 6
rounds: 24 of 120 seller-rounds. A score tells buyers about type, not quality, so
the same price also overpays low-quality honest sellers. Over the sweep the surplus
splits into buyer gain -6850, honest-seller gain 34546, opportunist cheating gain 23244
and identity sink -80 ticks. Buyers lose on average because they pay a type premium
for units worth less than the price.

Next to the earlier collapse bounds ([the Coase threshold](phase5-coase-threshold.md),
the [Grossman-Stiglitz information collapse](phase8-grossman-stiglitz.md) and the
Phase 9 certification bound), the deterrent cost bounds an exit friction, the cost of
discarding the past. It differs from the others in that friction below the bound adds
noise-driven exits instead of removing trades.

## Reproducibility

`tests/test_phase10_example.py` runs the example with `--small` twice and requires
byte-identical `rows.csv` and `report.html`. It also checks that the report parses as
XML with no external references, that the surplus share is non-decreasing in rating
accuracy, and that the numbers on this page match the run. Every number here comes
from master seed 59 and the small preset.
