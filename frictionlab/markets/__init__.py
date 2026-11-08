"""Market structures beyond the continuous limit order book.

- `ConstantProductAMM`: x * y = k pool with an integer-bps LP fee, settled
  through the ledger (the pool is an explicit `Account`).
- `ConcentratedLiquidityAMM`: v3-style pool with liquidity on integer-tick
  ranges `[tick_lower, tick_upper)`; swaps cross initialized ticks and share the
  quote/swap interface and ledger settlement of the constant-product pool.
  `sqrt_at`, `base_delta` and `quote_delta` are its Q96 fixed-point helpers.
- `PerpMarket`: perpetual futures venue; marks off its own book mid, reads an
  index (e.g. `FundamentalProcess`) and settles a clamped periodic funding
  rate in integer ticks through the ledger, remainders to a funding sink.
- `BatchAuction`: frequent batch auction with the order-book intent interface;
  each batch clears at one volume-maximizing integer-tick price with pro-rata
  allocation on the marginal level. `BatchVenue` / `batch_runner` settle it
  through the friction stack and ledger on a fixed batch grid.
- `FeeTierSchedule` / `FeeComponent`: trailing-volume maker/taker fee tiers
  as a drop-in "maker_taker" friction component; rebates are paid from the
  venue sink (`settle_tiered` can reject rebates that overdraw it).
- `VenueRouter`: routes a marketable intent across named books by integer
  all-in cost (tiered taker fee + walked slippage + fixed cost), ties by id,
  splitting only when no single venue has the depth.
- `SearchMarket` / `ReservationPolicy`: decentralized bilateral search venue;
  seeded Poisson meetings charge a per-search cost to a search-cost sink, pairs
  trade at an integer Nash split when reservations cross, unsold seller value
  may perish by a fixed tick decay down to a floor.
- `LemonsMarket` (fl-055): Akerlof hidden-quality market; a uniform pooled price
  is the buyer value of the mean offered quality, iterated to a fixed point, with
  an optional `CertificationPolicy` segment clearing certified units at q + premium.
- `ReputationMarket` (fl-058): repeated hidden-quality market; buyers price a
  seller's integer (good, bad) rating score at the buyer value of its integer
  posterior expected quality; opportunists milk, whitewashing pays an identity sink.
- `SwitchingMarket` (fl-061): repeated two-seller duopoly with seeded per-consumer
  switching costs (Klemperer); a consumer leaves its supplier only if rival price +
  its cost is strictly lower, the cost paid through a friction stack to a switch sink.
- `amm_v3_legacy`: deprecated shim re-exporting `amm_cl` (warns on import).
"""
from .amm_cl import (
    MAX_TICK, MIN_TICK, Q96, ConcentratedLiquidityAMM, Position, SwapPlan, base_delta,
    quote_delta, sqrt_at,
)
from .amm_cp import BPS_DENOM, ConstantProductAMM, SwapRejected, SwapResult, settle_swap
from .perp import PERP_SINK_ID, FundingEvent, PerpMarket
from .batch_auction import (
    BatchAuction, BatchVenue, Clearing, batch_runner, clearing_price, pro_rata,
)
from .fee_tiers import SINK_POLICIES, FeeComponent, FeeTier, FeeTierSchedule, settle_tiered
from .router import RouteLeg, RoutePlan, RouteVenue, VenueRouter, quote_leg
from .search_market import SEARCH_SINK_ID, Meeting, ReservationPolicy, SearchAgent, SearchMarket
from .lemons import CERTIFIED, POOLED, LemonsMarket, LemonsRound, pooled_price
from .reputation import (
    IDENTITY_SINK_ID, REPUTATION, ReputationMarket, ReputationRound, score_quality,
)
from .switching import SWITCH_SINK_ID, SWITCHING, SwitchingMarket, SwitchingRound

__all__ = [
    # constants
    "BPS_DENOM", "MAX_TICK", "MIN_TICK", "PERP_SINK_ID", "Q96", "SEARCH_SINK_ID",
    "SINK_POLICIES",
    # AMMs
    "ConcentratedLiquidityAMM", "ConstantProductAMM", "Position", "SwapPlan",
    "SwapRejected", "SwapResult", "base_delta", "quote_delta", "settle_swap", "sqrt_at",
    # perpetuals
    "FundingEvent", "PerpMarket",
    # batch auction
    "BatchAuction", "BatchVenue", "Clearing", "batch_runner", "clearing_price", "pro_rata",
    # fee tiers
    "FeeComponent", "FeeTier", "FeeTierSchedule", "settle_tiered",
    # routing
    "RouteLeg", "RoutePlan", "RouteVenue", "VenueRouter", "quote_leg",
    # search
    "Meeting", "ReservationPolicy", "SearchAgent", "SearchMarket",
    # quality uncertainty
    "CERTIFIED", "POOLED", "LemonsMarket", "LemonsRound", "pooled_price",
    # reputation
    "IDENTITY_SINK_ID", "REPUTATION", "ReputationMarket", "ReputationRound", "score_quality",
    # switching costs and lock-in
    "SWITCH_SINK_ID", "SWITCHING", "SwitchingMarket", "SwitchingRound",
]
