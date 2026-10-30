"""Market structures beyond the continuous limit order book.

- `ConstantProductAMM`: x * y = k pool with an integer-bps LP fee, settled
  through the ledger (the pool is an explicit `Account`).
- `ConcentratedLiquidityAMM`: v3-style pool with liquidity on integer-tick
  ranges `[tick_lower, tick_upper)`; swaps cross initialized ticks and share the
  quote/swap interface and ledger settlement of the constant-product pool.
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
- `amm_v3_legacy`: deprecated shim re-exporting `amm_cl` (warns on import).
"""
from .amm_cl import MAX_TICK, MIN_TICK, ConcentratedLiquidityAMM, Position, SwapPlan
from .amm_cp import BPS_DENOM, ConstantProductAMM, SwapRejected, SwapResult, settle_swap
from .perp import PERP_SINK_ID, FundingEvent, PerpMarket
from .batch_auction import (
    BatchAuction, BatchVenue, Clearing, batch_runner, clearing_price, pro_rata,
)
from .fee_tiers import SINK_POLICIES, FeeComponent, FeeTier, FeeTierSchedule, settle_tiered
from .router import RouteLeg, RoutePlan, RouteVenue, VenueRouter, quote_leg

__all__ = ["BPS_DENOM", "MAX_TICK", "MIN_TICK", "PERP_SINK_ID", "BatchAuction", "BatchVenue",
           "Clearing", "ConcentratedLiquidityAMM", "ConstantProductAMM", "FeeComponent",
           "FeeTier", "FeeTierSchedule", "FundingEvent", "PerpMarket", "Position",
           "RouteLeg", "RoutePlan", "RouteVenue", "SINK_POLICIES", "SwapPlan",
           "SwapRejected", "SwapResult", "VenueRouter", "batch_runner", "clearing_price",
           "pro_rata", "quote_leg", "settle_swap", "settle_tiered"]
