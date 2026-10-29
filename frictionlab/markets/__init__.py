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
- `amm_v3_legacy`: deprecated shim re-exporting `amm_cl` (warns on import).
"""
from .amm_cl import MAX_TICK, MIN_TICK, ConcentratedLiquidityAMM, Position, SwapPlan
from .amm_cp import BPS_DENOM, ConstantProductAMM, SwapRejected, SwapResult, settle_swap
from .perp import PERP_SINK_ID, FundingEvent, PerpMarket
from .batch_auction import (
    BatchAuction, BatchVenue, Clearing, batch_runner, clearing_price, pro_rata,
)

__all__ = ["BPS_DENOM", "MAX_TICK", "MIN_TICK", "PERP_SINK_ID", "BatchAuction", "BatchVenue",
           "Clearing", "ConcentratedLiquidityAMM", "ConstantProductAMM", "FundingEvent",
           "PerpMarket", "Position", "SwapPlan", "SwapRejected", "SwapResult",
           "batch_runner", "clearing_price", "pro_rata", "settle_swap"]
