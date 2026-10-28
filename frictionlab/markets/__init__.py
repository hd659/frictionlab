"""Market structures beyond the continuous limit order book.

- `ConstantProductAMM`: x * y = k pool with an integer-bps LP fee, settled
  through the ledger (the pool is an explicit `Account`).
- `ConcentratedLiquidityAMM`: v3-style pool with liquidity on integer-tick
  ranges `[tick_lower, tick_upper)`; swaps cross initialized ticks and share the
  quote/swap interface and ledger settlement of the constant-product pool.
- `amm_v3_legacy`: deprecated shim re-exporting `amm_cl` (warns on import).
"""
from .amm_cl import MAX_TICK, MIN_TICK, ConcentratedLiquidityAMM, Position, SwapPlan
from .amm_cp import BPS_DENOM, ConstantProductAMM, SwapRejected, SwapResult, settle_swap

__all__ = ["BPS_DENOM", "MAX_TICK", "MIN_TICK", "ConcentratedLiquidityAMM", "ConstantProductAMM",
           "Position", "SwapPlan", "SwapRejected", "SwapResult", "settle_swap"]
