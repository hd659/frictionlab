"""Market structures beyond the continuous limit order book.

- `ConstantProductAMM`: x * y = k pool with an integer-bps LP fee, settled
  through the ledger (the pool is an explicit `Account`).
- `amm_v3_legacy`: float-based concentrated-liquidity arbitrage model, kept
  for reference until it is ported onto the ledger (Phase 3).
"""
from .amm_cp import BPS_DENOM, ConstantProductAMM, SwapRejected, SwapResult

__all__ = ["BPS_DENOM", "ConstantProductAMM", "SwapRejected", "SwapResult"]
