"""Deprecated: use `frictionlab.markets.amm_cl` instead.

The original float-based single-range v3 model has been ported to the exact
integer, ledger-settled `ConcentratedLiquidityAMM` (multi-range, tick
crossing, per-step LP fee). This module only re-exports the new API and emits a
`DeprecationWarning` on import so existing `amm_v3_legacy` imports keep working.

The float CEX/DEX arbitrage helpers (`V3Pool`, `PerpVenue`, `GasModel`,
`evaluate`, `optimal_size`, ...) were removed; perpetuals and the cross-venue
arbitrageur are rebuilt on the ledger in later Phase 3 milestones.
"""
from __future__ import annotations

import warnings

from .amm_cl import (MAX_TICK, MIN_TICK, Q96, ConcentratedLiquidityAMM, Position,
                     SwapPlan, sqrt_at)
from .amm_cp import BPS_DENOM, SwapRejected, SwapResult

warnings.warn(
    "frictionlab.markets.amm_v3_legacy is deprecated; import ConcentratedLiquidityAMM "
    "from frictionlab.markets (or frictionlab.markets.amm_cl) instead",
    DeprecationWarning,
    stacklevel=2,
)

_REMOVED = frozenset({"V3Pool", "PerpVenue", "GasModel", "SpreadQuote", "Direction",
                      "evaluate", "best_direction", "optimal_size", "breakeven_gross_bps"})


def __getattr__(name: str):
    if name in _REMOVED:
        raise AttributeError(
            f"amm_v3_legacy.{name} was removed in the ledger port; use "
            "ConcentratedLiquidityAMM (perps/arbitrage return in later Phase 3 milestones)")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BPS_DENOM", "MAX_TICK", "MIN_TICK", "Q96", "ConcentratedLiquidityAMM", "Position",
           "SwapPlan", "SwapRejected", "SwapResult", "sqrt_at"]
