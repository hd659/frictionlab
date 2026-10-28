"""Analysis helpers that turn `MarketRunner` output into market-quality metrics.

- `spread`: time-weighted quoted spread from a sampled touch path, and
  per-agent / per-type PnL (net of fees and decision costs) plus markouts
  against a fundamental value.
"""
from .spread import (
    QuoteRecorder, SpreadStats, markout_pnl, pnl_by_agent, pnl_by_type,
    spread_stats, volume_between,
)

__all__ = [
    "QuoteRecorder", "SpreadStats", "markout_pnl", "pnl_by_agent", "pnl_by_type",
    "spread_stats", "volume_between",
]
