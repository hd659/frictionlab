"""Analysis helpers that turn `MarketRunner` output into market-quality metrics.

- `spread`: time-weighted quoted spread from a sampled touch path, and
  per-agent / per-type PnL (net of fees and decision costs) plus markouts
  against a fundamental value.
- `pnl`: per-agent PnL split into trading (gross surplus at a mark or a
  private valuation), fees and decision costs, with group totals.
"""
from .pnl import PnLBreakdown, breakdown_by, pnl_breakdown, total
from .spread import (
    QuoteRecorder, SpreadStats, markout_pnl, pnl_by_agent, pnl_by_type,
    spread_stats, volume_between,
)

__all__ = [
    "PnLBreakdown", "breakdown_by", "pnl_breakdown", "total",
    "QuoteRecorder", "SpreadStats", "markout_pnl", "pnl_by_agent", "pnl_by_type",
    "spread_stats", "volume_between",
]
