"""Analysis helpers that turn `MarketRunner` output into market-quality metrics.

- `spread`: time-weighted quoted spread from a sampled touch path, and
  per-agent / per-type PnL (net of fees and decision costs) plus markouts
  against a fundamental value.
- `pnl`: per-agent PnL split into trading (gross surplus at a mark or a
  private valuation), fees and decision costs, with group totals.
- `metrics`: canonical `TradeRecord` view of book, batch and AMM fills;
  effective = realized spread + price impact at horizon tau (exact integer
  identity), impact by trade size, and price efficiency (|mid - V| and the
  variance ratio).
- `welfare`: realized gains from trade vs the competitive-equilibrium
  maximum, net of friction sinks, and zero-sum PnL across every account.
- `sweep`: seeded (SeedSequence-spawned) parameter sweeps over friction
  levels, serial or process-parallel with identical rows, plus per-cell
  mean/std aggregation and a pandas-free CSV writer.
"""
from .metrics import (
    EfficiencyStats, MidPath, SpreadDecomposition, TradeMetrics, TradeRecord, decompose,
    impact_by_size, price_efficiency, spread_decomposition, trade_records, variance_ratio,
)
from .pnl import PnLBreakdown, breakdown_by, pnl_breakdown, total
from .spread import (
    QuoteRecorder, SpreadStats, markout_pnl, pnl_by_agent, pnl_by_type,
    spread_stats, volume_between,
)
from .sweep import (
    FRICTION_KEYS, SweepSpec, aggregate, expand_grid, friction_stack, run_sweep, write_csv,
)
from .welfare import (
    COST_SINK, POOL, VENUE_SINK, Equilibrium, WelfareReport, account_breakdown,
    competitive_equilibrium, is_valuation_agent, pnl_by_account, pnl_by_agent_type, welfare,
)

__all__ = [
    "PnLBreakdown", "breakdown_by", "pnl_breakdown", "total",
    "QuoteRecorder", "SpreadStats", "markout_pnl", "pnl_by_agent", "pnl_by_type",
    "spread_stats", "volume_between",
    "EfficiencyStats", "MidPath", "SpreadDecomposition", "TradeMetrics", "TradeRecord",
    "decompose", "impact_by_size", "price_efficiency", "spread_decomposition",
    "trade_records", "variance_ratio",
    "COST_SINK", "POOL", "VENUE_SINK", "Equilibrium", "WelfareReport", "account_breakdown",
    "competitive_equilibrium", "is_valuation_agent", "pnl_by_account", "pnl_by_agent_type",
    "welfare",
    "FRICTION_KEYS", "SweepSpec", "aggregate", "expand_grid", "friction_stack",
    "run_sweep", "write_csv",
]
