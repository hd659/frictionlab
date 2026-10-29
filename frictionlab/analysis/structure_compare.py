"""Compare market structures (frequent batch auction vs continuous book).

For a trade tape and a reference value V(t) (e.g. `FundamentalProcess.value`):

    effective spread_j = 2 * d_j * (p_j - V(t_j))      d_j = +1 buy-, -1 sell-aggressor
    avg effective spread = sum_j q_j * spread_j / sum_j q_j        (ticks, volume-weighted)

A fill is a latency-driven *snipe* when it prints within `window` seconds
after a jump of V (time in [jump, jump + window]) and the aggressor gains
against the new value, d_j * (V(t_j) - p_j) > 0: someone picked off a quote
that the jump made stale before its owner could update it. The snipe share
is snipe volume / total volume. In a continuous book the race is won by
the fastest taker at the stale resting price; a batch collects everyone's
response and clears at one price, so the stale-quote rent shrinks.

`compare` accepts trade lists or anything with a `.trades` attribute (e.g.
`RunResult`) keyed by structure name; `delta` reports b - a per metric.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, fields
from typing import Callable, Dict, Iterable, Mapping, Sequence

from ..core.types import Trade


@dataclass(frozen=True)
class StructureStats:
    name: str
    volume: int                 # lots executed
    n_trades: int
    avg_effective_spread: float  # ticks, volume-weighted (nan if no trades)
    snipe_volume: int
    snipe_share: float          # snipe_volume / volume (0.0 if no trades)
    avg_price: float            # VWAP (nan if no trades)

    def as_dict(self) -> Dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "name"}


def _trades_of(x) -> Sequence[Trade]:
    return list(getattr(x, "trades", x))


def effective_spread(trade: Trade, value: Callable[[float], int]) -> int:
    """2 * d * (p - V(t)) in ticks for one trade (may be negative)."""
    return 2 * trade.aggressor.value * (trade.price - value(trade.time))


def is_snipe(trade: Trade, value: Callable[[float], int],
             jump_times: Sequence[float], window: float) -> bool:
    """True if `trade` follows a jump within `window` and the aggressor gains."""
    i = bisect.bisect_right(jump_times, trade.time)
    if i == 0 or trade.time - jump_times[i - 1] > window:
        return False
    return trade.aggressor.value * (value(trade.time) - trade.price) > 0


def structure_stats(trades: Iterable[Trade], value: Callable[[float], int], *,
                    jump_times: Iterable[float] = (), window: float = 1.0,
                    name: str = "") -> StructureStats:
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    jt = sorted(float(t) for t in jump_times)
    vol = n = notional = snipe = 0
    spread_w = 0
    for t in trades:
        vol += t.qty
        n += 1
        notional += t.notional
        spread_w += t.qty * effective_spread(t, value)
        if is_snipe(t, value, jt, window):
            snipe += t.qty
    nan = float("nan")
    return StructureStats(name, vol, n, spread_w / vol if vol else nan, snipe,
                          snipe / vol if vol else 0.0, notional / vol if vol else nan)


def jump_times_of(value_process, until: float) -> list:
    """Jump times of a `FundamentalProcess` up to `until` (empty if constant)."""
    jumps = getattr(value_process, "jumps", None)
    return [t for t, _ in jumps(until)] if jumps else []


def compare(runs: Mapping[str, object], value: Callable[[float], int], *,
            jump_times: Iterable[float] = (), window: float = 1.0) -> Dict[str, StructureStats]:
    """`structure_stats` for every named run (trade list or `.trades` holder)."""
    jt = list(jump_times)
    return {k: structure_stats(_trades_of(r), value, jump_times=jt, window=window, name=k)
            for k, r in runs.items()}


def delta(a: StructureStats, b: StructureStats) -> Dict[str, float]:
    """b - a for every numeric metric (nan propagates)."""
    da, db = a.as_dict(), b.as_dict()
    return {k: db[k] - da[k] for k in da}


def summary(stats: Mapping[str, StructureStats]) -> str:
    """Fixed-width text table, one row per structure."""
    head = f"{'structure':<12}{'volume':>8}{'trades':>8}{'eff.spread':>12}{'snipe%':>8}"
    rows = [head]
    for k, s in stats.items():
        es = "nan" if math.isnan(s.avg_effective_spread) else f"{s.avg_effective_spread:.3f}"
        rows.append(f"{k:<12}{s.volume:>8}{s.n_trades:>8}{es:>12}{100 * s.snipe_share:>8.1f}")
    return "\n".join(rows)
