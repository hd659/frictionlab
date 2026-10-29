"""Market-quality metrics on runner output: realized spread, impact, efficiency.

Every fill (continuous book, batch auction, AMM swap) reduces to one
canonical `TradeRecord`. With d = +1 for buyer- and -1 for seller-initiated
fills and the doubled mid M = bid + ask (an exact integer):

    effective_j = d * (2 p_j - M(t_j-))          vs the mid just before the fill
    realized_j  = d * (2 p_j - M(t_j + tau))     what the liquidity supplier keeps
    impact_j    = d * (M(t_j + tau) - M(t_j-))   permanent (informational) move

so effective = realized + impact holds exactly for every trade, in ticks with
the usual 2 * d * (p - mid) scaling of `structure_compare.effective_spread`.

The mid path is the piecewise-constant touch sampled by `spread.QuoteRecorder`
(exact with zero latency: the book only changes inside wake-ups). M(t-) is
the last sample strictly before t, M(t) the last sample at or before t.

Price efficiency is the time-weighted mean |mid - V(t)| over time with a mid
(breakpoints include the jumps of a `FundamentalProcess`), plus the variance
ratio VR(k) = Var(k-step mid change) / (k * Var(1-step change)) on a regular
grid: VR ~ 1 for a random walk, < 1 under mean reversion (bid-ask bounce).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Side, Trade


@dataclass(frozen=True)
class TradeRecord:
    trade_id: int
    time: float
    price: int        # integer settlement price (ticks)
    qty: int          # lots
    side: Side        # aggressor / initiator side
    buyer_id: int
    seller_id: int
    venue: str        # "book" | "batch" | "amm"
    notional: int     # exact cash exchanged (AMM: price * qty + rounding remainder)

    @classmethod
    def from_trade(cls, t: Trade, venue: str = "book") -> "TradeRecord":
        return cls(t.trade_id, float(t.time), t.price, t.qty, Side(t.aggressor),
                   t.buyer_id, t.seller_id, venue, t.price * t.qty)

    @classmethod
    def from_swap(cls, s) -> "TradeRecord":
        st, side = s.settlement, Side(s.side)
        quote = s.amount_in if side is Side.BUY else s.amount_out
        return cls(st.trade_id, float(st.time), st.price, st.qty, side,
                   st.buyer_id, st.seller_id, "amm", quote)


def trade_records(result) -> List[TradeRecord]:
    """Canonical, time-ordered fills of a `RunResult` (or a plain trade list)."""
    if not hasattr(result, "trades"):
        return [TradeRecord.from_trade(t) for t in result]
    venue = "batch" if hasattr(result.book, "clear") else "book"
    recs = [TradeRecord.from_trade(t, venue) for t in result.trades]
    recs += [TradeRecord.from_swap(s) for s in getattr(result.venue, "swaps", ())]
    return sorted(recs, key=lambda r: r.time)  # stable: book fills first per instant


class MidPath:
    """Step function of the doubled mid M = bid + ask.

    With `carry` (default) a one-sided interval keeps the last two-sided mid,
    so a sweep that empties one side still has a post-trade mid; otherwise
    one-sided intervals are None. Before the first two-sided sample: None.
    """

    def __init__(self, samples, carry: bool = True) -> None:
        samples = list(getattr(samples, "samples", samples))
        if not samples:
            raise ValueError("need at least one sample")
        self.times = [float(s[0]) for s in samples]
        if any(b < a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("samples must be in time order")
        self.mid2 = [None if b is None or a is None else int(b) + int(a)
                     for _, b, a in samples]
        if carry:
            for i in range(1, len(self.mid2)):
                if self.mid2[i] is None:
                    self.mid2[i] = self.mid2[i - 1]

    def _get(self, i: int) -> Optional[int]:
        return None if i < 0 else self.mid2[i]

    def at(self, t: float) -> Optional[int]:
        return self._get(bisect.bisect_right(self.times, float(t)) - 1)

    def before(self, t: float) -> Optional[int]:
        return self._get(bisect.bisect_left(self.times, float(t)) - 1)

    def grid(self, dt: float, until: Optional[float] = None) -> np.ndarray:
        """Mid (ticks) on t0, t0+dt, ...; points before the first mid dropped."""
        end = self.times[-1] if until is None else float(until)
        ms = (self.at(t) for t in np.arange(self.times[0], end + 1e-12, float(dt)))
        return np.asarray([m / 2 for m in ms if m is not None], dtype=float)


@dataclass(frozen=True)
class TradeMetrics:
    record: TradeRecord
    effective: int
    realized: int
    impact: int


def decompose(records: Iterable[TradeRecord], path: MidPath,
              tau: float) -> List[TradeMetrics]:
    """Per-trade effective / realized / impact; trades without both mids skipped."""
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    out = []
    for r in records:
        pre, post = path.before(r.time), path.at(r.time + tau)
        if pre is None or post is None:
            continue
        d = r.side.value
        out.append(TradeMetrics(r, d * (2 * r.price - pre), d * (2 * r.price - post),
                                d * (post - pre)))
    return out


@dataclass(frozen=True)
class SpreadDecomposition:
    tau: float
    effective: float   # volume-weighted means, ticks (nan if no trades)
    realized: float
    impact: float
    volume: int
    n_trades: int
    n_skipped: int


def spread_decomposition(records: Sequence[TradeRecord], path: MidPath,
                         tau: float) -> SpreadDecomposition:
    ms = decompose(records, path, tau)
    vol = sum(m.record.qty for m in ms)
    avg = (lambda f: sum(f(m) * m.record.qty for m in ms) / vol) if vol else (
        lambda f: math.nan)
    return SpreadDecomposition(float(tau), avg(lambda m: m.effective),
                               avg(lambda m: m.realized), avg(lambda m: m.impact),
                               vol, len(ms), len(records) - len(ms))


def impact_by_size(metrics: Sequence[TradeMetrics], edges: Sequence[int] = (1,)
                   ) -> Dict[Tuple[int, Optional[int]], Tuple[int, float]]:
    """{(lo, hi): (n_trades, mean impact)} over size buckets [lo, hi); last hi=None."""
    edges = sorted(int(e) for e in edges)
    if not edges or edges[0] < 1 or len(set(edges)) != len(edges):
        raise ValueError(f"edges must be distinct positive ints, got {edges}")
    keys = list(zip(edges, edges[1:] + [None]))
    acc: Dict[Tuple[int, Optional[int]], List[int]] = {k: [] for k in keys}
    for m in metrics:
        i = bisect.bisect_right(edges, m.record.qty) - 1
        if i >= 0:
            acc[keys[i]].append(m.impact)
    return {k: (len(v), float(np.mean(v)) if v else math.nan) for k, v in acc.items()}


def variance_ratio(prices: Sequence[float], k: int = 2) -> float:
    """Lo-MacKinlay overlapping VR(k); nan if too short or flat."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    x = np.asarray(prices, dtype=float)
    r1 = np.diff(x)
    if len(r1) < k + 1 or np.var(r1) == 0:
        return math.nan
    return float(np.var(x[k:] - x[:-k]) / (k * np.var(r1)))


@dataclass(frozen=True)
class EfficiencyStats:
    mad: float              # time-weighted mean |mid - V| where a mid exists, ticks
    coverage: float         # time with a mid / total time
    variance_ratio: float


def price_efficiency(path: MidPath, value: Callable[[float], int], until: float,
                     *, k: int = 4, dt: float = 1.0) -> EfficiencyStats:
    t0, end = path.times[0], float(until)
    if end <= t0:
        raise ValueError(f"until={end} must exceed the first sample {t0}")
    proc = getattr(value, "__self__", value)  # bound FundamentalProcess.value
    jumps = [t for t, _ in proc.jumps(end)] if hasattr(proc, "jumps") else []
    cuts = sorted({t for t in path.times + jumps if t0 <= t < end})
    err = covered = 0.0
    for a, b in zip(cuts, cuts[1:] + [end]):
        m = path.at(a)
        if m is not None:
            err += abs(m - 2 * int(value(a))) / 2 * (b - a)
            covered += b - a
    vr = variance_ratio(path.grid(dt, end), k)
    return EfficiencyStats(err / covered if covered else math.nan, covered / (end - t0), vr)
