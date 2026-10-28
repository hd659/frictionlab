"""Quoted-spread and per-agent PnL metrics computed from runner output.

Quoted spread. With zero latency the book only changes inside an agent
wake-up, so sampling (time, best_bid, best_ask) after every wake-up gives the
exact piecewise-constant touch path. `QuoteRecorder` is a `stop_when` hook
for `MarketRunner.run` that does exactly that (and can wrap another stop
predicate). For samples s_0..s_n and end time T:

    TW spread = sum_i spread_i * (t_{i+1} - t_i) / sum_i (t_{i+1} - t_i)

summed only over two-sided intervals; `coverage` is the fraction of [t_0, T]
during which the book was two-sided (a one-sided book has no quoted spread).

PnL. For agent a with fills (side_j, price_j, qty_j) and a mark m:

    pnl_a(m) = sum_j side_j * (m - price_j) * qty_j - fees_a - decision_costs_a

which equals the change in a's mark-to-market equity, independent of the
endowment. `markout_pnl` values each fill at the fundamental V(t_j) at trade
time instead, optionally restricted to a set of counterparties: the gross
(pre-fee) adverse-selection bill one agent pays another.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.types import Trade

Sample = Tuple[float, Optional[int], Optional[int]]  # (time, bid, ask)


@dataclass(frozen=True)
class SpreadStats:
    tw_spread: float     # time-weighted quoted spread over two-sided time (nan if none)
    coverage: float      # two-sided time / total time
    mean_spread: float   # unweighted mean over two-sided samples
    max_spread: int
    n_samples: int


class QuoteRecorder:
    """`stop_when` hook that samples the touch after every wake-up."""

    def __init__(self, stop_when: Optional[Callable[[object], bool]] = None,
                 t0: float = 0.0) -> None:
        self.samples: List[Sample] = [(float(t0), None, None)]
        self.stop_when = stop_when

    def __call__(self, runner) -> bool:
        book = runner.book
        self.record(runner.clock.now, book.best_bid, book.best_ask)
        return bool(self.stop_when(runner)) if self.stop_when else False

    def record(self, t: float, bid: Optional[int], ask: Optional[int]) -> None:
        t = float(t)
        if self.samples and t < self.samples[-1][0]:
            raise ValueError(f"sample time {t} precedes {self.samples[-1][0]}")
        if self.samples and self.samples[-1][0] == t:
            self.samples[-1] = (t, bid, ask)  # same instant: keep the latest state
        else:
            self.samples.append((t, bid, ask))

    def stats(self, until: Optional[float] = None) -> SpreadStats:
        return spread_stats(self.samples, until)


def _spread(bid: Optional[int], ask: Optional[int]) -> Optional[int]:
    return None if bid is None or ask is None else ask - bid


def spread_stats(samples: Sequence[Sample], until: Optional[float] = None) -> SpreadStats:
    """Time-weighted quoted spread of a piecewise-constant touch path."""
    if not samples:
        raise ValueError("need at least one sample")
    times = [float(s[0]) for s in samples]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("samples must be in time order")
    end = times[-1] if until is None else float(until)
    if end < times[-1]:
        raise ValueError(f"until={end} precedes the last sample {times[-1]}")
    weighted = covered = 0.0
    seen: List[int] = []
    for i, (t, bid, ask) in enumerate(samples):
        s = _spread(bid, ask)
        if s is None:
            continue
        seen.append(s)
        dt = (times[i + 1] if i + 1 < len(samples) else end) - t
        weighted += s * dt
        covered += dt
    total = end - times[0]
    nan = float("nan")
    return SpreadStats(weighted / covered if covered > 0 else nan,
                       covered / total if total > 0 else 0.0,
                       sum(seen) / len(seen) if seen else nan,
                       max(seen) if seen else 0, len(samples))


# ---- PnL -------------------------------------------------------------------
def _fills(trades: Iterable[Trade]):
    for t in trades:
        yield t.buyer_id, +1, t.seller_id, t
        yield t.seller_id, -1, t.buyer_id, t


def pnl_by_agent(result, mark: int) -> Dict[int, int]:
    """Net PnL (tick-lots) of every agent marked at `mark`, after all frictions."""
    pnl = {aid: -result.ledger[aid].fees_paid - a.decision_costs_paid
           for aid, a in result.agents.items()}
    for aid, side, _, t in _fills(result.trades):
        if aid in pnl:
            pnl[aid] += side * (mark - t.price) * t.qty
    return pnl


def pnl_by_type(result, mark: int) -> Dict[str, int]:
    """`pnl_by_agent` summed by agent class name."""
    out: Dict[str, int] = defaultdict(int)
    for aid, v in pnl_by_agent(result, mark).items():
        out[type(result.agents[aid]).__name__] += v
    return dict(out)


def markout_pnl(trades: Iterable[Trade], agent_id: int, value: Callable[[float], int],
                counterparties: Optional[Iterable[int]] = None) -> int:
    """Gross PnL of `agent_id`'s fills valued at V(trade time)."""
    cps = None if counterparties is None else set(counterparties)
    total = 0
    for aid, side, cp, t in _fills(trades):
        if aid == agent_id and (cps is None or cp in cps):
            total += side * (value(t.time) - t.price) * t.qty
    return total


def volume_between(trades: Iterable[Trade], a: int, others: Iterable[int]) -> int:
    """Lots agent `a` traded against any agent in `others`."""
    o = set(others)
    return sum(t.qty for aid, _, cp, t in _fills(trades) if aid == a and cp in o)
