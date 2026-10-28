"""Event-driven simulation clock with a seeded RNG and a deterministic event queue.

Ordering contract: events fire by (time, priority, seq). Lower priority fires
first at equal time; seq (insertion order) breaks remaining ties, so every run
is fully reproducible from the seed.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

Handler = Callable[["SimClock", "Event"], None]


@dataclass(order=True)
class Event:
    time: float
    priority: int
    seq: int
    kind: str = field(compare=False)
    payload: Any = field(default=None, compare=False)
    cancelled: bool = field(default=False, compare=False)


class EventQueue:
    """Min-heap of events with O(1) lazy cancellation."""

    def __init__(self) -> None:
        self._heap: List[Event] = []
        self._seq = itertools.count()
        self._live = 0

    def push(self, time: float, kind: str, payload: Any = None, priority: int = 0) -> Event:
        if not np.isfinite(time):
            raise ValueError(f"event time must be finite, got {time!r}")
        ev = Event(float(time), int(priority), next(self._seq), kind, payload)
        heapq.heappush(self._heap, ev)
        self._live += 1
        return ev

    def cancel(self, ev: Event) -> bool:
        if ev.cancelled:
            return False
        ev.cancelled = True
        self._live -= 1
        return True

    def _drop_cancelled(self) -> None:
        while self._heap and self._heap[0].cancelled:
            heapq.heappop(self._heap)

    def peek(self) -> Optional[Event]:
        self._drop_cancelled()
        return self._heap[0] if self._heap else None

    def pop(self) -> Event:
        self._drop_cancelled()
        if not self._heap:
            raise IndexError("pop from empty EventQueue")
        ev = heapq.heappop(self._heap)
        self._live -= 1
        ev.cancelled = True  # consumed: a later cancel() is a no-op
        return ev

    def __len__(self) -> int:
        return self._live

    def __bool__(self) -> bool:
        return self._live > 0


class SimClock:
    """Discrete-event driver: owns simulated time, the RNG and handler dispatch."""

    def __init__(self, seed: int = 0, start: float = 0.0) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._seed_seq = np.random.SeedSequence(self.seed)
        self.now = float(start)
        self.queue = EventQueue()
        self.handlers: Dict[str, List[Handler]] = {}
        self.processed = 0
        self._stopped = False

    # -- randomness -------------------------------------------------------
    def spawn_rng(self) -> np.random.Generator:
        """Independent, reproducible child stream (e.g. one per agent)."""
        (child,) = self._seed_seq.spawn(1)
        return np.random.default_rng(child)

    # -- scheduling -------------------------------------------------------
    def on(self, kind: str, handler: Handler) -> None:
        self.handlers.setdefault(kind, []).append(handler)

    def schedule_at(self, time: float, kind: str, payload: Any = None, priority: int = 0) -> Event:
        if time < self.now:
            raise ValueError(f"cannot schedule in the past: {time} < now={self.now}")
        return self.queue.push(time, kind, payload, priority)

    def schedule(self, delay: float, kind: str, payload: Any = None, priority: int = 0) -> Event:
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")
        return self.queue.push(self.now + delay, kind, payload, priority)

    def schedule_poisson(self, rate: float, kind: str, payload: Any = None, priority: int = 0) -> Event:
        """Schedule after an Exp(rate) waiting time drawn from the clock RNG."""
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        return self.schedule(float(self.rng.exponential(1.0 / rate)), kind, payload, priority)

    def cancel(self, ev: Event) -> bool:
        return self.queue.cancel(ev)

    def stop(self) -> None:
        self._stopped = True

    # -- execution --------------------------------------------------------
    def step(self) -> Optional[Event]:
        if not self.queue:
            return None
        ev = self.queue.pop()
        self.now = ev.time
        self.processed += 1
        for h in self.handlers.get(ev.kind, ()):
            h(self, ev)
        return ev

    def run(self, until: Optional[float] = None, max_events: Optional[int] = None) -> int:
        """Process events with time <= until. Returns number of events fired.

        If `until` is given, the clock advances to `until` once the horizon is
        reached, so callers can resume from a well-defined time.
        """
        self._stopped = False
        fired = 0
        while not self._stopped:
            if max_events is not None and fired >= max_events:
                break
            nxt = self.queue.peek()
            if nxt is None or (until is not None and nxt.time > until):
                if until is not None and until > self.now:
                    self.now = float(until)
                break
            self.step()
            fired += 1
        return fired
