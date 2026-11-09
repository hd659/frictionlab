"""Observable-queue service venue with congestion frictions (fl-067, Naor 1969).

One server works on the simulation clock over steps t = 1 .. T. Each step runs:

1. arrival: one Bernoulli draw `u < arrival_prob_bps` on the arrival stream
   seeded by (seed, 0); an arriving customer observes the number n already in
   the system and joins iff (exact integer ceil, fl-040 conventions)

       R - toll - c * ceil((n + 1) * 10000 / service_prob_bps) >= 0   (ties join)

   A joiner pays the admission toll to the toll sink; a balker pays nothing.
2. waiting: every customer in the system (queue or service, the joiner too)
   pays c for the step to the waiting sink.
3. service: one Bernoulli draw `u < service_prob_bps` on the service stream
   seeded by (seed, 1); on success the head customer is served and collects R.

Both streams draw once every step, so changing one probability never moves the
other stream. A joiner with n ahead needs n + 1 successes, so its expected
steps in system are exactly (n + 1) * 10000 / service_prob_bps (negative
binomial) and the rule is the Naor individual-optimum: the system never holds
more than `threshold()` customers. Served customers reduce to fl-022
`TradeRecord`s tagged "queue"; cash moves only by `Ledger.transfer`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Union

import numpy as np

from ..core.ledger import Ledger
from ..core.types import Side, _check_int
from ..costs.congestion import CongestionCost

QUEUE = "queue"
BPS = 10_000


def ceil_div(a: int, b: int) -> int:
    """Exact integer ceil(a / b) for b > 0."""
    return -((-a) // b)


@dataclass(frozen=True)
class Arrival:
    """One arrival: customer id, queue length seen, expected net value, decision."""
    t: int
    customer: int
    n_seen: int
    expected_net: int
    joined: bool


@dataclass
class Customer:
    """A joined customer: join / departure steps, steps charged, served flag."""
    cid: int
    joined_t: int
    served_t: Optional[int] = None
    steps: int = 0

    @property
    def served(self) -> bool:
        return self.served_t is not None


def _bps(name: str, v: int, *, zero_ok: bool) -> int:
    v = _check_int(name, v, nonneg=True)
    if v > BPS or (v == 0 and not zero_ok):
        raise ValueError(f"{name} must be in [{0 if zero_ok else 1}, {BPS}] bps")
    return v


class QueueMarket:
    """Single-server observable queue with a toll and a per-step waiting cost."""

    def __init__(self, *, reward: int, arrival_prob_bps: int, service_prob_bps: int,
                 steps: int, congestion: Union[CongestionCost, None] = None, toll: int = 0,
                 waiting: int = 0, seed: int = 0, cash: int = 0) -> None:
        self.reward = _check_int("reward", reward)
        self.arrival_bps = _bps("arrival_prob_bps", arrival_prob_bps, zero_ok=True)
        self.service_bps = _bps("service_prob_bps", service_prob_bps, zero_ok=False)
        self.steps = _check_int("steps", steps, positive=True)
        self.congestion = congestion if congestion is not None else CongestionCost(toll, waiting)
        self.cash = _check_int("cash", cash)
        self._rng_a = np.random.default_rng([seed, 0])
        self._rng_s = np.random.default_rng([seed, 1])
        self.ledger = Ledger()
        self.congestion.attach(self.ledger)
        self.arrivals: List[Arrival] = []
        self.customers: Dict[int, Customer] = {}
        self.system: Deque[int] = deque()
        self.arrival_draws: List[bool] = []
        self.service_draws: List[bool] = []
        self.lengths: List[int] = []          # system size after step-1 arrivals
        self._next_id = 2
        self._ran = False

    # ---- model ---------------------------------------------------------
    def expected_steps(self, n: int) -> int:
        """ceil((n + 1) * 10000 / service_prob_bps): expected steps with n ahead."""
        return ceil_div((n + 1) * BPS, self.service_bps)

    def expected_net(self, n: int, cid: int = 0) -> int:
        """R - toll - c * expected_steps(n) for a customer seeing n in the system."""
        return self.congestion.expected_net(self.reward, self.expected_steps(n), cid)

    def threshold(self) -> Optional[int]:
        """Naor joining threshold N: join iff n < N; None means every arrival joins."""
        slack = self.reward - self.congestion.toll_for(0)
        c = self.congestion.wait_for(0)
        if slack < 0:
            return 0
        if c == 0:
            return None
        return (slack // c) * self.service_bps // BPS

    # ---- dynamics ------------------------------------------------------
    def _step(self, t: int) -> None:
        arrive = int(self._rng_a.integers(BPS)) < self.arrival_bps
        serve = int(self._rng_s.integers(BPS)) < self.service_bps
        self.arrival_draws.append(arrive)
        self.service_draws.append(serve)
        if arrive:
            cid, n = self._next_id, len(self.system)
            self._next_id += 1
            self.ledger.open_account(cid, self.cash)
            net = self.expected_net(n, cid)
            joined = net >= 0
            self.arrivals.append(Arrival(t, cid, n, net, joined))
            if joined:
                self.congestion.charge_toll(self.ledger, cid)
                self.customers[cid] = Customer(cid, t)
                self.system.append(cid)
        self.lengths.append(len(self.system))
        for cid in self.system:
            self.congestion.charge_wait(self.ledger, cid)
            self.customers[cid].steps += 1
        if serve and self.system:
            self.customers[self.system.popleft()].served_t = t

    def run(self) -> int:
        """Play steps 1 .. T once (idempotent); returns the number served."""
        if not self._ran:
            self._ran = True
            for t in range(1, self.steps + 1):
                self._step(t)
        return self.served

    # ---- views ---------------------------------------------------------
    @property
    def joins(self) -> int:
        """Arrivals that joined."""
        return sum(1 for a in self.arrivals if a.joined)

    @property
    def served(self) -> int:
        """Customers served so far."""
        return sum(1 for c in self.customers.values() if c.served)

    def records(self) -> list:
        """Served customers as fl-022 `TradeRecord`s, venue tag "queue".

        Price is the toll paid, notional the toll plus waiting cost paid.
        """
        from ..analysis.metrics import TradeRecord  # lazy: analysis sits above markets
        cc = self.congestion
        out = []
        for c in sorted(self.customers.values(), key=lambda c: (c.served_t or 0, c.cid)):
            if c.served:
                toll = cc.tolls.get(c.cid, 0)
                out.append(TradeRecord(len(out) + 1, float(c.served_t), toll, 1, Side.BUY,
                                       c.cid, cc.toll_sink_id, QUEUE,
                                       toll + cc.waits.get(c.cid, 0)))
        return out

    def __repr__(self) -> str:
        return (f"QueueMarket(steps={self.steps}, arrivals={len(self.arrivals)}, "
                f"joins={self.joins}, served={self.served}, in_system={len(self.system)})")
