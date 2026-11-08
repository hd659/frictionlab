"""Akerlof market for lemons (fl-055): hidden per-unit quality, pooled price, certification.

Seller i holds one unit of hidden integer-tick quality q_i (reservation r(q) = q);
buyers value a unit at q + premium but see only the pool actually offered. The
market posts a uniform price by fixed-point iteration from p_(-1) = infinity:

    O_k     = {i uncertified at p_(k-1) : q_i <= p_(k-1)}       (offered pool)
    p_k     = floor(sum_{O_k} q / |O_k|) + premium                (buyer value of the mean)

with the fl-040 remainder convention (a benefit rounds down, the remainder stays
with the buyers). It stops when a price repeats or O_k is empty. Without
certification the offered set is a lower set of qualities, so p_k is
non-increasing and the loop terminates. With premium 0 the pool unravels to the
lowest quality; with premium >= max(q) - min(q) the whole pool trades at p_0.

An optional `CertificationPolicy` lets a seller pay c(q) (to `CERT_SINK_ID`) to
sell in a separate certified segment at q + premium; it certifies iff
q + premium - c(q) >= max(p, q) at the last posted price p. Each seller sells at
most one unit to one buyer (ids in order); fills settle through the `Ledger` and
reduce to fl-022 `TradeRecord`s tagged "pooled" or "certified". A disabled or
absent policy reproduces the pure pooled run bit-for-bit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.ledger import Ledger
from ..core.types import Side, Trade, _check_int
from ..costs.certification import CertificationPolicy

POOLED, CERTIFIED = "pooled", "certified"


def pooled_price(qualities: Sequence[int], premium: int) -> Optional[int]:
    """floor(mean quality) + premium in ticks, or None for an empty pool."""
    qs = [int(q) for q in qualities]
    return sum(qs) // len(qs) + premium if qs else None


@dataclass(frozen=True)
class LemonsRound:
    """One posted price: the pool it was computed from and the certified count."""
    k: int
    price: int
    offered: Tuple[int, ...]     # seller ids in the pool
    certified: Tuple[int, ...]   # seller ids certifying at the previous price


class LemonsMarket:
    """Deterministic hidden-quality market with an optional certification segment."""

    def __init__(self, qualities: Sequence[int], *, premium: int = 0,
                 n_buyers: Optional[int] = None, cash: int = 0,
                 certification: Optional[CertificationPolicy] = None,
                 max_rounds: int = 10_000) -> None:
        if not qualities:
            raise ValueError("need at least one seller")
        self.qualities: Dict[int, int] = {i + 1: _check_int("quality", q, positive=True)
                                          for i, q in enumerate(qualities)}
        self.premium = _check_int("premium", premium, nonneg=True)
        n = len(self.qualities)
        nb = n if n_buyers is None else _check_int("n_buyers", n_buyers, nonneg=True)
        self.buyer_ids = list(range(n + 1, n + 1 + nb))
        self.cert = certification if certification is not None else CertificationPolicy.disabled()
        self.max_rounds = _check_int("max_rounds", max_rounds, positive=True)
        self.ledger = Ledger()
        for sid in self.qualities:
            self.ledger.open_account(sid, 0, 1)
        for bid in self.buyer_ids:
            self.ledger.open_account(bid, cash)
        self.cert.attach(self.ledger)
        self.rounds: List[LemonsRound] = []
        self.trades: List[Trade] = []
        self.segments: Dict[int, str] = {}       # trade_id -> segment
        self.cert_paid: Dict[int, int] = {}      # seller id -> certification ticks
        self.price: Optional[int] = None
        self.certified: Tuple[int, ...] = ()
        self.offered: Tuple[int, ...] = ()
        self._ran = False

    @classmethod
    def from_schedule(cls, n_sellers: int, schedule: Sequence[int], seed: int = 0,
                      **kw) -> "LemonsMarket":
        """Draw each seller's quality uniformly from the discrete `schedule` (seeded)."""
        n = _check_int("n_sellers", n_sellers, positive=True)
        sched = np.asarray([_check_int("quality", q, positive=True) for q in schedule])
        idx = np.random.default_rng(seed).integers(len(sched), size=n)
        return cls([int(q) for q in sched[idx]], **kw)

    # ---- model ---------------------------------------------------------
    def buyer_value(self, quality: int) -> int:
        """Buyer value of a unit of known `quality`."""
        return quality + self.premium

    def _split(self, prev: Optional[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if prev is None:
            return tuple(self.qualities), ()
        cert = tuple(i for i, q in self.qualities.items()
                     if self.cert.certifies(i, q, self.premium, prev))
        pool = tuple(i for i, q in self.qualities.items() if i not in cert and q <= prev)
        return pool, cert

    def iterate(self) -> Optional[int]:
        """Run the price fixed point; returns the final pooled price (None if no pool)."""
        if self.rounds:
            return self.price
        prev, seen = None, set()
        for k in range(self.max_rounds):
            pool, cert = self._split(prev)
            self.offered, self.certified = pool, cert
            if not pool:
                self.price = None
                return None
            p = pooled_price([self.qualities[i] for i in pool], self.premium)
            self.rounds.append(LemonsRound(k, p, pool, cert))
            if p in seen:
                break
            seen.add(p)
            prev = self.price = p
        return self.price

    # ---- settlement ----------------------------------------------------
    def _fill(self, price: int, buyer: int, seller: int, segment: str) -> None:
        tid = len(self.trades) + 1
        trade = Trade(tid, float(tid), price, 1, tid, tid, buyer, seller, Side.BUY)
        self.ledger.settle(trade)
        self.trades.append(trade)
        self.segments[tid] = segment

    def run(self) -> int:
        """Iterate, charge certifications, clear both segments; returns the trade count."""
        if self._ran:
            return len(self.trades)
        self._ran = True
        self.iterate()
        for sid in self.certified:
            self.cert_paid[sid] = self.cert.charge(self.ledger, sid, self.qualities[sid])
        # a cycle (only possible with certification) may end at p < prev: keep r(q) <= p
        orders = [(sid, self.price, POOLED) for sid in self.offered
                  if self.qualities[sid] <= self.price]
        orders += [(sid, self.buyer_value(self.qualities[sid]), CERTIFIED)
                   for sid in self.certified]
        for (sid, price, seg), bid in zip(orders, self.buyer_ids):
            self._fill(price, bid, sid, seg)
        return len(self.trades)

    @property
    def certification_costs(self) -> int:
        """Total ticks paid for certification in this run."""
        return sum(self.cert_paid.values())

    def sellers_sold(self, segment: Optional[str] = None) -> List[int]:
        """Seller ids that sold, optionally restricted to one segment."""
        return [t.seller_id for t in self.trades
                if segment is None or self.segments[t.trade_id] == segment]

    def records(self) -> list:
        """Fills as fl-022 `TradeRecord`s, venue tag "pooled" or "certified"."""
        from ..analysis.metrics import TradeRecord  # lazy: analysis sits above markets
        return [TradeRecord.from_trade(t, self.segments[t.trade_id]) for t in self.trades]

    def __repr__(self) -> str:
        return (f"LemonsMarket(sellers={len(self.qualities)}, premium={self.premium}, "
                f"price={self.price}, trades={len(self.trades)})")
