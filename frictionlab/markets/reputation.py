"""Repeated hidden-quality market with public reputation (fl-058).

Reputation as a dynamic substitute for fl-055 certification. Seller i holds a
hidden integer-tick quality q_i from the fl-055 schedule and plays T seeded
rounds under a persistent public id. Honest sellers always deliver q_i; an
opportunist (`ReputationSeller`) may deliver the schedule floor f and save
q_i - f. After each trade the buyer rates the delivery: the rating is correct
with probability `rating_accuracy_bps` / 10000, drawn from a per-seller stream
seeded by (seed, seller id) so ratings never move other agents' draws.

The public score is an integer (good, bad) count. Buyers hold a fixed integer
prior `prior_honest_bps` that a fresh id is honest and model an opportunist as
always delivering f. With a = accuracy, e = 10000 - a:

    w_H = prior * a^g * e^b,   w_O = (10000 - prior) * e^g * a^b
    E[q | g, b] = f + (m - f) * w_H // (w_H + w_O)        (f if w_H + w_O == 0)
    price(g, b) = E[q | g, b] + premium

where m + premium is the fl-055 pooled fixed point of the qualities, so with
T = 1 and no opportunists the market reproduces `LemonsMarket` prices. All
arithmetic is exact integer floor division (fl-040: remainders stay with
buyers). A seller trades iff the price covers its delivered quality; each round
it may first whitewash (fresh id, score (0, 0)) by paying `whitewash_cost`
through a friction stack to `IDENTITY_SINK_ID`. Fills settle through the
`Ledger` and reduce to fl-022 `TradeRecord`s tagged "reputation".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..agents.reputation_seller import ReputationSeller
from ..core.ledger import Ledger
from ..core.types import Side, Trade, _check_int
from ..costs.base import Fill, Role
from ..costs.fees import FixedCost
from ..costs.stack import FrictionStack
from .lemons import LemonsMarket

REPUTATION = "reputation"
IDENTITY_SINK_ID = -8
BPS = 10_000


def score_quality(good: int, bad: int, mean_q: int, floor_q: int,
                  accuracy_bps: int, prior_honest_bps: int) -> int:
    """Integer expected quality given a (good, bad) score (see module docstring)."""
    a, e = accuracy_bps, BPS - accuracy_bps
    w_h = prior_honest_bps * a ** good * e ** bad
    w_o = (BPS - prior_honest_bps) * e ** good * a ** bad
    if w_h + w_o == 0:
        return floor_q
    return floor_q + (mean_q - floor_q) * w_h // (w_h + w_o)


@dataclass(frozen=True)
class ReputationRound:
    """One seller-round: score seen by buyers, posted price and what happened."""
    t: int
    seller_id: int
    public_id: int
    score: Tuple[int, int]
    price: int
    delivered: Optional[int]     # None if the seller did not trade
    rating: Optional[bool]       # True = good
    whitewashed: bool


class ReputationMarket:
    """T-round posted-price market with ratings, scores, cheating and whitewashing."""

    def __init__(self, qualities: Sequence[int], *, opportunists: Iterable[int] = (),
                 rounds: int = 1, premium: int = 0, rating_accuracy_bps: int = BPS,
                 whitewash_cost: Optional[int] = None, prior_honest_bps: int = BPS,
                 floor: Optional[int] = None, seed: int = 0, cash: int = 0) -> None:
        if not qualities:
            raise ValueError("need at least one seller")
        opp = set(opportunists)
        self.sellers: Dict[int, ReputationSeller] = {
            i + 1: ReputationSeller(i + 1, q, opportunist=(i + 1) in opp)
            for i, q in enumerate(qualities)}
        self.rounds = _check_int("rounds", rounds, positive=True)
        self.premium = _check_int("premium", premium, nonneg=True)
        self.accuracy = _check_int("rating_accuracy_bps", rating_accuracy_bps, nonneg=True)
        self.prior = _check_int("prior_honest_bps", prior_honest_bps, nonneg=True)
        if self.accuracy > BPS or self.prior > BPS:
            raise ValueError("bps parameters must be <= 10000")
        qs = [s.quality for s in self.sellers.values()]
        self.floor = min(qs) if floor is None else _check_int("floor", floor, positive=True)
        self.mean_q = LemonsMarket(qs, premium=self.premium).iterate() - self.premium
        self.wash = None if whitewash_cost is None else FrictionStack(
            [FixedCost(_check_int("whitewash_cost", whitewash_cost, nonneg=True))])
        n = len(self.sellers)
        self.buyer_ids = list(range(n + 1, 2 * n + 1))
        self.ledger = Ledger()
        for sid in self.sellers:
            self.ledger.open_account(sid)
        for bid in self.buyer_ids:
            self.ledger.open_account(bid, cash)
        if self.wash is not None:
            self.ledger.open_account(IDENTITY_SINK_ID)
        self._rng = {sid: np.random.default_rng([seed, sid]) for sid in self.sellers}
        self._next_public = 0
        self.scores: Dict[int, Tuple[int, int]] = {}
        for s in self.sellers.values():
            self._fresh_id(s)
        self.log: List[ReputationRound] = []
        self.trades: List[Trade] = []
        self.delivered: Dict[int, int] = {}       # trade_id -> delivered quality
        self.wash_paid: Dict[int, int] = dict.fromkeys(self.sellers, 0)
        self._ran = False

    @classmethod
    def from_schedule(cls, n_sellers: int, schedule: Sequence[int], seed: int = 0,
                      **kw) -> "ReputationMarket":
        """Draw qualities exactly as fl-055 `LemonsMarket.from_schedule`; floor = min(schedule)."""
        n = _check_int("n_sellers", n_sellers, positive=True)
        sched = np.asarray([_check_int("quality", q, positive=True) for q in schedule])
        idx = np.random.default_rng(seed).integers(len(sched), size=n)
        kw.setdefault("floor", int(sched.min()))
        return cls([int(q) for q in sched[idx]], seed=seed, **kw)

    # ---- model ---------------------------------------------------------
    def buyer_value(self, quality: int) -> int:
        return quality + self.premium

    def expected_quality(self, score: Tuple[int, int]) -> int:
        return score_quality(score[0], score[1], self.mean_q, self.floor,
                             self.accuracy, self.prior)

    def price_for(self, score: Tuple[int, int]) -> int:
        """Buyer value of the score-implied expected quality."""
        return self.buyer_value(self.expected_quality(score))

    @property
    def fresh_price(self) -> int:
        return self.price_for((0, 0))

    @property
    def floor_price(self) -> int:
        return self.buyer_value(self.floor)

    def whitewash_fee(self, seller: ReputationSeller) -> Optional[int]:
        if self.wash is None:
            return None
        return self.wash.charge(Fill(seller.agent_id, Side.SELL, Role.TAKER, seller.quality, 1))

    def _fresh_id(self, s: ReputationSeller) -> None:
        self._next_public += 1
        s.public_id = self._next_public
        self.scores[s.agent_id] = (0, 0)

    # ---- dynamics ------------------------------------------------------
    def _seller_round(self, t: int, s: ReputationSeller, k: int) -> int:
        sid = s.agent_id
        fee = self.whitewash_fee(s)
        washed = s.will_whitewash(self.fresh_price, fee, self.price_for(self.scores[sid]))
        if washed:
            self.ledger.transfer(sid, IDENTITY_SINK_ID, fee)
            self.wash_paid[sid] += fee
            s.whitewashes += 1
            self._fresh_id(s)
        score = self.scores[sid]
        price = self.price_for(score)
        cheat = s.will_cheat(t, self.rounds, price, self.floor_price, self.floor)
        out = self.floor if cheat else s.quality
        if not s.will_sell(price, out):
            self.log.append(ReputationRound(t, sid, s.public_id, score, price, None, None, washed))
            return k
        if cheat:
            s.record_cheat(t)
        self.ledger.deposit(sid, position=1)                  # produce the unit
        tid = len(self.trades) + 1
        trade = Trade(tid, float(t), price, 1, tid, tid, self.buyer_ids[k], sid, Side.BUY)
        self.ledger.settle(trade)
        self.trades.append(trade)
        self.delivered[tid] = out
        correct = int(self._rng[sid].integers(BPS)) < self.accuracy
        good = (out >= s.quality) == correct
        g, b = score
        self.scores[sid] = (g + 1, b) if good else (g, b + 1)
        self.log.append(ReputationRound(t, sid, s.public_id, score, price, out, good, washed))
        return k + 1

    def run(self) -> int:
        """Play all T rounds (sellers in id order); returns the trade count."""
        if not self._ran:
            self._ran = True
            for t in range(self.rounds):
                k = 0
                for s in self.sellers.values():
                    k = self._seller_round(t, s, k)
        return len(self.trades)

    # ---- views ---------------------------------------------------------
    @property
    def whitewash_costs(self) -> int:
        return sum(self.wash_paid.values())

    def history(self, seller_id: int) -> List[ReputationRound]:
        return [r for r in self.log if r.seller_id == seller_id]

    def records(self) -> list:
        """Fills as fl-022 `TradeRecord`s, venue tag "reputation"."""
        from ..analysis.metrics import TradeRecord  # lazy: analysis sits above markets
        return [TradeRecord.from_trade(t, REPUTATION) for t in self.trades]

    def __repr__(self) -> str:
        return (f"ReputationMarket(sellers={len(self.sellers)}, rounds={self.rounds}, "
                f"trades={len(self.trades)}, whitewash={self.whitewash_costs})")
