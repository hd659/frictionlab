"""Bilateral hold-up market with costly contracting (fl-073).

Pair k is one `SpecificInvestor` seller with its own investment schedule
((investment_cost, buyer_value), ...). Over rounds t = 1 .. T each pair:

1. draws its buyer from `buyers` on the pairing stream seeded by (seed, k, 0) and
   an integer value shock e from `value_noise` on the noise stream seeded by
   (seed, k, 1); both streams draw once per round whatever anyone does, so one
   pair's schedule or choice never moves another pair's RNG. The shock lifts
   every buyer value by e (the outside value o does not move);
2. the seller picks a segment and level (`SpecificInvestor.choose`) and sinks
   investment_cost(i) into the investment sink (`INVEST_SINK_ID`);
3. hold-up: price = o + seller_share(Q(i), w) with Q(i) = buyer_value(i) - o,
   integer Nash bargaining under the fl-040 floor convention, so the buyer keeps
   buyer_share(Q(i), w) and the shares sum to Q(i) exactly;
   contract: the seller first pays the `ContractingCost` to the contract sink and
   both sides commit to the first best i* at price o + c(i*) + seller_share(S*, w),
   where S* = Q(i*) - c(i*) is the net first-best surplus;
4. the buyer receives buyer_value(i) from the value source (`VALUE_ID`) and pays
   the price to the seller.

Cash moves only by `Ledger.transfer`; trades reduce to fl-022 `TradeRecord`s
tagged "holdup" or "contract". Seller net PnL = revenue - investment - contracting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..agents.specific_investor import (
    SpecificInvestor, check_schedule, check_weight, seller_share,
)
from ..core.ledger import Ledger
from ..core.types import Side, _check_int
from ..costs.contracting import ContractingCost

HOLDUP = "holdup"
CONTRACT = "contract"
INVEST_SINK_ID = -16
VALUE_ID = -17


@dataclass(frozen=True)
class HoldupTrade:
    """One pair-round: segment, investment, value, price and split (ticks)."""
    t: int
    pair: int
    seller: int
    buyer: int
    segment: str
    level: int
    noise: int
    investment: int     # investment_cost(level), sunk by the seller
    value: int          # buyer_value(level) + noise
    price: int
    contract_cost: int  # 0 under hold-up

    @property
    def buyer_surplus(self) -> int:
        return self.value - self.price


class HoldupMarket:
    """Buyer-seller pairs trading a relationship-specific good, with optional contracts."""

    def __init__(self, sellers: Sequence[SpecificInvestor], schedules, buyers: Sequence[int], *,
                 rounds: int, outside_value: int, seller_weight_bps: int = 5_000,
                 value_noise: Sequence[int] = (0,),
                 contracting: Optional[ContractingCost] = None, seed: int = 0) -> None:
        if not sellers or len(sellers) != len(schedules):
            raise ValueError("need one investment schedule per seller (at least one seller)")
        ids = [s.agent_id for s in sellers]
        self.buyers = [_check_int("buyer", b, positive=True) for b in buyers]
        if not self.buyers or len(set(ids + self.buyers)) != len(ids) + len(self.buyers) \
                or min(ids) < 1:
            raise ValueError("seller and buyer ids must be distinct positive ints")
        self.sellers: List[SpecificInvestor] = list(sellers)
        self.schedules = [check_schedule(s) for s in schedules]
        self.rounds = _check_int("rounds", rounds, positive=True)
        self.outside = _check_int("outside_value", outside_value, nonneg=True)
        self.w = check_weight(seller_weight_bps)
        self.noise = [_check_int("value_noise", e) for e in value_noise]
        if not self.noise:
            raise ValueError("value_noise schedule must be non-empty")
        if min(s[0][1] for s in self.schedules) + min(self.noise) < self.outside:
            raise ValueError("buyer_value + noise must be >= outside_value (non-negative rent)")
        self.contracting = contracting or ContractingCost.disabled()
        self._pair_rng = [np.random.default_rng([seed, k, 0]) for k in range(len(ids))]
        self._noise_rng = [np.random.default_rng([seed, k, 1]) for k in range(len(ids))]
        self.ledger = Ledger()
        for a in (VALUE_ID, INVEST_SINK_ID, *ids, *self.buyers):
            self.ledger.open_account(a)
        if self.contracting.enabled:
            self.contracting.attach(self.ledger)
        self.log: List[HoldupTrade] = []
        self._ran = False

    # ---- one pair-round ------------------------------------------------
    def _trade(self, t: int, k: int) -> HoldupTrade:
        buyer = self.buyers[int(self._pair_rng[k].integers(len(self.buyers)))]
        e = self.noise[int(self._noise_rng[k].integers(len(self.noise)))]
        sched = tuple((c, v + e) for c, v in self.schedules[k])
        s, o, cc = self.sellers[k], self.outside, self.contracting
        cost = cc.cost_for(s.agent_id) if cc.enabled else 0
        seg, i = s.choose(sched, o, self.w, cost, cc.enabled)
        inv, value = sched[i]
        if seg == CONTRACT:
            paid = cc.charge(self.ledger, s.agent_id)
            price = o + inv + seller_share(value - o - inv, self.w)
        else:
            paid, price = 0, o + seller_share(value - o, self.w)
        self.ledger.transfer(s.agent_id, INVEST_SINK_ID, inv)
        self.ledger.transfer(VALUE_ID, buyer, value)
        self.ledger.transfer(buyer, s.agent_id, price)
        return HoldupTrade(t, k, s.agent_id, buyer, seg, i, e, inv, value, price, paid)

    def run(self) -> int:
        """Play rounds 1 .. T once (idempotent); returns the number of contracts."""
        if not self._ran:
            self._ran = True
            for t in range(1, self.rounds + 1):
                for k in range(len(self.sellers)):
                    self.log.append(self._trade(t, k))
        return self.contracts

    # ---- views ---------------------------------------------------------
    @property
    def contracts(self) -> int:
        return sum(1 for r in self.log if r.segment == CONTRACT)

    def pair(self, k: int) -> List[HoldupTrade]:
        """Per-round log of pair k."""
        return [r for r in self.log if r.pair == k]

    def draws(self, k: int) -> List[tuple]:
        """(buyer, noise) draws of pair k, one per round."""
        return [(r.buyer, r.noise) for r in self.pair(k)]

    def quasi_rent(self, r: HoldupTrade) -> int:
        """Ex-post quasi-rent buyer_value(i) - outside_value of a trade."""
        return r.value - self.outside

    def records(self) -> list:
        """Trades as fl-022 `TradeRecord`s tagged "holdup" / "contract" (one lot each)."""
        from ..analysis.metrics import TradeRecord  # lazy: analysis sits above markets
        return [TradeRecord(n + 1, float(r.t), r.price, 1, Side.BUY, r.buyer, r.seller,
                            r.segment, r.price) for n, r in enumerate(self.log)]

    def __repr__(self) -> str:
        return (f"HoldupMarket(pairs={len(self.sellers)}, rounds={self.rounds}, "
                f"w={self.w}, contracts={self.contracts})")


def seller_ids(market: HoldupMarket) -> Dict[int, int]:
    """Pair index -> seller id."""
    return {k: s.agent_id for k, s in enumerate(market.sellers)}
