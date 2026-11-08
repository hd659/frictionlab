"""Repeated two-seller market with consumer switching costs (fl-061, Klemperer).

Two `LockinSeller`s (ids 1 and 2) post one integer-tick price per round for T
seeded rounds. Consumer i (ids 3 .. n + 2) has an integer-tick valuation v_i,
a persistent current supplier and a switching cost s_i drawn once from a
discrete schedule on its own stream seeded by (seed, consumer id), so adding a
consumer never moves another agent's draw. With incumbent price p_k and rival
price p_j, an attached consumer

    switches to j   iff  p_j + s_i < p_k  and  p_j + s_i <= v_i   (ties stay)
    else buys at k  iff  p_k <= v_i                                (else sits out)

An unattached consumer (round 0, or one that never bought) picks the lower
price, ties to the lower seller id, buys iff that price <= v_i and becomes
attached. A seller with `entry_round` > t is absent and posts no price (None):
consumers then face only the present seller and nobody can switch to it. A switch charges s_i through a per-cost friction stack and moves it
to `SWITCH_SINK_ID`. Nobody pays above valuation: price + paid switching cost
<= v_i on every trade.

Timing: shares (bps of all consumers attached) are read at the start of the
round. A seller below its target share acquires, one at or above it harvests.
Targets are > 5000 bps, so at most one seller harvests in a round; it posts
after its rival and sees the rival's current price, which makes the harvest
price `rival + min s` (capped by min v) switch-proof by construction. Fills
settle through the `Ledger` and reduce to fl-022 `TradeRecord`s tagged
"switching".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..agents.lockin_seller import ACQUIRE, HARVEST, LockinSeller
from ..core.ledger import Ledger
from ..core.types import Side, Trade, _check_int
from ..costs.base import Fill, Role
from ..costs.fees import FixedCost
from ..costs.stack import FrictionStack

SWITCHING = "switching"
SWITCH_SINK_ID = -9
BPS = 10_000


@dataclass(frozen=True)
class SwitchingRound:
    """One round: posted prices and phases per seller, sales per seller and switches."""
    t: int
    prices: Tuple[Optional[int], Optional[int]]
    phases: Tuple[str, str]
    sales: Tuple[int, int]
    switches: int


class SwitchingMarket:
    """T-round duopoly where consumers pay a seeded switching cost to change supplier."""

    def __init__(self, valuations: Sequence[int], sellers: Sequence[LockinSeller], *,
                 switch_costs: Sequence[int] = (0,), rounds: int = 1, seed: int = 0,
                 cash: int = 0) -> None:
        if not valuations:
            raise ValueError("need at least one consumer")
        if sorted(s.agent_id for s in sellers) != [1, 2]:
            raise ValueError("SwitchingMarket needs exactly two sellers with ids 1 and 2")
        sched = [_check_int("switch_cost", c, nonneg=True) for c in switch_costs]
        if not sched:
            raise ValueError("switch_costs schedule must be non-empty")
        self.sellers: Dict[int, LockinSeller] = {s.agent_id: s for s in sorted(
            sellers, key=lambda s: s.agent_id)}
        self.rounds = _check_int("rounds", rounds, positive=True)
        ids = range(3, len(valuations) + 3)
        self.values: Dict[int, int] = {
            c: _check_int("valuation", v, positive=True) for c, v in zip(ids, valuations)}
        self.costs: Dict[int, int] = {
            c: sched[int(np.random.default_rng([seed, c]).integers(len(sched)))] for c in ids}
        self.supplier: Dict[int, Optional[int]] = dict.fromkeys(ids)
        self._stacks = {c: FrictionStack([FixedCost(c)]) for c in set(sched)}
        self.ledger = Ledger()
        for sid in self.sellers:
            self.ledger.open_account(sid)
        for cid in ids:
            self.ledger.open_account(cid, cash)
        self.ledger.open_account(SWITCH_SINK_ID)
        self.log: List[SwitchingRound] = []
        self.trades: List[Trade] = []
        self.phase_of: Dict[int, str] = {}           # trade_id -> seller phase
        self.switch_paid: Dict[int, int] = dict.fromkeys(ids, 0)
        self.switch_count = 0
        self._ran = False

    # ---- state ---------------------------------------------------------
    def customers(self, seller_id: int) -> List[int]:
        return [c for c, k in self.supplier.items() if k == seller_id]

    def share_bps(self, seller_id: int) -> int:
        return len(self.customers(seller_id)) * BPS // len(self.values)

    def switch_fee(self, consumer_id: int) -> int:
        s = self.costs[consumer_id]
        return self._stacks[s].charge(Fill(consumer_id, Side.BUY, Role.TAKER, 1, 1))

    def post_prices(self, t: int = 0) -> Tuple[Dict[int, Optional[int]], Dict[int, str]]:
        """Non-harvesters post first; a harvester then sees its rival's current price."""
        phases = {sid: s.phase(self.share_bps(sid), t) for sid, s in self.sellers.items()}
        prices: Dict[int, Optional[int]] = {}
        for sid, s in self.sellers.items():
            if phases[sid] != HARVEST:
                prices[sid] = s.post_price(phases[sid])
        for sid, s in self.sellers.items():
            if phases[sid] == HARVEST:
                rival = prices[3 - sid]
                own = [(self.values[c], self.costs[c]) for c in self.customers(sid)]
                prices[sid] = s.post_price(HARVEST, rival, own)
        return prices, phases

    # ---- dynamics ------------------------------------------------------
    def _choose(self, cid: int, prices: Dict[int, Optional[int]]) -> Tuple[Optional[int], bool]:
        v, k = self.values[cid], self.supplier[cid]
        live = [sid for sid in (1, 2) if prices[sid] is not None]
        if k is None:
            if not live:
                return None, False
            pick = min(live, key=lambda sid: (prices[sid], sid))
            return (pick, False) if prices[pick] <= v else (None, False)
        j, s = 3 - k, self.costs[cid]
        if prices[j] is not None and prices[j] + s <= v and (
                prices[k] is None or prices[j] + s < prices[k]):
            return j, True
        return (k, False) if prices[k] is not None and prices[k] <= v else (None, False)

    def _round(self, t: int) -> None:
        prices, phases = self.post_prices(t)
        sales = {1: 0, 2: 0}
        switches = 0
        for cid in self.values:
            sid, switched = self._choose(cid, prices)
            if sid is None:
                continue
            if switched:
                fee = self.switch_fee(cid)
                self.ledger.transfer(cid, SWITCH_SINK_ID, fee)
                self.switch_paid[cid] += fee
                switches += 1
            self.supplier[cid] = sid
            self.ledger.deposit(sid, position=1)          # produce the unit
            tid = len(self.trades) + 1
            trade = Trade(tid, float(t), prices[sid], 1, tid, tid, cid, sid, Side.BUY)
            self.ledger.settle(trade)
            self.trades.append(trade)
            self.phase_of[tid] = phases[sid]
            sales[sid] += 1
        self.switch_count += switches
        self.log.append(SwitchingRound(t, (prices[1], prices[2]), (phases[1], phases[2]),
                                       (sales[1], sales[2]), switches))

    def run(self) -> int:
        """Play all T rounds (consumers in id order); returns the trade count."""
        if not self._ran:
            self._ran = True
            for t in range(self.rounds):
                self._round(t)
        return len(self.trades)

    # ---- views ---------------------------------------------------------
    @property
    def switching_costs_paid(self) -> int:
        return sum(self.switch_paid.values())

    def acquisition_rounds(self, seller_id: int) -> List[int]:
        i = seller_id - 1
        return [r.t for r in self.log if r.phases[i] == ACQUIRE]

    def records(self) -> list:
        """Fills as fl-022 `TradeRecord`s, venue tag "switching"."""
        from ..analysis.metrics import TradeRecord  # lazy: analysis sits above markets
        return [TradeRecord.from_trade(t, SWITCHING) for t in self.trades]

    def __repr__(self) -> str:
        return (f"SwitchingMarket(consumers={len(self.values)}, rounds={self.rounds}, "
                f"trades={len(self.trades)}, switches={self.switch_count})")
