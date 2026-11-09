"""Two-sided platform market with network effects and cold start (fl-064).

A platform (account `PLATFORM_ID`) serves two sides, A and B (Rochet-Tirole,
Katz-Shapiro). Side A members have ids 2 .. n_a + 1, side B members follow.
Each potential member draws an integer-tick standalone value u_i once from its
side's discrete schedule on its own stream seeded by (seed, member id), so a
new member never moves another agent's draw. Side X has an integer-tick
cross-side benefit b_X >= 0 per member of the other side and a participation
fee f_X per round (negative = subsidy paid by the platform). With n_o(t-1) the
other side's membership at the end of round t - 1 and net_i = u_i + b_X *
n_o(t-1) - f_X, round t applies myopic best responses in member-id order:

    non-member joins  iff  net_i - adoption_i >= 0   (ties join)
    member leaves     iff  net_i < 0                 (rejoin pays adoption again)

Every member after the update pays f_X for round t. `seed_members` (k_a, k_b)
pre-enrols the first k ids of each side in round 0 with the adoption cost paid
by the platform. The run stops at the first repeated membership state (round 0
included) or after `rounds`, so it ends in a fixed point or a 2-cycle. With
b >= 0 the counts are monotone from empty and from full. Fees, subsidies and
adoption charges move by `Ledger.transfer` and reduce to fl-022 `TradeRecord`s
tagged "platform".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.ledger import Ledger
from ..core.types import Side, _check_int
from ..costs.adoption import AdoptionCost

PLATFORM = "platform"
PLATFORM_ID = 1
SIDES = ("A", "B")


@dataclass(frozen=True)
class PlatformRound:
    """One round: membership counts after the update, joins and leaves per side."""
    t: int
    members: Tuple[int, int]
    joins: Tuple[int, int]
    leaves: Tuple[int, int]


@dataclass(frozen=True)
class Flow:
    """One cash flow: `kind` in {"fee", "subsidy", "adoption"}, payer -> payee."""
    t: int
    kind: str
    member: int
    payer: int
    payee: int
    amount: int


def _pair(name: str, v, **kw) -> Tuple[int, int]:
    a, b = (v, v) if isinstance(v, int) else tuple(v)
    return _check_int(f"{name}_a", a, **kw), _check_int(f"{name}_b", b, **kw)


class PlatformMarket:
    """T-round two-sided platform with cross-side benefits, fees and an adoption friction."""

    def __init__(self, n_members: Sequence[int], *, standalone: Sequence[Sequence[int]],
                 benefit=(0, 0), fee=(0, 0), adoption: Union[int, AdoptionCost] = 0,
                 seed_members=(0, 0), rounds: int = 1, seed: int = 0, cash: int = 0) -> None:
        self.n = _pair("n_members", n_members, positive=True)
        self.benefit = _pair("benefit", benefit, nonneg=True)
        self.fee = _pair("fee", fee)
        k = _pair("seed_members", seed_members, nonneg=True)
        if k[0] > self.n[0] or k[1] > self.n[1]:
            raise ValueError("seed_members exceeds side size")
        scheds = [[_check_int("standalone", u) for u in s] for s in standalone]
        if len(scheds) != 2 or not all(scheds):
            raise ValueError("standalone needs one non-empty schedule per side")
        self.adoption = adoption if isinstance(adoption, AdoptionCost) else AdoptionCost(adoption)
        self.rounds = _check_int("rounds", rounds, positive=True)
        first = (2, 2 + self.n[0])
        self.side_of: Dict[int, int] = {}
        self.values: Dict[int, int] = {}
        for s in (0, 1):
            for mid in range(first[s], first[s] + self.n[s]):
                self.side_of[mid] = s
                rng = np.random.default_rng([seed, mid])
                self.values[mid] = scheds[s][int(rng.integers(len(scheds[s])))]
        self.platform_id = PLATFORM_ID
        self.ledger = Ledger()
        self.ledger.open_account(PLATFORM_ID, cash)
        for mid in self.side_of:
            self.ledger.open_account(mid, cash)
        self.adoption.attach(self.ledger)
        self.flows: List[Flow] = []
        self.log: List[PlatformRound] = []
        self.accrued: Dict[int, List[int]] = {m: [0, 0] for m in self.side_of}  # u, cross
        self.members: set = set()
        for s in (0, 1):
            for mid in range(first[s], first[s] + k[s]):
                self._adopt(0, mid, PLATFORM_ID)
        self.history: List[FrozenSet[int]] = [frozenset(self.members)]
        self._ran = False

    # ---- state ---------------------------------------------------------
    def count(self, side: int, members=None) -> int:
        """Members of `side` (0 = A, 1 = B) in `members` (default: current)."""
        pool = self.members if members is None else members
        return sum(1 for m in pool if self.side_of[m] == side)

    def net_value(self, mid: int, members=None) -> int:
        """u_i + b_X * n_other - f_X against `members` (default: current)."""
        s = self.side_of[mid]
        return self.values[mid] + self.benefit[s] * self.count(1 - s, members) - self.fee[s]

    # ---- dynamics ------------------------------------------------------
    def _move(self, t: int, kind: str, mid: int, src: int, dst: int, amount: int) -> None:
        if amount:
            self.ledger.transfer(src, dst, amount)
            self.flows.append(Flow(t, kind, mid, src, dst, amount))

    def _adopt(self, t: int, mid: int, payer: int) -> None:
        fee = self.adoption.charge(self.ledger, mid, payer)
        if fee:
            self.flows.append(Flow(t, "adoption", mid, payer, self.adoption.sink_id, fee))
        self.members.add(mid)

    def _round(self, t: int) -> None:
        prev = frozenset(self.members)
        joins, leaves = [0, 0], [0, 0]
        for mid in sorted(self.side_of):
            net, s = self.net_value(mid, prev), self.side_of[mid]
            if mid in prev and net < 0:
                self.members.discard(mid)
                leaves[s] += 1
            elif mid not in prev and self.adoption.joins(net, mid):
                self._adopt(t, mid, mid)
                joins[s] += 1
        for mid in sorted(self.members):
            s = self.side_of[mid]
            self.accrued[mid][0] += self.values[mid]
            self.accrued[mid][1] += self.benefit[s] * self.count(1 - s, prev)
            f = self.fee[s]
            if f >= 0:
                self._move(t, "fee", mid, mid, PLATFORM_ID, f)
            else:
                self._move(t, "subsidy", mid, PLATFORM_ID, mid, -f)
        self.log.append(PlatformRound(t, (self.count(0), self.count(1)),
                                      tuple(joins), tuple(leaves)))

    def run(self) -> int:
        """Play rounds 1 .. T until a membership state repeats; returns rounds played."""
        if not self._ran:
            self._ran = True
            for t in range(1, self.rounds + 1):
                self._round(t)
                state = frozenset(self.members)
                repeated = state in self.history
                self.history.append(state)
                if repeated:
                    break
        return len(self.log)

    # ---- views ---------------------------------------------------------
    def cutoff_members(self) -> FrozenSet[int]:
        """Static zero-benefit cutoff: u_i - f_X - adoption_i >= 0."""
        return frozenset(m for m in self.side_of if self.adoption.joins(
            self.values[m] - self.fee[self.side_of[m]], m))

    def records(self) -> list:
        """Cash flows as fl-022 `TradeRecord`s, venue tag "platform"."""
        from ..analysis.metrics import TradeRecord  # lazy: analysis sits above markets
        return [TradeRecord(i, float(f.t), f.amount, 1, Side.BUY, f.payer, f.payee,
                            PLATFORM, f.amount) for i, f in enumerate(self.flows, start=1)]

    def __repr__(self) -> str:
        return (f"PlatformMarket(n={self.n}, rounds={len(self.log)}, "
                f"members=({self.count(0)}, {self.count(1)}))")
