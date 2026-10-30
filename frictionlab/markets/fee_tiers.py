"""Volume-tiered maker/taker fee schedules (Phase 5 fee-tier wars).

- `FeeTier(min_volume, maker_bps, taker_bps)`: rates in bps of notional, as in
  `MakerTakerFee`; `maker_bps < 0` is a rebate. Charges are integer tick-lots
  (venue-favourable ceil, applied once per fill by the `FrictionStack`).
- `FeeTierSchedule(tiers, window)`: an account's tier at step `s` is picked by
  its trailing executed volume over steps `[s - window, s)`. The current step
  is excluded, so a tier is fixed for the whole step and fill order within a
  step cannot change fees. Tiers are monotone: more volume never raises a fee.
- `FeeComponent(schedule)`: a `CostModel` named "maker_taker", so it replaces a
  flat `MakerTakerFee` in a stack one-for-one; a one-tier schedule charges
  exactly what the flat fee charges. Executed volume is read from the ledger
  journal by `sync(ledger)`; `advance(step, ledger)` syncs, then moves on.
- `settle_tiered(ledger, trades, sink_policy)`: rebates are ordinary negative
  fees booked by the ledger, so they are paid out of the venue sink and cash
  stays conserved. With `sink_policy="reject"` a rebate that would drive the
  sink below zero is not paid (that leg's fee is clamped to 0).
"""
from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.ledger import Ledger, Settlement
from ..core.types import Trade, _check_int
from ..costs.base import BOTH, CostModel, Fill, Number, Role, to_frac
from ..costs.fees import BPS

SINK_POLICIES = ("allow", "reject")


@dataclass(frozen=True)
class FeeTier:
    min_volume: int
    maker_bps: Number = 0
    taker_bps: Number = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_volume",
                           _check_int("min_volume", self.min_volume, nonneg=True))
        object.__setattr__(self, "maker_bps", to_frac(self.maker_bps, "maker_bps"))
        object.__setattr__(self, "taker_bps", to_frac(self.taker_bps, "taker_bps"))
        if self.taker_bps < 0:
            raise ValueError("taker_bps must be >= 0")

    def rate(self, role: Role) -> Fraction:
        return (self.maker_bps if role is Role.MAKER else self.taker_bps) * BPS


class FeeTierSchedule:
    def __init__(self, tiers: Sequence[FeeTier], window: int = 100) -> None:
        tiers = tuple(tiers)
        if not tiers or tiers[0].min_volume != 0:
            raise ValueError("schedule needs a first tier with min_volume == 0")
        for a, b in zip(tiers, tiers[1:]):
            if b.min_volume <= a.min_volume:
                raise ValueError("tier min_volume must be strictly increasing")
            if b.taker_bps > a.taker_bps or b.maker_bps > a.maker_bps:
                raise ValueError("fees must be non-increasing in trailing volume")
        self.tiers = tiers
        self.window = _check_int("window", window, positive=True)
        self._cuts = [t.min_volume for t in tiers]

    @classmethod
    def flat(cls, maker_bps: Number = 0, taker_bps: Number = 0,
             window: int = 1) -> "FeeTierSchedule":
        return cls([FeeTier(0, maker_bps, taker_bps)], window)

    def index_for(self, volume: int) -> int:
        return bisect_right(self._cuts, _check_int("volume", volume, nonneg=True)) - 1

    def tier_for(self, volume: int) -> FeeTier:
        return self.tiers[self.index_for(volume)]

    def __len__(self) -> int:
        return len(self.tiers)

    def __repr__(self) -> str:
        return f"FeeTierSchedule({list(self.tiers)!r}, window={self.window})"


class FeeComponent(CostModel):
    """Tier-aware maker/taker fee; tracks per-account volume by step."""

    name = "maker_taker"

    def __init__(self, schedule: FeeTierSchedule, roles=BOTH, step: int = 0) -> None:
        super().__init__(roles)
        if not isinstance(schedule, FeeTierSchedule):
            raise TypeError("schedule must be a FeeTierSchedule")
        self.schedule = schedule
        self.step = _check_int("step", step, nonneg=True)
        self._hist: Dict[int, Deque[List[int]]] = {}
        self._ledger: Optional[Ledger] = None
        self._cursor = 0

    # ---- volume tracking ----------------------------------------------
    def record(self, agent_id: int, qty: int, step: Optional[int] = None) -> None:
        step = self.step if step is None else _check_int("step", step, nonneg=True)
        qty = _check_int("qty", qty, positive=True)
        dq = self._hist.setdefault(agent_id, deque())
        if step < self.step or (dq and step < dq[-1][0]):
            raise ValueError(f"volume for {agent_id} recorded out of step order")
        if dq and dq[-1][0] == step:
            dq[-1][1] += qty
        else:
            dq.append([step, qty])

    def sync(self, ledger: Ledger) -> int:
        """Record new journal settlements at the current step; returns count."""
        if self._ledger is None:
            self._ledger = ledger
        elif self._ledger is not ledger:
            raise ValueError("FeeComponent is bound to a different ledger")
        new = ledger.journal[self._cursor:]
        for s in new:
            self.record(s.buyer_id, s.qty)
            self.record(s.seller_id, s.qty)
        self._cursor = len(ledger.journal)
        return len(new)

    def advance(self, step: int, ledger: Optional[Ledger] = None) -> None:
        if ledger is not None:
            self.sync(ledger)
        step = _check_int("step", step, nonneg=True)
        if step < self.step:
            raise ValueError(f"step must not go backwards ({step} < {self.step})")
        self.step, lo = step, step - self.schedule.window
        for dq in self._hist.values():
            while dq and dq[0][0] < lo:
                dq.popleft()

    def trailing_volume(self, agent_id: int, step: Optional[int] = None) -> int:
        step = self.step if step is None else step
        lo = step - self.schedule.window
        return sum(q for s, q in self._hist.get(agent_id, ()) if lo <= s < step)

    def tier_of(self, agent_id: int) -> FeeTier:
        return self.schedule.tier_for(self.trailing_volume(agent_id))

    # ---- CostModel ----------------------------------------------------
    def raw_cost(self, fill: Fill) -> Fraction:
        return self.tier_of(fill.agent_id).rate(fill.role) * fill.notional

    def __repr__(self) -> str:
        return f"FeeComponent({self.schedule!r}, step={self.step})"


def settle_tiered(ledger: Ledger, trades: Iterable[Trade],
                  sink_policy: str = "allow") -> Tuple[List[Settlement], int]:
    """Settle trades; returns (settlements, total rebate rejected)."""
    if sink_policy not in SINK_POLICIES:
        raise ValueError(f"sink_policy must be one of {SINK_POLICIES}")
    out, rejected = [], 0
    for t in trades:
        bf, sf = ledger.fees_for(t)
        if sink_policy == "reject" and ledger.venue.cash + bf + sf < 0:
            rejected += max(-bf, 0) + max(-sf, 0)
            bf, sf = max(bf, 0), max(sf, 0)
        out.append(ledger.settle(t, (bf, sf)))
    return out, rejected
