"""One-time onboarding friction for joining a platform (fl-064, cold start).

An `AdoptionCost` is the integer-tick cost a potential member pays to join a
fl-064 `PlatformMarket`: account setup, learning the interface, moving data.
It is charged once per join (rejoining after a leave pays it again) and never
while a member stays, so it opens a wedge between the join and stay rules:

    join   iff  standalone + benefit * n_other - fee - adoption >= 0
    stay   iff  standalone + benefit * n_other - fee            >= 0

The cost is a friction stack (an int becomes a `FixedCost`) charged on a one-lot
taker `Fill` and moved by `Ledger.transfer` to an adoption sink
(`ADOPTION_SINK_ID`), so cash stays conserved. The payer is the joining member
unless the platform covers it (seeded members enrolled at a subsidy).
"""
from __future__ import annotations

from typing import Dict, Optional, Union

from ..core.ledger import Ledger
from ..core.types import Side, _check_int
from .base import CostModel, Fill, Role
from .fees import FixedCost
from .stack import FrictionStack

ADOPTION_SINK_ID = -10


class AdoptionCost:
    """Integer-tick one-time adoption friction routed to an adoption-sink account."""

    def __init__(self, cost: Union[int, CostModel, FrictionStack] = 0, *,
                 sink_id: int = ADOPTION_SINK_ID) -> None:
        if isinstance(cost, int) and not isinstance(cost, bool):
            cost = FixedCost(_check_int("cost", cost, nonneg=True))
        self.stack = cost if isinstance(cost, FrictionStack) else FrictionStack([cost])
        self.sink_id = _check_int("sink_id", sink_id)
        self.charges: Dict[int, int] = {}      # member id -> joins charged
        self.paid: Dict[int, int] = {}         # payer id -> ticks paid

    # -- model ----------------------------------------------------------------
    def cost_for(self, agent_id: int) -> int:
        """Integer-tick adoption cost for `agent_id` (one join, one lot)."""
        return self.stack.charge(Fill(_check_int("agent_id", agent_id), Side.BUY,
                                      Role.TAKER, 1, 1))

    def joins(self, net_value: int, agent_id: int = 0) -> bool:
        """True iff a non-member with current-round net value `net_value` joins (ties join)."""
        return _check_int("net_value", net_value) - self.cost_for(agent_id) >= 0

    # -- settlement -----------------------------------------------------------
    def attach(self, ledger: Ledger) -> None:
        """Open the adoption sink on `ledger` if absent."""
        if self.sink_id not in ledger:
            ledger.open_account(self.sink_id)

    def charge(self, ledger: Ledger, agent_id: int, payer: Optional[int] = None) -> int:
        """Move the adoption cost of `agent_id` from `payer` (default: itself) to the sink."""
        self.attach(ledger)
        src = agent_id if payer is None else payer
        fee = self.cost_for(agent_id)
        ledger.transfer(src, self.sink_id, fee)
        self.charges[agent_id] = self.charges.get(agent_id, 0) + 1
        self.paid[src] = self.paid.get(src, 0) + fee
        return fee

    @property
    def count(self) -> int:
        """Total joins charged."""
        return sum(self.charges.values())

    @property
    def revenue(self) -> int:
        """Total ticks charged (equals the sink's cash on a fresh ledger)."""
        return sum(self.paid.values())

    def __repr__(self) -> str:
        return f"AdoptionCost(stack={self.stack!r}, charged={self.count})"
