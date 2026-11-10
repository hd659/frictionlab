"""Contracting cost: the ex-ante cost of writing a complete contract (fl-073).

In a fl-073 `HoldupMarket` a seller makes a relationship-specific investment
whose value outside the match is fixed, so without a contract the buyer can
hold it up ex post (Williamson 1985, Grossman-Hart 1986). A contract fixes the
investment level and the price before the investment is sunk, which removes
the hold-up, but writing, negotiating and enforcing it costs real resources:
Coase's "cost of using the price mechanism". `ContractingCost` is that cost.

The cost is a friction stack (an int becomes a `FixedCost`) charged on a one-lot
taker `Fill` and moved by `Ledger.transfer` to a contract sink
(`CONTRACT_SINK_ID`), so cash stays conserved and
`revenue == contracts * cost` for a flat cost. `enabled=False` switches
contracting off entirely: the market then reproduces a pure hold-up run.
"""
from __future__ import annotations

from typing import Dict, Union

from ..core.ledger import Ledger
from ..core.types import Side, _check_int
from .base import CostModel, Fill, Role
from .fees import FixedCost
from .stack import FrictionStack

CONTRACT_SINK_ID = -15


class ContractingCost:
    """Integer-tick ex-ante contract-writing friction routed to a contract-sink account."""

    def __init__(self, cost: Union[int, CostModel, FrictionStack] = 0, *,
                 enabled: bool = True, sink_id: int = CONTRACT_SINK_ID) -> None:
        if isinstance(cost, int) and not isinstance(cost, bool):
            cost = FixedCost(_check_int("cost", cost, nonneg=True))
        self.stack = cost if isinstance(cost, FrictionStack) else FrictionStack([cost])
        self.enabled = bool(enabled)
        self.sink_id = _check_int("sink_id", sink_id)
        self.charges: Dict[int, int] = {}      # payer id -> contracts charged
        self.paid: Dict[int, int] = {}         # payer id -> ticks paid

    @classmethod
    def disabled(cls) -> "ContractingCost":
        """Contracting switched off (pure hold-up)."""
        return cls(0, enabled=False)

    # -- model ----------------------------------------------------------------
    def cost_for(self, agent_id: int) -> int:
        """Integer-tick cost of writing one contract paid by `agent_id`."""
        return self.stack.charge(Fill(_check_int("agent_id", agent_id), Side.SELL,
                                      Role.TAKER, 1, 1))

    # -- settlement -----------------------------------------------------------
    def attach(self, ledger: Ledger) -> None:
        """Open the contract sink on `ledger` if absent."""
        if self.sink_id not in ledger:
            ledger.open_account(self.sink_id)

    def charge(self, ledger: Ledger, agent_id: int) -> int:
        """Move one contract's cost from `agent_id` to the sink; returns ticks."""
        if not self.enabled:
            raise RuntimeError("contracting is disabled")
        self.attach(ledger)
        fee = self.cost_for(agent_id)
        ledger.transfer(agent_id, self.sink_id, fee)
        self.charges[agent_id] = self.charges.get(agent_id, 0) + 1
        self.paid[agent_id] = self.paid.get(agent_id, 0) + fee
        return fee

    def contracts_of(self, agent_id: int) -> int:
        """Contracts charged to `agent_id`."""
        return self.charges.get(agent_id, 0)

    @property
    def contracts(self) -> int:
        """Total contracts written."""
        return sum(self.charges.values())

    @property
    def revenue(self) -> int:
        """Total ticks charged (equals the sink's cash on a fresh ledger)."""
        return sum(self.paid.values())

    def __repr__(self) -> str:
        return (f"ContractingCost(stack={self.stack!r}, enabled={self.enabled}, "
                f"contracts={self.contracts})")
