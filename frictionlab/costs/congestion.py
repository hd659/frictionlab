"""Congestion frictions for a queueing venue: admission toll and waiting cost (fl-067).

A `CongestionCost` prices the two frictions a customer of a fl-067 `QueueMarket`
faces. The admission toll is charged once on joining and routed to a toll sink
(`TOLL_SINK_ID`, the operator's revenue); the waiting cost is charged once per
step the customer spends in the system (queue or service) and routed to a
waiting sink (`WAIT_SINK_ID`), the deadweight cost of delay. A customer who
balks pays nothing. Both are friction stacks (an int becomes a `FixedCost`)
charged on a one-lot taker `Fill`, and both settle by `Ledger.transfer`, so cash
stays conserved. The Naor (1969) join rule with W the expected steps in system:

    join  iff  reward - toll - waiting * W >= 0   (ties join)
"""
from __future__ import annotations

from typing import Dict, Union

from ..core.ledger import Ledger
from ..core.types import Side, _check_int
from .base import CostModel, Fill, Role
from .fees import FixedCost
from .stack import FrictionStack

TOLL_SINK_ID = -11
WAIT_SINK_ID = -12

Price = Union[int, CostModel, FrictionStack]


def _stack(name: str, cost: Price) -> FrictionStack:
    if isinstance(cost, int) and not isinstance(cost, bool):
        cost = FixedCost(_check_int(name, cost, nonneg=True))
    return cost if isinstance(cost, FrictionStack) else FrictionStack([cost])


def _one_lot(stack: FrictionStack, agent_id: int) -> int:
    return stack.charge(Fill(_check_int("agent_id", agent_id), Side.BUY, Role.TAKER, 1, 1))


class CongestionCost:
    """Integer-tick admission toll and per-step waiting cost, each routed to its own sink."""

    def __init__(self, toll: Price = 0, waiting: Price = 0, *,
                 toll_sink_id: int = TOLL_SINK_ID, waiting_sink_id: int = WAIT_SINK_ID) -> None:
        self.toll_stack = _stack("toll", toll)
        self.wait_stack = _stack("waiting", waiting)
        self.toll_sink_id = _check_int("toll_sink_id", toll_sink_id)
        self.wait_sink_id = _check_int("waiting_sink_id", waiting_sink_id)
        if self.toll_sink_id == self.wait_sink_id:
            raise ValueError("toll and waiting sinks must differ")
        self.tolls: Dict[int, int] = {}        # customer id -> toll ticks paid
        self.waits: Dict[int, int] = {}        # customer id -> waiting ticks paid
        self.wait_steps: Dict[int, int] = {}   # customer id -> steps charged

    # -- model ----------------------------------------------------------------
    def toll_for(self, agent_id: int) -> int:
        """Integer-tick admission toll for `agent_id`."""
        return _one_lot(self.toll_stack, agent_id)

    def wait_for(self, agent_id: int) -> int:
        """Integer-tick waiting cost per step in the system for `agent_id`."""
        return _one_lot(self.wait_stack, agent_id)

    def expected_net(self, reward: int, expected_steps: int, agent_id: int = 0) -> int:
        """reward - toll - waiting * expected_steps in ticks."""
        return (_check_int("reward", reward) - self.toll_for(agent_id)
                - self.wait_for(agent_id) * _check_int("expected_steps", expected_steps,
                                                       nonneg=True))

    def joins(self, reward: int, expected_steps: int, agent_id: int = 0) -> bool:
        """True iff the expected net value of joining is >= 0 (ties join)."""
        return self.expected_net(reward, expected_steps, agent_id) >= 0

    # -- settlement -----------------------------------------------------------
    def attach(self, ledger: Ledger) -> None:
        """Open both sinks on `ledger` if absent."""
        for sid in (self.toll_sink_id, self.wait_sink_id):
            if sid not in ledger:
                ledger.open_account(sid)

    def charge_toll(self, ledger: Ledger, agent_id: int) -> int:
        """Move the admission toll of `agent_id` to the toll sink; returns ticks."""
        self.attach(ledger)
        fee = self.toll_for(agent_id)
        ledger.transfer(agent_id, self.toll_sink_id, fee)
        self.tolls[agent_id] = self.tolls.get(agent_id, 0) + fee
        return fee

    def charge_wait(self, ledger: Ledger, agent_id: int) -> int:
        """Move one step of waiting cost of `agent_id` to the waiting sink; returns ticks."""
        self.attach(ledger)
        fee = self.wait_for(agent_id)
        ledger.transfer(agent_id, self.wait_sink_id, fee)
        self.waits[agent_id] = self.waits.get(agent_id, 0) + fee
        self.wait_steps[agent_id] = self.wait_steps.get(agent_id, 0) + 1
        return fee

    @property
    def toll_revenue(self) -> int:
        """Total tolls charged (equals the toll sink's cash on a fresh ledger)."""
        return sum(self.tolls.values())

    @property
    def waiting_paid(self) -> int:
        """Total waiting cost charged (equals the waiting sink's cash on a fresh ledger)."""
        return sum(self.waits.values())

    @property
    def customer_steps(self) -> int:
        """Total customer-steps in the system charged."""
        return sum(self.wait_steps.values())

    def __repr__(self) -> str:
        return (f"CongestionCost(toll={self.toll_stack!r}, waiting={self.wait_stack!r}, "
                f"joins={len(self.tolls)})")
