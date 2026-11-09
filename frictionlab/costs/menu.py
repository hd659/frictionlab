"""Menu cost: an integer-tick friction on every change of a posted price (fl-070).

A `MenuCost` is the cost a seller pays each time it changes its posted price in
a fl-070 `MenuCostMarket`: reprinting menus, relabelling shelves, the managerial
time of a pricing decision (Sheshinski-Weiss 1977, Mankiw 1985). Keeping the old
price is free, so even a small per-adjustment cost opens an inaction band:

    adjust  iff  |p - p*| > band       (ties keep the price)

The cost is a friction stack (an int becomes a `FixedCost`) charged on a one-lot
taker `Fill` and moved by `Ledger.transfer` to a menu sink (`MENU_SINK_ID`), so
cash stays conserved and `revenue == adjustments * cost` for a flat cost.
"""
from __future__ import annotations

from typing import Dict, Union

from ..core.ledger import Ledger
from ..core.types import Side, _check_int
from .base import CostModel, Fill, Role
from .fees import FixedCost
from .stack import FrictionStack

MENU_SINK_ID = -13


class MenuCost:
    """Integer-tick per-adjustment price-change friction routed to a menu-sink account."""

    def __init__(self, cost: Union[int, CostModel, FrictionStack] = 0, *,
                 sink_id: int = MENU_SINK_ID) -> None:
        if isinstance(cost, int) and not isinstance(cost, bool):
            cost = FixedCost(_check_int("cost", cost, nonneg=True))
        self.stack = cost if isinstance(cost, FrictionStack) else FrictionStack([cost])
        self.sink_id = _check_int("sink_id", sink_id)
        self.charges: Dict[int, int] = {}      # seller id -> adjustments charged
        self.paid: Dict[int, int] = {}         # seller id -> ticks paid

    # -- model ----------------------------------------------------------------
    def cost_for(self, agent_id: int) -> int:
        """Integer-tick cost of one price change by `agent_id`."""
        return self.stack.charge(Fill(_check_int("agent_id", agent_id), Side.SELL,
                                      Role.TAKER, 1, 1))

    # -- settlement -----------------------------------------------------------
    def attach(self, ledger: Ledger) -> None:
        """Open the menu sink on `ledger` if absent."""
        if self.sink_id not in ledger:
            ledger.open_account(self.sink_id)

    def charge(self, ledger: Ledger, agent_id: int) -> int:
        """Move one adjustment's menu cost from `agent_id` to the sink; returns ticks."""
        self.attach(ledger)
        fee = self.cost_for(agent_id)
        ledger.transfer(agent_id, self.sink_id, fee)
        self.charges[agent_id] = self.charges.get(agent_id, 0) + 1
        self.paid[agent_id] = self.paid.get(agent_id, 0) + fee
        return fee

    def adjustments_of(self, agent_id: int) -> int:
        """Price changes charged to `agent_id`."""
        return self.charges.get(agent_id, 0)

    @property
    def adjustments(self) -> int:
        """Total price changes charged."""
        return sum(self.charges.values())

    @property
    def revenue(self) -> int:
        """Total ticks charged (equals the sink's cash on a fresh ledger)."""
        return sum(self.paid.values())

    def __repr__(self) -> str:
        return f"MenuCost(stack={self.stack!r}, adjustments={self.adjustments})"
