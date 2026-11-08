"""Priced quality certification (fl-055): pay a friction to reveal hidden quality exactly.

In the fl-055 `LemonsMarket` a seller's quality q is hidden and buyers pay the
pooled price p for any uncertified unit. A `CertificationPolicy` lets a seller
pay an integer-tick cost c(q) to reveal q; a certified unit then clears in its
own segment at the buyer value q + premium. The seller's best alternative to
certifying is the pooled sale if q <= p, else keeping the unit (worth q), so

    certify  <=>  q + premium - c(q) >= max(p, q)

with ties certifying. For a quality-independent cost c <= premium this is the
separating cutoff q >= p + c - premium (every seller above p certifies too), so
certified qualities are never below uncertified ones. With c > premium nobody
certifies: certification is worth at most the buyers' premium.

The cost is a friction stack (an int becomes a `FixedCost`) charged on a one-lot
taker `Fill` at price q and moved by `Ledger.transfer` to a certification sink
(`CERT_SINK_ID`), so cash stays conserved. A disabled policy never certifies and
never opens the sink, so a market with it reproduces the pure pooled run.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

from ..core.ledger import Ledger
from ..core.types import Side, _check_int
from .base import CostModel, Fill, Role
from .fees import FixedCost
from .stack import FrictionStack

CERT_SINK_ID = -7


class CertificationPolicy:
    """Integer-tick certification friction with a deterministic certify rule and a sink."""

    def __init__(self, cost: Union[int, CostModel, FrictionStack] = 0, *,
                 enabled: bool = True, sink_id: int = CERT_SINK_ID) -> None:
        if isinstance(cost, int) and not isinstance(cost, bool):
            cost = FixedCost(_check_int("cost", cost, nonneg=True))
        self.stack = cost if isinstance(cost, FrictionStack) else FrictionStack([cost])
        self.enabled = bool(enabled)
        self.sink_id = _check_int("sink_id", sink_id)
        self.certifications: Dict[int, int] = {}
        self.paid: Dict[int, int] = {}

    @classmethod
    def disabled(cls) -> "CertificationPolicy":
        """A policy under which no seller ever certifies."""
        return cls(0, enabled=False)

    # -- model ----------------------------------------------------------------
    def cost_for(self, agent_id: int, quality: int) -> int:
        """Integer-tick certification cost for `agent_id` holding quality `quality`."""
        q = _check_int("quality", quality, positive=True)
        return self.stack.charge(Fill(agent_id, Side.SELL, Role.TAKER, q, 1))

    def certifies(self, agent_id: int, quality: int, premium: int,
                  pooled_price: Optional[int]) -> bool:
        """True iff q + premium - c(q) >= max(pooled price, q); None means no pool."""
        if not self.enabled:
            return False
        q = _check_int("quality", quality, positive=True)
        outside = q if pooled_price is None else max(q, pooled_price)
        return q + premium - self.cost_for(agent_id, q) >= outside

    def cutoff(self, qualities: Iterable[int], premium: int,
               pooled_price: Optional[int]) -> Optional[int]:
        """Smallest quality in `qualities` that certifies at `pooled_price` (None if none)."""
        return min((q for q in qualities if self.certifies(0, q, premium, pooled_price)),
                   default=None)

    # -- settlement -----------------------------------------------------------
    def attach(self, ledger: Ledger) -> None:
        """Open the certification sink on `ledger` if enabled and absent."""
        if self.enabled and self.sink_id not in ledger:
            ledger.open_account(self.sink_id)

    def charge(self, ledger: Ledger, agent_id: int, quality: int) -> int:
        """Move c(q) from `agent_id` to the sink and record one certification."""
        if not self.enabled:
            raise ValueError("certification is disabled")
        self.attach(ledger)
        fee = self.cost_for(agent_id, quality)
        ledger.transfer(agent_id, self.sink_id, fee)
        self.certifications[agent_id] = self.certifications.get(agent_id, 0) + 1
        self.paid[agent_id] = self.paid.get(agent_id, 0) + fee
        return fee

    @property
    def count(self) -> int:
        """Total certifications sold."""
        return sum(self.certifications.values())

    @property
    def revenue(self) -> int:
        """Total ticks charged (equals the sink's cash on a fresh ledger)."""
        return sum(self.paid.values())

    def __repr__(self) -> str:
        return f"CertificationPolicy(enabled={self.enabled}, certified={self.count})"
