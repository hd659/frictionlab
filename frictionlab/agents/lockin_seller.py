"""Lock-in seller (fl-061): deterministic invest-then-harvest pricing under switching costs.

A `LockinSeller` has an integer-tick marginal cost c and plays the repeated
`SwitchingMarket` duopoly (Klemperer). Its policy depends only on its share
(bps of all consumers currently attached to it) at the start of a round:

    acquire  (share < target_bps):  p = acquisition_price      (often < c: invest)
    harvest  (share >= target_bps): p = min(p_rival + min_i s_i, min_i v_i)

where i runs over its own locked-in customers, s_i is the consumer's
switching cost and v_i its valuation. A consumer switches only when
p_rival + s_i < p, strictly, so the harvest price is the largest price that
keeps every locked-in customer from switching, and the valuation cap keeps
every one of them buying. Harvest needs `target_bps > 5000` so the two
sellers can never both harvest in the same round.

In Bertrand mode the seller ignores lock-in and posts c every round, the
frictionless repeated Bertrand benchmark. A seller with `entry_round` > 0 is
absent (posts no price) before it enters: the Klemperer entrant facing an
incumbent whose customers are already locked in. With no rival price the
harvest price is capped by the valuations alone. The market drives all choices;
`decide` returns no book intents because the venue is posted-price.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..core.types import _check_int
from .base import Action, Agent, MarketView

ACQUIRE = "acquire"
HARVEST = "harvest"
BERTRAND = "bertrand"
ABSENT = "absent"
PHASES = (ACQUIRE, HARVEST, BERTRAND, ABSENT)
BPS = 10_000


def harvest_price(rival_price: Optional[int], customers: Sequence[Tuple[int, int]]) -> int:
    """Largest switch-proof price: rival + min switching cost, capped by min valuation.

    `customers` is a non-empty sequence of (valuation, switching_cost) pairs; an
    absent rival (`None`) leaves only the valuation cap.
    """
    if not customers:
        raise ValueError("harvest needs at least one locked-in customer")
    cap = min(v for v, _ in customers)
    if rival_price is None:
        return cap
    rival = _check_int("rival_price", rival_price, positive=True)
    return min(rival + min(s for _, s in customers), cap)


class LockinSeller(Agent):
    """Duopolist that buys market share below a target and harvests lock-in above it."""

    def __init__(self, agent_id: int, marginal_cost: int, *,
                 acquisition_price: Optional[int] = None, target_bps: int = 6000,
                 bertrand: bool = False, entry_round: int = 0,
                 decision_cost: int = 0) -> None:
        super().__init__(agent_id, decision_cost)
        self.marginal_cost = _check_int("marginal_cost", marginal_cost, positive=True)
        self.acquisition_price = self.marginal_cost if acquisition_price is None else (
            _check_int("acquisition_price", acquisition_price, positive=True))
        self.target_bps = _check_int("target_bps", target_bps, positive=True)
        if not 5000 < self.target_bps <= BPS:
            raise ValueError("target_bps must be in (5000, 10000]")
        self.bertrand = bool(bertrand)
        self.entry_round = _check_int("entry_round", entry_round, nonneg=True)
        self.prices: list = []          # posted price per round
        self.phases: list = []          # phase per round

    @classmethod
    def bertrand_seller(cls, agent_id: int, marginal_cost: int) -> "LockinSeller":
        """Seller that posts marginal cost every round."""
        return cls(agent_id, marginal_cost, bertrand=True)

    # ---- policy --------------------------------------------------------
    def phase(self, share_bps: int, t: int = 0) -> str:
        """Phase for round `t` that starts with `share_bps` of all consumers attached."""
        if t < self.entry_round:
            return ABSENT
        if self.bertrand:
            return BERTRAND
        return HARVEST if _check_int("share_bps", share_bps, nonneg=True) >= self.target_bps \
            else ACQUIRE

    def post_price(self, phase: str, rival_price: Optional[int] = None,
                   customers: Sequence[Tuple[int, int]] = ()) -> Optional[int]:
        """Posted price in `phase` (None while absent); harvest uses own customers."""
        if phase == ABSENT:
            p = None
        elif phase == BERTRAND:
            p = self.marginal_cost
        elif phase == ACQUIRE:
            p = self.acquisition_price
        elif phase == HARVEST:
            p = harvest_price(rival_price, customers)
        else:
            raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
        self.prices.append(p)
        self.phases.append(phase)
        return p

    def margin(self, price: int) -> int:
        """Per-unit profit at `price` (negative while investing in share)."""
        return price - self.marginal_cost

    def decide(self, view: MarketView) -> Sequence[Action]:
        """Posted-price venue: the market calls the policy methods, no book intents."""
        return ()

    def __repr__(self) -> str:
        mode = BERTRAND if self.bertrand else f"target={self.target_bps}bps"
        return (f"LockinSeller(id={self.agent_id}, c={self.marginal_cost}, "
                f"acq={self.acquisition_price}, {mode})")
