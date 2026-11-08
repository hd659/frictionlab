"""Reputation seller (fl-058): deterministic cheat / whitewash policy in a repeated lemons market.

A `ReputationSeller` holds one hidden integer-tick quality q drawn from the fl-055
schedule and a hidden type. An honest seller always delivers q. An opportunist
may deliver the schedule floor f instead and save the production cost q - f
(reservation r(q) = q as in fl-055). It plays a deterministic end-game rule:
in round t of a T-round market, with R = T - 1 - t rounds left after this one,

    cheat  <=>  R * (p_now - p_floor) < q - f        (ties stay honest)

where p_now is the price its current public score earns and p_floor = f + premium
is the price once a cheat is detected. The left side is the remaining-horizon
reputation premium, the right side the one-shot saving. With accurate ratings
an unblemished score never prices below the fresh-id score, so the live milking
round is never earlier than `switch_round(T, q - f, p_fresh - p_floor)`:

    t*  =  max(0, T - 1 - (saving - 1) // gap)   (gap <= 0 -> 0, saving <= 0 -> never)

since R * gap < saving  <=>  R <= (saving - 1) // gap for positive integers.

Any seller whitewashes (discards its history, re-enters under a fresh public id)
iff p_fresh - whitewash_cost > p_now, strictly. The market drives both choices;
`decide` returns no book intents because the reputation venue is posted-price.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..core.types import _check_int
from .base import Action, Agent, MarketView


def switch_round(rounds: int, saving: int, gap: int) -> Optional[int]:
    """Analytic first cheating round for a constant premium gap; None if it never cheats."""
    rounds = _check_int("rounds", rounds, positive=True)
    saving, gap = _check_int("saving", saving), _check_int("gap", gap)
    if saving <= 0:
        return None
    if gap <= 0:
        return 0
    return max(0, rounds - 1 - (saving - 1) // gap)


class ReputationSeller(Agent):
    """Seller with a persistent public id, a hidden quality and a hidden type."""

    def __init__(self, agent_id: int, quality: int, *, opportunist: bool = False,
                 decision_cost: int = 0) -> None:
        super().__init__(agent_id, decision_cost)
        self.quality = _check_int("quality", quality, positive=True)
        self.opportunist = bool(opportunist)
        self.public_id: Optional[int] = None
        self.cheats = 0
        self.whitewashes = 0
        self.milking_round: Optional[int] = None   # first round it cheated

    @property
    def honest(self) -> bool:
        return not self.opportunist

    def saving(self, floor_quality: int) -> int:
        """One-shot production saving from delivering the floor instead of q."""
        return max(0, self.quality - floor_quality)

    # ---- policy --------------------------------------------------------
    def will_cheat(self, t: int, rounds: int, price_now: int, floor_price: int,
                   floor_quality: int) -> bool:
        """End-game rule: cheat iff the remaining reputation premium < the saving."""
        if self.honest:
            return False
        remaining = rounds - 1 - t
        return remaining * (price_now - floor_price) < self.saving(floor_quality)

    def will_whitewash(self, fresh_price: int, cost: Optional[int], price_now: int) -> bool:
        """Re-enter under a fresh id iff fresh price - cost strictly beats the score price."""
        return cost is not None and fresh_price - cost > price_now

    def will_sell(self, price: int, delivered: int) -> bool:
        """Participate iff the posted price covers the production cost of the delivery."""
        return price >= delivered

    def record_cheat(self, t: int) -> None:
        self.cheats += 1
        if self.milking_round is None:
            self.milking_round = t

    def decide(self, view: MarketView) -> Sequence[Action]:
        """Posted-price venue: the market calls the policy methods, no book intents."""
        return ()

    def __repr__(self) -> str:
        kind = "opportunist" if self.opportunist else "honest"
        return (f"ReputationSeller(id={self.agent_id}, q={self.quality}, {kind}, "
                f"cheats={self.cheats}, whitewashes={self.whitewashes})")
