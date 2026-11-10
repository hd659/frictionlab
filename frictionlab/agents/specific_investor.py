"""Specific investor (fl-073): relationship-specific investment under hold-up.

A `SpecificInvestor` is the seller in a fl-073 `HoldupMarket`. Before trading it
picks an investment level i from a discrete schedule of (investment_cost,
buyer_value) pairs: costs strictly increase in i, buyer values never decrease.
Outside the relationship the good is worth `outside_value` o whatever i is, so
the quasi-rent of the match is Q(i) = buyer_value(i) - o (Williamson 1985).

Ex post the pair splits Q by integer Nash bargaining with seller weight w (bps),
under the fl-040 floor convention: the buyer gets floor((10000 - w) * Q / 10000)
and the seller the remainder, so the two shares sum to Q exactly in ticks.

Policy (deterministic, smallest index wins every tie):

    hold-up level  = argmax_i  seller_share(Q(i), w) - investment_cost(i)
    first best     = argmax_i  Q(i) - investment_cost(i)
    contract       iff  seller_share(S*, w) - contract_cost  >  hold-up payoff

where S* = Q(i*) - investment_cost(i*) is the net first-best surplus a contract
fixes up front. A tie on the contract test goes to no contract. Because the
buyer share is non-decreasing in Q, the hold-up level never exceeds the first
best, and with w = 10000 the two coincide (Grossman-Hart under-investment).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from ..core.types import _check_int
from .base import Action, Agent, MarketView

BPS = 10_000
Schedule = Tuple[Tuple[int, int], ...]


def check_schedule(schedule: Sequence[Sequence[int]]) -> Schedule:
    """Validate an investment schedule ((cost, value), ...); returns it as a tuple."""
    rows = tuple((_check_int("investment_cost", c, nonneg=True), _check_int("buyer_value", v))
                 for c, v in schedule)
    if not rows:
        raise ValueError("investment schedule must be non-empty")
    for (c0, v0), (c1, v1) in zip(rows, rows[1:]):
        if c1 <= c0:
            raise ValueError("investment_cost must be strictly increasing")
        if v1 < v0:
            raise ValueError("buyer_value must be non-decreasing")
    return rows


def check_weight(weight_bps: int) -> int:
    """Seller bargaining weight in bps, 0 <= w <= 10000."""
    w = _check_int("seller_weight_bps", weight_bps, nonneg=True)
    if w > BPS:
        raise ValueError(f"seller_weight_bps must be <= {BPS}")
    return w


def buyer_share(rent: int, weight_bps: int) -> int:
    """Buyer's integer share of `rent`: floor((10000 - w) * rent / 10000)."""
    return (BPS - check_weight(weight_bps)) * _check_int("rent", rent) // BPS


def seller_share(rent: int, weight_bps: int) -> int:
    """Seller's share of `rent`: the remainder after the buyer's floored share."""
    return rent - buyer_share(rent, weight_bps)


def first_best_level(schedule: Schedule, outside_value: int) -> int:
    """Smallest i maximizing the joint surplus Q(i) - investment_cost(i)."""
    vals = [v - outside_value - c for c, v in schedule]
    return vals.index(max(vals))


def holdup_level(schedule: Schedule, outside_value: int, weight_bps: int) -> int:
    """Smallest i maximizing seller_share(Q(i)) - investment_cost(i)."""
    vals = [seller_share(v - outside_value, weight_bps) - c for c, v in schedule]
    return vals.index(max(vals))


class SpecificInvestor(Agent):
    """Seller choosing a specific investment and whether to pay for a contract."""

    def __init__(self, agent_id: int, *, decision_cost: int = 0) -> None:
        super().__init__(agent_id, decision_cost)
        self.choices: List[Tuple[str, int]] = []      # (segment, level) per trade

    # ---- payoffs (ticks, net of the outside value) --------------------
    def holdup_payoff(self, schedule: Schedule, outside_value: int, weight_bps: int) -> int:
        """Seller's ex-post rent minus sunk cost at its hold-up level."""
        i = holdup_level(schedule, outside_value, weight_bps)
        c, v = schedule[i]
        return seller_share(v - outside_value, weight_bps) - c

    def contract_payoff(self, schedule: Schedule, outside_value: int, weight_bps: int) -> int:
        """Seller's share of the net first-best surplus fixed by a contract."""
        c, v = schedule[first_best_level(schedule, outside_value)]
        return seller_share(v - outside_value - c, weight_bps)

    def wants_contract(self, schedule: Schedule, outside_value: int, weight_bps: int,
                       contract_cost: int) -> bool:
        """Contract iff contract payoff - cost strictly beats hold-up (ties: no contract)."""
        cost = _check_int("contract_cost", contract_cost, nonneg=True)
        return (self.contract_payoff(schedule, outside_value, weight_bps) - cost
                > self.holdup_payoff(schedule, outside_value, weight_bps))

    def choose(self, schedule: Schedule, outside_value: int, weight_bps: int,
               contract_cost: int = 0, allow_contract: bool = False) -> Tuple[str, int]:
        """Segment ("holdup" / "contract") and investment level for one pair."""
        if allow_contract and self.wants_contract(schedule, outside_value, weight_bps,
                                                  contract_cost):
            out = ("contract", first_best_level(schedule, outside_value))
        else:
            out = ("holdup", holdup_level(schedule, outside_value, weight_bps))
        self.choices.append(out)
        return out

    def decide(self, view: MarketView) -> Sequence[Action]:
        """Bilateral venue: the market calls `choose`, no book intents."""
        return ()

    def __repr__(self) -> str:
        return f"SpecificInvestor(id={self.agent_id}, trades={len(self.choices)})"
