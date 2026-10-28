"""Friction stack: composable per-fill costs in integer tick-lots.

Example
    >>> from frictionlab.costs import MakerTakerFee, HalfSpreadCost, FixedCost
    >>> stack = MakerTakerFee(maker_bps=-1, taker_bps=5) + HalfSpreadCost(2) + FixedCost(3)
    >>> buyer_fee, seller_fee = stack.trade_fees(trade)  # doctest: +SKIP
"""
from .base import BOTH, TAKER_ONLY, CostModel, Fill, Role, fills_of, to_frac, venue_round
from .fees import BPS, FixedCost, HalfSpreadCost, LatencyCost, MakerTakerFee
from .stack import FrictionStack, zero_friction

__all__ = [
    "BOTH", "BPS", "TAKER_ONLY",
    "CostModel", "Fill", "Role", "fills_of", "to_frac", "venue_round",
    "FixedCost", "HalfSpreadCost", "LatencyCost", "MakerTakerFee",
    "FrictionStack", "zero_friction",
]
