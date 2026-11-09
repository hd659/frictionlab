"""Friction stack: composable per-fill costs in integer tick-lots.

Example
    >>> from frictionlab.costs import MakerTakerFee, HalfSpreadCost, FixedCost
    >>> stack = MakerTakerFee(maker_bps=-1, taker_bps=5) + HalfSpreadCost(2) + FixedCost(3)
    >>> buyer_fee, seller_fee = stack.trade_fees(trade)  # doctest: +SKIP

Time-varying gas: `GasProcess(calm, shock, p_shock, p_calm, seed=s).cost()` is a
`GasCost` component that charges the gas level at each fill's time.

Costly information (fl-052): `SignalSource(fundamental, accuracy_bps, price=k)` sells
noisy two-point state signals for an integer-tick price routed to `INFO_SINK_ID`.

Quality certification (fl-055): `CertificationPolicy(cost)` reveals a unit's hidden
quality for an integer-tick cost routed to `CERT_SINK_ID`.

Adoption friction (fl-064): `AdoptionCost(cost)` is a one-time integer-tick onboarding
cost for joining a `PlatformMarket`, routed to `ADOPTION_SINK_ID`.

Congestion (fl-067): `CongestionCost(toll, waiting)` charges a `QueueMarket` joiner an
admission toll (to `TOLL_SINK_ID`) and a per-step waiting cost (to `WAIT_SINK_ID`).
"""
from .base import BOTH, TAKER_ONLY, CostModel, Fill, Role, fills_of, to_frac, venue_round
from .fees import BPS, FixedCost, HalfSpreadCost, LatencyCost, MakerTakerFee
from .gas import CALM, SHOCK, GasCost, GasProcess
from .stack import FrictionStack, zero_friction
from .information import (
    BPS_ONE, HIGH, INFO_SINK_ID, LOW, SignalSource, breakeven_accuracy, expected_edge, two_point,
)
from .certification import CERT_SINK_ID, CertificationPolicy
from .adoption import ADOPTION_SINK_ID, AdoptionCost
from .congestion import TOLL_SINK_ID, WAIT_SINK_ID, CongestionCost

__all__ = [
    "BOTH", "BPS", "TAKER_ONLY",
    "CostModel", "Fill", "Role", "fills_of", "to_frac", "venue_round",
    "FixedCost", "HalfSpreadCost", "LatencyCost", "MakerTakerFee",
    "CALM", "SHOCK", "GasCost", "GasProcess",
    "FrictionStack", "zero_friction",
    "BPS_ONE", "HIGH", "INFO_SINK_ID", "LOW", "SignalSource", "breakeven_accuracy",
    "expected_edge", "two_point",
    "CERT_SINK_ID", "CertificationPolicy",
    "ADOPTION_SINK_ID", "AdoptionCost",
    "TOLL_SINK_ID", "WAIT_SINK_ID", "CongestionCost",
]
