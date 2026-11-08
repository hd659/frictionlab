"""Agents: observe -> decide -> act on a `Venue`, paying an explicit decision cost.

Agents never mutate the book or ledger; they emit typed `Submit` / `Cancel`
intents that the `Venue` executes and settles. The cross-venue arbitrageur
adds `Swap` (AMM leg) and `ArbPair` (book leg + AMM leg) intents, executed by
`core.runner.CrossVenue`. The funding arbitrageur reuses `ArbPair` for its
delta-neutral perp + spot legs. The `Middleman` posts bid/ask quotes over
perishable inventory as an optional counterparty of the search venue. The
`InfoAcquirer` buys a costly `SignalSource` signal only when its expected edge
clears signal cost plus decision cost (Grossman-Stiglitz), else trades as noise.
The `ReputationSeller` (fl-058) carries a hidden quality and type into the
repeated `ReputationMarket`: opportunists milk at a deterministic end-game round
and any seller whitewashes only when a fresh id net of its cost pays more.
"""
from ..costs.information import SignalSource
from .base import (
    COST_SINK_ID, Action, ActionError, Agent, Cancel, MarketView, Submit, Venue,
)
from .arbitrageur import (
    ArbPair, ArbQuote, CrossVenueArbitrageur, Swap, amm_leg, cp_optimal_size, min_quote_in,
)
from .bayesian import BayesianLearner, tick_grid, trade_likelihood
from .funding_arb import BasisPosition, BasisQuote, FundingArbitrageur
from .bounded_rational import MODES, BoundedRationalAgent, Deliberation
from .fundamental import FundamentalProcess
from .informed import EdgeQuote, InformedTrader
from .info_acquirer import InfoAcquirer, InfoChoice
from .market_maker import ASQuote, AvellanedaStoikovMM
from .middleman import (
    DECAY_SINK_ID, HOLD_SINK_ID, HeldUnit, Middleman, PerishCost,
)
from .reputation_seller import ReputationSeller, switch_round
from .noise import NoiseTrader, make_noise_population, order_flow
from .zero_intelligence import ZICTrader, ZITrader, ZIUTrader, make_zi_population

__all__ = [
    "COST_SINK_ID", "Action", "ActionError", "Agent", "Cancel", "MarketView",
    "Submit", "Venue",
    "ArbPair", "ArbQuote", "CrossVenueArbitrageur", "Swap", "amm_leg", "cp_optimal_size",
    "min_quote_in",
    "BayesianLearner", "tick_grid", "trade_likelihood",
    "BasisPosition", "BasisQuote", "FundingArbitrageur",
    "MODES", "BoundedRationalAgent", "Deliberation",
    "FundamentalProcess",
    "EdgeQuote", "InformedTrader",
    "InfoAcquirer", "InfoChoice", "SignalSource",
    "ASQuote", "AvellanedaStoikovMM",
    "DECAY_SINK_ID", "HOLD_SINK_ID", "HeldUnit", "Middleman",
    "PerishCost",
    "NoiseTrader", "make_noise_population", "order_flow",
    "ReputationSeller", "switch_round",
    "ZICTrader", "ZITrader", "ZIUTrader", "make_zi_population",
]
