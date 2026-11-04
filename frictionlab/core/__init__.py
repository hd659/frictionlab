"""Core engine: clock, primitives, order book, ledger and the market runner.

`frictionlab.costs` depends on this package, never the reverse; the ledger
accepts any object with `trade_fees(trade) -> (buyer_fee, seller_fee)`.

`MarketRunner`, `RunResult`, `CrossVenue` and `ArbFill` live in `core.runner`,
which builds on agents and markets. They are re-exported lazily (on first
attribute access) so importing `frictionlab.core` never pulls those packages in
and the agents -> core import edge stays acyclic.
"""
from importlib import import_module
from typing import Any

from .clock import Event, EventQueue, SimClock
from .ledger import FeeSource, InvariantViolation, Ledger, LedgerError, Settlement
from .order_book import OrderBook
from .types import Account, IdGen, Order, OrderType, Side, Trade, from_ticks, settle, to_ticks

_LAZY = {"ArbFill": "runner", "CrossVenue": "runner", "MarketRunner": "runner",
         "RunResult": "runner"}

__all__ = [
    "Account", "IdGen", "Order", "OrderType", "Side", "Trade",
    "from_ticks", "settle", "to_ticks",
    "Event", "EventQueue", "SimClock",
    "OrderBook",
    "FeeSource", "InvariantViolation", "Ledger", "LedgerError", "Settlement",
    "ArbFill", "CrossVenue", "MarketRunner", "RunResult",
]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy re-export of the runner symbols listed in `_LAZY`."""
    if name in _LAZY:
        value = getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
