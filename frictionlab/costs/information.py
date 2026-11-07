"""Costly information (Grossman-Stiglitz 1980): a priced, noisy signal of the fundamental.

The true state is two-point around a public prior `center`:

    state(t) = HIGH (+1) if V(t) >= center else LOW (-1)

where V is a `FundamentalProcess` (`two_point(center, jump, seed)` builds the
classic constant V = center +- jump from a seeded coin). One purchase reveals

    signal = state      with probability accuracy_bps / 10000
           = -state     otherwise

drawn as `u = integers(10000); correct = u < accuracy_bps`, exact in integers.
5000 bps is pure noise, 10000 bps is perfect. Each buyer has its own stream
seeded from (seed, agent_id), so one agent's purchases never move another
agent's draws (or any agent RNG outside the source).

The price is an integer-tick friction stack (an int becomes a `FixedCost`),
charged on a one-lot taker `Fill` at `center` and moved by `Ledger.transfer`
to an information sink (`INFO_SINK_ID`), so cash stays conserved.

Acting on signal s at the touch, the expected gross edge per lot is

    E = (2 * accuracy_bps - 10000) * jump / 10000 - spread / 2

summed as one exact rational and floored once (benefits round down, costs
round up, the fl-040 venue-favourable convention). A rational buyer pays for
the signal only if E > signal cost + decision cost.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.ledger import Ledger, LedgerError
from ..core.types import Side, _check_int
from .base import CostModel, Fill, Role
from .fees import FixedCost
from .stack import FrictionStack

INFO_SINK_ID = -6
HIGH, LOW = 1, -1
BPS_ONE = 10_000


def _bps(name: str, v: int) -> int:
    v = _check_int(name, v)
    if not 0 <= v <= BPS_ONE:
        raise ValueError(f"{name} must lie in [0, {BPS_ONE}] bps, got {v}")
    return v


def expected_edge(accuracy_bps: int, jump: int, spread: int) -> int:
    """floor((2a - 10000) * jump / 10000 - spread / 2) in ticks per lot, one rounding."""
    a = _bps("accuracy_bps", accuracy_bps)
    jump = _check_int("jump", jump, nonneg=True)
    spread = _check_int("spread", spread, nonneg=True)
    return ((2 * a - BPS_ONE) * jump - 5_000 * spread) // BPS_ONE


def breakeven_accuracy(jump: int, spread: int, hurdle: int) -> Optional[int]:
    """Smallest accuracy_bps with `expected_edge > hurdle`, or None if even 10000 fails."""
    jump = _check_int("jump", jump, positive=True)
    spread = _check_int("spread", spread, nonneg=True)
    hurdle = _check_int("hurdle", hurdle)
    # floor(x / 10000) > h  <=>  x >= 10000 * (h + 1), x = (2a - 10000) jump - 5000 spread
    need = BPS_ONE * (hurdle + 1) + 5_000 * spread + BPS_ONE * jump
    a = max(0, -(-need // (2 * jump)))
    return a if a <= BPS_ONE else None


def two_point(center: int, jump: int, seed=0):
    """Constant fundamental at center + jump or center - jump, chosen by a seeded coin."""
    from ..agents.fundamental import FundamentalProcess  # lazy: agents depend on costs
    center = _check_int("center", center, positive=True)
    jump = _check_int("jump", jump, positive=True)
    if center - jump < 1:
        raise ValueError("center - jump must stay a positive price")
    up = np.random.default_rng(seed).random() < 0.5
    return FundamentalProcess.constant(center + jump if up else center - jump)


class SignalSource:
    """Seller of noisy state signals at an integer-tick price, paid into an information sink."""

    def __init__(self, fundamental, accuracy_bps: int = 7_500, *,
                 price: Union[int, CostModel, FrictionStack] = 0,
                 center: Optional[int] = None, jump: Optional[int] = None,
                 seed: int = 0, sink_id: int = INFO_SINK_ID) -> None:
        if not hasattr(fundamental, "value"):
            raise TypeError("fundamental must expose value(t), e.g. a FundamentalProcess")
        self.fundamental = fundamental
        self.accuracy_bps = _bps("accuracy_bps", accuracy_bps)
        if isinstance(price, int) and not isinstance(price, bool):
            price = FixedCost(_check_int("price", price, nonneg=True))
        self.price = price if isinstance(price, FrictionStack) else FrictionStack([price])
        self.center = _check_int("center", fundamental.initial if center is None else center,
                                 positive=True)
        self.jump = _check_int("jump", fundamental.jump_size if jump is None else jump,
                               positive=True)
        self.seed = _check_int("seed", seed, nonneg=True)
        self.sink_id = _check_int("sink_id", sink_id)
        self._streams: Dict[int, np.random.Generator] = {}
        self.purchases: Dict[int, int] = {}
        self.paid: Dict[int, int] = {}
        self.log: List[Tuple[float, int, int, int]] = []  # (time, agent, signal, state)

    # -- model ----------------------------------------------------------------
    def state(self, t: float) -> int:
        """HIGH if V(t) >= center else LOW."""
        return HIGH if self.fundamental.value(t) >= self.center else LOW

    def cost(self, agent_id: int) -> int:
        """Integer-tick price of one signal for `agent_id` (friction-stack charge)."""
        return self.price.charge(Fill(agent_id, Side.BUY, Role.TAKER, self.center, 1))

    def edge(self, spread: int) -> int:
        """Expected gross edge per lot of trading on one signal at the given spread."""
        return expected_edge(self.accuracy_bps, self.jump, spread)

    def stream(self, agent_id: int) -> np.random.Generator:
        """The agent's private signal RNG, seeded from (seed, agent_id)."""
        g = self._streams.get(agent_id)
        if g is None:
            ss = np.random.SeedSequence([self.seed, abs(agent_id), int(agent_id < 0)])
            g = self._streams[agent_id] = np.random.default_rng(ss)
        return g

    def draw(self, agent_id: int, t: float) -> int:
        """One signal realization (no charge): the state with prob accuracy, else its flip."""
        state = self.state(t)
        correct = int(self.stream(agent_id).integers(BPS_ONE)) < self.accuracy_bps
        return state if correct else -state

    # -- settlement -----------------------------------------------------------
    def attach(self, ledger: Ledger) -> None:
        """Open the information sink on `ledger` if absent."""
        if self.sink_id not in ledger:
            ledger.open_account(self.sink_id)

    def sell(self, ledger: Ledger, agent_id: int, t: float) -> Optional[int]:
        """Charge one signal to `agent_id` and return it; None if the agent cannot pay."""
        self.attach(ledger)
        fee = self.cost(agent_id)
        try:
            ledger.transfer(agent_id, self.sink_id, fee)
        except LedgerError:
            return None
        self.purchases[agent_id] = self.purchases.get(agent_id, 0) + 1
        self.paid[agent_id] = self.paid.get(agent_id, 0) + fee
        sig = self.draw(agent_id, t)
        self.log.append((float(t), agent_id, sig, self.state(t)))
        return sig

    @property
    def revenue(self) -> int:
        """Total ticks charged for signals (equals the sink's cash on a fresh ledger)."""
        return sum(self.paid.values())

    def hit_rate(self) -> float:
        """Share of sold signals that matched the true state; 0.0 if none sold."""
        return sum(s == x for _, _, s, x in self.log) / len(self.log) if self.log else 0.0

    def __repr__(self) -> str:
        return (f"SignalSource(accuracy_bps={self.accuracy_bps}, center={self.center}, "
                f"jump={self.jump}, sold={len(self.log)})")
