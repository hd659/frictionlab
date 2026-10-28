"""Bayesian learner (Glosten-Milgrom 1985 style) that infers V from trade signs.

The agent does not see the fundamental. It holds a discrete posterior over an
integer-tick grid of candidate values v_1 < ... < v_n and, on each decide(),
folds in every *new* trade it has not seen (tracked by `trade_id`; its own
fills carry no information to itself and are skipped). For a trade at price
p with aggressor side s, under an assumed informed share mu and noise buy
probability eta:

    P(buy  | V=v) = mu * 1[v > p] + (1 - mu) * eta
    P(sell | V=v) = mu * 1[v < p] + (1 - mu) * (1 - eta)

    posterior(v) ~ prior(v) * P(s | V=v)             (renormalised)

With mu = 0 the likelihood is flat in v, so the posterior equals the prior.
An observation with zero likelihood everywhere (only possible at mu = 1) is
contradictory and ignored rather than wiping out the belief.

Trading reuses the informed trader's friction test with V replaced by the
posterior mean E[V]: it submits a marketable order at the touch only if
gross edge side * (E[V] - touch) * q strictly exceeds taker friction
(fee + half-spread + fixed knobs) plus its own decision_cost.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.types import Side, Trade, _check_int
from ..costs.base import Fill, Role
from .base import Action, Agent, Cancel, MarketView, Submit
from .informed import EdgeQuote, _as_stack
from .noise import _prob


def tick_grid(lo: int, hi: int, step: int = 1) -> np.ndarray:
    lo, hi = _check_int("lo", lo, positive=True), _check_int("hi", hi, positive=True)
    step = _check_int("step", step, positive=True)
    if hi < lo:
        raise ValueError("hi must be >= lo")
    return np.arange(lo, hi + 1, step, dtype=np.int64)


def trade_likelihood(grid: np.ndarray, price: int, side: Side,
                     mu: float, eta: float) -> np.ndarray:
    """P(side | V=v) for every v on the grid (Glosten-Milgrom trader mix)."""
    if Side(side) is Side.BUY:
        return mu * (grid > price) + (1.0 - mu) * eta
    return mu * (grid < price) + (1.0 - mu) * (1.0 - eta)


class BayesianLearner(Agent):
    def __init__(self, agent_id: int, grid: Sequence[int], *,
                 prior: Optional[Sequence[float]] = None, mu: float = 0.3,
                 noise_buy_prob: float = 0.5, frictions=None, order_size: int = 1,
                 max_position: Optional[int] = None, wake_rate: float = 1.0,
                 decision_cost: int = 0) -> None:
        super().__init__(agent_id, decision_cost)
        g = np.asarray(grid)
        if g.ndim != 1 or g.size == 0 or not np.issubdtype(g.dtype, np.integer):
            raise ValueError("grid must be a non-empty 1-D array of integer ticks")
        if np.any(np.diff(g) <= 0):
            raise ValueError("grid must be strictly increasing")
        self.grid = g.astype(np.int64)
        p = np.full(g.size, 1.0 / g.size) if prior is None else np.asarray(prior, float)
        if p.shape != g.shape or np.any(p < 0) or not np.isfinite(p).all() or p.sum() <= 0:
            raise ValueError("prior must be finite, non-negative, same length as grid")
        self.prior = p / p.sum()
        self.posterior = self.prior.copy()
        self.mu = _prob("mu", mu)
        self.eta = _prob("noise_buy_prob", noise_buy_prob)
        self.frictions = _as_stack(frictions)
        self.order_size = _check_int("order_size", order_size, positive=True)
        self.max_position = (None if max_position is None
                             else _check_int("max_position", max_position, positive=True))
        if not float(wake_rate) > 0:
            raise ValueError(f"wake_rate must be > 0, got {wake_rate}")
        self.wake_rate = float(wake_rate)
        self.last_seen_trade = 0
        self.n_updates = 0
        self.n_ignored = 0
        self.n_trades_sent = 0
        self.history: List[Optional[EdgeQuote]] = []
        self.mean_path: List[float] = []

    @classmethod
    def gaussian(cls, agent_id: int, center: int, half_width: int, sd: float,
                 step: int = 1, **kw) -> "BayesianLearner":
        g = tick_grid(max(1, center - half_width), center + half_width, step)
        return cls(agent_id, g, prior=np.exp(-0.5 * ((g - center) / float(sd)) ** 2), **kw)

    # -- beliefs ------------------------------------------------------------
    @property
    def mean(self) -> float:
        return float(self.posterior @ self.grid)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.posterior @ (self.grid - self.mean) ** 2))

    @property
    def map_value(self) -> int:
        return int(self.grid[int(np.argmax(self.posterior))])

    def update(self, price: int, side: Side) -> bool:
        """Fold one signed trade into the posterior; False if contradictory."""
        post = self.posterior * trade_likelihood(self.grid, price, side, self.mu, self.eta)
        z = post.sum()
        if not z > 0:
            self.n_ignored += 1
            return False
        self.posterior = post / z
        self.n_updates += 1
        return True

    def observe_trades(self, trades: Sequence[Trade]) -> int:
        """Update on trades newer than the last one seen; returns count used."""
        used = 0
        for t in sorted(trades, key=lambda t: t.trade_id):
            if t.trade_id <= self.last_seen_trade:
                continue
            self.last_seen_trade = t.trade_id
            if self.agent_id in (t.buyer_id, t.seller_id):
                continue
            used += self.update(t.price, t.aggressor)
        return used

    def reset(self) -> None:
        self.posterior, self.last_seen_trade = self.prior.copy(), 0

    # -- trading ------------------------------------------------------------
    def total_friction(self, side: Side, price: int, qty: int) -> Fraction:
        fill = Fill(self.agent_id, side, Role.TAKER, price, qty)
        return Fraction(self.frictions.charge(fill) + self.decision_cost)

    def _breakdown(self, side: Side, price: int, qty: int) -> Dict[str, Fraction]:
        parts = dict(self.frictions.breakdown(Fill(self.agent_id, side, Role.TAKER, price, qty)))
        parts["decision_cost"] = Fraction(self.decision_cost)
        return parts

    def evaluate(self, view: MarketView, side: Side) -> Optional[EdgeQuote]:
        price = view.best_ask if side is Side.BUY else view.best_bid
        if price is None:
            return None
        levels = view.asks if side is Side.BUY else view.bids
        room = self.order_size if self.max_position is None else max(
            0, min(self.order_size, self.max_position - side.value * view.position))
        qty = min(room, levels[0][1] if levels else self.order_size)
        if qty == 0:
            return EdgeQuote(side, price, 0, 0, Fraction(0), {})
        # snap E[V] to 1e-9 so float noise cannot break an exact edge == friction tie
        gross = side.value * (Fraction(round(self.mean, 9)) - price) * qty
        return EdgeQuote(side, price, qty, gross, self.total_friction(side, price, qty),
                         self._breakdown(side, price, qty))

    def best_opportunity(self, view: MarketView) -> Optional[EdgeQuote]:
        cands = [q for q in (self.evaluate(view, s) for s in (Side.BUY, Side.SELL))
                 if q is not None and q.profitable]
        return max(cands, key=lambda q: q.net_edge) if cands else None

    def decide(self, view: MarketView) -> List[Action]:
        tape = view.recent_trades or ((view.last_trade,) if view.last_trade else ())
        self.observe_trades(tape)
        self.mean_path.append(self.mean)
        actions: List[Action] = [Cancel(oid) for oid in view.open_orders]
        opp = self.best_opportunity(view)
        self.history.append(opp)
        if opp is not None:
            actions.append(Submit(opp.side, opp.qty, opp.price))
            self.n_trades_sent += 1
        return actions

    def __repr__(self) -> str:
        return (f"BayesianLearner(id={self.agent_id}, n={self.grid.size}, mu={self.mu}, "
                f"mean={self.mean:.3f}, decision_cost={self.decision_cost})")
