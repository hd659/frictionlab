"""Bounded-rational agent (Simon 1955): thinking is metered per step.

On each wake-up the agent draws `max_steps` candidate limit prices from its own
seeded RNG (always the full batch, so the stream advances identically in both
modes) and then evaluates them *one step at a time* against a private,
seeded valuation v. For a candidate p on side s the expected surplus is

    E[S](p) = s * (v - p) * P(fill | p),   P(fill | p) = exp(-kappa * d)

where d = max(0, s * (ref - p)) is the distance in ticks to the opposite touch
(ref = best ask for a buyer, best bid for a seller, v if that side is empty);
a marketable candidate has d = 0 and fills for sure.

- satisficing: stop at the first candidate with E[S] >= aspiration; if none
  qualifies the whole batch was searched and the best one seen is kept.
- optimizing:  evaluate all `max_steps` candidates and keep the argmax.

The effective decision cost of the call is steps_taken * cost_per_step. The
base `Agent.step` charges `deliberation_cost(view)` *before* `decide()`, so the
deliberation is run there, cached against the view, and replayed by
`decide()`; the charge flows agent -> cost sink via `Ledger.transfer`, which
keeps the cash invariant intact. If the agent cannot pay, the plan is dropped.
Because the optimizer's candidates are a superset of the satisficer's prefix,
on the same seed steps(sat) <= steps(opt) and gross(opt) >= gross(sat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..core.types import Side, _check_int
from .base import Action, Agent, Cancel, MarketView, Submit
from .zero_intelligence import Range, RngLike, _check_range

MODES = ("satisficing", "optimizing")


@dataclass(frozen=True)
class Deliberation:
    mode: str
    steps: int                                  # candidates actually evaluated
    evaluated: Tuple[Tuple[int, float], ...]    # (price, expected surplus) in order
    price: Optional[int]                        # chosen candidate, None if no search
    gross: float                                # expected surplus of the choice
    cost: int                                   # steps * cost_per_step
    satisfied: bool                             # aspiration met (satisficer stop)

    @property
    def submits(self) -> bool:
        return self.price is not None and self.gross > 0

    @property
    def net(self) -> float:
        return self.gross - self.cost


class BoundedRationalAgent(Agent):
    def __init__(self, agent_id: int, side: Side, rng: RngLike = None, *,
                 mode: str = "satisficing", max_steps: int = 10,
                 cost_per_step: int = 1, aspiration: float = 1.0,
                 valuation: Optional[int] = None, value_range: Range = (80, 120),
                 search_width: int = 10, kappa: float = 0.5, order_size: int = 1,
                 max_position: Optional[int] = None, wake_rate: float = 1.0) -> None:
        super().__init__(agent_id, 0)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.side, self.mode = Side(side), mode
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self.max_steps = _check_int("max_steps", max_steps, positive=True)
        self.cost_per_step = _check_int("cost_per_step", cost_per_step, nonneg=True)
        if not math.isfinite(aspiration):
            raise ValueError(f"aspiration must be finite, got {aspiration}")
        self.aspiration = float(aspiration)
        if valuation is None:
            lo, hi = _check_range("value_range", value_range)
            valuation = int(self.rng.integers(lo, hi + 1))
        self.valuation = _check_int("valuation", valuation, positive=True)
        self.search_width = _check_int("search_width", search_width, nonneg=True)
        if not (math.isfinite(kappa) and kappa >= 0):
            raise ValueError(f"kappa must be finite and >= 0, got {kappa}")
        self.kappa = float(kappa)
        self.order_size = _check_int("order_size", order_size, positive=True)
        self.max_position = (None if max_position is None
                             else _check_int("max_position", max_position, positive=True))
        if not float(wake_rate) > 0:
            raise ValueError(f"wake_rate must be > 0, got {wake_rate}")
        self.wake_rate = float(wake_rate)
        self._pending: Optional[Tuple[MarketView, Deliberation]] = None
        self.history: List[Deliberation] = []
        self.steps_taken = 0

    # -- valuation model ----------------------------------------------------
    def fill_prob(self, view: MarketView, price: int) -> float:
        ref = view.best_ask if self.side is Side.BUY else view.best_bid
        ref = self.valuation if ref is None else ref
        return math.exp(-self.kappa * max(0, self.side.value * (ref - price)))

    def expected_surplus(self, view: MarketView, price: int) -> float:
        return self.side.value * (self.valuation - price) * self.fill_prob(view, price)

    def room(self, view: MarketView) -> int:
        if self.max_position is None:
            return self.order_size
        return max(0, min(self.order_size, self.max_position - self.side.value * view.position))

    def draw_candidates(self) -> List[int]:
        lo = max(1, self.valuation - self.search_width)
        hi = self.valuation + self.search_width
        return self.rng.integers(lo, hi + 1, size=self.max_steps).tolist()

    # -- deliberation ---------------------------------------------------------
    def deliberate(self, view: MarketView) -> Deliberation:
        if self.room(view) == 0:
            return Deliberation(self.mode, 0, (), None, 0.0, 0, False)
        evaluated: List[Tuple[int, float]] = []
        best: Optional[Tuple[int, float]] = None
        satisfied = False
        for price in self.draw_candidates():        # one step per candidate
            es = self.expected_surplus(view, price)
            evaluated.append((price, es))
            if best is None or es > best[1]:
                best = (price, es)
            if self.mode == "satisficing" and es >= self.aspiration:
                best, satisfied = (price, es), True
                break
        steps = len(evaluated)
        return Deliberation(self.mode, steps, tuple(evaluated), best[0], best[1],
                            steps * self.cost_per_step, satisfied)

    def deliberation_cost(self, view: MarketView) -> int:
        plan = self.deliberate(view)
        self._pending = (view, plan)
        self.decision_cost = plan.cost            # effective cost of this call
        return plan.cost

    def decide(self, view: MarketView) -> List[Action]:
        pending, self._pending = self._pending, None
        plan = pending[1] if pending is not None and pending[0] is view else self.deliberate(view)
        self.history.append(plan)
        self.steps_taken += plan.steps
        actions: List[Action] = [Cancel(oid) for oid in view.open_orders]
        if plan.submits:
            actions.append(Submit(self.side, self.room(view), plan.price))
        return actions

    @property
    def last(self) -> Optional[Deliberation]:
        return self.history[-1] if self.history else None

    def __repr__(self) -> str:
        return (f"BoundedRationalAgent(id={self.agent_id}, side={self.side.name}, "
                f"mode={self.mode}, v={self.valuation}, steps={self.steps_taken})")
