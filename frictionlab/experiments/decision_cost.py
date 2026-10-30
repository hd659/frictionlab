"""Decision cost and over-analysis: when does a satisficer beat an optimizer?

`BoundedRationalAgent`s (private valuations near 100_000 ticks) trade against
a seeded ZI-C crowd. Each cell fixes (aspiration, cost_per_step, mode); under
common random numbers replicate r has the same crowd, valuations and candidate
streams in every cell, so satisficing and optimizing runs are paired. Per
cell, over the bounded-rational accounts (analysis.pnl):

    net = gross - fees - steps * cost_per_step

Step cost is charged through the ledger's decision-cost sink and never feeds
back into prices, so gross and steps are flat in cost_per_step and net is
linear in it with slope -steps. The satisficer wins at cost c iff

    gross_opt - gross_sat < (steps_opt - steps_sat) * c

`find_break_even_cost` finds the smallest such integer c by doubling plus
integer bisection over simulated probes; `BreakEven.analytic` is the closed
form floor(dG / dS) + 1 in exact Fractions. `population` runs the fl-024
replicator on a mixed population at one cost; `run_study` packages the grid,
the break-even-vs-aspiration table and three population runs as one result.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..agents import BoundedRationalAgent, ZICTrader
from ..analysis import (
    SimulatedFitness, StabilityDetector, evolve, mean_net_by_type, pnl_breakdown, total,
)
from ..analysis.welfare import pnl_by_account
from ..core.runner import MarketRunner
from ..core.types import Side
from .base import ExperimentResult, run_experiment

MODES = ("satisficing", "optimizing")


@dataclass(frozen=True)
class Config:
    n_br: int = 4                                  # bounded-rational slots (alternating sides)
    n_zi: int = 6                                  # ZI-C buyers and sellers each
    zi_values: Tuple[int, int] = (80_000, 120_000)
    zi_prices: Tuple[int, int] = (60_000, 140_000)
    br_values: Tuple[int, int] = (95_000, 105_000)
    max_steps: int = 40
    aspiration: float = 500.0
    search_width: int = 10_000
    kappa: float = 0.005
    wake_rate: float = 0.5
    horizon: float = 100.0
    cash: int = 10**12
    pop_values: Tuple[int, int] = (104_000, 96_000)   # (buyer, seller) valuation, population mode


def build(cfg: Config, params: Dict[str, Any], seed: int) -> MarketRunner:
    """One market. params: cost_per_step, and either mode (all slots) or counts."""
    counts = params.get("counts") or {params["mode"]: cfg.n_br}
    asp, steps = params.get("aspiration", cfg.aspiration), params.get("max_steps", cfg.max_steps)
    n = sum(counts.values())
    ss = np.random.SeedSequence(seed).spawn(n + 2 * cfg.n_zi)
    agents, aid = [], 1
    for mode in MODES:
        for k in range(counts.get(mode, 0)):
            agents.append(BoundedRationalAgent(
                aid, Side.BUY if aid % 2 else Side.SELL, np.random.default_rng(ss[aid - 1]),
                mode=mode, max_steps=int(steps), cost_per_step=int(params["cost_per_step"]),
                aspiration=float(asp), value_range=cfg.br_values, search_width=cfg.search_width,
                valuation=None if "mode" in params else cfg.pop_values[(aid + 1) % 2],
                kappa=cfg.kappa, wake_rate=cfg.wake_rate))
            aid += 1
    for i, s in enumerate(ss[n:]):
        rng = np.random.default_rng(s)
        v = int(rng.integers(cfg.zi_values[0], cfg.zi_values[1] + 1))
        agents.append(ZICTrader(100 + i, Side.BUY if i < cfg.n_zi else Side.SELL, rng,
                                schedule=[v] * 1000, price_range=cfg.zi_prices))
    return MarketRunner(agents, None, seed=int(ss[0].generate_state(1)[0]), cash=cfg.cash)


def _br(res) -> List[int]:
    return [a for a, ag in res.agents.items() if isinstance(ag, BoundedRationalAgent)]


def metrics(res) -> Dict[str, Any]:
    ids = _br(res)
    t = total(pnl_breakdown(res, lambda a: a.valuation, ids).values())
    steps = sum(res.agents[a].steps_taken for a in ids)
    cost = res.agents[ids[0]].cost_per_step if ids else 0
    return {"gross": t.trading, "decision": t.decision_costs, "fees": t.fees, "net": t.net,
            "steps": steps, "volume": t.volume,
            "decision_residual": t.decision_costs - cost * steps,
            "violations": len(res.ledger.violations()),
            "zero_sum_residual": sum(pnl_by_account(res, 100_000).values())}


def run_decision_cost(cfg: Config = Config(), costs: Sequence[int] = (0, 1, 2, 4, 8),
                      aspirations: Sequence[float] = (100.0, 500.0, 2000.0), *,
                      replicates: int = 4, master_seed: int = 29,
                      n_workers: int = 1) -> ExperimentResult:
    """cost_per_step x aspiration x mode grid at cfg.max_steps (paired by CRN)."""
    return run_experiment("decision_cost", partial(build, cfg), metrics,
                          {"aspiration": list(aspirations), "cost_per_step": list(costs),
                           "mode": list(MODES)}, base={"max_steps": cfg.max_steps},
                          replicates=replicates, master_seed=master_seed, crn=True,
                          run_kwargs={"until": cfg.horizon}, n_workers=n_workers)


@dataclass(frozen=True)
class BreakEven:
    aspiration: float
    max_steps: int
    cost: Optional[int]          # smallest integer c with mean sat net > opt net, None if none
    gross_gain: Fraction         # dG = mean gross_opt - mean gross_sat
    extra_steps: Fraction        # dS = mean steps_opt - mean steps_sat
    probes: Dict[int, Tuple[Fraction, Fraction]] = field(repr=False)  # c -> (sat, opt) net

    @property
    def analytic(self) -> Optional[int]:
        """Closed form: smallest integer c > dG / dS (0 if dG < 0, None if dS == 0)."""
        if self.gross_gain < 0:
            return 0
        return None if self.extra_steps <= 0 else self.gross_gain // self.extra_steps + 1

    def satisficer_wins(self, c: int) -> bool:
        s, o = self.probes[c]
        return s > o


def find_break_even_cost(cfg: Config = Config(), aspiration: Optional[float] = None,
                         max_steps: Optional[int] = None, *, replicates: int = 4,
                         master_seed: int = 29, cap: int = 1 << 16,
                         n_workers: int = 1) -> BreakEven:
    """Doubling then integer bisection on cost_per_step; every probe is simulated
    on the same per-replicate seeds, and means are exact `Fraction`s."""
    asp = cfg.aspiration if aspiration is None else float(aspiration)
    cfg = replace(cfg, aspiration=asp, max_steps=int(max_steps or cfg.max_steps))
    probes: Dict[int, Tuple[Fraction, Fraction]] = {}
    flat: Dict[str, Tuple[Fraction, Fraction]] = {}

    def wins(c: int) -> bool:
        if c not in probes:
            rows = run_decision_cost(cfg, [c], [asp], replicates=replicates,
                                     master_seed=master_seed, n_workers=n_workers).rows
            mean = lambda m, k: Fraction(sum(r[k] for r in rows if r["mode"] == m), replicates)
            probes[c] = (mean(MODES[0], "net"), mean(MODES[1], "net"))
            for k in ("gross", "steps"):
                flat.setdefault(k, (mean(MODES[0], k), mean(MODES[1], k)))
        return probes[c][0] > probes[c][1]

    lo, hi, found = -1, 0, wins(0)
    while not found and hi < cap:
        lo, hi = hi, max(1, 2 * hi)
        found = wins(hi)
    while found and hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if wins(mid) else (mid, hi)
    (g_s, g_o), (s_s, s_o) = flat["gross"], flat["steps"]
    return BreakEven(asp, cfg.max_steps, hi if found else None, g_o - g_s, s_o - s_s,
                     dict(sorted(probes.items())))


def population(cfg: Config, cost: int, *, n_agents: int = 8, generations: int = 8,
               replicates: int = 2, master_seed: int = 29, eta: float = 0.5):
    """fl-024 replicator over (satisficing, optimizing) shares at one step cost.

    Valuations are fixed per side (`cfg.pop_values`) so fitness differences
    come from the decision rule, not from which slots a mix happens to fill.
    An archetype is extinct below one agent's share (floor = 1 / n_agents).
    """
    fit = SimulatedFitness(partial(build, cfg), _type_fitness, MODES, n_agents=n_agents,
                           params={"cost_per_step": int(cost)}, master_seed=master_seed,
                           replicates=replicates, run_kwargs={"until": cfg.horizon / 2})
    det = StabilityDetector(epsilon=1e-6, window=3, floor=1.0 / n_agents)
    return evolve([0.5, 0.5], fit, eta=eta, generations=generations, normalize=True,
                  labels=MODES, detector=det)


def _type_fitness(res) -> Dict[str, float]:
    return mean_net_by_type(res, lambda a: a.valuation, key=lambda a: getattr(a, "mode", None),
                            labels=MODES)


def settled_share(eq) -> float:
    """Satisficer share at the detected fixed point, else at the last generation."""
    return float((eq.fixed_point if eq.fixed_point is not None else eq.shares[-1])[0])


def run_study(cfg: Config = Config(), costs: Sequence[int] = (0, 1, 2, 4, 8),
              aspirations: Sequence[float] = (100.0, 500.0, 2000.0), *, replicates: int = 4,
              master_seed: int = 29, high_cost: int = 64, generations: int = 8,
              n_workers: int = 1) -> ExperimentResult:
    """Grid + break-even table (findings["break_even"]) + replicator at
    (zero, near break-even, high) cost (findings["population"])."""
    res = run_decision_cost(cfg, costs, aspirations, replicates=replicates,
                            master_seed=master_seed, n_workers=n_workers)
    table = [find_break_even_cost(cfg, a, replicates=replicates, master_seed=master_seed,
                                  n_workers=n_workers) for a in aspirations]
    near = next((b.cost for b in table if b.aspiration == cfg.aspiration), table[0].cost) or 1
    pop = {}
    for label, c in (("zero", 0), ("break_even", near), ("high", max(high_cost, near + 1))):
        eq = population(cfg, c, generations=generations, master_seed=master_seed)
        pop[label] = {"cost": c, "reason": eq.reason, "generation": eq.generation,
                      "satisficer_share": settled_share(eq)}
    res.findings.update({
        "max_steps": cfg.max_steps,
        "break_even": [{"aspiration": b.aspiration, "cost": b.cost, "analytic": b.analytic,
                        "gross_gain": float(b.gross_gain), "extra_steps": float(b.extra_steps)}
                       for b in table],
        "population": pop})
    return res
