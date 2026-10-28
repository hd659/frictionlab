"""Phase 2 example: decision cost, or when over-analysis loses to satisficing.

Four `BoundedRationalAgent`s (two buyers, two sellers, private valuations
near 100_000 ticks) trade against a seeded ZI-C crowd through the
`MarketRunner`. In each market they run once as satisficers (stop at the first
candidate price whose expected surplus beats `aspiration`) and once as
optimizers (evaluate all `max_steps` candidates), on identical seeds, so the
two runs differ only in how hard the agents think. Every deliberation step
costs `cost_per_step` ticks, charged through the ledger's decision-cost sink.

    gross  = sum of fills valued at the agent's private valuation
    net    = gross - fees - steps * cost_per_step       (analysis.pnl)

The optimizer buys extra gross surplus with extra steps. With a marginal
return of r ticks per extra step, the satisficer wins once cost_per_step > r.
A second section shows a `BayesianLearner` posterior converging on a hidden
fundamental from informed-plus-noise order flow.

    python examples/phase2_decision_cost.py [seed]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from frictionlab.agents import (
    Agent, BayesianLearner, BoundedRationalAgent, Cancel, FundamentalProcess,
    InformedTrader, MarketView, Submit, ZICTrader, make_noise_population, tick_grid,
)
from frictionlab.analysis import pnl_breakdown, total
from frictionlab.core.runner import MarketRunner, RunResult
from frictionlab.core.types import Side
from frictionlab.costs import FixedCost

MODES = ("satisficing", "optimizing")
COSTS = (0, 1, 2, 4, 8, 16)  # ticks per deliberation step
BR_IDS = (1, 2, 3, 4)


@dataclass(frozen=True)
class Config:
    seed: int = 15
    markets: int = 12            # paired markets summed per sweep point
    horizon: float = 100.0
    n_zi: int = 6                # ZI-C buyers and sellers each
    zi_values: Tuple[int, int] = (80_000, 120_000)
    zi_prices: Tuple[int, int] = (60_000, 140_000)
    br_values: Tuple[int, int] = (95_000, 105_000)
    max_steps: int = 40
    aspiration: float = 500.0
    search_width: int = 10_000
    kappa: float = 0.005
    wake_rate: float = 0.5
    cash: int = 10**12


def build(cfg: Config, mode: str, cost: int, market: int) -> MarketRunner:
    """One market; ZI crowd and BR candidate streams depend only on the seed."""
    ss = np.random.SeedSequence([cfg.seed, market]).spawn(2 * cfg.n_zi + len(BR_IDS))
    agents: List[Agent] = []
    for i, (bid, s) in enumerate(zip(BR_IDS, ss[2 * cfg.n_zi:])):
        agents.append(BoundedRationalAgent(
            bid, Side.BUY if i % 2 == 0 else Side.SELL, np.random.default_rng(s),
            mode=mode, max_steps=cfg.max_steps, cost_per_step=cost,
            aspiration=cfg.aspiration, value_range=cfg.br_values,
            search_width=cfg.search_width, kappa=cfg.kappa, wake_rate=cfg.wake_rate))
    for i, s in enumerate(ss[:2 * cfg.n_zi]):
        rng = np.random.default_rng(s)
        v = int(rng.integers(cfg.zi_values[0], cfg.zi_values[1] + 1))
        agents.append(ZICTrader(10 + i, Side.BUY if i < cfg.n_zi else Side.SELL, rng,
                                schedule=[v] * 1000, price_range=cfg.zi_prices))
    return MarketRunner(agents, None, seed=int(ss[0].generate_state(1)[0]), cash=cfg.cash)


def simulate(cfg: Config, mode: str, cost: int, market: int) -> RunResult:
    return build(cfg, mode, cost, market).run(until=cfg.horizon)


def valuation(agent) -> int:
    return agent.valuation


def run_mode(cfg: Config, mode: str, cost: int) -> Dict[str, int]:
    """Summed PnL breakdown of the bounded-rational agents over all markets."""
    parts, steps = [], 0
    for m in range(cfg.markets):
        res = simulate(cfg, mode, cost, m)
        parts += pnl_breakdown(res, valuation, BR_IDS).values()
        steps += sum(res.agents[a].steps_taken for a in BR_IDS)
    t = total(parts)
    return {"gross": t.trading, "decision": t.decision_costs, "fees": t.fees,
            "net": t.net, "steps": steps, "volume": t.volume}


def sweep(cfg: Config = Config(), costs: Sequence[int] = COSTS) -> List[Dict[str, object]]:
    return [{"cost": c, **{m: run_mode(cfg, m, c) for m in MODES}} for c in costs]


def crossover(rows: Sequence[Dict[str, object]]) -> Optional[int]:
    """Smallest swept cost at which the satisficer's net PnL beats the optimizer's."""
    for r in rows:
        if r["satisficing"]["net"] > r["optimizing"]["net"]:
            return r["cost"]
    return None


# ---- BayesianLearner: posterior convergence under informed flow -------------
class ScanningProvider(Agent):
    """Re-quotes a 2-tick market at a uniformly random centre each wake-up."""

    def __init__(self, agent_id: int, seed: int, lo: int, hi: int, size: int = 5,
                 wake_rate: float = 4.0) -> None:
        super().__init__(agent_id)
        self.rng, self.lo, self.hi = np.random.default_rng(seed), lo, hi
        self.size, self.wake_rate = size, wake_rate

    def decide(self, view: MarketView):
        c = int(self.rng.integers(self.lo + 1, self.hi))
        return [Cancel(o) for o in view.open_orders] + [
            Submit(Side.BUY, self.size, c - 1), Submit(Side.SELL, self.size, c + 1)]


def learn(value: int = 108, mu: float = 0.4, seed: int = 1, horizon: float = 150.0,
          grid: Tuple[int, int] = (80, 130)) -> Tuple[BayesianLearner, RunResult]:
    """Observer learner (prohibitive friction, never trades) watching GM flow."""
    fund = FundamentalProcess.constant(value)
    agents = [ScanningProvider(1, seed, *grid),
              InformedTrader(2, fund, wake_rate=2 * mu),
              *make_noise_population(2, seed=seed, first_id=3, arrival_rate=1 - mu),
              BayesianLearner(9, tick_grid(*grid), mu=mu, frictions=FixedCost(10**9))]
    res = MarketRunner(agents, None, seed=seed, cash=10**7).run(until=horizon)
    return res.agents[9], res


def report(rows: Sequence[Dict[str, object]]) -> str:
    lines = [f"{'cost':>5}{'sat_gross':>11}{'sat_dec':>9}{'sat_net':>10}"
             f"{'opt_gross':>11}{'opt_dec':>9}{'opt_net':>10}{'winner':>12}"]
    for r in rows:
        s, o = r["satisficing"], r["optimizing"]
        win = "optimizing" if o["net"] > s["net"] else "satisficing" if s["net"] > o["net"] else "tie"
        lines.append(f"{r['cost']:>5}{s['gross']:>11}{s['decision']:>9}{s['net']:>10}"
                     f"{o['gross']:>11}{o['decision']:>9}{o['net']:>10}{win:>12}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cfg = Config(seed=int(argv[0])) if argv else Config()
    rows = sweep(cfg)
    s, o = rows[0]["satisficing"], rows[0]["optimizing"]
    r = (o["gross"] - s["gross"]) / max(1, o["steps"] - s["steps"])
    print(f"seed={cfg.seed} markets={cfg.markets} max_steps={cfg.max_steps} "
          f"steps sat={s['steps']} opt={o['steps']} return/extra step={r:.2f}")
    print(report(rows))
    print(f"crossover cost_per_step={crossover(rows)}")
    b, _ = learn(seed=cfg.seed)
    path = b.mean_path[:: max(1, len(b.mean_path) // 6)]
    print(f"BayesianLearner V=108 mu=0.4: E[V] path {[round(m, 1) for m in path]} "
          f"-> {b.mean:.2f} (sd {b.std:.2f}, MAP {b.map_value})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
