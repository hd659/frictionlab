"""Phase 2 example: ZI-C allocative efficiency vs the competitive equilibrium.

Gode & Sunder (1993) showed that budget-constrained zero-intelligence (ZI-C)
traders reach near-100% allocative efficiency in a double auction: market
structure, not trader intelligence, does the work. Here we ask what friction
does to that result. A taker fee of `bps` basis points enters each trader's
budget constraint (a buyer never bids where bid + fee > value, a seller never
asks where ask - fee < cost), so marginal units whose gains from trade are
smaller than the round-trip friction drop out: a Coase threshold on one tape.

    efficiency     = realized gains from trade / CE surplus      (allocation)
    net_efficiency = (gains - fees - decision costs) / CE surplus (welfare)

The fee levels run through `frictionlab.analysis.sweep`: each level is one
sweep cell, pinned to the config seed (common random numbers), so every
level sees the same schedules and the printed table is unchanged.

    python examples/phase2_zi_efficiency.py [seed] [n_workers]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from frictionlab.agents import ZICTrader
from frictionlab.agents.base import Cancel, MarketView
from frictionlab.analysis.sweep import SweepSpec, friction_stack, run_sweep
from frictionlab.core.runner import MarketRunner, RunResult
from frictionlab.core.types import Side
from frictionlab.costs import Fill, FrictionStack, MakerTakerFee, Role

FEE_LEVELS = (0, 25, 50, 100, 200, 400)  # taker bps


@dataclass(frozen=True)
class Config:
    seed: int = 11
    n_buyers: int = 10
    n_sellers: int = 10
    n_units: int = 3
    value_range: Tuple[int, int] = (500, 1500)
    price_range: Tuple[int, int] = (1, 2000)
    decision_cost: int = 0
    max_wakeups: int = 20_000


class FeeAwareZIC(ZICTrader):
    """ZI-C whose budget constraint is net of the taker fee it could pay."""

    def __init__(self, agent_id: int, side: Side, rng, stack: FrictionStack, **kw) -> None:
        super().__init__(agent_id, side, rng, **kw)
        self.stack = stack
        self.limits = tuple(self._limit(v) for v in self.schedule)

    def fee(self, price: int) -> int:
        return self.stack.charge(Fill(self.agent_id, self.side, Role.TAKER, price, 1))

    def _limit(self, v: int) -> int:
        if self.side is Side.BUY:
            return v - self.fee(v)  # fee is monotone in price
        a = v
        while a - self.fee(a) < v:
            a += 1
        return a

    def price_bounds(self, value: int) -> Tuple[int, int]:
        lo, hi = self.price_range
        lim = self.limits[self.units_traded]
        return (lo, lim) if self.side is Side.BUY else (lim, hi)

    def decide(self, view: MarketView):
        if not self.exhausted:
            lo, hi = self.price_bounds(self.unit_value)
            if lo > hi:  # friction prices this unit out of the market
                return [Cancel(o) for o in view.open_orders]
        return super().decide(view)


def population(cfg: Config, stack: FrictionStack) -> List[FeeAwareZIC]:
    """Seeded traders; the schedules depend on the seed only, never on fees."""
    sides = [Side.BUY] * cfg.n_buyers + [Side.SELL] * cfg.n_sellers
    streams = np.random.SeedSequence(cfg.seed).spawn(len(sides))
    return [FeeAwareZIC(i + 1, s, np.random.default_rng(ss), stack,
                        n_units=cfg.n_units, value_range=cfg.value_range,
                        price_range=cfg.price_range, decision_cost=cfg.decision_cost)
            for i, (s, ss) in enumerate(zip(sides, streams))]


def equilibrium(values: Sequence[int], costs: Sequence[int]) -> Tuple[int, int, Tuple[int, int]]:
    """(CE quantity, max surplus, CE price interval) from unit schedules."""
    d, s = sorted(values, reverse=True), sorted(costs)
    q = sum(1 for v, c in zip(d, s) if v >= c)
    if q == 0:
        return 0, 0, (0, 0)
    surplus = sum(d[i] - s[i] for i in range(q))
    lo = max(s[q - 1], d[q] if q < len(d) else 0)
    hi = min(d[q - 1], s[q] if q < len(s) else d[q - 1])
    return q, surplus, (lo, hi)


def _traded(res: RunResult, a: FeeAwareZIC) -> int:
    pos = res.ledger[a.agent_id].position
    return pos if a.side is Side.BUY else a.n_units - pos


def _no_feasible_trade(runner: MarketRunner) -> bool:
    open_bids, open_asks = [], []
    for a in runner.agents.values():
        pos = runner.ledger[a.agent_id].position
        k = pos if a.side is Side.BUY else a.n_units - pos
        if k < a.n_units:
            (open_bids if a.side is Side.BUY else open_asks).append(a.limits[k])
    return not open_bids or not open_asks or max(open_bids) < min(open_asks)


def build(cfg: Config, params: Dict[str, int], seed: int) -> MarketRunner:
    """Sweep factory: one seeded ZI-C market under the cell's frictions."""
    cfg = replace(cfg, seed=seed)
    stack = friction_stack(params)
    agents = population(cfg, stack)
    endow = {a.agent_id: (0, a.n_units if a.side is Side.SELL else 0) for a in agents}
    return MarketRunner(agents, stack, seed=cfg.seed, endowments=endow)


def zi_metrics(res: RunResult) -> Dict[str, float]:
    """Sweep metrics: allocation vs the CE benchmark, net of friction."""
    agents = list(res.agents.values())
    buyers = [a for a in agents if a.side is Side.BUY]
    sellers = [a for a in agents if a.side is Side.SELL]
    q_ce, s_ce, p_ce = equilibrium([v for a in buyers for v in a.schedule],
                                   [c for a in sellers for c in a.schedule])
    gains = (sum(sum(a.schedule[:_traded(res, a)]) for a in buyers)
             - sum(sum(a.schedule[:_traded(res, a)]) for a in sellers))
    volume = sum(t.qty for t in res.trades)
    prices = [t.price for t in res.trades]
    return {"volume": volume, "q_ce": q_ce, "surplus_ce": s_ce,
            "p_ce_lo": p_ce[0], "p_ce_hi": p_ce[1], "gains": gains,
            "fees": res.fees, "decision_costs": res.decision_costs,
            "efficiency": gains / s_ce if s_ce else 0.0,
            "net_efficiency": (gains - res.fees - res.decision_costs) / s_ce if s_ce else 0.0,
            "mean_price": float(np.mean(prices)) if prices else float("nan"),
            "wakeups": res.wakeups}


def spec(cfg: Config = Config(), levels: Sequence[int] = FEE_LEVELS) -> SweepSpec:
    return SweepSpec(partial(build, cfg), zi_metrics,
                     cells=[{"taker_bps": int(b)} for b in levels], seeds=[cfg.seed],
                     run_kwargs={"max_wakeups": cfg.max_wakeups,
                                 "stop_when": _no_feasible_trade})


def sweep(cfg: Config = Config(), levels: Sequence[int] = FEE_LEVELS,
          n_workers: int = 1) -> List[Dict[str, float]]:
    out = []
    for row in run_sweep(spec(cfg, levels), n_workers=n_workers):
        if row["error"] is not None:
            raise RuntimeError(f"taker_bps={row['taker_bps']}: {row['error']}")
        drop = ("cell", "replicate", "seed", "error", "taker_bps")
        out.append({"bps": row["taker_bps"], **{k: v for k, v in row.items() if k not in drop}})
    return out


def run(cfg: Config = Config(), taker_bps: int = 0) -> Dict[str, float]:
    return sweep(cfg, (taker_bps,))[0]


def report(rows: Sequence[Dict[str, float]]) -> str:
    r0 = rows[0]
    head = (f"CE: q*={r0['q_ce']} surplus={r0['surplus_ce']} "
            f"price in [{r0['p_ce_lo']}, {r0['p_ce_hi']}]")
    lines = [head, f"{'bps':>5}{'volume':>8}{'gains':>8}{'fees':>7}"
                   f"{'eff':>8}{'net_eff':>9}{'mean_px':>9}"]
    for r in rows:
        lines.append(f"{r['bps']:>5}{r['volume']:>8}{r['gains']:>8}{r['fees']:>7}"
                     f"{r['efficiency']:>8.3f}{r['net_efficiency']:>9.3f}"
                     f"{r['mean_price']:>9.1f}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cfg = Config(seed=int(argv[0])) if argv else Config()
    rows = sweep(cfg, n_workers=int(argv[1]) if len(argv) > 1 else 1)
    print(f"seed={cfg.seed} traders={cfg.n_buyers}x{cfg.n_sellers} units={cfg.n_units}")
    print(report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
