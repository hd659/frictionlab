"""Phase 4 example: the analysis toolkit end to end (sweep -> metrics -> replicator -> report).

Part 1 sweeps taker fee x latency over a ZI-C + maker + informed + noise market
(fl-023) with fl-022 metrics: effective = realized + impact per trade, and
welfare = sum_ZI side * (value_k - V_T) - fees - decision costs (all other
accounts marked at the final fundamental V_T). Part 2 runs satisficer-vs-optimizer
replicator dynamics (fl-024) at a low and a high cost_per_step. Part 3 writes one
offline HTML report (fl-026). See docs/phase4-analysis-toolkit.md.

    python examples/phase4_analysis_toolkit.py [--out DIR] [--seed N] [--small] [--workers N]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from frictionlab.agents import (
    AvellanedaStoikovMM, BoundedRationalAgent, FundamentalProcess, InformedTrader, ZICTrader,
    make_noise_population,
)
from frictionlab.analysis import (
    EquilibriumSection, MidPath, QuoteRecorder, Report, Section, SimulatedFitness,
    StabilityDetector, SweepSection, SweepSpec, aggregate, competitive_equilibrium, decompose,
    evolve, friction_stack, mean_net_by_type, pnl_by_account, run_sweep, spread_decomposition,
    trade_records,
)
from frictionlab.core.runner import MarketRunner, RunResult
from frictionlab.core.types import Side

FULL = {"taker_bps": (0, 50, 200, 500), "latency": (0.0, 0.5, 2.0)}
SMALL = {"taker_bps": (0, 500), "latency": (0.0, 1.0)}
LABELS = ("satisficing", "optimizing")
MM_ID, INF_ID, NOISE_ID, ZI_ID = 1, 2, 3, 10


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    horizon: float = 100.0
    replicates: int = 4
    tau: float = 5.0               # realized-spread / impact horizon
    n_zi: int = 4                  # buyers = sellers
    drift: int = 1                 # latency cost, ticks per second of delay
    costs: Sequence[int] = (0, 8)  # low / high cost_per_step
    n_br: int = 8                  # bounded-rational population (Part 2)
    generations: int = 12
    eq_replicates: int = 2


Outcome = NamedTuple("Outcome", [("result", RunResult), ("fund", FundamentalProcess),
                                 ("path", QuoteRecorder)])


class Scenario:
    """Built inside the worker; `run` returns the raw output the metrics need."""

    def __init__(self, runner: MarketRunner, fund: FundamentalProcess) -> None:
        self.runner, self.fund, self.rec = runner, fund, QuoteRecorder()

    def run(self, until: float) -> Outcome:
        return Outcome(self.runner.run(until=until, stop_when=self.rec), self.fund, self.rec)


def build_market(cfg: Config, params: Dict, seed: int) -> Scenario:
    stack = friction_stack({**params, "drift": cfg.drift})
    ss = np.random.SeedSequence(seed).spawn(2 * cfg.n_zi + 2)
    fund = FundamentalProcess.random_walk(100, 0.1, 2, seed=np.random.default_rng(ss[0]))
    mm = AvellanedaStoikovMM(MM_ID, gamma=0.01, sigma=1.0, horizon=cfg.horizon, order_size=2,
                             inventory_limit=20, fair_value=100, rng=np.random.default_rng(ss[1]))
    mm.wake_rate = 4.0
    agents = [mm, InformedTrader(INF_ID, fund, frictions=stack, max_position=10, wake_rate=0.5)]
    agents += make_noise_population(4, seed=int(ss[1].generate_state(1)[0]), first_id=NOISE_ID,
                                    arrival_rate=0.5)
    for i in range(2 * cfg.n_zi):
        agents.append(ZICTrader(ZI_ID + i, Side.BUY if i < cfg.n_zi else Side.SELL,
                                np.random.default_rng(ss[2 + i]), n_units=3,
                                value_range=(90, 110), price_range=(80, 120)))
    return Scenario(MarketRunner(agents, stack, seed=seed, cash=10**7), fund)


def market_metrics(cfg: Config, out: Outcome) -> Dict[str, float]:
    res, mark = out.result, out.fund.value(out.result.time)
    recs, path = trade_records(res), MidPath(out.path)
    sd = spread_decomposition(recs, path, cfg.tau)
    resid = max((abs(m.effective - m.realized - m.impact) for m in decompose(recs, path, cfg.tau)),
                default=0)
    zi = [a for a in res.agents.values() if isinstance(a, ZICTrader)]
    value = sum(a.side.value * sum(v - mark for v in a.schedule[:min(
        abs(res.ledger[a.agent_id].position), a.n_units)]) for a in zi)
    ce = competitive_equilibrium([v for a in zi if a.side is Side.BUY for v in a.schedule],
                                 [c for a in zi if a.side is Side.SELL for c in a.schedule])
    surplus = value - res.fees - res.decision_costs
    return {"effective_spread": sd.effective, "realized_spread": sd.realized,
            "price_impact": sd.impact, "identity_residual": resid,
            "volume": sd.volume, "fees": res.fees,
            "welfare_surplus": surplus, "welfare_loss": ce.max_surplus - surplus,
            "violations": len(res.ledger.violations()),
            "zero_sum_residual": sum(pnl_by_account(res, mark).values())}


def sweep_spec(cfg: Config, grid=FULL) -> SweepSpec:
    return SweepSpec(partial(build_market, cfg), partial(market_metrics, cfg), grid=grid,
                     replicates=cfg.replicates, master_seed=cfg.seed, crn=True,
                     run_kwargs={"until": cfg.horizon})


def br_market(cfg: Config, params: Dict, seed: int) -> MarketRunner:
    """Bounded-rational mix (params["counts"]) trading against a ZI-C crowd."""
    counts, ss = params["counts"], np.random.SeedSequence(seed).spawn(cfg.n_br + 6)
    agents, aid = [], 1
    for mode in LABELS:
        for k in range(counts[mode]):
            agents.append(BoundedRationalAgent(
                aid, Side.BUY if k % 2 == 0 else Side.SELL, np.random.default_rng(ss[aid - 1]),
                mode=mode, max_steps=20, cost_per_step=params["cost_per_step"], aspiration=1.0,
                value_range=(95, 105), search_width=10, kappa=0.3, wake_rate=0.5))
            aid += 1
    for i in range(6):
        rng = np.random.default_rng(ss[cfg.n_br + i])
        agents.append(ZICTrader(100 + i, Side.BUY if i < 3 else Side.SELL, rng,
                                schedule=[int(rng.integers(80, 121))] * 200, price_range=(60, 140)))
    return MarketRunner(agents, None, seed=seed, cash=10**9)


def br_metrics(res: RunResult) -> Dict[str, float]:
    return mean_net_by_type(res, lambda a: a.valuation, key=lambda a: getattr(a, "mode", None),
                            labels=LABELS)


def replicator(cfg: Config, cost: int, n_workers: int = 1):
    fit = SimulatedFitness(partial(br_market, cfg), br_metrics, LABELS, n_agents=cfg.n_br,
                           params={"cost_per_step": cost}, master_seed=cfg.seed,
                           replicates=cfg.eq_replicates, run_kwargs={"until": 30.0},
                           n_workers=n_workers)
    return evolve([0.5, 0.5], fit, eta=0.5, generations=cfg.generations, normalize=True,
                  labels=LABELS, detector=StabilityDetector(epsilon=1e-6, window=3, floor=0.05))


def settled_share(res) -> float:
    """Satisficer share at the detected fixed point, else at the last generation."""
    return float((res.fixed_point if res.fixed_point is not None else res.shares[-1])[0])


class ReplicatorSection(Section):
    """Part 2 as one section: an fl-026 EquilibriumSection body per cost level."""
    kind = "equilibrium"

    def __init__(self, runs: Dict[int, object], title: str) -> None:
        super().__init__(title)
        self.runs = runs

    def body(self) -> str:
        return "".join(f"<h3>cost_per_step = {c}</h3>" + EquilibriumSection(r).body()
                       for c, r in self.runs.items())


def build_report(cfg: Config, grid, rows: List[Dict], runs: Dict[int, object]) -> Report:
    rep = Report("frictionlab Phase 4: analysis toolkit", master_seed=cfg.seed, grid=dict(grid),
                 meta={"replicates": cfg.replicates, "horizon": cfg.horizon, "tau": cfg.tau,
                       "cost_per_step levels": ", ".join(map(str, runs))})
    rep.add(SweepSection(rows, by=tuple(grid), metric="welfare_surplus",
                         title="Part 1: taker fee x latency sweep"))
    return rep.add(ReplicatorSection(runs, "Part 2: satisficer vs optimizer replicator"))


def summary(table: Sequence[Dict], runs: Dict[int, object]) -> str:
    lines = [f"{'bps':>5}{'lat':>6}{'n':>3}{'realized':>10}{'impact':>9}{'welfare':>10}{'loss':>9}"]
    for r in table:
        lines.append(f"{r['taker_bps']:>5}{r['latency']:>6.1f}{r['n']:>3}"
                     f"{r['realized_spread_mean']:>10.2f}{r['price_impact_mean']:>9.2f}"
                     f"{r['welfare_surplus_mean']:>10.1f}{r['welfare_loss_mean']:>9.1f}")
    for c, r in runs.items():
        lines.append(f"cost_per_step={c}: reason={r.reason} gen={r.generation} "
                     f"satisficer share={settled_share(r):.3f}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="output dir (default: a fresh temp dir)")
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--small", action="store_true", help="2x2 grid, 2 replicates, 6 generations")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    small = dict(replicates=2, generations=6) if args.small else {}
    cfg, grid = Config(seed=args.seed, **small), SMALL if args.small else FULL
    rows = run_sweep(sweep_spec(cfg, grid), n_workers=args.workers)
    runs = {c: replicator(cfg, c) for c in cfg.costs}
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab-phase4-"))
    path = build_report(cfg, grid, rows, runs).write_report(out / "phase4_report.html")
    print(f"seed={cfg.seed} grid={dict(grid)} replicates={cfg.replicates}")
    print(summary(aggregate(rows, by=tuple(grid)), runs))
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
