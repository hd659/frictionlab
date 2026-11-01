"""Phase 5 example: funding-rate arbitrage under gas shocks.

Runs the fl-034 experiment: FundingArbitrageurs against an inventory-skewed
perp maker and one-sided retail flow, with a two-regime GasProcess swept over
shock gas level x shock probability. Prints the entry / premium table, the
analytic and bisected gas thresholds, and a regime-conditioned view (entries
per calm vs per shock step, positions held through a shock vs unwound). Writes
one offline fl-026 HTML report: findings, a shock gas x probability heatmap of
mean |premium| and the arbitrageur PnL with its friction breakdown.

    python examples/phase5_funding_arb.py [--seed N] [--replicates R]
           [--workers N] [--small] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from frictionlab.agents import FundingArbitrageur  # noqa: E402  (after sys.path tweak)
from frictionlab.analysis.pnl import PnLBreakdown  # noqa: E402
from frictionlab.analysis.report import Report  # noqa: E402
from frictionlab.analysis.report_sections import MetricsSection, PnLSection, SweepSection  # noqa: E402
from frictionlab.costs.gas import CALM, SHOCK  # noqa: E402
from frictionlab.experiments import (  # noqa: E402
    ExperimentResult, FundingArbConfig, GasThreshold, find_gas_threshold, run_funding_arb,
)
from frictionlab.experiments.funding_arb import LEVEL, PROB, arb_metrics, build, default_levels  # noqa: E402

CFG = FundingArbConfig()
SMALL = dict(levels=default_levels(CFG), probs=(0.0, 0.3, 0.6), replicates=1)
FULL = dict(levels=sorted({*default_levels(CFG), 21, 41, 81}), probs=(0.0, 0.1, 0.3, 0.6), replicates=2)
FRICTION = ("gross_funding", "basis_pnl", "perp_fee", "amm_cost", "gas", "decision_cost", "net_pnl")


@dataclass(frozen=True)
class Study:
    res: ExperimentResult
    threshold: GasThreshold
    regimes: List[Dict[str, Any]]         # one row per shock level at the top shock probability
    pnl: Dict[str, Dict[str, Any]]        # replicate-0 rerun: friction metrics + lots traded


def rerun(cfg: FundingArbConfig, res: ExperimentResult, level: int, prob: float) -> Dict[str, Any]:
    """Replay replicate 0 of one cell and split entries / exposure by gas regime."""
    seed = next(r["seed"] for r in res.rows if r[LEVEL] == level and r[PROB] == prob and r["replicate"] == 0)
    out = build(cfg, {LEVEL: level, PROB: prob}, seed).run(until=cfg.until)
    arbs = [a for a in out.agents.values() if isinstance(a, FundingArbitrageur)]
    gas, pos = arbs[0].gas, [p for a in arbs for p in a.report()]
    n = gas.step_of(cfg.until)
    reg = [int(x) for x in gas.regimes(n)]
    calm, shock = reg.count(CALM), reg.count(SHOCK)
    e_calm = sum(gas.regime(p.open_time) == CALM for p in pos)
    exposed = held = 0
    for p in pos:
        end = cfg.until if p.close_time is None else p.close_time
        for k in (k for k in range(1, n) if reg[k] == SHOCK and reg[k - 1] == CALM):
            if p.open_time < k * gas.dt < end:            # open when the shock hit
                spell_end = next((j for j in range(k, n) if reg[j] == CALM), n) * gas.dt
                exposed, held = exposed + 1, held + (end >= min(spell_end, cfg.until))
                break
    return {LEVEL: level, PROB: prob, "calm_steps": calm, "shock_steps": shock,
            "entries_calm": e_calm, "entries_shock": len(pos) - e_calm,
            "per_calm_step": e_calm / calm if calm else 0.0,
            "per_shock_step": (len(pos) - e_calm) / shock if shock else 0.0,
            "exposed": exposed, "held_through_shock": held, "unwound_in_shock": exposed - held,
            "lots": sum(2 * p.qty for p in pos), "metrics": arb_metrics(cfg, out)}


def premium_monotone(res: ExperimentResult) -> bool:
    agg = res.aggregate(["mean_abs_premium"])
    for p in res.params[PROB]:
        col = [r["mean_abs_premium_mean"] for r in sorted((r for r in agg if r[PROB] == p), key=lambda r: r[LEVEL])]
        if any(b < a - 1e-12 for a, b in zip(col, col[1:])):
            return False
    return True


def run(seed: int = 34, small: bool = False, replicates: Optional[int] = None,
        workers: int = 1, cfg: FundingArbConfig = CFG) -> Study:
    g = SMALL if small else FULL
    kw = dict(replicates=replicates or g["replicates"], master_seed=seed, n_workers=workers)
    res = run_funding_arb(cfg, g["levels"], g["probs"], **kw)
    top = g["probs"][-1]
    th = find_gas_threshold(cfg, top, grid=g["levels"], **kw)
    regimes = [rerun(cfg, res, lv, top) for lv in g["levels"]]
    base = rerun(cfg, res, g["levels"][0], g["probs"][0])
    pnl = {f"arb gas={r[LEVEL]} p={r[PROB]}": r for r in (base, regimes[-1])}
    hi, pi = regimes[-1], top / (top + cfg.p_calm)
    res.findings.update({
        "gas_threshold_analytic": th.analytic_bound, "gas_threshold_bisected": th.threshold,
        "threshold_within_bound": th.within_bound, "premium_monotone_in_gas": premium_monotone(res),
        "stationary_shock_share": round(pi, 6), "entries_per_calm_step": hi["per_calm_step"],
        "entries_per_shock_step": hi["per_shock_step"], "held_through_shock": hi["held_through_shock"],
        "unwound_in_shock": hi["unwound_in_shock"]})
    return Study(res, th, regimes, pnl)


def grid_table(res: ExperimentResult) -> str:
    agg, probs = res.aggregate(["entries", "mean_abs_premium"]), res.params[PROB]
    lines = [f"{'gas':>5}" + "".join(f"  p={p:<4} entries |prem|" for p in probs)]
    for lv in res.params[LEVEL]:
        cells = {r[PROB]: r for r in agg if r[LEVEL] == lv}
        lines.append(f"{lv:>5}" + "".join(f"  {cells[p]['entries_mean']:>14.2f}"
                                           f"{cells[p]['mean_abs_premium_mean']:>7.2f}" for p in probs))
    return "\n".join(lines)


def regime_table(regimes: List[Dict[str, Any]]) -> str:
    lines = [f"{'gas':>5}{'calm':>6}{'shock':>6}{'e_calm':>8}{'e_shock':>8}{'/calm':>8}{'/shock':>8}"
             f"{'exposed':>9}{'held':>6}{'unwound':>9}"]
    for r in regimes:
        lines.append(f"{r[LEVEL]:>5}{r['calm_steps']:>6}{r['shock_steps']:>6}{r['entries_calm']:>8}"
                     f"{r['entries_shock']:>8}{r['per_calm_step']:>8.3f}{r['per_shock_step']:>8.3f}"
                     f"{r['exposed']:>9}{r['held_through_shock']:>6}{r['unwound_in_shock']:>9}")
    return "\n".join(lines)


def breakdown(r: Dict[str, Any]) -> PnLBreakdown:
    m = r["metrics"]
    return PnLBreakdown(m["gross_funding"] + m["basis_pnl"], m["perp_fee"] + m["amm_cost"] + m["gas"],
                        m["decision_cost"], r["lots"])


def report(st: Study) -> Report:
    res = st.res
    fric = {f"{name}: {k}": r["metrics"][k] for name, r in st.pnl.items() for k in FRICTION}
    secs = [MetricsSection(extra=res.findings, title="Funding arbitrage: findings"),
            SweepSection(res.rows, [LEVEL, PROB], "mean_abs_premium",
                         title="Funding arbitrage: mean |premium| over shock gas x shock probability"),
            PnLSection({k: breakdown(r) for k, r in st.pnl.items()},
                       title="Arbitrageur PnL (fees = perp taker + AMM LP/impact + gas)"),
            MetricsSection(extra=fric, title="Arbitrageur friction breakdown (replicate 0)")]
    return Report("frictionlab experiment: funding_arb", master_seed=res.master_seed,
                  grid=res.params, meta={"experiment": res.name}, sections=secs)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--small", action="store_true", help="5 levels x 3 probabilities, 1 replicate")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory for funding_arb.html (default: a fresh temp dir)")
    a = ap.parse_args(argv)
    st = run(a.seed, a.small, a.replicates, a.workers)
    f, th = st.res.findings, st.threshold
    print(f"seed={a.seed} until={CFG.until} calm_gas={CFG.calm_gas} p_calm={CFG.p_calm} "
          f"horizon={CFG.horizon} max_size={CFG.max_size} replicates={len(st.res.rows) // len(st.res.aggregate())}")
    print(grid_table(st.res))
    print(f"premium non-decreasing in shock gas at every p: {f['premium_monotone_in_gas']}")
    print(f"gas threshold: analytic per-leg bound {th.analytic_bound}, bisected collapse level "
          f"{th.threshold} at p={th.shock_probability} (entries < {th.target:g} of baseline "
          f"{th.baseline_entries:g}); stationary shock share {f['stationary_shock_share']:.3f}")
    print(f"regime view at p={th.shock_probability} (replicate 0)")
    print(regime_table(st.regimes))
    out = a.out if a.out is not None else Path(tempfile.mkdtemp(prefix="frictionlab-funding-arb-"))
    print(f"report: {report(st).write_report(out / 'funding_arb.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
