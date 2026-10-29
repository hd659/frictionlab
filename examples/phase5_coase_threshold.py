"""Phase 5 example: the Coase threshold, the friction level at which trading collapses.

A ZI-C double auction with six buyers (values 160..110) and six sellers
(costs 60..110), one unit each, is swept over a total per-trade friction
F = 2 * fixed + taker fee in integer ticks. For each level it prints the
cap max_volume(F) (largest matching of pairs with gap >= F), mean
traded volume, realized surplus as a share of the competitive-equilibrium
maximum (net of friction), and the friction collected by the venue sink.
It then locates the Coase threshold (mean volume < fraction x frictionless
volume) by grid scan plus integer bisection, and checks it against the
analytic bound max(value) - min(cost), above which no trade is rational.

    python examples/phase5_coase_threshold.py [--seed N] [--fraction F]
           [--replicates R] [--workers N] [--small] [--report PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from frictionlab.experiments import (  # noqa: E402  (after sys.path tweak)
    CoaseConfig, CoaseThreshold, ExperimentResult, default_levels, find_coase_threshold,
    run_coase,
)

FULL = CoaseConfig()
SMALL = CoaseConfig(values=(150, 140, 130, 120), costs=(70, 80, 90, 100), max_wakeups=1_500)


def table(res: ExperimentResult) -> str:
    lines = [f"{'F':>5}{'cap':>5}{'volume':>9}{'surplus':>9}{'gross':>8}{'friction':>10}{'n':>4}"]
    for r in res.aggregate():
        lines.append(f"{r['friction']:>5}{r['max_volume_mean']:>5.0f}{r['volume_mean']:>9.2f}{r['surplus_share_mean']:>9.3f}"
                     f"{r['gross_share_mean']:>8.3f}{r['friction_collected_mean']:>10.1f}"
                     f"{r['n']:>4}")
    return "\n".join(lines)


def summary(th: CoaseThreshold, fraction: float) -> str:
    lo, hi = th.bracket
    return "\n".join([
        f"frictionless volume  {th.baseline_volume:.2f}  (target < {fraction:g} x = {th.target:.2f})",
        f"grid bracket         ({lo}, {hi}]  -> bisection over {len(th.probes)} probes",
        f"Coase threshold      F* = {th.threshold} ticks",
        f"analytic bound       {th.analytic_bound} ticks (largest value - cost gap)",
        f"F* / bound           {th.threshold / th.analytic_bound:.2f}" if th.analytic_bound else ""])


def run(cfg: CoaseConfig, seed: int, replicates: int, fraction: float, workers: int,
        levels: Optional[Sequence[int]] = None):
    levels = list(levels or default_levels(cfg))
    res = run_coase(cfg, levels, replicates=replicates, master_seed=seed, n_workers=workers)
    th = find_coase_threshold(cfg, fraction, grid=levels, replicates=replicates,
                              master_seed=seed, n_workers=workers)
    res.findings.update({"coase_threshold": th.threshold, "fraction": fraction,
                         "bracket_lo": th.bracket[0], "bracket_hi": th.bracket[1]})
    return res, th


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--small", action="store_true", help="4x4 schedule, 3 replicates")
    ap.add_argument("--report", type=Path, default=None, help="write an HTML report here")
    a = ap.parse_args(argv)
    cfg = SMALL if a.small else FULL
    reps = a.replicates or (3 if a.small else 4)
    res, th = run(cfg, a.seed, reps, a.fraction, a.workers)
    print(f"seed={a.seed} buyers={list(cfg.values)} sellers={list(cfg.costs)} "
          f"replicates={reps} friction=2*fixed+taker (ticks)")
    print(table(res))
    print(summary(th, a.fraction))
    if a.report is not None:
        print(f"report: {res.report(metric='volume').write_report(a.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
