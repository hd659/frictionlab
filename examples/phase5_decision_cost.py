"""Phase 5 example: decision cost, or when a satisficer beats an optimizer.

Promotes the Phase 2 seed (examples/phase2_decision_cost.py) onto the fl-028
experiment base. Four `BoundedRationalAgent`s trade against a ZI-C crowd over
a cost_per_step x aspiration grid, once as satisficers and once as optimizers
on identical seeds. Per cell it prints gross surplus, deliberation cost and
net PnL per mode (analysis/pnl.py). It then bisects, per aspiration, for the
smallest integer step cost at which the satisficer's mean net PnL wins:

    satisficer wins at c  <=>  dG < dS * c,   dG = gross_opt - gross_sat,
                                              dS = steps_opt - steps_sat

and runs the fl-024 replicator at zero, near-break-even and high cost.

    python examples/phase5_decision_cost.py [--seed N] [--replicates R]
           [--max-steps K] [--workers N] [--small] [--report PATH]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from frictionlab.experiments import (  # noqa: E402  (after sys.path tweak)
    DecisionCostConfig, ExperimentResult, run_decision_cost_study,
)

FULL = dict(cfg=DecisionCostConfig(), costs=(0, 1, 2, 4, 8, 16),
            aspirations=(100.0, 500.0, 2000.0), replicates=4, generations=8, high_cost=64)
SMALL = dict(cfg=DecisionCostConfig(max_steps=20, horizon=60.0), costs=(0, 4, 16),
             aspirations=(100.0, 500.0), replicates=2, generations=4, high_cost=256)


def grid_table(res: ExperimentResult) -> str:
    cells: Dict[tuple, Dict[str, dict]] = {}
    for r in res.aggregate():
        cells.setdefault((r["aspiration"], r["cost_per_step"]), {})[r["mode"]] = r
    lines = [f"{'asp':>7}{'cost':>5}{'sat_gross':>11}{'sat_dec':>9}{'sat_net':>11}"
             f"{'opt_gross':>11}{'opt_dec':>9}{'opt_net':>11}{'winner':>12}"]
    for (asp, c), m in cells.items():
        s, o = m["satisficing"], m["optimizing"]
        win = ("satisficing" if s["net_mean"] > o["net_mean"] else
               "optimizing" if o["net_mean"] > s["net_mean"] else "tie")
        lines.append(f"{asp:>7.0f}{c:>5}{s['gross_mean']:>11.0f}{s['decision_mean']:>9.0f}"
                     f"{s['net_mean']:>11.0f}{o['gross_mean']:>11.0f}{o['decision_mean']:>9.0f}"
                     f"{o['net_mean']:>11.0f}{win:>12}")
    return "\n".join(lines)


def break_even_table(res: ExperimentResult) -> str:
    lines = [f"{'asp':>7}{'dG':>11}{'dS':>10}{'dG/dS':>8}{'c*':>6}{'closed':>8}"]
    for b in res.findings["break_even"]:
        ratio = b["gross_gain"] / b["extra_steps"] if b["extra_steps"] else float("nan")
        lines.append(f"{b['aspiration']:>7.0f}{b['gross_gain']:>11.1f}{b['extra_steps']:>10.1f}"
                     f"{ratio:>8.2f}{str(b['cost']):>6}{str(b['analytic']):>8}")
    return "\n".join(lines)


def population_table(res: ExperimentResult) -> str:
    lines = [f"{'level':>11}{'cost':>6}{'reason':>12}{'gen':>5}{'sat_share':>11}"]
    for label, p in res.findings["population"].items():
        lines.append(f"{label:>11}{p['cost']:>6}{str(p['reason']):>12}"
                     f"{str(p['generation']):>5}{p['satisficer_share']:>11.3f}")
    return "\n".join(lines)


def run(small: bool = False, seed: int = 29, replicates: Optional[int] = None,
        max_steps: Optional[int] = None, workers: int = 1) -> ExperimentResult:
    spec = SMALL if small else FULL
    cfg = spec["cfg"] if max_steps is None else replace(spec["cfg"], max_steps=max_steps)
    return run_decision_cost_study(cfg, spec["costs"], spec["aspirations"],
                                   replicates=replicates or spec["replicates"],
                                   master_seed=seed, generations=spec["generations"],
                                   high_cost=spec["high_cost"],
                                   n_workers=workers)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--small", action="store_true", help="2x3 grid, 2 replicates, 4 gens")
    ap.add_argument("--report", type=Path, default=None, help="write an HTML report here")
    a = ap.parse_args(argv)
    res = run(a.small, a.seed, a.replicates, a.max_steps, a.workers)
    f = res.findings
    print(f"seed={a.seed} max_steps={f['max_steps']} grid={res.params} "
          f"replicates={len(res.rows) // len(res.aggregate())}")
    print(grid_table(res))
    print("break-even cost_per_step vs aspiration (c* = smallest c with sat net > opt net)")
    print(break_even_table(res))
    print("replicator fixed point (satisficer share)")
    print(population_table(res))
    if a.report is not None:
        print(f"report: {res.report(metric='net').write_report(a.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
