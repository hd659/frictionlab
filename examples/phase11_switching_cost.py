"""Phase 11 switching-cost walkthrough: bargain-then-ripoff pricing and the lock-in threshold.

The fl-062 `switching_cost` experiment is run through the public CLI entry point
`frictionlab.cli.main(argv)` on `examples/configs/switching_cost_small.json`, into OUT/run/
(rows.csv, resolved_config.json, report.html). From those artifacts it prints

1. harvest markup, churn rate and welfare change (against the zero-switching-cost
   Bertrand twin) over mean switching cost x target share;
2. one seller's posted price round by round in the traced cell, with the round in
   which it switches from acquisition to harvest marked;
3. the analytic and bisected lock-in thresholds.

It then renders one fl-026 HTML report, OUT/report.html, with a SweepSection heatmap
(switching cost x target share -> harvest markup), a line chart of both sellers' price
paths in the traced cell, and a PnLSection splitting realized surplus over the sweep into
consumer gain, acquisition loss, harvest gain and the switching sink. Two runs are
byte-identical.

    python examples/phase11_switching_cost.py --small [OUT]   # OUT defaults to a temp dir
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a checkout
sys.path.insert(0, str(HERE))

from frictionlab.agents.lockin_seller import HARVEST  # noqa: E402  (after sys.path tweak)
from frictionlab.analysis.pnl import consumer_surplus, switching_pnl  # noqa: E402
from frictionlab.analysis.plots import multi_line_chart  # noqa: E402
from frictionlab.analysis.report import Report  # noqa: E402
from frictionlab.analysis.report_sections import (  # noqa: E402
    MetricsSection, PnLSection, Section, SweepSection, inline_svg, table)
from frictionlab.experiments import switching_cost as sc  # noqa: E402
from phase9_lemons import cli, mean_by, read_rows  # noqa: E402  (shared CLI/CSV helpers)

CONFIG = HERE / "configs" / "switching_cost_small.json"
MEAN, TARGET = sc.MEAN, sc.TARGET
SPLIT_KEYS = ("consumer gain", "acquisition loss", "harvest gain", "switching sink")
TRACED = 1                                     # the seller whose path is printed
# (round, seller 1 price, seller 2 price, traced phase, traced sales, switches, mark)
Step = Tuple[int, Optional[int], Optional[int], str, int, int, str]


def surplus_split(cfg: sc.Config, rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sweep totals in ticks, rebuilt per cell seed; the four parts sum to realized surplus."""
    out = dict.fromkeys(SPLIT_KEYS, 0)
    for r in rows:
        m = sc.build(cfg, {MEAN: r[MEAN], TARGET: r[TARGET]}, r["seed"]).run().market
        out["consumer gain"] += sum(g for g, _, _ in consumer_surplus(m).values())
        for p in switching_pnl(m).values():
            out["acquisition loss"] += p.acquisition
            out["harvest gain"] += p.harvest
        out["switching sink"] -= m.switching_costs_paid
    return out


def trace_mean(cfg: sc.Config, costs: Sequence[int]) -> int:
    """Largest mean cost whose markup is set by the switching cost, not the valuation cap."""
    cap = sc.buyer_floor(cfg) - cfg.marginal_cost
    return max(m for m in costs if min(sc.cost_schedule(m, cfg.spread)) < cap)


def trace(m: Any, seller: int = TRACED) -> List[Step]:
    """Round-by-round prices of both sellers, marking the traced seller's first harvest."""
    i, steps, marked = seller - 1, [], False
    for r in m.log:
        mark = ""
        if r.phases[i] == HARVEST and not marked:
            mark, marked = "acquire -> harvest", True
        steps.append((r.t, r.prices[0], r.prices[1], r.phases[i], r.sales[i], r.switches, mark))
    return steps


class PathSection(Section):
    """Line chart of both sellers' price paths plus the traced seller's round table."""
    kind = "path"

    def __init__(self, steps: Sequence[Step], title: str) -> None:
        super().__init__(title)
        self.steps = list(steps)

    def body(self) -> str:
        xs = [float(s[0]) for s in self.steps]
        chart = multi_line_chart({"seller 1": [float(s[1] or 0) for s in self.steps],
                                  "seller 2": [float(s[2] or 0) for s in self.steps]},
                                 xs, title="Posted price paths", xlabel="round", ylabel="ticks")
        hdr = ("round", "seller 1 price", "seller 2 price", f"seller {TRACED} phase",
               f"seller {TRACED} sales", "switches", "mark")
        return inline_svg(chart) + table(hdr, self.steps)


@dataclass
class Phase11:
    """Everything the example reports, for tests and for the Phase 5 summary."""
    rows: List[Dict[str, Any]]
    cfg: sc.Config
    params: Dict[str, Any]
    master_seed: int
    mean: int                                    # traced cell's mean switching cost
    target: int                                  # traced cell's target share (bps)
    steps: List[Step]
    harvest_round: int                           # traced seller's first harvest round
    threshold: sc.LockinThreshold
    split: Dict[str, int]

    def harvest_markup(self, mean: int) -> float:
        """Harvest markup at one mean switching cost, averaged over target shares."""
        return mean_by(self.rows, "harvest_markup", **{MEAN: mean})


def run(out: Path, small: bool = True) -> Phase11:
    """Run the experiment via the CLI into OUT/run and derive every printed number."""
    flags = ("--small",) if small else ()
    cli("validate", str(CONFIG), *flags)
    cli("run", str(CONFIG), *flags, "--out", str(out / "run"))
    rows = read_rows(out / "run" / "rows.csv")
    resolved = json.loads((out / "run" / "resolved_config.json").read_text(encoding="utf-8"))
    p, seed = resolved["params"], int(resolved["master_seed"])
    cfg = sc.Config(tuple(p["valuations"]), p["n_consumers"], p["horizon"], p["marginal_cost"],
                    p["spread"])
    mean, target = trace_mean(cfg, p["mean_switching_costs"]), p["threshold_target"]
    cell = next(r for r in rows if r[MEAN] == mean and r[TARGET] == target)
    steps = trace(sc.build(cfg, {MEAN: mean, TARGET: target}, cell["seed"]).run().market)
    first = next((s[0] for s in steps if s[6]), cfg.horizon)
    th = sc.find_lockin_threshold(cfg, target, p["min_markup"], master_seed=seed)
    return Phase11(rows, cfg, p, seed, mean, target, steps, first, th, surplus_split(cfg, rows))


def report(p: Phase11) -> Report:
    """fl-026 report: headline numbers, markup heatmap, both price paths, surplus split."""
    th = p.threshold
    head = {f"seller {TRACED} first harvest round (mean cost {p.mean})": p.harvest_round,
            f"bisected lock-in threshold (target {th.target} bps, markup > {th.min_markup})": th.threshold,
            "analytic lock-in threshold": th.analytic, "horizon": p.cfg.horizon,
            "max harvest markup": max(r["harvest_markup"] for r in p.rows)}
    return Report("frictionlab Phase 11: switching costs, lock-in and harvest markup",
                  master_seed=p.master_seed, grid={MEAN: p.params["mean_switching_costs"],
                                                   TARGET: p.params["target_shares"]},
                  meta={"config": CONFIG.name, "consumers": p.cfg.n_consumers,
                        "marginal cost": p.cfg.marginal_cost, "cost spread": p.cfg.spread},
                  sections=[MetricsSection(extra=head, title="Lock-in and harvest"),
                            SweepSection(p.rows, [MEAN, TARGET], "harvest_markup",
                                         title="Harvest markup: switching cost x target share"),
                            PathSection(p.steps, f"Price paths (mean cost {p.mean}, "
                                                 f"target {p.target} bps)"),
                            PnLSection(p.split, title="Surplus split over the sweep (ticks)")])


def show(p: Phase11) -> None:
    print(f"switching-cost sweep (seed {p.master_seed}, {p.cfg.n_consumers} consumers, "
          f"horizon {p.cfg.horizon}, marginal cost {p.cfg.marginal_cost}):")
    print(f"  {'cost':>4} {'target':>6} {'markup':>6} {'churn':>5} {'welfare':>7} {'cs':>6}")
    for m in p.params["mean_switching_costs"]:
        for t in p.params["target_shares"]:
            r = next(r for r in p.rows if r[MEAN] == m and r[TARGET] == t)
            print(f"  {m:>4} {t:>6} {r['harvest_markup']:>6.1f} {r['churn_rate']:>5.2f} "
                  f"{r['welfare_change_vs_bertrand']:>7} {r['cs_change_vs_bertrand']:>6}")
    print(f"seller {TRACED} (mean cost {p.mean}, target {p.target} bps):")
    for t, p1, p2, phase, sold, sw, mark in p.steps:
        print(f"  round {t}: prices {p1}/{p2} {phase} sold {sold} switches {sw} {mark}".rstrip())
    th = p.threshold
    print(f"lock-in threshold (target {th.target}, markup > {th.min_markup}): "
          f"bisected {th.threshold}, analytic {th.analytic}")
    print("surplus split: " + ", ".join(f"{k} {v}" for k, v in p.split.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", help="output directory (default: a new temp dir)")
    ap.add_argument("--small", action="store_true", help="use the experiment's small preset")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab_phase11_"))
    out.mkdir(parents=True, exist_ok=True)
    p = run(out, small=args.small)
    show(p)
    print(f"report: {report(p).write_report(out / 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
