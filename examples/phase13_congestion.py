"""Phase 13 congestion walkthrough: Naor over-joining, the optimal toll and queue paths.

The fl-068 `congestion` experiment is run through the public CLI entry point
`frictionlab.cli.main(argv)` on `examples/configs/congestion_small.json`, into OUT/run/
(rows.csv, resolved_config.json, report.html). From those artifacts it prints

1. join rate, mean queue length and welfare over toll x waiting cost;
2. the individual (Naor), social and revenue-maximizing thresholds at the probe waiting
   cost, with the analytic and bisected optimal tolls;
3. the round-by-round queue-length path for the zero-toll and optimal-toll cells.

It then renders one fl-026 HTML report, OUT/report.html, with a SweepSection heatmap
(toll x waiting cost -> welfare), a line chart of both queue-length paths, and a
PnLSection splitting welfare into customer reward, the waiting-cost sink and the toll,
which customers pay and the operator receives. Two runs are byte-identical.

    python examples/phase13_congestion.py --small [OUT]   # OUT defaults to a temp dir
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a checkout
sys.path.insert(0, str(HERE))

from frictionlab.analysis.pnl import operator_pnl, queue_surplus  # noqa: E402
from frictionlab.analysis.plots import multi_line_chart  # noqa: E402
from frictionlab.analysis.report import Report  # noqa: E402
from frictionlab.analysis.report_sections import (  # noqa: E402
    MetricsSection, PnLSection, Section, SweepSection, inline_svg, table)
from phase9_lemons import cli  # noqa: E402  (shared CLI helper)
from phase12_critical_mass import read_rows  # noqa: E402  (exact-int CSV reader)

cg = importlib.import_module("frictionlab.experiments.congestion")
CONFIG = HERE / "configs" / "congestion_small.json"
TOLL, WAIT = cg.TOLL, cg.WAIT
SPLIT_KEYS = ("customer reward", "waiting-cost sink", "toll paid by customers", "toll revenue")
SHOWN = 30                                       # rounds of each path printed to stdout


def config(p: Dict[str, Any]) -> cg.Config:
    return cg.Config(**{k: p[k] for k in cg.Config.__dataclass_fields__})


def market(cfg: cg.Config, toll: int, c: int, seed: int):
    return cg.build(cfg, {TOLL: toll, WAIT: c}, seed).run().market


def queue_path(cfg: cg.Config, toll: int, c: int, seed: int) -> List[int]:
    """Customers waiting behind the one in service, after each step's arrival."""
    return [max(n - 1, 0) for n in market(cfg, toll, c, seed).lengths]


def welfare_split(cfg: cg.Config, rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sweep totals in ticks. The toll appears twice with opposite signs: a transfer."""
    out = dict.fromkeys(SPLIT_KEYS, 0)
    for r in rows:
        m = market(cfg, r[TOLL], r[WAIT], r["seed"])
        for s in queue_surplus(m).values():
            out["customer reward"] += s.reward
            out["waiting-cost sink"] -= s.waiting
            out["toll paid by customers"] -= s.toll
        out["toll revenue"] += operator_pnl(m).net
    return out


def revenue_threshold(cfg: cg.Config, c: int) -> Dict[str, int]:
    """Toll (smallest on ties) maximizing exact long-run toll revenue per step, and the
    threshold it induces; join rate per step is `welfare_per_step` with R = 1, c = 0."""
    p, q, R = cfg.arrival_prob_bps, cfg.service_prob_bps, cfg.reward
    best, rate = {"threshold": cg.naor_individual_threshold(R, c, q), "toll": 0}, 0
    for t in range(1, R + 1):
        s = cg.naor_individual_threshold(R, c, q, t)
        r = t * cg.welfare_per_step(1, 0, p, q, s)
        if r > rate:
            best, rate = {"threshold": s, "toll": t}, r
    return best


@dataclass
class Phase13:
    """Everything the example reports, for tests and for the Phase 5 summary."""
    rows: List[Dict[str, Any]]
    cfg: cg.Config
    params: Dict[str, Any]
    master_seed: int
    tol: cg.CongestionToll                       # bisected vs analytic optimal toll
    individual: int                              # Naor threshold at zero toll
    revenue: Dict[str, int]                      # revenue-maximizing threshold and toll
    zero_path: List[int]
    opt_path: List[int]
    welfare_zero: int
    welfare_opt: int
    split: Dict[str, int]

    def cell(self, toll: int, c: int) -> Dict[str, Any]:
        return next(r for r in self.rows if r[TOLL] == toll and r[WAIT] == c)


def run(out: Path, small: bool = True) -> Phase13:
    """Run the experiment via the CLI into OUT/run and derive every printed number."""
    flags = ("--small",) if small else ()
    cli("validate", str(CONFIG), *flags)
    cli("run", str(CONFIG), *flags, "--out", str(out / "run"))
    rows = read_rows(out / "run" / "rows.csv")
    resolved = json.loads((out / "run" / "resolved_config.json").read_text(encoding="utf-8"))
    p, seed = resolved["params"], int(resolved["master_seed"])
    cfg, c = config(p), int(p["probe_waiting_cost"])
    tol = cg.find_congestion_toll(cfg, c, master_seed=seed)
    cell_seed = rows[0]["seed"]                  # common random numbers: one seed per grid
    zero, opt = market(cfg, 0, c, cell_seed), market(cfg, tol.analytic, c, cell_seed)
    return Phase13(rows, cfg, p, seed, tol, cg.naor_individual_threshold(
        cfg.reward, c, cfg.service_prob_bps), revenue_threshold(cfg, c),
        queue_path(cfg, 0, c, cell_seed), queue_path(cfg, tol.analytic, c, cell_seed),
        cg.welfare(zero), cg.welfare(opt), welfare_split(cfg, rows))


class PathSection(Section):
    """Line chart and table of the zero-toll and optimal-toll queue-length paths."""
    kind = "path"

    def __init__(self, zero: Sequence[int], opt: Sequence[int], toll: int, title: str) -> None:
        super().__init__(title)
        self.zero, self.opt, self.toll = list(zero), list(opt), toll

    def body(self) -> str:
        series = {"toll 0": [float(x) for x in self.zero],
                  f"toll {self.toll}": [float(x) for x in self.opt]}
        chart = multi_line_chart(series, [float(i + 1) for i in range(len(self.zero))],
                                 title="Queue-length paths", xlabel="round", ylabel="queue length")
        rows = [(i + 1, a, b) for i, (a, b) in enumerate(zip(self.zero, self.opt)) if i < SHOWN]
        return inline_svg(chart) + table(("round", "toll 0", f"toll {self.toll}"), rows)


def report(p: Phase13) -> Report:
    """fl-026 report: thresholds and tolls, welfare heatmap, queue paths, welfare split."""
    c = p.tol.waiting_cost
    head = {f"individual threshold (waiting cost {c}, toll 0)": p.individual,
            "social threshold": p.tol.social_threshold,
            "revenue-maximizing threshold": p.revenue["threshold"],
            "analytic optimal toll": p.tol.analytic, "bisected optimal toll": p.tol.toll,
            "revenue-maximizing toll": p.revenue["toll"],
            "welfare at toll 0": p.welfare_zero, f"welfare at toll {p.tol.analytic}": p.welfare_opt}
    return Report("frictionlab Phase 13: congestion and queueing costs",
                  master_seed=p.master_seed, grid={TOLL: p.params["tolls"],
                                                   WAIT: p.params["waiting_costs"]},
                  meta={"config": CONFIG.name, "reward": p.cfg.reward,
                        "arrival / service (bps)": f"{p.cfg.arrival_prob_bps} / {p.cfg.service_prob_bps}",
                        "horizon": p.cfg.horizon},
                  sections=[MetricsSection(extra=head, title="Thresholds and the optimal toll"),
                            SweepSection(p.rows, [TOLL, WAIT], "welfare",
                                         title="Welfare: toll x waiting cost"),
                            PathSection(p.zero_path, p.opt_path, p.tol.analytic,
                                        f"Queue-length paths at waiting cost {c}"),
                            PnLSection(p.split, title="Welfare split over the sweep (ticks)")])


def show(p: Phase13) -> None:
    print(f"congestion sweep (seed {p.master_seed}, R {p.cfg.reward}, arrival "
          f"{p.cfg.arrival_prob_bps} bps, service {p.cfg.service_prob_bps} bps, horizon {p.cfg.horizon}):")
    print(f"  {'toll':>4} {'c':>2} {'join':>5} {'queue':>6} {'welfare':>7}")
    for t in p.params["tolls"]:
        for c in p.params["waiting_costs"]:
            r = p.cell(t, c)
            print(f"  {t:>4} {c:>2} {r['join_rate']:>5.3f} {r['mean_queue_length']:>6.3f} {r['welfare']:>7}")
    c = p.tol.waiting_cost
    print(f"thresholds at waiting cost {c}: individual {p.individual}, social "
          f"{p.tol.social_threshold}, revenue-maximizing {p.revenue['threshold']} "
          f"(toll {p.revenue['toll']})")
    print(f"optimal toll: analytic {p.tol.analytic}, bisected {p.tol.toll}")
    for t, path in ((0, p.zero_path), (p.tol.analytic, p.opt_path)):
        print(f"queue path, toll {t}: " + " ".join(map(str, path[:SHOWN])) + " ...")
    print(f"welfare at waiting cost {c}: toll 0 -> {p.welfare_zero}, "
          f"toll {p.tol.analytic} -> {p.welfare_opt}")
    print("welfare split: " + ", ".join(f"{k} {v}" for k, v in p.split.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", help="output directory (default: a new temp dir)")
    ap.add_argument("--small", action="store_true", help="use the experiment's small preset")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab_phase13_"))
    out.mkdir(parents=True, exist_ok=True)
    p = run(out, small=args.small)
    show(p)
    print(f"report: {report(p).write_report(out / 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
