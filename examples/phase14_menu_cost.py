"""Phase 14 menu-cost walkthrough: (S,s) bands, sawtooth prices and the rigidity threshold.

The fl-071 `menu_cost` experiment is run through the public CLI entry point
`frictionlab.cli.main(argv)` on `examples/configs/menu_cost_small.json`, into OUT/run/
(rows.csv, resolved_config.json, report.html). From those artifacts it prints

1. adjustment frequency, mean absolute gap and welfare change over menu cost x inflation;
2. one seller's posted-price and target-price paths step by step, each adjustment marked
   (the costliest menu cost at the probe inflation);
3. the analytic (S,s) band, its analytic adjustment frequency and the bisected rigidity
   threshold next to the analytic one.

It then renders one fl-026 HTML report, OUT/report.html, with a SweepSection heatmap
(menu cost x inflation -> adjustment frequency), a line chart of the traced seller's
sawtooth price against its target, and a PnLSection splitting seller surplus into
frictionless profit, misalignment loss and the menu sink. Two runs are byte-identical.

    python examples/phase14_menu_cost.py --small [OUT]   # OUT defaults to a temp dir
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a checkout
sys.path.insert(0, str(HERE))

from frictionlab.analysis.pnl import menu_pnl  # noqa: E402
from frictionlab.analysis.plots import multi_line_chart  # noqa: E402
from frictionlab.analysis.report import Report  # noqa: E402
from frictionlab.analysis.report_sections import (  # noqa: E402
    MetricsSection, PnLSection, Section, SweepSection, inline_svg, table)
from phase9_lemons import cli  # noqa: E402  (shared CLI helper)
from phase12_critical_mass import read_rows  # noqa: E402  (exact-int CSV reader)

mc = importlib.import_module("frictionlab.experiments.menu_cost")
CONFIG = HERE / "configs" / "menu_cost_small.json"
COST, INFL = mc.COST, mc.INFL
SPLIT_KEYS = ("frictionless profit", "misalignment loss", "menu sink")
SHOWN = 24                                       # steps of the traced path printed / tabled
TRACED = 1                                       # seller id whose path is traced


def config(p: Dict[str, Any]) -> mc.Config:
    return mc.Config(**{k: tuple(p[k]) if k == "shocks" else p[k]
                        for k in mc.Config.__dataclass_fields__})


def cell_market(cfg: mc.Config, f: int, d: int, seed: int):
    return mc.build(cfg, {COST: f, INFL: d}, seed).run().market


def surplus_split(cfg: mc.Config, rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sweep totals in ticks; the three entries sum to total seller net PnL."""
    out = dict.fromkeys(SPLIT_KEYS, 0)
    for r in rows:
        for s in menu_pnl(cell_market(cfg, r[COST], r[INFL], r["seed"])).values():
            out["frictionless profit"] += s.frictionless
            out["misalignment loss"] -= s.misalignment
            out["menu sink"] -= s.menu_cost
    return out


@dataclass
class Phase14:
    """Everything the example reports, for tests and for the Phase 5 summary."""
    rows: List[Dict[str, Any]]
    cfg: mc.Config
    params: Dict[str, Any]
    master_seed: int
    thr: mc.RigidityThreshold
    traced_cost: int
    traced_band: int
    traced_frequency: Fraction                   # analytic frequency at the traced band
    path: List[Any]                              # MenuStep rows of the traced seller
    split: Dict[str, int]

    def cell(self, f: int, d: int) -> Dict[str, Any]:
        return next(r for r in self.rows if r[COST] == f and r[INFL] == d)

    @property
    def drift(self) -> int:
        return int(self.params["probe_inflation"])


def run(out: Path, small: bool = True) -> Phase14:
    """Run the experiment via the CLI into OUT/run and derive every printed number."""
    flags = ("--small",) if small else ()
    cli("validate", str(CONFIG), *flags)
    cli("run", str(CONFIG), *flags, "--out", str(out / "run"))
    rows = read_rows(out / "run" / "rows.csv")
    resolved = json.loads((out / "run" / "resolved_config.json").read_text(encoding="utf-8"))
    p, seed = resolved["params"], int(resolved["master_seed"])
    cfg, d = config(p), int(p["probe_inflation"])
    thr = mc.find_rigidity_threshold(cfg, d, rigidity_bps=int(p["rigidity_bps"]), master_seed=seed)
    f = max(int(x) for x in p["menu_costs"])
    band = mc.optimal_ss_band(f, cfg.curvature, d)
    m = cell_market(cfg, f, d, rows[0]["seed"])  # common random numbers: one seed per grid
    return Phase14(rows, cfg, p, seed, thr, f, band, mc.adjustment_frequency(band, d),
                   m.path(TRACED), surplus_split(cfg, rows))


class SawtoothSection(Section):
    """Line chart and table of one seller's posted price against its target."""
    kind = "path"

    def __init__(self, path: Sequence[Any], title: str) -> None:
        super().__init__(title)
        self.path = list(path)

    def body(self) -> str:
        xs = [float(r.t) for r in self.path]
        chart = multi_line_chart({"posted price": [float(r.price) for r in self.path],
                                  "target price": [float(r.target) for r in self.path]}, xs,
                                 title="Sawtooth price path", xlabel="step", ylabel="price (ticks)")
        rows = [(r.t, r.target, r.price, r.gap, "adjust" if r.adjusted else "")
                for r in self.path[:SHOWN]]
        return inline_svg(chart) + table(("step", "target", "posted", "gap", "action"), rows)


def report(p: Phase14) -> Report:
    """fl-026 report: band and threshold, frequency heatmap, sawtooth path, surplus split."""
    head = {f"analytic band (menu cost {p.traced_cost}, inflation {p.drift})": p.traced_band,
            "analytic adjustment frequency": str(p.traced_frequency),
            "analytic rigidity threshold": p.thr.analytic,
            "bisected rigidity threshold": p.thr.menu_cost,
            "band at the threshold": p.thr.band,
            "rigidity cut-off (bps of flexible)": int(p.params["rigidity_bps"])}
    return Report("frictionlab Phase 14: menu costs and nominal price rigidity",
                  master_seed=p.master_seed, grid={COST: p.params["menu_costs"],
                                                   INFL: p.params["inflations"]},
                  meta={"config": CONFIG.name, "sellers": p.cfg.n_sellers,
                        "curvature": p.cfg.curvature, "horizon": p.cfg.horizon},
                  sections=[MetricsSection(extra=head, title="Band and rigidity threshold"),
                            SweepSection(p.rows, [COST, INFL], "adjustment_frequency",
                                         title="Adjustment frequency: menu cost x inflation"),
                            SawtoothSection(p.path, f"Seller {TRACED} at menu cost "
                                                    f"{p.traced_cost}, inflation {p.drift}"),
                            PnLSection(p.split, title="Seller surplus split over the sweep (ticks)")])


def show(p: Phase14) -> None:
    print(f"menu-cost sweep (seed {p.master_seed}, {p.cfg.n_sellers} sellers, curvature "
          f"{p.cfg.curvature}, horizon {p.cfg.horizon}):")
    print(f"  {'F':>3} {'d':>2} {'band':>4} {'freq':>5} {'gap':>5} {'welfare':>8}")
    for f in p.params["menu_costs"]:
        for d in p.params["inflations"]:
            r = p.cell(f, d)
            print(f"  {f:>3} {d:>2} {r['band']:>4} {r['adjustment_frequency']:>5.3f} "
                  f"{r['mean_abs_gap']:>5.2f} {r['welfare_change_vs_flexible']:>8}")
    print(f"seller {TRACED} at menu cost {p.traced_cost}, inflation {p.drift} (band {p.traced_band}):")
    for r in p.path[:SHOWN]:
        mark = "  <- adjust" if r.adjusted else ""
        print(f"  t {r.t:>3} target {r.target} posted {r.price} gap {r.gap:+d}{mark}")
    print(f"analytic band {p.traced_band}, analytic frequency {p.traced_frequency}; rigidity "
          f"threshold: analytic {p.thr.analytic}, bisected {p.thr.menu_cost} (band {p.thr.band})")
    print("surplus split: " + ", ".join(f"{k} {v}" for k, v in p.split.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", help="output directory (default: a new temp dir)")
    ap.add_argument("--small", action="store_true", help="use the experiment's small preset")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab_phase14_"))
    out.mkdir(parents=True, exist_ok=True)
    p = run(out, small=args.small)
    show(p)
    print(f"report: {report(p).write_report(out / 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
