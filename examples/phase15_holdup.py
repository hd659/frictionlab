"""Phase 15 hold-up walkthrough: underinvestment, costly contracts and the contracting threshold.

The fl-074 `holdup` experiment is run through the public CLI entry point
`frictionlab.cli.main(argv)` on `examples/configs/holdup_small.json`, into OUT/run/
(rows.csv, resolved_config.json, report.html). From those artifacts it prints

1. contracting rate, underinvestment gap and surplus share over contracting cost x seller weight;
2. the seller payoff curve over investment levels under hold-up and under a contract for one
   cell (probe weight, costliest contracting cost), with the hold-up and first-best choices marked;
3. the analytic max contracting cost next to the bisected contracting threshold.

It then renders one fl-026 HTML report, OUT/report.html, with a SweepSection heatmap
(contracting cost x seller weight -> surplus share), a line chart of seller payoff by
investment level under both regimes, and a PnLSection splitting first-best surplus into
buyer gain, seller net gain, sunk investment, contract sink and underinvestment deadweight
loss. Two runs are byte-identical.

    python examples/phase15_holdup.py --small [OUT]   # OUT defaults to a temp dir
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a checkout
sys.path.insert(0, str(HERE))

from frictionlab.agents.specific_investor import seller_share  # noqa: E402
from frictionlab.analysis.pnl import holdup_pnl  # noqa: E402
from frictionlab.analysis.plots import multi_line_chart  # noqa: E402
from frictionlab.analysis.report import Report  # noqa: E402
from frictionlab.analysis.report_sections import (  # noqa: E402
    MetricsSection, PnLSection, Section, SweepSection, inline_svg, table)
from phase9_lemons import cli  # noqa: E402  (shared CLI helper)
from phase12_critical_mass import read_rows  # noqa: E402  (exact-int CSV reader)

hu = importlib.import_module("frictionlab.experiments.holdup")
CONFIG = HERE / "configs" / "holdup_small.json"
COST, WEIGHT = hu.COST, hu.WEIGHT
SPLIT_KEYS = ("buyer gain", "seller net gain", "sunk investment", "contract sink",
              "underinvestment deadweight loss")
TUPLES = ("investment_costs", "buyer_values", "value_noise")


def config(p: Dict[str, Any]) -> hu.Config:
    return hu.Config(**{k: tuple(p[k]) if k in TUPLES else p[k] for k in hu.Config.__dataclass_fields__})


def surplus_split(cfg: hu.Config, rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sweep totals in ticks (all >= 0). Identities, exact in ticks:
    buyer gain + seller net gain + contract sink + deadweight loss = first-best surplus;
    seller net gain + sunk investment + contract sink = seller revenue over the outside value."""
    out = dict.fromkeys(SPLIT_KEYS, 0)
    for r in rows:
        m = hu.build(cfg, {COST: r[COST], WEIGHT: r[WEIGHT]}, r["seed"]).run().market
        out["buyer gain"] += r["buyer_surplus"]
        out["seller net gain"] += r["seller_surplus"]
        out["sunk investment"] += sum(s.investment for s in holdup_pnl(m).values())
        out["contract sink"] += r["contracting_cost_paid"]
        out["underinvestment deadweight loss"] += (r["max_surplus"] - r["welfare"]
                                                   - r["contracting_cost_paid"])
    return out


def payoff_curve(cfg: hu.Config, w: int, cost: int) -> List[Tuple[int, int, int]]:
    """(level, hold-up payoff, contract payoff) per investment level, net of the outside value:
    seller_share(Q(i)) - c(i) under hold-up, seller_share(Q(i) - c(i)) - cost under a contract."""
    o = cfg.outside_value
    return [(i, seller_share(v - o, w) - c, seller_share(v - o - c, w) - cost)
            for i, (c, v) in enumerate(cfg.schedule)]


@dataclass
class Phase15:
    """Everything the example reports, for tests and for the Phase 5 summary."""
    rows: List[Dict[str, Any]]
    cfg: hu.Config
    params: Dict[str, Any]
    master_seed: int
    thr: hu.ContractingThreshold
    weight: int                                  # traced cell: probe weight ...
    cost: int                                    # ... at the costliest contracting cost
    holdup_level: int
    first_best: int
    curve: List[Tuple[int, int, int]]
    split: Dict[str, int]

    def cell(self, c: int, w: int) -> Dict[str, Any]:
        return next(r for r in self.rows if r[COST] == c and r[WEIGHT] == w)


def run(out: Path, small: bool = True) -> Phase15:
    """Run the experiment via the CLI into OUT/run and derive every printed number."""
    flags = ("--small",) if small else ()
    cli("validate", str(CONFIG), *flags)
    cli("run", str(CONFIG), *flags, "--out", str(out / "run"))
    rows = read_rows(out / "run" / "rows.csv")
    resolved = json.loads((out / "run" / "resolved_config.json").read_text(encoding="utf-8"))
    p, seed = resolved["params"], int(resolved["master_seed"])
    cfg, w = config(p), int(p["probe_weight"])
    thr = hu.find_contracting_threshold(cfg, w, collapse_bps=int(p["collapse_bps"]), master_seed=seed)
    c = max(int(x) for x in p["contracting_costs"])
    s, o = cfg.schedule, cfg.outside_value
    return Phase15(rows, cfg, p, seed, thr, w, c, hu.holdup_investment(s, o, w),
                   hu.first_best_investment(s, o), payoff_curve(cfg, w, c), surplus_split(cfg, rows))


class PayoffSection(Section):
    """Line chart and table of seller payoff by investment level under both regimes."""
    kind = "payoff"

    def __init__(self, p: Phase15, title: str) -> None:
        super().__init__(title)
        self.p = p

    def body(self) -> str:
        xs = [float(i) for i, _, _ in self.p.curve]
        chart = multi_line_chart({"hold-up": [float(h) for _, h, _ in self.p.curve],
                                  "contract": [float(k) for _, _, k in self.p.curve]}, xs,
                                 title="Seller payoff by investment level", xlabel="investment level",
                                 ylabel="payoff over outside value (ticks)")
        rows = [(i, h, k, mark(self.p, i)) for i, h, k in self.p.curve]
        return inline_svg(chart) + table(("level", "hold-up", "contract", "choice"), rows)


def mark(p: Phase15, i: int) -> str:
    return " / ".join(n for n, lv in (("hold-up", p.holdup_level), ("first best", p.first_best))
                      if lv == i)


def report(p: Phase15) -> Report:
    """fl-026 report: thresholds, surplus-share heatmap, payoff curves, surplus split."""
    head = {f"hold-up investment (weight {p.weight})": p.holdup_level,
            "first-best investment": p.first_best,
            "analytic max contracting cost": p.thr.analytic,
            "bisected contracting threshold": p.thr.contracting_cost,
            "collapse cut-off (bps of pairs contracting)": int(p.params["collapse_bps"])}
    return Report("frictionlab Phase 15: hold-up and costly contracting",
                  master_seed=p.master_seed, grid={COST: p.params["contracting_costs"],
                                                   WEIGHT: p.params["seller_weights"]},
                  meta={"config": CONFIG.name, "pairs": p.cfg.n_pairs,
                        "outside value": p.cfg.outside_value, "rounds": p.cfg.rounds},
                  sections=[MetricsSection(extra=head, title="Investment and contracting threshold"),
                            SweepSection(p.rows, [COST, WEIGHT], "surplus_share",
                                         title="Surplus share: contracting cost x seller weight"),
                            PayoffSection(p, f"Seller payoff at weight {p.weight}, "
                                             f"contracting cost {p.cost}"),
                            PnLSection(p.split, title="First-best surplus split over the sweep (ticks)")])


def show(p: Phase15) -> None:
    print(f"hold-up sweep (seed {p.master_seed}, {p.cfg.n_pairs} pairs, outside value "
          f"{p.cfg.outside_value}, {p.cfg.rounds} rounds):")
    print(f"  {'cost':>4} {'w':>5} {'rate':>5} {'gap':>5} {'share':>5}")
    for c in p.params["contracting_costs"]:
        for w in p.params["seller_weights"]:
            r = p.cell(c, w)
            print(f"  {c:>4} {w:>5} {r['contracting_rate']:>5.2f} "
                  f"{r['underinvestment_gap']:>5.1f} {r['surplus_share']:>5.3f}")
    print(f"seller payoff at weight {p.weight}, contracting cost {p.cost}:")
    for i, h, k in p.curve:
        m = mark(p, i)
        print(f"  level {i} hold-up {h:>4} contract {k:>4}" + (f"  <- {m}" if m else ""))
    print(f"max contracting cost: analytic {p.thr.analytic}, bisected threshold "
          f"{p.thr.contracting_cost} (weight {p.weight})")
    print("surplus split: " + ", ".join(f"{k} {v}" for k, v in p.split.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", help="output directory (default: a new temp dir)")
    ap.add_argument("--small", action="store_true", help="use the experiment's small preset")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab_phase15_"))
    out.mkdir(parents=True, exist_ok=True)
    p = run(out, small=args.small)
    show(p)
    print(f"report: {report(p).write_report(out / 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
