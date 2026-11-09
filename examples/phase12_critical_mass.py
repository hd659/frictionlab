"""Phase 12 critical-mass walkthrough: cold-start traps, seeding subsidies and tipping.

The fl-065 `critical_mass` experiment is run through the public CLI entry point
`frictionlab.cli.main(argv)` on `examples/configs/critical_mass_small.json`, into OUT/run/
(rows.csv, resolved_config.json, report.html). From those artifacts it prints

1. final membership, platform PnL and welfare change (against the zero-adoption-cost
   twin) over adoption cost x seed members;
2. the round-by-round membership path of both sides for a seed just below and just at
   the bisected tipping seed, at the traced adoption cost;
3. the analytic critical mass per adoption cost and the bisected vs analytic tipping seed.

It then renders one fl-026 HTML report, OUT/report.html, with a SweepSection heatmap
(adoption cost x seed members -> final membership share), a line chart of both sides'
membership paths around the tipping point, and a PnLSection splitting the sweep into
member standalone value, cross-side gain, platform fee revenue, subsidy outlay and the
adoption sink. Two runs are byte-identical.

    python examples/phase12_critical_mass.py --small [OUT]   # OUT defaults to a temp dir
"""
from __future__ import annotations

import argparse
import csv
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

from frictionlab.analysis.pnl import member_surplus, platform_pnl  # noqa: E402
from frictionlab.analysis.plots import multi_line_chart  # noqa: E402
from frictionlab.analysis.report import Report  # noqa: E402
from frictionlab.analysis.report_sections import (  # noqa: E402
    MetricsSection, PnLSection, Section, SweepSection, inline_svg, table)
from phase9_lemons import cli  # noqa: E402  (shared CLI helper)

# the package re-exports a `critical_mass` function, so import the module by path
cm = importlib.import_module("frictionlab.experiments.critical_mass")
CONFIG = HERE / "configs" / "critical_mass_small.json"
COST, SEED = cm.COST, cm.SEED
SPLIT_KEYS = ("member standalone value", "cross-side gain", "platform fee revenue",
              "subsidy outlay", "adoption sink")
TRACE_COST = 0                                  # adoption cost of the traced tipping pair
# (round, side A members, side B members, joins A+B, leaves A+B)
Step = Tuple[int, int, int, int, int]


def _cell(v: str) -> Any:
    for cast in (int, float):
        try:
            return cast(v)                     # ints stay exact, also for 64-bit seeds
        except ValueError:
            pass
    return v                                   # outcome, booleans, error stay text


def read_rows(path: Path) -> List[Dict[str, Any]]:
    return [{k: _cell(v) for k, v in r.items()} for r in csv.DictReader(path.open(encoding="utf-8"))]


def config(p: Dict[str, Any]) -> cm.Config:
    return cm.Config(**{k: tuple(p[k]) if isinstance(p[k], list) else p[k]
                        for k in cm.Config.__dataclass_fields__})


def surplus_split(cfg: cm.Config, rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sweep totals in ticks; welfare = standalone + cross-side + adoption sink, and
    platform PnL = fee revenue + subsidy outlay (outlays carry a minus sign)."""
    out = dict.fromkeys(SPLIT_KEYS, 0)
    for r in rows:
        m = cm.build(cfg, {COST: r[COST], SEED: r[SEED]}, r["seed"]).run().market
        for s in member_surplus(m).values():
            out["member standalone value"] += s.standalone
            out["cross-side gain"] += s.cross_side
        pnl = platform_pnl(m)
        out["platform fee revenue"] += pnl.fee_revenue
        out["subsidy outlay"] -= pnl.subsidy_outlay
        out["adoption sink"] -= m.adoption.revenue
    return out


def trace(cfg: cm.Config, k: int, cost: int, seed: int) -> List[Step]:
    """Membership counts per side, from the seeded state (round 0) to the stop."""
    m = cm.build(cfg, {COST: cost, SEED: k}, seed).run().market
    k0 = cm.build(cfg, {COST: cost, SEED: k}, seed).k
    steps: List[Step] = [(0, k0[0], k0[1], 0, 0)]
    steps += [(r.t, r.members[0], r.members[1], sum(r.joins), sum(r.leaves)) for r in m.log]
    return steps


class PathSection(Section):
    """Line chart of both sides' membership just below and at the tipping seed."""
    kind = "path"

    def __init__(self, below: Sequence[Step], above: Sequence[Step], k: int, title: str) -> None:
        super().__init__(title)
        self.below, self.above, self.k = list(below), list(above), k

    def body(self) -> str:
        n = max(len(self.below), len(self.above))
        pad = lambda s, i: [float(x[i]) for x in s] + [float(s[-1][i])] * (n - len(s))  # noqa: E731
        series = {f"A, seed {self.k - 1}": pad(self.below, 1), f"B, seed {self.k - 1}": pad(self.below, 2),
                  f"A, seed {self.k}": pad(self.above, 1), f"B, seed {self.k}": pad(self.above, 2)}
        chart = multi_line_chart(series, [float(i) for i in range(n)], title="Membership paths",
                                 xlabel="round", ylabel="members")
        hdr = ("seed", "round", "side A", "side B", "joins", "leaves")
        rows = [(self.k - 1, *s) for s in self.below] + [(self.k, *s) for s in self.above]
        return inline_svg(chart) + table(hdr, rows)


@dataclass
class Phase12:
    """Everything the example reports, for tests and for the Phase 5 summary."""
    rows: List[Dict[str, Any]]
    cfg: cm.Config
    params: Dict[str, Any]
    master_seed: int
    trace_tip: cm.TippingSeed                    # bisected tipping seed at TRACE_COST
    below: List[Step]                            # path at trace_tip - 1
    above: List[Step]                            # path at trace_tip
    tip: cm.TippingSeed                          # bisected tipping seed at tipping_cost
    analytic: Dict[int, int]                     # critical_mass per adoption cost
    split: Dict[str, int]

    def members(self, cost: int, k: int) -> int:
        return next(r["members"] for r in self.rows if r[COST] == cost and r[SEED] == k)


def run(out: Path, small: bool = True) -> Phase12:
    """Run the experiment via the CLI into OUT/run and derive every printed number."""
    flags = ("--small",) if small else ()
    cli("validate", str(CONFIG), *flags)
    cli("run", str(CONFIG), *flags, "--out", str(out / "run"))
    rows = read_rows(out / "run" / "rows.csv")
    resolved = json.loads((out / "run" / "resolved_config.json").read_text(encoding="utf-8"))
    p, seed = resolved["params"], int(resolved["master_seed"])
    cfg, frac = config(p), p["tipping_fraction"]
    tt = cm.find_tipping_seed(cfg, TRACE_COST, frac, master_seed=seed)
    cell_seed = rows[0]["seed"]                  # common random numbers: one seed per grid
    below = trace(cfg, tt.seed_members - 1, TRACE_COST, cell_seed)
    above = trace(cfg, tt.seed_members, TRACE_COST, cell_seed)
    tip = cm.find_tipping_seed(cfg, p["tipping_cost"], frac, master_seed=seed)
    analytic = {c: next(r["critical_mass"] for r in rows if r[COST] == c) for c in p["adoption_costs"]}
    return Phase12(rows, cfg, p, seed, tt, below, above, tip, analytic, surplus_split(cfg, rows))


def report(p: Phase12) -> Report:
    """fl-026 report: headline numbers, share heatmap, membership paths, surplus split."""
    head = {f"bisected tipping seed (adoption cost {p.tip.adoption_cost}, > {p.tip.fraction:g})":
            p.tip.seed_members, "analytic critical mass": p.tip.analytic,
            f"tipping seed at adoption cost {TRACE_COST}": p.trace_tip.seed_members,
            "cold-start traps (seed 0, empty)": sum(1 for r in p.rows if r[SEED] == 0 and r["members"] == 0),
            "total seed subsidy": sum(r["seed_subsidy"] for r in p.rows)}
    k = p.trace_tip.seed_members
    return Report("frictionlab Phase 12: network effects, critical mass and cold start",
                  master_seed=p.master_seed, grid={COST: p.params["adoption_costs"],
                                                   SEED: p.params["seed_members"]},
                  meta={"config": CONFIG.name, "members per side": f"{p.cfg.n_a} / {p.cfg.n_b}",
                        "cross-side benefit": f"{p.cfg.benefit_a} / {p.cfg.benefit_b}",
                        "fees": f"{p.cfg.fee_a} / {p.cfg.fee_b}"},
                  sections=[MetricsSection(extra=head, title="Critical mass and tipping"),
                            SweepSection(p.rows, [COST, SEED], "share",
                                         title="Final membership share: adoption cost x seed members"),
                            PathSection(p.below, p.above, k, f"Membership paths around the tipping "
                                                             f"seed {k} (adoption cost {TRACE_COST})"),
                            PnLSection(p.split, title="Surplus split over the sweep (ticks)")])


def show(p: Phase12) -> None:
    print(f"critical-mass sweep (seed {p.master_seed}, {p.cfg.n_a} + {p.cfg.n_b} members, "
          f"benefit {p.cfg.benefit_a}/{p.cfg.benefit_b}, horizon {p.cfg.horizon}):")
    print(f"  {'cost':>4} {'seed':>4} {'members':>7} {'pnl':>5} {'welfare':>7}")
    for c in p.params["adoption_costs"]:
        for k in p.params["seed_members"]:
            r = next(r for r in p.rows if r[COST] == c and r[SEED] == k)
            print(f"  {c:>4} {k:>4} {r['members']:>7} {r['platform_pnl']:>5} "
                  f"{r['welfare_change_vs_free_adoption']:>7}")
    k = p.trace_tip.seed_members
    for label, kk, steps in (("below", k - 1, p.below), ("at", k, p.above)):
        path = " -> ".join(f"{a}/{b}" for _, a, b, _, _ in steps)
        print(f"seed {kk} ({label} tipping seed, adoption cost {TRACE_COST}): {path}")
    print("analytic critical mass: " + ", ".join(f"cost {c} -> {v}" for c, v in p.analytic.items()))
    print(f"tipping seed (adoption cost {p.tip.adoption_cost}, > {p.tip.fraction:g}): "
          f"bisected {p.tip.seed_members}, analytic {p.tip.analytic}")
    print("surplus split: " + ", ".join(f"{k} {v}" for k, v in p.split.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", help="output directory (default: a new temp dir)")
    ap.add_argument("--small", action="store_true", help="use the experiment's small preset")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab_phase12_"))
    out.mkdir(parents=True, exist_ok=True)
    p = run(out, small=args.small)
    show(p)
    print(f"report: {report(p).write_report(out / 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
