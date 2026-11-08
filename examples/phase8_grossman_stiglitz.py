"""Phase 8 Grossman-Stiglitz walkthrough: who pays for information prices already carry, one report.

The fl-053 `grossman_stiglitz` experiment is run through the public CLI entry point
`frictionlab.cli.main(argv)` on `examples/configs/grossman_stiglitz_small.json`, into
OUT/run/ (rows.csv, resolved_config.json, report.html). From those artifacts it prints

1. buyer fraction, quoted spread and price-efficiency error (mean |mid - V|) over
   signal cost x noise intensity, with the analytic informed share at each cost;
2. per population cost, the fl-024 replicator buyer share against the analytic
   fixed-point informed share from the fl-030 GM spread;
3. the bisected information-collapse cost (buyer fraction below the floor) and its
   analytic bound.

It then renders one fl-026 HTML report, OUT/report.html, with a SweepSection heatmap
(signal cost x noise intensity -> buyer fraction), an EquilibriumSection with the
buyer-share trajectory and its annotated fixed point, and a PnLSection splitting
buyer PnL over the sweep into trading gain, signal cost and decision cost. Nothing
time- or path-dependent is written, so two runs are byte-identical.

    python examples/phase8_grossman_stiglitz.py --small [OUT]   # OUT defaults to a temp dir
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a checkout

from frictionlab.analysis.report import Report  # noqa: E402  (after sys.path tweak)
from frictionlab.analysis.report_sections import (  # noqa: E402
    EquilibriumSection, MetricsSection, PnLSection, SweepSection)
from frictionlab.cli import main as cli_main  # noqa: E402
from frictionlab.experiments import grossman_stiglitz as gs  # noqa: E402

CONFIG = HERE / "configs" / "grossman_stiglitz_small.json"
COST, NOISE = gs.COST, gs.NOISE
SPLIT_KEYS = ("trading gain", "signal cost", "decision cost", "net")


def cli(*argv: str) -> str:
    """Run `frictionlab ARGV` in-process; return its output, raise on a non-zero exit."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli_main(list(argv))
    if code != 0:
        raise RuntimeError(f"`frictionlab {' '.join(argv)}` exited {code}\n{buf.getvalue()}")
    return buf.getvalue()


def _num(s: str) -> Any:
    try:
        return int(s)                 # exact, also for 64-bit seeds
    except ValueError:
        return float(s)


def read_rows(path: Path) -> List[Dict[str, Any]]:
    """rows.csv back to numbers: ints stay exact ints, everything else floats."""
    return [{k: (v if k == "error" or v == "" else _num(v)) for k, v in r.items()}
            for r in csv.DictReader(path.open(encoding="utf-8"))]


def mean_by(rows: Sequence[Dict[str, Any]], key: str, **where: Any) -> float:
    return float(np.mean([r[key] for r in rows if all(r[k] == v for k, v in where.items())]))


def buyer_split(cfg: gs.Config, rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sweep totals in ticks over every acquirer that bought a signal, rebuilt per cell seed."""
    out = dict.fromkeys(SPLIT_KEYS, 0)
    for r in rows:
        cell = gs.build(cfg, {COST: r[COST], NOISE: r[NOISE]}, r["seed"]).run(cfg.horizon)
        parts = cell.parts()[1]
        for a in cell.acquirers:
            if getattr(a, "n_signals", 0):
                p = parts[a.agent_id]
                out["trading gain"] += p.trading_gain
                out["signal cost"] -= p.signal_cost
                out["decision cost"] -= p.decision_cost
                out["net"] += p.net
    return out


@dataclass
class Phase8:
    """Everything the example reports, for tests and for the Phase 5 summary."""
    rows: List[Dict[str, Any]]
    cfg: gs.Config
    params: Dict[str, Any]
    master_seed: int
    population: List[Dict[str, Any]]
    eq: Any                              # EquilibriumResult at the middle population cost
    eq_cost: int
    collapse: gs.InformationCollapse
    split: Dict[str, int]

    def buyer_fraction(self, cost: int) -> float:
        """Buyer fraction at one signal cost, averaged over noise intensities."""
        return mean_by(self.rows, "buyer_fraction", **{COST: cost})


def run(out: Path, small: bool = True) -> Phase8:
    """Run the experiment via the CLI into OUT/run and derive every printed number."""
    flags = ("--small",) if small else ()
    cli("validate", str(CONFIG), *flags)
    cli("run", str(CONFIG), *flags, "--out", str(out / "run"))
    rows = read_rows(out / "run" / "rows.csv")
    resolved = json.loads((out / "run" / "resolved_config.json").read_text(encoding="utf-8"))
    p, seed = resolved["params"], int(resolved["master_seed"])
    cfg = gs.Config(**{f.name: p[f.name] for f in fields(gs.Config) if f.name in p})
    noises = p["noise_intensities"]
    lam = float(noises[len(noises) // 2])
    pop = []
    for c in p["population_costs"]:
        eq = gs.population(cfg, c, lam, generations=p["generations"], master_seed=seed)
        pop.append({"cost": int(c), "eq": eq, "analytic": gs.gs_equilibrium_informed_share(
            c, cfg.accuracy_bps, cfg.jump, cfg.noise_buy_prob), "buyer_share": float(
            (eq.shares[-1] if eq.fixed_point is None else eq.fixed_point)[0])})
    mid = pop[len(pop) // 2]
    col = gs.find_information_collapse(cfg, lam, p["collapse_floor"], master_seed=seed)
    table = [{k: v for k, v in d.items() if k != "eq"} for d in pop]
    return Phase8(rows, cfg, p, seed, table, mid["eq"], mid["cost"], col, buyer_split(cfg, rows))


def report(p: Phase8) -> Report:
    """fl-026 report: headline numbers, buyer-fraction heatmap, replicator path, PnL split."""
    head = {f"replicator buyer share (cost {d['cost']})": d["buyer_share"] for d in p.population}
    head.update({f"analytic informed share (cost {d['cost']})": d["analytic"] for d in p.population})
    head.update({"bisected collapse cost": p.collapse.threshold,
                 "analytic collapse cost": p.collapse.analytic, "collapse floor": p.collapse.floor})
    return Report("frictionlab Phase 8: the Grossman-Stiglitz paradox",
                  master_seed=p.master_seed, grid={COST: p.params["signal_costs"],
                                                   NOISE: p.params["noise_intensities"]},
                  meta={"config": CONFIG.name, "accuracy_bps": p.cfg.accuracy_bps,
                        "jump": p.cfg.jump, "acquirers": p.cfg.n_acquirers},
                  sections=[MetricsSection(extra=head, title="Informed share and collapse"),
                            SweepSection(p.rows, [COST, NOISE], "buyer_fraction",
                                         title="Buyer fraction: signal cost x noise intensity"),
                            EquilibriumSection(p.eq, title=f"Buyer share under the replicator "
                                                           f"(signal cost {p.eq_cost})"),
                            PnLSection(p.split, title="Buyer PnL split over the sweep (ticks)")])


def show(p: Phase8) -> None:
    c = p.cfg
    print(f"grossman-stiglitz sweep (seed {p.master_seed}, accuracy {c.accuracy_bps} bps, "
          f"jump {c.jump}, {c.n_acquirers} acquirers):")
    print(f"  {'cost':>4} {'noise':>5} {'mu*':>4} {'buyers':>6} {'spread':>6} {'|mid-V|':>7} "
          f"{'gain':>7} {'net':>7} {'sink':>5}")
    for cost in p.params["signal_costs"]:
        mu = gs.gs_equilibrium_informed_share(cost, c.accuracy_bps, c.jump, c.noise_buy_prob)
        for lam in p.params["noise_intensities"]:
            v = [mean_by(p.rows, k, **{COST: cost, NOISE: lam}) for k in
                 ("buyer_fraction", "quoted_spread", "efficiency_error",
                  "informed_gain_per_buyer", "buyer_net_pnl", "sink_revenue")]
            print(f"  {cost:>4} {lam:>5g} {mu:>4.2f} {v[0]:>6.2f} {v[1]:>6.2f} {v[2]:>7.2f} "
                  f"{v[3]:>7.1f} {v[4]:>7.1f} {v[5]:>5.0f}")
    for d in p.population:
        print(f"fixed point (cost {d['cost']}): replicator buyer share {d['buyer_share']:.4f}, "
              f"analytic informed share {d['analytic']:.2f}")
    print(f"information collapse: bisected {p.collapse.threshold}, analytic {p.collapse.analytic} "
          f"(floor {p.collapse.floor:g})")
    print("buyer PnL split: " + ", ".join(f"{k} {v}" for k, v in p.split.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", help="output directory (default: a new temp dir)")
    ap.add_argument("--small", action="store_true", help="use the experiment's small preset")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab_phase8gs_"))
    out.mkdir(parents=True, exist_ok=True)
    p = run(out, small=args.small)
    show(p)
    print(f"report: {report(p).write_report(out / 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
