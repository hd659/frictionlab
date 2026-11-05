"""Phase 7 search-cost walkthrough: costly bilateral search and perishable value, one report.

The fl-047 `search_cost` experiment is run through the public CLI entry point
`frictionlab.cli.main(argv)` on `examples/configs/search_cost_small.json`, into
OUT/run/ (rows.csv, resolved_config.json, report.html). From those artifacts it prints

1. volume, mean time to match and surplus share (of the fl-022 CE maximum) over
   search cost, at zero decay;
2. the analytic collapse bound (no meeting can cover its cost) and the bisected
   half-volume threshold, at zero and at the largest configured decay;
3. a decay view at one search cost: perishable sellers shade their asks down, so
   they match sooner at lower prices.

It then renders one fl-026 HTML report, OUT/report.html, with a SweepSection
heatmap (search cost x decay -> surplus share) and a PnLSection splitting the
sweep's surplus into buyer gain, seller gain, search-cost sink and decay loss.
Nothing time- or path-dependent is written, so two runs are byte-identical.

    python examples/phase7_search_cost.py --small [OUT]     # OUT defaults to a temp dir
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
from frictionlab.analysis.report_sections import MetricsSection, PnLSection, SweepSection  # noqa: E402
from frictionlab.cli import main as cli_main  # noqa: E402
from frictionlab.experiments import search_cost as sc  # noqa: E402

CONFIG = HERE / "configs" / "search_cost_small.json"
COST, DECAY = sc.COST, sc.DECAY
DECAY_VIEW_COST = 10
DECAY_VIEW = (0, 1, 2, 4, 8)
NUMERIC = ("volume", "mean_time_to_match", "surplus", "surplus_share", "search_cost_paid",
           "decay_loss", "seller_share", "seed", COST, DECAY)


def cli(*argv: str) -> str:
    """Run `frictionlab ARGV` in-process; return its output, raise on a non-zero exit."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli_main(list(argv))
    if code != 0:
        raise RuntimeError(f"`frictionlab {' '.join(argv)}` exited {code}\n{buf.getvalue()}")
    return buf.getvalue()


def read_rows(path: Path) -> List[Dict[str, Any]]:
    """rows.csv back to numbers (ints where integral)."""
    out = []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        row: Dict[str, Any] = dict(r)
        for k in NUMERIC:
            x = float(r[k])
            row[k] = int(x) if x.is_integer() and k not in ("surplus_share", "seller_share",
                                                             "mean_time_to_match") else x
        out.append(row)
    return out


def config_from(resolved: Dict[str, Any]) -> sc.Config:
    """The experiment `Config` the CLI actually ran, rebuilt from resolved_config.json."""
    p = resolved["params"]
    return sc.Config(**{f.name: tuple(p[f.name]) if isinstance(p[f.name], list) else p[f.name]
                        for f in fields(sc.Config) if f.name in p})


def mean_by(rows: Sequence[Dict[str, Any]], key: str, **where: Any) -> float:
    return float(np.mean([r[key] for r in rows if all(r[k] == v for k, v in where.items())]))


def decay_view(cfg: sc.Config, seeds: Sequence[int]) -> List[Dict[str, float]]:
    """Mean trade price and time to match vs decay at `DECAY_VIEW_COST`, on the run's seeds."""
    out = []
    for d in DECAY_VIEW:
        prices, times, vols = [], [], []
        for seed in seeds:
            cell = sc.build(cfg, {COST: DECAY_VIEW_COST, DECAY: d}, seed).run(cfg.until)
            m = sc.search_metrics(cfg, cell)
            prices += [t.price for t in cell.market.trades]
            times.append(m["mean_time_to_match"])
            vols.append(m["volume"])
        out.append({DECAY: d, "volume": float(np.mean(vols)),
                    "mean_price": float(np.mean(prices)) if prices else float("nan"),
                    "mean_time_to_match": float(np.mean(times))})
    return out


def surplus_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sweep totals in ticks: gains to each side, and what search and spoilage burned."""
    seller = sum(round(r["seller_share"] * r["surplus"]) for r in rows)
    return {"buyer gain": sum(r["surplus"] for r in rows) - seller, "seller gain": seller,
            "search-cost sink": -sum(r["search_cost_paid"] for r in rows),
            "decay loss": -sum(r["decay_loss"] for r in rows)}


@dataclass
class Phase7:
    """Everything the example reports, for tests and for the Phase 5 summary."""
    rows: List[Dict[str, Any]]
    cfg: sc.Config
    master_seed: int
    collapses: List[sc.SearchCollapse]
    decay: List[Dict[str, float]]
    split: Dict[str, int]


def run(out: Path, small: bool = True) -> Phase7:
    """Run the experiment via the CLI into OUT/run and derive every printed number."""
    flags = ("--small",) if small else ()
    cli("validate", str(CONFIG), *flags)
    cli("run", str(CONFIG), *flags, "--out", str(out / "run"))
    rows = read_rows(out / "run" / "rows.csv")
    resolved = json.loads((out / "run" / "resolved_config.json").read_text(encoding="utf-8"))
    cfg, seed = config_from(resolved), int(resolved["master_seed"])
    decays = sorted({r[DECAY] for r in rows})
    collapses = [sc.find_search_collapse(cfg, 0.5, d, master_seed=seed)
                 for d in dict.fromkeys((decays[0], decays[-1]))]
    seeds = sorted({r["seed"] for r in rows if r["cell"] == "0"})
    return Phase7(rows, cfg, seed, collapses, decay_view(cfg, seeds), surplus_split(rows))


def report(p: Phase7) -> Report:
    """fl-026 report: thresholds, surplus-share heatmap and the surplus split."""
    th = {f"collapse threshold (decay {c.decay})": c.threshold for c in p.collapses}
    th.update({f"analytic bound (decay {c.decay})": c.analytic_bound for c in p.collapses})
    return Report("frictionlab Phase 7: search costs and perishable value",
                  master_seed=p.master_seed, grid={COST: sorted({r[COST] for r in p.rows}),
                                                   DECAY: sorted({r[DECAY] for r in p.rows})},
                  meta={"config": CONFIG.name, "ce max surplus": sc.competitive_equilibrium(
                      p.cfg.values, p.cfg.costs).max_surplus},
                  sections=[MetricsSection(extra=th, title="Collapse thresholds"),
                            SweepSection(p.rows, [COST, DECAY], "surplus_share",
                                         title="Surplus share: search cost x decay"),
                            PnLSection(p.split, title="Surplus split over the sweep (ticks)")])


def show(p: Phase7) -> None:
    d0 = p.rows[0][DECAY]
    print(f"search cost sweep (decay {d0}, seed {p.master_seed}):")
    print(f"  {'cost':>5} {'volume':>7} {'time':>7} {'share':>6} {'paid':>6}")
    for c in sorted({r[COST] for r in p.rows}):
        v = [mean_by(p.rows, k, **{COST: c, DECAY: d0}) for k in
             ("volume", "mean_time_to_match", "surplus_share", "search_cost_paid")]
        print(f"  {c:>5} {v[0]:>7.1f} {v[1]:>7.2f} {v[2]:>6.2f} {v[3]:>6.0f}")
    for c in p.collapses:
        print(f"collapse (decay {c.decay}): bisected {c.threshold}, analytic bound "
              f"{c.analytic_bound}, within bound {c.within_bound}")
    print(f"decay view (search cost {DECAY_VIEW_COST}): perishability -> lower price, faster match")
    for r in p.decay:
        print(f"  decay {r[DECAY]:>2}: volume {r['volume']:.1f}  mean price {r['mean_price']:.2f}"
              f"  time to match {r['mean_time_to_match']:.2f}")
    print("surplus split: " + ", ".join(f"{k} {v}" for k, v in p.split.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", help="output directory (default: a new temp dir)")
    ap.add_argument("--small", action="store_true", help="use the experiment's small preset")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab_phase7_"))
    out.mkdir(parents=True, exist_ok=True)
    p = run(out, small=args.small)
    show(p)
    print(f"report: {report(p).write_report(out / 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
