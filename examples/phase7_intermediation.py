"""Phase 7 intermediation walkthrough: when a middleman's spread buys back search time, one report.

The fl-050 `intermediation` experiment is run through the public CLI entry point
`frictionlab.cli.main(argv)` on `examples/configs/intermediation_small.json`, into
OUT/run/ (rows.csv, resolved_config.json, report.html). From those artifacts it prints

1. intermediated share, volume, surplus share and the surplus change against the
   no-middleman twin over search cost x middleman half-spread;
2. per half-spread, the analytic search-cost bound (the spread is covered by the
   search time saved) and the bisected threshold where the share passes one half;
3. a middleman PnL view at the costliest cell: spread capture against holding cost
   and perishable decay as the per-step carry rises, with the break-even carry.

It then renders one fl-026 HTML report, OUT/report.html, with a SweepSection
heatmap (search cost x half-spread -> intermediated share) and a PnLSection
splitting the sweep's surplus into buyer gain, seller gain, middleman net,
search-cost sink, holding-cost sink and decay loss. Nothing time- or
path-dependent is written, so two runs are byte-identical.

    python examples/phase7_intermediation.py --small [OUT]     # OUT defaults to a temp dir
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a checkout

from frictionlab.analysis.pnl import middleman_pnl  # noqa: E402  (after sys.path tweak)
from frictionlab.analysis.report import Report  # noqa: E402
from frictionlab.analysis.report_sections import MetricsSection, PnLSection, SweepSection  # noqa: E402
from frictionlab.cli import main as cli_main  # noqa: E402
from frictionlab.experiments import intermediation as im  # noqa: E402

CONFIG = HERE / "configs" / "intermediation_small.json"
COST, SPREAD = im.COST, im.SPREAD
CARRY_VIEW = (0, 1, 2, 4, 8)          # holding cost = decay per step, in ticks
SPLIT_KEYS = ("buyer gain", "seller gain", "middleman net", "search-cost sink",
              "holding-cost sink", "decay loss")


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


def config_from(resolved: Dict[str, Any]) -> im.Config:
    """The experiment `Config` the CLI actually ran, rebuilt from resolved_config.json."""
    p = resolved["params"]
    return im.Config(**{f.name: tuple(p[f.name]) if isinstance(p[f.name], list) else p[f.name]
                        for f in fields(im.Config) if f.name in p})


def mean_by(rows: Sequence[Dict[str, Any]], key: str, **where: Any) -> float:
    return float(np.mean([r[key] for r in rows if all(r[k] == v for k, v in where.items())]))


def cell_market(cfg: im.Config, cost: int, spread: int, seed: int):
    return im.build(cfg, {COST: cost, SPREAD: spread}, seed).run(cfg.until).market


def surplus_split(cfg: im.Config, rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sweep totals in ticks; trader gains are rebuilt from each cell's fills on its seed."""
    out = dict.fromkeys(SPLIT_KEYS, 0)
    for r in rows:
        m = cell_market(cfg, r[COST], r[SPREAD], r["seed"])
        val, pnl = {i: a.value for i, a in m.agents.items()}, middleman_pnl(m)
        out["buyer gain"] += sum(val[t.buyer_id] - t.price for t in m.trades if t.buyer_id in val)
        out["seller gain"] += sum(t.price - val[t.seller_id] for t in m.trades if t.seller_id in val)
        out["middleman net"] += pnl.net
        out["search-cost sink"] -= m.search_costs
        out["holding-cost sink"] -= pnl.holding_cost
        out["decay loss"] -= pnl.decay_loss
    return out


def carry_view(cfg: im.Config, cost: int, spread: int, seed: int) -> List[Dict[str, int]]:
    """Middleman PnL split vs per-step carry (holding cost = decay) at one cell."""
    out = []
    for k in CARRY_VIEW:
        m = cell_market(replace(cfg, holding_cost=k, decay=k), cost, spread, seed)
        p = middleman_pnl(m)
        out.append({"carry": k, "sold": p.sold, "spread": p.spread_capture, "holding": p.holding_cost,
                    "decay": p.decay_loss, "net": p.net,
                    "peak": max((row[1] for row in m.carries), default=0)})
    return out


@dataclass
class Phase7:
    """Everything the example reports, for tests and for the Phase 5 summary."""
    rows: List[Dict[str, Any]]
    cfg: im.Config
    master_seed: int
    thresholds: List[im.IntermediationThreshold]
    carry: List[Dict[str, int]]
    split: Dict[str, int]
    view: Dict[str, int]

    @property
    def break_even_carry(self) -> Optional[int]:
        """Largest per-step carry at which the middleman still nets >= 0 in the PnL view."""
        return max((r["carry"] for r in self.carry if r["net"] >= 0), default=None)


def run(out: Path, small: bool = True) -> Phase7:
    """Run the experiment via the CLI into OUT/run and derive every printed number."""
    flags = ("--small",) if small else ()
    cli("validate", str(CONFIG), *flags)
    cli("run", str(CONFIG), *flags, "--out", str(out / "run"))
    rows = read_rows(out / "run" / "rows.csv")
    resolved = json.loads((out / "run" / "resolved_config.json").read_text(encoding="utf-8"))
    cfg, seed = config_from(resolved), int(resolved["master_seed"])
    spreads, costs = sorted({r[SPREAD] for r in rows}), sorted({r[COST] for r in rows})
    ths = [im.find_intermediation_threshold(cfg, h, 0.5, master_seed=seed) for h in spreads]
    view = {COST: costs[-1], SPREAD: spreads[min(1, len(spreads) - 1)], "seed": rows[0]["seed"]}
    carry = carry_view(cfg, view[COST], view[SPREAD], view["seed"])
    return Phase7(rows, cfg, seed, ths, carry, surplus_split(cfg, rows), view)


def report(p: Phase7) -> Report:
    """fl-026 report: thresholds, intermediated-share heatmap and the surplus split."""
    th = {f"bisected threshold (h {t.half_spread})": t.threshold for t in p.thresholds}
    th.update({f"analytic bound (h {t.half_spread})": t.analytic_bound for t in p.thresholds})
    th["break-even carry"] = p.break_even_carry
    return Report("frictionlab Phase 7: intermediation under search costs",
                  master_seed=p.master_seed, grid={COST: sorted({r[COST] for r in p.rows}),
                                                   SPREAD: sorted({r[SPREAD] for r in p.rows})},
                  meta={"config": CONFIG.name, "mid": p.cfg.mid, "ce max surplus":
                        im.competitive_equilibrium(p.cfg.values, p.cfg.costs).max_surplus},
                  sections=[MetricsSection(extra=th, title="Intermediation thresholds"),
                            SweepSection(p.rows, [COST, SPREAD], "intermediated_share",
                                         title="Intermediated share: search cost x half-spread"),
                            PnLSection(p.split, title="Surplus split over the sweep (ticks)")])


def show(p: Phase7) -> None:
    c = p.cfg
    print(f"intermediation sweep (seed {p.master_seed}, mid {c.mid}, rates direct "
          f"{c.meeting_rate:g} / middleman {c.middleman_meeting_rate:g}):")
    print(f"  {'cost':>4} {'h*':>3} {'h':>3} {'share':>6} {'vol':>4} {'surplus':>7} "
          f"{'d_surplus':>9} {'d_net':>6} {'mm_net':>6}")
    for cost in sorted({r[COST] for r in p.rows}):
        hstar = im.middleman_indifference_spread(cost, c.meeting_rate, c.middleman_meeting_rate)
        for h in sorted({r[SPREAD] for r in p.rows}):
            v = [mean_by(p.rows, k, **{COST: cost, SPREAD: h}) for k in
                 ("intermediated_share", "volume", "surplus_share", "surplus_change",
                  "net_surplus_change", "mm_net_pnl")]
            print(f"  {cost:>4} {hstar:>3} {h:>3} {v[0]:>6.2f} {v[1]:>4.0f} {v[2]:>7.2f} "
                  f"{v[3]:>9.0f} {v[4]:>6.0f} {v[5]:>6.0f}")
    for t in p.thresholds:
        print(f"threshold (h {t.half_spread}): bisected {t.threshold}, analytic bound "
              f"{t.analytic_bound}, within bound {t.within_bound}")
    print(f"middleman PnL view (search cost {p.view[COST]}, h {p.view[SPREAD]}, capacity "
          f"{c.capacity}): carry eats the spread")
    for r in p.carry:
        print(f"  carry {r['carry']:>2}: sold {r['sold']}  spread {r['spread']:>4}  holding "
              f"{r['holding']:>3}  decay {r['decay']:>3}  net {r['net']:>4}  peak inventory {r['peak']}")
    print(f"break-even carry: {p.break_even_carry}")
    print("surplus split: " + ", ".join(f"{k} {v}" for k, v in p.split.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", help="output directory (default: a new temp dir)")
    ap.add_argument("--small", action="store_true", help="use the experiment's small preset")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab_phase7im_"))
    out.mkdir(parents=True, exist_ok=True)
    p = run(out, small=args.small)
    show(p)
    print(f"report: {report(p).write_report(out / 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
