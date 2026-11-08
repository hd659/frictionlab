"""Phase 9 lemons walkthrough: how a hidden-quality pool unravels and who pays to escape it.

The fl-056 `lemons` experiment is run through the public CLI entry point
`frictionlab.cli.main(argv)` on `examples/configs/lemons_small.json`, into OUT/run/
(rows.csv, resolved_config.json, report.html). From those artifacts it prints

1. traded share, final pooled price and surplus share (of the full-information
   maximum) over certification cost x buyer premium;
2. the round-by-round unraveling path of the pooled price in the zero-premium cell
   with certification switched off, against `akerlof_pooled_price`;
3. the bisected certification threshold and its analytic bound.

It then renders one fl-026 HTML report, OUT/report.html, with a SweepSection heatmap
(certification cost x buyer premium -> surplus share), a line chart of the pooled-price
fixed-point iteration, and a PnLSection splitting realized surplus over the sweep into
buyer gain, pooled-seller gain, certified-seller gain and the certification sink.
Nothing time- or path-dependent is written, so two runs are byte-identical.

    python examples/phase9_lemons.py --small [OUT]   # OUT defaults to a temp dir
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a checkout

from frictionlab.analysis.plots import line_chart  # noqa: E402  (after sys.path tweak)
from frictionlab.analysis.report import Report  # noqa: E402
from frictionlab.analysis.report_sections import (  # noqa: E402
    MetricsSection, PnLSection, Section, SweepSection, inline_svg, table)
from frictionlab.cli import main as cli_main  # noqa: E402
from frictionlab.experiments import lemons as lm  # noqa: E402
from frictionlab.markets import CERTIFIED  # noqa: E402

CONFIG = HERE / "configs" / "lemons_small.json"
COST, PREMIUM = lm.COST, lm.PREMIUM
SPLIT_KEYS = ("buyer gain", "pooled-seller gain", "certified-seller gain", "certification sink")


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


def surplus_split(cfg: lm.Config, rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sweep totals in ticks, rebuilt per cell seed; the four parts sum to realized surplus."""
    out = dict.fromkeys(SPLIT_KEYS, 0)
    for r in rows:
        m = lm.build(cfg, {COST: r[COST], PREMIUM: r[PREMIUM]}, r["seed"]).run().market
        for t in m.trades:
            q = m.qualities[t.seller_id]
            out["buyer gain"] += q + m.premium - t.price
            seg = "certified" if m.segments[t.trade_id] == CERTIFIED else "pooled"
            out[f"{seg}-seller gain"] += t.price - q
        out["certification sink"] -= m.certification_costs
    return out


class PathSection(Section):
    """Line chart and table of the pooled-price fixed-point iteration."""
    kind = "path"

    def __init__(self, path: Sequence[Tuple[int, int, int]], title: str) -> None:
        super().__init__(title)
        self.path = list(path)                    # (round, price, units offered)

    def body(self) -> str:
        ks, ps = [float(k) for k, _, _ in self.path], [float(p) for _, p, _ in self.path]
        chart = line_chart(ps, ks, title="Pooled price by round", xlabel="round",
                           ylabel="price (ticks)", name="pooled price")
        return inline_svg(chart) + table(("round", "pooled price", "units offered"), self.path)


@dataclass
class Phase9:
    """Everything the example reports, for tests and for the Phase 5 summary."""
    rows: List[Dict[str, Any]]
    cfg: lm.Config
    params: Dict[str, Any]
    master_seed: int
    path: List[Tuple[int, int, int]]              # zero-premium, no-certification iteration
    akerlof: Tuple[int, int]                      # (fixed-point price, top quality offered)
    threshold: lm.CertificationThreshold
    split: Dict[str, int]

    def surplus_share(self, premium: int) -> float:
        """Surplus share at one buyer premium, averaged over certification costs."""
        return mean_by(self.rows, "surplus_share", **{PREMIUM: premium})


def run(out: Path, small: bool = True) -> Phase9:
    """Run the experiment via the CLI into OUT/run and derive every printed number."""
    flags = ("--small",) if small else ()
    cli("validate", str(CONFIG), *flags)
    cli("run", str(CONFIG), *flags, "--out", str(out / "run"))
    rows = read_rows(out / "run" / "rows.csv")
    resolved = json.loads((out / "run" / "resolved_config.json").read_text(encoding="utf-8"))
    p, seed = resolved["params"], int(resolved["master_seed"])
    cfg = lm.Config(**{f.name: tuple(p[f.name]) if isinstance(p[f.name], list) else p[f.name]
                       for f in fields(lm.Config) if f.name in p})
    p0 = min(p["buyer_premiums"])
    cell = next(r for r in rows if r[PREMIUM] == p0)
    m = lm.build(replace(cfg, certification=False), {COST: 0, PREMIUM: p0}, cell["seed"]).run().market
    path = [(r.k, r.price, len(r.offered)) for r in m.rounds]
    ak = lm.akerlof_pooled_price(list(m.qualities.values()), p0)
    th = lm.find_certification_threshold(cfg, p["threshold_premium"], p["uptake_fraction"],
                                         master_seed=seed)
    return Phase9(rows, cfg, p, seed, path, ak, th, surplus_split(cfg, rows))


def report(p: Phase9) -> Report:
    """fl-026 report: headline numbers, surplus heatmap, unraveling path, surplus split."""
    th = p.threshold
    head = {"Akerlof fixed-point price (premium 0)": p.akerlof[0],
            "top quality still offered (premium 0)": p.akerlof[1],
            "rounds to the fixed point": len(p.path),
            f"bisected certification threshold (premium {p.params['threshold_premium']})": th.threshold,
            "analytic certification bound": th.analytic, "uptake fraction": th.fraction}
    return Report("frictionlab Phase 9: lemons and certification",
                  master_seed=p.master_seed, grid={COST: p.params["certification_costs"],
                                                   PREMIUM: p.params["buyer_premiums"]},
                  meta={"config": CONFIG.name, "sellers": p.cfg.n_sellers,
                        "schedule": " ".join(map(str, p.cfg.schedule))},
                  sections=[MetricsSection(extra=head, title="Unraveling and certification"),
                            SweepSection(p.rows, [COST, PREMIUM], "surplus_share",
                                         title="Surplus share: certification cost x buyer premium"),
                            PathSection(p.path, "Pooled-price fixed point (premium 0, no certification)"),
                            PnLSection(p.split, title="Surplus split over the sweep (ticks)")])


def show(p: Phase9) -> None:
    print(f"lemons sweep (seed {p.master_seed}, {p.cfg.n_sellers} sellers, "
          f"schedule {p.cfg.schedule[0]}..{p.cfg.schedule[-1]}):")
    print(f"  {'cost':>4} {'prem':>4} {'traded':>6} {'price':>5} {'cert':>4} {'cutoff':>6} {'surplus':>7}")
    for c in p.params["certification_costs"]:
        for b in p.params["buyer_premiums"]:
            r = next(r for r in p.rows if r[COST] == c and r[PREMIUM] == b)
            print(f"  {c:>4} {b:>4} {r['traded_share']:>6.2f} {r['pooled_price']:>5} "
                  f"{r['certified']:>4} {r['cutoff']:>6} {r['surplus_share']:>7.2f}")
    print("unraveling (premium 0, no certification): "
          + " -> ".join(f"{pr} ({n} offered)" for _, pr, n in p.path))
    print(f"akerlof fixed point: price {p.akerlof[0]}, top quality offered {p.akerlof[1]}")
    th = p.threshold
    print(f"certification threshold (premium {p.params['threshold_premium']}): bisected "
          f"{th.threshold}, analytic {th.analytic} (uptake < {th.fraction:g})")
    print("surplus split: " + ", ".join(f"{k} {v}" for k, v in p.split.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", help="output directory (default: a new temp dir)")
    ap.add_argument("--small", action="store_true", help="use the experiment's small preset")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab_phase9_"))
    out.mkdir(parents=True, exist_ok=True)
    p = run(out, small=args.small)
    show(p)
    print(f"report: {report(p).write_report(out / 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
