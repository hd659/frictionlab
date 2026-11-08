"""Phase 10 reputation walkthrough: end-game milking, cheap identities and the deterrent cost.

The fl-059 `reputation` experiment is run through the public CLI entry point
`frictionlab.cli.main(argv)` on `examples/configs/reputation_small.json`, into OUT/run/
(rows.csv, resolved_config.json, report.html). From those artifacts it prints

1. cheating rate, whitewash count and surplus share (of the full-information
   maximum) over rating accuracy x whitewash cost;
2. one opportunist's public score and posted price round by round, with its
   milking round marked, against the analytic `milking_switch_round`;
3. the bisected whitewash threshold and the analytic deterrent bound.

It then renders one fl-026 HTML report, OUT/report.html, with a SweepSection heatmap
(rating accuracy x whitewash cost -> surplus share), a line chart of price vs the buyer
value of the delivered quality for the traced seller, and a PnLSection splitting
realized surplus over the sweep into buyer gain, honest-seller gain, opportunist
cheating gain and the identity sink. Two runs are byte-identical.

    python examples/phase10_reputation.py --small [OUT]   # OUT defaults to a temp dir
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a checkout
sys.path.insert(0, str(HERE))

from frictionlab.analysis.plots import multi_line_chart  # noqa: E402  (after sys.path tweak)
from frictionlab.analysis.report import Report  # noqa: E402
from frictionlab.analysis.report_sections import (  # noqa: E402
    MetricsSection, PnLSection, Section, SweepSection, inline_svg, table)
from frictionlab.experiments import reputation as rp  # noqa: E402
from phase9_lemons import cli, mean_by, read_rows  # noqa: E402  (shared CLI/CSV helpers)

CONFIG = HERE / "configs" / "reputation_small.json"
ACC, WASH = rp.ACC, rp.WASH
SPLIT_KEYS = ("buyer gain", "honest-seller gain", "opportunist cheating gain", "identity sink")
# (round, score, price, delivered quality, buyer value of delivery, rating, milking mark)
Step = Tuple[int, str, int, Optional[int], Optional[int], str, str]


def surplus_split(cfg: rp.Config, rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sweep totals in ticks, rebuilt per cell seed; the four parts sum to realized surplus.

    Opportunists' bucket is price minus delivered cost: trade surplus plus the cheating saving.
    """
    out = dict.fromkeys(SPLIT_KEYS, 0)
    for r in rows:
        m = rp.build(cfg, {ACC: r[ACC], WASH: r[WASH]}, r["seed"]).run().market
        for t in m.trades:
            d = m.delivered[t.trade_id]
            out["buyer gain"] += m.buyer_value(d) - t.price
            key = "opportunist cheating gain" if m.sellers[t.seller_id].opportunist else "honest-seller gain"
            out[key] += t.price - d
        out["identity sink"] -= m.whitewash_costs
    return out


def trace(m: Any) -> Tuple[int, List[Step]]:
    """First opportunist (by id) that milks: its round-by-round score and price path."""
    s = next(s for s in m.sellers.values() if s.opportunist and s.milking_round is not None)
    steps = [(r.t, f"{r.score[0]}/{r.score[1]}", r.price, r.delivered,
              None if r.delivered is None else m.buyer_value(r.delivered),
              "-" if r.rating is None else ("good" if r.rating else "bad"),
              ("milk" if r.t == s.milking_round else "") + (" wash" if r.whitewashed else ""))
             for r in m.history(s.agent_id)]
    return s.agent_id, steps


class PathSection(Section):
    """Line chart and table of the traced opportunist's price vs delivered value."""
    kind = "path"

    def __init__(self, steps: Sequence[Step], title: str) -> None:
        super().__init__(title)
        self.steps = list(steps)

    def body(self) -> str:
        xs = [float(s[0]) for s in self.steps]
        chart = multi_line_chart({"price": [float(s[2]) for s in self.steps],
                                  "delivered value": [float(s[4] or 0) for s in self.steps]},
                                 xs, title="Price vs delivered-quality value", xlabel="round",
                                 ylabel="ticks")
        hdr = ("round", "score good/bad", "price", "delivered q", "delivered value", "rating", "mark")
        return inline_svg(chart) + table(hdr, self.steps)


@dataclass
class Phase10:
    """Everything the example reports, for tests and for the Phase 5 summary."""
    rows: List[Dict[str, Any]]
    cfg: rp.Config
    params: Dict[str, Any]
    master_seed: int
    seller: int                                  # traced opportunist id
    steps: List[Step]
    milking_round: int                           # simulated, traced seller
    analytic_round: int                          # milking_switch_round at perfect accuracy
    threshold: rp.ReputationThreshold
    split: Dict[str, int]

    def surplus_share(self, accuracy: int) -> float:
        """Surplus share at one rating accuracy, averaged over whitewash costs."""
        return mean_by(self.rows, "surplus_share", **{ACC: accuracy})


def run(out: Path, small: bool = True) -> Phase10:
    """Run the experiment via the CLI into OUT/run and derive every printed number."""
    import json
    flags = ("--small",) if small else ()
    cli("validate", str(CONFIG), *flags)
    cli("run", str(CONFIG), *flags, "--out", str(out / "run"))
    rows = read_rows(out / "run" / "rows.csv")
    resolved = json.loads((out / "run" / "resolved_config.json").read_text(encoding="utf-8"))
    p, seed = resolved["params"], int(resolved["master_seed"])
    cfg = rp.Config(tuple(p["schedule"]), p["n_sellers"], p["horizon"], p["premium"], p["opportunist_bps"])
    acc, c0 = p["threshold_accuracy"], min(p["whitewash_costs"])
    cell = next(r for r in rows if r[ACC] == acc and r[WASH] == c0)
    m = rp.build(cfg, {ACC: acc, WASH: c0}, cell["seed"]).run().market
    sid, steps = trace(m)
    perfect = rp.build(cfg, {ACC: rp.BPS, WASH: c0}, cell["seed"]).market
    th = rp.find_reputation_threshold(cfg, acc, master_seed=seed)
    return Phase10(rows, cfg, p, seed, sid, steps, m.sellers[sid].milking_round,
                   rp.predicted_milking_round(perfect, sid), th, surplus_split(cfg, rows))


def report(p: Phase10) -> Report:
    """fl-026 report: headline numbers, surplus heatmap, traced price path, surplus split."""
    th = p.threshold
    head = {f"traced opportunist (seller {p.seller}) milking round": p.milking_round,
            "analytic milking round (perfect accuracy)": p.analytic_round,
            f"bisected whitewash threshold (accuracy {th.accuracy} bps)": th.threshold,
            "analytic deterrent whitewash bound": th.analytic, "horizon": p.cfg.horizon}
    return Report("frictionlab Phase 10: reputation, milking and whitewashing",
                  master_seed=p.master_seed, grid={ACC: p.params["rating_accuracies"],
                                                   WASH: p.params["whitewash_costs"]},
                  meta={"config": CONFIG.name, "sellers": p.cfg.n_sellers,
                        "opportunist share (bps)": p.cfg.opportunist_bps, "premium": p.cfg.premium},
                  sections=[MetricsSection(extra=head, title="Milking and whitewashing"),
                            SweepSection(p.rows, [ACC, WASH], "surplus_share",
                                         title="Surplus share: rating accuracy x whitewash cost"),
                            PathSection(p.steps, f"Seller {p.seller}: price vs delivered value "
                                                 f"(accuracy {th.accuracy} bps)"),
                            PnLSection(p.split, title="Surplus split over the sweep (ticks)")])


def show(p: Phase10) -> None:
    print(f"reputation sweep (seed {p.master_seed}, {p.cfg.n_sellers} sellers, "
          f"horizon {p.cfg.horizon}, opportunists {p.cfg.opportunist_bps} bps):")
    print(f"  {'acc':>5} {'wash':>4} {'cheat':>5} {'washes':>6} {'surplus':>7}")
    for a in p.params["rating_accuracies"]:
        for c in p.params["whitewash_costs"]:
            r = next(r for r in p.rows if r[ACC] == a and r[WASH] == c)
            print(f"  {a:>5} {c:>4} {r['cheating_rate']:>5.2f} {r['whitewashes']:>6} "
                  f"{r['surplus_share']:>7.2f}")
    print(f"seller {p.seller} (accuracy {p.threshold.accuracy} bps, cheapest whitewash):")
    for t, score, price, d, v, rating, mark in p.steps:
        print(f"  round {t}: score {score} price {price} delivered {d} ({rating}) {mark}".rstrip())
    print(f"milking round: simulated {p.milking_round}, analytic {p.analytic_round}")
    th = p.threshold
    print(f"whitewash threshold (accuracy {th.accuracy}): bisected {th.threshold}, "
          f"analytic deterrent bound {th.analytic}")
    print("surplus split: " + ", ".join(f"{k} {v}" for k, v in p.split.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", help="output directory (default: a new temp dir)")
    ap.add_argument("--small", action="store_true", help="use the experiment's small preset")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="frictionlab_phase10_"))
    out.mkdir(parents=True, exist_ok=True)
    p = run(out, small=args.small)
    show(p)
    print(f"report: {report(p).write_report(out / 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
