"""Phase 5 example: adverse selection, simulated spreads vs the Glosten-Milgrom benchmark.

Part 1 prints the analytic GM zero-profit quotes for a two-point value
100 +- L over a grid of informed shares mu. Part 2 sweeps informed share x
fundamental jump size over a GM maker + informed + noise market (fl-023)
and prints, per cell, the time-weighted quoted spread next to the GM
spread, realized spread and price impact (fl-022), and the maker's trading
PnL split into spread capture and adverse-selection loss (exact in ticks).
Part 3 reruns the highest-share cell as continuous book vs frequent batch
auction (fl-020) and compares maker adverse-selection loss per unit.

    python examples/phase5_adverse_selection.py [--seed N] [--replicates R]
           [--workers N] [--small] [--report PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from frictionlab.experiments import (  # noqa: E402  (after sys.path tweak)
    AdverseSelectionConfig, ExperimentResult, adverse_selection_report, gm_zero_profit_spread,
    run_adverse_selection_study,
)

FULL = dict(shares=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5), jumps=(2, 4, 8), replicates=3)
SMALL = dict(shares=(0.0, 0.25, 0.5), jumps=(2, 8), replicates=2)
MUS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def gm_table(L: int = 10, v0: int = 100) -> str:
    lines = [f"GM benchmark, V in {{{v0 - L}, {v0 + L}}}, noise buys w.p. 1/2",
             f"{'mu':>6}{'E[V|sell]':>11}{'E[V|buy]':>10}{'bid':>6}{'ask':>6}{'spread':>8}"]
    for mu in MUS:
        q = gm_zero_profit_spread(mu, v0 - L, v0 + L)
        lines.append(f"{mu:>6.2f}{float(q.bid_exact):>11.2f}{float(q.ask_exact):>10.2f}"
                     f"{q.bid:>6}{q.ask:>6}{q.spread:>8}")
    return "\n".join(lines)


def grid_table(res: ExperimentResult) -> str:
    lines = [f"{'mu':>5}{'J':>4}{'quoted':>8}{'GM':>5}{'realized':>10}{'impact':>8}"
             f"{'capture':>10}{'AS loss':>10}{'AS/unit':>9}{'resid':>6}"]
    for r in res.aggregate():
        lines.append(f"{r['share']:>5.2f}{r['jump']:>4}{r['tw_spread_mean']:>8.2f}"
                     f"{r['gm_spread_mean']:>5.0f}{r['realized_spread_mean']:>10.2f}"
                     f"{r['price_impact_mean']:>8.2f}{r['mm_spread_capture_mean']:>10.1f}"
                     f"{r['mm_as_loss_mean']:>10.1f}{r['mm_as_per_unit_mean']:>9.2f}"
                     f"{r['split_residual_mean']:>6.0f}")
    return "\n".join(lines)


def mitigation_table(res: ExperimentResult) -> str:
    f = res.findings
    lines = [f"mitigation at mu={f['mitigation_share']:g}, J={f['mitigation_jump']}",
             f"{'structure':<12}{'quoted':>8}{'mm vol':>8}{'AS/unit':>9}{'snipe%':>8}"]
    for s in ("continuous", "batch"):
        rows = [r for r in f["mitigation_rows"] if r["structure"] == s]
        m = lambda k: sum(r[k] for r in rows) / len(rows)
        lines.append(f"{s:<12}{m('tw_spread'):>8.2f}{m('mm_volume'):>8.1f}"
                     f"{m('mm_as_per_unit'):>9.2f}{100 * m('snipe_share'):>8.1f}")
    lines.append(f"batch - continuous AS/unit: {f['as_per_unit_batch_minus_continuous']:+.2f} ticks")
    return "\n".join(lines)


def run(seed: int = 30, small: bool = False, replicates: Optional[int] = None,
        workers: int = 1, cfg: AdverseSelectionConfig = AdverseSelectionConfig()) -> ExperimentResult:
    g = SMALL if small else FULL
    return run_adverse_selection_study(cfg, g["shares"], g["jumps"],
                                       replicates=replicates or g["replicates"],
                                       master_seed=seed, n_workers=workers)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--small", action="store_true", help="3 shares x 2 jumps, 2 replicates")
    ap.add_argument("--report", type=Path, default=None, help="write an HTML report here")
    a = ap.parse_args(argv)
    cfg = AdverseSelectionConfig()
    print(gm_table())
    res = run(a.seed, a.small, a.replicates, a.workers, cfg)
    print(f"\nseed={a.seed} horizon={cfg.horizon} lam={cfg.lam} L=alpha*J alpha={cfg.alpha} "
          f"taker_bps={cfg.taker_bps}")
    print(grid_table(res))
    print(f"quoted spread non-decreasing in mu at every J: {res.findings['spread_monotone_in_share']}")
    print("\n" + mitigation_table(res))
    if a.report is not None:
        print(f"report: {adverse_selection_report(res).write_report(a.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
