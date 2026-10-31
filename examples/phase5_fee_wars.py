"""Phase 5 example: fee-tier wars between two venues.

Two continuous-book venues share one population. ZI-C, noise and informed
takers route by all-in cost through the fl-031 VenueRouter; market makers
move each epoch to the venue with the best rebate net of measured
adverse-selection loss. Each venue picks an integer-tick (maker_fee,
taker_fee) pair by best response to the rival, round by round, until the
detector reports a pure fixed point or an Edgeworth-style cycle.

    python examples/phase5_fee_wars.py [--seed N] [--replicates R]
           [--workers N] [--small] [--report PATH]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from frictionlab.experiments import (  # noqa: E402  (after sys.path tweak)
    ExperimentResult, FeeWarsConfig, TwoVenueConfig, fee_wars_report, run_fee_wars,
)

FULL = FeeWarsConfig()
SMALL = FeeWarsConfig(market=TwoVenueConfig(steps=120, epoch=20), makers=(-2, -1, 0),
                      takers=(1, 2, 3, 4), start=((0, 4), (0, 4)), max_rounds=6,
                      window=2, replicates=1)


def path_table(res: ExperimentResult) -> str:
    lines = [f"{'rnd':>3}  {'fee_A':>8}  {'fee_B':>8}{'share_A':>9}{'rev_A':>8}{'rev_B':>8}"
             f"{'reb_A':>7}{'reb_B':>7}{'spr_A':>7}{'spr_B':>7}{'mm_A':>6}{'fric/lot':>9}"
             f"{'net_surp':>10}"]
    for r in res.findings["round_rows"]:
        lines.append(f"{r['round']:>3}  {str(r['fee_A']):>8}  {str(r['fee_B']):>8}"
                     f"{r['share_A_mean']:>9.3f}{r['revenue_A_mean']:>8.1f}"
                     f"{r['revenue_B_mean']:>8.1f}{r['rebates_A_mean']:>7.1f}"
                     f"{r['rebates_B_mean']:>7.1f}{r['spread_A_mean']:>7.2f}"
                     f"{r['spread_B_mean']:>7.2f}{r['makers_A_mean']:>6.2f}"
                     f"{r['friction_per_lot_mean']:>9.3f}{r['net_surplus_mean']:>10.1f}")
    return "\n".join(lines)


def summary(res: ExperimentResult) -> str:
    f = res.findings
    if f["outcome"] == "fixed_point":
        head = f"pure fixed point at round {f['detected_round']}: A={f['final_fee_A']} B={f['final_fee_B']}"
    elif f["outcome"] == "cycle":
        head = (f"Edgeworth cycle of length {f['cycle_length']} detected at round "
                f"{f['detected_round']}: {' -> '.join(map(str, f['cycle_states']))}")
    else:
        head = f"no fixed point or cycle within {f['rounds']} rounds"
    return (f"{head}\nfriction per lot: start {f['start_fee_level']} -> outcome "
            f"{f['outcome_friction_per_lot']:.3f} ticks; outcome share_A "
            f"{f['outcome_share_A']:.3f}; outcome net surplus {f['outcome_net_surplus']:.1f}")


def run(seed: int = 32, small: bool = False, replicates: Optional[int] = None,
        workers: int = 1) -> ExperimentResult:
    cfg = SMALL if small else FULL
    if replicates is not None:
        cfg = replace(cfg, replicates=replicates)
    return run_fee_wars(cfg, master_seed=seed, n_workers=workers)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--small", action="store_true", help="12-point grid, 120 steps, 1 replicate")
    ap.add_argument("--report", type=Path, default=None, help="write an HTML report here")
    a = ap.parse_args(argv)
    cfg = SMALL if a.small else FULL
    res = run(a.seed, a.small, a.replicates, a.workers)
    print(f"seed={a.seed} steps={cfg.market.steps} epoch={cfg.market.epoch} "
          f"makers={cfg.market.n_makers} grid={len(cfg.grid)} pairs "
          f"(maker {list(cfg.makers)} x taker {list(cfg.takers)}), start={cfg.start}")
    print(path_table(res))
    print(summary(res))
    if a.report is not None:
        print(f"report: {fee_wars_report(res).write_report(a.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
