"""Fee-tier wars: two venues best-responding in (maker_fee, taker_fee) space.

Market: `two_venue.TwoVenueMarket` (two fl-031 books, one population, takers
routed by `VenueRouter`, makers picking venues by rebate net of fl-030
adverse-selection loss). Strategy space: integer-tick pairs (m, t) from
`fee_grid`, with t >= 0 and m + t >= 0 so a venue never pays more rebate
than it collects.

Best response, alternating within a round: holding the rival fixed, a venue
probes every pair through the fl-023 sweep engine (`run_experiment`) with
the same per-replicate child seeds at every probe and every round, so the
map state -> best response is deterministic. It picks the pair maximising
its summed sink revenue; ties go to the lower taker fee, then the lower
maker fee. Revenue is integer ticks, so the comparison is exact.

`detect(path, window)` reads the (fee_A, fee_B) state path: a pure fixed
point once the state is unchanged for `window` rounds, an Edgeworth-style
cycle (with its length) as soon as a state recurs after a different one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple

import numpy as np

from .base import ExperimentResult, run_experiment
from .two_venue import VENUES, Fees, MarketConfig, TwoVenueMarket

State = Tuple[Fees, Fees]
ROUND_METRICS = ("share", "volume", "revenue", "rebates", "spread", "makers")


def fee_grid(makers: Sequence[int], takers: Sequence[int]) -> List[Fees]:
    """Valid (maker, taker) pairs in (taker, maker) order: t >= 0, m + t >= 0."""
    return sorted(((int(m), int(t)) for m in makers for t in takers if t >= 0 and m + t >= 0),
                  key=lambda p: (p[1], p[0]))


@dataclass(frozen=True)
class Config:
    """Fee-wars setup: two-venue market, maker/taker fee grids, start state and stopping rule."""
    market: MarketConfig = field(default_factory=MarketConfig)
    makers: Tuple[int, ...] = (-2, -1, 0, 1)
    takers: Tuple[int, ...] = (0, 1, 2, 3, 4)
    start: State = ((0, 4), (0, 4))
    max_rounds: int = 12
    window: int = 2
    replicates: int = 2

    @property
    def grid(self) -> List[Fees]:
        return fee_grid(self.makers, self.takers)


class Detection(NamedTuple):
    """Outcome of best-response dynamics: fixed point, cycle (with length) or none."""
    kind: str                  # "fixed_point" | "cycle" | "none"
    round: int                 # round at which it fired (last round if none)
    cycle_length: int          # 1 for a fixed point, 0 for none
    states: Tuple[State, ...]  # the fixed state or one period of the cycle


def detect(path: Sequence[State], window: int = 2) -> Detection:
    """Fixed point after `window` unchanged rounds, else first recurrence."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    last: Dict[State, int] = {}
    run = 0
    for t, s in enumerate(path):
        if t and s == path[t - 1]:
            run += 1
            if run >= window:
                return Detection("fixed_point", t, 1, (s,))
        else:
            run = 0
            if s in last:
                return Detection("cycle", t, t - last[s], tuple(path[last[s]:t]))
        last[s] = t
    return Detection("none", max(len(path) - 1, 0), 0, ())


def factory(cfg: MarketConfig, params: Dict[str, Any], seed: int) -> TwoVenueMarket:
    return TwoVenueMarket(cfg, {v: params[f"fee_{v}"] for v in VENUES}, seed)


def probe_seeds(master_seed: int, replicates: int) -> List[int]:
    """`replicates` deterministic 64-bit seeds spawned from `master_seed`."""
    kids = np.random.SeedSequence(master_seed).spawn(replicates)
    return [int(k.generate_state(1, np.uint64)[0]) for k in kids]


def _sweep(cfg: Config, mover: str, pairs: Sequence[Fees], state: State, rnd: int,
           seeds: List[int], master_seed: int, n_workers: int) -> ExperimentResult:
    rival = VENUES[1 - VENUES.index(mover)]
    base = {"round": rnd, "mover": mover, f"fee_{rival}": state[VENUES.index(rival)]}
    return run_experiment("fee_wars", partial(factory, cfg.market), dict,
                          {f"fee_{mover}": list(pairs)}, base=base, seeds=seeds,
                          master_seed=master_seed, n_workers=n_workers)


def best_response(rows: Sequence[Dict[str, Any]], mover: str) -> Fees:
    """argmax summed revenue; ties to lower taker fee, then lower maker fee."""
    rev: Dict[Fees, int] = {}
    for r in rows:
        p = tuple(r[f"fee_{mover}"])
        rev[p] = rev.get(p, 0) + int(r[f"revenue_{mover}"])
    return max(rev, key=lambda p: (rev[p], -p[1], -p[0]))


def round_record(rows: Sequence[Dict[str, Any]], rnd: int, state: State) -> Dict[str, Any]:
    """Mean per-round metrics at `state`, keyed `<metric>_mean` (fl-026 ready)."""
    keys = [f"{m}_{v}" for m in ROUND_METRICS for v in VENUES] + [
        "friction", "friction_per_lot", "surplus", "net_surplus"]
    out: Dict[str, Any] = {"round": rnd, "fee_A": state[0], "fee_B": state[1]}
    for k in keys:
        out[f"{k}_mean"] = float(np.mean([r[k] for r in rows]))
    return out


def run_fee_wars(cfg: Config = Config(), *, master_seed: int = 32,
                 n_workers: int = 1) -> ExperimentResult:
    """Alternating best responses from `cfg.start` until `detect` fires."""
    seeds, grid = probe_seeds(master_seed, cfg.replicates), cfg.grid
    state: State = (tuple(cfg.start[0]), tuple(cfg.start[1]))
    first = _sweep(cfg, "A", [state[0]], state, 0, seeds, master_seed, n_workers)
    rows, path, records = list(first.rows), [state], [round_record(first.rows, 0, state)]
    det = detect(path, cfg.window)
    for rnd in range(1, cfg.max_rounds + 1):
        at = []
        for i, mover in enumerate(VENUES):
            res = _sweep(cfg, mover, grid, state, rnd, seeds, master_seed, n_workers)
            rows += res.rows
            pick = best_response(res.rows, mover)
            state = (pick, state[1]) if i == 0 else (state[0], pick)
            at = [r for r in res.rows if tuple(r[f"fee_{mover}"]) == pick]
        path.append(state)
        records.append(round_record(at, rnd, state))
        det = detect(path, cfg.window)
        if det.kind != "none":
            break
    end = records[det.round] if det.kind != "none" else records[-1]
    start_level = max(sum(f) for f in cfg.start)
    findings = {"outcome": det.kind, "detected_round": det.round,
                "cycle_length": det.cycle_length, "rounds": len(path) - 1,
                "final_fee_A": path[-1][0], "final_fee_B": path[-1][1],
                "start_fee_level": start_level,
                "outcome_friction_per_lot": end["friction_per_lot_mean"],
                "outcome_share_A": end["share_A_mean"],
                "outcome_net_surplus": end["net_surplus_mean"],
                "path": path, "round_rows": records, "cycle_states": det.states}
    params = {"round": list(range(len(path))), "mover": ["A", "B"], "fee_A": grid, "fee_B": grid}
    return ExperimentResult("fee_wars", int(master_seed), params, rows, findings)


def study_report(res: ExperimentResult):
    """fl-026 HTML: findings, share / revenue / spread paths by round."""
    from ..analysis.report import Report
    from ..analysis.report_sections import MetricsSection, SweepSection
    skip = ("path", "round_rows", "cycle_states")
    flat = {k: (str(v) if isinstance(v, tuple) else v)
            for k, v in res.findings.items() if k not in skip}
    path = [{k: (str(v) if isinstance(v, tuple) else v) for k, v in r.items()}
            for r in res.findings["round_rows"]]
    secs = [MetricsSection(extra=flat, title="Fee wars: findings")]
    secs += [SweepSection(path, ["round"], m, title=f"Fee wars: {m} by round")
             for m in ("share_A", "revenue_A", "revenue_B", "friction_per_lot")]
    return Report("frictionlab experiment: fee_wars", master_seed=res.master_seed,
                  grid={"fee pairs": [str(p) for p in res.params["fee_A"]]},
                  meta={"experiment": res.name}, sections=secs)
