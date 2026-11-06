"""Phase 5 summary: rerun all five experiments (--small) into one offline HTML report.

Coase threshold, decision cost, adverse selection, fee wars and funding
arbitrage are rerun from their example modules with each one's own master
seed. The report header carries one headline line per experiment, and the
body holds exactly one `ExperimentSection` per `ExperimentResult`: a findings
table (scalars only) plus an fl-023 sweep table and chart of its key metric.
Nothing time-dependent is written, so two runs give byte-identical HTML.

`--include-phase7` appends the two Phase 7 experiments as sections six and
seven: fl-047 search cost (`examples/configs/search_cost_small.json`) and fl-050
intermediation (`examples/configs/intermediation_small.json`). Without it the
five-section report is byte-identical to what it was before Phase 7.

    python examples/phase5_summary_report.py [--out DIR] [--include-phase7]
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import numbers
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a checkout

from frictionlab.analysis.report import Report  # noqa: E402  (after sys.path tweak)
from frictionlab.analysis.report_sections import MetricsSection, Section, SweepSection, esc  # noqa: E402
from frictionlab.experiments import ExperimentResult  # noqa: E402


def load(name: str) -> ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def scalars(findings: Dict[str, Any]) -> Dict[str, Any]:
    """Flat, report-safe findings: numbers, bools and strings; tuples as text."""
    out: Dict[str, Any] = {}
    for k, v in findings.items():
        if isinstance(v, (numbers.Number, str)) or v is None:
            out[k] = v
        elif isinstance(v, tuple) and len(str(v)) <= 60:
            out[k] = str(v)
    return out


class ExperimentSection(Section):
    """One Phase 5 experiment: headline, flat findings and its key sweep."""
    kind = "experiment"

    def __init__(self, res: ExperimentResult, headline: str, rows: Sequence[Dict[str, Any]],
                 by: Sequence[str], metric: str) -> None:
        super().__init__(f"{res.name} (seed {res.master_seed})")
        self.res, self.headline = res, headline
        self.findings = MetricsSection(extra=scalars(res.findings))
        self.sweep = SweepSection(rows, by, metric)

    def body(self) -> str:
        return (f'<p class="headline">{esc(self.headline)}</p><h3>Findings</h3>{self.findings.body()}'
                f"<h3>Sweep: {esc(self.sweep.metric)} by {esc(', '.join(self.sweep.by))}</h3>"
                f"{self.sweep.body()}")


Spec = Tuple[ExperimentResult, str, Sequence[Dict[str, Any]], Sequence[str], str]


def coase() -> Spec:
    ex = load("phase5_coase_threshold")
    res, th = ex.run(ex.SMALL, 28, 3, 0.5, 1)
    f = res.findings
    return (res, f"half-volume collapse at F* = {f['coase_threshold']} ticks "
                 f"(analytic bound {f['analytic_bound']}, zero volume from {f['first_zero_volume_level']})",
            res.rows, res.param_keys, "volume")


def decision_cost() -> Spec:
    res = load("phase5_decision_cost").run(True, 29)
    be = ", ".join(f"c*={b['cost']} (analytic {b['analytic']}) at aspiration {b['aspiration']:g}"
                   for b in res.findings["break_even"])
    asp = res.params["aspiration"][0]          # 3-axis grid: chart one aspiration slice
    rows = [r for r in res.rows if r["aspiration"] == asp]
    return res, f"satisficer beats optimizer above {be}", rows, ["cost_per_step", "mode"], "net"


def adverse_selection() -> Spec:
    res = load("phase5_adverse_selection").run(30, True)
    f = res.findings
    return (res, f"quoted spread monotone in informed share: {f['spread_monotone_in_share']}; batch "
                 f"auction AS loss per unit {f['as_per_unit_continuous']:.2f} -> {f['as_per_unit_batch']:.2f}",
            res.rows, ["share", "jump"], "tw_spread")


def fee_wars() -> Spec:
    res = load("phase5_fee_wars").run(32, True)
    f = res.findings
    rows = [{k: (str(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in f["round_rows"]]
    return (res, f"{f['outcome']} at round {f['detected_round']}: A={f['final_fee_A']} B={f['final_fee_B']}, "
                 f"friction per lot {f['start_fee_level']} -> {f['outcome_friction_per_lot']:g}",
            rows, ["round"], "share_A")


def funding_arb() -> Spec:
    st = load("phase5_funding_arb").run(34, True)
    res, f = st.res, st.res.findings
    return (res, f"entries {f['baseline_entries']:g} -> {f['shock_entries']:g} and mean |premium| "
                 f"{f['baseline_mean_abs_premium']:.1f} -> {f['shock_mean_abs_premium']:.1f} bps under shocks; "
                 f"gas threshold {f['gas_threshold_bisected']} (analytic bound {f['gas_threshold_analytic']})",
            res.rows, res.param_keys, "mean_abs_premium")


def search_cost() -> Spec:
    from frictionlab.experiments import find_search_collapse, run_from_config, search_cost as sc
    raw = json.loads((HERE / "configs" / "search_cost_small.json").read_text(encoding="utf-8"))
    res = run_from_config(raw)
    cfg = registry_config(sc.Config, res.config["params"])
    col = find_search_collapse(cfg, 0.5, 0, master_seed=res.master_seed)
    f = res.findings
    return (res, f"volume {f['baseline_volume']:g} -> {f['costliest_volume']:g} and surplus share "
                 f"{f['baseline_surplus_share']:.2f} -> {f['costliest_surplus_share']:.2f} over search cost; "
                 f"half-volume collapse at {col.threshold} ticks (analytic bound {col.analytic_bound})",
            res.rows, [sc.COST, sc.DECAY], "surplus_share")


def intermediation() -> Spec:
    from frictionlab.experiments import (find_intermediation_threshold, intermediation as im,
                                         run_from_config)
    raw = json.loads((HERE / "configs" / "intermediation_small.json").read_text(encoding="utf-8"))
    res = run_from_config(raw)
    cfg, f = registry_config(im.Config, res.config["params"]), res.findings
    h = sorted(res.params[im.SPREAD])[min(1, len(res.params[im.SPREAD]) - 1)]
    th = find_intermediation_threshold(cfg, h, 0.5, master_seed=res.master_seed)
    return (res, f"intermediated share {f['baseline_intermediated_share']:.2f} at zero spread; at "
                 f"half-spread {h} it passes 0.5 at search cost {th.threshold} (analytic bound "
                 f"{th.analytic_bound}); middleman net {f['costliest_widest_mm_net_pnl']:g} ticks "
                 f"at the costliest, widest cell",
            res.rows, [im.COST, im.SPREAD], "intermediated_share")


def registry_config(cls: type, params: Dict[str, Any]) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in params.items() if k in names})


EXPERIMENTS: Tuple[Callable[[], Spec], ...] = (coase, decision_cost, adverse_selection, fee_wars, funding_arb)
PHASE7: Tuple[Callable[[], Spec], ...] = (search_cost, intermediation)


def build(include_phase7: bool = False) -> Report:
    specs = [run() for run in EXPERIMENTS + (PHASE7 if include_phase7 else ())]
    meta = {res.name: headline for res, headline, *_ in specs}
    seeds = {res.name: res.master_seed for res, *_ in specs}
    title = ("frictionlab Phase 5 summary: five friction experiments" if not include_phase7 else
             "frictionlab Phase 5 summary: five friction experiments plus Phase 7 search "
             "cost and intermediation")
    return Report(title,
                  master_seed=None, grid={"experiment": list(seeds)},
                  meta={"mode": "--small", **{f"seed {k}": v for k, v in seeds.items()}, **meta},
                  sections=[ExperimentSection(*s) for s in specs])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory for phase5_summary.html (default: a fresh temp dir)")
    ap.add_argument("--include-phase7", action="store_true",
                    help="append the Phase 7 search-cost and intermediation experiments "
                         "as sections six and seven")
    a = ap.parse_args(argv)
    rep = build(a.include_phase7)
    for k, v in rep.meta.items():
        print(f"{k:>24}: {v}")
    out = a.out if a.out is not None else Path(tempfile.mkdtemp(prefix="frictionlab-phase5-"))
    print(f"report: {rep.write_report(out / 'phase5_summary.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
