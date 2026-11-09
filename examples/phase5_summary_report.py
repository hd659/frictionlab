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

`--include-phase8` appends the fl-053 Grossman-Stiglitz experiment
(`examples/configs/grossman_stiglitz_small.json`) as one more section, after the
Phase 7 pair when both flags are given. The default and `--include-phase7`
outputs are unchanged by it.

`--include-phase9` appends the fl-056 lemons experiment
(`examples/configs/lemons_small.json`) as one more section, after any Phase 7 and
Phase 8 sections. The default, `--include-phase7` and `--include-phase8` outputs are
unchanged by it.

`--include-phase10` appends the fl-059 reputation experiment
(`examples/configs/reputation_small.json`) as one more section, after every earlier
optional section. The default and `--include-phase7/8/9` outputs are unchanged by it.

`--include-phase11` appends the fl-062 switching-cost experiment
(`examples/configs/switching_cost_small.json`) as one more section, after every earlier
optional section. The default and `--include-phase7/8/9/10` outputs are unchanged by it.

`--include-phase12` appends the fl-065 critical-mass experiment
(`examples/configs/critical_mass_small.json`) as one more section, after every earlier
optional section. The default and `--include-phase7/8/9/10/11` outputs are unchanged by it.

`--include-phase13` appends the fl-068 congestion experiment
(`examples/configs/congestion_small.json`) as one more section, after every earlier
optional section. The default and `--include-phase7` .. `--include-phase12` outputs are
unchanged by it.

    python examples/phase5_summary_report.py [--out DIR] [--include-phase7] [--include-phase8]
                                             [--include-phase9] [--include-phase10]
                                             [--include-phase11] [--include-phase12]
                                             [--include-phase13]
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


def grossman_stiglitz() -> Spec:
    from frictionlab.experiments import grossman_stiglitz as gs, run_from_config
    raw = json.loads((HERE / "configs" / "grossman_stiglitz_small.json").read_text(encoding="utf-8"))
    res = run_from_config(raw)
    f = res.findings
    fp = ", ".join(f"{d['buyer_share']:.2f} vs {d['analytic_share']:.2f} at cost {d['signal_cost']}"
                   for d in f["population"])
    return (res, f"buyer fraction {f['cheapest_buyer_fraction']:.2f} -> {f['costliest_buyer_fraction']:.2f} "
                 f"over signal cost; information collapse at {f['collapse_threshold']} ticks (analytic "
                 f"{f['collapse_analytic']}); replicator vs analytic informed share {fp}",
            res.rows, [gs.COST, gs.NOISE], "buyer_fraction")


def lemons() -> Spec:
    from frictionlab.experiments import lemons as lm, run_from_config
    raw = json.loads((HERE / "configs" / "lemons_small.json").read_text(encoding="utf-8"))
    res = run_from_config(raw)
    f = res.findings
    ak = ", ".join(f"{p} at premium {b}" for b, p in sorted(f["akerlof_pooled_price"].items()))
    return (res, f"Akerlof pooled price {ak}; certification stops at {f['certification_threshold']} "
                 f"ticks (analytic bound {f['certification_bound']}) at premium "
                 f"{f['threshold_premium']}; {f['total_certification_paid']} ticks paid to certify",
            res.rows, [lm.COST, lm.PREMIUM], "surplus_share")


def reputation() -> Spec:
    from frictionlab.experiments import reputation as rp, run_from_config
    raw = json.loads((HERE / "configs" / "reputation_small.json").read_text(encoding="utf-8"))
    res = run_from_config(raw)
    f = res.findings
    return (res, f"cheating rate {f['lowest_accuracy_cheating_rate']:.2f} -> "
                 f"{f['highest_accuracy_cheating_rate']:.2f} over rating accuracy; whitewashing stops at "
                 f"{f['whitewash_threshold']} ticks (deterrent bound {f['deterrent_bound']}) at accuracy "
                 f"{f['threshold_accuracy']} bps; {f['total_whitewash_paid']} ticks paid to whitewash",
            res.rows, [rp.ACC, rp.WASH], "surplus_share")


def switching_cost() -> Spec:
    from frictionlab.experiments import run_from_config, switching_cost as sc
    raw = json.loads((HERE / "configs" / "switching_cost_small.json").read_text(encoding="utf-8"))
    res = run_from_config(raw)
    f = res.findings
    return (res, f"harvest markup up to {f['max_harvest_markup']:g} ticks over marginal cost in "
                 f"{f['lockin_cells']} lock-in cells; markup exceeds {f['min_markup']} ticks from mean "
                 f"switching cost {f['lockin_threshold']} (analytic {f['analytic_lockin_threshold']}) at "
                 f"target {f['threshold_target']} bps; consumers lose up to "
                 f"{f['max_cs_loss_vs_bertrand']} ticks against Bertrand",
            res.rows, [sc.MEAN, sc.TARGET], "harvest_markup")


def critical_mass() -> Spec:
    from frictionlab.experiments import run_from_config
    raw = json.loads((HERE / "configs" / "critical_mass_small.json").read_text(encoding="utf-8"))
    res = run_from_config(raw)
    f, p = res.findings, res.config["params"]
    return (res, f"{f['cold_start_traps']} cold-start traps without seeding; membership tips past "
                 f"{f['tipping_fraction']:g} of {p['n_a'] + p['n_b']} members from seed "
                 f"{f['tipping_seed']} (analytic critical mass {f['analytic_critical_mass']}) at "
                 f"adoption cost {f['tipping_cost']}; critical mass up to {f['max_critical_mass']}; "
                 f"{f['total_seed_subsidy']} ticks of seeding subsidy over the sweep",
            res.rows, ["adoption_cost", "seed_members"], "share")


def congestion() -> Spec:
    from frictionlab.experiments import run_from_config
    raw = json.loads((HERE / "configs" / "congestion_small.json").read_text(encoding="utf-8"))
    res = run_from_config(raw)
    f = res.findings
    return (res, f"arrivals over-join: individual threshold {f['individual_threshold']} vs social "
                 f"{f['social_threshold']} at waiting cost {f['probe_waiting_cost']}; optimal toll "
                 f"{f['bisected_toll']} (analytic {f['naor_optimal_toll']}); welfare gain up to "
                 f"{f['max_welfare_gain']} ticks at toll {f['best_toll']}; {f['total_toll_revenue']} "
                 f"ticks of toll revenue over the sweep",
            res.rows, ["toll", "waiting_cost"], "welfare")


def registry_config(cls: type, params: Dict[str, Any]) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in params.items() if k in names})


EXPERIMENTS: Tuple[Callable[[], Spec], ...] = (coase, decision_cost, adverse_selection, fee_wars, funding_arb)
PHASE7: Tuple[Callable[[], Spec], ...] = (search_cost, intermediation)
PHASE8: Tuple[Callable[[], Spec], ...] = (grossman_stiglitz,)
PHASE9: Tuple[Callable[[], Spec], ...] = (lemons,)
PHASE10: Tuple[Callable[[], Spec], ...] = (reputation,)
PHASE11: Tuple[Callable[[], Spec], ...] = (switching_cost,)
PHASE12: Tuple[Callable[[], Spec], ...] = (critical_mass,)
PHASE13: Tuple[Callable[[], Spec], ...] = (congestion,)


def build(include_phase7: bool = False, include_phase8: bool = False,
          include_phase9: bool = False, include_phase10: bool = False,
          include_phase11: bool = False, include_phase12: bool = False,
          include_phase13: bool = False) -> Report:
    runs = (EXPERIMENTS + (PHASE7 if include_phase7 else ()) + (PHASE8 if include_phase8 else ())
            + (PHASE9 if include_phase9 else ()) + (PHASE10 if include_phase10 else ())
            + (PHASE11 if include_phase11 else ()) + (PHASE12 if include_phase12 else ())
            + (PHASE13 if include_phase13 else ()))
    specs = [run() for run in runs]
    meta = {res.name: headline for res, headline, *_ in specs}
    seeds = {res.name: res.master_seed for res, *_ in specs}
    title = ("frictionlab Phase 5 summary: five friction experiments" if not include_phase7 else
             "frictionlab Phase 5 summary: five friction experiments plus Phase 7 search "
             "cost and intermediation")
    if include_phase8:
        title += " plus Phase 8 Grossman-Stiglitz"
    if include_phase9:
        title += " plus Phase 9 lemons"
    if include_phase10:
        title += " plus Phase 10 reputation"
    if include_phase11:
        title += " plus Phase 11 switching costs"
    if include_phase12:
        title += " plus Phase 12 critical mass"
    if include_phase13:
        title += " plus Phase 13 congestion"
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
    ap.add_argument("--include-phase8", action="store_true",
                    help="append the Phase 8 Grossman-Stiglitz experiment as one more section")
    ap.add_argument("--include-phase9", action="store_true",
                    help="append the Phase 9 lemons and certification experiment as one more section")
    ap.add_argument("--include-phase10", action="store_true",
                    help="append the Phase 10 reputation experiment as one more section")
    ap.add_argument("--include-phase11", action="store_true",
                    help="append the Phase 11 switching-cost experiment as one more section")
    ap.add_argument("--include-phase12", action="store_true",
                    help="append the Phase 12 critical-mass experiment as one more section")
    ap.add_argument("--include-phase13", action="store_true",
                    help="append the Phase 13 congestion experiment as one more section")
    a = ap.parse_args(argv)
    rep = build(a.include_phase7, a.include_phase8, a.include_phase9, a.include_phase10,
                a.include_phase11, a.include_phase12, a.include_phase13)
    for k, v in rep.meta.items():
        print(f"{k:>24}: {v}")
    out = a.out if a.out is not None else Path(tempfile.mkdtemp(prefix="frictionlab-phase5-"))
    print(f"report: {rep.write_report(out / 'phase5_summary.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
