"""fl-035: examples/phase5_funding_arb.py (seeded funding-arb example + HTML report)."""
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab.experiments.funding_arb import LEVEL, PROB

PATH = Path(__file__).resolve().parents[1] / "examples" / "phase5_funding_arb.py"
BUDGET_S = 60.0


def load():
    if "phase5_funding_arb" in sys.modules:
        return sys.modules["phase5_funding_arb"]
    spec = importlib.util.spec_from_file_location("phase5_funding_arb", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase5_funding_arb"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ex():
    return load()


@pytest.fixture(scope="module")
def study(ex):
    return ex.run(34, small=True)


def parse(doc: str) -> ET.Element:
    assert doc.startswith("<!DOCTYPE html>\n")
    root = ET.fromstring(doc.split("\n", 1)[1])
    assert root.tag == "html"
    return root


def test_main_small_runs_within_budget_and_writes_report(ex, capsys, tmp_path):
    t0 = time.perf_counter()
    assert ex.main(["--small", "--out", str(tmp_path)]) == 0
    assert time.perf_counter() - t0 < BUDGET_S
    out = capsys.readouterr().out
    for needle in ("entries |prem|", "analytic per-leg bound", "bisected collapse level",
                   "regime view", "held", "unwound", "report:"):
        assert needle in out
    assert (tmp_path / "funding_arb.html").is_file()


def test_main_defaults_to_a_temp_dir(ex, capsys):
    assert ex.main(["--small"]) == 0
    path = Path(capsys.readouterr().out.strip().splitlines()[-1].split("report: ", 1)[1])
    assert path.is_file() and path.name == "funding_arb.html"


def test_reproducible_from_master_seed(ex, study):
    again = ex.run(34, small=True)
    assert again.res.rows == study.res.rows
    assert again.res.findings == study.res.findings
    assert again.regimes == study.regimes
    assert ex.report(again).to_html() == ex.report(study).to_html()


def test_premium_non_decreasing_in_shock_gas(study):
    res = study.res
    assert res.findings["premium_monotone_in_gas"] is True
    agg = res.aggregate(["mean_abs_premium", "entries"])
    for p in res.params[PROB]:
        col = sorted((r for r in agg if r[PROB] == p), key=lambda r: r[LEVEL])
        prem = [r["mean_abs_premium_mean"] for r in col]
        assert all(b >= a for a, b in zip(prem, prem[1:])), (p, prem)
        if p == 0.0:   # no shocks: every level is the constant-gas baseline
            assert len(set(prem)) == 1


def test_thresholds_and_regime_view_are_consistent(study):
    th, f = study.threshold, study.res.findings
    assert f["gas_threshold_analytic"] == th.analytic_bound
    assert f["gas_threshold_bisected"] == th.threshold and th.within_bound
    assert th.probes[th.threshold] < th.target <= th.probes[th.threshold - 1]
    for r in study.regimes:
        assert r["calm_steps"] + r["shock_steps"] == 200
        assert r["entries_calm"] + r["entries_shock"] == r["metrics"]["entries"]
        assert r["held_through_shock"] + r["unwound_in_shock"] == r["exposed"]
        if r[LEVEL] > th.analytic_bound:      # no entry is rational in a shock step
            assert r["entries_shock"] == 0
    assert f["entries_per_shock_step"] == 0.0 and f["entries_per_calm_step"] > 0


def test_friction_breakdown_reconciles(ex, study):
    for r in study.pnl.values():
        m = r["metrics"]
        net = (m["gross_funding"] + m["basis_pnl"] - m["perp_fee"] - m["amm_cost"]
               - m["gas"] - m["decision_cost"])
        assert abs(net - m["net_pnl"]) < 1e-6
        assert abs(ex.breakdown(r).net - m["net_pnl"]) < 1e-6


def test_report_is_wellformed_offline_html(ex, study):
    doc = ex.report(study).to_html()
    root = parse(doc)
    assert not re.search(r"https?://", doc, re.I)
    assert not list(root.iter("script"))
    kinds = [s.get("class") for s in root.iter("section")]
    assert kinds.count("sweep") == 1 and kinds.count("pnl") == 1
    sweep = next(s for s in root.iter("section") if s.get("class") == "sweep")
    assert "shock_gas_level x shock_probability" in "".join(sweep.itertext())
    assert "gas" in "".join(next(s for s in root.iter("section") if s.get("class") == "pnl").itertext())
