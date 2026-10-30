"""Tests for frictionlab.experiments.adverse_selection and examples/phase5_adverse_selection.py."""
import importlib.util
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frictionlab.agents import InformedTrader, NoiseTrader
from frictionlab.experiments import (
    AdverseSelectionConfig, GMMaker, adverse_selection_report, mitigation,
    run_adverse_selection, run_adverse_selection_study,
)
from frictionlab.experiments.adverse_selection import MM_ID, build

PATH = Path(__file__).resolve().parents[1] / "examples" / "phase5_adverse_selection.py"
CFG = AdverseSelectionConfig()
SHARES, JUMPS, REPS, SEED = (0.0, 0.2, 0.4), (2, 4, 8), 2, 30


def _load():
    spec = importlib.util.spec_from_file_location("phase5_adverse_selection", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def study():
    return run_adverse_selection_study(CFG, SHARES, JUMPS, replicates=REPS, master_seed=SEED)


def test_population_and_maker_premium():
    for share in (0.0, 0.3, 1.0):
        sc = build(CFG, {"share": share, "jump": 4}, 7)
        agents = list(sc.runner.agents.values())
        assert isinstance(agents[0], GMMaker) and agents[0].agent_id == MM_ID
        assert sum(isinstance(a, InformedTrader) for a in agents) == (share > 0)
        assert sum(isinstance(a, NoiseTrader) for a in agents) == (CFG.n_noise if share < 1 else 0)
        rate = sum(a.wake_rate for a in agents if a.agent_id != MM_ID)
        assert rate == pytest.approx(CFG.lam)              # total taker intensity held fixed
        gm = CFG.gm(share, 4)
        assert agents[0].p_ask + agents[0].p_bid == gm.spread


def test_ledger_invariants_hold_in_every_cell(study):
    assert len(study.rows) == len(SHARES) * len(JUMPS) * REPS and not study.errors
    mit = study.findings["mitigation_rows"]
    assert len(mit) == 2 * REPS
    for r in study.rows + mit:
        assert r["error"] is None
        assert r["violations"] == 0 and r["zero_sum_residual"] == 0
        assert r["identity_residual"] == 0                  # effective = realized + impact
        assert r["volume"] > 0 and r["mm_volume"] > 0


def test_maker_split_is_exact_in_integer_ticks(study):
    for r in study.rows + study.findings["mitigation_rows"]:
        cap, adv = r["mm_spread_capture"], r["mm_adverse_selection"]
        assert isinstance(cap, int) and isinstance(adv, int)
        assert cap + adv == r["mm_trading"] and r["split_residual"] == 0
        assert r["mm_as_loss"] == -adv


def test_quoted_spread_non_decreasing_in_informed_share(study):
    agg = study.aggregate()
    for j in JUMPS:
        col = [r["tw_spread_mean"] for r in agg if r["jump"] == j]
        assert all(b >= a for a, b in zip(col, col[1:])), (j, col)
        gm = [r["gm_spread_mean"] for r in agg if r["jump"] == j]
        assert all(s >= g for s, g in zip(col, gm))        # GM premium is a floor on the quote
    assert study.findings["spread_monotone_in_share"] is True


def test_price_impact_per_trade_rises_with_jump_size(study):
    f = study.findings
    imp = [f[f"impact_per_trade[J={j}]"] for j in JUMPS]
    assert all(b > a for a, b in zip(imp, imp[1:])), imp


def test_gm_findings(study):
    f = study.findings
    for j in JUMPS:
        assert f[f"gm_spread[mu=0,J={j}]"] == 0
        assert f[f"gm_spread[mu=0.4,J={j}]"] == CFG.gm(0.4, j).spread > 0


def test_mitigation_compares_structures(study):
    f = study.findings
    assert (f["mitigation_share"], f["mitigation_jump"]) == (max(SHARES), max(JUMPS))
    assert {r["structure"] for r in f["mitigation_rows"]} == {"continuous", "batch"}
    assert f["as_per_unit_batch_minus_continuous"] == pytest.approx(
        f["as_per_unit_batch"] - f["as_per_unit_continuous"])
    batch = [r for r in f["mitigation_rows"] if r["structure"] == "batch"]
    cont = [r for r in f["mitigation_rows"] if r["structure"] == "continuous"]
    assert [r["seed"] for r in batch] == [r["seed"] for r in cont]   # paired by CRN
    assert {r["share"] for r in batch} == {max(SHARES)}


def test_reproducible_from_master_seed(study):
    again = run_adverse_selection_study(CFG, SHARES, JUMPS, replicates=REPS, master_seed=SEED)
    assert again.rows == study.rows and again.findings == study.findings
    other = run_adverse_selection(CFG, SHARES, JUMPS, replicates=REPS, master_seed=SEED + 1)
    assert other.rows != study.rows


def test_serial_and_parallel_rows_identical(study):
    par = run_adverse_selection(CFG, SHARES, JUMPS, replicates=REPS, master_seed=SEED, n_workers=2)
    assert par.rows == study.rows
    mit = mitigation(CFG, max(SHARES), max(JUMPS), replicates=REPS, master_seed=SEED, n_workers=2)
    assert mit.rows == study.findings["mitigation_rows"]


def test_report_is_well_formed_and_deterministic(study, tmp_path):
    rep = adverse_selection_report(study)
    html = rep.to_html()
    assert html == adverse_selection_report(study).to_html()
    assert "http://" not in html and "https://" not in html
    for title in ("Adverse selection: findings", "Adverse selection: sweep",
                  "Adverse selection: batch vs continuous"):
        assert title in html
    body = html[html.index("<body"):html.index("</body>") + 7]
    ET.fromstring(body)
    assert Path(rep.write_report(tmp_path / "as.html")).stat().st_size > 0


def test_example_main_small_runs_fast(capsys, tmp_path):
    ex = _load()
    t0 = time.perf_counter()
    assert ex.main(["--small", "--report", str(tmp_path / "as.html")]) == 0
    assert time.perf_counter() - t0 < 20
    out = capsys.readouterr().out
    assert "GM benchmark" in out and "mitigation at mu=0.5" in out
    assert "non-decreasing in mu at every J: True" in out
    assert (tmp_path / "as.html").read_text().count("<section") == 3
