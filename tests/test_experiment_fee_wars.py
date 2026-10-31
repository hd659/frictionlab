"""Tests for frictionlab.experiments.fee_wars / two_venue and examples/phase5_fee_wars.py."""
import importlib.util
import sys
import time
from pathlib import Path

import pytest

from frictionlab.experiments import (
    FeeWarsConfig, LotFee, TwoVenueConfig, TwoVenueMarket, best_response, detect, fee_grid,
    fee_wars_report, probe_seeds, run_fee_wars, simulate_two_venue,
)

PATH = Path(__file__).resolve().parents[1] / "examples" / "phase5_fee_wars.py"
MKT = TwoVenueConfig(steps=120, epoch=20)
CFG = FeeWarsConfig(market=MKT, makers=(-2, -1, 0), takers=(1, 2, 3, 4), max_rounds=6,
                    replicates=1)
SEED = 32


def _load():
    if str(PATH.parent) not in sys.path:   # spawned sweep workers re-import by name
        sys.path.insert(0, str(PATH.parent))
    spec = importlib.util.spec_from_file_location("phase5_fee_wars", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def res():
    return run_fee_wars(CFG, master_seed=SEED)


def test_fee_grid_and_lot_fee_validation():
    g = fee_grid((-3, -1, 0, 2), (0, 1, 3))
    assert all(t >= 0 and m + t >= 0 for m, t in g)
    assert g == sorted(g, key=lambda p: (p[1], p[0])) and (-3, 3) in g and (-1, 0) not in g
    with pytest.raises(ValueError):
        LotFee(-3, 2)
    with pytest.raises(ValueError):
        LotFee(0, -1)


def test_shares_sum_to_one_every_round(res):
    rounds = res.findings["round_rows"]
    assert len(rounds) == res.findings["rounds"] + 1
    for r in rounds:
        assert r["share_A_mean"] + r["share_B_mean"] == pytest.approx(1.0)
    for r in res.rows:
        if r["volume"]:
            assert r["share_A"] + r["share_B"] == pytest.approx(1.0)


def test_revenue_plus_rebates_reconcile_with_sinks_exactly():
    for fees in ({"A": (-2, 3), "B": (0, 4)}, {"A": (-1, 1), "B": (-2, 2)}, {"A": (1, 0), "B": (0, 0)}):
        m = TwoVenueMarket(MKT, fees, 5)
        out = m.run()
        for v in ("A", "B"):
            led = m.venues[v].ledger
            assert out[f"revenue_{v}"] == led.venue.cash == led.fees_collected
            assert out[f"revenue_{v}"] + out[f"rebates_{v}"] == out[f"gross_fees_{v}"]
            assert led.venue.cash == sum(fees[v]) * out[f"volume_{v}"]
            assert out[f"rebates_{v}"] == max(-fees[v][0], 0) * out[f"volume_{v}"]
            assert led.venue.cash >= 0 and not led.violations()
        assert out["fee_residual"] == 0
        assert out["friction"] == out["revenue_A"] + out["revenue_B"]


def test_identical_schedules_split_flow_within_tolerance():
    for f in ((0, 4), (-2, 3), (0, 0)):
        out = simulate_two_venue(MKT, {"A": f, "B": f}, probe_seeds(SEED, 1)[0])
        assert abs(out["share_A"] - 0.5) <= 0.5
        assert out["volume_A"] > 0 and out["volume_B"] > 0


def test_unilateral_taker_cut_never_lowers_cutter_share():
    seeds = probe_seeds(SEED, 3)
    for seed in seeds:
        for rival in ((0, 4), (-2, 3), (0, 2)):
            for m, t in fee_grid((-2, -1, 0), (1, 2, 3, 4)):
                if m + t - 1 < 0:
                    continue
                before = simulate_two_venue(MKT, {"A": (m, t), "B": rival}, seed)["share_A"]
                after = simulate_two_venue(MKT, {"A": (m, t - 1), "B": rival}, seed)["share_A"]
                assert after >= before


def test_outcome_friction_not_above_start(res):
    f = res.findings
    assert f["outcome"] in ("fixed_point", "cycle")
    assert f["outcome_friction_per_lot"] <= f["start_fee_level"]
    assert f["round_rows"][0]["friction_per_lot_mean"] == f["start_fee_level"]


def test_best_response_tie_break_and_revenue():
    rows = [{"fee_A": (0, 3), "revenue_A": 10}, {"fee_A": (-1, 2), "revenue_A": 10},
            {"fee_A": (-2, 2), "revenue_A": 10}, {"fee_A": (0, 1), "revenue_A": 4}]
    assert best_response(rows, "A") == (-2, 2)
    rows.append({"fee_A": (0, 1), "revenue_A": 7})
    assert best_response(rows, "A") == (0, 1)


def test_detector_fixed_point_and_two_cycle():
    s, u = ((0, 3), (0, 3)), ((-1, 2), (0, 3))
    d = detect([s, s, s, s], window=2)
    assert (d.kind, d.round, d.cycle_length, d.states) == ("fixed_point", 2, 1, (s,))
    d = detect([((0, 4), (0, 4)), s, u, s, u], window=2)
    assert (d.kind, d.cycle_length, d.round, d.states) == ("cycle", 2, 3, (s, u))
    assert detect([s, u], window=2).kind == "none"
    with pytest.raises(ValueError):
        detect([s], window=0)


def test_ledger_invariants_every_cell(res):
    assert res.rows and not res.errors
    for r in res.rows:
        assert r["violations"] == 0 and r["fee_residual"] == 0
        assert r["revenue_A"] >= 0 and r["revenue_B"] >= 0


def test_serial_vs_parallel_and_reproducible(res):
    par = run_fee_wars(CFG, master_seed=SEED, n_workers=2)
    assert par.rows == res.rows and par.findings["path"] == res.findings["path"]
    again = run_fee_wars(CFG, master_seed=SEED)
    assert again.rows == res.rows and again.findings == res.findings


def test_report_has_sections(res, tmp_path):
    out = fee_wars_report(res).write_report(tmp_path / "fw.html")
    html = out.read_text()
    assert html.count("<section") == 5 and "share_A" in html and "outcome" in html


def test_example_main_small_runs_fast(capsys, tmp_path):
    ex = _load()
    t0 = time.perf_counter()
    assert ex.main(["--small", "--report", str(tmp_path / "fw.html")]) == 0
    assert time.perf_counter() - t0 < 20
    out = capsys.readouterr().out
    assert "friction per lot" in out and (tmp_path / "fw.html").exists()
