"""Regression tests for examples/phase1_friction_walkthrough.py.

The example is the Phase 1 integration test: clock -> book -> friction -> ledger.
These tests pin determinism and the PnL accounting identities it documents.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from frictionlab.costs import FixedCost, FrictionStack, LatencyCost, MakerTakerFee

PATH = Path(__file__).resolve().parents[1] / "examples" / "phase1_friction_walkthrough.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase1_friction_walkthrough", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(mod)
    return mod


ex = _load()
SMALL = ex.Config(seed=3, n_flow=120)


@pytest.fixture(scope="module")
def results():
    return ex.compare(SMALL)


def test_same_seed_is_bit_identical():
    a, b = ex.run(SMALL, ex.regimes()["retail"]), ex.run(SMALL, ex.regimes()["retail"])
    assert a.fingerprint() == b.fingerprint()
    assert [t.time for t in a.trades] == [t.time for t in b.trades]
    assert a.pnl() == b.pnl() and a.events == b.events and a.mark == b.mark


def test_different_seed_changes_tape():
    other = ex.run(ex.Config(seed=4, n_flow=120))
    assert other.fingerprint() != ex.run(SMALL).fingerprint()


def test_default_config_pins_seed7_regression():
    res = ex.compare()
    base = res["frictionless"]
    assert len(base.trades) == 321 and base.mark == 10039
    assert base.pnl()[ex.MAKER]["net"] == 2828
    assert res["retail"].ledger.fees_collected == 10498
    assert res["slow_retail"].ledger.fees_collected == 11133


def test_run_produces_trades_and_passes_invariants(results):
    for res in results.values():
        assert len(res.trades) > 0
        assert res.ledger.violations() == []
        assert len(res.book.trades) == len(res.trades)


def test_tape_is_friction_invariant(results):
    tapes = {name: r.fingerprint() for name, r in results.items()}
    base = tapes["frictionless"]
    assert all(t == base for t in tapes.values())


def test_latency_shifts_time_only(results):
    fast, slow = results["retail"], results["slow_retail"]
    lag = ex.regimes()["slow_retail"].delay
    assert lag == pytest.approx(0.05)
    for f, s in zip(fast.trades, slow.trades):
        assert s.time == pytest.approx(f.time + lag)


def test_net_equals_gross_minus_fees_and_gross_is_regime_free(results):
    base = results["frictionless"].pnl()
    for res in results.values():
        pnl = res.pnl()
        for aid, r in pnl.items():
            assert r["net"] == r["gross"] - r["fees"]
            assert r["gross"] == base[aid]["gross"]
            assert r["trades"] == base[aid]["trades"]


def test_zero_sum_before_friction_and_venue_takes_the_rest(results):
    for res in results.values():
        pnl = res.pnl()
        assert sum(r["gross"] for r in pnl.values()) == 0
        assert sum(r["net"] for r in pnl.values()) == -res.ledger.fees_collected
        assert sum(r["fees"] for r in pnl.values()) == res.ledger.venue.cash


def test_fees_rise_with_friction_level(results):
    fees = [results[k].ledger.fees_collected
            for k in ("frictionless", "rebate_venue", "retail", "slow_retail")]
    assert fees[0] == 0
    assert fees == sorted(fees) and len(set(fees)) == 4


def test_rebate_lifts_maker_net_above_gross(results):
    maker = results["rebate_venue"].pnl()[ex.MAKER]
    assert maker["fees"] < 0 and maker["net"] > maker["gross"]


def test_fixed_cost_scales_with_fill_count():
    per = 7
    res = ex.run(SMALL, FrictionStack([FixedCost(per)]))
    for a in res.ledger:
        assert a.fees_paid == per * a.n_trades
    assert res.ledger.fees_collected == 2 * per * len(res.trades)


def test_custom_stack_and_report_render():
    stacks = {"base": FrictionStack(),
              "lag": MakerTakerFee(taker_bps=3) + LatencyCost(0.2, drift_ticks_per_sec=5)}
    res = ex.compare(SMALL, stacks)
    text = ex.report(res)
    assert "base" in text and "lag" in text and "venue" in text
    assert res["lag"].ledger.fees_collected > 0


def test_main_exit_code_signals_identical_tape(capsys):
    assert ex.main(["5"]) == 0
    out = capsys.readouterr().out
    assert "seed=5" in out and "identical_tape_across_regimes=True" in out
