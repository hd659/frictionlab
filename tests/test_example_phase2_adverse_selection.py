"""Regression tests for examples/phase2_adverse_selection.py (informed flow vs spreads)."""
import importlib.util
import sys
from pathlib import Path

import pytest

from frictionlab.agents import InformedTrader, MarketView, NoiseTrader

PATH = Path(__file__).resolve().parents[1] / "examples" / "phase2_adverse_selection.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase2_adverse_selection", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(mod)
    return mod


ex = _load()
SEEDS = (1, 2, 3, 4, 7)


@pytest.fixture(scope="module")
def sweeps():
    return {s: ex.sweep(ex.Config(seed=s)) for s in SEEDS}


def test_ledger_invariants_hold_for_every_share():
    cfg = ex.Config(seed=3)
    for share in ex.SHARES + (1.0,):
        res, fund, _ = ex.simulate(cfg, share)
        assert res.ledger.violations() == []
        assert sum(t.qty for t in res.trades) > 0
        n_inf = sum(isinstance(a, InformedTrader) for a in res.agents.values())
        n_noise = sum(isinstance(a, NoiseTrader) for a in res.agents.values())
        assert n_inf == (share > 0) and n_noise == (cfg.n_noise if share < 1 else 0)
        # total taker intensity is held at lam
        rate = sum(a.wake_rate for a in res.agents.values() if a.agent_id != ex.MM_ID)
        assert rate == pytest.approx(cfg.lam)


def test_results_are_reproducible_from_a_seed():
    cfg = ex.Config(seed=5)
    assert ex.run(cfg, 0.3) == ex.run(cfg, 0.3)
    assert ex.run(cfg, 0.3) != ex.run(ex.Config(seed=6), 0.3)
    assert ex.run(cfg, 0.3, seed=6) == ex.run(ex.Config(seed=6), 0.3)


def test_default_seed_pins_regression(sweeps):
    rows = sweeps[7]
    r0, top = rows[0], rows[-1]
    assert (r0["volume"], r0["mm_pnl"], r0["informed_pnl"]) == (1159, -26179, 0)
    assert (top["volume"], top["mm_vs_informed"], top["informed_pnl"]) == (819, -4226, 3694)
    assert r0["spread"] == pytest.approx(2.8783, abs=1e-4)
    assert top["spread"] == pytest.approx(13.7683, abs=1e-4)


@pytest.mark.parametrize("seed", SEEDS)
def test_average_spread_is_non_decreasing_in_informed_share(sweeps, seed):
    spreads = [r["spread"] for r in sweeps[seed]]
    assert all(b >= a for a, b in zip(spreads, spreads[1:])), spreads
    assert spreads[-1] > spreads[0] + 5
    assert all(r["coverage"] > 0.8 for r in sweeps[seed])


@pytest.mark.parametrize("seed", SEEDS)
def test_market_maker_loses_to_informed_flow_in_aggregate(sweeps, seed):
    rows = sweeps[seed]
    assert rows[0]["mm_vs_informed"] == 0 and rows[0]["informed_volume"] == 0
    for r in rows[1:]:
        assert r["informed_volume"] > 0
        assert r["mm_vs_informed"] <= 0
        assert r["informed_pnl"] > 0  # information pays, net of taker fees
    assert sum(r["mm_vs_informed"] for r in rows) < 0


def test_noise_traders_fund_the_informed_on_average(sweeps):
    informed = [r for s in SEEDS for r in sweeps[s][1:]]
    assert sum(r["noise_pnl"] for r in informed) < 0
    # PnL is zero-sum net of fees (no decision costs in this example)
    for r in informed:
        assert r["mm_pnl"] + r["informed_pnl"] + r["noise_pnl"] == -r["fees"]


def test_premium_scales_with_share_and_widens_both_sides():
    mms = [ex.GMMarketMaker(1, s, 10, fair_value=100, horizon=100.0) for s in ex.SHARES]
    assert [m.premium for m in mms] == [0, 1, 2, 3, 4, 5]
    v = MarketView(0.0, 1, None, None, (), (), None, 0, 0, ())
    base, wide = mms[0].quote(v), mms[3].quote(v)
    assert (base.bid - wide.bid, wide.ask - base.ask) == (3, 3)
    assert wide.spread == pytest.approx(base.spread + 6)
    with pytest.raises(ValueError):
        ex.GMMarketMaker(1, 1.5, 10)
    with pytest.raises(ValueError):
        ex.GMMarketMaker(1, 0.2, 10, learn=2.0)


def test_main_prints_sweep(capsys):
    assert ex.main(["2"]) == 0
    out = capsys.readouterr().out
    assert "seed=2" in out and "mm_vs_inf" in out and len(out.splitlines()) == 2 + len(ex.SHARES)
