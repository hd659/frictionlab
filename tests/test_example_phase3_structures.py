"""Regression tests for examples/phase3_market_structures.py (Phase 3 close-out)."""
import importlib.util
import sys
from pathlib import Path

import pytest

from frictionlab.markets import PERP_SINK_ID

PATH = Path(__file__).resolve().parents[1] / "examples" / "phase3_market_structures.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase3_market_structures", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(mod)
    return mod


ex = _load()
SEEDS = (1, 3, 7)


@pytest.fixture(scope="module")
def sweep():
    return ex.latency_sweep(ex.Config())


@pytest.fixture(scope="module")
def arbs():
    return ex.gas_sweep(7)


def _fingerprint(res):
    return ([(t.time, t.price, t.qty, t.buyer_id, t.seller_id) for t in res.trades],
            sorted((a.agent_id, a.cash, a.position) for a in res.ledger))


# ---- ledger invariants -------------------------------------------------------
def test_ledger_invariants_hold_in_every_structure_run(sweep):
    assert [r["latency"] for r in sweep] == list(ex.LATENCIES)
    for row in sweep:
        for name, res in row["results"].items():
            assert res.ledger.violations() == []
            s = row["stats"][name]
            assert s.volume == sum(t.qty for t in res.trades) > 0
            assert 0.0 <= s.snipe_share <= 1.0
        by_time = {}
        for t in row["results"]["batch"].trades:  # one clearing price per batch
            by_time.setdefault(t.time, set()).add(t.price)
        assert all(len(p) == 1 for p in by_time.values())


def test_ledger_and_pool_invariants_hold_in_every_arb_run(arbs):
    assert {(r["kind"], r["gas"]) for r in arbs} == {(k, g) for k in ex.POOLS for g in ex.GAS}
    for r in arbs:
        res = r["result"]
        assert res.ledger.violations() == []
        pool = res.pools[ex.POOL_ID]
        if r["kind"] == "cl":
            assert pool.violations() == []
        else:  # constant product never decreases
            x, y = pool.reserves
            assert x * y >= 10_000 * 1_000_000
        assert r["n_arbs"] > 0
        assert all(a.pnl.net >= 0 for a in res.arbs)  # deterministic quotes: never loses
        assert r["residual_gap"] < 8                  # every run narrows the 8-tick gap


@pytest.mark.parametrize("funding", [True, False])
def test_ledger_invariants_hold_in_perp_runs(funding):
    res, perp = ex.run_perp(ex.Config(), funding)
    res.ledger.check()
    assert PERP_SINK_ID in res.ledger and perp.history
    assert perp.open_interest == perp.short_interest and sum(perp.positions.values()) == 0
    for ev in perp.history:
        assert sum(ev.payments.values()) + ev.remainder == 0
        assert -perp.cap_bps <= ev.rate_bps <= perp.cap_bps
    assert perp.sink_cash == sum(ev.remainder for ev in perp.history)
    if not funding:
        assert all(ev.rate_bps == 0 and not ev.payments for ev in perp.history)
        assert perp.position(ex.CARRY_ID) == 0


# ---- reproducibility ---------------------------------------------------------
def test_structure_runs_reproducible_from_seed(sweep):
    again = ex.latency_sweep(ex.Config(), latencies=(ex.LATENCIES[-1],))[0]
    last = sweep[-1]
    assert again["stats"] == last["stats"]
    assert again["informed_pnl"] == last["informed_pnl"]
    for k in ("continuous", "batch"):
        assert _fingerprint(again["results"][k]) == _fingerprint(last["results"][k])


def test_arb_and_perp_reproducible_from_seed(arbs):
    for r in (arbs[2], arbs[-1]):
        b = ex.run_arb(r["kind"], r["gas"], 7)
        assert (b["band"], b["residual_gap"], b["n_arbs"], b["arb_pnl"]) == \
            (r["band"], r["residual_gap"], r["n_arbs"], r["arb_pnl"])
        assert _fingerprint(b["result"]) == _fingerprint(r["result"])
    (a, pa), (b, pb) = ex.run_perp(ex.Config(seed=3)), ex.run_perp(ex.Config(seed=3))
    assert _fingerprint(a) == _fingerprint(b)
    assert [(e.rate_bps, e.payments) for e in pa.history] == \
        [(e.rate_bps, e.payments) for e in pb.history]


def test_different_seeds_give_different_paths():
    a = ex.run_structure(ex.Config(seed=1), "continuous", 0.0)[0]
    b = ex.run_structure(ex.Config(seed=2), "continuous", 0.0)[0]
    assert _fingerprint(a) != _fingerprint(b)


# ---- economic claims ---------------------------------------------------------
@pytest.mark.parametrize("kind", ex.POOLS)
def test_no_arbitrage_band_non_decreasing_in_gas(kind):
    gas = (0, 25, 50, 100, 200, 400, 800, 1600)
    bands = [ex.band_width(kind, g) for g in gas]
    assert all(a <= b for a, b in zip(bands, bands[1:])), bands
    assert bands[-1] > bands[0] > 0


def test_concentrated_liquidity_band_grows_slower(arbs):
    band = {(r["kind"], r["gas"]): r["band"] for r in arbs}
    assert band[("cl", ex.GAS[-1])] <= band[("cp", ex.GAS[-1])]


@pytest.mark.parametrize("seed", SEEDS)
def test_batch_snipe_share_le_continuous_at_highest_latency(seed, sweep):
    row = sweep[-1] if seed == 7 else \
        ex.latency_sweep(ex.Config(seed=seed), latencies=(ex.LATENCIES[-1],))[0]
    assert row["latency"] == max(ex.LATENCIES)
    s = row["stats"]
    assert s["batch"].snipe_share <= s["continuous"].snipe_share


def test_snipe_share_rises_with_latency_in_continuous_book(sweep):
    shares = [r["stats"]["continuous"].snipe_share for r in sweep]
    assert shares[-1] > shares[0]


@pytest.mark.parametrize("seed", SEEDS)
def test_funding_keeps_mark_closer_to_index(seed):
    cfg = ex.Config(seed=seed)
    on = ex.mean_abs_premium(ex.run_perp(cfg, True)[1])
    off = ex.mean_abs_premium(ex.run_perp(cfg, False)[1])
    assert on <= off


def test_main_prints_all_three_parts(capsys):
    assert ex.main(["7"]) == 0
    out = capsys.readouterr().out
    for marker in ("part 1", "part 2", "part 3", "continuous", "batch", "funding=on"):
        assert marker in out
