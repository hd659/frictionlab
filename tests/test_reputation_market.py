"""Tests for fl-058 `ReputationMarket`: score pricing, whitewashing, PnL split, fl-055 parity."""
from __future__ import annotations

from pathlib import Path

import pytest

from frictionlab import bench
from frictionlab.analysis import ReputationPnL, reputation_pnl
from frictionlab.markets import (
    IDENTITY_SINK_ID, REPUTATION, LemonsMarket, ReputationMarket, score_quality,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline_small.json"
SCHEDULE = [10, 20, 30, 40, 50, 60]


def _market(**kw):
    base = dict(rounds=8, premium=15, rating_accuracy_bps=8500, prior_honest_bps=7000,
                seed=58)
    base.update(kw)
    return ReputationMarket.from_schedule(18, SCHEDULE, opportunists=range(1, 19, 3), **base)


def test_price_every_round_is_buyer_value_of_score_implied_quality():
    mkt = _market(whitewash_cost=4)
    mkt.run()
    assert mkt.log
    for r in mkt.log:
        eq = score_quality(*r.score, mkt.mean_q, mkt.floor, mkt.accuracy, mkt.prior)
        assert r.price == eq + mkt.premium == mkt.price_for(r.score)
        assert mkt.floor <= eq <= mkt.mean_q
    traded = {t.trade_id: t.price for t in mkt.trades}
    assert sorted(traded.values()) == sorted(r.price for r in mkt.log if r.delivered)


def test_score_quality_exact_integer_posterior():
    # prior 5000, accuracy 8000: one good rating -> w_H : w_O = 4 : 1
    assert score_quality(1, 0, 50, 10, 8000, 5000) == 10 + 40 * 4 // 5
    assert score_quality(0, 0, 50, 10, 8000, 5000) == 30
    assert score_quality(0, 1, 50, 10, 10_000, 9000) == 10      # perfect rating, caught
    assert score_quality(3, 0, 50, 10, 10_000, 0) == 10         # zero prior weight


def test_whitewash_count_times_cost_reconciles_with_identity_sink():
    mkt = _market(whitewash_cost=3)
    mkt.run()
    washes = sum(s.whitewashes for s in mkt.sellers.values())
    assert washes > 0
    assert washes * 3 == mkt.whitewash_costs == mkt.ledger[IDENTITY_SINK_ID].cash
    for sid, s in mkt.sellers.items():
        assert mkt.wash_paid[sid] == 3 * s.whitewashes
    ids = [s.public_id for s in mkt.sellers.values()]
    assert len(set(ids)) == len(ids) and max(ids) == len(ids) + washes


def test_no_whitewash_cost_means_no_identity_sink():
    mkt = _market()
    mkt.run()
    assert IDENTITY_SINK_ID not in mkt.ledger
    assert all(not r.whitewashed for r in mkt.log)


def test_pnl_split_is_exact_in_ticks():
    mkt = _market(whitewash_cost=2)
    mkt.run()
    pnls = reputation_pnl(mkt)
    assert any(p.cheating_gain > 0 for p in pnls.values())
    for sid, p in pnls.items():
        assert isinstance(p, ReputationPnL)
        assert p.trade_surplus + p.cheating_gain - p.whitewash_cost == p.net
        cost = sum(mkt.delivered[t.trade_id] for t in mkt.trades if t.seller_id == sid)
        assert p.net == mkt.ledger[sid].cash - cost
        assert p.whitewash_cost == mkt.wash_paid[sid]


@pytest.mark.parametrize("premium", [0, 10, 25, 50])
def test_single_round_honest_market_reproduces_pooled_lemons_prices(premium):
    rep = ReputationMarket.from_schedule(20, SCHEDULE, seed=55, rounds=1, premium=premium)
    lem = LemonsMarket.from_schedule(20, SCHEDULE, seed=55, premium=premium)
    lem.run()
    rep.run()
    assert [s.quality for s in rep.sellers.values()] == list(lem.qualities.values())
    assert rep.fresh_price == lem.price
    assert {t.price for t in rep.trades} == {t.price for t in lem.trades}
    assert sorted(t.seller_id for t in rep.trades) == sorted(lem.sellers_sold())


def test_ledger_invariants_and_record_tags():
    mkt = _market(whitewash_cost=5)
    mkt.run()
    mkt.ledger.check()
    recs = mkt.records()
    assert len(recs) == len(mkt.trades) and {r.venue for r in recs} == {REPUTATION}


def test_runs_are_deterministic_under_fixed_seed():
    a, b = _market(whitewash_cost=3), _market(whitewash_cost=3)
    assert a.run() == b.run()
    assert a.log == b.log and a.records() == b.records()
    c = _market(whitewash_cost=3, seed=59)
    c.run()
    assert c.log != a.log


def test_ratings_use_per_seller_streams():
    qs = [30, 40, 50]
    a = ReputationMarket(qs, rounds=6, premium=20, rating_accuracy_bps=7000, seed=3)
    b = ReputationMarket(qs + [35], opportunists=[4], rounds=6, premium=20,
                         rating_accuracy_bps=7000, seed=3)
    a.run()
    b.run()
    for sid in (1, 2, 3):
        ra = [r.rating for r in a.history(sid) if r.rating is not None]
        rb = [r.rating for r in b.history(sid) if r.rating is not None]
        n = min(len(ra), len(rb))
        assert n and ra[:n] == rb[:n]


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        ReputationMarket([])
    with pytest.raises(ValueError):
        ReputationMarket([10], rating_accuracy_bps=10_001)


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name


def test_new_modules_under_line_ceiling():
    for rel in ("frictionlab/markets/reputation.py", "frictionlab/agents/reputation_seller.py"):
        assert len((ROOT / rel).read_text("utf-8").splitlines()) <= 250, rel
