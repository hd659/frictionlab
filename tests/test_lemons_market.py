"""Tests for fl-055 `LemonsMarket`: pooled price iteration, unraveling, settlement and PnL."""
from __future__ import annotations

from pathlib import Path

import pytest

from frictionlab import bench
from frictionlab.analysis import LemonsPnL, lemons_pnl
from frictionlab.costs import CERT_SINK_ID, CertificationPolicy
from frictionlab.markets import CERTIFIED, POOLED, LemonsMarket, pooled_price

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline_small.json"


def test_pooled_price_matches_floor_mean_plus_premium_every_round():
    mkt = LemonsMarket([10, 20, 30, 40, 50], premium=5)
    final_p = mkt.iterate()
    assert mkt.rounds and final_p == mkt.rounds[-1].price
    for rnd in mkt.rounds:
        qs = [mkt.qualities[i] for i in rnd.offered]
        assert rnd.price == sum(qs) // len(qs) + mkt.premium
        assert rnd.price == pooled_price(qs, mkt.premium)


def test_fixed_point_terminates_and_is_deterministic():
    m1 = LemonsMarket.from_schedule(24, [10, 20, 30, 40, 50, 60], seed=55, premium=8)
    m2 = LemonsMarket.from_schedule(24, [10, 20, 30, 40, 50, 60], seed=55, premium=8)
    assert m1.run() == m2.run()
    assert m1.rounds == m2.rounds
    assert m1.records() == m2.records()


def test_zero_premium_uniform_schedule_unravels_to_lowest_quality():
    schedule = [10, 20, 30, 40, 50]
    mkt = LemonsMarket(schedule * 4, premium=0)
    n_trades = mkt.run()
    assert mkt.price == 10
    assert n_trades == 4
    sold_qualities = {mkt.qualities[sid] for sid in mkt.sellers_sold()}
    assert sold_qualities == {10}


def test_premium_at_least_quality_range_trades_full_pool():
    qualities = [10, 20, 30, 40, 50]
    q_range = max(qualities) - min(qualities)
    mkt = LemonsMarket(qualities, premium=q_range)
    assert mkt.run() == len(qualities)
    assert len(mkt.rounds) == 2
    assert set(mkt.sellers_sold(POOLED)) == set(mkt.qualities)


def test_no_seller_sells_below_reservation_value():
    cert = CertificationPolicy(cost=6)
    mkt = LemonsMarket.from_schedule(30, [10, 25, 40, 55, 70, 85], seed=7,
                                     premium=12, certification=cert)
    mkt.run()
    for t in mkt.trades:
        q = mkt.qualities[t.seller_id]
        if mkt.segments[t.trade_id] == POOLED:
            assert t.price >= q
        else:
            assert t.price - mkt.cert_paid[t.seller_id] >= q


def test_records_carry_segment_tag_and_ledger_invariants_hold():
    cert = CertificationPolicy(cost=4)
    mkt = LemonsMarket([10, 20, 30, 40, 50], premium=10, certification=cert)
    mkt.run()
    recs = mkt.records()
    assert len(recs) == len(mkt.trades)
    assert {r.venue for r in recs} <= {POOLED, CERTIFIED}
    mkt.ledger.check()


def test_lemons_pnl_splits_surplus_and_certification_cost_exactly():
    cert = CertificationPolicy(cost=5)
    mkt = LemonsMarket([10, 20, 30, 40, 50, 60], premium=12, certification=cert)
    mkt.run()
    pnls = lemons_pnl(mkt)
    for sid, p in pnls.items():
        assert isinstance(p, LemonsPnL)
        assert p.trade_surplus - p.certification_cost == p.net
        acct = mkt.ledger[sid]
        q = mkt.qualities[sid]
        equity_change = acct.cash + (acct.position - 1) * q
        assert p.net == equity_change
    assert sum(p.certification_cost for p in pnls.values()) == mkt.certification_costs
    assert mkt.ledger[CERT_SINK_ID].cash == mkt.certification_costs


def test_disabled_certification_reproduces_pure_pooled_run_bit_for_bit():
    qs = [12, 18, 25, 33, 47, 60, 75]
    m_none = LemonsMarket(qs, premium=9, certification=None)
    m_dis = LemonsMarket(qs, premium=9, certification=CertificationPolicy.disabled())
    assert m_none.run() == m_dis.run()
    assert m_none.rounds == m_dis.rounds
    assert m_none.records() == m_dis.records()
    assert CERT_SINK_ID not in m_dis.ledger


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
