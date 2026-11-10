"""Tests for fl-073 `HoldupMarket`: prices, splits, PnL identities, streams, ledger."""
from __future__ import annotations

from pathlib import Path

import pytest

from frictionlab import bench
from frictionlab.agents import SpecificInvestor, first_best_level, holdup_level
from frictionlab.analysis import (
    HoldupBuyerSurplus, HoldupPnL, TradeRecord, holdup_buyer_surplus, holdup_pnl,
)
from frictionlab.costs import CONTRACT_SINK_ID, ContractingCost
from frictionlab.markets import CONTRACT, HOLDUP, INVEST_SINK_ID, VALUE_ID, HoldupMarket

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline_small.json"
A = ((0, 20), (4, 30), (10, 40), (18, 48), (30, 55))
B = ((0, 18), (6, 34), (12, 44))
O = 15


def _market(scheds=(A, B, A), w=5_000, cost=None, seed=73, rounds=30, noise=(-2, 0, 1, 4)):
    cc = None if cost is None else ContractingCost(cost)
    sellers = [SpecificInvestor(i + 1) for i in range(len(scheds))]
    return HoldupMarket(sellers, list(scheds), [10, 11, 12, 13], rounds=rounds, outside_value=O,
                        seller_weight_bps=w, value_noise=noise, contracting=cc, seed=seed)


def _fb(m, r):
    sched = tuple((c, v + r.noise) for c, v in m.schedules[r.pair])
    return sched, first_best_level(sched, O)


@pytest.mark.parametrize("w", [0, 2500, 5000, 8000])
def test_holdup_investment_never_above_first_best(w):
    m = _market(w=w)
    m.run()
    for r in m.log:
        assert r.segment == HOLDUP and r.level <= _fb(m, r)[1]


def test_full_seller_weight_reaches_first_best():
    m = _market(w=10_000)
    m.run()
    assert all(r.level == _fb(m, r)[1] for r in m.log)
    h = _market(w=5_000)
    h.run()
    assert any(r.level < _fb(h, r)[1] for r in h.log)       # hold-up bites at w < 10000


@pytest.mark.parametrize("w", [0, 3333, 5000, 10_000])
def test_bargained_price_bounds_and_exact_split(w):
    m = _market(w=w)
    m.run()
    for r in m.log:
        q = m.quasi_rent(r)
        assert O <= r.price <= r.value
        seller_part, buyer_part = r.price - O, r.buyer_surplus
        assert seller_part + buyer_part == q
        assert buyer_part == (10_000 - w) * q // 10_000


def test_contracted_pairs_invest_at_first_best():
    m = _market(cost=1)
    m.run()
    assert m.contracts > 0
    for r in m.log:
        if r.segment == CONTRACT:
            sched, i = _fb(m, r)
            assert r.level == i and r.investment == sched[i][0]
        else:
            assert r.level == holdup_level(_fb(m, r)[0], O, 5_000)


@pytest.mark.parametrize("cost", [None, 0, 2, 6])
def test_seller_pnl_split_exact_in_ticks(cost):
    m = _market(cost=cost)
    m.run()
    rows = holdup_pnl(m)
    for sid, p in rows.items():
        assert isinstance(p, HoldupPnL) and p.trades == m.rounds
        assert p.net == p.revenue - p.investment - p.contracting == m.ledger[sid].cash
    assert sum(p.investment for p in rows.values()) == m.ledger[INVEST_SINK_ID].cash
    for bid, b in holdup_buyer_surplus(m).items():
        assert isinstance(b, HoldupBuyerSurplus) and b.net == m.ledger[bid].cash
        assert b.as_dict()["net"] == b.value - b.price


def test_disabled_contracting_reproduces_pure_holdup_bit_for_bit():
    a, b = _market(), _market()
    b.contracting = ContractingCost.disabled()
    a.run()
    b.run()
    assert a.log == b.log and [x.cash for x in a.ledger] == [x.cash for x in b.ledger]
    assert CONTRACT_SINK_ID not in a.ledger


def test_pair_draws_independent_of_other_pairs_schedule():
    a, b = _market(scheds=(A, B, A)), _market(scheds=(A, A, A), cost=0)
    a.run()
    b.run()
    for k in range(3):
        assert a.draws(k) == b.draws(k)
    c = _market(scheds=(A, B, B))
    c.run()
    assert a.pair(0) == c.pair(0) and a.pair(1) == c.pair(1)
    assert a.pair(1) != b.pair(1)


def test_ledger_invariants_and_records():
    m = _market(cost=2)
    m.run()
    m.ledger.check()
    assert sum(acct.cash for acct in m.ledger) == 0
    recs = m.records()
    assert len(recs) == len(m.log) == 3 * 30
    assert all(isinstance(r, TradeRecord) and r.venue in (HOLDUP, CONTRACT) for r in recs)
    assert sum(r.venue == CONTRACT for r in recs) == m.contracts
    assert sum(r.value for r in m.log) == -m.ledger[VALUE_ID].cash


def test_deterministic_and_idempotent():
    a, b = _market(seed=9, cost=1), _market(seed=9, cost=1)
    assert a.run() == b.run() and a.run() == a.contracts
    assert a.log == b.log
    c = _market(seed=10, cost=1)
    c.run()
    assert [c.draws(k) for k in range(3)] != [a.draws(k) for k in range(3)]


def test_bad_inputs():
    with pytest.raises(ValueError):
        _market(noise=())
    with pytest.raises(ValueError):
        _market(w=10_001)
    with pytest.raises(ValueError):
        HoldupMarket([SpecificInvestor(1)], [A, B], [10], rounds=1, outside_value=O)
    with pytest.raises(ValueError):
        HoldupMarket([SpecificInvestor(1)], [A], [1], rounds=1, outside_value=O)
    with pytest.raises(ValueError):
        HoldupMarket([SpecificInvestor(1)], [A], [10], rounds=1, outside_value=O,
                     value_noise=(-10,))


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name


def test_new_modules_under_line_ceiling():
    for rel in ("frictionlab/markets/holdup.py", "frictionlab/costs/contracting.py",
                "frictionlab/agents/specific_investor.py", "frictionlab/analysis/pnl.py",
                "frictionlab/analysis/venue_pnl.py"):
        assert len((ROOT / rel).read_text("utf-8").splitlines()) <= 250, rel
