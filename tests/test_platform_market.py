"""Tests for fl-064 `PlatformMarket`: cutoffs, monotone dynamics, cold start, exact PnL."""
from __future__ import annotations

from pathlib import Path

import pytest

from frictionlab import bench
from frictionlab.analysis import (
    MemberSurplus, PlatformPnL, TradeRecord, member_surplus, platform_pnl,
)
from frictionlab.costs import ADOPTION_SINK_ID, AdoptionCost
from frictionlab.markets import PLATFORM, PLATFORM_ID, PlatformMarket

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline_small.json"
SCHED = ([-30, -20, -10, -4, 0, 3, 8], [-25, -12, -6, -1, 2, 6])


def _market(seed=64, benefit=(3, 2), fee=(2, -1), adoption=4, seed_members=(0, 0),
            rounds=40, n=(12, 14), sched=SCHED):
    return PlatformMarket(n, standalone=sched, benefit=benefit, fee=fee, adoption=adoption,
                          seed_members=seed_members, rounds=rounds, seed=seed)


def test_zero_benefit_is_static_cutoff_and_never_changes():
    mkt = _market(benefit=(0, 0))
    mkt.run()
    first = mkt.history[1]
    assert first == mkt.cutoff_members()
    assert all(state == first for state in mkt.history[1:])
    assert mkt.log[0].leaves == (0, 0) and len(mkt.log) == 2   # stops at the repeat


@pytest.mark.parametrize("seed", [1, 2, 3, 64])
def test_counts_monotone_from_empty(seed):
    mkt = _market(seed=seed)
    mkt.run()
    counts = [(0, 0)] + [r.members for r in mkt.log]
    for a, b in zip(counts, counts[1:]):
        assert b[0] >= a[0] and b[1] >= a[1]
    assert all(r.leaves == (0, 0) for r in mkt.log)


@pytest.mark.parametrize("seed", [1, 2, 3, 64])
def test_counts_monotone_from_full(seed):
    mkt = _market(seed=seed, fee=(9, 7), seed_members=(12, 14))
    mkt.run()
    counts = [(12, 14)] + [r.members for r in mkt.log]
    for a, b in zip(counts, counts[1:]):
        assert b[0] <= a[0] and b[1] <= a[1]
    assert all(r.joins == (0, 0) for r in mkt.log)
    assert counts[-1] < (12, 14)


@pytest.mark.parametrize("seed_members", [(0, 0), (3, 2), (12, 0)])
def test_no_member_holds_negative_net_value(seed_members):
    mkt = _market(seed_members=seed_members)
    mkt.run()
    for prev, now in zip(mkt.history, mkt.history[1:]):
        assert all(mkt.net_value(m, prev) >= 0 for m in now)
        for m in now - prev:                         # joiners also covered adoption
            assert mkt.net_value(m, prev) >= mkt.adoption.cost_for(m)


def test_cold_start_trap_and_seeding_tips():
    trap = _market(adoption=40, benefit=(6, 6))
    trap.run()
    assert trap.members == set() and trap.log[-1].members == (0, 0)
    assert trap.adoption.count == 0 and trap.ledger[ADOPTION_SINK_ID].cash == 0
    tipped = _market(adoption=40, benefit=(6, 6), seed_members=(6, 6))
    tipped.run()
    assert tipped.log[-1].members[0] > 6 and tipped.log[-1].members[1] > 6


def test_adoption_sink_reconciles_with_charges():
    mkt = _market(seed_members=(2, 3), adoption=AdoptionCost(5))
    mkt.run()
    joins = sum(sum(r.joins) for r in mkt.log) + 5
    assert mkt.adoption.count == joins
    assert mkt.ledger[ADOPTION_SINK_ID].cash == 5 * joins == mkt.adoption.revenue
    assert mkt.adoption.paid[PLATFORM_ID] == 5 * 5


def test_platform_pnl_equals_cash_change():
    for seed_members in [(0, 0), (4, 4)]:
        mkt = _market(seed_members=seed_members)
        mkt.run()
        p = platform_pnl(mkt)
        assert isinstance(p, PlatformPnL) and p.fee_revenue > 0 and p.subsidy_outlay > 0
        assert p.net == mkt.ledger[PLATFORM_ID].cash
        assert p.as_dict()["net"] == p.fee_revenue - p.subsidy_outlay


def test_member_surplus_sums_exactly():
    mkt = _market(seed_members=(3, 1))
    mkt.run()
    rows = member_surplus(mkt)
    assert set(rows) == set(mkt.side_of)
    for mid, r in rows.items():
        assert isinstance(r, MemberSurplus)
        assert r.net == r.standalone + r.cross_side - r.fees - r.adoption
        assert -mkt.ledger[mid].cash == r.fees + r.adoption       # cash moves only
    fees = sum(r.fees for r in rows.values())
    paid = sum(r.adoption for r in rows.values())
    assert fees == platform_pnl(mkt).fee_revenue - sum(
        f.amount for f in mkt.flows if f.kind == "subsidy")
    assert paid + mkt.adoption.paid.get(PLATFORM_ID, 0) == mkt.adoption.revenue


def test_ledger_invariants_and_records():
    mkt = _market(seed_members=(2, 2))
    mkt.run()
    mkt.ledger.check()
    assert sum(a.cash for a in mkt.ledger) == 0
    recs = mkt.records()
    assert recs and all(isinstance(r, TradeRecord) and r.venue == PLATFORM for r in recs)
    assert sum(r.notional for r in recs) == sum(f.amount for f in mkt.flows)


def test_deterministic_under_seed_and_run_idempotent():
    a, b = _market(seed=7), _market(seed=7)
    assert a.run() == b.run() and a.run() == len(a.log)
    assert a.values == b.values and a.history == b.history and a.flows == b.flows
    assert _market(seed=8).values != a.values


def test_per_member_streams_are_independent():
    small, big = _market(n=(5, 5)), _market(n=(5, 9))
    assert all(small.values[m] == big.values[m] for m in range(2, 7))


def test_stops_at_repeat_or_horizon():
    mkt = _market(rounds=1)
    assert mkt.run() == 1
    full = _market()
    full.run()
    assert full.history[-1] in full.history[:-1]


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        _market(seed_members=(13, 0))
    with pytest.raises(ValueError):
        _market(benefit=(-1, 0))
    with pytest.raises(ValueError):
        PlatformMarket((2, 2), standalone=([1],), rounds=1)
    with pytest.raises(TypeError):
        _market(fee=(1.5, 0))


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name


def test_new_modules_under_line_ceiling():
    for rel in ("frictionlab/markets/platform.py", "frictionlab/costs/adoption.py",
                "frictionlab/analysis/pnl.py"):
        assert len((ROOT / rel).read_text("utf-8").splitlines()) <= 250, rel
