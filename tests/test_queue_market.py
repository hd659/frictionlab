"""Tests for fl-067 `QueueMarket`: Naor threshold, join rule, streams, exact surplus."""
from __future__ import annotations

from pathlib import Path

import pytest

from frictionlab import bench
from frictionlab.analysis import (
    OperatorPnL, QueueSurplus, TradeRecord, operator_pnl, queue_surplus,
)
from frictionlab.costs import TOLL_SINK_ID, WAIT_SINK_ID, CongestionCost
from frictionlab.markets import QUEUE, QueueMarket, ceil_div

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline_small.json"


def _market(seed=67, reward=60, toll=6, waiting=4, arrival=7000, service=4000, steps=300):
    return QueueMarket(reward=reward, arrival_prob_bps=arrival, service_prob_bps=service,
                       steps=steps, toll=toll, waiting=waiting, seed=seed)


def test_ceil_div_is_exact():
    assert [ceil_div(a, 4) for a in (0, 1, 4, 5, 8, 9)] == [0, 1, 1, 2, 2, 3]
    m = _market(service=3000)
    assert m.expected_steps(0) == 4 and m.expected_steps(2) == 10


@pytest.mark.parametrize("seed,reward,toll,waiting,service", [
    (1, 60, 6, 4, 4000), (2, 100, 0, 3, 5000), (3, 40, 10, 7, 6000), (67, 25, 3, 2, 2500)])
def test_queue_never_exceeds_naor_threshold(seed, reward, toll, waiting, service):
    m = _market(seed=seed, reward=reward, toll=toll, waiting=waiting, service=service)
    m.run()
    n_max = m.threshold()
    assert n_max == ((reward - toll) // waiting) * service // 10_000
    assert max(m.lengths) <= n_max
    for a in m.arrivals:                          # join iff n < N exactly
        assert a.joined == (a.n_seen < n_max)
    assert any(not a.joined for a in m.arrivals)  # the threshold binds on these runs


@pytest.mark.parametrize("seed", [4, 5, 67])
def test_no_join_with_negative_expected_net(seed):
    m = _market(seed=seed)
    m.run()
    assert m.arrivals
    for a in m.arrivals:
        assert a.expected_net == 60 - 6 - 4 * ceil_div((a.n_seen + 1) * 10_000, 4000)
        assert a.joined == (a.expected_net >= 0)


def test_ties_join_and_negative_slack_blocks_all():
    tie = QueueMarket(reward=10, arrival_prob_bps=10_000, service_prob_bps=10_000, steps=5,
                      toll=6, waiting=4)
    tie.run()
    assert all(a.joined and a.expected_net == 0 for a in tie.arrivals)
    none = _market(reward=5, toll=6)
    none.run()
    assert none.threshold() == 0 and none.joins == 0
    assert none.ledger[TOLL_SINK_ID].cash == 0 and none.ledger[WAIT_SINK_ID].cash == 0


def test_free_queue_admits_every_arrival():
    m = _market(toll=0, waiting=0, arrival=9000, service=2000)
    m.run()
    assert m.threshold() is None
    assert m.joins == len(m.arrivals) > 0 and max(m.lengths) > 5


def test_surplus_decomposition_exact_per_customer():
    m = _market()
    m.run()
    rows = queue_surplus(m)
    assert set(rows) == set(m.customers)
    for cid, r in rows.items():
        assert isinstance(r, QueueSurplus)
        assert r.net == r.reward - r.toll - r.waiting
        assert r.toll == 6 and r.waiting == 4 * r.steps
        assert r.reward == (60 if m.customers[cid].served else 0)
        assert -m.ledger[cid].cash == r.toll + r.waiting      # cash moves only
    p = operator_pnl(m)
    assert isinstance(p, OperatorPnL) and p.net == p.toll_revenue == 6 * m.joins
    assert p.as_dict()["joins"] == m.joins


def test_arrivals_independent_of_service_prob():
    a, b = _market(service=4000), _market(service=8500)
    a.run()
    b.run()
    assert a.arrival_draws == b.arrival_draws
    assert a.service_draws != b.service_draws
    c = _market(arrival=3000)
    c.run()
    assert c.service_draws == a.service_draws


def test_ledger_invariants_and_records():
    m = _market()
    m.run()
    m.ledger.check()
    assert sum(acct.cash for acct in m.ledger) == 0
    recs = m.records()
    assert len(recs) == m.served > 0
    assert all(isinstance(r, TradeRecord) and r.venue == QUEUE for r in recs)
    served = [c for c in m.customers.values() if c.served]
    assert sum(r.notional for r in recs) == sum(6 + 4 * c.steps for c in served)


def test_deterministic_and_run_idempotent():
    a, b = _market(seed=9), _market(seed=9)
    assert a.run() == b.run() and a.run() == a.served
    assert a.arrivals == b.arrivals and a.lengths == b.lengths
    c = _market(seed=10)
    c.run()
    assert c.arrival_draws != a.arrival_draws


def test_fifo_and_steps_accounting():
    m = _market()
    m.run()
    served = sorted((c for c in m.customers.values() if c.served), key=lambda c: c.joined_t)
    assert [c.served_t for c in served] == sorted(c.served_t for c in served)
    for c in served:
        assert c.steps == c.served_t - c.joined_t + 1


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        _market(service=0)
    with pytest.raises(ValueError):
        _market(arrival=10_001)
    with pytest.raises(ValueError):
        _market(toll=-1)
    with pytest.raises(TypeError):
        _market(reward=1.5)
    assert isinstance(_market().congestion, CongestionCost)


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name


def test_new_modules_under_line_ceiling():
    for rel in ("frictionlab/markets/queue_market.py", "frictionlab/costs/congestion.py",
                "frictionlab/analysis/pnl.py"):
        assert len((ROOT / rel).read_text("utf-8").splitlines()) <= 250, rel
