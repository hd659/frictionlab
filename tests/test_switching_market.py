"""Tests for fl-061 `SwitchingMarket`: switch rule, valuation cap, Bertrand parity, sinks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frictionlab import bench
from frictionlab.agents import LockinSeller
from frictionlab.analysis import SwitchingPnL, consumer_surplus, switching_pnl
from frictionlab.core.types import Side, Trade
from frictionlab.markets import SWITCH_SINK_ID, SWITCHING, SwitchingMarket

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline_small.json"
COSTS = [1, 3, 6, 9]


def _values(n=30, lo=10, hi=80, seed=1):
    return [int(x) for x in np.random.default_rng(seed).integers(lo, hi, size=n)]


def _market(seed=61, rounds=10, costs=COSTS, **kw):
    """Incumbent keeps acquiring (high target); a cheaper entrant arrives in round 2."""
    a = LockinSeller(1, 12, acquisition_price=16, target_bps=9500)
    b = LockinSeller(2, 10, acquisition_price=8, target_bps=6000, entry_round=2, **kw)
    return SwitchingMarket(_values(), [a, b], switch_costs=costs, rounds=rounds, seed=seed)


def _bertrand(values, costs, rounds=6, c=20, seed=7):
    sellers = [LockinSeller.bertrand_seller(1, c), LockinSeller.bertrand_seller(2, c)]
    return SwitchingMarket(values, sellers, switch_costs=costs, rounds=rounds, seed=seed)


def test_no_switch_when_gap_within_switching_cost():
    mkt = _market()
    mkt.run()
    bought = {(int(t.time), t.buyer_id): t.seller_id for t in mkt.trades}
    prev = dict.fromkeys(mkt.values)
    checked = 0
    for r in mkt.log:
        for cid, v in mkt.values.items():
            k, now = prev[cid], bought.get((r.t, cid))
            if k is None or r.prices[0] is None or r.prices[1] is None:
                prev[cid] = now if now is not None else k
                continue
            gap = r.prices[k - 1] - r.prices[2 - k]
            if gap <= mkt.costs[cid]:
                assert now in (k, None), (r.t, cid)         # never leaves
                assert (now == k) == (r.prices[k - 1] <= v)
                checked += 1
            if now is not None and now != k:
                assert gap > mkt.costs[cid]
            prev[cid] = now if now is not None else k
    assert checked > 0 and mkt.switch_count > 0


def test_no_trade_above_valuation_including_switching_cost():
    mkt = _market()
    mkt.run()
    for t in mkt.trades:
        assert t.price <= mkt.values[t.buyer_id]
    gross, cost, net = zip(*consumer_surplus(mkt).values())
    assert all(g >= 0 for g in gross) and sum(cost) == mkt.switching_costs_paid
    assert sum(net) == sum(gross) - sum(cost)
    # a switcher's price + cost is within its valuation in the switching round
    for (cid, row) in consumer_surplus(mkt).items():
        fills = [t for t in mkt.trades if t.buyer_id == cid]
        assert row[0] == sum(mkt.values[cid] - t.price for t in fills)


def _reference_bertrand(values, rounds, c):
    """Frictionless repeated Bertrand: both at c, ties to seller 1, buy iff c <= v."""
    trades = []
    for t in range(rounds):
        for cid, v in enumerate(values, start=3):
            if c <= v:
                tid = len(trades) + 1
                trades.append(Trade(tid, float(t), c, 1, tid, tid, cid, 1, Side.BUY))
    return trades


def test_zero_switching_cost_bertrand_reproduces_frictionless_run_bit_for_bit():
    vals = _values(seed=5)
    mkt = _bertrand(vals, [0])
    mkt.run()
    assert all(r.prices == (20, 20) for r in mkt.log)
    assert mkt.trades == _reference_bertrand(vals, 6, 20)
    assert mkt.switch_count == 0 and mkt.ledger[SWITCH_SINK_ID].cash == 0
    assert all(p.net == 0 for p in switching_pnl(mkt).values())
    # switching costs are irrelevant when nobody has a reason to switch
    other = _bertrand(vals, [0, 5, 9])
    other.run()
    assert other.trades == mkt.trades and other.log == mkt.log


def test_switch_count_times_cost_reconciles_with_sink():
    mkt = _market()
    mkt.run()
    assert mkt.switch_count == sum(r.switches for r in mkt.log) > 0
    assert mkt.switching_costs_paid == mkt.ledger[SWITCH_SINK_ID].cash
    for cid, fee in mkt.switch_paid.items():
        assert fee % mkt.costs[cid] == 0 if mkt.costs[cid] else fee == 0


def test_uniform_cost_switch_count_times_cost_equals_sink():
    mkt = _market(costs=[4])
    mkt.run()
    assert mkt.switch_count > 0
    assert mkt.switch_count * 4 == mkt.ledger[SWITCH_SINK_ID].cash


def test_acquisition_plus_harvest_equals_net_in_ticks():
    mkt = _market()
    mkt.run()
    pnls = switching_pnl(mkt)
    assert pnls[2].acquisition < 0 < pnls[2].harvest
    for sid, p in pnls.items():
        assert isinstance(p, SwitchingPnL)
        assert p.acquisition + p.harvest == p.net
        units = p.sold_acquire + p.sold_harvest
        assert p.net == mkt.ledger[sid].cash - mkt.sellers[sid].marginal_cost * units


def test_ledger_invariants_and_record_tags():
    mkt = _market()
    mkt.run()
    mkt.ledger.check()
    recs = mkt.records()
    assert len(recs) == len(mkt.trades) and {r.venue for r in recs} == {SWITCHING}


def test_runs_are_deterministic_and_costs_use_per_consumer_streams():
    a, b = _market(), _market()
    assert a.run() == b.run() and a.log == b.log and a.records() == b.records()
    c = _market(seed=62)
    assert c.costs != a.costs
    small = SwitchingMarket(_values()[:10], [LockinSeller(1, 5), LockinSeller(2, 5)],
                            switch_costs=COSTS, seed=61)
    assert all(small.costs[k] == a.costs[k] for k in small.costs)


def test_invalid_parameters_rejected():
    s = [LockinSeller(1, 5), LockinSeller(2, 5)]
    with pytest.raises(ValueError):
        SwitchingMarket([], s)
    with pytest.raises(ValueError):
        SwitchingMarket([10], [LockinSeller(1, 5)])
    with pytest.raises(ValueError):
        SwitchingMarket([10], s, switch_costs=[])


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name


def test_new_modules_under_line_ceiling():
    for rel in ("frictionlab/markets/switching.py", "frictionlab/agents/lockin_seller.py",
                "frictionlab/analysis/pnl.py", "frictionlab/analysis/venue_pnl.py"):
        assert len((ROOT / rel).read_text("utf-8").splitlines()) <= 250, rel
