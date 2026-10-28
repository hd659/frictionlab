"""Unit tests for frictionlab.analysis.spread (quoted spread and PnL metrics)."""
import math

import pytest

from frictionlab.agents import (
    AvellanedaStoikovMM, FundamentalProcess, InformedTrader, make_noise_population,
)
from frictionlab.analysis import (
    QuoteRecorder, markout_pnl, pnl_by_agent, pnl_by_type, spread_stats, volume_between,
)
from frictionlab.core import Side, Trade
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import FixedCost, MakerTakerFee


def trade(tid, t, price, qty, buyer, seller):
    return Trade(tid, float(t), price, qty, 100 + tid, 200 + tid, buyer, seller, Side.BUY)


# -- quoted spread ---------------------------------------------------------------
def test_time_weighted_spread_on_hand_built_path():
    # spread 2 for 1s, 4 for 3s, then one-sided for 1s, then 6 until t=6
    path = [(0.0, 99, 101), (1.0, 98, 102), (4.0, None, 102), (5.0, 97, 103)]
    st = spread_stats(path, until=6.0)
    assert st.tw_spread == pytest.approx((2 * 1 + 4 * 3 + 6 * 1) / 5)
    assert st.coverage == pytest.approx(5 / 6)
    assert st.mean_spread == pytest.approx(4.0)
    assert st.max_spread == 6 and st.n_samples == 4


def test_one_sided_book_has_no_spread():
    st = spread_stats([(0.0, None, None), (2.0, 99, None)], until=3.0)
    assert math.isnan(st.tw_spread) and math.isnan(st.mean_spread)
    assert st.coverage == 0.0 and st.max_spread == 0


def test_spread_stats_rejects_bad_input():
    with pytest.raises(ValueError):
        spread_stats([])
    with pytest.raises(ValueError):
        spread_stats([(1.0, 1, 2), (0.5, 1, 2)])
    with pytest.raises(ValueError):
        spread_stats([(0.0, 1, 2), (2.0, 1, 2)], until=1.0)


def test_recorder_keeps_last_state_per_instant_and_orders_time():
    rec = QuoteRecorder()
    rec.record(1.0, 99, 101)
    rec.record(1.0, 98, 103)  # same instant: overwritten
    rec.record(2.0, 99, 100)
    assert rec.samples == [(0.0, None, None), (1.0, 98, 103), (2.0, 99, 100)]
    assert rec.stats(until=3.0).tw_spread == pytest.approx((5 + 1) / 2)
    with pytest.raises(ValueError):
        rec.record(1.5, 99, 100)


# -- PnL ------------------------------------------------------------------------
def test_markout_and_volume_between_on_hand_built_tape():
    trades = [trade(1, 0.0, 100, 2, buyer=1, seller=2),   # 1 buys 2 @100
              trade(2, 1.0, 104, 1, buyer=3, seller=1),   # 1 sells 1 @104
              trade(3, 2.0, 98, 1, buyer=2, seller=3)]    # 1 not involved
    value = lambda t: 101 if t < 1.0 else 106
    assert markout_pnl(trades, 1, value) == 2 * (101 - 100) - (106 - 104)
    assert markout_pnl(trades, 1, value, counterparties=[2]) == 2
    assert markout_pnl(trades, 1, value, counterparties=[3]) == -2
    assert markout_pnl(trades, 1, value, counterparties=[]) == 0
    assert volume_between(trades, 1, [2, 3]) == 3 and volume_between(trades, 2, [3]) == 1
    # zero-sum before fees: everyone's markouts add to zero
    assert sum(markout_pnl(trades, a, value) for a in (1, 2, 3)) == 0


def run_market(seed=5, decision_cost=3):
    fund = FundamentalProcess.random_walk(100, jump_rate=0.2, jump_size=3, seed=seed)
    fr = MakerTakerFee(taker_bps=20) + FixedCost(1)
    agents = [AvellanedaStoikovMM(1, gamma=0.01, sigma=1.0, horizon=120.0,
                                  inventory_limit=30, rng=seed),
              InformedTrader(2, fund, frictions=fr, max_position=15,
                             decision_cost=decision_cost),
              *make_noise_population(3, seed=seed, first_id=3, arrival_rate=0.7)]
    rec = QuoteRecorder()
    res = MarketRunner(agents, fr, seed=seed, cash=10**6).run(until=120.0, stop_when=rec)
    return res, fund, rec


def test_pnl_matches_equity_change_and_is_zero_sum_net_of_frictions():
    res, fund, _ = run_market()
    assert res.ledger.violations() == []
    mark = fund.value(res.time)
    pnl = pnl_by_agent(res, mark)
    eq = res.ledger.equity(mark)
    for aid in res.agents:  # every agent starts flat with 10**6 cash
        assert pnl[aid] == eq[aid] - 10**6
    assert sum(pnl.values()) == -(res.fees + res.decision_costs)
    assert res.fees > 0 and res.decision_costs > 0
    by_type = pnl_by_type(res, mark)
    assert set(by_type) == {"AvellanedaStoikovMM", "InformedTrader", "NoiseTrader"}
    assert sum(by_type.values()) == sum(pnl.values())


def test_recorder_tracks_the_runner_touch_and_composes_stop_when():
    res, _, rec = run_market()
    assert rec.samples[-1][1:] == (res.book.best_bid, res.book.best_ask)
    assert len(rec.samples) <= res.wakeups + 1
    st = rec.stats(until=res.time)
    assert st.tw_spread >= 1 and 0 < st.coverage <= 1
    calls = []
    stop = QuoteRecorder(lambda r: calls.append(r.wakeups) or r.wakeups >= 25)
    fr = MakerTakerFee(taker_bps=20)
    agents = make_noise_population(2, seed=1, first_id=3) + [AvellanedaStoikovMM(1, rng=1)]
    r = MarketRunner(agents, fr, seed=1, cash=10**6).run(until=1e6, stop_when=stop)
    assert r.wakeups == 25 and calls[-1] == 25 and len(stop.samples) >= 2


def test_metrics_are_deterministic_for_a_seed():
    (a, fa, ra), (b, fb, rb) = run_market(9), run_market(9)
    assert ra.samples == rb.samples
    assert pnl_by_agent(a, fa.value(a.time)) == pnl_by_agent(b, fb.value(b.time))
    assert markout_pnl(a.trades, 1, fa.value, [2]) == markout_pnl(b.trades, 1, fb.value, [2])
