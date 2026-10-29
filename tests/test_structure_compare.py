"""structure_compare: effective spread, snipe share, batch vs continuous runs."""
import math

import pytest

from frictionlab.agents import FundamentalProcess, make_zi_population
from frictionlab.analysis.structure_compare import (
    compare, delta, effective_spread, is_snipe, jump_times_of, structure_stats, summary,
)
from frictionlab.core import Side, Trade
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import MakerTakerFee
from frictionlab.markets import batch_runner


def tr(tid, t, price, qty, aggr):
    return Trade(tid, t, price, qty, 1, 2, 10, 20, aggr)


def step_value(t):
    return 100 if t < 5.0 else 104   # one jump at t = 5


def test_effective_spread_sign_and_volume_weighting():
    buy = tr(1, 1.0, 101, 1, Side.BUY)    # buyer pays 1 over V -> spread 2
    sell = tr(2, 1.0, 98, 3, Side.SELL)   # seller gets 2 under V -> spread 4
    assert effective_spread(buy, step_value) == 2
    assert effective_spread(sell, step_value) == 4
    s = structure_stats([buy, sell], step_value, name="x")
    assert s.volume == 4 and s.n_trades == 2
    assert s.avg_effective_spread == pytest.approx((2 * 1 + 4 * 3) / 4)
    assert s.avg_price == pytest.approx((101 + 98 * 3) / 4)


def test_snipe_requires_recent_jump_and_aggressor_gain():
    jumps = [5.0]
    stale = tr(1, 5.2, 100, 2, Side.BUY)      # buys at old value after up-jump
    late = tr(2, 7.0, 100, 1, Side.BUY)       # outside window
    fair = tr(3, 5.3, 105, 1, Side.BUY)       # pays above new value: no gain
    before = tr(4, 4.9, 99, 1, Side.BUY)      # before the jump
    assert is_snipe(stale, step_value, jumps, 1.0)
    assert not any(is_snipe(t, step_value, jumps, 1.0) for t in (late, fair, before))
    s = structure_stats([stale, late, fair, before], step_value, jump_times=jumps, window=1.0)
    assert s.snipe_volume == 2 and s.snipe_share == pytest.approx(2 / 5)
    with pytest.raises(ValueError):
        structure_stats([], step_value, window=-1)


def test_empty_tape_and_delta_and_summary():
    e = structure_stats([], step_value, name="empty")
    assert e.volume == 0 and e.snipe_share == 0.0 and math.isnan(e.avg_effective_spread)
    a = structure_stats([tr(1, 1.0, 101, 1, Side.BUY)], step_value, name="a")
    b = structure_stats([tr(2, 1.0, 101, 3, Side.BUY)], step_value, name="b")
    d = delta(a, b)
    assert d["volume"] == 2 and d["avg_effective_spread"] == 0
    text = summary({"a": a, "empty": e})
    assert "structure" in text and "nan" in text and len(text.splitlines()) == 3


def test_jump_times_of_fundamental_process():
    assert jump_times_of(FundamentalProcess.constant(100), 50.0) == []
    v = FundamentalProcess.random_walk(100, jump_rate=1.0, seed=1)
    jt = jump_times_of(v, 20.0)
    assert jt and jt == sorted(jt) and all(0 < t <= 20.0 for t in jt)


def _runs(seed=5):
    kw = dict(seed=seed, cash=10**6, position=20, check_every=True)
    fees = MakerTakerFee(0, 10)
    cont = MarketRunner(make_zi_population("zi-c", 6, 6, seed=seed, n_units=3), fees, **kw)
    batch = batch_runner(make_zi_population("zi-c", 6, 6, seed=seed, n_units=3), fees,
                         interval=1.0, **kw)
    return {"continuous": cont.run(until=40.0), "batch": batch.run(until=40.0)}


def test_compare_batch_and_continuous_runs():
    runs = _runs()
    v = FundamentalProcess.random_walk(100, jump_rate=0.5, seed=2)
    jt = jump_times_of(v, 40.0)
    stats = compare(runs, v.value, jump_times=jt, window=1.0)
    for name, res in runs.items():
        s = stats[name]
        res.ledger.check()
        assert s.name == name and s.volume == sum(t.qty for t in res.trades) > 0
        assert 0.0 <= s.snipe_share <= 1.0 and math.isfinite(s.avg_effective_spread)
    # every batch fill prints at its batch's single clearing price
    by_time = {}
    for t in runs["batch"].trades:
        by_time.setdefault(t.time, set()).add(t.price)
    assert all(len(p) == 1 for p in by_time.values())
    again = compare(_runs(), v.value, jump_times=jt, window=1.0)
    assert again == stats  # deterministic under a fixed seed
