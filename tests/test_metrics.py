"""frictionlab.analysis.metrics: realized spread, price impact, price efficiency."""
import math

import numpy as np
import pytest

from frictionlab.agents import (
    Agent, AvellanedaStoikovMM, FundamentalProcess, InformedTrader, Submit,
    make_noise_population, make_zi_population,
)
from frictionlab.analysis import (
    MidPath, QuoteRecorder, TradeRecord, decompose, impact_by_size, price_efficiency,
    spread_decomposition, trade_records, variance_ratio,
)
from frictionlab.core import Side, Trade
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import FixedCost, MakerTakerFee
from frictionlab.markets import batch_runner


def rec(tid, t, price, qty=1, side=Side.BUY):
    b, s = (1, 2) if side is Side.BUY else (2, 1)
    return TradeRecord.from_trade(Trade(tid, float(t), price, qty, tid, tid, b, s, side))


def run_market(seed=5):
    fund = FundamentalProcess.random_walk(100, jump_rate=0.2, jump_size=3, seed=seed)
    fr = MakerTakerFee(taker_bps=20) + FixedCost(1)
    agents = [AvellanedaStoikovMM(1, gamma=0.01, sigma=1.0, horizon=120.0,
                                  inventory_limit=30, rng=seed),
              InformedTrader(2, fund, frictions=fr, max_position=15, decision_cost=2),
              *make_noise_population(3, seed=seed, first_id=3, arrival_rate=0.7)]
    qr = QuoteRecorder()
    res = MarketRunner(agents, fr, seed=seed, cash=10**6).run(until=120.0, stop_when=qr)
    return res, fund, MidPath(qr)


@pytest.mark.parametrize("tau", [0.0, 0.5, 2.0, 10.0])
def test_effective_equals_realized_plus_impact_for_every_trade(tau):
    res, _, path = run_market()
    ms = decompose(trade_records(res), path, tau)
    assert len(ms) >= 15
    for m in ms:
        assert all(isinstance(x, int) for x in (m.effective, m.realized, m.impact))
        assert m.effective == m.realized + m.impact
    agg = spread_decomposition(trade_records(res), path, tau)
    assert agg.effective == pytest.approx(agg.realized + agg.impact)
    assert agg.n_trades + agg.n_skipped == len(res.trades)


def test_flat_mid_gives_zero_impact_and_zero_realized_at_midpoint_prints():
    path = MidPath([(0.0, 99, 101)])  # mid 100 forever
    tape = [rec(1, 1.0, 101), rec(2, 2.0, 99, side=Side.SELL), rec(3, 3.0, 100, 4),
            rec(4, 4.0, 100, side=Side.SELL)]
    ms = decompose(tape, path, tau=5.0)
    assert [m.impact for m in ms] == [0, 0, 0, 0]
    assert [m.realized for m in ms] == [m.effective for m in ms] == [2, 2, 0, 0]
    assert decompose(tape[2:], path, 1.0)[0].realized == 0
    # pre-trade mid is taken strictly before the fill; never two-sided -> skipped
    p2 = MidPath([(0.0, None, None), (1.0, 99, 101), (2.0, 100, 104), (3.0, None, 104)])
    assert decompose([rec(5, 1.0, 101)], p2, 0.0) == []
    (m,) = decompose([rec(6, 2.0, 103)], p2, 0.0)
    assert (m.effective, m.realized, m.impact) == (6, 2, 4)
    assert p2.at(3.5) == 204  # one-sided: last two-sided mid carried forward
    half = [(0.0, 9, 11), (1.0, None, 11)]
    assert MidPath(half, carry=False).at(1.0) is None and MidPath(half).at(1.0) == 20
    with pytest.raises(ValueError):
        decompose(tape, path, -1.0)


class Ladder(Agent):
    """Posts a static two-sided one-lot-per-tick ladder once, then idles."""

    def __init__(self, agent_id=1):
        super().__init__(agent_id)
        self.done = False

    def decide(self, view):
        if self.done:
            return []
        self.done = True
        return ([Submit(Side.SELL, 1, 101 + i) for i in range(200)]
                + [Submit(Side.BUY, 1, 99 - i) for i in range(90)])


class Lifter(Agent):
    def __init__(self, agent_id=2, seed=0):
        super().__init__(agent_id)
        self.rng = np.random.default_rng(seed)

    def decide(self, view):
        if view.best_ask is None:
            return []
        return [Submit(Side.BUY, int(self.rng.integers(1, 5)))]


def test_buyer_initiated_flow_against_static_book_has_positive_impact():
    qr = QuoteRecorder()
    # bound by time, not wake-ups: the budget stop skips the final touch sample
    res = MarketRunner([Ladder(), Lifter()], seed=2, cash=10**6).run(until=15.0, stop_when=qr)
    path = MidPath(qr)
    ms = decompose(trade_records(res), path, tau=1.0)
    assert ms and all(m.record.side is Side.BUY for m in ms)
    assert all(m.impact > 0 for m in ms)
    buckets = impact_by_size(ms, edges=[1, 3])
    assert sum(n for n, _ in buckets.values()) == len(ms)
    assert all(v > 0 for n, v in buckets.values() if n)


def test_impact_by_size_buckets_hand_built():
    path = MidPath([(0.0, 99, 101), (1.0, 100, 102), (2.0, 102, 104)])
    tape = [rec(1, 1.0, 101, 1), rec(2, 2.0, 103, 5)]
    out = impact_by_size(decompose(tape, path, 0.0), edges=[1, 2, 10])
    assert out[(1, 2)] == (1, 2.0) and out[(10, None)][0] == 0
    assert out[(2, 10)] == (1, 4.0)
    with pytest.raises(ValueError):
        impact_by_size([], edges=[0, 2])


def test_price_efficiency_zero_when_mid_tracks_fundamental():
    fund = FundamentalProcess.random_walk(100, jump_rate=0.5, jump_size=2, seed=4)
    pts = [(0.0, fund.value(0.0))] + fund.jumps(200.0)
    assert len(pts) > 50
    exact = MidPath([(t, v - 1, v + 1) for t, v in pts])
    st = price_efficiency(exact, fund.value, until=200.0)
    assert st.mad == 0.0 and st.coverage == 1.0
    off = MidPath([(t, v, v + 2) for t, v in pts])  # mid = V + 1 everywhere
    assert price_efficiency(off, fund.value, until=200.0).mad == pytest.approx(1.0)
    # stale mid (never updated) is penalized
    stale = MidPath([(0.0, 99, 101)])
    assert price_efficiency(stale, fund.value, until=200.0).mad > 0
    res, f, path = run_market()
    real = price_efficiency(path, f.value, until=res.time)
    assert real.mad > 0 and 0 < real.coverage <= 1


def test_variance_ratio_random_walk_vs_bounce():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.choice([-1, 1], size=5000))
    assert variance_ratio(walk, 4) == pytest.approx(1.0, abs=0.1)
    bounce = np.array([100, 101] * 500, dtype=float)
    assert variance_ratio(bounce, 2) < 0.1
    assert math.isnan(variance_ratio([1.0, 1.0, 1.0], 2))


def test_batch_fills_reduce_to_canonical_records():
    agents = make_zi_population("zi-c", 4, 4, seed=3, n_units=3, value_range=(60, 140),
                                price_range=(40, 160))
    res = batch_runner(agents, interval=2.0, seed=3).run(max_wakeups=300)
    recs = trade_records(res)
    assert recs and {r.venue for r in recs} == {"batch"}
    assert all(r.notional == r.price * r.qty for r in recs)
    assert [r.trade_id for r in recs] == [t.trade_id for t in res.trades]


def test_metrics_deterministic_under_fixed_seed():
    (a, fa, pa), (b, fb, pb) = run_market(9), run_market(9)
    assert decompose(trade_records(a), pa, 2.0) == decompose(trade_records(b), pb, 2.0)
    assert (price_efficiency(pa, fa.value, until=a.time)
            == price_efficiency(pb, fb.value, until=b.time))
