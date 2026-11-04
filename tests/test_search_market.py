"""fl-046: decentralized bilateral search market (Phase 7)."""
from pathlib import Path

import numpy as np
import pytest

from frictionlab.analysis.metrics import TradeRecord
from frictionlab.analysis.welfare import competitive_equilibrium
from frictionlab.core import Side
from frictionlab.costs import FixedCost
from frictionlab.markets import ReservationPolicy, SearchAgent, SearchMarket
from frictionlab.markets.search_market import BPS_DENOM, SEARCH_SINK_ID

ROOT = Path(__file__).resolve().parents[1]


def make(values, costs, policy=None, floor=1):
    pol = policy or ReservationPolicy.fixed()
    return ([SearchAgent(i, Side.BUY, v, pol) for i, v in enumerate(values, 1)]
            + [SearchAgent(100 + i, Side.SELL, c, pol, floor) for i, c in enumerate(costs, 1)])


def random_market(seed, **kw):
    rng = np.random.default_rng(seed)
    return SearchMarket(make(rng.integers(20, 120, 6).tolist(), rng.integers(20, 120, 6).tolist()),
                        seed=seed, **kw)


@pytest.mark.parametrize("seed", range(8))
def test_trades_only_when_reservations_cross_and_price_in_band(seed):
    m = random_market(seed, buyer_weight_bps=int(seed * 1250), search_cost=1)
    m.run(200)
    assert m.meetings
    for mt in m.meetings:
        b, s = m.agents[mt.buyer_id], m.agents[mt.seller_id]
        if mt.trade is None:  # no decay: reservations are fixed, so they did not cross
            assert b.reservation < s.reservation
            continue
        p = mt.trade.price
        assert s.reservation <= p <= b.reservation  # values fixed: no decay
        surplus = b.reservation - s.reservation
        buyer_share, seller_share = b.reservation - p, p - s.reservation
        assert buyer_share + seller_share == surplus
        assert buyer_share == surplus * m.w // BPS_DENOM


def test_no_trade_below_either_reservation():
    pol_b, pol_s = ReservationPolicy.fixed(50), ReservationPolicy.fixed(60)
    agents = [SearchAgent(1, Side.BUY, 100, pol_b), SearchAgent(2, Side.SELL, 10, pol_s)]
    m = SearchMarket(agents, seed=1)
    m.run(50)
    assert m.trades == [] and m.exhausted()
    assert agents[0].reservation == 50 and agents[1].reservation == 60


def test_split_extremes_and_remainder_to_seller():
    m = SearchMarket(make([100], [41]), buyer_weight_bps=5_000)
    assert m.split(100, 41) == 71  # surplus 59: buyer floor(29.5) = 29, seller 30
    assert SearchMarket(make([9], [1]), buyer_weight_bps=0).split(9, 1) == 9
    assert SearchMarket(make([9], [1]), buyer_weight_bps=BPS_DENOM).split(9, 1) == 1
    with pytest.raises(ValueError):
        SearchMarket(make([9], [1]), buyer_weight_bps=BPS_DENOM + 1)


@pytest.mark.parametrize("cost", [1, 3, FixedCost(2)])
def test_search_costs_equal_meetings_times_cost_and_reconcile_with_sink(cost):
    m = SearchMarket(make([100, 90, 30, 25], [20, 50, 95, 99]), search_cost=cost, seed=7)
    m.run(100)
    per = cost if isinstance(cost, int) else 2
    assert m.search_costs == 2 * per * len(m.meetings)
    assert m.ledger[SEARCH_SINK_ID].cash == m.search_costs
    paid = sum(-m.ledger[a].cash for a in m.agents if m.agents[a].side is Side.BUY)
    recv = sum(m.ledger[a].cash for a in m.agents if m.agents[a].side is Side.SELL)
    notional = sum(t.price for t in m.trades)
    assert paid - recv == m.search_costs and paid >= notional


@pytest.mark.parametrize("seed", range(6))
def test_frictionless_volume_equals_competitive_equilibrium(seed):
    values, costs = [100, 90, 80, 20], [30, 40, 50, 110]
    m = SearchMarket(make(values, costs), seed=seed)
    m.run(10_000)
    assert m.exhausted() and m.search_costs == 0
    assert len(m.trades) == competitive_equilibrium(values, costs).quantity == 3


def test_perishable_value_never_below_floor():
    agents = [SearchAgent(1, Side.BUY, 10), SearchAgent(2, Side.SELL, 100, floor=35)]
    m = SearchMarket(agents, decay=7, decay_interval=1.0, seed=4)
    seen = []
    for t in range(1, 30):
        m.run(float(t))
        seen.append(agents[1].value)
    assert min(seen) >= 35 and seen[-1] == 35 and m.trades == []
    assert all(a - b in (0, 7) or b == 35 for a, b in zip(seen, seen[1:]))


def test_waiting_has_a_cost_perished_seller_trades_lower():
    agents = [SearchAgent(1, Side.BUY, 40), SearchAgent(2, Side.SELL, 100, floor=35)]
    m = SearchMarket(agents, decay=7, meeting_rate=0.5, seed=4)
    m.run(500)
    assert len(m.trades) == 1 and 35 <= m.trades[0].price <= 40
    assert m.trades[0].time >= 9.0  # 100 - 7 * 9 = 37 is the first value <= 40
    assert 35 <= agents[1].value <= 40 and m.exhausted()
    m.ledger.check()


def test_stopping_policy_thresholds():
    pol = ReservationPolicy.stopping(50, 100, 5)
    rb, rs = pol.threshold(Side.BUY), pol.threshold(Side.SELL)
    n = 51
    assert sum(rb - x for x in range(50, rb + 1)) >= 5 * n > sum(rb - 1 - x for x in range(50, rb))
    assert sum(x - rs for x in range(rs, 101)) >= 5 * n > sum(x - rs - 1 for x in range(rs + 1, 101))
    assert pol.reservation(Side.BUY, 60) == 60 and pol.reservation(Side.SELL, 90) == 90
    zero = ReservationPolicy.stopping(50, 100, 0)
    assert zero.threshold(Side.BUY) == 50 and zero.threshold(Side.SELL) == 100
    with pytest.raises(ValueError):
        ReservationPolicy.stopping(10, 5, 1)


def test_ledger_invariants_and_trade_records():
    for seed in range(5):
        m = random_market(seed, search_cost=2, decay=3)
        m.run(100)
        m.ledger.check()
        assert sum(a.position for a in m.ledger) == sum(
            1 for a in m.agents.values() if a.side is Side.SELL)
        recs = m.records()
        assert all(isinstance(r, TradeRecord) and r.venue == "search" and r.qty == 1 for r in recs)
        assert [r.notional for r in recs] == [t.price for t in m.trades]
        assert all(m.ledger[r.buyer_id].position == 1 for r in recs)


def test_deterministic_under_seed():
    def run(seed):
        m = random_market(seed, search_cost=1, decay=2)
        m.run(150)
        return [(x.time, x.buyer_id, x.seller_id, x.trade and x.trade.price) for x in m.meetings]
    assert run(11) == run(11)
    assert run(11) != run(12)


def test_validation():
    with pytest.raises(ValueError):
        SearchMarket(make([10], [5]), meeting_rate=0)
    with pytest.raises(ValueError):
        SearchMarket([SearchAgent(1, Side.BUY, 10), SearchAgent(1, Side.SELL, 5)])
    with pytest.raises(ValueError):
        SearchMarket([SearchAgent(SEARCH_SINK_ID, Side.BUY, 10)])


def test_module_under_hygiene_line_ceiling():
    n = len((ROOT / "frictionlab/markets/search_market.py").read_text().splitlines())
    assert n <= 250
