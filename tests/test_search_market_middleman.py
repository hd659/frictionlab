"""fl-049: the Middleman as an optional counterparty of the fl-046 SearchMarket."""
import hashlib

import numpy as np
import pytest

from frictionlab.agents import DECAY_SINK_ID, HOLD_SINK_ID, Middleman
from frictionlab.analysis import MiddlemanPnL, TradeRecord, middleman_pnl
from frictionlab.core import Side
from frictionlab.markets import ReservationPolicy, SearchAgent, SearchMarket
from frictionlab.markets.search_market import SEARCH_SINK_ID

MM = 500
# sha256 of fl-046 meeting rows captured before fl-049 touched search_market.py
FL046_ROWS = "d32266baaaa2f1fc2d5b8066cf3e0d191730eb6225a1fa322748a1d68f935d3e"


def population(seed, floor=1):
    rng = np.random.default_rng(seed)
    pol = ReservationPolicy.fixed()
    return ([SearchAgent(i, Side.BUY, int(v), pol) for i, v in enumerate(rng.integers(20, 120, 6), 1)]
            + [SearchAgent(100 + i, Side.SELL, int(c), pol, floor)
               for i, c in enumerate(rng.integers(20, 120, 6), 1)])


def market(seed, *, bid=55, ask=75, K=2, hold=1, decay=2, search_cost=1, rate=1.5, until=150):
    mm = Middleman(MM, bid, ask, K, holding_cost=hold, decay=decay, floor=30)
    m = SearchMarket(population(seed, 15), search_cost=search_cost, decay=3, seed=seed,
                     middleman=mm, middleman_meeting_rate=rate, cash=10_000)
    m.run(until)
    return m


def fl046_rows(**extra):
    rows = []
    for seed in range(6):
        m = SearchMarket(population(seed, 15), search_cost=2, decay=3, seed=seed, **extra)
        m.run(120)
        rows += [(x.time, x.buyer_id, x.seller_id, x.buyer_cost, x.seller_cost,
                  x.trade and x.trade.price) for x in m.meetings]
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def test_middleman_none_reproduces_fl046_rows_bit_for_bit():
    assert fl046_rows() == fl046_rows(middleman=None) == FL046_ROWS


@pytest.mark.parametrize("seed", range(10))
def test_inventory_within_capacity_after_every_step(seed):
    m = SearchMarket(population(seed, 15), search_cost=1, decay=3, seed=seed, cash=10_000,
                     middleman=Middleman(MM, 60, 70, 2, holding_cost=1), middleman_meeting_rate=2.0)
    m.run(0.0)  # schedule the processes, then step event by event
    seen = set()
    while m.clock.step() is not None and m.clock.now < 200:
        units = m.middleman.units
        assert 0 <= units <= 2 and m.ledger[MM].position == units
        seen.add(units)
    assert seen


@pytest.mark.parametrize("seed", range(10))
def test_quotes_and_reservations_bind_every_middleman_fill(seed):
    m = market(seed)
    for mt in m.mm_meetings:
        t, trader = mt.trade, m.agents[mt.seller_id if mt.buyer_id == MM else mt.buyer_id]
        if t is None:
            continue
        if t.buyer_id == MM:   # middleman buys: never above its bid nor below the seller's reservation
            assert t.price <= m.middleman.bid and t.price >= trader.reservation
        else:                  # middleman sells: never below its ask nor above the buyer's reservation
            assert t.price >= m.middleman.ask and t.price <= trader.reservation
        assert not trader.active


def test_some_seed_exercises_both_middleman_legs():
    legs = set()
    for seed in range(10):
        legs |= {t.buyer_id == MM for t in market(seed).trades if MM in (t.buyer_id, t.seller_id)}
    assert legs == {True, False}


@pytest.mark.parametrize("seed", range(8))
def test_pnl_split_is_exact_and_costs_reconcile_with_sinks(seed):
    m = market(seed, hold=2, decay=3)
    p = middleman_pnl(m)
    assert isinstance(p, MiddlemanPnL)
    assert p.net == p.spread_capture - p.holding_cost - p.decay_loss
    assert p.net == m.ledger[MM].cash - 10_000 + p.held_basis
    assert p.held_basis == sum(u.basis for u in m.middleman.inventory)
    assert p.holding_cost == m.ledger[HOLD_SINK_ID].cash == m.middleman.holding_paid
    assert p.decay_loss == m.ledger[DECAY_SINK_ID].cash == m.middleman.decay_paid
    assert p.holding_cost == 2 * sum(units for _, units, _, _ in m.carries)
    assert p.spread_capture == p.sold * (m.middleman.ask - m.middleman.bid)
    assert p.bought - p.sold == m.middleman.units == m.ledger[MM].position


def test_search_cost_charged_to_traders_only_and_sink_reconciles():
    m = market(3, search_cost=2)
    assert m.mm_meetings
    assert all(x.buyer_cost == 2 and x.seller_cost == 0 if x.seller_id == MM
               else x.seller_cost == 2 and x.buyer_cost == 0 for x in m.mm_meetings)
    assert m.ledger[SEARCH_SINK_ID].cash == m.search_costs == 2 * (2 * len(m.meetings) + len(m.mm_meetings))


@pytest.mark.parametrize("seed", range(6))
def test_ledger_invariants_and_venue_tags(seed):
    m = market(seed)
    m.ledger.check()
    recs = m.records()
    assert all(isinstance(r, TradeRecord) for r in recs) and len(recs) == len(m.trades)
    mm_ids = {x.trade.trade_id for x in m.mm_meetings if x.trade}
    assert {r.trade_id for r in recs if r.venue == "middleman"} == mm_ids
    assert all(MM in (r.buyer_id, r.seller_id) for r in recs if r.venue == "middleman")
    assert all(MM not in (r.buyer_id, r.seller_id) for r in recs if r.venue == "search")
    assert sorted(r.trade_id for r in recs) == list(range(1, len(recs) + 1))


def test_unattractive_quotes_mean_no_middleman_fill():
    m = market(2, bid=1, ask=10_000, until=300)
    assert all(x.trade is None for x in m.mm_meetings)
    assert middleman_pnl(m) == MiddlemanPnL()
    assert all(r.venue == "search" for r in m.records())


def test_deterministic_under_seed():
    def run(seed):
        m = market(seed)
        return ([(x.time, x.buyer_id, x.seller_id, x.trade and x.trade.price) for x in m.mm_meetings],
                m.carries, [(t.time, t.price) for t in m.trades], middleman_pnl(m).as_dict())
    assert run(9) == run(9)
    assert run(9) != run(10)


def test_validation():
    with pytest.raises(ValueError):
        SearchMarket(population(0), middleman=Middleman(1, 40, 60))   # id taken by a buyer
    with pytest.raises(ValueError):
        SearchMarket(population(0), middleman=Middleman(MM, 40, 60), middleman_meeting_rate=0)
    with pytest.raises(ValueError):
        middleman_pnl(SearchMarket(population(0)))
