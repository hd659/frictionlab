import dataclasses

import pytest

from frictionlab.agents import (
    COST_SINK_ID, ActionError, Agent, Cancel, MarketView, Submit, Venue,
)
from frictionlab.core import Ledger, LedgerError, OrderBook, Side
from frictionlab.costs import FixedCost


class Scripted(Agent):
    """Replays a fixed list of action batches, one batch per decide()."""

    def __init__(self, agent_id, script=(), decision_cost=0):
        super().__init__(agent_id, decision_cost)
        self.script = list(script)
        self.seen = []

    def decide(self, view):
        self.seen.append(view)
        return self.script.pop(0) if self.script else []


def venue(frictions=None, **kw):
    lg = Ledger(frictions, check_every=True, **kw)
    lg.open_account(1, cash=100_000, position=50)
    lg.open_account(2, cash=100_000, position=50)
    return Venue(OrderBook(), lg)


def test_market_view_is_read_only_snapshot():
    v = venue()
    maker = Scripted(1, [[Submit(Side.SELL, 5, 1010), Submit(Side.BUY, 3, 990)]])
    maker.step(v, 0.0)
    view = v.view(maker, 1.0)
    assert (view.best_bid, view.best_ask, view.spread, view.mid) == (990, 1010, 20, 1000.0)
    assert view.asks == ((1010, 5),) and view.bids == ((990, 3),)
    assert view.last_trade is None and len(view.open_orders) == 2
    assert (view.cash, view.position) == (100_000, 50)
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.cash = 0


def test_submit_intent_matches_and_settles_through_ledger():
    v = venue()
    Scripted(1, [[Submit(Side.SELL, 5, 1000)]]).step(v, 0.0)
    taker = Scripted(2, [[Submit(Side.BUY, 2)]])
    trades = taker.step(v, 1.0)
    assert [(t.price, t.qty, t.buyer_id, t.seller_id) for t in trades] == [(1000, 2, 2, 1)]
    assert (v.ledger[2].cash, v.ledger[2].position) == (98_000, 52)
    assert v.view(taker, 2.0).last_trade == trades[0]
    assert not taker.open_orders  # market order never rests
    v.ledger.check()


def test_decision_cost_charged_per_decide_to_cost_sink():
    v = venue()
    a = Scripted(1, decision_cost=7)
    for t in range(3):
        a.step(v, float(t))
    assert (a.n_decisions, a.decision_costs_paid) == (3, 21)
    assert v.ledger[1].cash == 100_000 - 21
    assert v.decision_costs_collected == v.ledger[COST_SINK_ID].cash == 21
    assert v.ledger.fees_collected == 0  # decision cost is not a venue fee
    v.ledger.check()


def test_cash_conserved_with_both_fees_and_decision_costs():
    v = venue(FixedCost(4))
    Scripted(1, [[Submit(Side.SELL, 5, 1000)]], decision_cost=3).step(v, 0.0)
    Scripted(2, [[Submit(Side.BUY, 5, 1000)]], decision_cost=5).step(v, 1.0)
    lg = v.ledger
    assert lg.fees_collected == 8 and v.decision_costs_collected == 8
    assert sum(a.cash for a in lg) + lg.venue.cash == lg.cash_in
    lg.check()


def test_agent_priced_out_when_it_cannot_afford_to_think():
    v = venue(allow_overdraft=False)
    v.ledger.open_account(3, cash=10)
    a = Scripted(3, [[Submit(Side.BUY, 1, 900)]], decision_cost=6)
    a.step(v, 0.0)
    assert a.step(v, 1.0) == [] and a.n_priced_out == 1
    assert (a.n_decisions, len(a.seen), v.ledger[3].cash) == (1, 1, 4)
    v.ledger.check()


def test_state_dependent_deliberation_cost_override():
    class Pricey(Scripted):
        def deliberation_cost(self, view):
            return 2 * len(view.open_orders) + 1

    v = venue()
    a = Pricey(1, [[Submit(Side.SELL, 1, 1100), Submit(Side.SELL, 1, 1200)]])
    a.step(v, 0.0)
    a.step(v, 1.0)
    assert a.decision_costs_paid == 1 + 5
    bad = Pricey(2)
    bad.deliberation_cost = lambda view: -1
    with pytest.raises(ValueError):
        bad.step(v, 2.0)


def test_cancel_intent_and_open_order_tracking():
    v = venue()
    a = Scripted(1, [[Submit(Side.BUY, 4, 990)]])
    a.step(v, 0.0)
    (oid,) = a.open_orders
    a.script = [[Cancel(oid, qty=1)], [Cancel(oid)]]
    a.step(v, 1.0)
    assert v.book.volume_at(Side.BUY, 990) == 3
    a.step(v, 2.0)
    assert v.book.best_bid is None
    assert v.view(a, 3.0).open_orders == () and not a.open_orders


def test_filled_orders_drop_out_of_open_orders():
    v = venue()
    maker = Scripted(1, [[Submit(Side.SELL, 2, 1000)]])
    maker.step(v, 0.0)
    Scripted(2, [[Submit(Side.BUY, 2, 1000)]]).step(v, 1.0)
    assert v.view(maker, 2.0).open_orders == ()


def test_cannot_cancel_another_agents_order():
    v = venue()
    owner = Scripted(1, [[Submit(Side.SELL, 2, 1000)]])
    owner.step(v, 0.0)
    (oid,) = owner.open_orders
    with pytest.raises(ActionError):
        Scripted(2, [[Cancel(oid)]]).step(v, 1.0)
    assert oid in v.book


def test_intent_validation():
    with pytest.raises(ValueError):
        Submit(Side.BUY, 0, 100)
    with pytest.raises(TypeError):
        Submit(Side.BUY, 1, 100.5)
    with pytest.raises(ValueError):
        Cancel(1, qty=0)
    with pytest.raises(ValueError):
        Scripted(1, decision_cost=-1)
    assert Submit(1, 1).side is Side.BUY and Submit(Side.SELL, 1).price is None
    with pytest.raises(TypeError):
        Agent(1)  # abstract


def test_ledger_transfer_is_atomic_internal_move():
    lg = Ledger(allow_overdraft=False)
    lg.open_account(1, cash=5)
    lg.open_account(2)
    with pytest.raises(LedgerError):
        lg.transfer(1, 2, 6)
    with pytest.raises(LedgerError):
        lg.transfer(1, 1, 1)
    lg.transfer(1, 2, 5)
    assert (lg[1].cash, lg[2].cash, lg.cash_in, lg.transferred) == (0, 5, 5, 5)
    lg.check()
    assert isinstance(MarketView(0.0, 1, None, None, (), (), None, 0, 0).spread, type(None))
