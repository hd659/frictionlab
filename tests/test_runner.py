"""MarketRunner: scheduling, routing, settlement, decision costs, latency."""
import pytest

from frictionlab.agents import Agent, Cancel, Submit, make_zi_population
from frictionlab.core import Side
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import FixedCost, FrictionStack, LatencyCost, MakerTakerFee


class Scripted(Agent):
    """Emits one pre-set action list per wake-up, then idles; logs wake times."""

    def __init__(self, agent_id, script=(), decision_cost=0, wake_rate=None):
        super().__init__(agent_id, decision_cost)
        self.script, self.times = list(script), []
        if wake_rate is not None:
            self.wake_rate = wake_rate

    def decide(self, view):
        self.times.append(view.time)
        return self.script.pop(0) if self.script else []


def _zi_runner(seed=0, frictions=None, **kw):
    agents = make_zi_population("zi-c", 4, 4, seed=seed, n_units=2)
    return MarketRunner(agents, frictions, seed=seed, cash=10**6, position=10, **kw)


def test_requires_a_stopping_bound_and_valid_agents():
    with pytest.raises(ValueError):
        MarketRunner([])
    with pytest.raises(ValueError):
        MarketRunner([Scripted(1), Scripted(1)])
    with pytest.raises(ValueError):
        MarketRunner([Scripted(0)])  # venue id is reserved
    with pytest.raises(ValueError):
        MarketRunner([Scripted(1)], wake_rate=0)
    with pytest.raises(ValueError):
        MarketRunner([Scripted(1)]).run()


def test_wakeups_are_monotone_and_respect_budget():
    a, b = Scripted(1), Scripted(2)
    res = MarketRunner([a, b], seed=1, wake_rate=2.0).run(max_wakeups=50)
    assert res.wakeups == 50 and len(a.times) + len(b.times) == 50
    for ts in (a.times, b.times):
        assert ts == sorted(ts) and len(ts) > 0


def test_until_bounds_time_and_run_resumes():
    r = MarketRunner([Scripted(1)], seed=2, wake_rate=5.0)
    first = r.run(until=3.0)
    assert first.time == pytest.approx(3.0)
    w = first.wakeups
    second = r.run(until=6.0)
    assert second.time == pytest.approx(6.0) and second.wakeups > w
    assert all(t <= 6.0 for t in r.agents[1].times)


def test_per_agent_wake_rate_override():
    fast, slow = Scripted(1, wake_rate=20.0), Scripted(2, wake_rate=1.0)
    MarketRunner([fast, slow], seed=3).run(until=20.0)
    assert len(fast.times) > 5 * len(slow.times)


def test_intents_route_to_book_and_settle_through_friction():
    seller = Scripted(1, [[Submit(Side.SELL, 3, 100)]])
    buyer = Scripted(2, [[], [Submit(Side.BUY, 2)]], wake_rate=0.5)
    stack = FrictionStack([FixedCost(4)])
    r = MarketRunner([seller, buyer], stack, seed=4, cash=10_000, position=5)
    res = r.run(until=50.0)
    assert [(t.price, t.qty, t.buyer_id, t.seller_id) for t in res.trades] == [(100, 2, 2, 1)]
    assert res.ledger[2].position == 7 and res.ledger[1].position == 3
    assert res.ledger[2].cash == 10_000 - 200 - 4
    assert res.fees == 8 and res.ledger.violations() == []
    assert r.book.volume_at(Side.SELL, 100) == 1  # remainder still rests


def test_decision_costs_charged_each_wakeup_and_conserved():
    thinker = Scripted(1, decision_cost=3)
    res = MarketRunner([thinker, Scripted(2)], seed=5, cash=1_000).run(max_wakeups=40)
    assert thinker.n_decisions == len(thinker.times) > 0
    assert res.decision_costs == 3 * thinker.n_decisions == thinker.decision_costs_paid
    assert res.ledger[1].cash == 1_000 - res.decision_costs
    assert res.ledger.violations() == []


def test_decision_cost_prices_out_broke_agent():
    broke = Scripted(1, [[Submit(Side.BUY, 1, 10)]], decision_cost=5)
    r = MarketRunner([broke, Scripted(2)], seed=6, cash=12, allow_overdraft=False)
    r.run(max_wakeups=30)
    assert broke.n_decisions == 2 and broke.n_priced_out > 0
    assert r.ledger[1].cash == 2 and r.ledger.violations() == []


def test_latency_defers_execution_by_stack_delay():
    lag = 0.25
    seller = Scripted(1, [[Submit(Side.SELL, 1, 50)]], wake_rate=10.0)
    buyer = Scripted(2, [[]] * 3 + [[Submit(Side.BUY, 1)]], wake_rate=1.0)
    stack = FrictionStack([LatencyCost(lag)])
    r = MarketRunner([seller, buyer], stack, seed=7, position=5)
    res = r.run(until=30.0)
    assert len(res.trades) == 1
    assert res.trades[0].time == pytest.approx(buyer.times[3] + lag)
    assert r.delay == pytest.approx(lag)


def test_cancel_intents_clear_resting_orders():
    a = Scripted(1, [[Submit(Side.BUY, 1, 10), Submit(Side.BUY, 1, 11)]])
    r = MarketRunner([a], seed=8)
    r.run(max_wakeups=1)
    assert len(r.book) == 2
    a.script.append([Cancel(o) for o in sorted(a.open_orders)])
    r.run(max_wakeups=1)
    assert len(r.book) == 0


def test_zi_market_is_deterministic_per_seed():
    stack = MakerTakerFee(taker_bps=20) + FixedCost(1)
    tape = lambda res: [(t.time, t.price, t.qty, t.buyer_id, t.seller_id) for t in res.trades]
    a = _zi_runner(9, stack).run(max_wakeups=300)
    b = _zi_runner(9, stack).run(max_wakeups=300)
    c = _zi_runner(10, stack).run(max_wakeups=300)
    assert tape(a) == tape(b) and len(a.trades) > 0
    assert tape(a) != tape(c)
    assert [l.cash for l in a.ledger] == [l.cash for l in b.ledger]


def test_zi_invariants_hold_with_check_every():
    stack = MakerTakerFee(maker_bps=-1, taker_bps=10) + FixedCost(2)
    res = _zi_runner(11, stack, check_every=True).run(max_wakeups=400)
    assert res.ledger.violations() == []
    total_units = sum(a.n_units for a in res.agents.values())
    assert 0 < sum(t.qty for t in res.trades) <= total_units
    assert sum(a.position for a in res.ledger) == 8 * 10
    assert res.fees == sum(a.fees_paid for a in res.ledger)


def test_stop_when_predicate_halts_run():
    r = MarketRunner([Scripted(1)], seed=12)
    res = r.run(max_wakeups=1_000, stop_when=lambda m: m.wakeups >= 7)
    assert res.wakeups == 7
