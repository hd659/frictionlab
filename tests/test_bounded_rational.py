import math

import pytest

from frictionlab.agents import (
    AvellanedaStoikovMM, BoundedRationalAgent, Cancel, MarketView, Submit,
    make_noise_population,
)
from frictionlab.core import Side
from frictionlab.core.ledger import Ledger
from frictionlab.core.order_book import OrderBook
from frictionlab.core.runner import MarketRunner
from frictionlab.agents.base import Venue
from frictionlab.costs import MakerTakerFee


def view(bid=95, ask=105, position=0, open_orders=()):
    bids = ((bid, 5),) if bid is not None else ()
    asks = ((ask, 5),) if ask is not None else ()
    return MarketView(0.0, 1, bid, ask, bids, asks, None, 10**6, position,
                      tuple(open_orders), ())


def agent(mode, seed=0, side=Side.BUY, **kw):
    kw.setdefault("valuation", 100)
    kw.setdefault("aspiration", 2.0)
    return BoundedRationalAgent(1, side, seed, mode=mode, **kw)


def paired(seed, **kw):
    return agent("satisficing", seed, **kw), agent("optimizing", seed, **kw)


def test_steps_never_exceed_max_steps():
    for mode in ("satisficing", "optimizing"):
        for max_steps in (1, 3, 8):
            a = agent(mode, seed=max_steps, max_steps=max_steps, aspiration=1e9)
            for _ in range(20):
                plan = a.deliberate(view())
                assert 1 <= plan.steps <= max_steps and len(plan.evaluated) == plan.steps
            assert agent(mode, max_steps=max_steps).deliberate(view()).steps <= max_steps
    capped = agent("optimizing", max_position=2)
    assert capped.deliberate(view(position=2)).steps == 0
    assert capped.decide(view(position=2, open_orders=(4,))) == [Cancel(4)]
    with pytest.raises(ValueError):
        agent("greedy")
    with pytest.raises(ValueError):
        agent("optimizing", max_steps=0)


def test_satisficer_uses_no_more_steps_and_optimizer_gross_dominates():
    stopped_early = 0
    for seed in range(60):
        # agent values the asset above the ask (buyer) / below the bid (seller)
        for side, v in ((Side.BUY, view(90, 98)), (Side.SELL, view(102, 110))):
            sat, opt = paired(seed, side=side, max_steps=12, kappa=0.3)
            for _ in range(5):  # rng streams stay aligned across calls
                ps, po = sat.deliberate(v), opt.deliberate(v)
                assert ps.steps <= po.steps == 12
                assert ps.evaluated == po.evaluated[:ps.steps]  # same candidates
                assert po.gross >= ps.gross
                assert po.gross == max(es for _, es in po.evaluated)
                stopped_early += ps.steps < po.steps
                if ps.satisfied:
                    assert ps.gross >= sat.aspiration
                    assert all(es < sat.aspiration for _, es in ps.evaluated[:-1])
    assert stopped_early > 100


def test_expected_surplus_model():
    a = agent("optimizing", kappa=0.5)
    v = view(95, 105)
    assert a.fill_prob(v, 105) == 1.0 and a.fill_prob(v, 110) == 1.0
    assert a.fill_prob(v, 101) == pytest.approx(math.exp(-2.0))
    assert a.expected_surplus(v, 98) == pytest.approx(2 * math.exp(-3.5))
    assert a.expected_surplus(v, 104) < 0  # marketable but above valuation
    s = agent("optimizing", side=Side.SELL, kappa=0.0)
    assert s.expected_surplus(v, 103) == pytest.approx(3.0)
    assert a.fill_prob(view(None, None), 100) == 1.0  # empty book: ref = v


def test_charged_cost_equals_steps_times_cost_per_step():
    for mode in ("satisficing", "optimizing"):
        for cps in (0, 1, 3):
            a = BoundedRationalAgent(7, Side.BUY, 11, mode=mode, valuation=100,
                                     max_steps=9, cost_per_step=cps, aspiration=1.5)
            venue = Venue(OrderBook(), Ledger())
            venue.ledger.open_account(7, cash=10**6)
            for t in range(15):
                before = venue.decision_costs_collected
                a.step(venue, float(t))
                plan = a.last
                assert plan.cost == plan.steps * cps == a.decision_cost
                assert venue.decision_costs_collected - before == plan.cost
            assert a.decision_costs_paid == a.steps_taken * cps
            assert venue.decision_costs_collected == a.decision_costs_paid
            venue.ledger.check()


def test_priced_out_agent_does_not_act():
    a = BoundedRationalAgent(7, Side.BUY, 0, valuation=100, mode="optimizing",
                             max_steps=5, cost_per_step=4)
    led = Ledger(allow_overdraft=False)
    led.open_account(7, cash=10)  # can afford exactly zero 20-tick thoughts
    venue = Venue(OrderBook(), led)
    assert a.step(venue, 0.0) == [] and a.n_priced_out == 1
    assert a.history == [] and venue.decision_costs_collected == 0
    assert len(venue.book.depth(Side.BUY, 5)) == 0
    led.check()


def test_submits_chosen_price_when_surplus_positive():
    a = agent("optimizing", kappa=0.0, max_steps=6)
    acts = a.decide(view(open_orders=(3,)))
    plan = a.last
    assert acts[0] == Cancel(3)
    if plan.submits:
        assert acts[1:] == [Submit(Side.BUY, 1, plan.price)]
        assert plan.price < a.valuation and plan.gross == a.valuation - plan.price


def run_market(seed, mode, cps=1):
    brs = [BoundedRationalAgent(10 + i, Side(1 - 2 * (i % 2)), seed * 100 + i,
                                mode=mode, max_steps=8, cost_per_step=cps,
                                aspiration=1.0, value_range=(95, 115),
                                max_position=20, wake_rate=0.5) for i in range(4)]
    agents = [AvellanedaStoikovMM(1, gamma=0.01, sigma=1.0, horizon=150.0,
                                  inventory_limit=40, rng=seed),
              *make_noise_population(3, seed=seed, first_id=2, arrival_rate=0.5), *brs]
    fees = MakerTakerFee(taker_bps=10)
    return MarketRunner(agents, fees, seed=seed, cash=10**6, check_every=True).run(until=150.0)


def test_ledger_invariants_hold_through_runner():
    for mode in ("satisficing", "optimizing"):
        r = run_market(3, mode, cps=2)
        r.ledger.check()
        brs = [a for a in r.agents.values() if isinstance(a, BoundedRationalAgent)]
        assert all(a.steps_taken > 0 for a in brs)
        assert r.decision_costs == sum(a.decision_costs_paid for a in r.agents.values())
        assert sum(a.decision_costs_paid for a in brs) == 2 * sum(a.steps_taken for a in brs)
        assert all(p.steps <= 8 for a in brs for p in a.history)
        assert len(r.trades) > 10


def test_deterministic_under_fixed_seed():
    for mode in ("satisficing", "optimizing"):
        r1, r2 = run_market(7, mode), run_market(7, mode)
        key = lambda r: [(t.price, t.qty, t.buyer_id, t.seller_id, t.time) for t in r.trades]
        assert key(r1) == key(r2)
        for aid in range(10, 14):
            assert r1.agents[aid].history == r2.agents[aid].history
            assert r1.agents[aid].valuation == r2.agents[aid].valuation
    sat, opt = run_market(7, "satisficing"), run_market(7, "optimizing")
    steps = lambda r: sum(r.agents[a].steps_taken for a in range(10, 14))
    assert steps(sat) < steps(opt)
