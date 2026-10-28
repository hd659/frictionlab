import numpy as np
import pytest

from frictionlab.agents import (
    Cancel, MarketView, Submit, Venue, ZICTrader, ZIUTrader, make_zi_population,
)
from frictionlab.core import Ledger, OrderBook, Side

PR = (1, 200)


def view(agent_id=1, position=0, open_orders=()):
    return MarketView(0.0, agent_id, None, None, (), (), None, 10_000, position,
                      tuple(open_orders))


def run_market(kind, seed, n=6, steps=400, **kw):
    agents = make_zi_population(kind, n, n, seed=seed, n_units=3,
                                value_range=(60, 140), price_range=PR, **kw)
    lg = Ledger(check_every=True)
    for a in agents:
        lg.open_account(a.agent_id, cash=100_000, position=10)
    v = Venue(OrderBook(), lg)
    order = np.random.default_rng(seed).integers(0, len(agents), size=steps)
    for t, i in enumerate(order):
        agents[i].step(v, float(t))
    lg.check()
    return agents, v


def fills(v, agent):
    key = "buyer_id" if agent.side is Side.BUY else "seller_id"
    return [t.price for t in v.book.trades if getattr(t, key) == agent.agent_id]


def test_zic_buyer_never_bids_above_value_and_seller_never_asks_below_cost():
    buyer = ZICTrader(1, Side.BUY, 7, schedule=[90], price_range=PR)
    seller = ZICTrader(2, Side.SELL, 8, schedule=[110], price_range=PR)
    bids = [buyer.decide(view(1))[-1].price for _ in range(2_000)]
    asks = [seller.decide(view(2))[-1].price for _ in range(2_000)]
    assert PR[0] <= min(bids) and max(bids) <= 90
    assert 110 <= min(asks) and max(asks) <= PR[1]
    assert max(bids) == 90 and min(asks) == 110  # bounds are inclusive


def test_ziu_ignores_its_schedule_and_spans_the_full_range():
    buyer = ZIUTrader(1, Side.BUY, 7, schedule=[90], price_range=PR)
    bids = [buyer.decide(view(1))[-1].price for _ in range(2_000)]
    assert min(bids) >= PR[0] and max(bids) <= PR[1]
    assert any(b > 90 for b in bids)  # would bid above its own value


def test_zic_never_trades_a_unit_at_a_loss_in_a_live_market():
    agents, v = run_market("zi-c", seed=3)
    assert v.book.trades, "market should clear some volume"
    for a in agents:
        prices = fills(v, a)
        assert len(prices) == a.units_traded <= a.n_units
        for unit, p in zip(a.schedule, prices):
            assert a.side.value * (unit - p) >= 0
        assert a.surplus(prices) >= 0


def test_ziu_market_can_produce_loss_making_trades():
    agents, v = run_market("zi-u", seed=3)
    losses = [a.side.value * (u - p) for a in agents
              for u, p in zip(a.schedule, fills(v, a))]
    assert losses and min(losses) < 0


def test_seed_determinism_for_schedules_quotes_and_trades():
    def signature(seed):
        agents, v = run_market("zi-c", seed=seed)
        return ([a.schedule for a in agents], [a.quotes for a in agents],
                [(t.price, t.buyer_id, t.seller_id) for t in v.book.trades])
    assert signature(11) == signature(11)
    assert signature(11) != signature(12)
    a = ZICTrader(1, Side.BUY, 5, n_units=4)
    b = ZICTrader(1, Side.BUY, np.random.default_rng(5), n_units=4)
    assert a.schedule == b.schedule


def test_schedule_ordering_and_value_range():
    buyer = ZICTrader(1, Side.BUY, 0, n_units=20, value_range=(70, 80))
    seller = ZICTrader(2, Side.SELL, 0, n_units=20, value_range=(70, 80))
    assert list(buyer.schedule) == sorted(buyer.schedule, reverse=True)
    assert list(seller.schedule) == sorted(seller.schedule)
    assert all(70 <= u <= 80 for u in buyer.schedule + seller.schedule)


def test_default_decision_cost_is_zero_and_nothing_reaches_cost_sink():
    agents, v = run_market("zi-u", seed=1)
    assert all(a.decision_cost == 0 and a.decision_costs_paid == 0 for a in agents)
    assert v.decision_costs_collected == 0
    assert sum(a.n_decisions for a in agents) == 400


def test_decide_emits_typed_intents_and_replaces_resting_quote():
    t = ZICTrader(1, Side.SELL, 0, schedule=[100], price_range=PR)
    acts = t.decide(view(1, open_orders=(4, 9)))
    assert acts[:2] == [Cancel(4), Cancel(9)]
    assert isinstance(acts[2], Submit) and acts[2].qty == 1 and acts[2].side is Side.SELL


def test_trader_goes_quiet_once_schedule_is_exhausted():
    t = ZICTrader(1, Side.BUY, 0, schedule=[100, 90], price_range=PR)
    t.observe(view(1, position=5))
    assert t.unit_value == 100
    t.observe(view(1, position=6))
    assert (t.units_traded, t.unit_value) == (1, 90)
    t.observe(view(1, position=7, open_orders=(3,)))
    assert t.exhausted and t.decide(t.view) == [Cancel(3)]


def test_population_ids_sides_and_independent_streams():
    pop = make_zi_population("ZI-C", 2, 3, seed=0, first_id=10, n_units=5)
    assert [a.agent_id for a in pop] == [10, 11, 12, 13, 14]
    assert [a.side for a in pop] == [Side.BUY] * 2 + [Side.SELL] * 3
    assert len({a.schedule for a in pop}) == 5
    assert all(isinstance(a, ZICTrader) for a in pop)


def test_validation_errors():
    with pytest.raises(ValueError):
        make_zi_population("zi-x", 1, 1)
    with pytest.raises(ValueError):
        ZICTrader(1, Side.BUY, 0, schedule=[300], price_range=PR)
    with pytest.raises(ValueError):
        ZIUTrader(1, Side.BUY, 0, price_range=(10, 5))
    with pytest.raises(ValueError):
        ZIUTrader(1, Side.BUY, 0, schedule=[])
    with pytest.raises(TypeError):
        ZIUTrader(1, Side.BUY, 0, schedule=[1.5])
