"""fl-052: InfoAcquirer (Grossman-Stiglitz buy-or-not) in isolation and in seeded runs."""
from pathlib import Path

import numpy as np
import pytest

import frictionlab.agents as agents
from frictionlab import bench
from frictionlab.agents import (
    Agent, AvellanedaStoikovMM, Cancel, FundamentalProcess, InfoAcquirer, InformedTrader, MarketView,
    NoiseTrader, SignalSource, Submit, make_noise_population, make_zi_population,
)
from frictionlab.analysis import info_pnl
from frictionlab.bench.scenarios import ledger_balances
from frictionlab.core import Side
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import HIGH, INFO_SINK_ID, LOW, MakerTakerFee, two_point

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline_small.json"
MARK = 100


def view(bid=99, ask=101, depth=5, position=0):
    return MarketView(0.0, 1, bid, ask, ((bid, depth),), ((ask, depth),), None, 10**6,
                      position)


class Quoter(Agent):
    """Static liquidity: re-posts 99 x 101 with depth 5 on every wake-up."""
    wake_rate = 4.0

    def decide(self, view):
        return ([Cancel(o) for o in view.open_orders]
                + [Submit(Side.BUY, 5, 99), Submit(Side.SELL, 5, 101)])


def run(acc=9_000, price=2, decision_cost=1, n_acq=3, seed=4, wakeups=600, fseed=1):
    fund = two_point(100, 10, seed=fseed)
    src = SignalSource(fund, acc, price=price, center=100, jump=10, seed=2)
    acq = [InfoAcquirer(40 + i, src, rng=100 + i, decision_cost=decision_cost)
           for i in range(n_acq)]
    pop = ([Quoter(20)]
           + make_noise_population(3, seed=9, first_id=30) + acq)
    r = MarketRunner(pop, MakerTakerFee(taker_bps=10), seed=seed, cash=10**5, position=20,
                     check_every=True)
    return r.run(max_wakeups=wakeups), src, acq


def test_reexported_and_built_on_agent_base():
    assert {"InfoAcquirer", "SignalSource", "InfoChoice"} <= set(agents.__all__)
    src = SignalSource(FundamentalProcess.constant(110), center=100, jump=10)
    assert isinstance(InfoAcquirer(1, src), Agent)
    with pytest.raises(TypeError):
        InfoAcquirer(1, object())


@pytest.mark.parametrize("price", [1, 2, 5, 50])
def test_uninformative_source_never_bought_and_equals_noise(price):
    res, src, acq = run(acc=5_000, price=price, decision_cost=0)
    assert src.revenue == 0 and all(a.n_signals == 0 for a in acq)
    assert all(not c.bought and not c.traded for a in acq for c in a.history)
    assert INFO_SINK_ID not in res.ledger                 # sink opens on first purchase
    ref, _, _ = run(acc=5_000, price=price, decision_cost=0)
    assert ledger_balances(ref.ledger) == ledger_balances(res.ledger)


def test_uninformative_acquirer_is_bit_identical_to_a_noise_trader():
    def market(last):
        pop = ([Quoter(20)]
               + make_noise_population(2, seed=9, first_id=30) + [last])
        r = MarketRunner(pop, MakerTakerFee(taker_bps=10), seed=4, cash=10**5, position=20)
        res = r.run(max_wakeups=500)
        return [(t.price, t.qty, t.buyer_id, t.seller_id) for t in res.trades], res
    src = SignalSource(two_point(100, 10, seed=1), 5_000, price=1, center=100, jump=10)
    a, ra = market(InfoAcquirer(40, src, rng=77))
    b, rb = market(NoiseTrader(40, 77))
    assert a == b and len(a) > 20
    assert INFO_SINK_ID not in ra.ledger
    assert ledger_balances(ra.ledger) == ledger_balances(rb.ledger)


@pytest.mark.parametrize("acc,price,dc", [(6_000, 1, 0), (6_000, 0, 1), (7_500, 3, 1),
                                          (9_000, 7, 1), (9_000, 6, 2), (10_000, 9, 0),
                                          (9_000, 6, 0), (10_000, 3, 5)])
def test_never_buys_when_edge_le_hurdle(acc, price, dc):
    res, src, acq = run(acc=acc, price=price, decision_cost=dc)
    for a in acq:
        for c in a.history:
            assert c.bought == (c.expected_edge is not None and c.expected_edge > c.hurdle)
            assert c.hurdle == price + dc
    edges = {c.expected_edge for a in acq for c in a.history} - {None}
    assert edges == {src.edge(2)}                       # static 99 x 101 touch
    assert (src.revenue > 0) == (src.edge(2) > price + dc)
    assert all(c.traded for a in acq for c in a.history if c.bought)


def test_buys_and_trades_with_signal_direction():
    src = SignalSource(FundamentalProcess.constant(110), 10_000, price=2, center=100, jump=10)
    acq = InfoAcquirer(1, src, rng=0)
    assert acq.expected_edge(view()) == 9 and acq.should_buy(view())
    assert acq.informed_order(view(), HIGH) == Submit(Side.BUY, 1, 101)
    assert acq.informed_order(view(), LOW).price == 99
    assert acq.informed_order(view(position=3), HIGH) is not None
    capped = InfoAcquirer(1, src, rng=0, max_position=2)
    assert capped.informed_order(view(position=2), HIGH) is None
    assert acq.decide(view()) is not None and acq.n_signals == 0   # no venue -> no purchase


def test_signal_purchases_reconcile_with_sink_and_pnl_identity():
    res, src, acq = run(acc=9_000, price=2, decision_cost=1)
    assert sum(a.n_signals for a in acq) > 0
    assert res.ledger[INFO_SINK_ID].cash == src.revenue
    assert src.revenue == sum(src.purchases.values()) * 2
    assert all(a.signal_costs_paid == src.paid.get(a.agent_id, 0) for a in acq)
    parts = info_pnl(res, src, MARK, [a.agent_id for a in acq])
    for a in acq:
        p, acct = parts[a.agent_id], res.ledger[a.agent_id]
        equity_change = acct.cash + acct.position * MARK - (10**5 + 20 * MARK)
        assert p.net == p.trading_gain - p.signal_cost - p.decision_cost == equity_change
        assert p.signals == a.n_signals and p.decision_cost == a.decision_costs_paid
    res.ledger.check()


def test_perfect_signal_beats_the_market_maker():
    res, src, acq = run(acc=10_000, price=1, decision_cost=0, fseed=1)
    v = src.fundamental.value(0.0)
    parts = info_pnl(res, src, v, [a.agent_id for a in acq])
    assert src.hit_rate() == 1.0
    assert sum(p.trading_gain for p in parts.values()) > sum(p.signal_cost for p in parts.values())


def test_runs_are_deterministic_under_fixed_seed():
    a, sa, _ = run(seed=13)
    b, sb, _ = run(seed=13)
    assert [(t.price, t.qty, t.buyer_id) for t in a.trades] == \
        [(t.price, t.qty, t.buyer_id) for t in b.trades]
    assert sa.log == sb.log and ledger_balances(a.ledger) == ledger_balances(b.ledger)


def test_zero_acquirers_reproduce_seeded_informed_noise_scenario():
    ss = np.random.SeedSequence(38).spawn(4)
    child = lambda s: int(s.generate_state(1)[0])
    stack = MakerTakerFee(maker_bps=0, taker_bps=10)
    fund = FundamentalProcess.random_walk(100, 0.2, 1, seed=np.random.default_rng(ss[0]))
    pop = (make_zi_population("zi-c", 4, 4, seed=child(ss[1]), n_units=3)
           + [AvellanedaStoikovMM(20, rng=np.random.default_rng(ss[2])),
              InformedTrader(21, fund, frictions=stack, max_position=10)]
           + make_noise_population(2, seed=child(ss[3]), first_id=30))
    assert not any(isinstance(a, InfoAcquirer) for a in pop)        # zero acquirers
    res = MarketRunner(pop, stack, seed=38, cash=10**6, position=10).run(
        max_wakeups=int(bench.get_scenario("runner_zi_market").sizes["small"]["wakeups"]))
    work = bench.Work(res.wakeups, len(res.trades), sum(t.qty for t in res.trades),
                      ledger_balances(res.ledger))
    ref = bench.load_json(BASELINE)["scenarios"]["runner_zi_market"]
    assert work.checksum == ref["checksum"] and work.trades == ref["trades"]


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name
