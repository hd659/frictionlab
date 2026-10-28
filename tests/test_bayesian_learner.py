import numpy as np
import pytest

from frictionlab.agents import (
    AvellanedaStoikovMM, BayesianLearner, Cancel, FundamentalProcess, InformedTrader,
    MarketView, Submit, make_noise_population, tick_grid, trade_likelihood,
)
from frictionlab.core import Side, Trade
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import HalfSpreadCost, MakerTakerFee

GRID = tick_grid(90, 120)


def stack():
    # taker at ~100: 10 bps -> 0.1, half-spread 1 -> total ceil(1.1) = 2 per lot
    return MakerTakerFee(taker_bps=10) + HalfSpreadCost(2)


def tr(tid, price, side, buyer=7, seller=8):
    return Trade(tid, float(tid), price, 1, 1000 + tid, 2000 + tid, buyer, seller, Side(side))


def view(bid=99, ask=101, trades=(), position=0, depth=5, open_orders=()):
    bids = ((bid, depth),) if bid is not None else ()
    asks = ((ask, depth),) if ask is not None else ()
    tape = tuple(trades)
    return MarketView(0.0, 1, bid, ask, bids, asks, tape[-1] if tape else None,
                      10**6, position, tuple(open_orders), tape)


def learner(**kw):
    kw.setdefault("frictions", stack())
    return BayesianLearner(1, GRID, **kw)


def informed_flow(v, mu, n, seed):
    """Seeded GM flow: prices uniform on the grid, informed w.p. mu, else 50/50 noise."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = int(rng.integers(GRID[0], GRID[-1] + 1))
        if rng.random() < mu and v != p:
            side = Side.BUY if v > p else Side.SELL
        else:
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
        out.append((p, side))
    return out


def test_posterior_stays_normalized_and_non_negative():
    b = learner(mu=0.6)
    for p, s in informed_flow(104, 0.6, 400, seed=1):
        b.update(p, s)
        assert np.all(b.posterior >= 0) and b.posterior.sum() == pytest.approx(1.0, abs=1e-12)
    lik = trade_likelihood(GRID, 100, Side.BUY, 0.6, 0.5)
    assert lik.min() == pytest.approx(0.2) and lik.max() == pytest.approx(0.8)
    with pytest.raises(ValueError):
        BayesianLearner(1, [3, 2, 1])
    with pytest.raises(ValueError):
        BayesianLearner(1, GRID, prior=-np.ones(GRID.size))


def test_buy_flow_raises_and_sell_flow_lowers_mean():
    up, down, m0 = learner(), learner(), learner().mean
    assert m0 == pytest.approx(105.0)
    for _ in range(5):
        up.update(105, Side.BUY)
        down.update(105, Side.SELL)
    assert up.mean > m0 > down.mean
    assert up.mean - m0 == pytest.approx(m0 - down.mean)  # symmetric grid, eta = 0.5


def test_mu_zero_posterior_equals_prior():
    g = BayesianLearner.gaussian(1, 105, 15, sd=4.0, mu=0.0)
    for p, s in informed_flow(112, 0.9, 300, seed=3):
        g.update(p, s)
    np.testing.assert_allclose(g.posterior, g.prior, rtol=0, atol=1e-12)
    assert g.mean == pytest.approx(105.0)


def test_posterior_concentrates_on_true_value():
    for v in (97, 108, 115):
        b = learner(mu=0.4)
        for p, s in informed_flow(v, 0.4, 3000, seed=v):
            b.update(p, s)
        near = b.posterior[np.abs(GRID - v) <= 1].sum()
        assert abs(b.map_value - v) <= 1 and near > 0.9
        assert abs(b.mean - v) < 1.0 and b.std < 1.5


def test_tape_dedupe_skips_own_fills_and_contradictions():
    b = learner(mu=0.5)
    tape = [tr(1, 100, Side.BUY), tr(2, 100, Side.BUY, buyer=1)]
    b.decide(view(trades=tape))
    b.decide(view(trades=tape))  # same tape again: nothing new
    assert b.n_updates == 1 and b.last_seen_trade == 2
    hard = BayesianLearner(1, tick_grid(100, 102), mu=1.0)
    assert hard.update(105, Side.BUY) is False and hard.n_ignored == 1
    assert hard.posterior.sum() == pytest.approx(1.0)


def test_no_trade_when_edge_le_friction():
    flat = BayesianLearner(1, tick_grid(90, 110), frictions=stack())  # mean 100
    assert flat.decide(view(99, 101)) == []  # edge -1 both sides
    for decision_cost in range(5):  # GRID mean 105 -> buy edge 4, friction 2 + dc
        b = learner(decision_cost=decision_cost)
        need = 2 + decision_cost
        sent = [a for a in b.decide(view(99, 101)) if isinstance(a, Submit)]
        assert bool(sent) == (4 > need)  # dc = 2 is the tie: no trade
        for side, px in ((Side.BUY, 101), (Side.SELL, 99)):
            q = b.evaluate(view(99, 101), side)
            assert q.friction == need
            assert q.gross_edge == side.value * (105 - px)
            assert q.profitable == (q.gross_edge > need)
    go = learner(decision_cost=1)
    assert go.decide(view(99, 101, open_orders=(5,))) == [Cancel(5), Submit(Side.BUY, 1, 101)]
    sell = learner()
    for i in range(40):
        sell.update(100, Side.SELL)
    assert sell.mean < 96 and sell.decide(view(99, 101)) == [Submit(Side.SELL, 1, 99)]


def run_market(seed):
    fund = FundamentalProcess.constant(108)
    agents = [AvellanedaStoikovMM(1, gamma=0.01, sigma=1.0, horizon=200.0,
                                  inventory_limit=50, rng=seed),
              InformedTrader(2, fund, frictions=stack(), max_position=20),
              *make_noise_population(4, seed=seed, first_id=3, arrival_rate=0.5),
              BayesianLearner(9, GRID, mu=0.3, frictions=stack(), max_position=10)]
    return MarketRunner(agents, stack(), seed=seed, cash=10**7).run(until=200.0)


def test_runs_are_deterministic_under_fixed_seed():
    r1, r2 = run_market(5), run_market(5)
    key = lambda r: [(t.price, t.qty, t.buyer_id, t.seller_id, t.time) for t in r.trades]
    assert key(r1) == key(r2) and len(r1.trades) > 20
    b1, b2 = r1.agents[9], r2.agents[9]
    np.testing.assert_array_equal(b1.posterior, b2.posterior)
    assert b1.mean_path == b2.mean_path and b1.n_updates > 0
    assert all(q is None or q.profitable for q in b1.history)
    r1.ledger.check()
