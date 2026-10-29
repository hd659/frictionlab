"""frictionlab.analysis.welfare: CE surplus, friction reconciliation, zero-sum PnL."""
import itertools

import numpy as np
import pytest

from frictionlab.agents import (
    Agent, AvellanedaStoikovMM, CrossVenueArbitrageur, FundamentalProcess, InformedTrader,
    Submit, ZICTrader, make_noise_population, make_zi_population,
)
from frictionlab.analysis import (
    COST_SINK, VENUE_SINK, account_breakdown, competitive_equilibrium, pnl_breakdown,
    pnl_by_account, pnl_by_agent_type, trade_records, welfare,
)
from frictionlab.core import Side
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import FixedCost, FrictionStack, HalfSpreadCost, MakerTakerFee
from frictionlab.markets import ConstantProductAMM, batch_runner


def zi_market(seed=0, taker_bps=50, decision_cost=1, batch=False):
    agents = make_zi_population("zi-c", 6, 6, seed=seed, n_units=3, value_range=(60, 140),
                                price_range=(40, 160), decision_cost=decision_cost)
    fr = MakerTakerFee(maker_bps=0, taker_bps=taker_bps) + FixedCost(1)
    endow = {a.agent_id: (10**6, a.n_units if a.side is Side.SELL else 0) for a in agents}
    kw = dict(seed=seed, endowments=endow, check_every=True)
    r = batch_runner(agents, fr, interval=1.5, **kw) if batch else MarketRunner(agents, fr, **kw)
    return r.run(max_wakeups=600)


def brute_max_surplus(values, costs):
    best = 0
    for k in range(min(len(values), len(costs)) + 1):
        for b in itertools.combinations(values, k):
            for s in itertools.combinations(costs, k):
                best = max(best, sum(b) - sum(s))
    return best


@pytest.mark.parametrize("seed", range(5))
def test_competitive_equilibrium_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    v, c = rng.integers(1, 30, size=5).tolist(), rng.integers(1, 30, size=4).tolist()
    ce = competitive_equilibrium(v, c)
    assert ce.max_surplus == brute_max_surplus(v, c)
    assert ce.price_lo <= ce.price_hi or ce.quantity == 0
    assert competitive_equilibrium([5], [9]).quantity == 0


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_surplus_bounded_by_ce_and_reconciles_with_friction_sinks(seed, batch):
    res = zi_market(seed, batch=batch)
    w = welfare(res)
    assert res.trades and res.fees > 0 and res.decision_costs > 0
    assert 0 < w.gross_gains <= w.equilibrium.max_surplus
    assert w.surplus <= w.equilibrium.max_surplus and w.efficiency <= 1.0
    # prices are transfers between valuation agents: surplus == gains from trade
    assert w.surplus == w.gross_gains
    assert w.net_surplus + res.fees + res.decision_costs == w.gross_gains
    assert w.friction == res.fees + res.decision_costs
    assert w.net_efficiency <= w.efficiency
    # ZI-C never trades a unit at a loss before fees
    assert all(b.trading >= 0 for b in w.per_agent.values())


def test_welfare_rejects_role_violations_and_requires_valuations():
    res = zi_market(0)
    buyer = next(a for a in res.agents.values() if a.side is Side.BUY)
    fake = ZICTrader(buyer.agent_id, Side.SELL, 0, schedule=[50, 60, 70],
                     price_range=(40, 160))
    assert res.ledger[buyer.agent_id].volume > 0
    with pytest.raises(ValueError):
        welfare(res, [fake])
    with pytest.raises(ValueError):
        welfare(res, [])


def mixed_market(seed=5):
    fund = FundamentalProcess.random_walk(100, jump_rate=0.2, jump_size=3, seed=seed)
    fr = MakerTakerFee(taker_bps=20) + FixedCost(1)
    agents = [AvellanedaStoikovMM(1, gamma=0.01, sigma=1.0, horizon=120.0,
                                  inventory_limit=30, rng=seed),
              InformedTrader(2, fund, frictions=fr, max_position=15, decision_cost=3),
              *make_noise_population(3, seed=seed, first_id=3, arrival_rate=0.7)]
    res = MarketRunner(agents, fr, seed=seed, cash=10**6).run(until=120.0)
    return res, fund.value(res.time)


def test_pnl_by_type_sums_to_zero_and_matches_pnl_breakdown():
    res, mark = mixed_market()
    acct = pnl_by_account(res, mark)
    assert sum(acct.values()) == 0
    assert acct[VENUE_SINK] == res.fees > 0 and acct[COST_SINK] == res.decision_costs > 0
    by_type = pnl_by_agent_type(res, mark)
    assert sum(by_type.values()) == 0
    assert {"AvellanedaStoikovMM", "InformedTrader", "NoiseTrader"} <= set(by_type)
    ref = pnl_breakdown(res, mark)
    assert account_breakdown(res, mark) == ref
    for aid in res.agents:  # equals the equity change (agents start flat, 10**6 cash)
        assert acct[aid] == res.ledger[aid].equity(mark) - 10**6


class Maker(Agent):
    def __init__(self, quotes, agent_id=2):
        super().__init__(agent_id)
        self.quotes, self.wake_rate = list(quotes), 50.0

    def decide(self, view):
        out, self.quotes = [Submit(s, q, p) for s, q, p in self.quotes], []
        return out


def test_pnl_zero_sum_with_amm_pool_and_canonical_swap_records():
    stack = FrictionStack([FixedCost(2)])
    pool = lambda lg: ConstantProductAMM(lg, 900, 10_000, 1_000_000, fee_bps=30,
                                         frictions=stack)
    book_fr = MakerTakerFee(taker_bps=10) + HalfSpreadCost(1)
    agents = [Maker([(Side.SELL, 1000, 95)]),
              CrossVenueArbitrageur(1, frictions=book_fr, decision_cost=1)]
    res = MarketRunner(agents, book_fr, seed=3, cash=10**8, position=10**5, pools=[pool],
                       check_every=True).run(max_wakeups=200)
    recs = trade_records(res)
    assert {r.venue for r in recs} == {"book", "amm"}
    for r in recs:
        if r.venue == "amm":
            assert r.price * r.qty <= r.notional < (r.price + 1) * r.qty
    for mark in (90, 100, 110):
        acct = pnl_by_account(res, mark)
        assert 900 in acct and sum(acct.values()) == 0
        assert sum(pnl_by_agent_type(res, mark).values()) == 0
    ref = pnl_breakdown(res, 100)
    assert account_breakdown(res, 100)[2] == ref[2]  # maker only traded on the book


def test_welfare_and_pnl_deterministic_under_fixed_seed():
    a, b = zi_market(7), zi_market(7)
    wa, wb = welfare(a), welfare(b)
    assert (wa.gross_gains, wa.per_agent, wa.equilibrium) == (
        wb.gross_gains, wb.per_agent, wb.equilibrium)
    assert pnl_by_account(a, 100) == pnl_by_account(b, 100)
