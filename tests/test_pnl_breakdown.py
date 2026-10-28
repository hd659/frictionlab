"""Unit tests for frictionlab.analysis.pnl (trading / fee / decision-cost split)."""
from types import SimpleNamespace

import pytest

from frictionlab.agents import (
    AvellanedaStoikovMM, BoundedRationalAgent, make_noise_population,
)
from frictionlab.analysis import (
    PnLBreakdown, breakdown_by, pnl_breakdown, pnl_by_agent, total,
)
from frictionlab.core import Side, Trade
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import MakerTakerFee


def trade(tid, price, qty, buyer, seller):
    return Trade(tid, float(tid), price, qty, 100 + tid, 200 + tid, buyer, seller, Side.BUY)


def fake_result(trades, fees, costs, valuations=None):
    agents = {aid: SimpleNamespace(agent_id=aid, decision_costs_paid=c,
                                   valuation=(valuations or {}).get(aid))
              for aid, c in costs.items()}
    ledger = {aid: SimpleNamespace(fees_paid=f) for aid, f in fees.items()}
    return SimpleNamespace(trades=trades, agents=agents, ledger=ledger)


def test_hand_built_breakdown_at_common_mark():
    # agent 1 buys 2 @ 98 and sells 1 @ 104; agent 2 is the other side
    res = fake_result([trade(1, 98, 2, 1, 2), trade(2, 104, 1, 2, 1)],
                      fees={1: 3, 2: -1}, costs={1: 5, 2: 0})
    b = pnl_breakdown(res, 100)
    assert b[1] == PnLBreakdown(trading=2 * 2 + 4, fees=3, decision_costs=5, volume=3)
    assert b[2] == PnLBreakdown(trading=-8, fees=-1, decision_costs=0, volume=3)
    assert b[1].net == 0 and b[1].friction == 8 and b[2].net == -7
    assert b[1].trading + b[2].trading == 0  # zero-sum at a common mark
    assert b[1].as_dict() == {"trading": 8, "fees": 3, "decision_costs": 5, "net": 0,
                              "volume": 3, "n_agents": 1}


def test_mark_forms_mapping_and_callable_agree():
    res = fake_result([trade(1, 98, 2, 1, 2)], fees={1: 0, 2: 0}, costs={1: 0, 2: 0},
                      valuations={1: 110, 2: 90})
    by_val = pnl_breakdown(res, lambda a: a.valuation)
    by_map = pnl_breakdown(res, {1: 110, 2: 90})
    assert by_val == by_map
    # buyer surplus 2 * (110 - 98), seller surplus 2 * (98 - 90)
    assert (by_val[1].trading, by_val[2].trading) == (24, 16)
    assert pnl_breakdown(res, 100, agents=[2]) == {2: PnLBreakdown(-4, 0, 0, 2)}
    with pytest.raises(KeyError):
        pnl_breakdown(res, 100, agents=[3])


def test_total_and_grouping():
    a, b = PnLBreakdown(5, 1, 2, 3), PnLBreakdown(-1, 0, 4, 1)
    t = total([a, b])
    assert t == PnLBreakdown(4, 1, 6, 4, n_agents=2) and t.net == -3
    assert total([]) == PnLBreakdown(n_agents=0)
    res = fake_result([trade(1, 98, 2, 1, 2), trade(2, 101, 1, 3, 2)],
                      fees={1: 1, 2: 1, 3: 1}, costs={1: 0, 2: 2, 3: 0})
    groups = breakdown_by(res, 100, key=lambda ag: "buyer" if ag.agent_id != 2 else "seller")
    assert groups["buyer"].n_agents == 2 and groups["buyer"].trading == 4 - 1
    # seller: 2 @ 98 (-4) and 1 @ 101 (+1) at mark 100, fee 1, thinking 2
    assert (groups["seller"].trading, groups["seller"].net) == (-3, -6)
    assert groups["seller"] == pnl_breakdown(res, 100)[2]


def run_market(seed=3, cps=2):
    brs = [BoundedRationalAgent(10 + i, Side(1 - 2 * (i % 2)), seed * 100 + i,
                                mode=("satisficing", "optimizing")[i // 2], max_steps=8,
                                cost_per_step=cps, value_range=(95, 115),
                                max_position=20, wake_rate=0.5) for i in range(4)]
    agents = [AvellanedaStoikovMM(1, gamma=0.01, sigma=1.0, horizon=150.0,
                                  inventory_limit=40, rng=seed),
              *make_noise_population(3, seed=seed, first_id=2, arrival_rate=0.5), *brs]
    return MarketRunner(agents, MakerTakerFee(taker_bps=50), seed=seed,
                        cash=10**6).run(until=150.0)


def test_runner_output_reconciles_with_ledger_and_spread_pnl():
    res = run_market()
    res.ledger.check()
    b = pnl_breakdown(res, 100)
    assert {k: v.net for k, v in b.items()} == pnl_by_agent(res, 100)
    t = total(b.values())
    assert t.trading == 0
    assert t.fees == res.fees and t.decision_costs == res.decision_costs > 0
    assert t.net == -(res.fees + res.decision_costs)
    assert t.volume == 2 * sum(tr.qty for tr in res.trades)
    br = [res.agents[a] for a in range(10, 14)]
    assert all(b[a.agent_id].decision_costs == 2 * a.steps_taken for a in br)
    by_type = breakdown_by(res, 100)
    assert by_type["BoundedRationalAgent"] == total(b[a.agent_id] for a in br)
    assert sum(g.n_agents for g in by_type.values()) == len(res.agents)


def test_private_valuation_surplus_for_bounded_rational_agents():
    res = run_market(seed=5, cps=0)
    ids = range(10, 14)
    priv = pnl_breakdown(res, lambda a: a.valuation, ids)
    for aid in ids:
        a = res.agents[aid]
        expect = sum((a.valuation - t.price) * t.qty if t.buyer_id == aid else
                     (t.price - a.valuation) * t.qty
                     for t in res.trades if aid in (t.buyer_id, t.seller_id))
        assert priv[aid].trading == expect and priv[aid].decision_costs == 0
