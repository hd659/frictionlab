"""FundingArbitrageur: delta-neutral perp/spot basis trade under seeded gas shocks."""
import pytest

from frictionlab.agents import Agent, FundingArbitrageur, Submit
from frictionlab.analysis.pnl import total
from frictionlab.core import Side
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import TAKER_ONLY, FrictionStack, GasProcess, MakerTakerFee
from frictionlab.markets import ConstantProductAMM, PerpMarket

ARB, MAKER, POOL = 1, 2, 900
X, Y, FEE = 10_000, 1_000_000, 30  # spot 100 ticks, 30 bps LP fee
CASH = 10**8


class Maker(Agent):
    """Posts one deep two-sided ladder so the perp mark is pinned at the mid."""

    def __init__(self, bid, ask, depth=5_000):
        super().__init__(MAKER)
        self.quotes, self.wake_rate = [(Side.BUY, bid), (Side.SELL, ask)], 50.0
        self.depth = depth

    def decide(self, view):
        out = [Submit(s, self.depth, p) for s, p in self.quotes]
        self.quotes = []
        return out


def run(bid=104, ask=106, gas=None, seed=7, until=80.0, decision_cost=0, horizon=6,
        max_size=50, taker_bps=10):
    gas = gas or GasProcess.constant(5, seed=seed)
    stack = MakerTakerFee(taker_bps=taker_bps) + gas.cost(TAKER_ONLY)
    pool_stack = FrictionStack([gas.cost(TAKER_ONLY)])
    pf = lambda lg: ConstantProductAMM(lg, POOL, X, Y, fee_bps=FEE, frictions=pool_stack)
    # perp is bound after the runner has built the shared ledger
    arb = FundingArbitrageur(ARB, None, POOL, gas=gas, horizon=horizon, max_size=max_size,
              wake_rate=2.0, decision_cost=decision_cost)
    runner = MarketRunner([Maker(bid, ask), arb], stack, seed=seed, cash=CASH,
                          position=10**5, pools=[pf], check_every=True)
    perp = PerpMarket(runner.ledger, lambda t: 100, runner.book, k=1, cap_bps=75, interval=4.0)
    perp.attach(runner.clock)
    arb.perp = perp
    res = runner.run(until=until)
    return res, arb, perp


def test_opens_short_perp_long_spot_when_funding_positive():
    res, arb, perp = run()
    assert arb.n_entries >= 1 and arb.report()
    for p in arb.report():
        assert p.direction == Side.SELL.value
    assert all(e.rate_bps > 0 for e in perp.history)


def test_opens_long_perp_short_spot_when_funding_negative():
    res, arb, perp = run(bid=94, ask=96)
    assert arb.n_entries >= 1
    assert all(p.direction == Side.BUY.value for p in arb.report())


def test_never_opens_when_expected_funding_within_friction():
    for kw in ({"bid": 99, "ask": 101}, {"decision_cost": 10**6}, {"horizon": 1}):
        res, arb, _ = run(**kw)
        assert arb.n_entries == 0 and not res.arbs


def test_entry_quotes_clear_total_friction_with_four_gas_legs():
    _, arb, _ = run()
    opened = [q for q in arb.history if q is not None and q.friction["gas"] == 4 * 5]
    assert opened
    for q in opened:
        assert q.net_edge > 0 and q.expected_funding > q.total_friction
        assert set(q.friction) == {"perp_fee", "perp_spread", "lp_fee", "price_impact",
                                   "gas", "decision_cost"}


def test_net_delta_zero_while_open():
    res, arb, perp = run()
    perp.sync()
    assert arb.net_delta == 0
    assert perp.position(ARB) == arb.perp_pos
    assert res.ledger[ARB].position - 10**5 == arb.net_delta
    for a in res.arbs:
        assert a.hedged


def test_shock_gas_blocks_entries_and_is_monotone():
    blocked = GasProcess(5, 10**5, p_shock=0.0, p_calm=0.0, start=1, seed=7)
    assert run(gas=blocked)[1].n_entries == 0
    counts = [run(gas=GasProcess(5, g, p_shock=0.5, p_calm=0.05, dt=1.0, seed=7),
                  until=120.0)[1].n_entries for g in (5, 8, 12, 40, 400, 10**5)]
    assert counts[0] >= 1 and counts[-1] < counts[0]  # shocks do bind on this seed
    assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_funding_fees_gas_reconcile_with_ledger_exactly():
    gas = GasProcess(3, 40, p_shock=0.1, p_calm=0.3, dt=1.0, seed=11)
    res, arb, perp = run(gas=gas, until=120.0, decision_cost=1)
    pos = arb.report()
    assert pos and len(arb.positions) >= 1
    cash_delta = res.ledger[ARB].cash - CASH
    pnl = total(p.pnl for p in pos)
    assert cash_delta == pnl.trading - pnl.fees - arb.decision_costs_paid
    received = sum(ev.payments.get(ARB, 0) for ev in perp.history)
    assert sum(p.funding for p in pos) == received
    assert sum(p.fees for p in pos) == res.ledger[ARB].fees_paid
    assert sum(p.gas for p in pos) == sum(p.friction["gas"] for p in pos)
    for p in pos:
        assert p.pnl.net == p.trading + p.funding - p.fees - p.decision_costs
        assert 0 <= p.gas <= p.fees


def test_ledger_invariants_through_runner():
    res, arb, perp = run(gas=GasProcess(2, 30, 0.2, 0.3, seed=3), until=100.0)
    res.ledger.check()
    assert perp.open_interest == perp.short_interest
    assert perp.sink_cash == sum(ev.remainder for ev in perp.history) >= 0
    pool = res.pools[POOL]
    assert pool.k >= X * Y


def test_deterministic_under_fixed_seed():
    def sig(seed):
        res, arb, perp = run(gas=GasProcess(2, 30, 0.2, 0.3, seed=seed), seed=seed)
        return ([(t.price, t.qty, t.buyer_id) for t in res.trades],
                [(p.qty, p.funding, p.fees, p.trading) for p in arb.report()],
                [e.payments for e in perp.history], [a.cash for a in res.ledger])
    assert sig(5) == sig(5)


def test_validation():
    with pytest.raises(ValueError):
        FundingArbitrageur(ARB, None, POOL, horizon=0)
    with pytest.raises(ValueError):
        FundingArbitrageur(ARB, None, POOL, wake_rate=0)
