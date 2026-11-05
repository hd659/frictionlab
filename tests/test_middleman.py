"""fl-049: Middleman agent in isolation (quotes, capacity, carry costs, PerishCost)."""
from pathlib import Path

import pytest

import frictionlab.agents as agents
from frictionlab.agents import DECAY_SINK_ID, HOLD_SINK_ID, Agent, Middleman, PerishCost
from frictionlab.core import Side
from frictionlab.core.ledger import Ledger
from frictionlab.costs import FixedCost
from frictionlab.costs.base import Fill, Role
from frictionlab.costs.fees import HalfSpreadCost

ROOT = Path(__file__).resolve().parents[1]


def ledger_for(mm, cash=0):
    led = Ledger()
    for i, c in ((mm.agent_id, cash), (mm.hold_sink_id, 0), (mm.decay_sink_id, 0)):
        led.open_account(i, c)
    return led


def test_reexported_and_built_on_agent_base():
    assert "Middleman" in agents.__all__ and agents.Middleman is Middleman
    mm = Middleman(500, 40, 60, 2)
    assert isinstance(mm, Agent) and mm.decide(None) == ()
    assert (mm.hold_sink_id, mm.decay_sink_id) == (HOLD_SINK_ID, DECAY_SINK_ID)


def test_quote_acceptance_respects_capacity_and_reservations():
    mm = Middleman(500, 40, 60, capacity=2)
    assert not mm.sells_to(100)                      # empty: nothing to sell
    assert mm.meet(Side.SELL, 41, 0.0) is None       # seller wants more than the bid
    assert mm.meet(Side.SELL, 40, 0.0) == 40 and mm.meet(Side.SELL, 10, 1.0) == 40
    assert mm.units == 2 and not mm.buys_from(1)     # at capacity K
    assert mm.meet(Side.BUY, 59, 2.0) is None        # buyer below the ask keeps searching
    assert mm.meet(Side.BUY, 60, 2.0) == 60 and mm.units == 1
    assert (mm.bought, mm.sold, mm.basis_sold) == (2, 1, 40)


def test_fifo_release_and_errors():
    mm = Middleman(500, 30, 35, capacity=1)
    with pytest.raises(ValueError):
        mm.give()
    mm.take(0.0)
    with pytest.raises(ValueError):
        mm.take(0.0)
    assert mm.give() == 35 and mm.units == 0


def test_alive_is_false_only_when_no_fill_can_ever_happen():
    mm = Middleman(500, 40, 60, capacity=1)
    assert mm.alive([70], [30])            # can buy, then sell
    assert mm.alive([50], [30]) and mm.alive([], [40])  # a buy is still a fill
    assert not mm.alive([70], [41])        # empty and no seller ever reaches the bid
    mm.take(0.0)
    assert mm.alive([60], [])              # full: can still sell
    assert not mm.alive([59], [10])        # full and no buyer at the ask


def test_perish_cost_is_floor_bounded():
    pc = PerishCost(7, floor=35)
    f = lambda v: Fill(1, Side.SELL, Role.MAKER, v, 1)  # noqa: E731
    assert [pc.charge(f(v)) for v in (100, 40, 35, 36)] == [7, 5, 0, 1]
    with pytest.raises(ValueError):
        PerishCost(1, floor=0)


@pytest.mark.parametrize("holding", [3, FixedCost(3), HalfSpreadCost(6, roles="maker")])
def test_carry_routes_holding_and_decay_to_sinks(holding):
    mm = Middleman(500, 50, 70, capacity=3, holding_cost=holding, decay=4, floor=43)
    led = ledger_for(mm, cash=1_000)
    mm.take(0.0), mm.take(0.5)
    rows = [mm.carry(led, float(t)) for t in range(1, 4)]
    assert rows == [(6, 8), (6, 6), (6, 0)]                 # 50 -> 46 -> 43 (floor) -> 43
    assert [u.value for u in mm.inventory] == [43, 43] and [u.basis for u in mm.inventory] == [50, 50]
    assert led[HOLD_SINK_ID].cash == mm.holding_paid == 18
    assert led[DECAY_SINK_ID].cash == mm.decay_paid == 14
    assert led[500].cash == 1_000 - 32
    led.check()


def test_empty_carry_charges_nothing():
    mm = Middleman(500, 50, 70, holding_cost=5, decay=5)
    led = ledger_for(mm)
    assert mm.carry(led, 1.0) == (0, 0) and led.transferred == 0


def test_validation():
    with pytest.raises(ValueError):
        Middleman(500, 60, 40)                       # crossed quotes
    with pytest.raises(ValueError):
        Middleman(500, 40, 60, capacity=0)
    with pytest.raises(ValueError):
        Middleman(500, 40, 60, holding_cost=-1)
    with pytest.raises(ValueError):
        Middleman(HOLD_SINK_ID, 40, 60)              # id collides with its own sink
    with pytest.raises(TypeError):
        Middleman(500, 40.5, 60)


def test_module_under_hygiene_line_ceiling():
    for rel in ("frictionlab/agents/middleman.py", "frictionlab/markets/search_market.py",
                "frictionlab/analysis/pnl.py"):
        assert len((ROOT / rel).read_text().splitlines()) <= 250, rel
