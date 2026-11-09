"""Tests for fl-070 `MenuCostMarket`: menu sink, exact PnL split, streams, ledger."""
from __future__ import annotations

from pathlib import Path

import pytest

from frictionlab import bench
from frictionlab.agents import SsPricer
from frictionlab.analysis import MenuPnL, TradeRecord, menu_pnl
from frictionlab.costs import MENU_SINK_ID, MenuCost
from frictionlab.markets import DEMAND_ID, MENU, MenuCostMarket

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline_small.json"
SHOCKS = (-3, -1, 0, 0, 1, 3)


def _market(bands=(0, 4, 10), seed=70, cost=5, drift=1, shocks=SHOCKS, steps=120,
            curvature=2, base=50):
    sellers = [SsPricer(i + 1, b, drift=drift) for i, b in enumerate(bands)]
    return MenuCostMarket(sellers, [100 + 10 * i for i in range(len(bands))], steps=steps,
                          inflation_ticks_per_step=drift, shocks=shocks, menu=cost,
                          curvature=curvature, base_profit=base, seed=seed)


def test_frictionless_limit_tracks_target():
    m = _market(bands=(0, 0), cost=0, drift=0, shocks=(0,))
    m.run()
    assert all(r.price == r.target and r.loss == 0 for r in m.log)
    assert m.adjustments == 0 and m.menu.revenue == 0
    assert all(p.misalignment == 0 and p.net == 50 * 120 for p in menu_pnl(m).values())


def test_flexible_pricer_tracks_moving_target():
    m = _market(bands=(0,), cost=3)
    m.run()
    assert all(r.gap == 0 and r.loss == 0 for r in m.log)
    assert m.adjustments == sum(1 for r in m.log if r.gap_before != 0)


def test_adjustments_reconcile_with_menu_sink():
    m = _market()
    m.run()
    assert m.adjustments > 0
    assert m.menu.adjustments == m.adjustments
    assert m.ledger[MENU_SINK_ID].cash == 5 * m.adjustments == m.menu.revenue
    for i in m.sellers:
        assert m.menu.adjustments_of(i) == sum(r.adjusted for r in m.path(i))


@pytest.mark.parametrize("seed", [1, 2, 70])
def test_pnl_split_exact_in_ticks(seed):
    m = _market(seed=seed)
    m.run()
    rows = menu_pnl(m)
    for i, p in rows.items():
        assert isinstance(p, MenuPnL)
        assert p.frictionless == 50 * 120
        assert p.net == p.frictionless - p.misalignment - p.menu_cost
        assert p.net == m.ledger[i].cash
        assert p.as_dict()["adjustments"] == p.adjustments
    assert sum(p.menu_cost for p in rows.values()) == m.ledger[MENU_SINK_ID].cash


def test_misalignment_loss_nonnegative_and_zero_iff_aligned():
    m = _market(bands=(3, 8), curvature=3)
    m.run()
    for r in m.log:
        assert r.loss >= 0 and r.loss == 3 * r.gap ** 2
        assert (r.loss == 0) == (r.gap == 0)
    assert any(r.loss > 0 for r in m.log)


def test_shock_draws_independent_of_other_sellers_band():
    a, b = _market(bands=(0, 4, 10)), _market(bands=(0, 4, 30))
    a.run()
    b.run()
    assert a.shock_draws[1] == b.shock_draws[1] and a.shock_draws[2] == b.shock_draws[2]
    assert a.path(1) == b.path(1)
    assert a.path(3) != b.path(3)


def test_ledger_invariants_and_records():
    m = _market()
    m.run()
    m.ledger.check()
    assert sum(acct.cash for acct in m.ledger) == 0
    recs = m.records()
    assert len(recs) == len(m.log) == 3 * 120
    assert all(isinstance(r, TradeRecord) and r.venue == MENU for r in recs)
    assert all(r.buyer_id == DEMAND_ID for r in recs)
    assert sum(r.notional for r in recs) == -m.ledger[DEMAND_ID].cash


def test_deterministic_and_idempotent():
    a, b = _market(seed=9), _market(seed=9)
    assert a.run() == b.run() and a.run() == a.adjustments
    assert a.log == b.log
    c = _market(seed=10)
    c.run()
    assert c.shock_draws != a.shock_draws


def test_menu_cost_component_and_bad_inputs():
    mc = MenuCost(7)
    assert mc.cost_for(1) == 7 and mc.adjustments == 0
    with pytest.raises(ValueError):
        MenuCost(-1)
    with pytest.raises(ValueError):
        _market(shocks=())
    with pytest.raises(ValueError):
        MenuCostMarket([SsPricer(1)], [100, 101], steps=5)
    with pytest.raises(ValueError):
        MenuCostMarket([SsPricer(1), SsPricer(1)], [100, 101], steps=5)


@pytest.mark.parametrize("name", list(bench.SCENARIOS))
def test_fl038_checksums_unchanged(name):
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert bench.get_scenario(name).run("small").checksum == ref["checksum"], name


def test_new_modules_under_line_ceiling():
    for rel in ("frictionlab/markets/menu_cost.py", "frictionlab/costs/menu.py",
                "frictionlab/agents/ss_pricer.py", "frictionlab/analysis/pnl.py"):
        assert len((ROOT / rel).read_text("utf-8").splitlines()) <= 250, rel
