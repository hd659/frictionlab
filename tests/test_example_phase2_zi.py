"""Regression tests for examples/phase2_zi_efficiency.py (ZI-C efficiency vs fees)."""
import importlib.util
import sys
from pathlib import Path

import pytest

from frictionlab.core import Side
from frictionlab.costs import FrictionStack, MakerTakerFee

PATH = Path(__file__).resolve().parents[1] / "examples" / "phase2_zi_efficiency.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase2_zi_efficiency", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(mod)
    return mod


ex = _load()


@pytest.fixture(scope="module")
def rows():
    return ex.sweep()


def test_equilibrium_on_hand_built_schedules():
    q, s, (lo, hi) = ex.equilibrium([10, 8, 6, 2], [1, 3, 7, 9])
    assert q == 2 and s == (10 - 1) + (8 - 3)
    assert lo <= hi and 3 <= lo and hi <= 7
    assert ex.equilibrium([1], [5]) == (0, 0, (0, 0))


def test_same_seed_is_bit_identical():
    assert ex.run(ex.Config(seed=4), 100) == ex.run(ex.Config(seed=4), 100)
    assert ex.run(ex.Config(seed=4), 0) != ex.run(ex.Config(seed=5), 0)


def test_default_seed_pins_regression(rows):
    r0, top = rows[0], rows[-1]
    assert (r0["q_ce"], r0["surplus_ce"], r0["p_ce_lo"], r0["p_ce_hi"]) == (18, 7872, 1040, 1045)
    assert (r0["volume"], r0["gains"], r0["fees"]) == (19, 7792, 0)
    assert (top["bps"], top["volume"], top["gains"], top["fees"]) == (400, 16, 7721, 689)


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 11])
def test_near_full_efficiency_at_zero_friction(seed):
    r = ex.run(ex.Config(seed=seed), 0)
    assert r["fees"] == 0 and r["decision_costs"] == 0
    assert r["efficiency"] == r["net_efficiency"]
    assert 0.9 <= r["efficiency"] <= 1.0
    assert r["gains"] <= r["surplus_ce"]


def test_mean_zero_friction_efficiency_is_gode_sunder_level():
    effs = [ex.run(ex.Config(seed=s), 0)["efficiency"] for s in range(1, 9)]
    assert sum(effs) / len(effs) >= 0.95


def test_fees_and_welfare_gap_grow_with_taker_fee(rows):
    fees = [r["fees"] for r in rows]
    assert fees[0] == 0 and fees == sorted(fees)
    gap = [r["efficiency"] - r["net_efficiency"] for r in rows]
    assert gap == sorted(gap)
    assert rows[-1]["net_efficiency"] < rows[0]["net_efficiency"]
    assert rows[-1]["volume"] <= rows[0]["volume"]


def test_schedules_and_equilibrium_do_not_depend_on_fees(rows):
    keys = ("q_ce", "surplus_ce", "p_ce_lo", "p_ce_hi")
    assert len({tuple(r[k] for k in keys) for r in rows}) == 1


def test_fee_aware_limits_never_lose_money_net_of_fee():
    stack = FrictionStack([MakerTakerFee(taker_bps=300)])
    for a in ex.population(ex.Config(seed=7), stack):
        for v, lim in zip(a.schedule, a.limits):
            if a.side is Side.BUY:
                assert lim + a.fee(lim) <= v
            else:
                assert lim - a.fee(lim) >= v and (lim - 1) - a.fee(lim - 1) < v


def test_no_trader_ends_with_negative_net_surplus():
    cfg = ex.Config(seed=6)
    for bps in (0, 200):
        stack = FrictionStack([MakerTakerFee(taker_bps=bps)])
        agents = ex.population(cfg, stack)
        endow = {a.agent_id: (0, a.n_units if a.side is Side.SELL else 0) for a in agents}
        res = ex.MarketRunner(agents, stack, seed=cfg.seed, endowments=endow).run(
            max_wakeups=cfg.max_wakeups, stop_when=ex._no_feasible_trade)
        for a in agents:
            acct = res.ledger[a.agent_id]
            k = ex._traded(res, a)
            net = (sum(a.schedule[:k]) + acct.cash if a.side is Side.BUY
                   else acct.cash - sum(a.schedule[:k]))
            assert net >= 0
        assert res.ledger.violations() == []


def test_prohibitive_fee_shuts_the_market_down():
    r = ex.run(ex.Config(seed=11), 5000)  # 50% taker fee > any unit's gains
    assert r["volume"] == 0 and r["gains"] == 0 and r["efficiency"] == 0.0


def test_main_prints_sweep(capsys):
    assert ex.main(["2"]) == 0
    out = capsys.readouterr().out
    assert "seed=2" in out and "CE: q*=" in out and "net_eff" in out
