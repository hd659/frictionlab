"""Regression tests for examples/phase2_decision_cost.py (satisficing vs optimizing)."""
import importlib.util
import sys
from pathlib import Path

import pytest

from frictionlab.agents import BayesianLearner, BoundedRationalAgent, ZICTrader
from frictionlab.analysis import pnl_breakdown

PATH = Path(__file__).resolve().parents[1] / "examples" / "phase2_decision_cost.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase2_decision_cost", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(mod)
    return mod


ex = _load()
SMALL = ex.Config(markets=3)


@pytest.fixture(scope="module")
def rows():
    return ex.sweep(ex.Config())


def test_ledger_invariants_hold():
    for mode in ex.MODES:
        for cost in (0, 3):
            res = ex.simulate(SMALL, mode, cost, market=1)
            res.ledger.check()
            assert res.ledger.violations() == []
            brs = [res.agents[a] for a in ex.BR_IDS]
            assert all(isinstance(a, BoundedRationalAgent) and a.mode == mode for a in brs)
            assert sum(isinstance(a, ZICTrader) for a in res.agents.values()) == 2 * SMALL.n_zi
            assert res.decision_costs == cost * sum(a.steps_taken for a in brs)
            assert all(p.steps <= SMALL.max_steps for a in brs for p in a.history)
            b = pnl_breakdown(res, ex.valuation, ex.BR_IDS)
            assert all(b[a.agent_id].decision_costs == a.decision_costs_paid for a in brs)
            assert sum(t.qty for t in res.trades) > 0 and res.fees == 0


def test_results_are_reproducible_from_a_seed():
    a = ex.run_mode(SMALL, "optimizing", 2)
    assert a == ex.run_mode(SMALL, "optimizing", 2)
    assert a != ex.run_mode(ex.Config(seed=16, markets=3), "optimizing", 2)
    r1, r2 = ex.simulate(SMALL, "satisficing", 1, 0), ex.simulate(SMALL, "satisficing", 1, 0)
    key = lambda r: [(t.price, t.qty, t.buyer_id, t.seller_id, t.time) for t in r.trades]
    assert key(r1) == key(r2)


def test_modes_share_valuations_and_thinking_is_metered(rows):
    sat = ex.simulate(SMALL, "satisficing", 0, 2)
    opt = ex.simulate(SMALL, "optimizing", 0, 2)
    assert [sat.agents[a].valuation for a in ex.BR_IDS] == \
        [opt.agents[a].valuation for a in ex.BR_IDS]
    for r in rows:
        s, o = r["satisficing"], r["optimizing"]
        assert s["steps"] < o["steps"]
        assert (s["decision"], o["decision"]) == (r["cost"] * s["steps"], r["cost"] * o["steps"])
        assert s["net"] == s["gross"] - s["fees"] - s["decision"]
        # step cost never feeds back into prices: gross is flat across the sweep
        assert (s["gross"], o["gross"]) == (rows[0]["satisficing"]["gross"],
                                            rows[0]["optimizing"]["gross"])


def test_default_seed_pins_regression(rows):
    s, o = rows[0]["satisficing"], rows[0]["optimizing"]
    assert (s["gross"], s["steps"]) == (2730214, 79003)
    assert (o["gross"], o["steps"]) == (2738040, 95800)
    assert ex.crossover(rows) == 1


@pytest.mark.parametrize("seed", (15, 1, 2, 3))
def test_at_zero_cost_optimizer_net_is_at_least_satisficer(rows, seed):
    r = rows[0] if seed == 15 else ex.sweep(ex.Config(seed=seed), (0,))[0]
    assert r["cost"] == 0
    assert r["optimizing"]["net"] >= r["satisficing"]["net"]


@pytest.mark.parametrize("seed", (15, 1, 3, 8))
def test_some_step_cost_makes_the_satisficer_win(rows, seed):
    sweep = rows if seed == 15 else ex.sweep(ex.Config(seed=seed))
    c = ex.crossover(sweep)
    assert c is not None and c in ex.COSTS
    # past the crossover the gap only widens: net is linear in cost with
    # slope -steps, and the optimizer always takes more steps
    gaps = [r["satisficing"]["net"] - r["optimizing"]["net"] for r in sweep]
    assert all(b > a for a, b in zip(gaps, gaps[1:]))
    assert gaps[-1] > 0


def test_bayesian_learner_converges_under_informed_flow():
    for value in (92, 108, 121):
        b, res = ex.learn(value, seed=1)
        assert isinstance(b, BayesianLearner) and b.n_trades_sent == 0
        assert b.posterior.sum() == pytest.approx(1.0) and (b.posterior >= 0).all()
        assert abs(b.mean - value) < 1.5 and b.std < 1.5 and abs(b.map_value - value) <= 1
        early = abs(b.mean_path[0] - value)
        assert abs(b.mean - value) < max(early, 1.5)
        res.ledger.check()


def test_main_prints_sweep_and_learner(capsys):
    assert ex.main(["15"]) == 0
    out = capsys.readouterr().out
    assert "seed=15" in out and "crossover cost_per_step=1" in out
    assert "BayesianLearner" in out and len(out.splitlines()) == 4 + len(ex.COSTS)
