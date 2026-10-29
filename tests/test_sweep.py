"""Tests for frictionlab.analysis.sweep (seeded, order-independent friction sweeps)."""
import importlib.util
import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from frictionlab.agents import ZICTrader
from frictionlab.analysis import SweepSpec, aggregate, expand_grid, friction_stack, run_sweep, write_csv
from frictionlab.core.runner import MarketRunner
from frictionlab.core.types import Side

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def zi_market(params, seed):
    """Small seeded ZI-C market; module-level so worker processes can unpickle it."""
    if params.get("boom"):
        raise ValueError("bad cell")
    stack = friction_stack(params)
    streams = np.random.SeedSequence(seed).spawn(6)
    agents = [ZICTrader(i + 1, Side.BUY if i < 3 else Side.SELL, np.random.default_rng(ss),
                        n_units=2, value_range=(50, 150), price_range=(1, 200),
                        decision_cost=params.get("decision_cost", 0))
              for i, ss in enumerate(streams)]
    endow = {a.agent_id: (0, 2 if a.side is Side.SELL else 0) for a in agents}
    return MarketRunner(agents, stack, seed=seed, endowments=endow)


def market_metrics(res):
    return {"volume": sum(t.qty for t in res.trades), "fees": res.fees,
            "decision_costs": res.decision_costs, "wakeups": res.wakeups}


def make_spec(**kw):
    kw.setdefault("grid", {"taker_bps": [0, 100], "fixed_cost": [0, 2]})
    return SweepSpec(zi_market, market_metrics, replicates=kw.pop("replicates", 3),
                     master_seed=kw.pop("master_seed", 7), run_kwargs={"max_wakeups": 300}, **kw)


@pytest.fixture(scope="module")
def rows():
    return run_sweep(make_spec())


def test_grid_is_full_cartesian_product_in_stable_order():
    grid = {"taker_bps": [0, 50, 100], "latency": [0.0, 0.5], "decision_cost": [0, 1]}
    pts = expand_grid(grid)
    assert len(pts) == 3 * 2 * 2 and len({tuple(p.values()) for p in pts}) == 12
    assert pts[0] == {"taker_bps": 0, "latency": 0.0, "decision_cost": 0}
    assert pts[1] == {"taker_bps": 0, "latency": 0.0, "decision_cost": 1}
    assert pts[-1] == {"taker_bps": 100, "latency": 0.5, "decision_cost": 1}
    assert expand_grid(grid) == pts
    spec = SweepSpec(zi_market, market_metrics, grid=grid, base={"maker_bps": 0})
    assert spec.points == [{"maker_bps": 0, **p} for p in pts]


def test_rows_are_tidy_and_ordered(rows):
    assert len(rows) == 4 * 3
    assert [(r["cell"], r["replicate"]) for r in rows] == [(c, k) for c in range(4) for k in range(3)]
    assert all(r["error"] is None for r in rows)
    assert {(r["taker_bps"], r["fixed_cost"]) for r in rows} == {(0, 0), (0, 2), (100, 0), (100, 2)}
    assert all(r["fees"] == 0 for r in rows if r["taker_bps"] == 0 and r["fixed_cost"] == 0)
    assert any(r["fees"] > 0 for r in rows if r["taker_bps"] == 100)


def test_serial_and_parallel_rows_are_identical(rows):
    assert run_sweep(make_spec(), n_workers=2) == rows


def test_same_master_seed_is_bit_identical(rows):
    again = run_sweep(make_spec())
    assert again == rows and write_csv(again) == write_csv(rows)
    other = run_sweep(make_spec(master_seed=8))
    assert [r["seed"] for r in other] != [r["seed"] for r in rows]


def test_child_seeds_distinct_and_order_free():
    spec = make_spec(replicates=5)
    seeds = spec.child_seeds()
    flat = [s for cell in seeds for s in cell]
    assert len(flat) == len(set(flat)) == 4 * 5
    assert all(len(set(cell)) == 5 for cell in seeds)
    assert seeds == make_spec(replicates=5).child_seeds()
    crn = make_spec(replicates=5, crn=True).child_seeds()
    assert all(cell == crn[0] for cell in crn) and len(set(crn[0])) == 5
    pinned = make_spec(seeds=[11, 12]).child_seeds()
    assert pinned == [[11, 12]] * 4


def test_failing_cell_becomes_error_row_without_aborting():
    spec = SweepSpec(zi_market, market_metrics, cells=[{"taker_bps": 0}, {"boom": 1}, {"taker_bps": 50}],
                     replicates=2, run_kwargs={"max_wakeups": 100})
    for n_workers in (1, 2):
        out = run_sweep(spec, n_workers=n_workers)
        assert len(out) == 6
        bad = [r for r in out if r["error"] is not None]
        assert [r["cell"] for r in bad] == [1, 1]
        assert all(r["error"] == "ValueError: bad cell" and "volume" not in r for r in bad)
        assert all("volume" in r for r in out if r["error"] is None)
    agg = aggregate(out, by=("cell",))
    assert [(a["n"], a["errors"]) for a in agg] == [(2, 0), (0, 2), (2, 0)]


def test_aggregate_matches_manual_mean_std(rows):
    agg = aggregate(rows, by=("taker_bps", "fixed_cost"))
    assert len(agg) == 4
    for a in agg:
        grp = [r for r in rows if (r["taker_bps"], r["fixed_cost"]) == (a["taker_bps"], a["fixed_cost"])]
        assert a["n"] == 3 and a["errors"] == 0
        for m in ("volume", "fees", "wakeups"):
            xs = [float(r[m]) for r in grp]
            mean = sum(xs) / len(xs)
            assert a[f"{m}_mean"] == pytest.approx(mean)
            assert a[f"{m}_std"] == pytest.approx(math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs)))
        assert "seed_mean" not in a and "replicate_mean" not in a
    sample = aggregate(rows, by=("taker_bps", "fixed_cost"), metrics=["volume"], ddof=1)[0]
    xs = [r["volume"] for r in rows if r["cell"] == 0]
    assert sample["volume_std"] == pytest.approx(float(np.std(xs, ddof=1)))


def test_friction_stack_and_decision_cost_params():
    names = [m.name for m in friction_stack({"taker_bps": 5, "fixed_cost": 1, "latency": 0.25}).models]
    assert names == ["maker_taker", "fixed", "latency"]
    assert friction_stack({}).models == []
    out = run_sweep(SweepSpec(zi_market, market_metrics, grid={"decision_cost": [0, 3]},
                              seeds=[5], run_kwargs={"max_wakeups": 200}))
    assert out[0]["decision_costs"] == 0 < out[1]["decision_costs"]


def test_csv_writer_round_trips(tmp_path, rows):
    text = write_csv(rows, tmp_path / "s.csv")
    assert (tmp_path / "s.csv").read_text() == text
    lines = text.splitlines()
    assert lines[0].split(",")[:4] == ["cell", "replicate", "seed", "taker_bps"]
    assert len(lines) == len(rows) + 1
    buf = io.StringIO()
    write_csv(rows[:1], buf)
    assert buf.getvalue() == write_csv(rows[:1])


def test_spec_validation():
    with pytest.raises(ValueError):
        SweepSpec(zi_market, market_metrics)
    with pytest.raises(ValueError):
        SweepSpec(zi_market, market_metrics, grid={"a": [1]}, cells=[{"a": 1}])
    with pytest.raises(ValueError):
        SweepSpec(zi_market, market_metrics, grid={"a": [1]}, replicates=0)
    with pytest.raises(ValueError):
        run_sweep(make_spec(), n_workers=0)


def test_zi_example_on_sweep_engine_keeps_regression():
    sys.path.insert(0, str(EXAMPLES))  # lets worker processes import the example module
    spec = importlib.util.spec_from_file_location("phase2_zi_efficiency", EXAMPLES / "phase2_zi_efficiency.py")
    ex = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ex
    spec.loader.exec_module(ex)
    rows = ex.sweep()
    r0, top = rows[0], rows[-1]
    assert (r0["q_ce"], r0["surplus_ce"], r0["p_ce_lo"], r0["p_ce_hi"]) == (18, 7872, 1040, 1045)
    assert (r0["volume"], r0["gains"], r0["fees"]) == (19, 7792, 0)
    assert (top["bps"], top["volume"], top["gains"], top["fees"]) == (400, 16, 7721, 689)
    assert ex.sweep(n_workers=2) == rows
    assert ex.run(ex.Config(), 400) == top
