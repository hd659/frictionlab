"""Tests for frictionlab.experiments.base (shared Phase 5 experiment shape)."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from frictionlab.agents import ZICTrader
from frictionlab.analysis import Report, friction_stack
from frictionlab.core.runner import MarketRunner
from frictionlab.core.types import Side
from frictionlab.experiments import ExperimentError, ExperimentResult, run_experiment

GRID = {"taker_bps": [0, 200, 800]}


def zi_market(params, seed):
    """Module-level so worker processes can unpickle it."""
    if params.get("boom"):
        raise ValueError("bad cell")
    streams = np.random.SeedSequence(seed).spawn(6)
    agents = [ZICTrader(i + 1, Side.BUY if i < 3 else Side.SELL, np.random.default_rng(ss),
                        n_units=2, value_range=(50, 150), price_range=(1, 200))
              for i, ss in enumerate(streams)]
    endow = {a.agent_id: (0, 2 if a.side is Side.SELL else 0) for a in agents}
    return MarketRunner(agents, friction_stack(params), seed=seed, endowments=endow)


def metrics(res):
    return {"volume": sum(t.qty for t in res.trades), "fees": res.fees,
            "violations": len(res.ledger.violations())}


def headline(rows):
    return {"total_volume": sum(r["volume"] for r in rows), "cells": len(rows)}


def run(**kw):
    kw.setdefault("replicates", 3)
    kw.setdefault("master_seed", 5)
    return run_experiment("toy_fees", zi_market, metrics, kw.pop("grid", GRID),
                          run_kwargs={"max_wakeups": 200}, findings=headline, **kw)


@pytest.fixture(scope="module")
def res():
    return run()


def test_result_shape(res):
    assert isinstance(res, ExperimentResult)
    assert res.name == "toy_fees" and res.master_seed == 5
    assert res.params == {"taker_bps": [0, 200, 800]} and res.param_keys == ["taker_bps"]
    assert len(res.rows) == 3 * 3 and not res.errors and len(res.ok_rows) == 9
    assert [r["taker_bps"] for r in res.rows] == [0, 0, 0, 200, 200, 200, 800, 800, 800]
    assert res.findings == {"total_volume": sum(r["volume"] for r in res.rows), "cells": 9}
    assert {"volume", "fees", "violations"} <= set(res.metric_names())


def test_aggregate_and_column_match_manual_mean(res):
    agg = res.aggregate()
    assert [a["taker_bps"] for a in agg] == GRID["taker_bps"] and all(a["n"] == 3 for a in agg)
    for lv, mean in zip(GRID["taker_bps"], res.column("volume")):
        vals = [r["volume"] for r in res.rows if r["taker_bps"] == lv]
        assert math.isclose(mean, float(np.mean(vals)))
    assert res.column("volume", "std")[0] == pytest.approx(
        float(np.std([r["volume"] for r in res.rows if r["taker_bps"] == 0])))


def test_crn_default_shares_seeds_across_levels(res):
    by_rep = {}
    for r in res.rows:
        by_rep.setdefault(r["replicate"], set()).add(r["seed"])
    assert all(len(s) == 1 for s in by_rep.values()) and len(by_rep) == 3
    indep = run(crn=False)
    assert len({r["seed"] for r in indep.rows}) == 9


def test_reproducible_and_parallel_identical(res):
    assert run().rows == res.rows
    assert run(n_workers=2).rows == res.rows
    assert run(master_seed=6).rows != res.rows


def test_ledger_invariants_in_every_cell(res):
    assert all(r["violations"] == 0 for r in res.rows)


def test_strict_raises_and_lenient_keeps_error_rows():
    grid = {"boom": [False, True]}
    with pytest.raises(ExperimentError, match="1 failed cell"):
        run(grid=grid, replicates=1)
    loose = run(grid=grid, replicates=1, strict=False)
    assert len(loose.errors) == 1 and "ValueError: bad cell" in loose.errors[0]["error"]
    assert loose.findings["cells"] == 1   # findings only see ok rows


def test_rejects_empty_grid():
    with pytest.raises(ValueError):
        run_experiment("x", zi_market, metrics, {})
    with pytest.raises(ValueError):
        run_experiment("x", zi_market, metrics, {"taker_bps": []})


def test_report_has_findings_sweep_and_provenance(res):
    rep = res.report(metric="volume")
    assert isinstance(rep, Report) and len(rep.sections) == 2
    html = rep.to_html()
    ET.fromstring(html.split("\n", 1)[1])          # well-formed after the doctype
    assert "toy_fees" in html and "total_volume" in html and "taker_bps" in html
    assert "http://" not in html.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert res.report(metric="volume").to_html() == html
