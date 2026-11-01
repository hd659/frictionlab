"""fl-038 benchmark harness: deterministic checksums, timing sanity, compare, CLI.

No test asserts an absolute timing threshold; timings only need to be
positive and ordered. Correctness is pinned by the committed checksums.
"""
import json
import time
from pathlib import Path

import pytest

from frictionlab import bench
from frictionlab.cli import main

BASELINE = Path(__file__).resolve().parents[1] / "benchmarks" / "baseline_small.json"
NAMES = list(bench.SCENARIOS)


@pytest.fixture(scope="module")
def small():
    return bench.run_bench(size="small", warmup=0, repeat=2)


def test_registry_names_and_sizes():
    assert NAMES == ["book_random_flow", "runner_zi_market", "batch_auction_clear",
                     "amm_swap_stream", "sweep_coase_small"]
    for sc in bench.SCENARIOS.values():
        assert set(sc.sizes) == set(bench.SIZES) and sc.doc
    with pytest.raises(KeyError, match="known"):
        bench.get_scenario("nope")
    with pytest.raises(KeyError):
        bench.SCENARIOS["amm_swap_stream"].run("huge")


@pytest.mark.parametrize("name", NAMES)
def test_checksum_is_identical_across_runs_and_matches_baseline(name):
    sc = bench.get_scenario(name)
    a, b = sc.run("small"), sc.run("small")
    assert a == b and a.checksum == b.checksum
    assert len(a.checksum) == 16 and int(a.checksum, 16) >= 0
    ref = bench.load_json(BASELINE)["scenarios"][name]
    assert a.checksum == ref["checksum"]
    assert (a.ops, a.trades, a.volume) == (ref["ops"], ref["trades"], ref["volume"])
    assert a.trades > 0 and a.volume > 0


def test_checksum_sensitive_to_results():
    w = bench.get_scenario("book_random_flow").run("small")
    assert bench.Work(w.ops, w.trades + 1, w.volume, w.balances).checksum != w.checksum
    assert bench.Work(w.ops + 1, w.trades, w.volume, w.balances).checksum == w.checksum
    bal = list(w.balances)
    bal[0] = (bal[0][0], bal[0][1] + 1, bal[0][2])
    assert bench.Work(w.ops, w.trades, w.volume, bal).checksum != w.checksum


def test_baseline_has_checksums_only():
    data = bench.load_json(BASELINE)
    assert data["size"] == "small" and set(data["scenarios"]) == set(NAMES)
    for r in data["scenarios"].values():
        assert set(r) == {"checksum", "ops", "trades", "volume"}


def test_timing_fields_positive_and_ordered(small):
    assert set(small["scenarios"]) == set(NAMES)
    for r in small["scenarios"].values():
        assert r["repeat"] == 2 and r["warmup"] == 0
        assert 0 < r["min_s"] <= r["median_s"] <= r["max_s"]
        assert r["ops_per_s"] > 0
        assert r["ops_per_s"] == pytest.approx(r["ops"] / r["median_s"])


def test_harness_rejects_bad_counts_and_detects_drift(monkeypatch):
    with pytest.raises(ValueError):
        bench.run_scenario("amm_swap_stream", repeat=0)
    sc = bench.get_scenario("amm_swap_stream")
    calls = iter(range(100))
    fake = bench.Scenario(sc.name, sc.doc, lambda p, s: bench.Work(1, next(calls), 1),
                          sc.sizes)
    monkeypatch.setitem(bench.SCENARIOS, sc.name, fake)
    with pytest.raises(bench.ChecksumDrift):
        bench.run_scenario(sc.name, warmup=0, repeat=2)


def test_compare_flags_tampered_checksum(small, tmp_path):
    tampered = json.loads(bench.to_json(bench.checksums_only(small)))
    tampered["scenarios"]["amm_swap_stream"]["checksum"] = "0" * 16
    p = bench.write_json(tampered, tmp_path / "tampered.json")
    cmp = bench.compare(small, bench.load_json(p))
    assert not cmp.ok and len(cmp.errors) == 1
    assert "amm_swap_stream" in cmp.errors[0] and "checksum mismatch" in cmp.errors[0]
    assert "ERROR" in cmp.format()
    assert bench.compare(small, bench.load_json(BASELINE)).ok


def test_compare_speedup_ratio_and_structural_errors():
    def res(size, **meds):
        return {"size": size, "scenarios": {n: {"checksum": "ab", "median_s": m}
                                            for n, m in meds.items()}}
    cmp = bench.compare(res("small", a=0.5, b=2.0), res("small", a=1.0, b=1.0))
    assert cmp.ok and cmp.speedups == {"a": pytest.approx(2.0), "b": pytest.approx(0.5)}
    assert "2.00x" in cmp.format() and "0.50x" in cmp.format()
    assert bench.compare(res("small", a=1.0), res("small", a=1.0, b=1.0)).ok  # subset ok
    assert not bench.compare(res("small", c=1.0), res("small", a=1.0)).ok
    assert not bench.compare(res("medium", a=1.0), res("small", a=1.0)).ok
    nt = bench.compare(res("small", a=1.0), {"size": "small", "scenarios": {"a": {"checksum": "ab"}}})
    assert nt.ok and nt.speedups == {"a": None}


def test_json_output_has_stable_sorted_keys(small, tmp_path):
    text = bench.to_json(small)
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"
    assert list(json.loads(text)) == sorted(json.loads(text))
    assert list(json.loads(text)["scenarios"]) == sorted(NAMES)
    p = bench.write_json(small, tmp_path / "x" / "r.json")
    assert bench.load_json(p) == json.loads(text)
    with pytest.raises(ValueError):
        (tmp_path / "bad.json").write_text("{}")
        bench.load_json(tmp_path / "bad.json")


def test_format_table_lists_every_scenario(small):
    table = bench.format_table(small)
    assert table.startswith("frictionlab bench (size=small)")
    for n, r in small["scenarios"].items():
        assert n in table and r["checksum"] in table


def test_cli_bench_small_within_budget(tmp_path, capsys):
    out = tmp_path / "bench.json"
    t0 = time.perf_counter()
    code = main(["bench", "--size", "small", "--repeat", "1", "--warmup", "0",
                 "--json", str(out), "--compare", str(BASELINE)])
    assert code == 0 and time.perf_counter() - t0 < 60     # generous test budget
    text = capsys.readouterr().out
    assert "compare: ok" in text and all(n in text for n in NAMES)
    data = bench.load_json(out)
    assert data == json.loads(out.read_text()) and set(data["scenarios"]) == set(NAMES)


def test_cli_checksums_only_reproduces_baseline(tmp_path, capsys):
    out = tmp_path / "base.json"
    assert main(["bench", "--repeat", "1", "--warmup", "0", "--json", str(out),
                 "--checksums-only"]) == 0
    assert out.read_text() == BASELINE.read_text()


def test_cli_compare_mismatch_and_usage_errors(tmp_path, capsys):
    bad = json.loads(BASELINE.read_text())
    bad["scenarios"]["runner_zi_market"]["checksum"] = "f" * 16
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    assert main(["bench", "--scenario", "runner_zi_market", "--repeat", "1",
                 "--compare", str(p)]) == 1
    assert "checksum mismatch" in capsys.readouterr().out
    assert main(["bench", "--scenario", "nope"]) == 2
    assert "--scenario" in capsys.readouterr().err
    assert main(["bench", "--compare", str(tmp_path / "missing.json")]) == 2
    assert main(["bench", "--size", "huge"]) == 2
