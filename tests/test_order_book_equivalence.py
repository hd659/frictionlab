"""fl-039: the indexed-level OrderBook is observationally identical to the frozen oracle.

Every seeded operation stream is replayed through both books with twin
`Order` objects; after *every* operation the returned fills, the order
states, best bid/ask, full depth and the live-id index must match exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from frictionlab.bench import scenarios as sc
from frictionlab.core import IdGen, Order, OrderBook, OrderType, Side
from frictionlab.core import book_levels
from frictionlab.core.book_levels import BookSide

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_order_book import ReferenceOrderBook  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / "benchmarks" / "baseline_small.json"


def trade_key(t):
    return (t.trade_id, t.time, t.price, t.qty, t.buy_order_id, t.sell_order_id,
            t.buyer_id, t.seller_id, t.aggressor)


def snapshot(b):
    return (b.best_bid, b.best_ask, b.depth(Side.BUY), b.depth(Side.SELL), len(b))


def random_ops(seed, n=600, n_agents=6, width=12):
    """Limits (some marketable), markets, full/partial cancels, dead/unknown ids."""
    rng, ops, oid, seen = np.random.default_rng(seed), [], 0, []
    for i in range(n):
        u = rng.random()
        if u < 0.55 or not seen:
            oid += 1
            side = 1 if rng.random() < 0.5 else -1
            off = int(rng.integers(-3, width))  # negative offset -> crosses
            price = 100 - off if side == 1 else 101 + off
            ops.append(("L", oid, int(rng.integers(1, n_agents + 1)), side,
                        int(rng.integers(1, 8)), price, float(i)))
            seen.append(oid)
        elif u < 0.72:
            oid += 1
            ops.append(("M", oid, int(rng.integers(1, n_agents + 1)),
                        1 if rng.random() < 0.5 else -1, int(rng.integers(1, 12)), None, float(i)))
        elif u < 0.9:  # may target filled, cancelled or never-seen ids
            target = seen[int(rng.integers(len(seen)))] if rng.random() < 0.9 else oid + 1000
            qty = None if rng.random() < 0.6 else int(rng.integers(1, 5))
            ops.append(("C", target, qty))
        else:
            ops.append(("C", -int(rng.integers(1, 50)), None))  # unknown id
    return ops


def replay(ops, stp="cancel_resting", check_every=True):
    ref, new = ReferenceOrderBook(IdGen(), stp=stp), OrderBook(IdGen(), stp=stp)
    twins = {}
    for op in ops:
        if op[0] == "C":
            assert new.cancel(op[1], op[2]) == ref.cancel(op[1], op[2]), op
        else:
            kind, oid, aid, side, qty, price, t = op
            otype = OrderType.LIMIT if kind == "L" else OrderType.MARKET
            a, b = (Order(oid, aid, Side(side), qty, price, t, otype) for _ in range(2))
            twins[oid] = (a, b)
            ra, rb = ref.submit(a), new.submit(b)
            assert [trade_key(x) for x in rb] == [trade_key(x) for x in ra], op
        if check_every:
            assert snapshot(new) == snapshot(ref), op
            assert set(new._live) == set(ref._live)
    for a, b in twins.values():
        assert (b.filled, b.cancelled, b.qty) == (a.filled, a.cancelled, a.qty)
    assert [trade_key(t) for t in new.trades] == [trade_key(t) for t in ref.trades]
    return ref, new


@pytest.mark.parametrize("seed", range(40))
def test_random_streams_match_oracle_after_every_op(seed):
    ref, _ = replay(random_ops(seed))
    assert ref.trades, "stream should exercise matching"


@pytest.mark.parametrize("seed", range(10))
def test_random_streams_match_oracle_cancel_incoming(seed):
    replay(random_ops(1000 + seed, n_agents=3), stp="cancel_incoming")


def test_partial_fills_walk_levels_identically():
    ops = [("L", i, i % 4 + 1, -1, 3, 100 + i % 5, float(i)) for i in range(1, 21)]
    ops += [("L", 50, 9, 1, 25, 103, 21.0), ("M", 51, 9, 1, 40, None, 22.0)]
    ref, new = replay(ops)
    assert any(t.qty < 3 for t in new.trades)  # at least one partial fill
    assert snapshot(new) == snapshot(ref)


def test_cancel_filled_or_unknown_id_matches_oracle():
    ops = [("L", 1, 1, -1, 2, 100, 0.0), ("M", 2, 2, 1, 2, None, 1.0),
           ("C", 1, None), ("C", 1, 1), ("C", 999, None), ("C", 2, None)]
    ref, new = replay(ops)
    assert new.cancel(1) == ref.cancel(1) == 0
    assert new.cancel(12345, 3) == ref.cancel(12345, 3) == 0
    assert 1 not in new and len(new) == 0


def test_lazily_deleted_levels_never_surface_as_best():
    half, rng = BookSide(Side.SELL), np.random.default_rng(39)
    live, oid = {}, 0
    for _ in range(3000):
        if live and rng.random() < 0.5:  # kill a whole level in O(1)
            p = list(live)[int(rng.integers(len(live)))]
            half.reduce(p, live.pop(p))
        else:  # (re)birth of a level, possibly over a buried stale heap key
            oid += 1
            p, q = int(rng.integers(90, 110)), int(rng.integers(1, 4))
            half.add(Order(oid, 1, Side.SELL, q, p))
            live[p] = live.get(p, 0) + q
        assert half.best() == (min(live) if live else None)
        assert half.levels() == sorted(live.items())
        assert len(half._heap) == len(half._in_heap) == len(set(half._heap))


def test_bid_side_best_is_highest_and_depth_truncates():
    half = BookSide(Side.BUY)
    for i, p in enumerate([97, 101, 99, 101, 95]):
        half.add(Order(i + 1, 1, Side.BUY, 2, p))
    half.reduce(101, 4)
    assert half.best() == 99
    assert half.levels(2) == [(99, 2), (97, 2)] and half.prices(0) == []


@pytest.mark.parametrize("name", sc.scenario_names())
def test_fl038_scenario_checksums_equal_baseline(name):
    base = json.loads(BASELINE.read_text())["scenarios"][name]
    assert sc.get_scenario(name).run("small").checksum == base["checksum"]


def _book_with_depth(seed=7):
    _, new = replay(random_ops(seed, n=300), check_every=False)
    return new


def test_depth_array_matches_depth():
    book = _book_with_depth()
    for side in (Side.BUY, Side.SELL):
        for n in (None, 1, 3):
            arr, lv = book.depth_array(side, n), book.depth(side, n)
            assert arr.dtype == np.int64 and arr.shape == (len(lv), 3)
            assert arr[:, :2].tolist() == [list(x) for x in lv]
            assert arr[:, 2].tolist() == np.cumsum([q for _, q in lv]).tolist()
    assert OrderBook().depth_array(Side.BUY).shape == (0, 3)


def test_numba_flag_without_numba_falls_back_silently(monkeypatch):
    book = _book_with_depth()
    book_levels.reset_kernel_cache()
    plain = [book.depth_array(s).tolist() for s in (Side.BUY, Side.SELL)]
    monkeypatch.setitem(sys.modules, "numba", None)  # import numba -> ImportError
    monkeypatch.setenv(book_levels.NUMBA_ENV, "1")
    book_levels.reset_kernel_cache()
    assert book_levels.numba_requested()
    assert book_levels.cumulative_kernel() is book_levels._cumsum_numpy
    assert [book.depth_array(s).tolist() for s in (Side.BUY, Side.SELL)] == plain
    ref, new = replay(random_ops(3))
    assert snapshot(new) == snapshot(ref)
    book_levels.reset_kernel_cache()
