"""fl-041: the sorted-sweep batch clear matches the frozen pre-change oracle exactly.

Every batch is replayed through `BatchAuction` and `ReferenceBatchAuction` with
twin `Order` objects. After each clear the clearing price, volume, n_orders,
the full trade list, every order's fill / cancel state, the live set and the
depth of both sides must agree. Stream shapes cover empty, one-sided,
all-at-one-price and multi-way-tie batches, market orders, partial cancels,
self-crossing agents and both carry modes.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from frictionlab.bench import scenarios as sc
from frictionlab.core import IdGen, Order, OrderType, Side
from frictionlab.markets import BatchAuction, clearing_price
from tests import reference_batch_auction as ref
from tests.test_batch_auction import brute

BASELINE = Path(__file__).resolve().parents[1] / "benchmarks" / "baseline_small.json"
SHAPES = ("mixed", "empty", "bids_only", "asks_only", "one_price", "ties", "wide")


def quote(rng, shape):
    side = Side.BUY if rng.random() < 0.5 else Side.SELL
    if shape == "bids_only":
        side = Side.BUY
    elif shape == "asks_only":
        side = Side.SELL
    if shape == "one_price":
        price = 100
    elif shape == "ties":                       # symmetric two-level book: wide V plateaus
        price = int(rng.choice([97, 103]))
    elif shape == "wide":
        price = int(rng.integers(1, 400))
    else:
        price = int(rng.integers(92, 109))
    market = shape not in ("one_price", "ties") and rng.random() < 0.1
    return side, (2 if shape == "ties" else int(rng.integers(1, 7))), None if market else price


def twins(oid, aid, side, qty, price, t):
    ot = OrderType.MARKET if price is None else OrderType.LIMIT
    return [Order(oid, aid, side, qty, price, t, ot) for _ in range(2)]


def snapshot(auc, orders):
    return (sorted(auc._live), auc.depth(Side.BUY), auc.depth(Side.SELL), auc.last_price,
            [(o.filled, o.cancelled, o.qty) for o in orders])


def clearing_tuple(c):
    return (c.time, c.price, c.volume, c.n_orders,
            [(t.trade_id, t.time, t.price, t.qty, t.buy_order_id, t.sell_order_id,
              t.buyer_id, t.seller_id, t.aggressor) for t in c.trades])


def replay(seed, shape, carry=True, n_batches=12, n_agents=4, last_price=None):
    rng, ids = np.random.default_rng(seed), IdGen()
    new = BatchAuction(IdGen(), carry=carry, last_price=last_price)
    old = ref.ReferenceBatchAuction(IdGen(), carry=carry, last_price=last_price)
    mine, theirs = [], []
    for b in range(n_batches):
        for k in range(0 if shape == "empty" else int(rng.integers(0, 12))):
            side, qty, price = quote(rng, shape)
            a, o = twins(ids(), int(rng.integers(1, n_agents + 1)), side, qty, price, b + k / 20)
            new.submit(a), old.submit(o)
            mine.append(a), theirs.append(o)
        if mine and rng.random() < 0.3:          # full / partial cancel, maybe of a dead id
            oid = mine[int(rng.integers(len(mine)))].order_id + (10**6 if rng.random() < 0.1 else 0)
            q = None if rng.random() < 0.5 else int(rng.integers(1, 4))
            assert new.cancel(oid, q) == old.cancel(oid, q)
        cn, co = new.clear(b + 0.5), old.clear(b + 0.5)
        assert clearing_tuple(cn) == clearing_tuple(co), (seed, shape, b)
        assert snapshot(new, mine) == snapshot(old, theirs), (seed, shape, b)
        bought = sum(t.qty for t in cn.trades)
        assert bought == cn.volume                                 # nothing lost or created
        assert sum(o.filled for o in mine) == 2 * sum(t.qty for t in new.trades)
    return new


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("carry", [True, False])
def test_seeded_batches_match_oracle_after_every_clear(shape, carry):
    for seed in range(30):
        replay(seed, shape, carry, last_price=None if seed % 3 else 100)


def test_clearing_price_matches_oracle_and_brute_force():
    rng = np.random.default_rng(41)
    for i in range(3000):
        n_b, n_a = int(rng.integers(0, 8)), int(rng.integers(0, 8))
        w = int(rng.choice([1, 3, 12, 300]))
        mk = lambda: (None if rng.random() < 0.1 else 100 + int(rng.integers(-w, w + 1)),
                      int(rng.integers(1, 6)))
        bids, asks = [mk() for _ in range(n_b)], [mk() for _ in range(n_a)]
        r = None if i % 4 == 0 else 100 + int(rng.integers(-2 * w, 2 * w + 1))
        got = clearing_price(bids, asks, r)
        assert got == ref.clearing_price(bids, asks, r), (bids, asks, r)
        if any(p is not None for p, _ in bids + asks):
            assert got[1] == brute(bids, asks)


def test_edge_batches_match_oracle():
    cases = [([], [], None), ([], [], 100), ([(None, 3)], [(None, 3)], None),
             ([(None, 3)], [(None, 2)], 7), ([(100, 4)], [], 100), ([], [(1, 4)], None),
             ([(100, 2), (100, 3)], [(100, 5)], None),       # all at one price
             ([(103, 2), (103, 2)], [(97, 2), (97, 2)], 50),  # flat plateau, far ref
             ([(105, 1), (101, 1)], [(99, 1), (95, 1)], None),  # multi-segment tie
             ([(99, 2)], [(101, 2)], 100)]                    # uncrossed
    for bids, asks, r in cases:
        assert clearing_price(bids, asks, r) == ref.clearing_price(bids, asks, r)


def test_marginal_pro_rata_and_conservation_on_wide_batches():
    for seed in range(10):
        auc = replay(100 + seed, "wide", n_batches=6, n_agents=8)
        for c in auc.history:
            buys = sum(t.qty for t in c.trades)
            assert buys == c.volume and all(t.price == c.price for t in c.trades)


@pytest.mark.parametrize("name", sc.scenario_names())
def test_fl038_scenario_checksums_equal_baseline(name):
    base = json.loads(BASELINE.read_text())["scenarios"][name]
    w = sc.get_scenario(name).run("small")
    assert (w.checksum, w.ops, w.trades, w.volume) == (
        base["checksum"], base["ops"], base["trades"], base["volume"])


def test_deep_batch_workload_matches_oracle_checksum(monkeypatch):
    p = dict(sc.DEEP_BATCH, n_batches=30)
    fast = sc.batch_auction_clear(p, 41)
    monkeypatch.setattr(sc, "BatchAuction", ref.ReferenceBatchAuction)
    slow = sc.batch_auction_clear(p, 41)
    assert fast.trades > 0 and fast == slow and fast.checksum == slow.checksum
