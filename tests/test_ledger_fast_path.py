"""fl-040: the ledger hot path is a pure speedup.

The cheap `type(x) is int` guard must keep every rejection message, the
per-trade conservation check must catch what the per-leg checks catch, and
strict mode (FRICTIONLAB_STRICT_LEDGER) must give identical balances and
identical fl-038 checksums.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from frictionlab.bench import scenarios as sc
from frictionlab.core import (IdGen, InvariantViolation, Ledger, LedgerError, Order, OrderBook,
                              OrderType, Side, Trade)
from frictionlab.core.ledger import STRICT_ENV, _int, strict_from_env
from frictionlab.core.types import _check_int
from frictionlab.costs import FixedCost, HalfSpreadCost, MakerTakerFee

BASELINE = Path(__file__).resolve().parents[1] / "benchmarks" / "baseline_small.json"
BAD = [True, False, 1.0, 2.5, np.float64(3.0), np.float32(1.0), "1", None, 1 + 0j]


def message(fn, *args, **kw):
    with pytest.raises((TypeError, ValueError)) as ei:
        fn(*args, **kw)
    return type(ei.value), str(ei.value)


@pytest.mark.parametrize("v", BAD)
def test_fast_int_guard_rejects_with_identical_messages(v):
    assert message(_int, "cash", v) == message(_check_int, "cash", v)
    assert message(_int, "amount", v, nonneg=True) == message(_check_int, "amount", v, nonneg=True)
    kind, text = message(_int, "cash", v)
    assert kind is TypeError and text == f"cash must be an integer, got {type(v).__name__}"


def test_fast_int_guard_accepts_ints_and_numpy_ints():
    assert _int("x", 5) == 5 and type(_int("x", np.int64(-7))) is int
    assert message(_int, "amount", -1, nonneg=True) == (ValueError, "amount must be >= 0, got -1")


@pytest.mark.parametrize("v", [True, 1.0, np.float64(2.0)])
def test_ledger_entry_points_reject_bool_and_floats(v):
    lg = Ledger(MakerTakerFee(0, 10))
    lg.open_account(1, cash=1000)
    lg.open_account(2, position=10)
    t = Trade(1, 0.0, 10, 1, 5, 6, 1, 2, Side.BUY)
    cases = [(lg.open_account, (v,), "agent_id"), (lg.deposit, (1, v), "cash"),
             (lg.deposit, (1, 0, v), "position"), (lg.transfer, (1, 2, v), "amount"),
             (lg.settle, (t, (v, 0)), "buyer_fee"), (lg.settle, (t, (0, v)), "seller_fee")]
    for fn, args, name in cases:
        assert message(fn, *args) == (TypeError,
                                      f"{name} must be an integer, got {type(v).__name__}")
    assert not lg.journal and lg[1].cash == 1000 and lg.venue.cash == 0


def flow(strict, seed=7, n=600):
    stack = MakerTakerFee(maker_bps=-1, taker_bps=7) + HalfSpreadCost(1) + FixedCost(2)
    lg = Ledger(stack, strict=strict)
    rng = np.random.default_rng(seed)
    for aid in range(1, 9):
        lg.open_account(aid, cash=int(rng.integers(0, 10**6)), position=int(rng.integers(0, 500)))
    book, oid = OrderBook(), IdGen()
    for t in range(n):
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        aid, qty = int(rng.integers(1, 9)), int(rng.integers(1, 20))
        px = None if rng.random() < 0.2 else int(rng.integers(95, 106))
        kind = OrderType.MARKET if px is None else OrderType.LIMIT
        lg.settle_all(book.submit(Order(oid(), aid, side, qty, px, float(t), kind)))
    return lg


def state(lg):
    return (sc.ledger_balances(lg), [(a.fees_paid, a.volume, a.n_trades) for a in lg],
            lg.journal)


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_strict_and_default_modes_book_identically(seed):
    fast, strict = flow(False, seed), flow(True, seed)
    assert len(fast.journal) > 50 and state(fast) == state(strict)
    fast.check()
    strict.check()


def test_strict_mode_from_env(monkeypatch):
    monkeypatch.delenv(STRICT_ENV, raising=False)
    assert not strict_from_env() and not Ledger().strict
    for on in ("1", "true", "YES"):
        monkeypatch.setenv(STRICT_ENV, on)
        assert strict_from_env() and Ledger().strict
    monkeypatch.setenv(STRICT_ENV, "0")
    assert not Ledger().strict and Ledger(strict=True).strict


def test_seeded_runner_scenario_identical_in_strict_mode(monkeypatch):
    p = sc.get_scenario("runner_zi_market").sizes["small"]
    monkeypatch.delenv(STRICT_ENV, raising=False)
    fast = sc.runner_zi_market(dict(p), 4)
    monkeypatch.setenv(STRICT_ENV, "1")
    strict = sc.runner_zi_market(dict(p), 4)
    assert fast.trades > 0 and fast == strict and fast.checksum == strict.checksum


@pytest.mark.parametrize("strict", ["0", "1"])
def test_every_fl038_checksum_matches_baseline(monkeypatch, strict):
    monkeypatch.setenv(STRICT_ENV, strict)
    base = json.loads(BASELINE.read_text())["scenarios"]
    for name, scen in sc.SCENARIOS.items():
        assert scen.run("small").checksum == base[name]["checksum"], (name, strict)


def test_per_trade_conservation_check_catches_a_broken_account(monkeypatch):
    class Leaky(type(Ledger().venue)):
        """An account whose cash setter drops one tick on every write."""
        def __setattr__(self, k, v):
            object.__setattr__(self, k, v - 1 if k == "cash" and hasattr(self, "cash") else v)

    for strict in (False, True):
        lg = Ledger(FixedCost(1), strict=strict)
        lg.open_account(1, cash=10_000)
        lg.open_account(2, position=5)
        lg.accounts[2] = Leaky(2, lg[2].cash, lg[2].position)
        with pytest.raises(InvariantViolation):
            lg.settle(Trade(1, 0.0, 10, 1, 5, 6, 1, 2, Side.BUY))


def test_rejections_still_leave_ledger_untouched():
    for strict in (False, True):
        lg = Ledger(FixedCost(2_000), allow_overdraft=False, strict=strict)
        lg.open_account(1, cash=100_000)
        lg.open_account(2, position=10)
        with pytest.raises(LedgerError, match="cannot pay fee"):
            lg.settle(Trade(1, 0.0, 10, 10, 5, 6, 1, 2, Side.BUY))
        assert not lg.journal and lg[1].cash == 100_000 and lg.venue.cash == 0
