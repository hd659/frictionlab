"""Named, seeded benchmark workloads.

Every scenario is a pure function `(params, seed) -> Work`: it builds its own
book / ledger / runner from scratch, so repeated calls are independent and
bit-identical. `Work.checksum` hashes the *results* (trade count, total volume,
final ledger balances), never the timing, so an optimization that changes the
checksum changed behaviour, not just speed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..agents import (AvellanedaStoikovMM, FundamentalProcess, InformedTrader,
                      make_noise_population, make_zi_population)
from ..core import IdGen, Ledger, Order, OrderBook, OrderType, Side
from ..costs import FrictionStack, MakerTakerFee
from ..core.runner import MarketRunner
from ..markets import BatchAuction, ConstantProductAMM, SwapRejected

SIZES = ("small", "medium")
Balances = List[Tuple[int, int, int]]


@dataclass(frozen=True)
class Work:
    """What a scenario did: `ops` drives ops/sec, the rest feeds the checksum."""
    ops: int
    trades: int
    volume: int
    balances: Balances = field(default_factory=list)

    @property
    def checksum(self) -> str:
        blob = json.dumps({"trades": self.trades, "volume": self.volume,
                           "balances": [list(b) for b in self.balances]},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Scenario:
    name: str
    doc: str
    fn: Callable[[Dict[str, Any], int], Work]
    sizes: Dict[str, Dict[str, Any]]
    seed: int = 38

    def run(self, size: str = "small") -> Work:
        if size not in self.sizes:
            raise KeyError(f"scenario {self.name!r} has no size {size!r}; use one of {SIZES}")
        return self.fn(dict(self.sizes[size]), self.seed)


def ledger_balances(lg: Ledger) -> Balances:
    accts = [lg.venue] + list(lg)
    return sorted((a.agent_id, a.cash, a.position) for a in accts)


def _fees() -> FrictionStack:
    return FrictionStack([MakerTakerFee(maker_bps=0, taker_bps=10)])


def _ledger(n_agents: int, cash: int = 10**7, position: int = 10**4) -> Ledger:
    lg = Ledger(_fees())
    for aid in range(1, n_agents + 1):
        lg.open_account(aid, cash, position)
    return lg


def _random_limit(rng, oid: int, n_agents: int, t: float, spread: int = 10,
                  mid: int = 100) -> Order:
    side = Side.BUY if rng.random() < 0.5 else Side.SELL
    off = int(rng.integers(0, spread + 1))
    price = mid - off if side is Side.BUY else mid + off - spread // 2
    return Order(oid, int(rng.integers(1, n_agents + 1)), side, int(rng.integers(1, 6)),
                 max(1, price), t)


def book_random_flow(p: Dict[str, Any], seed: int) -> Work:
    """Random limit / market / cancel stream into the continuous book."""
    rng, n = np.random.default_rng(seed), int(p["n_agents"])
    lg, book, ids, live = _ledger(n), OrderBook(IdGen()), IdGen(), []
    for i in range(int(p["n_ops"])):
        u, t = rng.random(), float(i)
        if u < 0.55:
            o = _random_limit(rng, ids(), n, t)
            live.append(o.order_id)
        elif u < 0.8:
            o = Order(ids(), int(rng.integers(1, n + 1)), Side(1 if rng.random() < 0.5 else -1),
                      int(rng.integers(1, 4)), None, t, OrderType.MARKET)
        else:
            if live:
                book.cancel(live.pop(int(rng.integers(len(live)))))
            continue
        lg.settle_all(book.submit(o))
    lg.check()
    tr = book.trades
    return Work(int(p["n_ops"]), len(tr), sum(x.qty for x in tr), ledger_balances(lg))


def runner_zi_market(p: Dict[str, Any], seed: int) -> Work:
    """ZI-C + Avellaneda-Stoikov maker + informed + noise through MarketRunner."""
    ss = np.random.SeedSequence(seed).spawn(4)
    child = lambda s: int(s.generate_state(1)[0])
    stack = _fees()
    fund = FundamentalProcess.random_walk(100, 0.2, 1, seed=np.random.default_rng(ss[0]))
    agents = (make_zi_population("zi-c", 4, 4, seed=child(ss[1]), n_units=3)
              + [AvellanedaStoikovMM(20, rng=np.random.default_rng(ss[2])),
                 InformedTrader(21, fund, frictions=stack, max_position=10)]
              + make_noise_population(2, seed=child(ss[3]), first_id=30))
    r = MarketRunner(agents, stack, seed=seed, cash=10**6, position=10)
    res = r.run(max_wakeups=int(p["wakeups"]))
    return Work(res.wakeups, len(res.trades), sum(t.qty for t in res.trades),
                ledger_balances(res.ledger))


def batch_auction_clear(p: Dict[str, Any], seed: int) -> Work:
    """Seeded limit-order batches cleared at a uniform price, settled on a ledger.

    Optional `spread` / `mid` widen the price grid (defaults 10 / 100 are the
    fl-038 workload whose checksums are pinned); `DEEP_BATCH` is a many-level
    preset used by docs/phase6-performance.md and the fl-041 oracle test.
    """
    rng, n = np.random.default_rng(seed), int(p["n_agents"])
    spread, mid = int(p.get("spread", 10)), int(p.get("mid", 100))
    lg, auc, ids = _ledger(n), BatchAuction(IdGen(), last_price=mid), IdGen()
    per = int(p["orders_per_batch"])
    for b in range(int(p["n_batches"])):
        for _ in range(per):
            auc.submit(_random_limit(rng, ids(), n, float(b), spread, mid))
        lg.settle_all(auc.clear(float(b) + 0.5).trades)
    lg.check()
    tr = auc.trades
    return Work(int(p["n_batches"]) * per, len(tr), sum(x.qty for x in tr), ledger_balances(lg))


def amm_swap_stream(p: Dict[str, Any], seed: int) -> Work:
    """Random two-sided swap stream against a constant-product pool."""
    rng, n = np.random.default_rng(seed), int(p["n_agents"])
    lg = _ledger(n)
    amm = ConstantProductAMM(lg, 900, 10_000, 1_000_000, fee_bps=30)
    done = vol = 0
    for i in range(int(p["n_swaps"])):
        aid, buy = int(rng.integers(1, n + 1)), rng.random() < 0.5
        amt = int(rng.integers(1_000, 20_000)) if buy else int(rng.integers(10, 200))
        try:
            res = amm.swap_exact_in(aid, Side.BUY if buy else Side.SELL, amt, time=float(i))
        except SwapRejected:
            continue
        done += 1
        vol += res.amount_out if buy else res.amount_in      # base lots moved
    lg.check()
    return Work(int(p["n_swaps"]), done, vol, ledger_balances(lg))


def sweep_coase_small(p: Dict[str, Any], seed: int) -> Work:
    """The fl-028 Coase grid, serial (n_workers=1)."""
    from ..experiments.coase import Config, run_coase   # lazy: experiments are heavy
    res = run_coase(Config(max_wakeups=int(p["wakeups"])), levels=list(p["levels"]),
                    replicates=int(p["replicates"]), master_seed=seed, n_workers=1)
    rows = res.rows
    if res.errors:
        raise RuntimeError(f"sweep_coase_small: {len(res.errors)} error rows")
    bal = [(int(r["cell"]), int(r["replicate"]), int(r["friction_collected"])) for r in rows]
    return Work(sum(int(r["wakeups"]) for r in rows), len(rows),
                sum(int(r["volume"]) for r in rows), bal)


# Many price levels: ~600 distinct ticks, deep carried books (fl-041 docs and oracle test).
DEEP_BATCH: Dict[str, Any] = {"n_batches": 250, "orders_per_batch": 40, "n_agents": 32,
                              "spread": 400, "mid": 1000}


SCENARIOS: Dict[str, Scenario] = {s.name: s for s in (
    Scenario("book_random_flow", "limit/market/cancel stream into the continuous book",
             book_random_flow, {"small": {"n_ops": 3_000, "n_agents": 8},
                                "medium": {"n_ops": 30_000, "n_agents": 32}}),
    Scenario("runner_zi_market", "ZI-C + maker + informed + noise through the runner",
             runner_zi_market, {"small": {"wakeups": 1_500}, "medium": {"wakeups": 15_000}}),
    Scenario("batch_auction_clear", "seeded batches cleared at a uniform price",
             batch_auction_clear, {"small": {"n_batches": 100, "orders_per_batch": 20, "n_agents": 8},
                                   "medium": {"n_batches": 250, "orders_per_batch": 40,
                                              "n_agents": 32}}),
    Scenario("amm_swap_stream", "two-sided swap stream on a constant-product pool",
             amm_swap_stream, {"small": {"n_swaps": 2_000, "n_agents": 8},
                               "medium": {"n_swaps": 20_000, "n_agents": 32}}),
    Scenario("sweep_coase_small", "fl-028 Coase grid with n_workers=1",
             sweep_coase_small, {"small": {"levels": [0, 40, 80], "replicates": 2, "wakeups": 600},
                                 "medium": {"levels": [0, 20, 40, 60, 80, 100], "replicates": 4,
                                            "wakeups": 2_000}}),
)}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {', '.join(sorted(SCENARIOS))}")
    return SCENARIOS[name]


def scenario_names(names: Sequence[str] = ()) -> List[str]:
    return [get_scenario(n).name for n in names] if names else list(SCENARIOS)
