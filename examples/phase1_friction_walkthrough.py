"""Phase 1 walkthrough: one seeded market replayed under several friction regimes.

Wires every Phase 1 component end to end:

    SimClock (seeded events) -> OrderBook (price-time matching)
        -> FrictionStack (per-fill costs) -> Ledger (settlement + invariants)

The order flow is a fixed seeded process that does not react to costs
(agents that react arrive in Phase 2). That makes each run a controlled
experiment: one seed yields the same tape under every friction regime, so the
gap in realized PnL between two regimes is exactly the friction each agent
paid. Latency only shifts timestamps here, because every order is delayed by
the same `stack.delay` and the relative event order is preserved.

    python examples/phase1_friction_walkthrough.py [seed]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from frictionlab.core import IdGen, Ledger, Order, OrderBook, OrderType, Side, Trade
from frictionlab.core.clock import SimClock
from frictionlab.costs import (
    FixedCost, FrictionStack, HalfSpreadCost, LatencyCost, MakerTakerFee, zero_friction,
)

MAKER = 1
TAKERS = (2, 3, 4, 5)
CASH0, POS0 = 10_000_000, 1_000  # per-agent opening cash (tick-lots) and inventory (lots)


@dataclass(frozen=True)
class Config:
    seed: int = 7
    n_flow: int = 400        # taker decisions before the flow stops
    fair0: int = 10_000      # initial fair value (ticks)
    flow_rate: float = 5.0   # taker decisions / sec
    quote_rate: float = 2.0  # maker re-quotes / sec
    fair_rate: float = 1.0   # fair-value jumps / sec
    fair_step: int = 3       # +- ticks per jump
    half_spread: int = 2     # maker quotes fair +- half_spread
    quote_size: int = 5
    p_limit: float = 0.3     # share of taker decisions resting a passive limit
    max_qty: int = 4


@dataclass
class RunResult:
    trades: List[Trade]
    ledger: Ledger
    book: OrderBook
    mark: int    # final fair value used to mark inventory
    events: int

    def fingerprint(self) -> List[tuple]:
        """Friction-invariant tape: who traded how much at which price."""
        return [(t.price, t.qty, t.buyer_id, t.seller_id, t.aggressor) for t in self.trades]

    def pnl(self) -> Dict[int, Dict[str, int]]:
        """Per-agent PnL marked at final fair value; gross = net + fees paid."""
        start = CASH0 + POS0 * self.mark
        out = {}
        for a in self.ledger:
            net = a.equity(self.mark) - start
            out[a.agent_id] = {"trades": a.n_trades, "volume": a.volume,
                               "fees": a.fees_paid, "gross": net + a.fees_paid, "net": net}
        return out


def run(cfg: Config = Config(), frictions: Optional[FrictionStack] = None) -> RunResult:
    stack = zero_friction() if frictions is None else frictions
    clock = SimClock(cfg.seed)
    rng, delay, oid = clock.rng, stack.delay, IdGen()
    book = OrderBook(trade_ids=IdGen())
    ledger = Ledger(stack, venue_id=0, check_every=True)
    for aid in (MAKER, *TAKERS):
        ledger.open_account(aid, cash=CASH0, position=POS0)
    st = {"fair": cfg.fair0, "flow": 0, "quotes": []}

    def live() -> bool:
        return st["flow"] < cfg.n_flow

    def on_fair(c: SimClock, ev) -> None:
        step = cfg.fair_step if rng.random() < 0.5 else -cfg.fair_step
        st["fair"] = max(cfg.half_spread + 1, st["fair"] + step)
        if live():
            c.schedule_poisson(cfg.fair_rate, "fair")

    def on_quote(c: SimClock, ev) -> None:
        f, t = st["fair"], c.now + delay
        quotes = [Order(oid(), MAKER, Side.BUY, cfg.quote_size, f - cfg.half_spread, t),
                  Order(oid(), MAKER, Side.SELL, cfg.quote_size, f + cfg.half_spread, t)]
        c.schedule(delay, "arrive", ("quote", quotes))
        if live():
            c.schedule_poisson(cfg.quote_rate, "quote")

    def on_flow(c: SimClock, ev) -> None:
        st["flow"] += 1
        aid = int(TAKERS[int(rng.integers(len(TAKERS)))])
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        qty, t = int(rng.integers(1, cfg.max_qty + 1)), c.now + delay
        if rng.random() < cfg.p_limit:  # passive: buy below fair, sell above
            px = st["fair"] - side.value * int(rng.integers(0, cfg.half_spread + 3))
            order = Order(oid(), aid, side, qty, px, t)
        else:
            order = Order(oid(), aid, side, qty, None, t, OrderType.MARKET)
        c.schedule(delay, "arrive", ("order", [order]))
        if live():
            c.schedule_poisson(cfg.flow_rate, "flow")

    def on_arrive(c: SimClock, ev) -> None:
        kind, orders = ev.payload
        if kind == "quote":  # maker replaces its two-sided quote atomically
            for old in st["quotes"]:
                book.cancel(old)
            st["quotes"] = [o.order_id for o in orders]
        for o in orders:
            ledger.settle_all(book.submit(o))

    for kind, h in (("fair", on_fair), ("quote", on_quote),
                    ("flow", on_flow), ("arrive", on_arrive)):
        clock.on(kind, h)
    clock.schedule_at(0.0, "quote")
    clock.schedule_poisson(cfg.fair_rate, "fair")
    clock.schedule_poisson(cfg.flow_rate, "flow")
    events = clock.run()  # flow stops after n_flow decisions; queue then drains
    ledger.check()
    return RunResult(list(book.trades), ledger, book, st["fair"], events)


def regimes() -> Dict[str, FrictionStack]:
    """Frictionless baseline plus three increasingly costly venues."""
    retail = MakerTakerFee(maker_bps=0, taker_bps=10) + HalfSpreadCost(2) + FixedCost(5)
    return {
        "frictionless": zero_friction(),
        "rebate_venue": FrictionStack([MakerTakerFee(maker_bps=-1, taker_bps=5)]),
        "retail": retail,
        "slow_retail": retail + LatencyCost(0.05, drift_ticks_per_sec=20),
    }


def compare(cfg: Config = Config(),
            stacks: Optional[Mapping[str, FrictionStack]] = None) -> Dict[str, RunResult]:
    stacks = regimes() if stacks is None else stacks
    return {name: run(cfg, s) for name, s in stacks.items()}


def report(results: Mapping[str, RunResult]) -> str:
    lines = [f"{'regime':<14}{'agent':>6}{'trades':>8}{'volume':>8}"
             f"{'fees':>9}{'gross':>10}{'net':>10}"]
    for name, res in results.items():
        for aid, r in sorted(res.pnl().items()):
            role = "maker" if aid == MAKER else f"t{aid}"
            lines.append(f"{name:<14}{role:>6}{r['trades']:>8}{r['volume']:>8}"
                         f"{r['fees']:>9}{r['gross']:>10}{r['net']:>10}")
        lines.append(f"{name:<14}{'venue':>6}{len(res.trades):>8}{'':>8}"
                     f"{res.ledger.fees_collected:>9}  mark={res.mark}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cfg = Config(seed=int(argv[0])) if argv else Config()
    results = compare(cfg)
    base = next(iter(results.values())).fingerprint()
    same = all(r.fingerprint() == base for r in results.values())
    print(f"seed={cfg.seed} trades={len(base)} identical_tape_across_regimes={same}")
    print(report(results))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
