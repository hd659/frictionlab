"""Phase 3 example: one agent mix, three market structures, every friction knob.

Part 1  continuous book vs frequent batch auction across a latency sweep.
        Value-anchored ZI-C traders quote in [V - w, V] (buyers) / [V, V + w]
        (sellers) using the V(t) seen at wake-up; latency delays their orders,
        so quotes go stale after every jump. An informed trader and noise flow
        take liquidity. Reports volume, effective spread, snipe share
        (`analysis.structure_compare`) and informed net PnL at V(T).
Part 2  CrossVenueArbitrageur between a deep static book and a constant-product
        or concentrated-liquidity pool across a gas sweep. Band = buy-side +
        sell-side venue gap (ticks) needed to trigger a trade; residual gap =
        pool spot - book ask once the arbitrageur stops.
Part 3  perp with buy-skewed noise flow. A carry trader takes the side that
        receives funding; with k = 0 (funding off) nobody leans against the
        flow and mark drifts away from index.

    python examples/phase3_market_structures.py [seed]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from frictionlab.agents import (
    Agent, Cancel, CrossVenueArbitrageur, FundamentalProcess, InformedTrader, MarketView,
    Submit, ZICTrader, make_noise_population, make_zi_population,
)
from frictionlab.analysis import pnl_breakdown
from frictionlab.analysis.structure_compare import compare, jump_times_of, summary
from frictionlab.core import Ledger, Side
from frictionlab.core.runner import MarketRunner
from frictionlab.costs import FixedCost, FrictionStack, LatencyCost, MakerTakerFee
from frictionlab.markets import ConcentratedLiquidityAMM, ConstantProductAMM, PerpMarket, batch_runner

LATENCIES, GAS, POOLS = (0.0, 0.5, 1.0, 2.0), (0, 50, 200, 800), ("cp", "cl")
ARB_ID, MAKER_ID, CARRY_ID, INFORMED_ID, POOL_ID = 1, 2, 5, 50, 900


@dataclass(frozen=True)
class Config:
    seed: int = 7
    horizon: float = 200.0
    batch_interval: float = 1.0
    width: int = 3               # ZI-C quote band around V(t), ticks
    jump_rate: float = 0.2
    jump_size: int = 5
    taker_bps: int = 10
    perp_horizon: float = 400.0
    noise_buy_prob: float = 0.6


# ---- part 1: continuous vs batch ------------------------------------------
class AnchoredZIC(ZICTrader):
    """ZI-C whose unit value is the V(t) it observed at its last wake-up."""
    fundamental: FundamentalProcess
    width = 3

    def price_bounds(self, value: int):  # schedule value ignored: V(t) anchors
        v = self.fundamental.value(self.view.time)
        return (max(1, v - self.width), v) if self.side is Side.BUY else (v, v + self.width)


def run_structure(cfg: Config, kind: str, latency: float):
    """(RunResult, FundamentalProcess) for 'continuous' or 'batch' at one latency."""
    stack = FrictionStack([MakerTakerFee(0, cfg.taker_bps)] + ([LatencyCost(latency)] if latency else []))
    fund = FundamentalProcess.random_walk(100, cfg.jump_rate, cfg.jump_size, seed=cfg.seed)
    agents = make_zi_population("zi-c", 6, 6, seed=cfg.seed, n_units=10**4, value_range=(100, 100))
    for a in agents:
        a.__class__, a.fundamental, a.width = AnchoredZIC, fund, cfg.width
    agents.append(InformedTrader(INFORMED_ID, fund, frictions=stack, order_size=2,
                                 max_position=30, wake_rate=2.0))
    agents += make_noise_population(3, seed=cfg.seed + 1, first_id=60, arrival_rate=0.5)
    kw = dict(seed=cfg.seed, cash=10**7, check_every=True)
    r = (MarketRunner(agents, stack, **kw) if kind == "continuous"
         else batch_runner(agents, stack, interval=cfg.batch_interval, **kw))
    return r.run(until=cfg.horizon), fund


def latency_sweep(cfg: Config = Config(), latencies=LATENCIES) -> List[Dict]:
    rows = []
    for d in latencies:
        runs = {k: run_structure(cfg, k, d) for k in ("continuous", "batch")}
        res = {k: r for k, (r, _) in runs.items()}
        fund = runs["batch"][1]
        stats = compare(res, fund.value, jump_times=jump_times_of(fund, cfg.horizon),
                        window=cfg.batch_interval + d)
        inf = {k: pnl_breakdown(r, fund.value(r.time), [INFORMED_ID])[INFORMED_ID].net
               for k, r in res.items()}
        rows.append({"latency": d, "stats": stats, "informed_pnl": inf, "results": res})
    return rows


# ---- part 2: cross-venue arbitrage under gas ------------------------------
class StaticMaker(Agent):
    """Posts one deep two-sided quote once, then idles (deterministic book)."""

    def __init__(self, bid: int, ask: int, depth: int = 10**5) -> None:
        super().__init__(MAKER_ID)
        self.quotes, self.wake_rate, self.depth = [(Side.BUY, bid), (Side.SELL, ask)], 50.0, depth

    def decide(self, view: MarketView):
        out, self.quotes = [Submit(s, self.depth, p) for s, p in self.quotes], []
        return out


def make_pool(kind: str, gas: int):
    st = FrictionStack([FixedCost(gas)]) if gas else None

    def factory(lg: Ledger):
        if kind == "cp":
            return ConstantProductAMM(lg, POOL_ID, 10_000, 1_000_000, fee_bps=30, frictions=st)
        pool = ConcentratedLiquidityAMM(lg, POOL_ID, 100, fee_bps=30, frictions=st)
        pool.add_position(80, 125, 10**6)  # ~ cp capital, concentrated on +/-20%
        return pool
    return factory


def band_width(kind: str, gas: int, max_gap: int = 40) -> int:
    """Buy-side + sell-side trigger gap (ticks) around the pool spot of 100."""
    pool = make_pool(kind, gas)(Ledger())
    arb = CrossVenueArbitrageur(ARB_ID, frictions=MakerTakerFee(0, 10), max_size=400)
    width = 0
    for side in (Side.BUY, Side.SELL):
        for gap in range(max_gap + 1):
            p, lv = 100 - side.value * gap, ((100 - side.value * gap, 10**4),)
            buy = side is Side.BUY
            v = MarketView(0.0, ARB_ID, None if buy else p, p if buy else None,
                           () if buy else lv, lv if buy else (), None, 10**8, 0)
            if arb.best_opportunity(v, {POOL_ID: pool}) is not None:
                break
        width += gap
    return width


def run_arb(kind: str, gas: int, seed: int = 7, ask: int = 92) -> Dict:
    """Book ask well below pool spot; the arbitrageur closes the gap net of friction."""
    r = MarketRunner([StaticMaker(ask - 4, ask), CrossVenueArbitrageur(ARB_ID, max_size=400)],
                     FrictionStack([MakerTakerFee(0, 10)]), seed=seed, cash=10**9,
                     position=10**6, pools=[make_pool(kind, gas)], check_every=True)
    res = r.run(max_wakeups=300)
    gap = float(res.pools[POOL_ID].spot_fraction) - res.book.best_ask
    return {"kind": kind, "gas": gas, "band": band_width(kind, gas), "residual_gap": gap,
            "n_arbs": len(res.arbs), "arb_pnl": res.arb_pnl(ARB_ID).net, "result": res}


def gas_sweep(seed: int = 7, gas=GAS) -> List[Dict]:
    return [run_arb(k, g, seed) for k in POOLS for g in gas]


# ---- part 3: perp funding -------------------------------------------------
class BeliefMaker(Agent):
    """Quotes +/- half around a belief that chases prints (no index knowledge)."""

    def __init__(self, agent_id: int = 3, half: int = 1, size: int = 3, learn: float = 0.25) -> None:
        super().__init__(agent_id)
        self.belief, self.half, self.size, self.learn = 100.0, half, size, learn
        self.wake_rate, self._seen = 8.0, 0

    def decide(self, view: MarketView):
        for t in view.recent_trades:
            if t.trade_id > self._seen:
                self._seen, self.belief = t.trade_id, self.belief + self.learn * (t.price - self.belief)
        m = round(self.belief)
        return [Cancel(o) for o in view.open_orders] + [
            Submit(Side.BUY, self.size, max(1, m - self.half)), Submit(Side.SELL, self.size, m + self.half)]


class CarryTrader(Agent):
    """Takes the side that receives funding whenever |last rate| >= min_bps."""

    def __init__(self, agent_id: int = CARRY_ID, size: int = 2, min_bps: int = 2) -> None:
        super().__init__(agent_id)
        self.perp: Optional[PerpMarket] = None
        self.size, self.min_bps, self.wake_rate = size, min_bps, 2.0

    def decide(self, view: MarketView):
        h = self.perp.history if self.perp else []
        if not h or abs(h[-1].rate_bps) < self.min_bps:
            return []
        side = Side.SELL if h[-1].rate_bps > 0 else Side.BUY
        px = view.best_bid if side is Side.SELL else view.best_ask
        return [] if px is None else [Submit(side, self.size, px)]


def run_perp(cfg: Config = Config(), funding: bool = True):
    """(RunResult, PerpMarket); funding=False sets k = 0 (rate always zero)."""
    carry = CarryTrader()
    noise = make_noise_population(4, seed=cfg.seed, first_id=10, arrival_rate=1.0,
                                  buy_prob=cfg.noise_buy_prob, max_size=2)
    r = MarketRunner([BeliefMaker(), carry] + noise, FrictionStack([MakerTakerFee(0, 5)]),
                     seed=cfg.seed, cash=10**7)
    index = FundamentalProcess.random_walk(100, 0.1, 2, seed=cfg.seed)
    carry.perp = PerpMarket(r.ledger, index, r.book, k=int(funding), cap_bps=100, interval=4.0)
    carry.perp.attach(r.clock)
    res = r.run(until=cfg.perp_horizon)
    carry.perp.sync()
    return res, carry.perp


def mean_abs_premium(perp: PerpMarket) -> float:
    h = perp.history
    return sum(abs(float(e.premium_bps)) for e in h) / len(h) if h else 0.0


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cfg = Config(seed=int(argv[0])) if argv else Config()
    print(f"seed={cfg.seed}  part 1: continuous vs batch (interval={cfg.batch_interval})")
    for row in latency_sweep(cfg):
        inf = row["informed_pnl"]
        print(f"-- latency={row['latency']}  informed_pnl continuous={inf['continuous']} "
              f"batch={inf['batch']}\n{summary(row['stats'])}")
    print(f"part 2: cross-venue arbitrage vs gas\n{'pool':<5}{'gas':>5}{'band':>6}"
          f"{'resid_gap':>11}{'arbs':>6}{'arb_pnl':>9}")
    for r in gas_sweep(cfg.seed):
        print(f"{r['kind']:<5}{r['gas']:>5}{r['band']:>6}{r['residual_gap']:>11.2f}"
              f"{r['n_arbs']:>6}{r['arb_pnl']:>9}")
    print(f"part 3: perp funding (noise buy_prob={cfg.noise_buy_prob:.2f})")
    for on in (True, False):
        _, perp = run_perp(cfg, on)
        path = " ".join(f"{float(e.rate_bps):.0f}" for e in perp.history[:12])
        print(f"funding={'on ' if on else 'off'} mean|premium|={mean_abs_premium(perp):8.1f} bps "
              f"carry_pos={perp.position(CARRY_ID):>5} rates[:12]=[{path}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
