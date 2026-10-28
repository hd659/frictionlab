"""Phase 2 example: adverse selection, or how informed flow widens spreads.

One inventory-risk market maker (Avellaneda-Stoikov) quotes against a fixed
total taker arrival rate `lam`. A share `pi` of that flow comes from an
informed trader who sees the latent fundamental V(t) (a seeded jump random
walk) and only trades when its edge beats friction; the rest is value-blind
noise. Following Glosten-Milgrom, the maker knows `pi` (rational
expectations) and adds an adverse-selection premium to each side,

    premium = ceil(pi * adverse_ticks)        (ticks per side)

where `adverse_ticks` is the value move it expects to lose per informed fill.
Its mid is a belief, not the (self-made) book mid: each new print moves it by
`learn * (price - mid)`, a reduced-form Glosten-Milgrom update (buys push the
belief up, sells push it down), so informed flow drags quotes toward V(t).
Per run we report the time-weighted quoted spread, the maker's net PnL, the
maker's markout against informed flow, and the informed trader's net PnL.

    python examples/phase2_adverse_selection.py [seed]
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from frictionlab.agents import (
    ASQuote, AvellanedaStoikovMM, FundamentalProcess, InformedTrader,
    MarketView, make_noise_population,
)
from frictionlab.analysis import QuoteRecorder, markout_pnl, pnl_by_agent, volume_between
from frictionlab.core.runner import MarketRunner, RunResult
from frictionlab.costs import FrictionStack, MakerTakerFee

SHARES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
MM_ID, INFORMED_ID, FIRST_NOISE_ID = 1, 2, 3


@dataclass(frozen=True)
class Config:
    seed: int = 7
    horizon: float = 300.0
    lam: float = 4.0             # total taker arrivals per time unit
    n_noise: int = 4
    v0: int = 100
    jump_rate: float = 0.2
    jump_size: int = 4
    adverse_ticks: int = 10      # maker's expected loss per informed fill
    taker_bps: int = 10
    gamma: float = 0.01
    sigma: float = 1.0
    inventory_limit: int = 30
    informed_cap: int = 20
    mm_rate: float = 8.0         # maker re-quotes per time unit
    quote_size: int = 3
    learn: float = 0.5
    cash: int = 10**7


class GMMarketMaker(AvellanedaStoikovMM):
    """A-S quotes widened by a Glosten-Milgrom adverse-selection premium."""

    def __init__(self, agent_id: int, informed_share: float, adverse_ticks: int, *,
                 learn: float = 0.5, wake_rate: float = 1.0, **kw) -> None:
        super().__init__(agent_id, **kw)
        if not 0.0 <= informed_share <= 1.0:
            raise ValueError(f"informed_share must lie in [0, 1], got {informed_share}")
        if not 0.0 <= learn <= 1.0:
            raise ValueError(f"learn must lie in [0, 1], got {learn}")
        self.informed_share = float(informed_share)
        self.premium = math.ceil(round(self.informed_share * adverse_ticks, 9))
        self.learn = float(learn)
        self.wake_rate = float(wake_rate)  # read by MarketRunner
        self._seen: Optional[int] = None

    def estimate_mid(self, view: MarketView) -> float:
        t = view.last_trade
        if t is not None and t.trade_id != self._seen:
            self._seen = t.trade_id
            self._mid += self.learn * (t.price - self._mid)
        return self._mid

    def quote(self, view: MarketView) -> ASQuote:
        q = super().quote(view)
        if not self.premium:
            return q
        bid = None if q.bid is None else max(1, q.bid - self.premium)
        ask = None if q.ask is None else q.ask + self.premium
        return replace(q, bid=bid, ask=ask, spread=q.spread + 2 * self.premium)


def build(cfg: Config, share: float, seed: int):
    stack = FrictionStack([MakerTakerFee(maker_bps=0, taker_bps=cfg.taker_bps)])
    fund = FundamentalProcess.random_walk(cfg.v0, cfg.jump_rate, cfg.jump_size, seed=seed)
    mm = GMMarketMaker(MM_ID, share, cfg.adverse_ticks, gamma=cfg.gamma, sigma=cfg.sigma,
                       horizon=cfg.horizon, inventory_limit=cfg.inventory_limit,
                       fair_value=cfg.v0, order_size=cfg.quote_size, learn=cfg.learn,
                       wake_rate=cfg.mm_rate, rng=seed)
    agents = [mm]
    if share > 0:
        agents.append(InformedTrader(INFORMED_ID, fund, frictions=stack,
                                     max_position=cfg.informed_cap,
                                     wake_rate=share * cfg.lam))
    if share < 1:
        rate = (1 - share) * cfg.lam / cfg.n_noise
        agents += make_noise_population(cfg.n_noise, seed=seed, first_id=FIRST_NOISE_ID,
                                        arrival_rate=rate)
    return MarketRunner(agents, stack, seed=seed, cash=cfg.cash), fund


def simulate(cfg: Config, share: float, seed: Optional[int] = None):
    """(RunResult, FundamentalProcess, QuoteRecorder) of one seeded run."""
    runner, fund = build(cfg, share, cfg.seed if seed is None else seed)
    rec = QuoteRecorder()
    res = runner.run(until=cfg.horizon, stop_when=rec)
    return res, fund, rec


def metrics(cfg: Config, share: float, res: RunResult, fund: FundamentalProcess,
            rec: QuoteRecorder) -> Dict[str, float]:
    v_end = fund.value(res.time)
    pnl = pnl_by_agent(res, v_end)
    st = rec.stats(until=cfg.horizon)
    inf = [INFORMED_ID] if INFORMED_ID in res.agents else []
    noise = [a for a in res.agents if a >= FIRST_NOISE_ID]
    return {"share": share, "spread": st.tw_spread, "coverage": st.coverage,
            "trades": len(res.trades), "volume": sum(t.qty for t in res.trades),
            "mm_pnl": pnl[MM_ID], "informed_pnl": pnl.get(INFORMED_ID, 0),
            "noise_pnl": sum(v for a, v in pnl.items() if a >= FIRST_NOISE_ID),
            "mm_vs_informed": markout_pnl(res.trades, MM_ID, fund.value, inf),
            "mm_vs_noise": markout_pnl(res.trades, MM_ID, fund.value, noise),
            "informed_volume": volume_between(res.trades, MM_ID, inf),
            "fees": res.fees, "v_end": v_end}


def run(cfg: Config = Config(), share: float = 0.0, seed: Optional[int] = None) -> Dict[str, float]:
    res, fund, rec = simulate(cfg, share, seed)
    return metrics(cfg, share, res, fund, rec)


def sweep(cfg: Config = Config(), shares: Sequence[float] = SHARES) -> List[Dict[str, float]]:
    return [run(cfg, s) for s in shares]


def report(rows: Sequence[Dict[str, float]]) -> str:
    lines = [f"{'pi':>5}{'spread':>8}{'cover':>7}{'volume':>8}{'mm_pnl':>9}"
             f"{'mm_vs_inf':>10}{'mm_vs_noise':>12}{'inf_pnl':>9}{'noise_pnl':>10}"]
    for r in rows:
        lines.append(f"{r['share']:>5.2f}{r['spread']:>8.2f}{r['coverage']:>7.2f}"
                     f"{r['volume']:>8}{r['mm_pnl']:>9}{r['mm_vs_informed']:>10}{r['mm_vs_noise']:>12}"
                     f"{r['informed_pnl']:>9}{r['noise_pnl']:>10}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cfg = Config(seed=int(argv[0])) if argv else Config()
    print(f"seed={cfg.seed} lam={cfg.lam} horizon={cfg.horizon} "
          f"adverse_ticks={cfg.adverse_ticks} taker_bps={cfg.taker_bps}")
    print(report(sweep(cfg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
