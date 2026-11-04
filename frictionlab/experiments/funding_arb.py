"""Funding-rate arbitrage under gas shocks: how much basis survives when gas spikes?

An `InventoryMaker` quotes a perp ladder at index - inventory / kappa (Ho-Stoll
skew, clamped to +/- band); one-sided `RetailFlow` leaves it short, so mark sits
above index (mirror for direction=-1). fl-033 FundingArbitrageurs absorb that
inventory and pull mark back while they hold. Gas: fl-033 GasProcess, calm gas
fixed, grid = shock_gas_level x shock_probability; p = 0 is the constant-gas
baseline. Flow is book-blind, so gas reaches the premium only via the arb.

Entry is individually rational iff E * h - fees - impact - decision_cost > 4 g,
hence `entry_threshold_gas`; `analytic_bound` evaluates it at the friendliest
state the market allows (capped rate, top ladder price, fee floor, no impact).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..agents import FundingArbitrageur
from ..agents.base import Agent, Cancel, MarketView, Submit
from ..agents.funding_arb import LEGS
from ..core.runner import MarketRunner
from ..core.types import Side
from ..costs import TAKER_ONLY, Fill, FrictionStack, GasProcess, MakerTakerFee, Role
from ..markets import ConstantProductAMM, PerpMarket
from ..markets.amm_cp import BPS_DENOM
from .base import ExperimentResult, run_experiment

LEVEL, PROB = "shock_gas_level", "shock_probability"
POOL_ID, MAKER_ID, FIRST_ARB, FIRST_FLOW = 900, 50, 1, 100
CASH, POS = 10**9, 10**5


@dataclass(frozen=True)
class Config:
    """Funding-arb market: index, gas process, retail flow skew and arbitrageur limits."""
    index: int = 1_000
    calm_gas: int = 1
    p_calm: float = 0.05             # SHOCK -> CALM per gas step (persistent shocks)
    gas_dt: float = 1.0
    direction: int = 1               # +1: retail long, mark > index; -1 mirror
    flow_skew: float = 0.8
    n_flow: int = 4; flow_cap: int = 6; n_arbs: int = 1
    horizon: int = 4; max_size: int = 12; decision_cost: int = 0
    taker_bps: int = 10; lp_fee_bps: int = 5; pool_base: int = 20_000
    k: int = 1; cap_bps: int = 100; interval: float = 4.0
    kappa: int = 4                   # lots of maker inventory per tick of skew
    band: int = 6; half: int = 1; ladder: int = 4; depth: int = 10
    until: float = 200.0


class InventoryMaker(Agent):
    """Perp maker that skews its quotes against inventory, opening the basis under one-sided flow."""
    def __init__(self, cfg: Config) -> None:
        super().__init__(MAKER_ID)
        self.cfg, self.wake_rate = cfg, 8.0

    def decide(self, view: MarketView):
        c = self.cfg
        inv = view.position - POS
        m = c.index + max(-c.band, min(c.band, int(round(-inv / c.kappa))))
        out = [Cancel(o) for o in view.open_orders]
        for i in range(c.ladder):
            out += [Submit(Side.BUY, c.depth, m - c.half - i), Submit(Side.SELL, c.depth, m + c.half + i)]
        return out


class RetailFlow(Agent):
    """Book-blind one-sided market orders of 1 lot, position kept within [-cap, cap]."""

    def __init__(self, agent_id: int, rng, cfg: Config) -> None:
        super().__init__(agent_id)
        self.rng, self.cfg, self.wake_rate = rng, cfg, 1.0

    def decide(self, view: MarketView):
        c, u = self.cfg, self.rng.random()
        side = Side.BUY if (u < c.flow_skew) == (c.direction > 0) else Side.SELL
        inv = view.position - POS
        return [] if abs(inv + side.value) > c.flow_cap else [Submit(side, 1)]


def entry_threshold_gas(expected_funding, horizon, fees, impact, decision_cost) -> int:
    """Largest integer per-leg gas g with E * h - fees - impact - dc - 4 g > 0; -1 if none."""
    slack = Fraction(expected_funding) * horizon - Fraction(fees) - Fraction(impact) - decision_cost
    return -1 if slack <= 0 else -(-slack.numerator // (LEGS * slack.denominator)) - 1


def analytic_bound(cfg: Config) -> int:
    """Per-leg gas level above which entry cannot pay, from the config's worst case."""
    hi, lo = cfg.index + cfg.band + cfg.half + cfg.ladder, cfg.index - cfg.band - cfg.half - cfg.ladder
    e, fee = Fraction(hi * cfg.cap_bps, BPS_DENOM), Fraction(2 * lo * cfg.taker_bps, BPS_DENOM)
    return entry_threshold_gas(cfg.max_size * e, cfg.horizon, cfg.max_size * fee, 0, cfg.decision_cost)


def build(cfg: Config, params: Dict[str, Any], seed: int) -> MarketRunner:
    level, p = int(params[LEVEL]), float(params[PROB])
    if level < cfg.calm_gas:
        raise ValueError(f"shock gas {level} below calm gas {cfg.calm_gas}")
    ss = np.random.SeedSequence(seed).spawn(cfg.n_flow + 1)
    gas = GasProcess(cfg.calm_gas, level, p, cfg.p_calm, dt=cfg.gas_dt,
                     seed=int(ss[0].generate_state(1, np.uint64)[0]))
    pool = partial(ConstantProductAMM, pool_id=POOL_ID, reserve_base=cfg.pool_base,
                   reserve_quote=cfg.pool_base * cfg.index, fee_bps=cfg.lp_fee_bps,
                   frictions=FrictionStack([gas.cost(TAKER_ONLY)]))
    arbs = [FundingArbitrageur(FIRST_ARB + i, None, POOL_ID, gas=gas, horizon=cfg.horizon,
                               max_size=cfg.max_size, wake_rate=2.0, decision_cost=cfg.decision_cost)
            for i in range(cfg.n_arbs)]
    flow = [RetailFlow(FIRST_FLOW + i, np.random.default_rng(s), cfg) for i, s in enumerate(ss[1:])]
    stack = MakerTakerFee(0, cfg.taker_bps) + gas.cost(TAKER_ONLY)
    runner = MarketRunner([InventoryMaker(cfg), *arbs, *flow], stack, seed=seed,
                          cash=CASH, position=POS, pools=[pool])
    perp = PerpMarket(runner.ledger, lambda t: cfg.index, runner.book, k=cfg.k,
                      cap_bps=cfg.cap_bps, interval=cfg.interval)
    perp.attach(runner.clock)
    for a in arbs:
        a.perp = perp
    return runner


def arb_metrics(cfg: Config, res) -> Dict[str, Any]:
    """Per-run arbitrage metrics: entries, funding, friction breakdown, net PnL and residuals."""
    arbs = [a for a in res.agents.values() if isinstance(a, FundingArbitrageur)]
    perp, gas = arbs[0].perp, arbs[0].gas
    perp.sync()
    taker = FrictionStack([MakerTakerFee(0, cfg.taker_bps)])
    keys = ("entries", "gross_funding", "perp_fee", "gas", "decision_cost", "trading", "net_pnl",
            "net_delta", "recon_residual", "fee_residual", "gas_residual", "funding_residual")
    m, holds, entry_gas = dict.fromkeys(keys, 0), [], [-1]
    for a in arbs:
        aid, pos, acct = a.agent_id, a.report(), res.ledger[a.agent_id]
        fee_i = gas_i = 0
        for f in (f for f in res.arbs if f.agent_id == aid):
            side = f.pair.book.side
            fee_i += sum(taker.charge(Fill(aid, side, Role.TAKER, t.price, t.qty, t.time))
                         for t in f.book_trades)
            gas_i += gas.level(f.time) * (len(f.book_trades) + (f.swap is not None))
        funding, trading, fees = (sum(getattr(p, k) for p in pos) for k in ("funding", "trading", "fees"))
        cash, dc = acct.cash - CASH, a.decision_costs_paid
        for k, v in (("entries", a.n_entries), ("gross_funding", funding), ("perp_fee", fee_i),
                     ("gas", gas_i), ("decision_cost", dc), ("trading", trading),
                     ("net_delta", abs(acct.position - POS)),
                     ("net_pnl", cash + (acct.position - POS) * cfg.index),
                     ("recon_residual", abs(cash - (trading + funding - fees - dc))),
                     ("fee_residual", abs(acct.fees_paid - fee_i - gas_i)),
                     ("gas_residual", abs(sum(p.gas for p in pos) - gas_i)),
                     ("funding_residual", abs(funding - sum(e.payments.get(aid, 0) for e in perp.history)))):
            m[k] += v
        holds += [p.close_time - p.open_time for p in pos if p.close_time is not None]
        entry_gas += [gas.level(p.open_time) for p in pos]
    x, y, amm = cfg.pool_base, cfg.pool_base * cfg.index, Fraction(0)
    for s in (f.swap for f in res.arbs if f.swap is not None):   # replay: LP fee + impact vs spot
        spot, buy = Fraction(y, x), s.side is Side.BUY
        amm += s.amount_in - s.amount_out * spot if buy else s.amount_in * spot - s.amount_out
        x, y = (x - s.amount_out, y + s.amount_in) if buy else (x + s.amount_in, y - s.amount_out)
    h = perp.history
    m.update(amm_cost=float(amm), basis_pnl=float(m.pop("trading") + amm), entry_gas_max=max(entry_gas),
             mean_holding=float(np.mean(holds)) if holds else 0.0,
             mean_abs_premium=sum(abs(float(e.premium_bps)) for e in h) / len(h) if h else 0.0,
             pool_residual=int(res.pools[POOL_ID].reserves != (x, y)),
             oi_residual=perp.open_interest - perp.short_interest,
             violations=len(res.ledger.violations()))
    return m


def default_levels(cfg: Config = Config()) -> List[int]:
    """Default shock gas grid: calm, two small steps, the midpoint and just past the bound."""
    b, c = max(analytic_bound(cfg), cfg.calm_gas), cfg.calm_gas
    return sorted({c, c + 4, c + 10, (c + b) // 2, b + 1})


def run_funding_arb(cfg: Config = Config(), levels: Optional[Sequence[int]] = None,
                    probs: Sequence[float] = (0.0, 0.3, 0.6), *, replicates: int = 1,
                    master_seed: int = 34, n_workers: int = 1) -> ExperimentResult:
    """Sweep shock gas level x shock probability and report arbitrage activity per cell."""
    levels = [int(v) for v in (default_levels(cfg) if levels is None else levels)]
    probs = [float(p) for p in probs]

    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        cell = lambda lv, p, k: float(np.mean([r[k] for r in rows if r[LEVEL] == lv and r[PROB] == p]))
        out: Dict[str, Any] = {"analytic_bound": analytic_bound(cfg)}
        for tag, lv, p in (("baseline", levels[0], probs[0]), ("shock", levels[-1], probs[-1])):
            for k in ("entries", "net_pnl", "mean_abs_premium"):
                out[f"{tag}_{k}"] = cell(lv, p, k)
        return out

    return run_experiment("funding_arb", partial(build, cfg), partial(arb_metrics, cfg),
                          {LEVEL: levels, PROB: probs}, replicates=replicates,
                          master_seed=master_seed, crn=True, run_kwargs={"until": cfg.until},
                          n_workers=n_workers, findings=headline)


@dataclass(frozen=True)
class GasThreshold:
    """Entry-collapse shock gas level with the analytic bound and probes."""
    threshold: Optional[int]         # smallest shock level with mean entries < target
    analytic_bound: int
    shock_probability: float
    baseline_entries: float
    target: float
    probes: Dict[int, float] = field(repr=False)

    @property
    def within_bound(self) -> bool:
        return self.threshold is not None and self.threshold <= self.analytic_bound + 1


def find_gas_threshold(cfg: Config = Config(), shock_probability: float = 0.6, fraction: float = 0.5,
                       *, grid: Optional[Sequence[int]] = None, replicates: int = 1,
                       master_seed: int = 34, n_workers: int = 1) -> GasThreshold:
    """Grid scan up to bound + 1, then integer bisection; same child seeds at every probe."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    kw = dict(replicates=replicates, master_seed=master_seed, n_workers=n_workers)
    base = run_funding_arb(cfg, [cfg.calm_gas], [0.0], **kw).column("entries")[0]
    probes: Dict[int, float] = {}

    def probe(levels: Sequence[int]) -> None:
        todo = sorted({int(v) for v in levels} - set(probes))
        if todo:
            probes.update(zip(todo, run_funding_arb(cfg, todo, [shock_probability], **kw).column("entries")))

    bound, target = analytic_bound(cfg), fraction * base
    scan = sorted({cfg.calm_gas, max(cfg.calm_gas, bound + 1), *(grid or default_levels(cfg))})
    probe(scan)
    hi = next((lv for lv in scan if probes[lv] < target), None)
    if hi is not None and hi > scan[0]:
        lo = max(lv for lv in scan if lv < hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            probe([mid])
            lo, hi = (lo, mid) if probes[mid] < target else (mid, hi)
    return GasThreshold(hi, bound, shock_probability, base, target, dict(sorted(probes.items())))
