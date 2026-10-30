"""Adverse selection: how informed flow widens spreads, vs the Glosten-Milgrom benchmark.

Analytic part. V is low or high with prior 1/2. A share mu of takers is
informed (buys iff V high), the rest is noise that buys with probability q.
A risk-neutral competitive maker breaks even per fill at

    ask = E[V | buy]  = v_low + (v_high - v_low) * P(H | buy)
    bid = E[V | sell] = v_low + (v_high - v_low) * P(H | sell)
    P(H | buy) = (mu + (1-mu) q) / (mu + 2 (1-mu) q),  P(H | sell) likewise with 1-q

computed in exact Fractions and put on the tick grid as (floor bid, ceil ask),
so the integer quote never loses in expectation. With q = 1/2: ask = mean + mu * L.

Simulated part. A `GMMaker` (Avellaneda-Stoikov base, belief mid updated from
prints) widens each side by the GM premium for a two-point value mean +- L,
L = alpha * jump, at its known informed share. It faces an `InformedTrader`
(rate mu * lam) and noise (rate (1 - mu) * lam) on a jump random walk, swept
over share x jump on the fl-023 engine. Per maker fill j with maker side s_j,
integer reference r_j = floor(pre-trade book mid) and final value V_T:

    trading = sum s_j (V_T - p_j) q_j = capture + adverse
    capture = sum s_j (r_j - p_j) q_j      adverse = sum s_j (V_T - r_j) q_j

exactly in integer ticks; the adverse-selection loss is -adverse.
`mitigation` reruns one cell as continuous book vs batch auction (fl-020).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import partial
from typing import Any, Dict, List, NamedTuple, Sequence

import numpy as np

from ..agents import AvellanedaStoikovMM, FundamentalProcess, InformedTrader, make_noise_population
from ..analysis import (
    MidPath, QuoteRecorder, decompose, pnl_breakdown, spread_decomposition, trade_records,
)
from ..analysis.structure_compare import jump_times_of, structure_stats
from ..analysis.welfare import pnl_by_account
from ..core.runner import MarketRunner
from ..costs import FrictionStack, MakerTakerFee
from ..markets import batch_runner
from .base import ExperimentResult, run_experiment

MM_ID, INF_ID, NOISE_ID = 1, 2, 3
SHARES, JUMPS, STRUCTURES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5), (2, 4, 8), ("continuous", "batch")


def _prob(name: str, x, open_: bool = False) -> Fraction:
    f = Fraction(str(x)) if isinstance(x, float) else Fraction(x)
    if not 0 <= f <= 1 or (open_ and f in (0, 1)):
        raise ValueError(f"{name} must lie in {'(0, 1)' if open_ else '[0, 1]'}, got {x}")
    return f


@dataclass(frozen=True)
class GMQuote:
    bid: int                 # floor(E[V | sell])
    ask: int                 # ceil(E[V | buy])
    bid_exact: Fraction
    ask_exact: Fraction
    mean: Fraction           # prior mean (v_low + v_high) / 2

    @property
    def spread(self) -> int:
        return self.ask - self.bid


def gm_zero_profit_spread(mu, v_low: int, v_high: int, noise_buy_prob=0.5) -> GMQuote:
    """Break-even GM bid/ask (integer ticks) for informed share mu, two-point V."""
    m, q = _prob("mu", mu), _prob("noise_buy_prob", noise_buy_prob, open_=True)
    if int(v_low) != v_low or int(v_high) != v_high or v_low > v_high:
        raise ValueError(f"need integer ticks v_low <= v_high, got {v_low}, {v_high}")
    lo, w = int(v_low), int(v_high) - int(v_low)
    ask = lo + w * (m + (1 - m) * q) / (m + 2 * (1 - m) * q)
    bid = lo + w * ((1 - m) * (1 - q)) / (m + 2 * (1 - m) * (1 - q))
    return GMQuote(math.floor(bid), math.ceil(ask), bid, ask, Fraction(lo + int(v_high), 2))


@dataclass(frozen=True)
class Config:
    horizon: float = 200.0
    lam: float = 4.0              # total taker arrivals per time unit
    n_noise: int = 4
    v0: int = 100
    jump_rate: float = 0.2
    alpha: int = 5                # GM half-width L = alpha * jump (ticks)
    noise_buy_prob: float = 0.5
    taker_bps: int = 10
    gamma: float = 0.01
    sigma: float = 1.0
    inventory_limit: int = 30
    informed_cap: int = 20
    mm_rate: float = 8.0
    quote_size: int = 3
    learn: float = 0.5
    tau: float = 5.0              # realized-spread / impact horizon
    batch_interval: float = 1.0
    cash: int = 10**7

    def gm(self, share: float, jump: int) -> GMQuote:
        L = self.alpha * int(jump)
        return gm_zero_profit_spread(share, self.v0 - L, self.v0 + L, self.noise_buy_prob)


class GMMaker(AvellanedaStoikovMM):
    """A-S quotes around a print-learned belief, widened by the GM premium."""

    def __init__(self, agent_id: int, gm: GMQuote, *, learn: float, wake_rate: float, **kw):
        super().__init__(agent_id, **kw)
        self.p_ask, self.p_bid = int(gm.ask - self.fair_value), int(self.fair_value - gm.bid)
        self.learn, self.wake_rate, self._seen = float(learn), float(wake_rate), None

    def estimate_mid(self, view) -> float:
        t = view.last_trade
        if t is not None and t.trade_id != self._seen:
            self._seen = t.trade_id
            self._mid += self.learn * (t.price - self._mid)
        return self._mid

    def quote(self, view):
        q = super().quote(view)
        bid = None if q.bid is None else max(1, q.bid - self.p_bid)
        ask = None if q.ask is None else q.ask + self.p_ask
        return replace(q, bid=bid, ask=ask, spread=q.spread + self.p_bid + self.p_ask)


class Outcome(NamedTuple):
    result: Any
    fund: FundamentalProcess
    path: QuoteRecorder
    params: Dict[str, Any]


class Scenario:
    def __init__(self, runner, fund, params) -> None:
        self.runner, self.fund, self.params = runner, fund, params

    def run(self, until: float) -> Outcome:
        rec = QuoteRecorder()
        return Outcome(self.runner.run(until=until, stop_when=rec), self.fund, rec, self.params)


def build(cfg: Config, params: Dict[str, Any], seed: int) -> Scenario:
    share, jump = float(params["share"]), int(params["jump"])
    ss = np.random.SeedSequence(seed).spawn(3)
    stack = FrictionStack([MakerTakerFee(maker_bps=0, taker_bps=cfg.taker_bps)])
    fund = FundamentalProcess.random_walk(cfg.v0, cfg.jump_rate, jump, seed=np.random.default_rng(ss[0]))
    agents = [GMMaker(MM_ID, cfg.gm(share, jump), learn=cfg.learn, wake_rate=cfg.mm_rate,
                      gamma=cfg.gamma, sigma=cfg.sigma, horizon=cfg.horizon,
                      inventory_limit=cfg.inventory_limit, fair_value=cfg.v0,
                      order_size=cfg.quote_size, rng=np.random.default_rng(ss[1]))]
    if share > 0:
        agents.append(InformedTrader(INF_ID, fund, frictions=stack, max_position=cfg.informed_cap,
                                     wake_rate=share * cfg.lam))
    if share < 1:
        agents += make_noise_population(cfg.n_noise, seed=int(ss[2].generate_state(1)[0]),
                                        first_id=NOISE_ID, buy_prob=cfg.noise_buy_prob,
                                        arrival_rate=(1 - share) * cfg.lam / cfg.n_noise)
    kw = dict(seed=seed, cash=cfg.cash)
    batch = params.get("structure", "continuous") == "batch"
    runner = (batch_runner(agents, stack, interval=cfg.batch_interval, **kw) if batch
              else MarketRunner(agents, stack, **kw))
    return Scenario(runner, fund, dict(params))


def maker_split(res, path: MidPath, mark: int) -> Dict[str, int]:
    """Maker trading PnL at `mark` = spread capture + adverse-selection component."""
    cap = adv = vol = 0
    for t in res.trades:
        if MM_ID not in (t.buyer_id, t.seller_id):
            continue
        s, pre = (1 if t.buyer_id == MM_ID else -1), path.before(t.time)
        ref = t.price if pre is None else pre // 2
        cap += s * (ref - t.price) * t.qty
        adv += s * (mark - ref) * t.qty
        vol += t.qty
    return {"mm_volume": vol, "mm_spread_capture": cap, "mm_adverse_selection": adv,
            "mm_as_loss": -adv, "mm_as_per_unit": -adv / vol if vol else 0.0}


def metrics(cfg: Config, out: Outcome) -> Dict[str, Any]:
    res, fund, p = out.result, out.fund, out.params
    mark, path, recs = fund.value(res.time), MidPath(out.path), trade_records(out.result)
    sd, per = spread_decomposition(recs, path, cfg.tau), decompose(recs, path, cfg.tau)
    st = out.path.stats(until=res.time)
    gm = cfg.gm(p["share"], p["jump"])
    split = maker_split(res, path, mark)
    pb = pnl_breakdown(res, mark, [a for a in (MM_ID, INF_ID) if a in res.agents])
    ss = structure_stats(res.trades, fund.value, jump_times=jump_times_of(fund, res.time),
                         window=cfg.batch_interval)
    z = lambda x: 0.0 if x != x else float(x)          # nan -> 0 keeps rows comparable
    return {"tw_spread": z(st.tw_spread), "coverage": st.coverage, "gm_spread": gm.spread,
            "excess_spread": z(st.tw_spread) - gm.spread, "effective_spread": z(sd.effective),
            "realized_spread": z(sd.realized), "price_impact": z(sd.impact),
            "impact_per_trade": float(np.mean([m.impact for m in per])) if per else 0.0,
            "identity_residual": max((abs(m.effective - m.realized - m.impact) for m in per), default=0),
            "n_trades": len(res.trades), "volume": sum(t.qty for t in res.trades),
            "mm_trading": pb[MM_ID].trading, **split,
            "split_residual": split["mm_spread_capture"] + split["mm_adverse_selection"] - pb[MM_ID].trading,
            "mm_net": pb[MM_ID].net, "informed_net": pb[INF_ID].net if INF_ID in pb else 0,
            "snipe_share": ss.snipe_share, "violations": len(res.ledger.violations()),
            "zero_sum_residual": sum(pnl_by_account(res, mark).values())}


def run_grid(cfg: Config = Config(), shares: Sequence[float] = SHARES, jumps: Sequence[int] = JUMPS,
             *, replicates: int = 3, master_seed: int = 30, n_workers: int = 1) -> ExperimentResult:
    def headline(rows) -> Dict[str, Any]:
        mean = lambda k, s, j: float(np.mean([r[k] for r in rows if r["share"] == s and r["jump"] == j]))
        informed = [r for r in rows if r["share"] > 0]
        impact = {j: float(np.mean([r["impact_per_trade"] for r in informed if r["jump"] == j]))
                  if informed else 0.0 for j in jumps}
        return {f"gm_spread[mu={s:g},J={j}]": cfg.gm(s, j).spread for s in (shares[0], shares[-1]) for j in jumps} | {
            **{f"impact_per_trade[J={j}]": v for j, v in impact.items()},
            "spread_monotone_in_share": all(
                all(b >= a for a, b in zip(col, col[1:]))
                for col in ([mean("tw_spread", s, j) for s in shares] for j in jumps))}

    return run_experiment("adverse_selection", partial(build, cfg), partial(metrics, cfg),
                          {"share": list(shares), "jump": list(jumps)}, replicates=replicates,
                          master_seed=master_seed, crn=True, run_kwargs={"until": cfg.horizon},
                          n_workers=n_workers, findings=headline)


def mitigation(cfg: Config, share: float, jump: int, *, replicates: int = 3,
               master_seed: int = 30, n_workers: int = 1) -> ExperimentResult:
    """One (share, jump) cell as continuous book vs frequent batch auction."""
    return run_experiment("adverse_selection_mitigation", partial(build, cfg), partial(metrics, cfg),
                          {"structure": list(STRUCTURES)}, base={"share": share, "jump": jump},
                          replicates=replicates, master_seed=master_seed, crn=True,
                          run_kwargs={"until": cfg.horizon}, n_workers=n_workers)


def run_study(cfg: Config = Config(), shares: Sequence[float] = SHARES, jumps: Sequence[int] = JUMPS,
              *, replicates: int = 3, master_seed: int = 30, n_workers: int = 1) -> ExperimentResult:
    """Grid + mitigation at the highest share (largest jump), one ExperimentResult."""
    res = run_grid(cfg, shares, jumps, replicates=replicates, master_seed=master_seed, n_workers=n_workers)
    mit = mitigation(cfg, max(shares), max(jumps), replicates=replicates,
                     master_seed=master_seed, n_workers=n_workers)
    agg = {r["structure"]: r for r in mit.aggregate()}
    c, b = agg["continuous"]["mm_as_per_unit_mean"], agg["batch"]["mm_as_per_unit_mean"]
    res.findings.update({"mitigation_share": max(shares), "mitigation_jump": max(jumps),
                         "as_per_unit_continuous": c, "as_per_unit_batch": b,
                         "as_per_unit_batch_minus_continuous": b - c,
                         "mitigation_rows": mit.rows})
    return res


def study_report(res: ExperimentResult):
    """fl-026 report: findings, share x jump heatmap, structure comparison."""
    from ..analysis.report import Report
    from ..analysis.report_sections import MetricsSection, SweepSection
    flat = {k: v for k, v in res.findings.items() if k != "mitigation_rows"}
    secs = [MetricsSection(extra=flat, title="Adverse selection: findings"),
            SweepSection(res.rows, res.param_keys, "tw_spread", title="Adverse selection: sweep"),
            SweepSection(res.findings.get("mitigation_rows", []), ["structure"], "mm_as_per_unit",
                         title="Adverse selection: batch vs continuous")]
    return Report("frictionlab experiment: adverse_selection", master_seed=res.master_seed,
                  grid=res.params, meta={"experiment": res.name}, sections=secs)
