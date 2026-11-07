"""Grossman-Stiglitz: who pays for information when prices already carry it?

V = center +- jump. `GSMaker` quotes its posterior mid +- half the fl-030 GM spread at
the live informed share mu; fl-052 `InfoAcquirer`s buy iff the edge net of that half-spread
beats the price. More buyers widen the spread, more noise invites buyers; once the price
eats the edge nobody buys and the mid stays at the prior: prices stop carrying information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..agents import InfoAcquirer, NoiseTrader, make_noise_population
from ..agents.base import Agent, Cancel, Submit
from ..analysis import MidPath, QuoteRecorder, info_pnl, price_efficiency
from ..analysis.equilibrium import SimulatedFitness, StabilityDetector, evolve
from ..core.runner import MarketRunner
from ..core.types import Side
from ..costs import INFO_SINK_ID, SignalSource, expected_edge, two_point
from .adverse_selection import gm_zero_profit_spread
from .base import ExperimentResult, run_experiment

COST, NOISE, LABELS = "signal_cost", "noise_intensity", ("buyer", "non_buyer")
MM_ID, FIRST_ACQ, FIRST_NOISE = 1, 100, 200
COSTS, NOISES = (0, 1, 2, 4, 6, 8, 10), (1.0, 4.0, 16.0)


@dataclass(frozen=True)
class Config:
    """Two-point fundamental, fixed signal accuracy, population sizes, dealer and horizon."""
    center: int = 100
    jump: int = 10
    accuracy_bps: int = 9_000
    n_acquirers: int = 8
    acquirer_rate: float = 1.0
    n_noise: int = 4
    noise_buy_prob: float = 0.5
    maker_rate: float = 4.0
    quote_size: int = 50
    horizon: float = 60.0
    cash: int = 10**7


def gs_equilibrium_informed_share(signal_cost: int, accuracy_bps: int, jump: int,
                                  noise_buy_prob: float = 0.5) -> float:
    """Largest integer-percent mu at which the edge net of the GM half-spread is >= cost."""
    for pct in range(100, 0, -1):
        gm = gm_zero_profit_spread(Fraction(pct, 100), 0, 2 * int(jump), noise_buy_prob)
        if expected_edge(accuracy_bps, jump, gm.spread) >= int(signal_cost):
            return pct / 100
    return 0.0


def analytic_collapse_cost(cfg: Config) -> int:
    """Smallest signal cost with a zero analytic informed share."""
    gm = gm_zero_profit_spread(Fraction(1, 100), 0, 2 * cfg.jump, cfg.noise_buy_prob)
    return max(0, expected_edge(cfg.accuracy_bps, cfg.jump, gm.spread) + 1)


class GSMaker(Agent):
    """Dealer quoting its posterior mid +- half the GM spread at the live informed share."""

    def __init__(self, agent_id: int, cfg: Config, acquirers: Sequence[Agent], total_rate: float):
        super().__init__(agent_id)
        self.lo, self.w, self.size = cfg.center - cfg.jump, 2 * cfg.jump, cfg.quote_size
        self.a, self.q = cfg.accuracy_bps / 10_000, cfg.noise_buy_prob
        self.acq, self.rate, self.total = list(acquirers), cfg.acquirer_rate, float(total_rate)
        self.wake_rate, self.p, self._seen = cfg.maker_rate, 0.5, -1

    def informed_share(self) -> float:
        """Taker-flow share of acquirers whose latest decision bought a signal."""
        k = sum(1 for x in self.acq if x.history and x.history[-1].bought)
        return min(1.0, k * self.rate / self.total) if self.total > 0 else 0.0

    def update(self, buy: bool, mu: float) -> None:
        """Bayes step on P(V high) after one taker buy (or sell) at informed share mu."""
        a, q = (self.a, self.q) if buy else (1 - self.a, 1 - self.q)
        lh, ll = mu * a + (1 - mu) * q, mu * (1 - a) + (1 - mu) * q
        self.p = min(1 - 1e-9, max(1e-9, self.p * lh / (self.p * lh + (1 - self.p) * ll)))

    def quote(self, mu: float) -> tuple:
        """(bid, ask) = round(posterior mid) -+ max(1, ceil(GM spread at mu / 2))."""
        gm = gm_zero_profit_spread(Fraction(mu).limit_denominator(10_000), self.lo, self.lo + self.w, self.q)
        h, m = max(1, -(-gm.spread // 2)), int(round(self.lo + self.w * self.p))
        return max(1, m - h), m + h

    def decide(self, view) -> List[Any]:
        mu = self.informed_share()
        for t in (t for t in view.recent_trades if t.trade_id > self._seen):
            if self.agent_id in (t.buyer_id, t.seller_id):
                self.update(t.seller_id == self.agent_id, mu)
            self._seen = t.trade_id
        bid, ask = self.quote(mu)
        return ([Cancel(o) for o in view.open_orders]
                + [Submit(Side.BUY, self.size, bid), Submit(Side.SELL, self.size, ask)])


class CommittedBuyer(InfoAcquirer):
    """Replicator buyer archetype: buys the signal on every wake-up, whatever its price."""
    hurdle = -(10**9)          # shadows the fl-052 property: no expected edge misses it


class GSCell:
    """One seeded market: dealer, acquirers, noise, the signal source and a quote path."""

    def __init__(self, runner: MarketRunner, source: SignalSource, acquirers: List[Agent]):
        self.runner, self.source, self.acquirers, self.quotes = runner, source, acquirers, QuoteRecorder()

    def run(self, until: float) -> "GSCell":
        """Run to `until`, recording the touch after every wake-up."""
        self.result, self.until = self.runner.run(until=until, stop_when=self.quotes), float(until)
        return self

    def parts(self):
        """(true V, fl-052 per-acquirer PnL split) for the finished run."""
        v = int(self.source.fundamental.value(0.0))
        return v, info_pnl(self.result, self.source, v, [a.agent_id for a in self.acquirers])

    def fitness(self) -> Dict[str, float]:
        """Mean net PnL per archetype (buyer / non_buyer), the replicator payoff."""
        parts, groups = self.parts()[1], {}
        for a in self.acquirers:
            groups.setdefault(LABELS[isinstance(a, NoiseTrader)], []).append(parts[a.agent_id].net)
        return {k: sum(x) / len(x) for k, x in groups.items()}


def build(cfg: Config, params: Dict[str, Any], seed: int) -> GSCell:
    """fl-023 factory; `params["counts"]` (replicator) swaps in buyer / non-buyer archetypes."""
    cost, lam, counts = int(params[COST]), float(params[NOISE]), params.get("counts")
    ss, child = np.random.SeedSequence(seed).spawn(4), lambda s: int(s.generate_state(1)[0])
    src = SignalSource(two_point(cfg.center, cfg.jump, seed=child(ss[0])), cfg.accuracy_bps,
                       price=cost, center=cfg.center, jump=cfg.jump, seed=child(ss[1]))
    n_buy, n_non = (cfg.n_acquirers, 0) if counts is None else (int(counts.get(k, 0)) for k in LABELS)
    n, kw = n_buy + n_non, {"buy_prob": cfg.noise_buy_prob}
    kind, rngs = InfoAcquirer if counts is None else CommittedBuyer, ss[2].spawn(max(n, 1))
    pop: List[Agent] = [kind(FIRST_ACQ + i, src, rng=np.random.default_rng(rngs[i]),
                             wake_rate=cfg.acquirer_rate, **kw) for i in range(n_buy)]
    pop += [NoiseTrader(FIRST_ACQ + i, np.random.default_rng(rngs[i]), arrival_rate=cfg.acquirer_rate,
                        **kw) for i in range(n_buy, n)]
    noise = make_noise_population(cfg.n_noise, seed=child(ss[3]), first_id=FIRST_NOISE,
                                  arrival_rate=lam / cfg.n_noise, **kw) if lam > 0 else []
    mm = GSMaker(MM_ID, cfg, pop[:n_buy], n * cfg.acquirer_rate + lam)
    return GSCell(MarketRunner([mm] + pop + noise, None, seed=seed, cash=cfg.cash), src, pop)


def gs_metrics(cfg: Config, cell: GSCell) -> Dict[str, Any]:
    """Buyer fraction, spread, efficiency, buyer / non-buyer PnL, sink revenue, residuals."""
    src, led, (v, parts) = cell.source, cell.result.ledger, cell.parts()
    acq = [a for a in cell.acquirers if isinstance(a, InfoAcquirer)]
    decisions, buyers = sum(len(a.history) for a in acq), {a.agent_id for a in acq if a.n_signals}
    mean = lambda ids, k: sum(getattr(parts[i], k) for i in ids) / len(ids) if ids else 0.0
    sink, spread = led[INFO_SINK_ID].cash if INFO_SINK_ID in led else 0, cell.quotes.stats(cell.until).tw_spread
    err = price_efficiency(MidPath(cell.quotes), src.fundamental.value, cell.until).mad
    return {"buyer_fraction": sum(a.n_signals for a in acq) / decisions if decisions else 0.0,
            "n_buyers": len(buyers), "signals": len(src.log),
            "quoted_spread": spread if spread == spread else 0.0,
            "efficiency_error": err if err == err else float(cfg.jump),
            "informed_gain_per_buyer": mean(buyers, "trading_gain"), "buyer_net_pnl":
            mean(buyers, "net"), "non_buyer_net_pnl": mean([i for i in parts if i not in buyers], "net"),
            "sink_revenue": sink, "sink_residual": abs(sink - src.revenue),
            "pnl_residual": sum(abs(parts[i].net - led[i].cash - led[i].position * v + cfg.cash)
                                for i in parts), "violations": len(led.violations())}


def run_gs_grid(cfg: Config = Config(), signal_costs: Sequence[int] = COSTS,
                noise_intensities: Sequence[float] = NOISES, *, replicates: int = 1,
                master_seed: int = 53, n_workers: int = 1) -> ExperimentResult:
    """Sweep signal_cost x noise_intensity through the fl-023 engine (common random numbers)."""
    costs, noises = [int(c) for c in signal_costs], [float(x) for x in noise_intensities]

    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        avg = lambda c, k: float(np.mean([r[k] for r in rows if r[COST] == c]))
        return {"analytic_collapse_cost": analytic_collapse_cost(cfg), **{
            f"{tag}_{k}": avg(c, k) for tag, c in (("cheapest", costs[0]), ("costliest", costs[-1]))
            for k in ("buyer_fraction", "quoted_spread", "efficiency_error", "sink_revenue")}}

    return run_experiment("grossman_stiglitz", partial(build, cfg), partial(gs_metrics, cfg),
                          {COST: costs, NOISE: noises}, replicates=replicates,
                          master_seed=master_seed, crn=True, run_kwargs={"until": cfg.horizon},
                          n_workers=n_workers, findings=headline)


def population(cfg: Config, cost: int, noise_intensity: float, *, generations: int = 8,
               replicates: int = 1, master_seed: int = 53, eta: float = 0.5):
    """fl-024 replicator over (committed buyer, non-buyer) shares at one signal cost."""
    fit = SimulatedFitness(partial(build, cfg), GSCell.fitness, LABELS, n_agents=cfg.n_acquirers,
                           params={COST: int(cost), NOISE: float(noise_intensity)}, master_seed=master_seed,
                           replicates=replicates, run_kwargs={"until": cfg.horizon})
    det = StabilityDetector(epsilon=1e-6, window=3, floor=1.0 / cfg.n_acquirers)
    return evolve([0.5, 0.5], fit, eta=eta, generations=generations, normalize=True,
                  labels=LABELS, detector=det)


@dataclass(frozen=True)
class InformationCollapse:
    """Smallest signal cost at which the buyer fraction falls below `floor`, with the bound."""
    threshold: Optional[int]
    analytic: int
    floor: float
    probes: Dict[int, float] = field(repr=False)


def find_information_collapse(cfg: Config = Config(), noise_intensity: float = 4.0,
                              floor: float = 0.05, *, upper: Optional[int] = None,
                              replicates: int = 1, master_seed: int = 53,
                              n_workers: int = 1) -> InformationCollapse:
    """Integer bisection on [0, upper]; every probe reuses the same master seed."""
    if not 0 < floor <= 1:
        raise ValueError(f"floor must be in (0, 1], got {floor}")
    upper, probes = int(upper if upper is not None else expected_edge(cfg.accuracy_bps, cfg.jump, 0) + 1), {}

    def frac(c: int) -> float:
        if c not in probes:
            probes[c] = run_gs_grid(cfg, [c], [noise_intensity], replicates=replicates, master_seed=master_seed,
                                    n_workers=n_workers).column("buyer_fraction")[0]
        return probes[c]

    lo, hi = 0, (0 if frac(0) < floor else upper if frac(upper) < floor else None)
    if hi:                                      # invariant: frac(lo) >= floor > frac(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if frac(mid) < floor else (mid, hi)
    return InformationCollapse(hi, analytic_collapse_cost(cfg), float(floor), dict(sorted(probes.items())))


def run_grossman_stiglitz(cfg: Config = Config(), signal_costs: Sequence[int] = COSTS,
                          noise_intensities: Sequence[float] = NOISES, *,
                          population_costs: Sequence[int] = (0, 4, 9), generations: int = 8,
                          collapse_floor: float = 0.05, replicates: int = 1,
                          master_seed: int = 53, n_workers: int = 1) -> ExperimentResult:
    """Grid + replicator at three costs (findings["population"]) + bisected collapse."""
    res = run_gs_grid(cfg, signal_costs, noise_intensities, replicates=replicates,
                      master_seed=master_seed, n_workers=n_workers)
    lam, pop = float(list(noise_intensities)[len(noise_intensities) // 2]), []
    for c in population_costs:
        eq = population(cfg, c, lam, generations=generations, master_seed=master_seed)
        pop.append({"signal_cost": int(c), "reason": eq.reason, "generation": eq.generation,
                    "buyer_share": float((eq.shares[-1] if eq.fixed_point is None else eq.fixed_point)[0]),
                    "analytic_share": gs_equilibrium_informed_share(
                        c, cfg.accuracy_bps, cfg.jump, cfg.noise_buy_prob)})
    col = find_information_collapse(cfg, lam, collapse_floor, replicates=replicates,
                                    master_seed=master_seed, n_workers=n_workers)
    res.findings.update({"population": pop, "collapse_noise_intensity": lam,
                         "collapse_threshold": col.threshold, "collapse_analytic": col.analytic})
    return res
