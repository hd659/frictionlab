"""Menu-cost and price-rigidity experiment (fl-071) on the fl-070 `MenuCostMarket`.

Horizon, curvature k and the shock schedule are fixed; the grid is menu_cost F x
inflation_ticks_per_step d. Every seller is an `SsPricer` at the analytic band for its
cell, opening on its reset price (target + overshoot, as if it had just adjusted), so a
zero-shock run is in its steady-state sawtooth from step 1.

Under deterministic drift the (S, s) gap after a reset is o = overshoot(b, d) and falls
by d per step; the cycle length L(b, d) is the smallest j >= 1 with |o - j * d| > b, the
gaps over one cycle are o, o - d, ..., o - (L - 1) d, and the long-run loss per step is

    V(b) = (k * sum gap^2 + F) / L        (exact `Fraction`)

`optimal_ss_band` returns the smallest minimizer. L is non-decreasing in b, so the
band is non-decreasing in F (and 0 at F = 0). A band with k * (b - |d|)^2 / 12 > F
loses more than band 0 (whose loss is F), which bounds the search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from math import isqrt
from typing import Any, Dict, List, Sequence, Tuple

from ..agents.ss_pricer import SsPricer, overshoot
from ..analysis.pnl import menu_pnl
from ..config.schema import Param
from ..markets import MenuCostMarket
from .base import ExperimentResult, run_experiment

COST, INFL, BPS = "menu_cost", "inflation_ticks_per_step", 10_000
COSTS, INFLS = (0, 1, 2, 4, 8, 16, 32, 64), (1, 2, 4)


@dataclass(frozen=True)
class Config:
    """Fixed market: sellers, opening price, horizon, curvature, base profit, shocks."""
    n_sellers: int = 4
    start_price: int = 1_000
    horizon: int = 240
    curvature: int = 1
    base_profit: int = 200
    shocks: Tuple[int, ...] = (-1, 0, 0, 1)


def _cycle(band: int, drift: int) -> List[int]:
    """Gaps over one deterministic sawtooth cycle (reset step first)."""
    o, g = overshoot(band, drift), [overshoot(band, drift)]
    while abs(o - len(g) * drift) <= band:
        g.append(o - len(g) * drift)
    return g


def per_step_loss(band: int, menu_cost: int, curvature: int, drift_per_step: int) -> Fraction:
    """Exact long-run misalignment + menu cost per step of band `band` under drift d."""
    if drift_per_step == 0:
        return Fraction(0)
    g = _cycle(band, drift_per_step)
    return Fraction(curvature * sum(x * x for x in g) + menu_cost, len(g))


def adjustment_frequency(band: int, drift_per_step: int) -> Fraction:
    """Exact share of steps that adjust under deterministic drift (1 / L, 0 without drift)."""
    return Fraction(0) if drift_per_step == 0 else Fraction(1, len(_cycle(band, drift_per_step)))


def optimal_ss_band(menu_cost: int, curvature: int, drift_per_step: int) -> int:
    """Smallest integer band minimizing `per_step_loss` (0 when F == 0 or d == 0)."""
    if menu_cost < 0 or curvature < 0:
        raise ValueError("menu_cost and curvature must be >= 0")
    if menu_cost == 0 or drift_per_step == 0:
        return 0
    if curvature == 0:
        raise ValueError("with zero curvature and a positive menu cost no band is optimal")
    top = 2 * abs(drift_per_step) + 2 * isqrt(12 * menu_cost // curvature + 1) + 4
    vals = [per_step_loss(b, menu_cost, curvature, drift_per_step) for b in range(top + 1)]
    return vals.index(min(vals))


class SteadyPricer(SsPricer):
    """`SsPricer` that opens on its reset price (free): steady state from step 1."""

    def start(self, target: int) -> int:
        self.price = int(target) + self.overshoot
        return self.price


class MenuCell:
    """One seeded market at (F, d, band) plus its zero-cost flexible twin on the same draws."""

    def __init__(self, market: MenuCostMarket, twin: MenuCostMarket) -> None:
        self.market, self.twin = market, twin

    def run(self) -> "MenuCell":
        self.market.run()
        self.twin.run()
        return self


def market(cfg: Config, menu_cost: int, drift: int, band: int, seed: int) -> MenuCostMarket:
    """Seeded market with every seller on `band`; opening targets staggered by one tick."""
    ids = range(1, cfg.n_sellers + 1)
    return MenuCostMarket([SteadyPricer(i, band, drift=drift) for i in ids],
                          [cfg.start_price + i for i in ids], steps=cfg.horizon,
                          inflation_ticks_per_step=drift, shocks=cfg.shocks, menu=menu_cost,
                          curvature=cfg.curvature, base_profit=cfg.base_profit, seed=seed)


def build(cfg: Config, params: Dict[str, Any], seed: int) -> MenuCell:
    """fl-023 factory: analytic-band market and its flexible zero-cost twin, both on `seed`."""
    f, d = int(params[COST]), int(params[INFL])
    band = int(params.get("band", optimal_ss_band(f, cfg.curvature, d)))
    return MenuCell(market(cfg, f, d, band, seed), market(cfg, 0, d, 0, seed))


def welfare(m: MenuCostMarket) -> int:
    """fl-022 welfare: total seller net PnL (menu costs are a deadweight resource cost)."""
    return sum(p.net for p in menu_pnl(m).values())


def menu_metrics(cell: MenuCell) -> Dict[str, Any]:
    """Frequency, gaps, adjustment size, loss, menu cost, net PnL and welfare change."""
    m, log = cell.market, cell.market.log
    adj = [r for r in log if r.adjusted]
    pnl = menu_pnl(m).values()
    loss, paid = sum(p.misalignment for p in pnl), sum(p.menu_cost for p in pnl)
    net, band = welfare(m), next(iter(m.sellers.values())).band
    return {"band": band, "adjustments": len(adj), "seller_steps": len(log),
            "adjustment_frequency": len(adj) / len(log),
            "analytic_frequency": float(adjustment_frequency(band, m.drift)),
            "mean_abs_gap": sum(abs(r.gap) for r in log) / len(log),
            "mean_adjustment_size": (sum(abs(r.gap - r.gap_before) for r in adj) / len(adj)
                                     if adj else 0.0),
            "misalignment_loss": loss, "menu_cost_paid": paid, "per_step_loss": (loss + paid) / len(log),
            "seller_net_pnl": net, "welfare_change_vs_flexible": net - welfare(cell.twin),
            "violations": len(m.ledger.violations()) + len(cell.twin.ledger.violations())}


def run_menu_grid(cfg: Config = Config(), menu_costs: Sequence[int] = COSTS,
                  inflations: Sequence[int] = INFLS, *, replicates: int = 1,
                  master_seed: int = 71, n_workers: int = 1) -> ExperimentResult:
    """Sweep menu_cost x inflation through the fl-023 engine (CRN)."""
    def headline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"max_band": max(r["band"] for r in rows),
                "min_adjustment_frequency": min(r["adjustment_frequency"] for r in rows),
                "max_mean_abs_gap": max(r["mean_abs_gap"] for r in rows),
                "total_menu_cost_paid": sum(r["menu_cost_paid"] for r in rows),
                "worst_welfare_change": min(r["welfare_change_vs_flexible"] for r in rows)}

    return run_experiment("menu_cost", partial(build, cfg), menu_metrics,
                          {COST: [int(x) for x in menu_costs], INFL: [int(x) for x in inflations]},
                          replicates=replicates, master_seed=master_seed, crn=True,
                          n_workers=n_workers, findings=headline)


@dataclass(frozen=True)
class RigidityThreshold:
    """Smallest menu cost whose simulated adjustment frequency is < rigidity_bps of flexible."""
    menu_cost: int
    analytic: int
    band: int
    analytic_frequency: Fraction
    baseline_adjustments: int
    probes: Dict[int, int] = field(repr=False)     # menu cost -> simulated adjustments


def _rigid(adjustments: int, baseline: int, rigidity_bps: int) -> bool:
    return adjustments * BPS < rigidity_bps * baseline


def find_rigidity_threshold(cfg: Config = Config(), drift: int = 2, *, rigidity_bps: int = 5_000,
                            master_seed: int = 71, max_cost: int = 1 << 20) -> RigidityThreshold:
    """Integer bisection on menu cost; every probe reuses the master seed."""
    if drift == 0 or not 0 < rigidity_bps <= BPS:
        raise ValueError("need a non-zero drift and 0 < rigidity_bps <= 10000")
    probes: Dict[int, int] = {}

    def adjustments(f: int) -> int:
        if f not in probes:
            probes[f] = run_menu_grid(cfg, [f], [drift], master_seed=master_seed).rows[0]["adjustments"]
        return probes[f]

    base = adjustments(0)
    rigid = lambda f: _rigid(adjustments(f), base, rigidity_bps)  # noqa: E731
    lo, hi = 0, 1                                     # F = 0 is the flexible baseline: not rigid
    while not rigid(hi):
        lo, hi = hi, 2 * hi
        if hi > max_cost:
            raise ValueError(f"no menu cost <= {max_cost} makes the market rigid")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if rigid(mid) else (mid, hi)
    target = Fraction(rigidity_bps, BPS) * adjustment_frequency(0, drift)
    ana = next(f for f in range(max_cost + 1) if adjustment_frequency(
        optimal_ss_band(f, cfg.curvature, drift), drift) < target)
    band = optimal_ss_band(hi, cfg.curvature, drift)
    return RigidityThreshold(hi, ana, band, adjustment_frequency(band, drift), base,
                             dict(sorted(probes.items())))


def run_menu_cost(cfg: Config = Config(), menu_costs: Sequence[int] = COSTS,
                  inflations: Sequence[int] = INFLS, *, probe_inflation: int = 2,
                  rigidity_bps: int = 5_000, replicates: int = 1, master_seed: int = 71,
                  n_workers: int = 1) -> ExperimentResult:
    """Grid + analytic bands + bisected rigidity threshold."""
    res = run_menu_grid(cfg, menu_costs, inflations, replicates=replicates,
                        master_seed=master_seed, n_workers=n_workers)
    thr = find_rigidity_threshold(cfg, probe_inflation, rigidity_bps=rigidity_bps, master_seed=master_seed)
    res.findings.update({"probe_inflation": int(probe_inflation), "rigidity_bps": int(rigidity_bps),
                         "bisected_threshold": thr.menu_cost, "analytic_threshold": thr.analytic,
                         "threshold_band": thr.band,
                         "threshold_analytic_frequency": float(thr.analytic_frequency)})
    return res


# ---- fl-036 registry spec (kept here so the registry module stays under the line ceiling)
_POS = Param("int", min=1)
PARAM_SCHEMA: Dict[str, Param] = {
    "menu_costs": Param("int", seq=True, friction=True), "inflations": Param("int", seq=True),
    "n_sellers": _POS, "start_price": _POS, "horizon": _POS, "curvature": _POS,
    "base_profit": Param("int", min=0), "shocks": Param("int", seq=True), "probe_inflation": Param("int"),
    "rigidity_bps": Param("int", min=1, max=BPS)}
SMALL: Dict[str, Any] = {"menu_costs": [0, 2, 8, 32], "inflations": [1, 2, 3], "n_sellers": 2,
                         "horizon": 120, "shocks": [0]}


def defaults() -> Dict[str, Any]:
    """Registered defaults: `Config()` fields plus the grid and threshold-probe settings."""
    base = {f: getattr(Config(), f) for f in Config.__dataclass_fields__}
    return {"menu_costs": list(COSTS), "inflations": list(INFLS), **base,
            "shocks": list(Config().shocks), "probe_inflation": 2, "rigidity_bps": 5_000}


def registry_runner(p: Dict[str, Any], master_seed: int, replicates: int, n_workers: int) -> ExperimentResult:
    """fl-036 runner: resolved params -> `run_menu_cost`."""
    cfg = Config(**{k: tuple(p[k]) if k == "shocks" else p[k] for k in Config.__dataclass_fields__})
    return run_menu_cost(cfg, p["menu_costs"], p["inflations"], probe_inflation=p["probe_inflation"],
                         rigidity_bps=p["rigidity_bps"], replicates=replicates,
                         master_seed=master_seed, n_workers=n_workers)
