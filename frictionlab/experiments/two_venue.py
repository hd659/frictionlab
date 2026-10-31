"""Two continuous-book venues sharing one agent population (fee-tier-wars market).

Each venue is a fl-031 `RouteVenue` (own book, ledger and fee sink) priced by
`LotFee(maker, taker)`: integer ticks per lot, `maker < 0` is a rebate paid
out of the venue sink. `maker + taker >= 0`, so a sink never goes negative.
Maker quotes are post-only (clamped one tick inside the rival maker's touch).

One step: V moves by +-jump with prob `jump_prob` (makers see last step's
value, informed takers see the current one). Every maker cancels and requotes
`quote_size` lots at `center -+ half_spread` on its venue, center skewed by
inventory, so more makers on a venue means a tighter touch. Then 0..max
takers arrive (ZI-C, noise or informed), are routed by `VenueRouter` to the
minimum all-in venue (exact ties by a pre-drawn coin) and trade only if the
all-in price beats their value (noise always trades). All random draws are
made whatever the fees are, so probes at different fees share one path.

Every `epoch` steps each maker re-picks a venue by expected rebate net of
measured adverse-selection loss per unit, `-maker_fee - as_loss`, where the
loss is the fl-030 adverse component `-sum s (V_t - ref) q` over all maker
fills on that venue so far (shrunk by `as_prior` lots), moving only for a gain
of at least `switch_cost` ticks.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from ..agents.base import Submit
from ..analysis.pnl import PnLBreakdown, total
from ..core.ledger import Ledger
from ..core.order_book import OrderBook
from ..core.types import IdGen, Order, Side
from ..costs import CostModel, FrictionStack, Role
from ..markets import RouteVenue, VenueRouter
from ..markets.router import RoutePlan

VENUES = ("A", "B")
MM0, ZIC0, NOISE0, INF0 = 1, 100, 200, 300
Fees = Tuple[int, int]                     # (maker, taker) ticks per lot


class LotFee(CostModel):
    """Per-lot maker/taker fee in integer ticks; a negative maker fee is a rebate."""

    name = "maker_taker"

    def __init__(self, maker: int, taker: int) -> None:
        super().__init__()
        self.maker, self.taker = int(maker), int(taker)
        if self.taker < 0 or self.maker + self.taker < 0:
            raise ValueError(f"need taker >= 0 and maker + taker >= 0, got {(maker, taker)}")

    def raw_cost(self, fill) -> Fraction:
        return Fraction((self.maker if fill.role is Role.MAKER else self.taker) * fill.qty)


@dataclass(frozen=True)
class MarketConfig:
    steps: int = 200
    epoch: int = 25
    n_makers: int = 4
    quote_size: int = 4
    half_spread: int = 2
    skew: int = 6                          # 1 tick of center skew per `skew` lots held
    switch_cost: int = 1
    as_prior: int = 20
    v0: int = 1000
    jump_prob: float = 0.15
    jump: int = 6
    max_arrivals: int = 3
    mix: Tuple[float, float] = (0.55, 0.85)   # cumulative: ZI-C < 0.55 <= noise < 0.85 <= informed
    zic_range: int = 6
    max_qty: int = 2
    n_each: int = 4
    cash: int = 10**8


def fees_of(params: Mapping[str, Any]) -> Dict[str, Fees]:
    return {v: (int(params[f"maker_{v}"]), int(params[f"taker_{v}"])) for v in VENUES}


class TwoVenueMarket:
    def __init__(self, cfg: MarketConfig, fees: Mapping[str, Fees], seed: int) -> None:
        self.cfg, self.fees = cfg, {v: tuple(fees[v]) for v in VENUES}
        self.rng = np.random.default_rng(seed)
        self.makers = list(range(MM0, MM0 + cfg.n_makers))
        ids = self.makers + [b + i for b in (ZIC0, NOISE0, INF0) for i in range(cfg.n_each)]
        self.venues: Dict[str, RouteVenue] = {}
        for v in VENUES:
            led = Ledger(FrictionStack([LotFee(*self.fees[v])]))
            for aid in ids:
                led.open_account(aid, cfg.cash)
            self.venues[v] = RouteVenue(v, OrderBook(trade_ids=IdGen()), led)
        self.router = VenueRouter(list(self.venues.values()))
        self.loc = {k: VENUES[i % 2] for i, k in enumerate(self.makers)}
        self.orders: Dict[int, List[int]] = {k: [] for k in self.makers}
        self.as_loss, self.as_vol = dict.fromkeys(VENUES, 0), dict.fromkeys(VENUES, 0)
        self.spread_sum, self.spread_n = dict.fromkeys(VENUES, 0), dict.fromkeys(VENUES, 0)
        self.maker_epochs, self.n_epochs = dict.fromkeys(VENUES, 0), 0
        self.gross: List[PnLBreakdown] = []

    def score(self, v: str) -> Fraction:
        return -self.fees[v][0] - Fraction(self.as_loss[v], self.as_vol[v] + self.cfg.as_prior)

    def choose_venues(self) -> None:
        s = {v: self.score(v) for v in VENUES}
        for k in self.makers:
            other = VENUES[1 - VENUES.index(self.loc[k])]
            if s[other] - s[self.loc[k]] >= self.cfg.switch_cost:
                self.loc[k] = other
        self.n_epochs += 1
        for k in self.makers:
            self.maker_epochs[self.loc[k]] += 1

    def quote(self, v_pub: int, t: float) -> None:
        c = self.cfg
        for k in self.makers:
            for v in VENUES:
                for oid in self.orders[k]:
                    self.venues[v].book.cancel(oid)
            self.orders[k], ven = [], self.venues[self.loc[k]]
            inv = sum(self.venues[v].ledger[k].position for v in VENUES)
            mid = v_pub - inv // c.skew
            for side, px in ((Side.BUY, mid - c.half_spread), (Side.SELL, mid + c.half_spread)):
                opp = ven.book.best_ask if side is Side.BUY else ven.book.best_bid
                if opp is not None:                     # post-only: never cross a rival maker
                    px = min(px, opp - 1) if side is Side.BUY else max(px, opp + 1)
                oid = ven.order_ids()
                ven.book.submit(Order(oid, k, side, c.quote_size, px, t))
                self.orders[k].append(oid)
        for v in VENUES:
            if self.venues[v].book.spread is not None:
                self.spread_sum[v] += self.venues[v].book.spread
                self.spread_n[v] += 1

    def taker(self, v_true: int, v_pub: int, t: float) -> None:
        c, r = self.cfg, self.rng
        u, side_u, qty = r.random(), r.random(), int(r.integers(1, c.max_qty + 1))
        val_zic, acct, coin = v_pub + int(r.integers(-c.zic_range, c.zic_range + 1)), int(r.integers(c.n_each)), r.random()
        if u < c.mix[0]:
            aid, value, side, strict = ZIC0 + acct, val_zic, Side.BUY if side_u < 0.5 else Side.SELL, False
        elif u < c.mix[1]:
            aid, value, side, strict = NOISE0 + acct, v_true, Side.BUY if side_u < 0.5 else Side.SELL, None
        else:
            if v_true == v_pub:
                return
            aid, value, strict = INF0 + acct, v_true, True
            side = Side.BUY if v_true > v_pub else Side.SELL
        plan = self.router.plan(aid, Submit(side, qty))
        if not plan.legs:
            return
        full = [l for l in plan.quotes.values() if l is not None and l.qty == qty]
        best = [l for l in full if l.all_in == plan.all_in] if not plan.split else []
        if len(best) > 1:
            plan = RoutePlan(aid, side, qty, (best[int(coin * len(best))],), plan.quotes)
        gain = side.value * value * plan.qty - plan.all_in      # value minus all-in, signed
        if strict is not None and (gain < 0 or (strict and gain == 0)):
            return
        for v, trades in self.router.execute(plan, t).items():
            for tr in trades:
                m = tr.seller_id if tr.buyer_id == aid else tr.buyer_id
                s = 1 if tr.buyer_id == m else -1
                self.as_loss[v] -= s * (v_true - v_pub) * tr.qty
                self.as_vol[v] += tr.qty
                self.gross += [PnLBreakdown(-s * (value - tr.price) * tr.qty, volume=tr.qty),
                               PnLBreakdown(s * (v_true - tr.price) * tr.qty, volume=tr.qty)]

    def run(self) -> Dict[str, Any]:
        c, r = self.cfg, self.rng
        v_pub = v_true = c.v0
        for step in range(c.steps):
            if step % c.epoch == 0:
                self.choose_venues()
            v_pub = v_true
            if r.random() < c.jump_prob:
                v_true += c.jump if r.random() < 0.5 else -c.jump
            self.quote(v_pub, float(step))
            for _ in range(int(r.integers(0, c.max_arrivals + 1))):
                self.taker(v_true, v_pub, float(step))
        return self.metrics()

    def metrics(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        vol = {v: sum(s.qty for s in self.venues[v].ledger.journal) for v in VENUES}
        tot, friction, resid, viol = sum(vol.values()), 0, 0, 0
        for v in VENUES:
            led, (m, t) = self.venues[v].ledger, self.fees[v]
            fees = [f for s in led.journal for f in (s.buyer_fee, s.seller_fee)]
            reb, gross = -sum(f for f in fees if f < 0), sum(f for f in fees if f > 0)
            out |= {f"volume_{v}": vol[v], f"share_{v}": vol[v] / tot if tot else 0.5,
                    f"revenue_{v}": led.venue.cash, f"rebates_{v}": reb, f"gross_fees_{v}": gross,
                    f"spread_{v}": self.spread_sum[v] / self.spread_n[v] if self.spread_n[v] else 0.0,
                    f"makers_{v}": self.maker_epochs[v] / max(self.n_epochs, 1),
                    f"as_per_unit_{v}": self.as_loss[v] / self.as_vol[v] if self.as_vol[v] else 0.0}
            friction += led.venue.cash
            resid += abs(led.venue.cash - (m + t) * vol[v]) + abs(reb - max(-m, 0) * vol[v])
            viol += len(led.violations())
        g = total(self.gross)
        return out | {"volume": tot, "friction": friction, "friction_per_lot": friction / tot if tot else 0.0,
                      "surplus": g.trading, "net_surplus": g.trading - friction,
                      "fee_residual": resid, "violations": viol}


def build(cfg: MarketConfig, params: Dict[str, Any], seed: int) -> TwoVenueMarket:
    return TwoVenueMarket(cfg, fees_of(params), seed)


def simulate(cfg: MarketConfig, fees: Mapping[str, Fees], seed: int) -> Dict[str, Any]:
    return TwoVenueMarket(cfg, fees, seed).run()
