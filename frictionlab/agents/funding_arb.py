"""Funding-rate arbitrageur: delta-neutral perp/spot basis trade, net of friction.

Observes the `PerpMarket` mark, index and last funding rate (predicted rate
before the first settlement) and a constant-product spot pool. With funding
rate r (bps per period, > 0 means longs pay shorts) and size q lots:

  r > 0 : SELL q perp + BUY q spot      r < 0 : BUY q perp + SELL q spot
  expected funding E = q * mark * |r| / 10_000 * periods

Round-trip friction at size q (entry legs priced at the current state, exit
legs assumed symmetric):
  perp_fee      2 * taker fees of walking q lots through the perp book (gas excluded)
  perp_spread   2 * |book notional - q * mark|        (crossing the perp spread)
  lp_fee        2 * LP fee of the spot swap (valued in ticks)
  price_impact  2 * (AMM implicit cost - LP fee)
  gas           current gas * 4 legs (one fill per leg)
  decision_cost the agent's own decision cost
The agent opens the q with the largest E - friction only if it is > 0 (ties ->
no trade). While open it unwinds when the horizon (`horizon` funding periods
from entry) expires or the expected remaining funding falls below the exit
friction (perp + pool + 2 * gas + decision cost) it would pay by holding on,
i.e. exit friction - carry > exit friction: the carry has turned adverse. Legs are emitted as one
`ArbPair` (perp leg IOC, spot leg sized to the actual fill) so the net delta,
perp position + spot inventory, stays zero. Each position reports gross
funding, fees, gas and net PnL as an `analysis.pnl.PnLBreakdown`; zero-latency
venues only (fills are read back from `CrossVenue.arbs`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional

from ..analysis.pnl import PnLBreakdown
from ..core.types import Side, _check_int
from ..costs.base import Fill, Role
from ..costs.gas import GasProcess
from ..markets.amm_cp import BPS_DENOM, SwapRejected
from .arbitrageur import ArbPair, Swap, _walk, min_quote_in
from .base import Action, Agent, Cancel, MarketView, Submit
from .informed import _as_stack

LEGS = 4


@dataclass(frozen=True)
class BasisQuote:
    perp_side: Side
    qty: int
    expected_funding: Fraction
    friction: Dict[str, Fraction]
    intent: ArbPair

    @property
    def total_friction(self) -> Fraction:
        return sum(self.friction.values(), Fraction(0))

    @property
    def net_edge(self) -> Fraction:
        return self.expected_funding - self.total_friction


@dataclass
class BasisPosition:
    direction: int            # perp side at entry: -1 short perp / long spot
    qty: int
    open_time: float
    expiry: float
    close_time: Optional[float] = None
    trading: int = 0          # pre-fee cash flow of every leg (spread, LP, impact, basis)
    funding: int = 0          # funding received (negative = paid)
    fees: int = 0             # ledger fees incl. gas
    gas: int = 0
    decision_costs: int = 0
    legs: int = 0

    @property
    def pnl(self) -> PnLBreakdown:
        return PnLBreakdown(self.trading + self.funding, self.fees, self.decision_costs,
                            2 * self.qty)

    @property
    def friction(self) -> Dict[str, int]:
        return {"fees": self.fees - self.gas, "gas": self.gas,
                "decision_cost": self.decision_costs}


class FundingArbitrageur(Agent):
    def __init__(self, agent_id: int, perp, pool_id: int, *, gas: Optional[GasProcess] = None,
                 frictions=None, horizon: int = 3, max_size: int = 50,
                 wake_rate: float = 1.0, decision_cost: int = 0) -> None:
        super().__init__(agent_id, decision_cost)
        self.perp, self.pool_id, self.gas, self.pool = perp, pool_id, gas, None
        self._own = frictions is not None
        self.frictions = _as_stack(frictions)
        self.horizon = _check_int("horizon", horizon, positive=True)
        self.max_size = _check_int("max_size", max_size, positive=True)
        if not float(wake_rate) > 0:
            raise ValueError(f"wake_rate must be > 0, got {wake_rate}")
        self.wake_rate = float(wake_rate)
        self.perp_pos = self.spot_inv = self.n_entries = 0
        self.open: Optional[BasisPosition] = None
        self.positions: List[BasisPosition] = []
        self.history: List[Optional[BasisQuote]] = []
        self._fund_cur = self._arb_cur = 0

    @property
    def net_delta(self) -> int:
        return self.perp_pos + self.spot_inv

    def gas_level(self, t: float) -> int:
        return 0 if self.gas is None else self.gas.level(t)

    def _fee_stack(self):
        return self.frictions.without("gas") if "gas" in self.frictions else self.frictions

    def quote(self, view: MarketView, perp_side: Side, q: int, mark: Fraction,
              rate_abs: Fraction, periods: Fraction, round_trip: bool) -> Optional[BasisQuote]:
        """Exact friction of trading q lots on perp_side plus the offsetting spot leg."""
        perp_side, pool = Side(perp_side), self.pool
        fills = _walk(view.asks if perp_side is Side.BUY else view.bids, q)
        if fills is None or pool is None:
            return None
        stack, t = self._fee_stack(), view.time
        fee = sum(stack.charge(Fill(self.agent_id, perp_side, Role.TAKER, p, n, t)) for p, n in fills)
        notional, spot = sum(p * n for p, n in fills), pool.spot_fraction
        try:
            if perp_side is Side.SELL:   # buy q base on the pool
                a = min_quote_in(pool, q)
                if pool.quote_exact_in(Side.BUY, a) != q:
                    return None  # spot leg cannot hit q exactly: would leave delta
                implicit, lp, swap = a - q * spot, Fraction(a * pool.fee_bps, BPS_DENOM), \
                    Swap(pool.pool_id, Side.BUY, a, q)
            else:                        # sell q base into the pool
                out = pool.quote_exact_in(Side.SELL, q)
                implicit, lp = q * spot - out, Fraction(q * pool.fee_bps, BPS_DENOM) * spot
                swap = Swap(pool.pool_id, Side.SELL, q, out)
        except SwapRejected:
            return None
        m = 2 if round_trip else 1
        fr = {"perp_fee": Fraction(m * fee),
              "perp_spread": m * abs(notional - q * mark),
              "lp_fee": m * lp, "price_impact": m * (implicit - lp),
              "gas": Fraction(self.gas_level(t) * (LEGS if round_trip else 2)),
              "decision_cost": Fraction(self.decision_cost)}
        e = q * mark * rate_abs / BPS_DENOM * periods
        return BasisQuote(perp_side, q, e, fr, ArbPair(Submit(perp_side, q, fills[-1][0]), swap))

    # ---- lifecycle -----------------------------------------------------
    def _ingest_funding(self) -> None:
        for ev in self.perp.history[self._fund_cur:]:
            if self.open is not None:
                self.open.funding += ev.payments.get(self.agent_id, 0)
        self._fund_cur = len(self.perp.history)

    def _ingest_fills(self, venue) -> None:
        arbs = getattr(venue, "arbs", [])
        for a in arbs[self._arb_cur:]:
            if a.agent_id != self.agent_id or not a.qty:
                continue
            side = a.pair.book.side
            self.perp_pos += side.value * a.qty
            if a.swap is not None:
                s = a.swap
                self.spot_inv += s.side.value * (s.amount_out if s.side is Side.BUY else s.amount_in)
            if self.open is None:
                self.open = BasisPosition(side.value, a.qty, a.time,
                                          a.time + self.horizon * self.perp.interval)
                self.n_entries += 1
            pos, legs = self.open, len(a.book_trades) + (a.swap is not None)
            pos.trading += a.pnl.trading
            pos.fees += a.pnl.fees
            pos.decision_costs += a.pnl.decision_costs
            pos.gas += self.gas_level(a.time) * legs
            pos.legs += legs
            if self.perp_pos == 0:
                pos.close_time = a.time
                self.positions.append(pos)
                self.open = None
        self._arb_cur = len(arbs)

    def step(self, venue, time: float):
        pools = getattr(venue, "pools", None) or {}
        self.pool = pools.get(self.pool_id, self.pool)
        lf = getattr(getattr(venue, "ledger", None), "frictions", None)
        if not self._own and lf is not None:
            self.frictions = _as_stack(lf)
        self._ingest_funding()
        trades = super().step(venue, time)
        self._ingest_fills(venue)
        return trades

    def report(self) -> List[BasisPosition]:
        """Closed positions plus the open one, with all funding to date attributed."""
        self._ingest_funding()
        return self.positions + ([self.open] if self.open is not None else [])

    def decide(self, view: MarketView) -> List[Action]:
        actions: List[Action] = [Cancel(oid) for oid in view.open_orders]
        t, perp = view.time, self.perp
        mark, index = perp.mark_price(t), perp.index_price(t)
        rate = perp.history[-1].rate_bps if perp.history else perp.funding_rate_bps(mark, index)[1]
        best: Optional[BasisQuote] = None
        if self.perp_pos:
            q, close = abs(self.perp_pos), Side.BUY if self.perp_pos < 0 else Side.SELL
            left = max(Fraction(0), Fraction(self.open.expiry - t) / Fraction(perp.interval))
            qt = self.quote(view, close, q, mark, abs(rate), left, round_trip=False)
            if qt is not None:  # hold: pay exit friction later minus remaining carry
                carry = qt.expected_funding * (1 if (rate > 0) == (self.perp_pos < 0) else -1)
            if qt is not None and (left == 0 or carry < 0):
                best = qt
        elif rate != 0:
            side = Side.SELL if rate > 0 else Side.BUY
            levels = view.bids if side is Side.SELL else view.asks
            for q in range(1, min(self.max_size, sum(n for _, n in levels)) + 1):
                qt = self.quote(view, side, q, mark, abs(rate), Fraction(self.horizon), True)
                if qt is not None and qt.net_edge > 0 and (best is None or qt.net_edge > best.net_edge):
                    best = qt
        self.history.append(best)
        if best is not None:
            actions.append(best.intent)
        return actions

    def __repr__(self) -> str:
        return (f"FundingArbitrageur(id={self.agent_id}, pool={self.pool_id}, "
                f"horizon={self.horizon}, decision_cost={self.decision_cost})")
