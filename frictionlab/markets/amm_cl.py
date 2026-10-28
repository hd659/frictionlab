"""Concentrated-liquidity AMM (Uniswap-v3 style) settled through the ledger.

Prices are integer ticks (quote tick-lots per base lot), so a position's range
`[tick_lower, tick_upper)` is a price interval and liquidity `L` is active iff
`tick_lower <= tick < tick_upper`. State is exact integers: `sqrt_price` is
floor(sqrt(P) * 2**96) and per range the virtual reserves follow
    base  = L * (1/sqrtP - 1/sqrtP_upper),   quote = L * (sqrtP - sqrtP_lower)
which inside one range is exactly x * y = L**2. A swap walks across initialized
tick boundaries, adding/removing `liquidity_net` as it crosses, and charges the
LP fee per step on the input consumed by that step (the curve moves by the exact
fee-net input; its integer reserve share is rounded up). Every rounding favours
the pool, so tracked reserves always cover what positions are owed. Unlike the
constant-product pool, fees accrue outside the curve (as in v3), not into k.

The pool is a ledger `Account` holding `reserve + accrued LP fees`; swaps share
`settle_swap` (and the quote/swap interface) with `ConstantProductAMM`.
"""
from __future__ import annotations

from bisect import bisect_right, insort
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Dict, List, Optional, Tuple

from ..core.ledger import Ledger
from ..core.types import IdGen, Side, _check_int
from ..costs.stack import FrictionStack
from .amm_cp import BPS_DENOM, SwapRejected, SwapResult, settle_swap

Q96 = 1 << 96
MIN_TICK, MAX_TICK = 0, 1 << 64  # [MIN_TICK, MAX_TICK) is "full range"


def sqrt_at(price) -> int:
    """floor(sqrt(price) * 2**96) for an int tick or Fraction price."""
    p = Fraction(price)
    return isqrt(p.numerator * (1 << 192) // p.denominator)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def base_delta(L: int, sa: int, sb: int, up: bool = False) -> int:
    num, den = L * Q96 * (sb - sa), sa * sb
    return _cdiv(num, den) if up else num // den


def quote_delta(L: int, sa: int, sb: int, up: bool = False) -> int:
    num = L * (sb - sa)
    return _cdiv(num, Q96) if up else num // Q96


@dataclass(frozen=True)
class Position:
    tick_lower: int
    tick_upper: int
    liquidity: int


@dataclass(frozen=True)
class SwapPlan:
    side: Side
    amount_in: int        # gross input actually consumed (incl. LP fee)
    amount_out: int
    lp_fee: int
    net_in: int           # input that moved the curve
    sqrt_price: int
    tick: int
    liquidity: int
    crossed: Tuple[int, ...]
    partial: bool


class ConcentratedLiquidityAMM:
    def __init__(self, ledger: Ledger, pool_id: int, price, fee_bps: int = 30,
                 frictions: Optional[FrictionStack] = None, ids: Optional[IdGen] = None,
                 partial_fill: bool = True) -> None:
        p = Fraction(price)
        if not MIN_TICK < p < MAX_TICK:
            raise ValueError(f"price must be in ({MIN_TICK}, {MAX_TICK})")
        self.fee_bps = _check_int("fee_bps", fee_bps, nonneg=True)
        if self.fee_bps >= BPS_DENOM:
            raise ValueError("fee_bps must be < 10000")
        self.ledger, self.pool_id, self.partial_fill = ledger, pool_id, partial_fill
        self.account = ledger.open_account(pool_id)
        self.frictions = frictions if frictions is not None else FrictionStack()
        self.ids = ids if ids is not None else IdGen(start=10**9)
        self.sqrt_price, self.tick, self.liquidity = sqrt_at(p), p.numerator // p.denominator, 0
        self.liquidity_net: Dict[int, int] = {}
        self._ticks: List[int] = []
        self.positions: List[Position] = []
        self.reserve_base = self.reserve_quote = self.fees_base = self.fees_quote = 0

    @classmethod
    def full_range(cls, ledger: Ledger, pool_id: int, reserve_base: int, reserve_quote: int,
                   **kw) -> "ConcentratedLiquidityAMM":
        """Single [MIN_TICK, MAX_TICK) position: behaves like x * y = k."""
        _check_int("reserve_base", reserve_base, positive=True)
        _check_int("reserve_quote", reserve_quote, positive=True)
        amm = cls(ledger, pool_id, Fraction(reserve_quote, reserve_base), **kw)
        amm.add_position(MIN_TICK, MAX_TICK, isqrt(reserve_base * reserve_quote))
        return amm

    # ---- liquidity -------------------------------------------------------
    def amounts_for(self, lo: int, hi: int, L: int, up: bool = True) -> Tuple[int, int]:
        """(base, quote) backing liquidity L on [lo, hi) at the current price."""
        sl, su, s = sqrt_at(lo), sqrt_at(hi), self.sqrt_price
        if s <= sl:
            return base_delta(L, sl, su, up), 0
        if s >= su:
            return 0, quote_delta(L, sl, su, up)
        return base_delta(L, s, su, up), quote_delta(L, sl, s, up)

    def add_position(self, tick_lower: int, tick_upper: int, liquidity: int) -> Tuple[int, int]:
        """Deposit the (ceil) amounts for `liquidity` on [tick_lower, tick_upper)."""
        lo, hi = _check_int("tick_lower", tick_lower), _check_int("tick_upper", tick_upper)
        L = _check_int("liquidity", liquidity, positive=True)
        if not MIN_TICK <= lo < hi <= MAX_TICK:
            raise ValueError("need MIN_TICK <= tick_lower < tick_upper <= MAX_TICK")
        b, q = self.amounts_for(lo, hi, L)
        self.ledger.deposit(self.pool_id, cash=q, position=b)
        for t, d in ((lo, L), (hi, -L)):
            if t not in self.liquidity_net:
                insort(self._ticks, t)
            self.liquidity_net[t] = self.liquidity_net.get(t, 0) + d
        if lo <= self.tick < hi:
            self.liquidity += L
        self.reserve_base += b
        self.reserve_quote += q
        self.positions.append(Position(lo, hi, L))
        return b, q

    # ---- state -----------------------------------------------------------
    @property
    def reserves(self) -> tuple:
        return self.account.position, self.account.cash

    @property
    def spot_fraction(self) -> Fraction:
        return Fraction(self.sqrt_price ** 2, 1 << 192)

    @property
    def spot_price(self) -> int:
        return round(self.spot_fraction)

    def marginal_price(self, side: Side) -> int:
        keep = Fraction(BPS_DENOM - self.fee_bps, BPS_DENOM)
        if Side(side) is Side.BUY:
            p = self.spot_fraction / keep
            return _cdiv(p.numerator, p.denominator)
        return int(self.spot_fraction * keep)

    def violations(self) -> List[str]:
        """Pool account == reserves + fees, and reserves cover all positions."""
        out = []
        if self.reserves != (self.reserve_base + self.fees_base, self.reserve_quote + self.fees_quote):
            out.append("pool account != reserves + accrued fees")
        owed = [self.amounts_for(p.tick_lower, p.tick_upper, p.liquidity, up=False)
                for p in self.positions]
        if sum(b for b, _ in owed) > self.reserve_base or sum(q for _, q in owed) > self.reserve_quote:
            out.append("reserves do not cover positions")
        return out

    # ---- quoting ---------------------------------------------------------
    def simulate(self, side: Side, amount_in: int) -> SwapPlan:
        """Pure dry run of an exact-in swap; raises SwapRejected if unfillable."""
        side, rem = Side(side), _check_int("amount_in", amount_in, positive=True)
        up, keep = side is Side.BUY, BPS_DENOM - self.fee_bps
        s, tick, L = self.sqrt_price, self.tick, self.liquidity
        out = fee = net_in = 0
        crossed: List[int] = []
        while rem > 0:
            i = bisect_right(self._ticks, tick)
            nxt = (self._ticks[i] if i < len(self._ticks) else None) if up else \
                (self._ticks[i - 1] if i > 0 else None)
            st = sqrt_at(nxt) if nxt is not None else 0
            if nxt is None or (st == 0 and L == 0):
                break  # no liquidity left in this direction
            availx = rem * keep  # fee-net input, scaled by BPS_DENOM (exact)
            if L == 0:
                need = 0
            elif st == 0:  # price 0 is unreachable: never cross it
                need = availx + 1
            else:
                need = quote_delta(L, s, st, True) if up else base_delta(L, st, s, True)
            if availx >= need * BPS_DENOM:  # reach the boundary and cross it
                gross = _cdiv(need * BPS_DENOM, keep)
                out += 0 if L == 0 else base_delta(L, s, st) if up else quote_delta(L, st, s)
                fee, net_in, rem, s = fee + gross - need, net_in + need, rem - gross, st
                L, tick = (L + self.liquidity_net[nxt], nxt) if up else \
                    (L - self.liquidity_net[nxt], nxt - 1)
                crossed.append(nxt)
                continue
            if up:
                sn = min(st, s + availx * Q96 // (L * BPS_DENOM))  # never skip a tick
                out += base_delta(L, s, sn)
                tick = min(nxt - 1, max(tick, sn * sn >> 192))
            else:
                sn = max(st, _cdiv(L * Q96 * s * BPS_DENOM, L * Q96 * BPS_DENOM + availx * s))
                out += quote_delta(L, sn, s)
                tick = max(nxt, min(tick, sn * sn >> 192))
            avail = _cdiv(availx, BPS_DENOM)  # reserve share rounds up, fee share down
            fee, net_in, rem, s = fee + rem - avail, net_in + avail, 0, sn
        if out <= 0:
            raise SwapRejected("swap output rounds to zero or no liquidity")
        if rem and not self.partial_fill:
            raise SwapRejected(f"insufficient liquidity: {rem} of {amount_in} unfilled")
        return SwapPlan(side, amount_in - rem, out, fee, net_in, s, tick, L, tuple(crossed), rem > 0)

    def quote_exact_in(self, side: Side, amount_in: int) -> int:
        return self.simulate(side, amount_in).amount_out

    def price_impact(self, side: Side, amount_in: int) -> Fraction:
        plan, spot = self.simulate(side, amount_in), self.spot_fraction
        if plan.side is Side.BUY:
            return Fraction(plan.amount_in, plan.amount_out) / spot - 1
        return 1 - Fraction(plan.amount_out, plan.amount_in) / spot

    # ---- execution -------------------------------------------------------
    def swap_exact_in(self, agent_id: int, side: Side, amount_in: int, min_out: int = 0,
                      time: float = 0.0) -> SwapResult:
        plan = self.simulate(side, amount_in)
        if plan.amount_out < min_out:
            raise SwapRejected(f"slippage: out {plan.amount_out} < min_out {min_out}")
        res = settle_swap(self, agent_id, plan.side, plan.amount_in, plan.amount_out,
                          plan.lp_fee, time)
        if plan.side is Side.BUY:
            self.reserve_quote += plan.net_in
            self.fees_quote += plan.lp_fee
            self.reserve_base -= plan.amount_out
        else:
            self.reserve_base += plan.net_in
            self.fees_base += plan.lp_fee
            self.reserve_quote -= plan.amount_out
        self.sqrt_price, self.tick, self.liquidity = plan.sqrt_price, plan.tick, plan.liquidity
        return res
