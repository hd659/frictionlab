"""Constant-product AMM venue (x * y = k) settled through the ledger.

Units: base reserve `x` in lots, quote reserve `y` in tick-lots (the same
units as `Account.position` / `Account.cash`). The pool *is* a ledger
`Account`: its position is `x` and its cash is `y`, so there is exactly one
source of truth and the ledger's conservation invariants cover the pool.

Swap math (exact integers, LP fee `f = fee_bps / 10_000` taken on input):
    sell base dx : dy_out = floor(y * dx * (1-f) / (x + dx * (1-f)))
    buy base  dy : dx_out = floor(x * dy * (1-f) / (y + dy * (1-f)))
The full input is added to reserves, so the LP fee accrues to the pool, and
flooring the output rounds in the pool's favour: k never decreases.

Settlement: a swap is booked as a `Trade` at price `p = quote // base`
(trader is the aggressor/taker, the pool the maker) plus an internal cash
transfer of the remainder `quote - p * base`, so cash moves exactly. Extra
per-trade friction (gas, ticket fee) comes from an optional `FrictionStack`
charged on the trader's leg only and is routed to the ledger's venue sink.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from ..core.ledger import Ledger, LedgerError, Settlement
from ..core.types import IdGen, Side, Trade, _check_int
from ..costs.base import fills_of
from ..costs.stack import FrictionStack

BPS_DENOM = 10_000


class SwapRejected(LedgerError):
    """The swap is invalid (drain, zero output, slippage); nothing changed."""


@dataclass(frozen=True)
class SwapResult:
    trade_id: int
    side: Side          # trader's side in base: BUY = quote in, base out
    amount_in: int
    amount_out: int
    lp_fee: int         # LP fee retained by the pool, in input units (ceil)
    friction_fee: int   # per-trade friction paid to the venue sink
    settlement: Settlement

    @property
    def avg_price(self) -> Fraction:
        """Average execution price in ticks per lot."""
        base = self.amount_out if self.side is Side.BUY else self.amount_in
        quote = self.amount_in if self.side is Side.BUY else self.amount_out
        return Fraction(quote, base)


class ConstantProductAMM:
    def __init__(self, ledger: Ledger, pool_id: int, reserve_base: int, reserve_quote: int,
                 fee_bps: int = 30, frictions: Optional[FrictionStack] = None,
                 ids: Optional[IdGen] = None, min_reserve: int = 1) -> None:
        _check_int("reserve_base", reserve_base, positive=True)
        _check_int("reserve_quote", reserve_quote, positive=True)
        self.fee_bps = _check_int("fee_bps", fee_bps, nonneg=True)
        if self.fee_bps >= BPS_DENOM:
            raise ValueError("fee_bps must be < 10000")
        self.min_reserve = _check_int("min_reserve", min_reserve, positive=True)
        self.ledger = ledger
        self.pool_id = pool_id
        self.account = ledger.open_account(pool_id, cash=reserve_quote, position=reserve_base)
        self.frictions = frictions if frictions is not None else FrictionStack()
        self.ids = ids if ids is not None else IdGen(start=10**9)

    # ---- state ---------------------------------------------------------
    @property
    def reserves(self) -> tuple:
        return self.account.position, self.account.cash

    @property
    def k(self) -> int:
        x, y = self.reserves
        return x * y

    @property
    def spot_fraction(self) -> Fraction:
        x, y = self.reserves
        return Fraction(y, x)

    @property
    def spot_price(self) -> int:
        """Mid price y / x rounded to the nearest integer tick."""
        return round(self.spot_fraction)

    def marginal_price(self, side: Side) -> int:
        """Fee-inclusive price of an infinitesimal trade, in ticks.

        Buying base costs spot / (1-f) (ceil); selling base earns spot * (1-f) (floor).
        """
        keep = Fraction(BPS_DENOM - self.fee_bps, BPS_DENOM)
        if Side(side) is Side.BUY:
            p = self.spot_fraction / keep
            return -((-p.numerator) // p.denominator)
        return int(self.spot_fraction * keep)

    # ---- quoting -------------------------------------------------------
    def _io_reserves(self, side: Side) -> tuple:
        x, y = self.reserves
        return (y, x) if Side(side) is Side.BUY else (x, y)  # (reserve_in, reserve_out)

    def quote_exact_in(self, side: Side, amount_in: int) -> int:
        """Output for `amount_in` (quote if side=BUY, base if side=SELL)."""
        amount_in = _check_int("amount_in", amount_in, positive=True)
        r_in, r_out = self._io_reserves(side)
        eff = amount_in * (BPS_DENOM - self.fee_bps)
        out = r_out * eff // (r_in * BPS_DENOM + eff)
        if out <= 0:
            raise SwapRejected("swap output rounds to zero")
        if r_out - out < self.min_reserve:
            raise SwapRejected(f"swap would drain reserve below {self.min_reserve}")
        return out

    def quote_exact_out(self, side: Side, amount_out: int) -> int:
        """Minimum input needed to receive `amount_out` (ceil, pool-favourable)."""
        amount_out = _check_int("amount_out", amount_out, positive=True)
        r_in, r_out = self._io_reserves(side)
        if r_out - amount_out < self.min_reserve:
            raise SwapRejected(f"swap would drain reserve below {self.min_reserve}")
        num = r_in * amount_out * BPS_DENOM
        den = (r_out - amount_out) * (BPS_DENOM - self.fee_bps)
        return -(-num // den)

    def price_impact(self, side: Side, amount_in: int) -> Fraction:
        """Relative shortfall of the average price vs spot (>= 0 is worse)."""
        out = self.quote_exact_in(side, amount_in)
        spot = self.spot_fraction
        if Side(side) is Side.BUY:
            return Fraction(amount_in, out) / spot - 1
        return 1 - Fraction(out, amount_in) / spot

    # ---- execution -----------------------------------------------------
    def swap_exact_in(self, agent_id: int, side: Side, amount_in: int, min_out: int = 0,
                      time: float = 0.0) -> SwapResult:
        side = Side(side)
        out = self.quote_exact_in(side, amount_in)
        if out < min_out:
            raise SwapRejected(f"slippage: out {out} < min_out {min_out}")
        lp_fee = -(-amount_in * self.fee_bps // BPS_DENOM)
        return settle_swap(self, agent_id, side, amount_in, out, lp_fee, time)


def settle_swap(pool, agent_id: int, side: Side, amount_in: int, out: int, lp_fee: int,
                time: float = 0.0) -> SwapResult:
    """Book a priced swap between `agent_id` and `pool.pool_id` on `pool.ledger`.

    Shared by every AMM venue (`pool` needs `ledger`, `pool_id`, `frictions`,
    `ids`). All checks precede the first ledger mutation, so a raise leaves
    the ledger untouched and the caller may commit its own state afterwards.
    """
    base, quote = (out, amount_in) if side is Side.BUY else (amount_in, out)
    price, rem = divmod(quote, base)
    if price < 1:
        raise SwapRejected("average price below one tick")
    lg = pool.ledger
    trader = lg[agent_id]
    buyer, seller = (agent_id, pool.pool_id) if side is Side.BUY else (pool.pool_id, agent_id)
    tid = pool.ids()
    trade = Trade(tid, time, price, base, tid, tid, buyer, seller, aggressor=side)
    fee = pool.frictions.charge(fills_of(trade)[0 if side is Side.BUY else 1])
    if not lg.allow_overdraft:
        need = quote + fee if side is Side.BUY else fee - quote
        if trader.cash < need:
            raise LedgerError(f"account {agent_id} cannot fund swap")
    if not lg.allow_short and side is Side.SELL and trader.position < base:
        raise LedgerError(f"account {agent_id} lacks inventory for swap")
    rec = lg.settle(trade, fees=(fee, 0) if side is Side.BUY else (0, fee))
    if rem:
        lg.transfer(buyer, seller, rem)
    return SwapResult(tid, side, amount_in, out, lp_fee, fee, rec)
