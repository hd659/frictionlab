"""Cross-venue arbitrageur: limit order book <-> AMM pool(s), net of every friction.

For an integer size q (lots of base) and pool spot S = y / x (exact Fraction):

  BUY book / SELL amm : lift q lots off the asks (cost C), swap q base into the
                        pool for O quote.          gross = q*S - C, implicit = q*S - O
  SELL book / BUY amm : hit q lots on the bids (proceeds R), swap the least quote
                        A that returns >= q base.  gross = R - q*S, implicit = A - q*S

The AMM implicit cost splits into the LP fee (fee_bps of the input, valued at
S) and price impact (the rest, >= 0 for a constant-product pool). Then

  friction = book fees (ledger stack: taker fee + half-spread + fixed, one charge
             per touched level) + LP fee + price impact + gas (pool stack)
             + decision_cost
  net      = gross - friction  == exact cash PnL of the pair, in tick-lots

The agent scans q = 1..max_size (bounded by visible depth, pool reserves and,
for a constant-product pool, the analytic optimum `cp_optimal_size` at the
touch: beyond it the pool's marginal price is worse than the best book
price, so a larger q can only lose), keeps the q with the largest net and acts only if net > 0, i.e. gross edge >
total friction (ties -> no trade). Fees per level are exact when each level is
one resting order. The opportunity is emitted as one `ArbPair` intent; the
`CrossVenue` in `core.runner` fills the book leg IOC, sizes the AMM leg to the
actual fill and settles both legs through the ledger.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..analysis.pnl import PnLBreakdown
from ..core.types import Side, _check_int
from ..costs.base import Fill, Role
from ..markets.amm_cp import BPS_DENOM, ConstantProductAMM, SwapRejected
from .base import Action, Agent, Cancel, MarketView, Submit
from .informed import _as_stack


@dataclass(frozen=True)
class Swap:
    """AMM intent: swap `amount_in` (quote if side=BUY, base if SELL) on `pool_id`."""
    pool_id: int
    side: Side
    amount_in: int
    min_out: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", Side(self.side))
        _check_int("amount_in", self.amount_in, positive=True)
        _check_int("min_out", self.min_out, nonneg=True)


@dataclass(frozen=True)
class ArbPair:
    """Paired intent: a marketable book leg plus the offsetting AMM leg."""
    book: Submit
    amm: Swap

    def __post_init__(self) -> None:
        if self.book.price is None:
            raise ValueError("book leg must be a marketable limit order")
        if self.book.side is self.amm.side:
            raise ValueError("book and AMM legs must be opposite sides in base")


def _out(pool, side: Side, amount_in: int) -> int:
    try:
        return pool.quote_exact_in(side, amount_in)
    except SwapRejected as e:
        if "zero" in str(e):
            return 0
        raise


def min_quote_in(pool, base_out: int) -> int:
    """Least quote input that buys >= `base_out` lots (raises SwapRejected)."""
    exact = getattr(pool, "quote_exact_out", None)
    if exact is not None:
        a = exact(Side.BUY, base_out)
        if _out(pool, Side.BUY, a) >= base_out:
            return a
    lo, hi = 0, max(1, -(-base_out * pool.spot_fraction.numerator // pool.spot_fraction.denominator))
    while _out(pool, Side.BUY, hi) < base_out:
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if _out(pool, Side.BUY, mid) >= base_out else (mid, hi)
    return hi


def cp_optimal_size(x: int, y: int, fee_bps: int, price: int, book_side: Side) -> float:
    """Continuous optimum vs. a flat book at `price` (no book fees), x*y=k pool.

    book BUY  / sell dq into pool: y*x*k' / (x + q*k')^2 = P -> q* = (sqrt(x*y*k'/P) - x) / k'
    book SELL / buy  q out of pool: x*y / ((x - q)^2 * k') = P -> q* = x - sqrt(x*y / (P*k'))
    with k' = 1 - fee. Returns 0.0 when no size is profitable.
    """
    keep = (BPS_DENOM - fee_bps) / BPS_DENOM
    if Side(book_side) is Side.BUY:
        q = (math.sqrt(x * y * keep / price) - x) / keep
    else:
        q = x - math.sqrt(x * y / (price * keep))
    return max(0.0, q)


def amm_leg(pool, book_side: Side, base: int) -> Tuple[Swap, int]:
    """AMM leg offsetting `base` lots traded on the book; returns (intent, out)."""
    if Side(book_side) is Side.BUY:  # bought on the book -> sell base into the pool
        out = pool.quote_exact_in(Side.SELL, base)
        return Swap(pool.pool_id, Side.SELL, base, out), out
    a = min_quote_in(pool, base)
    return Swap(pool.pool_id, Side.BUY, a, base), pool.quote_exact_in(Side.BUY, a)


@dataclass(frozen=True)
class ArbQuote:
    pool_id: int
    book_side: Side
    qty: int
    limit: int              # worst book level touched (marketable limit price)
    book_notional: int      # C (book buy) or R (book sell), tick-lots
    amm_in: int
    amm_out: int
    spot: Fraction
    book_fees: int
    gas: int
    lp_fee: Fraction
    decision_cost: int
    breakdown: Dict[str, Fraction] = field(default_factory=dict, compare=False)

    @property
    def implicit(self) -> Fraction:
        """AMM cost vs. trading q lots at spot: LP fee + price impact."""
        if self.book_side is Side.BUY:
            return self.qty * self.spot - self.amm_out
        return self.amm_in - self.qty * self.spot

    @property
    def price_impact(self) -> Fraction:
        return self.implicit - self.lp_fee

    @property
    def gross_edge(self) -> Fraction:
        return self.book_side.value * (self.qty * self.spot - self.book_notional)

    @property
    def friction(self) -> Fraction:
        return self.book_fees + self.gas + self.implicit + self.decision_cost

    @property
    def net_edge(self) -> Fraction:
        return self.gross_edge - self.friction

    @property
    def profitable(self) -> bool:
        return self.net_edge > 0

    @property
    def pnl(self) -> PnLBreakdown:
        """Planned PnL of the pair; `pnl.net == net_edge` exactly."""
        trading = (self.amm_out - self.book_notional if self.book_side is Side.BUY
                   else self.book_notional - self.amm_in)
        return PnLBreakdown(trading, self.book_fees + self.gas, self.decision_cost, self.qty)

    def intent(self) -> ArbPair:
        min_out = self.amm_out if self.book_side is Side.BUY else self.qty
        return ArbPair(Submit(self.book_side, self.qty, self.limit),
                       Swap(self.pool_id, self.book_side.opposite, self.amm_in, min_out))


def _walk(levels, q: int) -> Optional[List[Tuple[int, int]]]:
    fills, left = [], q
    for price, size in levels:
        take = min(size, left)
        fills.append((price, take))
        left -= take
        if not left:
            return fills
    return None


class CrossVenueArbitrageur(Agent):
    def __init__(self, agent_id: int, *, frictions=None, pools: Sequence = (),
                 pool_ids: Optional[Sequence[int]] = None, max_size: int = 100,
                 wake_rate: float = 1.0, decision_cost: int = 0) -> None:
        super().__init__(agent_id, decision_cost)
        self._own_frictions = frictions is not None  # None -> adopt the venue ledger's stack
        self.frictions = _as_stack(frictions)
        self.pools: Dict[int, object] = {p.pool_id: p for p in pools}
        self.pool_ids = None if pool_ids is None else frozenset(pool_ids)
        self.max_size = _check_int("max_size", max_size, positive=True)
        if not float(wake_rate) > 0:
            raise ValueError(f"wake_rate must be > 0, got {wake_rate}")
        self.wake_rate = float(wake_rate)
        self.history: List[Optional[ArbQuote]] = []
        self.sent: List[ArbQuote] = []

    def step(self, venue, time: float):
        live = getattr(venue, "pools", None)
        if live:  # observe the venue's pools exactly like the book
            self.pools = dict(live)
        lf = getattr(getattr(venue, "ledger", None), "frictions", None)
        if not self._own_frictions and lf is not None:
            self.frictions = _as_stack(lf)
        return super().step(venue, time)

    def quote_size(self, view: MarketView, pool, book_side: Side, q: int) -> Optional[ArbQuote]:
        """Exact economics of a size-q pair, or None if depth/reserves cannot fill it."""
        book_side = Side(book_side)
        fills = _walk(view.asks if book_side is Side.BUY else view.bids, q)
        if fills is None:
            return None
        parts: Dict[str, Fraction] = {}
        book_fees = 0
        for price, n in fills:
            f = Fill(self.agent_id, book_side, Role.TAKER, price, n)
            book_fees += self.frictions.charge(f)
            for k, v in self.frictions.breakdown(f).items():
                parts["book_" + k] = parts.get("book_" + k, Fraction(0)) + v
        try:
            swap, out = amm_leg(pool, book_side, q)
        except SwapRejected:
            return None
        base, quote = (q, out) if book_side is Side.BUY else (out, swap.amount_in)
        if quote < base:
            return None  # average price below one tick: the pool would reject
        gas = pool.frictions.charge(Fill(self.agent_id, swap.side, Role.TAKER, quote // base, base))
        spot = pool.spot_fraction
        lp = Fraction(swap.amount_in * pool.fee_bps, BPS_DENOM) * (spot if book_side is Side.BUY else 1)
        qt = ArbQuote(pool.pool_id, book_side, q, fills[-1][0], sum(p * n for p, n in fills),
                      swap.amount_in, out, spot, book_fees, gas, lp, self.decision_cost, parts)
        parts.update(lp_fee=lp, price_impact=qt.price_impact, amm_gas=Fraction(gas),
                     decision_cost=Fraction(self.decision_cost))
        return qt

    def best_opportunity(self, view: MarketView,
                         pools: Optional[Mapping[int, object]] = None) -> Optional[ArbQuote]:
        best: Optional[ArbQuote] = None
        for pid, pool in sorted((self.pools if pools is None else pools).items()):
            if self.pool_ids is not None and pid not in self.pool_ids:
                continue
            for side in (Side.BUY, Side.SELL):
                levels = view.asks if side is Side.BUY else view.bids
                if not levels or side.value * (pool.spot_fraction - levels[0][0]) <= 0:
                    continue  # touch on the wrong side of spot: gross <= 0 for every q
                cap = min(self.max_size, sum(n for _, n in levels))
                if isinstance(pool, ConstantProductAMM):
                    x, y = pool.reserves
                    cap = min(cap, int(cp_optimal_size(x, y, pool.fee_bps, levels[0][0], side)))
                for q in range(1, cap + 1):
                    qt = self.quote_size(view, pool, side, q)
                    if qt is None:
                        break
                    if qt.profitable and (best is None or qt.net_edge > best.net_edge):
                        best = qt
        return best

    def decide(self, view: MarketView) -> List[Action]:
        actions: List[Action] = [Cancel(oid) for oid in view.open_orders]
        opp = self.best_opportunity(view)
        self.history.append(opp)
        if opp is not None:
            actions.append(opp.intent())
            self.sent.append(opp)
        return actions

    def __repr__(self) -> str:
        return (f"CrossVenueArbitrageur(id={self.agent_id}, pools={sorted(self.pools)}, "
                f"max_size={self.max_size}, decision_cost={self.decision_cost})")
