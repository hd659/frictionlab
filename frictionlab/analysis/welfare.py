"""Welfare / surplus vs the competitive equilibrium, and account-level PnL.

Valuation agents (ZI-style: a `side` and a unit `schedule` of redemption
values or costs) trade their units in schedule order, so the k-th lot a
buyer acquires is worth schedule[k] and the k-th lot a seller delivers cost
schedule[k]. For the realized allocation

    gross gains     = sum(values of units bought) - sum(costs of units sold)
    surplus_a       = sum_k side_a * (schedule_a[k] - p_k)      (at valuation)
    net surplus     = sum_a (surplus_a - fees_a - decision_a)

Prices are transfers: when every fill is between valuation agents,
sum_a surplus_a == gross gains and net surplus + friction sinks == gross
gains. No allocation beats the competitive equilibrium (CE): gross gains
<= CE max surplus, the sum of value - cost over the q* efficient pairs.

`pnl_by_account` marks every ledger account at a common price using the
canonical `metrics.TradeRecord` tape (book, batch and AMM fills), so pools
are included and the venue fee sink and decision-cost sink close the books:
the values sum to exactly zero. Agent rows are `pnl.PnLBreakdown`s and match
`pnl.pnl_breakdown` on book-only runs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from ..core.types import Side
from .metrics import TradeRecord, trade_records
from .pnl import PnLBreakdown, total

VENUE_SINK, COST_SINK, POOL = "venue_sink", "cost_sink", "pool"


@dataclass(frozen=True)
class Equilibrium:
    quantity: int
    max_surplus: int
    price_lo: int
    price_hi: int


def competitive_equilibrium(values: Sequence[int], costs: Sequence[int]) -> Equilibrium:
    """CE quantity, max surplus and clearing-price interval of unit schedules."""
    d, s = sorted(values, reverse=True), sorted(costs)
    q = sum(1 for v, c in zip(d, s) if v >= c)
    if q == 0:
        return Equilibrium(0, 0, 0, 0)
    lo = max(s[q - 1], d[q] if q < len(d) else 0)
    hi = min(d[q - 1], s[q] if q < len(s) else d[q - 1])
    return Equilibrium(q, sum(d[i] - s[i] for i in range(q)), lo, hi)


def is_valuation_agent(agent) -> bool:
    return hasattr(agent, "schedule") and hasattr(agent, "side")


def _lots(records: Iterable[TradeRecord], agent) -> List[int]:
    """Fill price of each lot the agent traded in its own role, in fill order."""
    aid, side = agent.agent_id, Side(agent.side)
    out: List[int] = []
    for r in records:
        if aid not in (r.buyer_id, r.seller_id):
            continue
        if (aid == r.buyer_id) != (side is Side.BUY):
            raise ValueError(f"agent {aid} traded against its {side.name} role")
        out += [r.price] * r.qty
    if len(out) > len(agent.schedule):
        raise ValueError(f"agent {aid} traded {len(out)} > {len(agent.schedule)} units")
    return out


@dataclass(frozen=True)
class WelfareReport:
    equilibrium: Equilibrium
    gross_gains: int                      # value of the realized allocation
    per_agent: Dict[int, PnLBreakdown] = field(repr=False)

    @property
    def totals(self) -> PnLBreakdown:
        return total(self.per_agent.values())

    @property
    def surplus(self) -> int:             # at valuation, before frictions
        return self.totals.trading

    @property
    def friction(self) -> int:
        return self.totals.friction

    @property
    def net_surplus(self) -> int:
        return self.totals.net

    @property
    def efficiency(self) -> float:
        m = self.equilibrium.max_surplus
        return self.gross_gains / m if m else 0.0

    @property
    def net_efficiency(self) -> float:
        m = self.equilibrium.max_surplus
        return self.net_surplus / m if m else 0.0


def welfare(result, agents: Optional[Iterable] = None) -> WelfareReport:
    """Surplus of the valuation agents (default: all in `result.agents`)."""
    pop = [a for a in (result.agents.values() if agents is None else agents)
           if is_valuation_agent(a)]
    if not pop:
        raise ValueError("no valuation agents (need `side` and `schedule`)")
    recs = trade_records(result)
    gains, per = 0, {}
    for a in pop:
        lots = _lots(recs, a)
        units = a.schedule[:len(lots)]
        sign = Side(a.side).value
        gains += sign * sum(units)
        per[a.agent_id] = PnLBreakdown(sum(sign * (v - p) for v, p in zip(units, lots)),
                                       result.ledger[a.agent_id].fees_paid,
                                       a.decision_costs_paid, len(lots))
    ce = competitive_equilibrium(
        [v for a in pop if Side(a.side) is Side.BUY for v in a.schedule],
        [c for a in pop if Side(a.side) is Side.SELL for c in a.schedule])
    return WelfareReport(ce, gains, per)


# ---- account-level PnL -------------------------------------------------------
def account_breakdown(result, mark: int) -> Dict[int, PnLBreakdown]:
    """PnLBreakdown at `mark` for every agent and pool account (not the sinks)."""
    pools = set(getattr(result, "pools", {}) or {})
    ids = list(result.agents) + sorted(pools)
    trading, volume = dict.fromkeys(ids, 0), dict.fromkeys(ids, 0)
    for r in trade_records(result):
        for aid, sign in ((r.buyer_id, 1), (r.seller_id, -1)):
            if aid in trading:
                trading[aid] += sign * (mark * r.qty - r.notional)
                volume[aid] += r.qty
    return {aid: PnLBreakdown(trading[aid], result.ledger[aid].fees_paid,
                              result.agents[aid].decision_costs_paid if aid in result.agents
                              else 0, volume[aid])
            for aid in ids}


def pnl_by_account(result, mark: int) -> Dict[object, int]:
    """Net PnL of every account plus both friction sinks; sums to zero."""
    out: Dict[object, int] = {aid: b.net for aid, b in account_breakdown(result, mark).items()}
    out[VENUE_SINK] = result.fees
    out[COST_SINK] = result.decision_costs
    return out


def pnl_by_agent_type(result, mark: int) -> Dict[str, int]:
    """`pnl_by_account` grouped by agent class name, pools and sinks."""
    out: Dict[str, int] = defaultdict(int)
    for k, v in pnl_by_account(result, mark).items():
        if k in result.agents:
            out[type(result.agents[k]).__name__] += v
        else:
            out[k if isinstance(k, str) else POOL] += v
    return dict(out)
