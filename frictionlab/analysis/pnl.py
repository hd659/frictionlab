"""Per-agent PnL split into trading, fee and decision-cost components.

For agent a with fills (side_j, price_j, qty_j) and a valuation mark m_a:

    trading_a  = sum_j side_j * (m_a - price_j) * qty_j   (gross surplus)
    fees_a     = friction-stack charges settled by the ledger
    decision_a = deliberation charges routed to the decision-cost sink
    net_a      = trading_a - fees_a - decision_a

The mark can be one common price (mark-to-market, so `net` equals
`spread.pnl_by_agent`), a per-agent mapping, or a callable on the agent; the
callable form values each fill at a private valuation (Gode-Sunder surplus,
or a `BoundedRationalAgent.valuation`). Summed over all agents at a common
mark, trading PnL is zero and the net totals equal minus the fees plus
decision costs collected: trading between agents only moves surplus around.

Venue-specific splits, each exact in ticks against the ledger:
- `middleman_pnl` (fl-049): spread capture - holding cost - decay loss = cash change
  plus the basis of units still held.
- `info_pnl` (fl-052): trading gain - signal cost - decision cost = equity change.
- `lemons_pnl` (fl-055): trade surplus (price - q) - certification cost.
- `reputation_pnl` (fl-058): trade surplus + cheating gain (q - delivered) - whitewash
  cost = cash minus the production cost of what was delivered.
- `switching_pnl` (fl-061): seller acquisition-phase margin + harvest-phase margin
  = cash - c * units sold; `consumer_surplus` gives v - price minus switching cost paid.
- `platform_pnl` (fl-064): fee revenue - subsidy outlay (negative fees plus seeded
  adoption paid) = platform cash change; `member_surplus` splits each member into
  standalone value + cross-side gain - net fees - adoption cost paid.
- `queue_surplus` (fl-067): customer reward (if served) - toll - waiting cost = net
  surplus; `operator_pnl` gives the queue operator's toll revenue (= toll sink cash).
- `menu_pnl` (fl-070): seller frictionless profit - misalignment loss - menu cost = cash.
- `holdup_pnl` / `holdup_buyer_surplus` (fl-073): revenue - investment - contracting; value - price.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, Mapping, Optional, Union

Mark = Union[int, Mapping[int, int], Callable[[object], int]]


@dataclass(frozen=True)
class PnLBreakdown:
    """Agent or group PnL split into trading, fees and decision costs (`net` is the total)."""
    trading: int = 0          # gross PnL of fills at the mark
    fees: int = 0             # friction charges paid (rebates negative)
    decision_costs: int = 0   # deliberation charges paid
    volume: int = 0           # lots traded
    n_agents: int = 1

    @property
    def net(self) -> int:
        return self.trading - self.fees - self.decision_costs

    @property
    def friction(self) -> int:
        return self.fees + self.decision_costs

    def __add__(self, other: "PnLBreakdown") -> "PnLBreakdown":
        return PnLBreakdown(self.trading + other.trading, self.fees + other.fees,
                            self.decision_costs + other.decision_costs,
                            self.volume + other.volume, self.n_agents + other.n_agents)

    def as_dict(self) -> Dict[str, int]:
        return {"trading": self.trading, "fees": self.fees,
                "decision_costs": self.decision_costs, "net": self.net,
                "volume": self.volume, "n_agents": self.n_agents}


ZERO = PnLBreakdown(n_agents=0)


def _resolve(mark: Mark, agent) -> int:
    if callable(mark):
        return int(mark(agent))
    if isinstance(mark, Mapping):
        return int(mark[agent.agent_id])
    return int(mark)


def pnl_breakdown(result, mark: Mark,
                  agents: Optional[Iterable[int]] = None) -> Dict[int, PnLBreakdown]:
    """Breakdown per agent id (default: every agent in `result.agents`)."""
    ids = list(result.agents) if agents is None else list(agents)
    unknown = [a for a in ids if a not in result.agents]
    if unknown:
        raise KeyError(f"unknown agent ids {unknown}")
    marks = {aid: _resolve(mark, result.agents[aid]) for aid in ids}
    trading = dict.fromkeys(ids, 0)
    volume = dict.fromkeys(ids, 0)
    for t in result.trades:
        for aid, side in ((t.buyer_id, 1), (t.seller_id, -1)):
            if aid in trading:
                trading[aid] += side * (marks[aid] - t.price) * t.qty
                volume[aid] += t.qty
    return {aid: PnLBreakdown(trading[aid], result.ledger[aid].fees_paid,
                              result.agents[aid].decision_costs_paid, volume[aid])
            for aid in ids}


def total(parts: Iterable[PnLBreakdown]) -> PnLBreakdown:
    """Sum of `PnLBreakdown`s; the empty sum is a zero breakdown."""
    out = ZERO
    for p in parts:
        out = out + p
    return out


def breakdown_by(result, mark: Mark, key: Callable[[object], Hashable] = None,
                 agents: Optional[Iterable[int]] = None) -> Dict[Hashable, PnLBreakdown]:
    """Sum `pnl_breakdown` over groups; default group is the agent class name."""
    key = key or (lambda a: type(a).__name__)
    out: Dict[Hashable, PnLBreakdown] = {}
    for aid, b in pnl_breakdown(result, mark, agents).items():
        k = key(result.agents[aid])
        out[k] = out.get(k, ZERO) + b
    return out


@dataclass(frozen=True)
class PlatformPnL:
    """Platform PnL in ticks: `net = fee_revenue - subsidy_outlay` (= its cash change)."""
    fee_revenue: int = 0       # positive participation fees collected
    subsidy_outlay: int = 0    # negative-fee payouts plus adoption costs it covered

    @property
    def net(self) -> int:
        return self.fee_revenue - self.subsidy_outlay

    def as_dict(self) -> Dict[str, int]:
        return {"fee_revenue": self.fee_revenue, "subsidy_outlay": self.subsidy_outlay,
                "net": self.net}


@dataclass(frozen=True)
class MemberSurplus:
    """Member surplus: `net = standalone + cross_side - fees - adoption` in ticks."""
    standalone: int = 0        # sum of u_i over rounds as a member
    cross_side: int = 0        # sum of b * n_other(t-1) over rounds as a member
    fees: int = 0              # fees paid minus subsidies received
    adoption: int = 0          # adoption costs paid out of own cash

    @property
    def net(self) -> int:
        return self.standalone + self.cross_side - self.fees - self.adoption

    def as_dict(self) -> Dict[str, int]:
        return {"standalone": self.standalone, "cross_side": self.cross_side,
                "fees": self.fees, "adoption": self.adoption, "net": self.net}


def platform_pnl(market) -> PlatformPnL:
    """Split the platform account of a fl-064 `PlatformMarket` run from its flow log."""
    pid = market.platform_id
    rev = sum(f.amount for f in market.flows if f.kind == "fee")
    out = sum(f.amount for f in market.flows if f.payer == pid and f.kind != "fee")
    return PlatformPnL(rev, out)


def member_surplus(market) -> Dict[int, MemberSurplus]:
    """Per-member `MemberSurplus` of a fl-064 `PlatformMarket` run."""
    fees = dict.fromkeys(market.side_of, 0)
    adopt = dict.fromkeys(market.side_of, 0)
    for f in market.flows:
        if f.kind == "fee":
            fees[f.member] += f.amount
        elif f.kind == "subsidy":
            fees[f.member] -= f.amount
        elif f.payer == f.member:
            adopt[f.member] += f.amount
    return {m: MemberSurplus(u, x, fees[m], adopt[m])
            for m, (u, x) in market.accrued.items()}


@dataclass(frozen=True)
class QueueSurplus:
    """Queue customer surplus: `net = reward - toll - waiting` in ticks."""
    reward: int = 0            # R if served by the horizon, else 0
    toll: int = 0              # admission toll paid
    waiting: int = 0           # waiting cost paid over steps in system
    steps: int = 0             # steps spent in the system

    @property
    def net(self) -> int:
        return self.reward - self.toll - self.waiting

    def as_dict(self) -> Dict[str, int]:
        return {"reward": self.reward, "toll": self.toll, "waiting": self.waiting,
                "steps": self.steps, "net": self.net}


@dataclass(frozen=True)
class OperatorPnL:
    """Queue operator PnL in ticks: `net = toll_revenue` (no service cost modelled)."""
    toll_revenue: int = 0
    joins: int = 0

    @property
    def net(self) -> int:
        return self.toll_revenue

    def as_dict(self) -> Dict[str, int]:
        return {"toll_revenue": self.toll_revenue, "joins": self.joins, "net": self.net}


def queue_surplus(market) -> Dict[int, QueueSurplus]:
    """Per-customer `QueueSurplus` of a fl-067 `QueueMarket` run (joiners only)."""
    cc = market.congestion
    return {cid: QueueSurplus(market.reward if c.served else 0, cc.tolls.get(cid, 0),
                              cc.waits.get(cid, 0), c.steps)
            for cid, c in sorted(market.customers.items())}


def operator_pnl(market) -> OperatorPnL:
    """Toll revenue of a fl-067 `QueueMarket` run."""
    return OperatorPnL(market.congestion.toll_revenue, market.joins)


@dataclass(frozen=True)
class MenuPnL:
    """Menu-cost seller PnL: `net = frictionless - misalignment - menu_cost` in ticks."""
    frictionless: int = 0      # steps * base_profit (profit at the target every step)
    misalignment: int = 0      # sum curvature * (p - p*)^2
    menu_cost: int = 0         # menu cost paid to the menu sink
    adjustments: int = 0

    @property
    def net(self) -> int:
        return self.frictionless - self.misalignment - self.menu_cost

    def as_dict(self) -> Dict[str, int]:
        return {"frictionless": self.frictionless, "misalignment": self.misalignment,
                "menu_cost": self.menu_cost, "adjustments": self.adjustments, "net": self.net}


def menu_pnl(market) -> Dict[int, MenuPnL]:
    """Per-seller `MenuPnL` of a fl-070 `MenuCostMarket` run."""
    out = {}
    for i in market.sellers:
        rows = market.path(i)
        out[i] = MenuPnL(market.base_profit * len(rows), sum(r.loss for r in rows),
                         market.menu.paid.get(i, 0), market.menu.adjustments_of(i))
    return out


from .venue_pnl import (  # noqa: E402  (imported last: venue_pnl needs the names above)
    HoldupBuyerSurplus, HoldupPnL, InfoPnL, LemonsPnL, MiddlemanPnL, ReputationPnL, SwitchingPnL,
    consumer_surplus, holdup_buyer_surplus, holdup_pnl, info_pnl, lemons_pnl, middleman_pnl,
    reputation_pnl, switching_pnl,
)
