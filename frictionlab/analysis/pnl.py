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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, Mapping, Optional, Union

Mark = Union[int, Mapping[int, int], Callable[[object], int]]


@dataclass(frozen=True)
class PnLBreakdown:
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
