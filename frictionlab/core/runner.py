"""Market runner: schedule agent wake-ups on the clock and drive the venue.

One wake-up of agent `a` at time t:
  1. `a.step(venue, t)` builds the MarketView, charges the decision cost
     (agent -> cost sink via `Ledger.transfer`) and asks `decide()` for intents.
  2. With zero latency the intents execute immediately: the venue routes them
     to the book and settles every fill through the friction stack + ledger.
     With `frictions.delay > 0` the intents are captured and executed by an
     `arrive` event `delay` seconds later, so a slow agent acts on a stale view.
  3. The agent's next wake-up is drawn as Exp(rate) from the clock RNG.

AMM pools are optional: pass `pools=[factory]` where `factory(ledger)` builds
a pool bound to the runner's ledger. The venue is then a `CrossVenue`, which
also executes `Swap` intents and paired `ArbPair` intents (book leg filled
immediate-or-cancel, AMM leg sized to the actual book fill) and records each
pair with its realized `PnLBreakdown`.

Every run is reproducible from `seed`: wake times come from the clock RNG and
agents carry their own seeded streams. The ledger invariants (cash incl. fee
and decision-cost sinks, inventory, pool accounts) are checked at the end of
every `run`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..agents.arbitrageur import ArbPair, Swap, amm_leg
from ..agents.base import COST_SINK_ID, Action, Agent, MarketView, Venue
from ..analysis.pnl import PnLBreakdown, total
from ..markets.amm_cp import SwapRejected, SwapResult
from .clock import SimClock
from .ledger import InvariantViolation, Ledger, LedgerError, Settlement
from .order_book import OrderBook
from .types import IdGen, Order, Trade


class _Deferred:
    """Venue proxy that observes and charges now but defers execution."""

    def __init__(self, venue: Venue) -> None:
        self.venue = venue
        self.captured: List[Action] = []

    def __getattr__(self, name):  # pools, ledger, ... read through to the venue
        return getattr(self.venue, name)

    def view(self, agent: Agent, time: float) -> MarketView:
        return self.venue.view(agent, time)

    def charge(self, agent_id: int, amount: int) -> bool:
        return self.venue.charge(agent_id, amount)

    def execute(self, agent: Agent, action: Action, time: float) -> List[Trade]:
        self.captured.append(action)
        return []


@dataclass(frozen=True)
class ArbFill:
    """One executed (or partially hedged) arbitrage pair."""
    agent_id: int
    time: float
    pair: ArbPair
    book_trades: Tuple[Trade, ...]
    swap: Optional[SwapResult]
    pnl: PnLBreakdown           # realized: cash delta, fees, decision cost

    @property
    def qty(self) -> int:
        return sum(t.qty for t in self.book_trades)

    @property
    def hedged(self) -> bool:
        return self.swap is not None


@dataclass
class CrossVenue(Venue):
    """Order book plus AMM pools, all settled on the same ledger."""
    pools: Dict[int, object] = field(default_factory=dict)
    swaps: List[SwapResult] = field(default_factory=list)
    arbs: List[ArbFill] = field(default_factory=list)

    def _swap(self, agent: Agent, s: Swap, time: float) -> SwapResult:
        pool = self.pools.get(s.pool_id)
        if pool is None:
            raise LedgerError(f"unknown pool {s.pool_id}")
        res = pool.swap_exact_in(agent.agent_id, s.side, s.amount_in, s.min_out, time)
        self.swaps.append(res)
        return res

    def execute(self, agent: Agent, action: Action, time: float) -> List[Trade]:
        if isinstance(action, Swap):
            self._swap(agent, action, time)
            return []
        if not isinstance(action, ArbPair):
            return super().execute(agent, action, time)
        acct, b = self.ledger[agent.agent_id], action.book
        cash0, fees0 = acct.cash, acct.fees_paid
        oid = self.order_ids()
        trades = self.book.submit(Order(oid, agent.agent_id, b.side, b.qty, b.price, float(time)))
        self.ledger.settle_all(trades)
        if oid in self.book:
            self.book.cancel(oid)  # immediate-or-cancel: never rest a hedge leg
        filled = sum(t.qty for t in trades)
        if not filled:
            return trades
        res = None
        try:
            leg = action.amm if filled == b.qty else amm_leg(self.pools[action.amm.pool_id],
                                                            b.side, filled)[0]
            res = self._swap(agent, leg, time)
        except (SwapRejected, LedgerError):
            pass  # unhedged inventory stays with the agent (leg risk)
        fees = acct.fees_paid - fees0
        pnl = PnLBreakdown(acct.cash - cash0 + fees, fees, agent.decision_cost, filled)
        self.arbs.append(ArbFill(agent.agent_id, float(time), action, tuple(trades), res, pnl))
        return trades


@dataclass
class RunResult:
    trades: List[Trade]
    ledger: Ledger
    book: OrderBook
    venue: Venue
    agents: Dict[int, Agent]
    wakeups: int
    events: int
    time: float

    @property
    def settlements(self) -> List[Settlement]:
        return self.ledger.journal

    @property
    def fees(self) -> int:
        return self.ledger.fees_collected

    @property
    def decision_costs(self) -> int:
        return self.venue.decision_costs_collected

    @property
    def pools(self) -> Dict[int, object]:
        return dict(getattr(self.venue, "pools", {}))

    @property
    def arbs(self) -> List[ArbFill]:
        return list(getattr(self.venue, "arbs", []))

    def arb_pnl(self, agent_id: Optional[int] = None) -> PnLBreakdown:
        """Sum of realized per-pair PnL (optionally for one agent)."""
        return total(a.pnl for a in self.arbs if agent_id in (None, a.agent_id))

    def fill_prices(self, agent_id: int) -> List[int]:
        """Prices of every lot the agent traded, in fill order."""
        return [t.price for t in self.trades for _ in range(t.qty)
                if agent_id in (t.buyer_id, t.seller_id)]

    def fees_paid(self, agent_id: int) -> int:
        return self.ledger[agent_id].fees_paid


class MarketRunner:
    def __init__(self, agents: Sequence[Agent], frictions=None, *, seed: int = 0,
                 wake_rate: float = 1.0, cash: int = 0, position: int = 0,
                 endowments: Optional[Mapping[int, Tuple[int, int]]] = None,
                 venue_id: int = 0, cost_sink_id: int = COST_SINK_ID,
                 allow_short: bool = True, allow_overdraft: bool = True,
                 check_every: bool = False, depth: int = 5,
                 pools: Sequence[Callable[[Ledger], object]] = ()) -> None:
        if wake_rate <= 0:
            raise ValueError(f"wake_rate must be > 0, got {wake_rate}")
        self.agents: Dict[int, Agent] = {}
        for a in agents:
            if a.agent_id in self.agents:
                raise ValueError(f"duplicate agent_id {a.agent_id}")
            if a.agent_id in (venue_id, cost_sink_id):
                raise ValueError(f"agent_id {a.agent_id} is reserved")
            self.agents[a.agent_id] = a
        if not self.agents:
            raise ValueError("runner needs at least one agent")
        self.clock = SimClock(seed)
        self.book = OrderBook(trade_ids=IdGen())
        self.ledger = Ledger(frictions, venue_id, allow_short=allow_short,
                             allow_overdraft=allow_overdraft, check_every=check_every)
        if pools:
            self.venue: Venue = CrossVenue(self.book, self.ledger, IdGen(), cost_sink_id, depth)
        else:
            self.venue = Venue(self.book, self.ledger, IdGen(), cost_sink_id, depth)
        endow = dict(endowments or {})
        for aid in self.agents:
            c, p = endow.get(aid, (cash, position))
            self.ledger.open_account(aid, c, p)
        for make in pools:
            pool = make(self.ledger)
            if pool.pool_id in self.agents or pool.pool_id in self.venue.pools:
                raise ValueError(f"pool_id {pool.pool_id} collides with an agent or pool")
            self.venue.pools[pool.pool_id] = pool
        self.delay = float(getattr(self.ledger.frictions, "delay", 0.0) or 0.0)
        self.rates = {aid: float(getattr(a, "wake_rate", wake_rate))
                      for aid, a in self.agents.items()}
        self.wakeups = 0
        self._budget: Optional[int] = None
        self._stop_when: Optional[Callable[["MarketRunner"], bool]] = None
        self.clock.on("wake", self._on_wake)
        self.clock.on("arrive", self._on_arrive)
        for aid in self.agents:  # insertion order -> deterministic draw order
            self.clock.schedule_poisson(self.rates[aid], "wake", aid)

    # ---- event handlers ----------------------------------------------
    def _on_wake(self, clock: SimClock, ev) -> None:
        agent = self.agents[ev.payload]
        self.wakeups += 1
        if self.delay > 0:
            proxy = _Deferred(self.venue)
            agent.step(proxy, clock.now)
            if proxy.captured:
                clock.schedule(self.delay, "arrive", (agent.agent_id, proxy.captured))
        else:
            agent.step(self.venue, clock.now)
        clock.schedule_poisson(self.rates[agent.agent_id], "wake", agent.agent_id)
        if self._budget is not None and self.wakeups >= self._budget:
            clock.stop()
        elif self._stop_when is not None and self._stop_when(self):
            clock.stop()

    def _on_arrive(self, clock: SimClock, ev) -> None:
        aid, actions = ev.payload
        self.agents[aid].act(self.venue, actions, clock.now)

    # ---- driver --------------------------------------------------------
    def check(self) -> None:
        self.ledger.check()
        bad = [f"pool {pid}: {v}" for pid, p in getattr(self.venue, "pools", {}).items()
               for v in (p.violations() if hasattr(p, "violations") else [])]
        if bad:
            raise InvariantViolation("; ".join(bad))

    def run(self, until: Optional[float] = None, max_wakeups: Optional[int] = None,
            stop_when: Optional[Callable[["MarketRunner"], bool]] = None) -> RunResult:
        """Advance to `until` (sim seconds) and/or `max_wakeups` more wake-ups.

        Wake-ups recur forever, so at least one bound is required. Calling
        `run` again resumes from where the previous call stopped.
        """
        if until is None and max_wakeups is None:
            raise ValueError("run needs `until` or `max_wakeups`")
        self._budget = None if max_wakeups is None else self.wakeups + int(max_wakeups)
        self._stop_when = stop_when
        events = self.clock.run(until=until)
        self.check()
        return RunResult(list(self.book.trades), self.ledger, self.book, self.venue,
                         dict(self.agents), self.wakeups, events, self.clock.now)
