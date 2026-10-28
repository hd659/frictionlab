"""Ledger: settle trades net of friction and enforce conservation invariants.

Accounting model (all integers, tick-lots for cash, lots for inventory)
- Agents hold `Account`s. External flows (`open_account`, `deposit`) are the
  only way cash or inventory enters the system; they are tracked as totals.
- Every friction charge is routed to an explicit venue sink account, so cash
  is conserved *including* fees: a rebate (negative fee) is paid by the venue.
- Invariants (checked by `check`, optionally after every settlement):
    sum(agent cash) + venue cash == cash deposited
    sum(agent position)          == inventory deposited
    venue cash == sum(agent fees_paid) == sum(journal fees)
    venue position == 0

Settlement is atomic: all validation happens before any account is mutated,
so a rejected trade leaves the ledger untouched. Each trade id settles once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Protocol, Tuple

from .types import Account, Side, Trade, _check_int


class FeeSource(Protocol):
    def trade_fees(self, trade: Trade) -> Tuple[int, int]: ...


class LedgerError(ValueError):
    """A settlement was rejected; the ledger is unchanged."""


class InvariantViolation(AssertionError):
    """A conservation invariant does not hold (indicates a bug)."""


@dataclass(frozen=True)
class Settlement:
    trade_id: int
    time: float
    buyer_id: int
    seller_id: int
    price: int
    qty: int
    buyer_fee: int
    seller_fee: int

    @property
    def total_fee(self) -> int:
        return self.buyer_fee + self.seller_fee


class Ledger:
    def __init__(self, frictions: Optional[FeeSource] = None, venue_id: int = 0, *,
                 allow_short: bool = True, allow_overdraft: bool = True,
                 check_every: bool = False) -> None:
        if frictions is not None and not hasattr(frictions, "trade_fees"):
            from ..costs.stack import FrictionStack  # lazy: costs depends on core
            frictions = FrictionStack([frictions])  # bare CostModel -> 1-knob stack
        self.frictions = frictions
        self.venue = Account(_check_int("venue_id", venue_id))
        self.allow_short = allow_short
        self.allow_overdraft = allow_overdraft
        self.check_every = check_every
        self.accounts: Dict[int, Account] = {}
        self.journal: List[Settlement] = []
        self._settled: set = set()
        self.cash_in = 0
        self.inventory_in = 0
        self.transferred = 0

    # ---- accounts / external flows -----------------------------------
    def open_account(self, agent_id: int, cash: int = 0, position: int = 0) -> Account:
        agent_id = _check_int("agent_id", agent_id)
        if agent_id == self.venue.agent_id:
            raise LedgerError(f"agent_id {agent_id} is reserved for the venue")
        if agent_id in self.accounts:
            raise LedgerError(f"account {agent_id} already exists")
        self.accounts[agent_id] = acct = Account(agent_id)
        try:
            self.deposit(agent_id, cash, position)
        except (LedgerError, TypeError, ValueError):
            del self.accounts[agent_id]  # atomic: no half-opened account
            raise
        return acct

    def deposit(self, agent_id: int, cash: int = 0, position: int = 0) -> None:
        """External transfer in (negative = withdrawal)."""
        acct = self[agent_id]
        cash, position = _check_int("cash", cash), _check_int("position", position)
        if not self.allow_overdraft and acct.cash + cash < 0:
            raise LedgerError(f"withdrawal overdraws account {agent_id}")
        if not self.allow_short and acct.position + position < 0:
            raise LedgerError(f"withdrawal shorts account {agent_id}")
        acct.cash += cash
        acct.position += position
        self.cash_in += cash
        self.inventory_in += position

    def transfer(self, src: int, dst: int, amount: int) -> None:
        """Internal cash move between two agent accounts (e.g. a cost sink).

        Not an external flow: `cash_in` is untouched, so conservation holds.
        Atomic: validation precedes mutation.
        """
        amount = _check_int("amount", amount, nonneg=True)
        a, b = self[src], self[dst]
        if a is b:
            raise LedgerError(f"transfer from account {src} to itself")
        if not self.allow_overdraft and a.cash < amount:
            raise LedgerError(f"transfer overdraws account {src}")
        a.cash -= amount
        b.cash += amount
        self.transferred += amount

    def __getitem__(self, agent_id: int) -> Account:
        try:
            return self.accounts[agent_id]
        except KeyError:
            raise LedgerError(f"unknown account {agent_id}") from None

    def __contains__(self, agent_id: int) -> bool:
        return agent_id in self.accounts

    def __iter__(self) -> Iterator[Account]:
        return iter(self.accounts.values())

    def __len__(self) -> int:
        return len(self.accounts)

    # ---- settlement --------------------------------------------------
    def fees_for(self, trade: Trade) -> Tuple[int, int]:
        if self.frictions is None:
            return 0, 0
        bf, sf = self.frictions.trade_fees(trade)
        return _check_int("buyer_fee", bf), _check_int("seller_fee", sf)

    def settle(self, trade: Trade, fees: Optional[Tuple[int, int]] = None) -> Settlement:
        if trade.trade_id in self._settled:
            raise LedgerError(f"trade {trade.trade_id} already settled")
        buyer, seller = self[trade.buyer_id], self[trade.seller_id]
        bf, sf = self.fees_for(trade) if fees is None else (
            _check_int("buyer_fee", fees[0]), _check_int("seller_fee", fees[1]))
        n = trade.notional
        if not self.allow_overdraft:
            if buyer.cash - n - bf < 0:
                raise LedgerError(f"buyer {buyer.agent_id} cannot fund trade {trade.trade_id}")
            if seller.cash + n - sf < 0:
                raise LedgerError(f"seller {seller.agent_id} cannot pay fee on {trade.trade_id}")
        if not self.allow_short and seller.position < trade.qty:
            raise LedgerError(f"seller {seller.agent_id} lacks inventory for {trade.trade_id}")

        buyer.apply_fill(Side.BUY, trade.price, trade.qty, bf)
        seller.apply_fill(Side.SELL, trade.price, trade.qty, sf)
        self.venue.cash += bf + sf
        rec = Settlement(trade.trade_id, trade.time, trade.buyer_id, trade.seller_id,
                         trade.price, trade.qty, bf, sf)
        self.journal.append(rec)
        self._settled.add(trade.trade_id)
        if self.check_every:
            self.check()
        return rec

    def settle_all(self, trades: Iterable[Trade]) -> List[Settlement]:
        return [self.settle(t) for t in trades]

    # ---- invariants & reporting -------------------------------------
    @property
    def fees_collected(self) -> int:
        return self.venue.cash

    def violations(self) -> List[str]:
        out = []
        cash = sum(a.cash for a in self) + self.venue.cash
        if cash != self.cash_in:
            out.append(f"cash not conserved: {cash} != deposited {self.cash_in}")
        inv = sum(a.position for a in self)
        if inv != self.inventory_in:
            out.append(f"inventory not conserved: {inv} != deposited {self.inventory_in}")
        paid = sum(a.fees_paid for a in self)
        booked = sum(s.total_fee for s in self.journal)
        if not (self.venue.cash == paid == booked):
            out.append(f"fee sink mismatch: venue={self.venue.cash} paid={paid} journal={booked}")
        if self.venue.position != 0:
            out.append(f"venue holds inventory: {self.venue.position}")
        if not self.allow_short:
            out += [f"account {a.agent_id} short {a.position}" for a in self if a.position < 0]
        if not self.allow_overdraft:
            out += [f"account {a.agent_id} overdrawn {a.cash}" for a in self if a.cash < 0]
        return out

    def check(self) -> None:
        bad = self.violations()
        if bad:
            raise InvariantViolation("; ".join(bad))

    def equity(self, mark: int) -> Dict[int, int]:
        return {a.agent_id: a.equity(mark) for a in self}

    def total_equity(self, mark: int) -> int:
        """Agents' total equity; equals deposits at `mark` minus fees collected."""
        return sum(self.equity(mark).values())
