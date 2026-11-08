"""Venue-specific PnL splits (fl-049 .. fl-061), each exact in ticks against the ledger.

Split out of `analysis.pnl` (which re-exports every name here) to keep both
modules under the hygiene line ceiling. See the `analysis.pnl` docstring for
the identity each split satisfies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from .pnl import Mark, pnl_breakdown

@dataclass(frozen=True)
class MiddlemanPnL:
    """Middleman PnL in ticks: `net = spread_capture - holding_cost - decay_loss`."""
    spread_capture: int = 0   # sum(ask) - sum(basis) over units sold
    holding_cost: int = 0     # carry charges routed to the hold sink
    decay_loss: int = 0       # spoilage routed to the decay sink
    bought: int = 0
    sold: int = 0
    held_basis: int = 0       # cost basis of units still in inventory

    @property
    def net(self) -> int:
        return self.spread_capture - self.holding_cost - self.decay_loss

    def as_dict(self) -> Dict[str, int]:
        return {"spread_capture": self.spread_capture, "holding_cost": self.holding_cost,
                "decay_loss": self.decay_loss, "net": self.net, "bought": self.bought,
                "sold": self.sold, "held_basis": self.held_basis}


@dataclass(frozen=True)
class InfoPnL:
    """Costly-information PnL in ticks: `net = trading_gain - signal_cost - decision_cost`."""
    trading_gain: int = 0     # gross fills at the mark minus execution fees
    signal_cost: int = 0      # ticks paid for signals
    decision_cost: int = 0    # deliberation charges
    signals: int = 0          # signals bought
    volume: int = 0

    @property
    def net(self) -> int:
        return self.trading_gain - self.signal_cost - self.decision_cost

    def as_dict(self) -> Dict[str, int]:
        return {"trading_gain": self.trading_gain, "signal_cost": self.signal_cost,
                "decision_cost": self.decision_cost, "net": self.net,
                "signals": self.signals, "volume": self.volume}


def info_pnl(result, source, mark: Mark,
             agents: Optional[Iterable[int]] = None) -> Dict[int, InfoPnL]:
    """Per-agent `InfoPnL` of a run against one `SignalSource` (default: every agent)."""
    out: Dict[int, InfoPnL] = {}
    for aid, b in pnl_breakdown(result, mark, agents).items():
        out[aid] = InfoPnL(b.trading - b.fees, source.paid.get(aid, 0), b.decision_costs,
                           source.purchases.get(aid, 0), b.volume)
    return out


@dataclass(frozen=True)
class LemonsPnL:
    """Lemons seller PnL in ticks: `net = trade_surplus - certification_cost`."""
    trade_surplus: int = 0        # price - quality over units sold
    certification_cost: int = 0   # ticks paid to the certification sink
    sold: int = 0
    certified: int = 0

    @property
    def net(self) -> int:
        return self.trade_surplus - self.certification_cost

    def as_dict(self) -> Dict[str, int]:
        return {"trade_surplus": self.trade_surplus,
                "certification_cost": self.certification_cost, "net": self.net,
                "sold": self.sold, "certified": self.certified}


def lemons_pnl(market) -> Dict[int, LemonsPnL]:
    """Per-seller `LemonsPnL` of a fl-055 `LemonsMarket` run."""
    out = {sid: LemonsPnL() for sid in market.qualities}
    for t in market.trades:
        q, p = market.qualities[t.seller_id], out[t.seller_id]
        out[t.seller_id] = LemonsPnL(p.trade_surplus + (t.price - q) * t.qty, 0, p.sold + t.qty)
    for sid, fee in market.cert_paid.items():
        p = out[sid]
        out[sid] = LemonsPnL(p.trade_surplus, p.certification_cost + fee, p.sold, p.certified + 1)
    return out


@dataclass(frozen=True)
class ReputationPnL:
    """Reputation seller PnL in ticks: `net = trade_surplus + cheating_gain - whitewash_cost`."""
    trade_surplus: int = 0     # sum(price - q) over sales
    cheating_gain: int = 0     # sum(q - delivered) over sales
    whitewash_cost: int = 0    # ticks paid to the identity sink
    sold: int = 0
    cheats: int = 0
    whitewashes: int = 0

    @property
    def net(self) -> int:
        return self.trade_surplus + self.cheating_gain - self.whitewash_cost

    def as_dict(self) -> Dict[str, int]:
        return {"trade_surplus": self.trade_surplus, "cheating_gain": self.cheating_gain,
                "whitewash_cost": self.whitewash_cost, "net": self.net, "sold": self.sold,
                "cheats": self.cheats, "whitewashes": self.whitewashes}


def reputation_pnl(market) -> Dict[int, ReputationPnL]:
    """Per-seller `ReputationPnL` of a fl-058 `ReputationMarket` run."""
    acc = {sid: [0, 0, 0, 0] for sid in market.sellers}
    for t in market.trades:
        q, out = market.sellers[t.seller_id].quality, market.delivered[t.trade_id]
        row = acc[t.seller_id]
        row[0] += (t.price - q) * t.qty
        row[1] += (q - out) * t.qty
        row[2] += t.qty
        row[3] += int(out < q)
    return {sid: ReputationPnL(r[0], r[1], market.wash_paid[sid], r[2], r[3],
                               market.sellers[sid].whitewashes) for sid, r in acc.items()}


@dataclass(frozen=True)
class SwitchingPnL:
    """Lock-in seller PnL in ticks: `net = acquisition + harvest` (margins over cost c)."""
    acquisition: int = 0       # sum(price - c) over acquire-phase sales (the investment)
    harvest: int = 0           # sum(price - c) over harvest / Bertrand-phase sales
    sold_acquire: int = 0
    sold_harvest: int = 0

    @property
    def net(self) -> int:
        return self.acquisition + self.harvest

    def as_dict(self) -> Dict[str, int]:
        return {"acquisition": self.acquisition, "harvest": self.harvest, "net": self.net,
                "sold_acquire": self.sold_acquire, "sold_harvest": self.sold_harvest}


def switching_pnl(market) -> Dict[int, SwitchingPnL]:
    """Per-seller `SwitchingPnL` of a fl-061 `SwitchingMarket` run."""
    acc = {sid: [0, 0, 0, 0] for sid in market.sellers}
    for t in market.trades:
        row, m = acc[t.seller_id], (t.price - market.sellers[t.seller_id].marginal_cost) * t.qty
        k = 0 if market.phase_of[t.trade_id] == "acquire" else 1
        row[k] += m
        row[k + 2] += t.qty
    return {sid: SwitchingPnL(*r) for sid, r in acc.items()}


def consumer_surplus(market) -> Dict[int, Tuple[int, int, int]]:
    """Per consumer (gross v - price, switching cost paid, net) of a `SwitchingMarket`."""
    gross = dict.fromkeys(market.values, 0)
    for t in market.trades:
        gross[t.buyer_id] += (market.values[t.buyer_id] - t.price) * t.qty
    return {c: (g, market.switch_paid[c], g - market.switch_paid[c]) for c, g in gross.items()}


def middleman_pnl(market) -> MiddlemanPnL:
    """Split the middleman PnL of a fl-046 `SearchMarket` run from its fills and carry log."""
    mm = market.middleman
    if mm is None:
        raise ValueError("market has no middleman")
    fills = [t for t in market.trades if mm.agent_id in (t.buyer_id, t.seller_id)]
    asks = sum(t.price for t in fills if t.seller_id == mm.agent_id)
    bids = sum(t.price for t in fills if t.buyer_id == mm.agent_id)
    sold = sum(1 for t in fills if t.seller_id == mm.agent_id)
    held = sum(u.basis for u in mm.inventory)
    hold = sum(row[2] for row in market.carries)
    spoil = sum(row[3] for row in market.carries)
    return MiddlemanPnL(asks - (bids - held), hold, spoil, len(fills) - sold, sold, held)
