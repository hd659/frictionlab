"""Tests for fl-055 `CertificationPolicy`: monotone cutoff, sink reconciliation and line ceiling."""
from __future__ import annotations

from pathlib import Path

import pytest

from frictionlab.costs import CERT_SINK_ID, CertificationPolicy, FixedCost, FrictionStack, HalfSpreadCost
from frictionlab.markets import CERTIFIED, POOLED, LemonsMarket

ROOT = Path(__file__).resolve().parents[1]


def test_monotone_separating_cutoff_certified_never_below_uncertified():
    cert = CertificationPolicy(cost=5)
    mkt = LemonsMarket([10, 20, 30, 40, 50, 60, 70, 80], premium=12, certification=cert)
    mkt.run()
    cert_qs = [mkt.qualities[sid] for sid in mkt.sellers_sold(CERTIFIED)]
    pool_qs = [mkt.qualities[sid] for sid in mkt.sellers_sold(POOLED)]
    unsold_qs = [q for sid, q in mkt.qualities.items() if sid not in mkt.sellers_sold()]
    assert cert_qs and pool_qs
    assert min(cert_qs) >= max(pool_qs + unsold_qs)
    cutoff = cert.cutoff(mkt.qualities.values(), mkt.premium, mkt.price)
    assert cutoff == min(cert_qs)


def test_certification_count_times_cost_reconciles_with_sink():
    cost = 7
    cert = CertificationPolicy(cost=cost)
    mkt = LemonsMarket([15, 25, 35, 45, 55, 65], premium=14, certification=cert)
    mkt.run()
    assert cert.count == len(mkt.certified) > 0
    assert cert.revenue == cert.count * cost
    assert mkt.certification_costs == cert.revenue
    assert mkt.ledger[CERT_SINK_ID].cash == cert.revenue
    mkt.ledger.check()


def test_cost_above_premium_never_certifies():
    cert = CertificationPolicy(cost=15)
    mkt = LemonsMarket([10, 20, 30, 40, 50], premium=10, certification=cert)
    mkt.run()
    assert cert.count == 0
    assert mkt.certified == ()
    assert mkt.ledger[CERT_SINK_ID].cash == 0


def test_custom_friction_stack_and_disabled_guard():
    stack = FrictionStack([FixedCost(3), HalfSpreadCost(4)])
    cert = CertificationPolicy(stack)
    assert cert.cost_for(1, 50) == 5
    dis = CertificationPolicy.disabled()
    assert not dis.certifies(1, 50, 20, 10)
    mkt = LemonsMarket([10, 20], premium=5)
    with pytest.raises(ValueError, match="disabled"):
        dis.charge(mkt.ledger, 1, 10)


def test_new_modules_stay_under_line_ceiling():
    for rel in ("frictionlab/markets/lemons.py",
                "frictionlab/costs/certification.py",
                "frictionlab/analysis/pnl.py"):
        lines = len((ROOT / rel).read_text("utf-8").splitlines())
        assert lines <= 250, f"{rel} has {lines} lines (> 250)"
