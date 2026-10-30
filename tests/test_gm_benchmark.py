"""Tests for the analytic Glosten-Milgrom zero-profit spread (gm_zero_profit_spread)."""
from fractions import Fraction

import pytest

from frictionlab.experiments import AdverseSelectionConfig, gm_zero_profit_spread

MUS = [Fraction(k, 20) for k in range(21)]


@pytest.mark.parametrize("lo,hi", [(90, 110), (95, 105), (1, 2), (100, 100), (37, 81)])
def test_zero_informed_share_gives_zero_spread(lo, hi):
    q = gm_zero_profit_spread(0, lo, hi)
    assert q.bid_exact == q.ask_exact == q.mean == Fraction(lo + hi, 2)
    assert q.spread == (0 if (lo + hi) % 2 == 0 else 1)   # odd sum: mean sits between ticks
    assert q.bid <= q.bid_exact and q.ask >= q.ask_exact


@pytest.mark.parametrize("mu", MUS)
def test_symmetric_around_prior_mean(mu):
    q = gm_zero_profit_spread(mu, 90, 110)
    assert q.ask_exact - q.mean == q.mean - q.bid_exact == mu * 10   # 2 * mu * L with L = 10
    assert q.ask - 100 == 100 - q.bid
    for nb in (Fraction(1, 4), Fraction(1, 3), Fraction(2, 3)):     # q <-> 1-q mirrors quotes
        a, b = gm_zero_profit_spread(mu, 90, 110, nb), gm_zero_profit_spread(mu, 90, 110, 1 - nb)
        assert a.ask_exact - 100 == 100 - b.bid_exact
        assert a.bid_exact - 100 == 100 - b.ask_exact


@pytest.mark.parametrize("nb", [Fraction(1, 2), Fraction(1, 5), Fraction(4, 5)])
@pytest.mark.parametrize("lo,hi", [(90, 110), (60, 140), (99, 101)])
def test_spread_non_decreasing_in_mu(nb, lo, hi):
    qs = [gm_zero_profit_spread(m, lo, hi, nb) for m in MUS]
    exact = [q.ask_exact - q.bid_exact for q in qs]
    assert all(b >= a for a, b in zip(exact, exact[1:]))
    assert all(b.spread >= a.spread for a, b in zip(qs, qs[1:]))
    assert qs[-1].bid_exact == lo and qs[-1].ask_exact == hi          # mu = 1: full revelation
    assert all(lo <= q.bid <= q.ask <= hi for q in qs)


def test_quotes_are_conditional_expectations():
    """Ask = E[V | buy] by Bayes on the order-arrival tree, in exact arithmetic."""
    mu, nb, lo, hi = Fraction(3, 10), Fraction(2, 5), 80, 120
    p_buy_h, p_buy_l = mu + (1 - mu) * nb, (1 - mu) * nb
    p_sell_h, p_sell_l = (1 - mu) * (1 - nb), mu + (1 - mu) * (1 - nb)
    q = gm_zero_profit_spread(mu, lo, hi, nb)
    assert q.ask_exact == (hi * p_buy_h + lo * p_buy_l) / (p_buy_h + p_buy_l)
    assert q.bid_exact == (hi * p_sell_h + lo * p_sell_l) / (p_sell_h + p_sell_l)
    # zero expected profit per fill: E[(ask - V) | buy] == 0 and E[(V - bid) | sell] == 0
    assert p_buy_h * (q.ask_exact - hi) + p_buy_l * (q.ask_exact - lo) == 0
    assert p_sell_h * (hi - q.bid_exact) + p_sell_l * (lo - q.bid_exact) == 0


def test_integer_quote_never_loses_in_expectation():
    for mu in MUS:
        q = gm_zero_profit_spread(mu, 90, 110, Fraction(3, 10))
        assert q.bid == int(q.bid) and q.ask == int(q.ask)
        assert 0 <= q.ask - q.ask_exact < 1 and 0 <= q.bid_exact - q.bid < 1


def test_float_inputs_match_fractions():
    assert gm_zero_profit_spread(0.3, 90, 110) == gm_zero_profit_spread(Fraction(3, 10), 90, 110)
    assert gm_zero_profit_spread(0.25, 90, 110, 0.5).spread == 6


@pytest.mark.parametrize("bad", [
    dict(mu=-0.1), dict(mu=1.1), dict(noise_buy_prob=0), dict(noise_buy_prob=1),
    dict(v_low=110, v_high=90), dict(v_low=90.5),
])
def test_rejects_invalid_inputs(bad):
    args = dict(mu=0.2, v_low=90, v_high=110, noise_buy_prob=0.5) | bad
    with pytest.raises(ValueError):
        gm_zero_profit_spread(**args)


def test_config_gm_scales_with_jump():
    cfg = AdverseSelectionConfig()
    for j in (1, 2, 4, 8):
        q = cfg.gm(0.5, j)
        assert q.ask_exact - q.bid_exact == 2 * Fraction(1, 2) * cfg.alpha * j   # 2 * mu * L
        assert q.spread in (cfg.alpha * j, cfg.alpha * j + 1)          # outward rounding
        assert q.mean == cfg.v0
    assert [cfg.gm(m, 4).spread for m in (0, 0.1, 0.5)] == [0, 4, 20]
