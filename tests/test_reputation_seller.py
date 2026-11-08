"""Tests for fl-058 `ReputationSeller`: cheat rule, milking round, whitewash rule, scores."""
from __future__ import annotations

import pytest

from frictionlab.agents import Agent, ReputationSeller, switch_round
from frictionlab.markets import ReputationMarket

SCHEDULE = [10, 20, 30, 40, 50, 60]


def _market(seed=58, **kw):
    base = dict(rounds=10, premium=15, rating_accuracy_bps=10_000, prior_honest_bps=7000)
    base.update(kw)
    return ReputationMarket.from_schedule(24, SCHEDULE, seed=seed,
                                          opportunists=range(2, 25, 2), **base)


def test_is_agent_and_decides_no_book_intents():
    s = ReputationSeller(1, 40, opportunist=True)
    assert isinstance(s, Agent) and s.decide(None) == ()
    assert not ReputationSeller(2, 40).opportunist


def test_honest_seller_never_cheats():
    s = ReputationSeller(1, 60)
    assert not any(s.will_cheat(t, 5, 20, 10, 10) for t in range(5))


def test_cheat_rule_is_end_game_threshold():
    s = ReputationSeller(1, 50, opportunist=True)    # saving 40 vs floor 10
    # gap 10: cheat iff (T - 1 - t) * 10 < 40 -> remaining <= 3
    flags = [s.will_cheat(t, 10, 30, 20, 10) for t in range(10)]
    assert flags == [False] * 6 + [True] * 4
    assert flags.index(True) == switch_round(10, 40, 10) == 6


@pytest.mark.parametrize("rounds,saving,gap,expected", [
    (10, 40, 10, 6), (10, 41, 10, 5), (5, 100, 1, 0), (5, 0, 7, None), (5, 9, 0, 0),
    (1, 3, 100, 0),
])
def test_switch_round_formula(rounds, saving, gap, expected):
    assert switch_round(rounds, saving, gap) == expected


def test_whitewash_rule_is_strict():
    s = ReputationSeller(1, 30, opportunist=True)
    assert s.will_whitewash(30, 4, 25)
    assert not s.will_whitewash(30, 5, 25)          # tie keeps the old id
    assert not s.will_whitewash(30, None, 10)       # whitewashing unavailable


def test_perfect_ratings_never_pay_a_detected_cheat_above_floor():
    mkt = _market()
    mkt.run()
    caught = 0
    for sid, s in mkt.sellers.items():
        if not s.opportunist:
            continue
        detected = False
        for r in mkt.history(sid):
            if detected and r.delivered is not None:
                assert r.price <= mkt.floor_price
            if r.rating is False:
                detected, caught = True, caught + 1
    assert caught > 0


def test_honest_scores_non_decreasing_in_good_count():
    mkt = _market(rating_accuracy_bps=8000, seed=11)
    mkt.run()
    for sid, s in mkt.sellers.items():
        if s.opportunist:
            continue
        goods = [r.score[0] for r in mkt.history(sid)] + [mkt.scores[sid][0]]
        assert goods == sorted(goods)


def test_milking_round_deterministic_and_not_before_analytic_switch():
    a, b = _market(), _market()
    a.run()
    b.run()
    gap = a.fresh_price - a.floor_price
    milked = 0
    for sid, s in a.sellers.items():
        assert s.milking_round == b.sellers[sid].milking_round
        if not s.opportunist:
            assert s.milking_round is None and s.cheats == 0
            continue
        t_star = switch_round(a.rounds, s.saving(a.floor), gap)
        if s.milking_round is not None:
            milked += 1
            assert t_star is not None and s.milking_round >= t_star
    assert milked > 0


def test_whitewash_only_when_fresh_id_net_beats_score_price():
    mkt = _market(rating_accuracy_bps=8000, prior_honest_bps=6000, whitewash_cost=2, seed=5)
    mkt.run()
    washes = [r for r in mkt.log if r.whitewashed]
    assert washes
    for r in washes:
        assert r.score == (0, 0) and r.price == mkt.fresh_price
        prev = [h for h in mkt.history(r.seller_id) if h.t < r.t]
        assert prev
        last = prev[-1]
        g, b = last.score
        if last.rating is not None:
            g, b = (g + 1, b) if last.rating else (g, b + 1)
        assert mkt.fresh_price - 2 > mkt.price_for((g, b))
