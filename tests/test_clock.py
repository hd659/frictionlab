import numpy as np
import pytest

from frictionlab.core.clock import EventQueue, SimClock


def _recorder(log):
    def h(clock, ev):
        log.append((clock.now, ev.kind, ev.payload))
    return h


def test_queue_orders_by_time_priority_then_seq():
    q = EventQueue()
    q.push(2.0, "c")
    q.push(1.0, "b", priority=1)
    q.push(1.0, "a", priority=0)
    q.push(1.0, "b2", priority=1)
    assert [q.pop().kind for _ in range(4)] == ["a", "b", "b2", "c"]
    assert not q and len(q) == 0


def test_queue_cancel_is_lazy_and_idempotent():
    q = EventQueue()
    e1 = q.push(1.0, "x")
    q.push(2.0, "y")
    assert q.cancel(e1) is True
    assert q.cancel(e1) is False
    assert len(q) == 1
    assert q.peek().kind == "y"
    assert q.pop().kind == "y"
    with pytest.raises(IndexError):
        q.pop()


def test_cancel_after_pop_is_noop():
    q = EventQueue()
    e = q.push(1.0, "x")
    q.pop()
    assert q.cancel(e) is False
    assert len(q) == 0


def test_queue_rejects_non_finite_time():
    q = EventQueue()
    with pytest.raises(ValueError):
        q.push(float("inf"), "x")
    with pytest.raises(ValueError):
        q.push(float("nan"), "x")


def test_clock_dispatch_advances_time():
    clk = SimClock(seed=1)
    log = []
    clk.on("tick", _recorder(log))
    clk.schedule(0.5, "tick", 1)
    clk.schedule_at(0.2, "tick", 0)
    assert clk.run() == 2
    assert log == [(0.2, "tick", 0), (0.5, "tick", 1)]
    assert clk.now == 0.5 and clk.processed == 2


def test_handlers_can_reschedule_and_run_until_horizon():
    clk = SimClock(seed=0)
    times = []

    def tick(c, ev):
        times.append(c.now)
        c.schedule(1.0, "tick")

    clk.on("tick", tick)
    clk.schedule_at(0.0, "tick")
    fired = clk.run(until=3.5)
    assert fired == 4 and times == [0.0, 1.0, 2.0, 3.0]
    assert clk.now == 3.5
    assert clk.queue.peek().time == 4.0
    clk.run(until=5.0)
    assert times[-1] == 5.0


def test_cannot_schedule_into_the_past():
    clk = SimClock()
    clk.schedule_at(2.0, "x")
    clk.run()
    with pytest.raises(ValueError):
        clk.schedule_at(1.0, "x")
    with pytest.raises(ValueError):
        clk.schedule(-0.1, "x")
    with pytest.raises(ValueError):
        clk.schedule_poisson(0.0, "x")


def test_stop_and_max_events():
    clk = SimClock()
    seen = []

    def h(c, ev):
        seen.append(ev.payload)
        if ev.payload == 2:
            c.stop()

    clk.on("e", h)
    for i in range(6):
        clk.schedule_at(float(i), "e", i)
    assert clk.run() == 3 and seen == [0, 1, 2]
    assert clk.run(max_events=2) == 2 and seen[-1] == 4
    assert clk.now == 4.0 and len(clk.queue) == 1


def test_cancelled_event_never_fires():
    clk = SimClock()
    log = []
    clk.on("e", _recorder(log))
    ev = clk.schedule(1.0, "e", "dead")
    clk.schedule(2.0, "e", "alive")
    clk.cancel(ev)
    clk.run()
    assert [p for _, _, p in log] == ["alive"]


def test_multiple_handlers_fire_in_registration_order():
    clk = SimClock()
    order = []
    clk.on("e", lambda c, ev: order.append("first"))
    clk.on("e", lambda c, ev: order.append("second"))
    clk.schedule(0.0, "e")
    clk.step()
    assert order == ["first", "second"]
    assert clk.step() is None


def _poisson_trace(seed, n=50):
    clk = SimClock(seed=seed)
    times = []

    def arrive(c, ev):
        times.append(c.now)
        if len(times) < n:
            c.schedule_poisson(2.0, "arrive")

    clk.on("arrive", arrive)
    clk.schedule_poisson(2.0, "arrive")
    clk.run()
    return np.array(times)


def test_seed_reproducibility_and_divergence():
    a, b, c = _poisson_trace(7), _poisson_trace(7), _poisson_trace(8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(np.diff(a) >= 0)


def test_poisson_rate_is_statistically_sane():
    t = _poisson_trace(123, n=4000)
    mean_gap = np.diff(np.concatenate([[0.0], t])).mean()
    assert abs(mean_gap - 0.5) < 0.05


def test_spawned_rngs_are_independent_and_reproducible():
    c1, c2 = SimClock(seed=42), SimClock(seed=42)
    r1a, r1b = c1.spawn_rng(), c1.spawn_rng()
    r2a, r2b = c2.spawn_rng(), c2.spawn_rng()
    xa, xb = r1a.random(5), r1b.random(5)
    assert not np.array_equal(xa, xb)
    assert np.array_equal(xa, r2a.random(5))
    assert np.array_equal(xb, r2b.random(5))
