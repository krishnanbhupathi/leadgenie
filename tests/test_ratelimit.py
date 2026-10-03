import asyncio

import pytest

from leadgenie.ratelimit import KeyedLimiter, TokenBucket


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


async def test_burst_then_throttle():
    t = FakeTime()
    bucket = TokenBucket(rate=2, burst=2, clock=t.clock, sleep=t.sleep)
    for _ in range(4):
        await bucket.acquire()
    # Two free (burst), then two more at 0.5s each.
    assert t.sleeps == pytest.approx([0.5, 0.5])
    assert t.now == pytest.approx(1.0)


async def test_refills_over_time():
    t = FakeTime()
    bucket = TokenBucket(rate=1, burst=1, clock=t.clock, sleep=t.sleep)
    await bucket.acquire()
    t.now += 5
    await bucket.acquire()
    assert t.sleeps == []


async def test_concurrent_waiters_are_spaced():
    t = FakeTime()
    bucket = TokenBucket(rate=10, burst=1, clock=t.clock, sleep=t.sleep)
    await asyncio.gather(*(bucket.acquire() for _ in range(5)))
    assert t.now == pytest.approx(0.4)


def test_keyed_limiter_isolates_keys():
    limiter = KeyedLimiter(rate=1)
    assert limiter.for_key("a.example") is limiter.for_key("a.example")
    assert limiter.for_key("a.example") is not limiter.for_key("b.example")


def test_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        TokenBucket(rate=0)
