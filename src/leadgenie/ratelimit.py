"""Async rate limiting: a token bucket, and a keyed variant for per-domain politeness."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


class TokenBucket:
    """Allows `rate` acquisitions per second on average, with bursts up to `burst`.

    Waiters are served in arrival order because the lock is held while sleeping.
    """

    def __init__(
        self,
        rate: float,
        burst: int = 1,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = asyncio.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
        self._last = now

    async def acquire(self) -> None:
        async with self._lock:
            self._refill()
            if self._tokens < 1:
                await self._sleep((1 - self._tokens) / self.rate)  # type: ignore[misc]
                self._refill()
            self._tokens -= 1

    async def __aenter__(self) -> TokenBucket:
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class KeyedLimiter:
    """One TokenBucket per key (e.g. per hostname), created on first use."""

    def __init__(self, rate: float, burst: int = 1) -> None:
        self.rate = rate
        self.burst = burst
        self._buckets: dict[str, TokenBucket] = {}

    def for_key(self, key: str) -> TokenBucket:
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(self.rate, self.burst)
        return self._buckets[key]
