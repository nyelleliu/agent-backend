import threading
import time
from collections import OrderedDict


class RateLimiter:
    """In-memory sliding-window rate limiter with bounded memory.

    Tracks attempts per key (for example ``login:203.0.113.9``) inside
    a sliding time window. Thread-safe and dependency-free.

    Memory is bounded on purpose: a limiter keyed by client input (IP
    address, user id) would otherwise grow forever, because keys that
    are never used again are never revisited. Three mechanisms keep
    the map small:

    1. Every ``hit`` prunes that key's expired events.
    2. A periodic sweep (``sweep_interval``) drops keys whose events
       have all expired, so idle keys disappear on their own.
    3. At most ``max_keys`` keys are tracked; exceeding the cap evicts
       the least recently used key, so spraying unique keys cannot
       grow the map without bound.

    Note: the counters live in process memory, so they only protect a
    single-process deployment. With multiple uvicorn workers, back
    this with a shared store such as Redis.
    """

    DEFAULT_MAX_KEYS = 10_000

    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: float = 300.0,
        max_keys: int = DEFAULT_MAX_KEYS,
        sweep_interval: float | None = None,
    ):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.max_keys = max(1, max_keys)

        # Sweep at least once per window, at most once a minute, so
        # long-lived idle keys are reaped without scanning on every hit.
        self.sweep_interval = (
            float(sweep_interval)
            if sweep_interval is not None
            else max(1.0, min(window_seconds, 60.0))
        )

        self._events: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def hit(self, key: str) -> tuple[bool, int]:
        """Record one attempt.

        Returns ``(allowed, retry_after_seconds)``. When the limit is
        exceeded the attempt is not recorded and ``retry_after`` tells
        the caller how long to wait.
        """
        now = time.monotonic()

        with self._lock:
            self._sweep_if_due(now)

            events = [
                moment
                for moment in self._events.get(key, [])
                if now - moment < self.window_seconds
            ]

            if len(events) >= self.max_attempts:
                if events:
                    self._events[key] = events
                    self._events.move_to_end(key)
                    oldest = events[0]
                else:
                    # max_attempts <= 0: nothing is ever allowed.
                    self._events.pop(key, None)
                    oldest = now

                retry_after = self.window_seconds - (now - oldest)

                return False, max(1, int(retry_after) + 1)

            events.append(now)

            if key in self._events:
                self._events[key] = events
                self._events.move_to_end(key)
            else:
                while len(self._events) >= self.max_keys:
                    self._events.popitem(last=False)

                self._events[key] = events

            return True, 0

    def reset(self, key: str) -> None:
        """Forget all attempts for a key, e.g. after a success."""
        with self._lock:
            self._events.pop(key, None)

    def key_count(self) -> int:
        """How many keys are currently tracked (for tests/metrics)."""
        with self._lock:
            return len(self._events)

    def _sweep_if_due(self, now: float) -> None:
        """Drop keys whose events have all expired. Caller holds lock."""
        if now - self._last_sweep < self.sweep_interval:
            return

        self._last_sweep = now

        cutoff = now - self.window_seconds

        for key in [
            name
            for name, events in self._events.items()
            if not events or events[-1] <= cutoff
        ]:
            del self._events[key]
