import threading
import time

from app.core.rate_limit import RateLimiter


def test_allows_attempts_within_limit():
    limiter = RateLimiter(max_attempts=3, window_seconds=60)

    results = [limiter.hit("key") for _ in range(3)]

    assert [allowed for allowed, _ in results] == [True, True, True]


def test_blocks_attempts_over_limit():
    limiter = RateLimiter(max_attempts=2, window_seconds=60)

    assert limiter.hit("key")[0] is True
    assert limiter.hit("key")[0] is True

    allowed, retry_after = limiter.hit("key")

    assert allowed is False
    assert retry_after > 0


def test_blocked_attempt_does_not_extend_window():
    limiter = RateLimiter(max_attempts=1, window_seconds=1)

    assert limiter.hit("key")[0] is True
    assert limiter.hit("key")[0] is False

    time.sleep(1.1)

    assert limiter.hit("key")[0] is True


def test_reset_clears_attempts():
    limiter = RateLimiter(max_attempts=1, window_seconds=60)

    assert limiter.hit("key")[0] is True
    assert limiter.hit("key")[0] is False

    limiter.reset("key")

    assert limiter.hit("key")[0] is True


def test_keys_are_isolated():
    limiter = RateLimiter(max_attempts=1, window_seconds=60)

    assert limiter.hit("user-a")[0] is True
    assert limiter.hit("user-a")[0] is False
    assert limiter.hit("user-b")[0] is True


# ---------------------------------------------------------------------------
# Bounded memory: expired events and idle keys must not accumulate
# ---------------------------------------------------------------------------


def test_expired_events_are_pruned_on_hit():
    limiter = RateLimiter(
        max_attempts=2,
        window_seconds=0.05,
        sweep_interval=3600,
    )

    assert limiter.hit("key")[0] is True
    assert limiter.hit("key")[0] is True
    assert limiter.hit("key")[0] is False

    time.sleep(0.12)

    # The window rolled over: both old events were dropped, so the
    # key records a fresh attempt instead of carrying dead history.
    assert limiter.hit("key")[0] is True
    assert limiter.key_count() == 1


def test_idle_keys_are_swept_automatically():
    limiter = RateLimiter(
        max_attempts=1,
        window_seconds=0.05,
        sweep_interval=0.01,
    )

    assert limiter.hit("idle-key")[0] is True
    assert limiter.key_count() == 1

    time.sleep(0.12)

    # Any later hit triggers the periodic sweep, which drops keys
    # whose events have all expired — including keys never touched
    # again (the case an attacker spraying unique keys relies on).
    assert limiter.hit("another-key")[0] is True

    assert limiter.key_count() == 1
    assert limiter.hit("idle-key")[0] is True


def test_key_count_is_bounded_and_evicts_least_recently_used():
    limiter = RateLimiter(
        max_attempts=1,
        window_seconds=60,
        max_keys=3,
    )

    for index in range(10):
        assert limiter.hit(f"key-{index}")[0] is True
        assert limiter.key_count() <= 3

    # The oldest key was evicted to respect the cap, so it starts
    # with a fresh budget instead of growing the map.
    assert limiter.hit("key-0")[0] is True
    assert limiter.key_count() <= 3


def test_blocked_attempts_do_not_record_and_do_not_grow_map():
    limiter = RateLimiter(
        max_attempts=1,
        window_seconds=60,
        sweep_interval=3600,
    )

    assert limiter.hit("key")[0] is True

    for _ in range(5):
        assert limiter.hit("key")[0] is False

    assert limiter.key_count() == 1


def test_reset_removes_key_entirely():
    limiter = RateLimiter(max_attempts=1, window_seconds=60)

    limiter.hit("key")
    assert limiter.key_count() == 1

    limiter.reset("key")

    assert limiter.key_count() == 0
    assert limiter.hit("key")[0] is True


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_hits_allow_exactly_max_attempts():
    limiter = RateLimiter(max_attempts=10, window_seconds=60)
    barrier = threading.Barrier(16)
    results = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        allowed, _ = limiter.hit("shared-key")
        with lock:
            results.append(allowed)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 16
    assert sum(1 for allowed in results if allowed) == 10
    assert sum(1 for allowed in results if not allowed) == 6
    assert limiter.key_count() == 1


def test_concurrent_unique_keys_stay_within_cap():
    limiter = RateLimiter(
        max_attempts=1,
        window_seconds=60,
        max_keys=8,
    )
    barrier = threading.Barrier(16)
    errors = []

    def worker(index):
        barrier.wait()
        try:
            for step in range(5):
                limiter.hit(f"key-{index}-{step}")
                assert limiter.key_count() <= 8
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(index,))
        for index in range(16)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert limiter.key_count() <= 8
