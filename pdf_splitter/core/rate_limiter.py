"""Rate limiting utilities for preventing resource exhaustion."""

import asyncio
import time
from collections import deque
from contextlib import contextmanager
from threading import BoundedSemaphore, Lock
from typing import Optional

from pdf_splitter.core.exceptions import PDFSplitterError


class RateLimitExceeded(PDFSplitterError):
    """Raised when rate limit is exceeded."""

    pass


class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter for controlling request rates.

    This implementation provides:
    - Configurable rate and burst capacity
    - Thread-safe operation
    - Non-blocking and blocking acquire modes
    - Context manager support
    """

    def __init__(
        self, rate: float, capacity: int, initial_tokens: Optional[int] = None
    ):
        """
        Initialize rate limiter.

        Args:
            rate: Number of tokens replenished per second
            capacity: Maximum number of tokens (burst capacity)
            initial_tokens: Initial token count (defaults to capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(initial_tokens if initial_tokens is not None else capacity)
        self.last_update = time.monotonic()
        self._lock = Lock()

    def _replenish_tokens(self) -> None:
        """Replenish tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update

        # Add tokens based on rate and elapsed time
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_update = now

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        with self._lock:
            self._replenish_tokens()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds

        Returns:
            True if tokens were acquired

        Raises:
            RateLimitExceeded: If timeout is reached
        """
        start_time = time.monotonic()

        while True:
            if self.try_acquire(tokens):
                return True

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    raise RateLimitExceeded(
                        f"Failed to acquire {tokens} tokens within {timeout}s"
                    )

            # Sleep briefly before retrying
            time.sleep(0.01)  # 10ms

    @contextmanager
    def __call__(self, tokens: int = 1):
        """Context manager for rate limiting."""
        self.acquire(tokens)
        yield


class ConcurrencyLimiter:
    """
    Limit concurrent operations to prevent resource exhaustion.

    This implementation provides:
    - Maximum concurrent operations limit
    - Thread-safe operation
    - Context manager support
    - Queue length monitoring
    """

    def __init__(self, max_concurrent: int):
        """
        Initialize concurrency limiter.

        Args:
            max_concurrent: Maximum number of concurrent operations
        """
        self.semaphore = BoundedSemaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.active_count = 0
        self._lock = Lock()

    @property
    def available_slots(self) -> int:
        """Get number of available concurrency slots."""
        with self._lock:
            return self.max_concurrent - self.active_count

    def try_acquire(self) -> bool:
        """Try to acquire a slot without blocking."""
        acquired = self.semaphore.acquire(blocking=False)
        if acquired:
            with self._lock:
                self.active_count += 1
        return acquired

    def release(self) -> None:
        """Release a concurrency slot."""
        self.semaphore.release()
        with self._lock:
            self.active_count -= 1

    @contextmanager
    def __call__(self, timeout: Optional[float] = None):
        """
        Context manager for concurrency limiting.

        Args:
            timeout: Maximum time to wait for a slot

        Raises:
            RateLimitExceeded: If timeout is reached
        """
        acquired = self.semaphore.acquire(timeout=timeout)
        if not acquired:
            raise RateLimitExceeded(
                f"Failed to acquire concurrency slot within {timeout}s"
            )

        with self._lock:
            self.active_count += 1

        try:
            yield
        finally:
            self.release()


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for more precise rate control.

    This implementation tracks request timestamps and enforces
    limits over a sliding time window.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        """
        Initialize sliding window rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque[float] = deque()
        self._lock = Lock()

    def _cleanup_old_requests(self, current_time: float) -> None:
        """Remove requests outside the current window."""
        cutoff_time = current_time - self.window_seconds
        while self.requests and self.requests[0] <= cutoff_time:
            self.requests.popleft()

    def try_acquire(self) -> bool:
        """Try to make a request."""
        with self._lock:
            current_time = time.monotonic()
            self._cleanup_old_requests(current_time)

            if len(self.requests) < self.max_requests:
                self.requests.append(current_time)
                return True
            return False

    def get_wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        with self._lock:
            if not self.requests or len(self.requests) < self.max_requests:
                return 0.0

            # Calculate when the oldest request will expire
            oldest_request = self.requests[0]
            current_time = time.monotonic()
            wait_time = (oldest_request + self.window_seconds) - current_time
            return max(0.0, wait_time)


# Async versions for use with asyncio
class AsyncTokenBucketRateLimiter:
    """Async version of TokenBucketRateLimiter."""

    def __init__(self, rate: float, capacity: int):
        """Initialize async rate limiter."""
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens asynchronously."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_update

                # Replenish tokens
                tokens_to_add = elapsed * self.rate
                self.tokens = min(self.capacity, self.tokens + tokens_to_add)
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

            # Wait before retrying
            await asyncio.sleep(0.01)
