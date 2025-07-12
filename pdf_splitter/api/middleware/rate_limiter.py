"""
Enhanced Rate Limiting and Throttling Middleware

Provides flexible rate limiting with different strategies and storage backends.
"""
import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import redis
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from pdf_splitter.api.config import config

logger = logging.getLogger(__name__)


class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception."""

    def __init__(self, retry_after: Optional[int] = None):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)} if retry_after else None,
        )


class RateLimitStrategy:
    """Base class for rate limiting strategies."""

    async def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """
        Check if request is allowed.

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        raise NotImplementedError

    async def record_request(self, key: str):
        """Record a request."""
        raise NotImplementedError

    async def reset(self, key: str):
        """Reset rate limit for a key."""
        raise NotImplementedError


class FixedWindowStrategy(RateLimitStrategy):
    """Fixed window rate limiting strategy."""

    def __init__(self, requests: int, window: int):
        """
        Initialize fixed window strategy.

        Args:
            requests: Number of requests allowed
            window: Time window in seconds
        """
        self.requests = requests
        self.window = window
        self.storage: Dict[str, Dict[str, Any]] = {}

    async def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed."""
        now = time.time()

        if key not in self.storage:
            return True, None

        data = self.storage[key]
        window_start = data["window_start"]

        # Check if window has expired
        if now - window_start >= self.window:
            # New window
            return True, None

        # Check request count
        if data["count"] >= self.requests:
            # Calculate retry after
            retry_after = int(self.window - (now - window_start))
            return False, retry_after

        return True, None

    async def record_request(self, key: str):
        """Record a request."""
        now = time.time()

        if key not in self.storage:
            self.storage[key] = {"window_start": now, "count": 1}
        else:
            data = self.storage[key]

            # Check if window has expired
            if now - data["window_start"] >= self.window:
                # New window
                data["window_start"] = now
                data["count"] = 1
            else:
                # Increment count
                data["count"] += 1

    async def reset(self, key: str):
        """Reset rate limit for a key."""
        if key in self.storage:
            del self.storage[key]


class SlidingWindowStrategy(RateLimitStrategy):
    """Sliding window rate limiting strategy."""

    def __init__(self, requests: int, window: int):
        """
        Initialize sliding window strategy.

        Args:
            requests: Number of requests allowed
            window: Time window in seconds
        """
        self.requests = requests
        self.window = window
        self.storage: Dict[str, deque] = defaultdict(deque)

    async def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed."""
        now = time.time()
        queue = self.storage[key]

        # Remove expired timestamps
        cutoff = now - self.window
        while queue and queue[0] < cutoff:
            queue.popleft()

        # Check request count
        if len(queue) >= self.requests:
            # Calculate retry after
            oldest_request = queue[0]
            retry_after = int(oldest_request + self.window - now)
            return False, retry_after

        return True, None

    async def record_request(self, key: str):
        """Record a request."""
        now = time.time()
        self.storage[key].append(now)

    async def reset(self, key: str):
        """Reset rate limit for a key."""
        if key in self.storage:
            del self.storage[key]


class TokenBucketStrategy(RateLimitStrategy):
    """Token bucket rate limiting strategy."""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket strategy.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.storage: Dict[str, Dict[str, Any]] = {}

    async def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed."""
        now = time.time()

        if key not in self.storage:
            self.storage[key] = {"tokens": self.capacity, "last_refill": now}

        bucket = self.storage[key]

        # Refill tokens
        time_passed = now - bucket["last_refill"]
        tokens_to_add = time_passed * self.refill_rate
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now

        # Check if token available
        if bucket["tokens"] >= 1:
            return True, None

        # Calculate retry after
        tokens_needed = 1 - bucket["tokens"]
        retry_after = int(tokens_needed / self.refill_rate)
        return False, retry_after

    async def record_request(self, key: str):
        """Record a request."""
        if key in self.storage:
            self.storage[key]["tokens"] -= 1

    async def reset(self, key: str):
        """Reset rate limit for a key."""
        if key in self.storage:
            del self.storage[key]


class RedisRateLimitStrategy(RateLimitStrategy):
    """Redis-backed rate limiting strategy."""

    def __init__(self, redis_client: redis.Redis, requests: int, window: int):
        """
        Initialize Redis rate limit strategy.

        Args:
            redis_client: Redis client
            requests: Number of requests allowed
            window: Time window in seconds
        """
        self.redis = redis_client
        self.requests = requests
        self.window = window

    async def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed."""
        redis_key = f"rate_limit:{key}"

        # Get current count
        count = await asyncio.get_event_loop().run_in_executor(
            None, self.redis.get, redis_key
        )

        if count is None:
            return True, None

        count = int(count)
        if count >= self.requests:
            # Get TTL for retry after
            ttl = await asyncio.get_event_loop().run_in_executor(
                None, self.redis.ttl, redis_key
            )
            return False, max(1, ttl)

        return True, None

    async def record_request(self, key: str):
        """Record a request."""
        redis_key = f"rate_limit:{key}"

        # Increment with expiry
        pipe = self.redis.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, self.window)

        await asyncio.get_event_loop().run_in_executor(None, pipe.execute)

    async def reset(self, key: str):
        """Reset rate limit for a key."""
        redis_key = f"rate_limit:{key}"
        await asyncio.get_event_loop().run_in_executor(
            None, self.redis.delete, redis_key
        )


class EnhancedRateLimitMiddleware(BaseHTTPMiddleware):
    """Enhanced rate limiting middleware with multiple strategies."""

    def __init__(
        self,
        app,
        strategy: Optional[RateLimitStrategy] = None,
        key_func: Optional[callable] = None,
        exclude_paths: Optional[set] = None,
    ):
        super().__init__(app)

        # Use provided strategy or default
        self.strategy = strategy or SlidingWindowStrategy(requests=60, window=60)

        # Key function to identify clients
        self.key_func = key_func or self._default_key_func

        # Paths to exclude from rate limiting
        self.exclude_paths = exclude_paths or {
            "/api/health",
            "/api/docs",
            "/api/openapi.json",
        }

        # Rate limit groups with different limits
        self.route_groups = {
            "auth": {"requests": 5, "window": 60},  # 5 requests per minute
            "upload": {"requests": 10, "window": 300},  # 10 uploads per 5 minutes
            "download": {"requests": 100, "window": 3600},  # 100 downloads per hour
            "default": {"requests": 60, "window": 60},  # 60 requests per minute
        }

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        # Check if path is excluded
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get client key
        key = self.key_func(request)

        # Get route group
        group = self._get_route_group(request.url.path)
        group_key = f"{key}:{group}"

        # Check rate limit
        allowed, retry_after = await self.strategy.is_allowed(group_key)

        if not allowed:
            # Log rate limit exceeded
            logger.warning(f"Rate limit exceeded for {key} on {request.url.path}")

            # Return rate limit response
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "type": "rate_limit_exceeded",
                            "message": "Too many requests",
                            "retry_after": retry_after,
                        }
                    }
                ),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.route_groups[group]["requests"]),
                    "X-RateLimit-Window": str(self.route_groups[group]["window"]),
                    "Content-Type": "application/json",
                },
            )

        # Record request
        await self.strategy.record_request(group_key)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        limit_info = self.route_groups[group]
        response.headers["X-RateLimit-Limit"] = str(limit_info["requests"])
        response.headers["X-RateLimit-Window"] = str(limit_info["window"])

        return response

    def _default_key_func(self, request: Request) -> str:
        """Default key function using client IP."""
        if request.client:
            return f"ip:{request.client.host}"

        # Fallback to forwarded IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        return "ip:unknown"

    def _get_route_group(self, path: str) -> str:
        """Determine route group for a path."""
        if "/auth" in path or "/login" in path:
            return "auth"
        elif "/upload" in path:
            return "upload"
        elif "/download" in path:
            return "download"
        else:
            return "default"


class AdaptiveRateLimitMiddleware(EnhancedRateLimitMiddleware):
    """Adaptive rate limiting that adjusts based on server load."""

    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)

        # Load monitoring
        self.request_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.load_factor = 1.0

    async def dispatch(self, request: Request, call_next):
        """Apply adaptive rate limiting."""
        start_time = time.time()

        try:
            # Apply adjusted rate limit
            response = await super().dispatch(request, call_next)

            # Record successful request
            self.request_times.append(time.time() - start_time)

            # Update load factor
            self._update_load_factor()

            return response

        except Exception as e:
            # Record error
            key = self.key_func(request)
            self.error_counts[key] += 1

            # Apply stricter limits for error-prone clients
            if self.error_counts[key] > 10:
                logger.warning(f"High error rate from {key}")
                # Could implement temporary ban here

            raise

    def _update_load_factor(self):
        """Update load factor based on response times."""
        if len(self.request_times) < 100:
            return

        # Calculate average response time
        avg_time = sum(self.request_times) / len(self.request_times)

        # Adjust load factor
        if avg_time > 1.0:  # Slow responses
            self.load_factor = max(0.5, self.load_factor - 0.1)
        elif avg_time < 0.1:  # Fast responses
            self.load_factor = min(2.0, self.load_factor + 0.1)
