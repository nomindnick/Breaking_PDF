"""Advanced caching for PDF processing with metrics and memory management."""

import logging
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

import numpy as np
import psutil
from PIL import Image

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheMetrics:
    """Tracks cache performance metrics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_time_saved: float = 0.0
    total_memory_saved_mb: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_time_saved_per_hit(self) -> float:
        """Average time saved per cache hit."""
        return self.total_time_saved / self.hits if self.hits > 0 else 0.0

    def log_summary(self) -> None:
        """Log cache performance summary with hit rate and time saved."""
        logger.info(
            f"Cache Performance - Hit Rate: {self.hit_rate:.1%}, "
            f"Time Saved: {self.total_time_saved:.2f}s, "
            f"Memory Saved: {self.total_memory_saved_mb:.1f}MB"
        )


@dataclass
class CacheEntry(Generic[T]):
    """Wrapper for cached items with metadata."""

    value: T
    size_mb: float
    creation_time: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)

    def access(self) -> None:
        """Update access statistics for LRU tracking."""
        self.access_count += 1
        self.last_access = time.time()


class AdvancedLRUCache(Generic[T]):
    """
    Production-ready LRU cache with memory management and metrics.

    Features:
    - Memory-based eviction (not just count-based)
    - Performance metrics tracking
    - System memory awareness
    - TTL support for entries
    - Warm-up capability for predictive caching
    """

    def __init__(
        self,
        max_memory_mb: int = 100,
        max_items: int = 1000,
        ttl_seconds: Optional[int] = None,
        memory_pressure_threshold: float = 0.8,
        eviction_ratio: float = 0.5,
    ):
        """
        Initialize advanced LRU cache with memory management.

        Args:
            max_memory_mb: Maximum memory usage in MB
            max_items: Maximum number of items to cache
            ttl_seconds: Time-to-live for cache entries (None = no expiry)
            memory_pressure_threshold: System memory threshold for aggressive eviction
            eviction_ratio: Ratio of items to evict during aggressive eviction
        """
        self.cache: OrderedDict[Any, CacheEntry[T]] = OrderedDict()
        self.max_memory_mb = max_memory_mb
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self.memory_pressure_threshold = memory_pressure_threshold
        self.eviction_ratio = eviction_ratio
        self.current_memory_mb = 0.0
        self.metrics = CacheMetrics()

    def get(
        self,
        key: Any,
        compute_func: Optional[Callable[[], T]] = None,
        compute_time: float = 0.0,
    ) -> Optional[T]:
        """
        Get item from cache or compute if missing.

        Args:
            key: Cache key
            compute_func: Optional function to compute value if missing
            compute_time: Estimated time to compute (for metrics)
        """
        # Check for expired entries
        if key in self.cache:
            entry = self.cache[key]

            # Check TTL
            if (
                self.ttl_seconds
                and time.time() - entry.creation_time > self.ttl_seconds
            ):
                self.evict(key)
            else:
                # Cache hit!
                self.metrics.hits += 1
                self.metrics.total_time_saved += compute_time
                self.metrics.total_memory_saved_mb += entry.size_mb

                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.access()

                return entry.value

        # Cache miss
        self.metrics.misses += 1

        # Compute if function provided
        if compute_func:
            value = compute_func()
            self.put(key, value)
            return value

        return None

    def put(self, key: Any, value: T, size_mb: Optional[float] = None) -> None:
        """Add item to cache with automatic memory management."""
        # Calculate size if not provided
        if size_mb is None:
            size_mb = self._estimate_size(value)

        # Check system memory pressure
        if self._check_memory_pressure():
            self._aggressive_eviction()

        # Normal eviction to make space
        while (
            self.current_memory_mb + size_mb > self.max_memory_mb
            or len(self.cache) >= self.max_items
        ):
            if not self.cache:
                break
            self._evict_lru()

        # Add new entry
        entry = CacheEntry(value=value, size_mb=size_mb, creation_time=time.time())
        self.cache[key] = entry
        self.current_memory_mb += size_mb

    def evict(self, key: Any) -> None:
        """Manually evict a specific key."""
        if key in self.cache:
            entry = self.cache.pop(key)

            # Close PIL Image if applicable
            if hasattr(entry.value, "close"):
                try:
                    entry.value.close()
                except Exception:
                    pass  # Ignore errors during cleanup

            self.current_memory_mb -= entry.size_mb
            self.metrics.evictions += 1

    def _evict_lru(self):
        """Evict least recently used item."""
        if self.cache:
            key, entry = self.cache.popitem(last=False)

            # Close PIL Image if applicable
            if hasattr(entry.value, "close"):
                try:
                    entry.value.close()
                except Exception:
                    pass  # Ignore errors during cleanup

            self.current_memory_mb -= entry.size_mb
            self.metrics.evictions += 1
            logger.debug(f"Evicted cache entry: {key}")

    def _aggressive_eviction(self):
        """Evict portion of cache when system memory is under pressure."""
        target_size = int(len(self.cache) * (1 - self.eviction_ratio))
        while len(self.cache) > target_size:
            self._evict_lru()
        logger.warning(f"Memory pressure detected - evicted to {len(self.cache)} items")

    def _check_memory_pressure(self) -> bool:
        """Check if system memory is under pressure."""
        try:
            mem = psutil.virtual_memory()
            return mem.percent / 100 > self.memory_pressure_threshold
        except Exception as e:
            logger.warning(f"Failed to check memory pressure: {e}")
            return False

    def _estimate_size(self, value: Any) -> float:
        """Estimate memory size of object in MB."""
        try:
            if isinstance(value, np.ndarray):
                # Numpy array size
                return value.nbytes / (1024 * 1024)
            elif isinstance(value, Image.Image):
                # PIL Image size estimation
                return (value.width * value.height * len(value.getbands())) / (
                    1024 * 1024
                )
            elif isinstance(value, str):
                return len(value.encode("utf-8")) / (1024 * 1024)
            else:
                return sys.getsizeof(value) / (1024 * 1024)
        except Exception as e:
            logger.debug(f"Failed to estimate size for {type(value)}: {e}")
            return 1.0  # Default 1MB if estimation fails

    def warmup(self, keys: List[Any], compute_func: Callable[[Any], T]) -> None:
        """Pre-populate cache with likely-to-be-used items."""
        for key in keys:
            if key not in self.cache:
                try:
                    value = compute_func(key)
                    self.put(key, value)
                except Exception as e:
                    logger.debug(f"Failed to warmup cache for {key}: {e}")

    def clear(self) -> None:
        """Clear entire cache and cleanup resources."""
        # Close PIL Images before clearing
        for item in self.cache.values():
            if hasattr(item, "close"):
                try:
                    item.close()
                except Exception:
                    pass  # Ignore errors during cleanup

        self.cache.clear()
        self.current_memory_mb = 0.0
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "size": len(self.cache),
            "memory_mb": self.current_memory_mb,
            "hit_rate": f"{self.metrics.hit_rate:.1%}",
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "evictions": self.metrics.evictions,
            "time_saved_seconds": self.metrics.total_time_saved,
            "memory_saved_mb": self.metrics.total_memory_saved_mb,
            "oldest_entry_age": self._get_oldest_age(),
        }

    def _get_oldest_age(self) -> Optional[float]:
        """Get age of oldest cache entry in seconds."""
        if self.cache:
            oldest = next(iter(self.cache.values()))
            return time.time() - oldest.creation_time
        return None


class PDFProcessingCache:
    """Specialized cache manager for PDF processing with multiple cache types."""

    def __init__(self, config):
        """Initialize specialized caches for different PDF data types."""
        # Rendered pages (memory-heavy)
        self.render_cache = AdvancedLRUCache[Image.Image](
            max_memory_mb=config.render_cache_memory_mb,
            max_items=config.page_cache_size,
            ttl_seconds=3600,  # 1 hour TTL
            eviction_ratio=config.cache_aggressive_eviction_ratio,
        )

        # Extracted text (memory-light, access-heavy)
        self.text_cache = AdvancedLRUCache[dict](
            max_memory_mb=config.text_cache_memory_mb,
            max_items=config.page_cache_size * 2,
            ttl_seconds=7200,  # 2 hour TTL
            eviction_ratio=config.cache_aggressive_eviction_ratio,
        )

        # Page analysis results (very light)
        self.analysis_cache = AdvancedLRUCache[dict](
            max_memory_mb=10,
            max_items=1000,
            ttl_seconds=3600,
            eviction_ratio=config.cache_aggressive_eviction_ratio,
        )

    def get_rendered_page(
        self, pdf_path: str, page_num: int, dpi: int, render_func=None
    ) -> Optional[Image.Image]:
        """Get rendered page with cache management."""
        cache_key = (pdf_path, page_num, dpi)
        return self.render_cache.get(
            cache_key,
            compute_func=render_func,
            compute_time=0.03,  # Estimated render time
        )

    def get_page_text(
        self, pdf_path: str, page_num: int, extract_func=None
    ) -> Optional[Any]:
        """Get extracted text with cache management."""
        cache_key = (pdf_path, page_num, "text")
        result = self.text_cache.get(
            cache_key,
            compute_func=extract_func,
            compute_time=0.08,  # Estimated extraction time
        )
        return result

    def warmup_pages(
        self,
        pdf_path: str,
        page_range: range,
        render_func,
        extract_func,
        dpi: int = 300,
    ):
        """Pre-cache likely-to-be-accessed pages."""
        # Warmup render cache
        render_keys = [(pdf_path, p, dpi) for p in page_range]
        self.render_cache.warmup(render_keys[:10], lambda k: render_func(*k))

        # Warmup text cache
        text_keys = [(pdf_path, p, "text") for p in page_range]
        self.text_cache.warmup(text_keys[:10], lambda k: extract_func(k[0], k[1]))

    def get_combined_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            "render_cache": self.render_cache.get_stats(),
            "text_cache": self.text_cache.get_stats(),
            "analysis_cache": self.analysis_cache.get_stats(),
            "total_memory_mb": (
                self.render_cache.current_memory_mb
                + self.text_cache.current_memory_mb
                + self.analysis_cache.current_memory_mb
            ),
        }

    def log_performance(self):
        """Log performance metrics for all caches."""
        logger.info("=== Cache Performance Report ===")
        self.render_cache.metrics.log_summary()
        self.text_cache.metrics.log_summary()
        self.analysis_cache.metrics.log_summary()
