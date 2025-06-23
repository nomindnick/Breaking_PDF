"""Tests for advanced caching functionality."""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.preprocessing.advanced_cache import (
    AdvancedLRUCache,
    CacheMetrics,
    PDFProcessingCache,
)


class TestAdvancedLRUCache:
    """Test suite for AdvancedLRUCache."""

    def test_basic_get_put(self):
        """Test basic cache operations."""
        cache = AdvancedLRUCache[str](max_memory_mb=10, max_items=5)

        # Test put and get
        cache.put("key1", "value1", size_mb=1.0)
        assert cache.get("key1") == "value1"
        assert cache.metrics.hits == 1
        assert cache.metrics.misses == 0

        # Test cache miss
        assert cache.get("key2") is None
        assert cache.metrics.misses == 1

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = AdvancedLRUCache[int](max_memory_mb=10, max_items=3)

        # Fill cache
        cache.put("a", 1, size_mb=1.0)
        cache.put("b", 2, size_mb=1.0)
        cache.put("c", 3, size_mb=1.0)

        # Access 'a' to make it recently used
        cache.get("a")

        # Add new item - should evict 'b' (least recently used)
        cache.put("d", 4, size_mb=1.0)

        assert cache.get("a") == 1  # Still in cache
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3  # Still in cache
        assert cache.get("d") == 4  # New item

    def test_memory_based_eviction(self):
        """Test memory-based eviction."""
        cache = AdvancedLRUCache[str](max_memory_mb=5, max_items=100)

        # Add items that exceed memory limit
        cache.put("1", "x" * 1000000, size_mb=2.0)
        cache.put("2", "y" * 1000000, size_mb=2.0)
        cache.put("3", "z" * 1000000, size_mb=2.0)  # Should trigger eviction

        assert cache.current_memory_mb <= 5.0
        assert cache.metrics.evictions > 0

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = AdvancedLRUCache[str](max_memory_mb=10, ttl_seconds=1)

        cache.put("key", "value", size_mb=1.0)
        assert cache.get("key") == "value"

        # Wait for TTL to expire
        time.sleep(1.1)
        assert cache.get("key") is None  # Should be expired

    def test_compute_function(self):
        """Test compute function on cache miss."""
        cache = AdvancedLRUCache[int](max_memory_mb=10)

        compute_called = False

        def compute():
            nonlocal compute_called
            compute_called = True
            return 42

        # First call should compute
        result = cache.get("key", compute_func=compute, compute_time=0.1)
        assert result == 42
        assert compute_called
        assert cache.metrics.misses == 1

        # Second call should use cache
        compute_called = False
        result = cache.get("key", compute_func=compute, compute_time=0.1)
        assert result == 42
        assert not compute_called
        assert cache.metrics.hits == 1
        assert cache.metrics.total_time_saved == pytest.approx(0.1, rel=0.01)

    @patch("psutil.virtual_memory")
    def test_memory_pressure_handling(self, mock_memory):
        """Test memory pressure detection and handling."""
        # Simulate high memory usage
        mock_memory.return_value = Mock(percent=85)

        cache = AdvancedLRUCache[str](max_memory_mb=10, memory_pressure_threshold=0.8)

        # Fill cache
        for i in range(10):
            cache.put(f"key{i}", f"value{i}", size_mb=0.5)

        # Trigger memory pressure eviction
        cache.put("new_key", "new_value", size_mb=0.5)

        # Should have aggressively evicted items
        assert len(cache.cache) <= 5  # Half of original

    def test_size_estimation(self):
        """Test automatic size estimation for different object types."""
        cache = AdvancedLRUCache[any](max_memory_mb=100)

        # Test numpy array
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        cache.put("numpy", arr)
        assert cache.current_memory_mb == pytest.approx(0.029, rel=0.1)

        # Test string
        text = "x" * 1000000  # 1MB string
        cache.put("string", text)
        assert cache.current_memory_mb > 0.9  # Allow for some overhead

    def test_cache_stats(self):
        """Test cache statistics reporting."""
        cache = AdvancedLRUCache[str](max_memory_mb=10)

        # Generate some activity
        cache.put("a", "value1", size_mb=1.0)
        cache.get("a")  # Hit
        cache.get("b")  # Miss
        cache.put("b", "value2", size_mb=1.0)

        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == "50.0%"
        assert stats["memory_mb"] == 2.0


class TestPDFProcessingCache:
    """Test suite for PDFProcessingCache."""

    def test_initialization(self):
        """Test cache manager initialization."""
        config = PDFConfig(
            render_cache_memory_mb=50, text_cache_memory_mb=25, cache_ttl_seconds=1800
        )

        cache_manager = PDFProcessingCache(config)

        assert cache_manager.render_cache.max_memory_mb == 50
        assert cache_manager.text_cache.max_memory_mb == 25
        assert cache_manager.render_cache.ttl_seconds == 3600  # Config default is 3600

    def test_render_cache_operations(self):
        """Test render cache functionality."""
        config = PDFConfig()
        cache_manager = PDFProcessingCache(config)

        # Create test image
        test_array = np.ones((100, 100, 3), dtype=np.uint8) * 255

        def render_func():
            return test_array

        # First call should miss and compute
        result = cache_manager.get_rendered_page("test.pdf", 0, 150, render_func)
        assert np.array_equal(result, test_array)
        assert cache_manager.render_cache.metrics.misses == 1

        # Second call should hit cache
        result2 = cache_manager.get_rendered_page("test.pdf", 0, 150, render_func)
        assert np.array_equal(result2, test_array)
        assert cache_manager.render_cache.metrics.hits == 1

    def test_text_cache_operations(self):
        """Test text cache functionality."""
        config = PDFConfig()
        cache_manager = PDFProcessingCache(config)

        test_data = {"text": "Sample text", "quality_score": 0.95, "word_count": 2}

        def extract_func():
            return test_data

        # Test caching
        result = cache_manager.get_page_text("test.pdf", 0, extract_func)
        assert result == test_data

        # Verify cache hit on second call
        result2 = cache_manager.get_page_text("test.pdf", 0, extract_func)
        assert result2 == test_data
        assert cache_manager.text_cache.metrics.hits == 1

    def test_cache_warmup(self):
        """Test cache warmup functionality."""
        config = PDFConfig()
        cache_manager = PDFProcessingCache(config)

        render_calls = 0
        extract_calls = 0

        def mock_render(pdf_path, page_num, dpi):
            nonlocal render_calls
            render_calls += 1
            return np.zeros((100, 100, 3), dtype=np.uint8)

        def mock_extract(pdf_path, page_num):
            nonlocal extract_calls
            extract_calls += 1
            return {"text": f"Page {page_num}"}

        # Warmup first 5 pages
        cache_manager.warmup_pages("test.pdf", range(5), mock_render, mock_extract)

        # Verify warmup occurred (limited to 10 by default)
        assert render_calls > 0
        assert extract_calls > 0

    def test_combined_stats(self):
        """Test combined statistics reporting."""
        config = PDFConfig()
        cache_manager = PDFProcessingCache(config)

        # Generate some cache activity
        cache_manager.render_cache.put(
            ("test.pdf", 0, 150), np.zeros((100, 100, 3), dtype=np.uint8)
        )
        cache_manager.text_cache.put(("test.pdf", 0, "text"), {"text": "sample"})

        stats = cache_manager.get_combined_stats()

        assert "render_cache" in stats
        assert "text_cache" in stats
        assert "analysis_cache" in stats
        assert "total_memory_mb" in stats
        assert stats["total_memory_mb"] > 0


class TestCacheMetrics:
    """Test cache metrics tracking."""

    def test_metrics_calculation(self):
        """Test metrics calculations."""
        metrics = CacheMetrics()

        # Simulate cache activity
        metrics.hits = 8
        metrics.misses = 2
        metrics.total_time_saved = 1.5
        metrics.total_memory_saved_mb = 100

        assert metrics.hit_rate == 0.8
        assert metrics.avg_time_saved_per_hit == pytest.approx(0.1875)

    def test_metrics_with_no_activity(self):
        """Test metrics with no cache activity."""
        metrics = CacheMetrics()

        assert metrics.hit_rate == 0.0
        assert metrics.avg_time_saved_per_hit == 0.0
