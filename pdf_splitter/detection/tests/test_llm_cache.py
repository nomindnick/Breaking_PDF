"""
Tests for the LLM response cache.
"""

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from pdf_splitter.detection.llm_cache import LLMResponseCache


class TestLLMResponseCache:
    """Test the persistent LLM response cache."""

    @pytest.fixture
    def temp_cache(self):
        """Create a temporary cache for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = Path(f.name)

        cache = LLMResponseCache(cache_path, max_age_days=1, max_size_mb=10)
        yield cache

        # Cleanup
        cache_path.unlink(missing_ok=True)

    def test_cache_initialization(self, temp_cache):
        """Test cache initializes properly."""
        assert temp_cache.cache_path.exists()

        # Check database schema
        with sqlite3.connect(temp_cache.cache_path) as conn:
            # Check tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]

            assert "cache_entries" in table_names
            assert "cache_metadata" in table_names

    def test_cache_put_and_get(self, temp_cache):
        """Test storing and retrieving from cache."""
        text1 = "End of first page"
        text2 = "Start of second page"
        model = "gemma3:latest"
        response = "<thinking>Test</thinking>\n<answer>DIFFERENT</answer>"

        # Store in cache
        temp_cache.put(
            text1,
            text2,
            model,
            response,
            is_boundary=True,
            confidence=0.95,
            reasoning="Test reasoning",
        )

        # Retrieve from cache
        result = temp_cache.get(text1, text2, model)

        assert result is not None
        assert result[0] is True  # is_boundary
        assert result[1] == 0.95  # confidence
        assert result[2] == "Test reasoning"

        # Check stats
        stats = temp_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_miss(self, temp_cache):
        """Test cache miss behavior."""
        result = temp_cache.get("nonexistent1", "nonexistent2", "model")

        assert result is None

        stats = temp_cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1

    def test_cache_expiration(self, temp_cache):
        """Test that old entries are not returned."""
        text1 = "Old content"
        text2 = "Old page 2"
        model = "gemma3:latest"

        # Store in cache with old timestamp
        with sqlite3.connect(temp_cache.cache_path) as conn:
            old_time = time.time() - (2 * 86400)  # 2 days ago
            key = temp_cache._generate_key(text1, text2, model, None)

            conn.execute(
                """
                INSERT INTO cache_entries
                (key, page1_hash, page2_hash, response, is_boundary,
                 confidence, reasoning, model_name, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    key,
                    temp_cache._hash_text(text1),
                    temp_cache._hash_text(text2),
                    "old response",
                    True,
                    0.9,
                    "old reasoning",
                    model,
                    old_time,
                    old_time,
                ),
            )
            conn.commit()

        # Should not retrieve expired entry
        result = temp_cache.get(text1, text2, model)
        assert result is None

    def test_cache_size_limit(self, temp_cache):
        """Test cache size limiting."""
        # Fill cache with many entries
        for i in range(100):
            temp_cache.put(
                f"text1_{i}",
                f"text2_{i}",
                "model",
                f"response_{i}",
                True,
                0.9,
                f"reasoning_{i}",
            )

        # Force size check with very small limit
        temp_cache.max_size_mb = 0.001  # 1KB

        # Add one more entry to trigger cleanup
        temp_cache.put(
            "trigger", "cleanup", "model", "response", True, 0.9, "reasoning"
        )

        # Check that entries were removed
        stats = temp_cache.get_stats()
        assert stats["total_entries"] < 100

    def test_access_count_tracking(self, temp_cache):
        """Test that access counts are tracked properly."""
        text1 = "Access test 1"
        text2 = "Access test 2"
        model = "gemma3:latest"

        # Store entry
        temp_cache.put(text1, text2, model, "response", True, 0.9, "reasoning")

        # Access multiple times
        for _ in range(5):
            temp_cache.get(text1, text2, model)

        # Check access count
        with sqlite3.connect(temp_cache.cache_path) as conn:
            key = temp_cache._generate_key(text1, text2, model, None)
            result = conn.execute(
                "SELECT access_count FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()

            assert result[0] == 6  # 1 initial + 5 gets

    def test_find_similar(self, temp_cache):
        """Test finding similar entries by hash."""
        text1 = "Similar content 1"
        text2 = "Similar content 2"
        model = "gemma3:latest"

        # Store entry
        temp_cache.put(text1, text2, model, "response", True, 0.95, "reasoning")

        # Find by hash (exact match for now)
        result = temp_cache.find_similar(text1, text2, model)

        assert result is not None
        assert result[0] is True
        assert result[1] == 0.95
        assert result[2] == "reasoning"

    def test_clear_old_entries(self, temp_cache):
        """Test cleaning old entries."""
        # Add old and new entries
        with sqlite3.connect(temp_cache.cache_path) as conn:
            old_time = time.time() - (2 * 86400)  # 2 days ago
            new_time = time.time()

            # Old entry
            conn.execute(
                """
                INSERT INTO cache_entries
                (key, page1_hash, page2_hash, response, is_boundary,
                 confidence, reasoning, model_name, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    "old_key",
                    "hash1",
                    "hash2",
                    "response",
                    True,
                    0.9,
                    "reasoning",
                    "model",
                    old_time,
                    old_time,
                ),
            )

            # New entry
            conn.execute(
                """
                INSERT INTO cache_entries
                (key, page1_hash, page2_hash, response, is_boundary,
                 confidence, reasoning, model_name, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    "new_key",
                    "hash3",
                    "hash4",
                    "response",
                    True,
                    0.9,
                    "reasoning",
                    "model",
                    new_time,
                    new_time,
                ),
            )

            conn.commit()

        # Clear old entries
        temp_cache.clear_old_entries()

        # Check results
        stats = temp_cache.get_stats()
        assert stats["total_entries"] == 1  # Only new entry remains

    def test_prompt_version_tracking(self, temp_cache):
        """Test that different prompt versions are cached separately."""
        text1 = "Version test 1"
        text2 = "Version test 2"
        model = "gemma3:latest"

        # Store with version 1
        temp_cache.put(
            text1,
            text2,
            model,
            "response_v1",
            True,
            0.9,
            "reasoning_v1",
            prompt_version="v1",
        )

        # Store with version 2
        temp_cache.put(
            text1,
            text2,
            model,
            "response_v2",
            False,
            0.8,
            "reasoning_v2",
            prompt_version="v2",
        )

        # Retrieve version 1
        result_v1 = temp_cache.get(text1, text2, model, prompt_version="v1")
        assert result_v1[0] is True
        assert result_v1[1] == 0.9

        # Retrieve version 2
        result_v2 = temp_cache.get(text1, text2, model, prompt_version="v2")
        assert result_v2[0] is False
        assert result_v2[1] == 0.8

    def test_error_handling(self, temp_cache):
        """Test graceful error handling."""
        # Close the database to cause errors
        temp_cache.cache_path.unlink()

        # Operations should not raise exceptions
        result = temp_cache.get("text1", "text2", "model")
        assert result is None

        temp_cache.put("text1", "text2", "model", "response", True, 0.9, "reasoning")

        stats = temp_cache.get_stats()
        assert stats["errors"] > 0
