"""
Persistent caching system for LLM detector responses.

This module provides a SQLite-based cache to store LLM responses across
sessions, significantly improving performance for repeated document processing.
"""

import hashlib
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMResponseCache:
    """
    SQLite-based persistent cache for LLM responses.

    Stores responses with metadata for efficient retrieval and management.
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        max_age_days: int = 30,
        max_size_mb: int = 500,
    ):
        """
        Initialize the cache.

        Args:
            cache_path: Path to SQLite database file
            max_age_days: Maximum age of cache entries in days
            max_size_mb: Maximum cache size in megabytes
        """
        if cache_path is None:
            cache_path = Path.home() / ".cache" / "pdf_splitter" / "llm_cache.db"

        self.cache_path = cache_path
        self.max_age_days = max_age_days
        self.max_size_mb = max_size_mb

        # Ensure cache directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Stats tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
        }

    def _init_database(self):
        """Initialize the SQLite database schema."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    page1_hash TEXT NOT NULL,
                    page2_hash TEXT NOT NULL,
                    response TEXT NOT NULL,
                    is_boundary BOOLEAN NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    model_name TEXT NOT NULL,
                    prompt_version TEXT,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """
            )

            # Create indices for efficient lookups
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON cache_entries(created_at)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_accessed_at
                ON cache_entries(accessed_at)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_hashes
                ON cache_entries(page1_hash, page2_hash)
            """
            )

            # Create metadata table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """
            )

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.cache_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            self.stats["errors"] += 1
            raise
        finally:
            if conn:
                conn.close()

    def get(
        self,
        text1: str,
        text2: str,
        model_name: str,
        prompt_version: Optional[str] = None,
    ) -> Optional[Tuple[bool, float, str]]:
        """
        Retrieve a cached response.

        Args:
            text1: Bottom text from first page
            text2: Top text from second page
            model_name: Name of the LLM model
            prompt_version: Version of the prompt template

        Returns:
            Tuple of (is_boundary, confidence, reasoning) or None if not found
        """
        key = self._generate_key(text1, text2, model_name, prompt_version)

        try:
            with self._get_connection() as conn:
                # Check if entry exists and is not too old
                max_age = time.time() - (self.max_age_days * 86400)

                result = conn.execute(
                    """
                    SELECT is_boundary, confidence, reasoning
                    FROM cache_entries
                    WHERE key = ? AND created_at > ?
                """,
                    (key, max_age),
                ).fetchone()

                if result:
                    # Update access time and count
                    conn.execute(
                        """
                        UPDATE cache_entries
                        SET accessed_at = ?, access_count = access_count + 1
                        WHERE key = ?
                    """,
                        (time.time(), key),
                    )
                    conn.commit()

                    self.stats["hits"] += 1
                    return (
                        bool(result["is_boundary"]),
                        result["confidence"],
                        result["reasoning"],
                    )
                else:
                    self.stats["misses"] += 1
                    return None

        except sqlite3.Error as e:
            logger.error(f"Cache retrieval error: {e}")
            self.stats["errors"] += 1
            return None

    def put(
        self,
        text1: str,
        text2: str,
        model_name: str,
        response: str,
        is_boundary: bool,
        confidence: float,
        reasoning: str,
        prompt_version: Optional[str] = None,
    ):
        """
        Store a response in the cache.

        Args:
            text1: Bottom text from first page
            text2: Top text from second page
            model_name: Name of the LLM model
            response: Raw LLM response
            is_boundary: Whether a boundary was detected
            confidence: Confidence score
            reasoning: Reasoning provided by LLM
            prompt_version: Version of the prompt template
        """
        key = self._generate_key(text1, text2, model_name, prompt_version)
        page1_hash = self._hash_text(text1)
        page2_hash = self._hash_text(text2)

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (key, page1_hash, page2_hash, response, is_boundary,
                     confidence, reasoning, model_name, prompt_version,
                     created_at, accessed_at, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            COALESCE((SELECT access_count FROM cache_entries WHERE key = ?), 0) + 1)
                """,
                    (
                        key,
                        page1_hash,
                        page2_hash,
                        response,
                        is_boundary,
                        confidence,
                        reasoning,
                        model_name,
                        prompt_version,
                        time.time(),
                        time.time(),
                        key,
                    ),
                )
                conn.commit()

                # Check cache size and clean if needed
                self._check_cache_size(conn)

        except sqlite3.Error as e:
            logger.error(f"Cache storage error: {e}")
            self.stats["errors"] += 1

    def _generate_key(
        self,
        text1: str,
        text2: str,
        model_name: str,
        prompt_version: Optional[str] = None,
    ) -> str:
        """Generate a unique cache key."""
        components = [
            self._hash_text(text1),
            self._hash_text(text2),
            model_name,
            prompt_version or "default",
        ]
        return ":".join(components)

    def _hash_text(self, text: str) -> str:
        """Generate a hash for text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _check_cache_size(self, conn: sqlite3.Connection):
        """Check cache size and clean if necessary."""
        # Get current database size
        db_size = self.cache_path.stat().st_size / (1024 * 1024)  # MB

        if db_size > self.max_size_mb:
            logger.info(f"Cache size ({db_size:.1f}MB) exceeds limit, cleaning...")

            # Remove oldest entries (least recently accessed)
            target_entries = int(
                conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0] * 0.7
            )  # Keep 70% of entries

            conn.execute(
                """
                DELETE FROM cache_entries
                WHERE key IN (
                    SELECT key FROM cache_entries
                    ORDER BY accessed_at ASC
                    LIMIT (
                        SELECT COUNT(*) - ? FROM cache_entries
                    )
                )
            """,
                (target_entries,),
            )
            conn.commit()

            # VACUUM in a separate transaction
            conn.close()
            vacuum_conn = sqlite3.connect(self.cache_path)
            vacuum_conn.execute("VACUUM")
            vacuum_conn.close()

    def clear_old_entries(self):
        """Remove entries older than max_age_days."""
        try:
            with self._get_connection() as conn:
                max_age = time.time() - (self.max_age_days * 86400)

                deleted = conn.execute(
                    """
                    DELETE FROM cache_entries
                    WHERE created_at < ?
                """,
                    (max_age,),
                ).rowcount

                if deleted > 0:
                    conn.commit()
                    logger.info(f"Removed {deleted} old cache entries")

                    # VACUUM in a separate transaction
                    conn.close()
                    vacuum_conn = sqlite3.connect(self.cache_path)
                    vacuum_conn.execute("VACUUM")
                    vacuum_conn.close()
                    return

        except sqlite3.Error as e:
            logger.error(f"Error cleaning cache: {e}")

    def clear(self):
        """Clear all cache entries."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
                logger.info("Cache cleared")

                # Reset stats
                self.stats = {
                    "hits": 0,
                    "misses": 0,
                    "errors": 0,
                }
        except sqlite3.Error as e:
            logger.error(f"Error clearing cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.copy()

        try:
            with self._get_connection() as conn:
                # Get additional stats from database
                result = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_entries,
                        AVG(access_count) as avg_access_count,
                        SUM(access_count) as total_accesses
                    FROM cache_entries
                """
                ).fetchone()

                stats.update(
                    {
                        "total_entries": result["total_entries"],
                        "avg_access_count": result["avg_access_count"] or 0,
                        "total_accesses": result["total_accesses"] or 0,
                        "hit_rate": (
                            stats["hits"] / (stats["hits"] + stats["misses"])
                            if (stats["hits"] + stats["misses"]) > 0
                            else 0
                        ),
                        "cache_size_mb": self.cache_path.stat().st_size / (1024 * 1024),
                    }
                )

        except (sqlite3.Error, FileNotFoundError) as e:
            logger.error(f"Error getting cache stats: {e}")

        return stats

    def find_similar(
        self,
        text1: str,
        text2: str,
        model_name: str,
        similarity_threshold: float = 0.9,
    ) -> Optional[Tuple[bool, float, str]]:
        """
        Find similar cached entries using text hashes.

        This is useful for finding responses for slightly different text
        that likely has the same boundary decision.

        Args:
            text1: Bottom text from first page
            text2: Top text from second page
            model_name: Name of the LLM model
            similarity_threshold: Minimum similarity score (not implemented)

        Returns:
            Tuple of (is_boundary, confidence, reasoning) or None
        """
        # For now, just check exact hash matches
        # Future: implement fuzzy matching
        page1_hash = self._hash_text(text1)
        page2_hash = self._hash_text(text2)

        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    """
                    SELECT is_boundary, confidence, reasoning
                    FROM cache_entries
                    WHERE page1_hash = ? AND page2_hash = ? AND model_name = ?
                    ORDER BY accessed_at DESC
                    LIMIT 1
                """,
                    (page1_hash, page2_hash, model_name),
                ).fetchone()

                if result:
                    self.stats["hits"] += 1
                    return (
                        bool(result["is_boundary"]),
                        result["confidence"],
                        result["reasoning"],
                    )
                else:
                    self.stats["misses"] += 1
                    return None

        except sqlite3.Error as e:
            logger.error(f"Similar search error: {e}")
            self.stats["errors"] += 1
            return None
