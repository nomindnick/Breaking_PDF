"""Test configuration for detection module tests."""
import tempfile
from pathlib import Path

import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.llm_detector import LLMDetector


@pytest.fixture
def pdf_config():
    """Provide a test PDF configuration."""
    return PDFConfig()


@pytest.fixture
def test_detector_no_cache(pdf_config):
    """Create an LLM detector with caching disabled for testing."""
    return LLMDetector(config=pdf_config, cache_enabled=False)


@pytest.fixture
def test_detector_with_temp_cache(pdf_config):
    """Create an LLM detector with a temporary cache that's cleared after test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "test_cache.db"
        detector = LLMDetector(
            config=pdf_config, cache_enabled=True, cache_path=cache_path
        )
        yield detector
        # Cache is automatically cleaned up with temp directory


@pytest.fixture(autouse=True)
def clear_default_cache():
    """Clear the default cache before each test to ensure fresh results."""
    # Only clear if detector uses default cache location
    default_cache = Path.home() / ".cache" / "pdf_splitter" / "llm_cache.db"
    if default_cache.exists():
        # Create a temporary detector just to clear the cache
        detector = LLMDetector(cache_enabled=True)
        if detector._cache:
            detector.clear_cache()
