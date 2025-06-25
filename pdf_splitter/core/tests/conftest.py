"""Shared fixtures for core module tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    env_vars = {}

    def _mock_env(**kwargs):
        env_vars.update(kwargs)
        return patch.dict(os.environ, env_vars)

    return _mock_env


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    import structlog

    structlog.reset_defaults()

    # Reset standard logging
    import logging

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
