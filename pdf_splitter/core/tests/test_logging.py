"""Tests for logging configuration."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import structlog

from pdf_splitter.core.logging import get_logger, setup_logging


class TestLoggingSetup:
    """Test logging setup and configuration."""

    def test_setup_logging_defaults(self):
        """Test default logging setup."""
        # Reset structlog configuration
        structlog.reset_defaults()

        setup_logging()

        # Get a logger to test
        logger = get_logger(__name__)
        assert logger is not None

        # Check that it's a structlog logger
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_setup_logging_with_level(self):
        """Test logging setup with specific level."""
        # Test different log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            structlog.reset_defaults()
            setup_logging(level=level)

            # Standard library logging should respect the level
            assert logging.getLogger().level == getattr(logging, level)

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            structlog.reset_defaults()
            setup_logging(log_file=str(log_file))

            # Log a message
            logger = get_logger(__name__)
            logger.info("Test message", key="value")

            # File should be created
            assert log_file.exists()

            # Check file content
            content = log_file.read_text()
            assert "Test message" in content
            assert "key" in content

    def test_get_logger(self):
        """Test getting logger instances."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        get_logger("module1")  # Would be same instance with caching enabled

        # Different modules get different logger instances
        assert logger1 is not logger2

        # Check that loggers have expected attributes
        assert hasattr(logger1, "info")
        assert hasattr(logger1, "debug")
        assert hasattr(logger1, "error")

    def test_logger_context(self):
        """Test logger with context binding."""
        logger = get_logger(__name__)

        # Bind context
        logger_with_context = logger.bind(user_id=123, request_id="abc")

        # The bound logger should have the context
        assert logger_with_context is not logger


class TestLoggingOutput:
    """Test logging output formatting."""

    @patch("sys.stderr", new_callable=MagicMock)
    def test_structured_output(self, mock_stderr):
        """Test structured log output."""
        structlog.reset_defaults()
        setup_logging(level="INFO")

        logger = get_logger(__name__)
        logger.info("Test event", param1="value1", param2=42)

        # Check that something was written to stderr
        assert mock_stderr.write.called

    def test_json_output(self):
        """Test JSON formatted output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            structlog.reset_defaults()
            setup_logging(log_file=str(log_file), json_logs=True)

            logger = get_logger(__name__)
            logger.info("JSON test", number=123, flag=True)

            # Read and parse JSON
            import json

            content = log_file.read_text().strip()

            # Each line should be valid JSON
            for line in content.split("\n"):
                if line:
                    data = json.loads(line)
                    assert "event" in data
                    assert data.get("event") == "JSON test"
                    assert data.get("number") == 123
                    assert data.get("flag") is True


class TestLoggingIntegration:
    """Test logging integration with application."""

    def test_exception_logging(self):
        """Test exception logging."""
        logger = get_logger(__name__)

        try:
            raise ValueError("Test exception")
        except ValueError:
            # This should not raise
            logger.exception("Error occurred")

    def test_performance_logging(self):
        """Test performance metric logging."""
        logger = get_logger(__name__)

        # Log performance metrics
        logger.info(
            "processing_complete",
            duration_seconds=1.23,
            pages_processed=10,
            pages_per_second=8.13,
        )

    def test_multiline_logging(self):
        """Test logging with multiline content."""
        logger = get_logger(__name__)

        multiline_text = """Line 1
        Line 2
        Line 3"""

        # Should handle multiline content gracefully
        logger.info("Multiline content", content=multiline_text)

    def test_unicode_logging(self):
        """Test logging with unicode content."""
        logger = get_logger(__name__)

        # Test various unicode characters
        logger.info(
            "Unicode test", emoji="🔥", chinese="中文", arabic="العربية", special="α β γ δ"
        )


class TestLoggingConfiguration:
    """Test logging configuration options."""

    def test_colored_output(self):
        """Test colored console output."""
        structlog.reset_defaults()
        setup_logging(level="DEBUG", colored=True)

        logger = get_logger(__name__)
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_timestamp_formats(self):
        """Test different timestamp formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            structlog.reset_defaults()
            setup_logging(log_file=str(log_file))

            logger = get_logger(__name__)
            logger.info("Timestamp test")

            content = log_file.read_text()
            # Should contain ISO format timestamp
            import re

            assert re.search(r"\d{4}-\d{2}-\d{2}", content)
