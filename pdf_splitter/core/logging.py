"""Centralized logging configuration for PDF Splitter application."""

import logging
from pathlib import Path
from typing import Optional, Union

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import settings


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    structured: bool = True,
    colored: bool = True,
    json_logs: bool = False,
) -> None:
    """
    Configure application logging with both structured and human-readable formats.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        structured: Whether to use structured logging
        colored: Whether to use colored output (when not using json_logs)
        json_logs: Whether to output logs in JSON format
    """
    log_level = level or settings.log_level

    # Reset structlog configuration to ensure clean state
    structlog.reset_defaults()

    # Configure standard logging level first
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if json_logs or structured:
        # Configure structlog for structured logging
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
            if json_logs
            else structlog.dev.ConsoleRenderer(),
        ]

        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,  # Enable caching for same instances
        )

        # Ensure console output for structured logs if no file specified
        if not log_file and not json_logs:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level))
            root_logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            log_file_path = Path(log_file) if isinstance(log_file, str) else log_file
            file_handler = logging.FileHandler(str(log_file_path))
            file_handler.setLevel(getattr(logging, log_level))
            root_logger.addHandler(file_handler)
    else:
        # Configure standard logging with Rich handler for console output
        console = Console()

        # Console handler with Rich
        console_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=settings.debug,
            markup=True,
        )
        console_handler.setLevel(getattr(logging, log_level))
        root_logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            log_file_path = Path(log_file) if isinstance(log_file, str) else log_file
            file_handler = logging.FileHandler(str(log_file_path))
            file_handler.setLevel(getattr(logging, log_level))
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


# Configure logging on import
# Commented out to avoid interfering with tests
# setup_logging(structured=not settings.debug)
