"""Centralized logging configuration for PDF Splitter application."""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    structured: bool = True
) -> None:
    """
    Configure application logging with both structured and human-readable formats.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        structured: Whether to use structured logging
    """
    level = log_level or settings.log_level
    
    if structured:
        # Configure structlog for structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Configure standard logging with Rich handler for console output
        console = Console()
        
        # Remove existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler with Rich
        console_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=settings.debug,
            markup=True
        )
        console_handler.setLevel(getattr(logging, level))
        
        # Configure format
        logging.basicConfig(
            level=getattr(logging, level),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[console_handler]
        )
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level))
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
setup_logging(structured=not settings.debug)