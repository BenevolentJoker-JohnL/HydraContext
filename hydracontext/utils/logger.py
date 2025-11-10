"""
Structured logging for HydraContext.

Provides consistent logging across all modules with configurable levels.
"""

import logging
import sys
from typing import Optional


# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger(name: str = "hydra_context") -> logging.Logger:
    """
    Get or create logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    global _logger

    if _logger is None:
        _logger = logging.getLogger(name)
        _logger.setLevel(logging.WARNING)  # Default to WARNING

        # Add console handler if none exists
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            _logger.addHandler(handler)

    return _logger


def set_log_level(level: str):
    """
    Set logging level.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = get_logger()
    logger.setLevel(getattr(logging, level.upper()))


def disable_logging():
    """Disable all logging."""
    logger = get_logger()
    logger.setLevel(logging.CRITICAL + 1)


def enable_debug():
    """Enable debug logging."""
    set_log_level("DEBUG")
