"""Structured logging with structlog — JSON-formatted output for RRRIE-CDSS."""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON output and standard library integration."""
    global _configured
    if _configured:
        return

    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard library root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # structlog configuration
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a named structlog logger.

    Args:
        name: Logger name (typically module __name__).

    Returns:
        Bound structlog logger instance.
    """
    if not _configured:
        setup_logging()
    return structlog.get_logger(name)
