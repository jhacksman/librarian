"""Logging utilities for the ingest module."""

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | Path = "logs",
    log_file: str | None = None,
    verbose: bool = False,
) -> logging.Logger:
    """Set up logging for the ingest module.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        log_file: Optional specific log file name
        verbose: Enable verbose console output

    Returns:
        Configured logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"ingest_{timestamp}.log"

    log_path = log_dir / log_file

    level = getattr(logging, log_level.upper(), logging.INFO)
    if verbose:
        level = logging.DEBUG

    root_logger = logging.getLogger("ingest")
    root_logger.setLevel(level)

    root_logger.handlers.clear()

    console = Console(stderr=True)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=verbose,
        rich_tracebacks=True,
        tracebacks_show_locals=verbose,
    )
    console_handler.setLevel(level)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    root_logger.info(f"Logging initialized. Log file: {log_path}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Name of the module (will be prefixed with 'ingest.')

    Returns:
        Logger instance
    """
    if not name.startswith("ingest."):
        name = f"ingest.{name}"
    return logging.getLogger(name)


class LogContext:
    """Context manager for logging with additional context."""

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        **context: str | int | float,
    ) -> None:
        """Initialize the log context.

        Args:
            logger: Logger instance to use
            operation: Name of the operation being performed
            **context: Additional context key-value pairs
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time: datetime | None = None

    def __enter__(self) -> "LogContext":
        """Enter the context and log the start."""
        self.start_time = datetime.now()
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        self.logger.info(f"Starting {self.operation} [{context_str}]")
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: object,
    ) -> None:
        """Exit the context and log the result."""
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())

        if exc_type is not None:
            self.logger.error(
                f"Failed {self.operation} [{context_str}] after {duration:.2f}s: {exc_val}"
            )
        else:
            self.logger.info(f"Completed {self.operation} [{context_str}] in {duration:.2f}s")

    def log_progress(self, message: str, **extra: str | int | float) -> None:
        """Log progress within the context.

        Args:
            message: Progress message
            **extra: Additional context for this log entry
        """
        all_context = {**self.context, **extra}
        context_str = ", ".join(f"{k}={v}" for k, v in all_context.items())
        self.logger.debug(f"{self.operation}: {message} [{context_str}]")
