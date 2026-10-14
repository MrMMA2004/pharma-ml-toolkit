"""Centralised logging configuration.

The toolkit never configures the root logger implicitly.  Library code
logs through module-level loggers obtained via :func:`get_logger`;
applications call :func:`configure_logging` once (the CLI does this in
``main()``) to attach a console handler with a readable, timestamped
format, and optionally a rotating file handler.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track which loggers we already configured so repeated calls are no-ops
# (idempotent behaviour keeps notebooks and REPLs pleasant).
_CONFIGURED: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, prefixed with ``pharmaml``.

    Parameters
    ----------
    name:
        Dotted module name (``__name__`` is the right value).  Any
        leading ``pharmaml.`` duplication is stripped.

    Returns
    -------
    logging.Logger
        A logger whose name starts with ``pharmaml.`` so the whole
        toolkit can be silenced/enabled with a single handler.
    """
    if name.startswith("pharmaml."):
        clean = name
    else:
        clean = f"pharmaml.{name.lstrip('.')}"
    return logging.getLogger(clean)


def configure_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """Configure the ``pharmaml`` logger tree exactly once.

    Parameters
    ----------
    level:
        Root threshold for the ``pharmaml`` logger (``logging.DEBUG``
        for verbose runs, ``logging.WARNING`` for quiet pipelines).
    log_file:
        Optional path for a rotating file handler (10 MB chunks, five
        backups kept).  Parent directories are created automatically.
    console:
        Attach a stderr console handler (default ``True``).

    Returns
    -------
    logging.Logger
        The configured ``pharmaml`` logger.
    """
    root = logging.getLogger("pharmaml")
    if "pharmaml" in _CONFIGURED:
        root.setLevel(level)
        return root

    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet duplicate records when the host application also attaches
    # handlers to the root logger.
    root.propagate = False
    _CONFIGURED.add("pharmaml")
    return root
