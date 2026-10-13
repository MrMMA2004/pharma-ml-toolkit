"""Custom exception hierarchy for PharmaML.

Every error raised intentionally by this package derives from
:class:`PharmaMLError`, so callers can reliably distinguish toolkit
failures from third-party stack traces with a single ``except`` clause.
"""

from __future__ import annotations


class PharmaMLError(Exception):
    """Base class for all intentional PharmaML errors."""


class DataValidationError(PharmaMLError):
    """Raised when input data fails validation (bad columns, NaN-only
    targets, non-monotonic time series, non-positive concentrations, ...)."""


class DependencyError(PharmaMLError):
    """Raised when an optional dependency is required but not installed.

    The message always contains the exact ``pip`` command that installs
    the missing dependency so the failure is immediately actionable.
    """


class UnitConversionError(PharmaMLError):
    """Raised for unknown units or dimensionally invalid conversions."""


class ModelError(PharmaMLError):
    """Raised for unknown model names or invalid model configurations."""


class PersistenceError(PharmaMLError):
    """Raised when saving/loading model artifacts fails."""


class DockingError(PharmaMLError):
    """Raised when a docking preparation or execution step fails."""


class DockingParseError(DockingError):
    """Raised when a docking log/output file cannot be parsed."""
