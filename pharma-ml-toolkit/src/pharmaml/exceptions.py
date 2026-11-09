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


class JobCancelled(PharmaMLError):
    """Raised when a long batch is stopped at a checkpoint by its caller.

    (round 205) The cooperative-stop channel: every compute loop that
    already reports progress (the feature factory's molecule batches,
    ``recalculate_selected_features``, the progressive table reader, the
    ``compare_models`` / ``benchmark_feature_families`` leaderboards)
    checks the callback's RETURN value -- ``progress(...) is False``
    means "stop at this checkpoint".  Raising this named error (instead
    of a bare ``RuntimeError``) lets the playground's job poller tell a
    user-requested stop apart from a failure: the status bar reads
    "cancelled after 42 s", the window and every loaded input survive.

    A callback that returns ``None`` or any truthy value never cancels,
    which keeps every historical callback (and every test mock) running
    byte-identically.
    """

    def __init__(self, message: str = "job cancelled at a checkpoint") -> None:
        super().__init__(message)


class PersistenceError(PharmaMLError):
    """Raised when saving/loading model artifacts fails."""


class DockingError(PharmaMLError):
    """Raised when a docking preparation or execution step fails."""


class DockingParseError(DockingError):
    """Raised when a docking log/output file cannot be parsed."""


class MDAnalysisError(PharmaMLError):
    """Raised for invalid molecular-dynamics inputs: non-3D coordinate
    arrays, atom-count mismatches between topology and trajectory,
    empty selections, degenerate alignments, unparseable ``.xvg``/``.ndx``
    exports, ligand folders without a single parsable metric, or
    interaction types outside the ProLIF vocabulary."""
