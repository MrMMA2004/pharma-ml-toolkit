"""Input validation guards.

The legacy scripts sprinkled ad-hoc checks (``if col not in df.columns:
raise ValueError``) across every file with mixed-language messages.
These helpers centralise that logic behind typed, English-language,
testable functions.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "validate_positive",
    "validate_smiles_syntax",
    "require_columns",
    "validate_concentration_series",
    "coerce_frame",
    "looks_continuous",
]

# Cheap syntactic sanity check for SMILES: balanced brackets and only
# characters that are legal in openSMILES.  Chemical *validity* still
# requires RDKit -- this is a fast pre-filter used when RDKit is absent.
_SMILES_TOKENS = re.compile(r"^[A-Za-z0-9@\[\]\(\)=+#\-/\.\*\\%$~:']+$")
_BRACKET_PAIRS = {"(": ")", "[": "]"}


def validate_positive(value: float, name: str, allow_zero: bool = False) -> float:
    """Assert ``value`` is a finite, (strictly) positive number.

    Parameters
    ----------
    value:
        Number to check.
    name:
        Human-readable identifier used in the error message.
    allow_zero:
        Accept ``0`` when True (still rejects negatives/NaN/inf).

    Returns
    -------
    float
        The validated value cast to float.

    Raises
    ------
    DataValidationError
        If the value is NaN/inf or out of the allowed range.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise DataValidationError(f"{name} must be finite, got {numeric}")
    if numeric < 0 or (numeric == 0 and not allow_zero):
        raise DataValidationError(f"{name} must be > 0 (or >= 0 if allow_zero), got {numeric}")
    return numeric


def validate_smiles_syntax(smiles: str) -> bool:
    """Return True if ``smiles`` passes lightweight syntax checks.

    A ``False`` result is a definite rejection; a ``True`` result means
    "plausible" -- full validation needs :mod:`rdkit`.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    candidate = smiles.strip()
    if not _SMILES_TOKENS.match(candidate):
        return False
    stack: list[str] = []
    for char in candidate:
        if char in _BRACKET_PAIRS:
            stack.append(_BRACKET_PAIRS[char])
        elif char in (")", "]"):
            if not stack or stack.pop() != char:
                return False
    return not stack


def require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Assert that every name in ``columns`` exists in ``frame``.

    Raises
    ------
    DataValidationError
        Listing *all* missing columns at once (the legacy code checked
        one column at a time and failed confusingly on the second one).
    """
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s) {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def validate_concentration_series(
    times: Sequence[float],
    concentrations: Sequence[float],
    require_positive_conc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a paired time/concentration series for PK math.

    Checks enforced:

    * both sequences are the same, non-empty length (>= 2),
    * all values are finite numbers,
    * times are strictly ascending (duplicate or decreasing sample
      points would silently corrupt trapezoid AUCs),
    * concentrations are non-negative (and positive when
      ``require_positive_conc`` -- log-trapezoids need this).

    Returns
    -------
    tuple of numpy.ndarray
        ``(times, concentrations)`` as sorted float arrays.

    Raises
    ------
    DataValidationError
        With a precise description of the first violated rule.
    """
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)
    if t.shape != c.shape or t.ndim != 1:
        raise DataValidationError(
            "times and concentrations must be 1-D sequences of equal length "
            f"(got {t.shape} vs {c.shape})"
        )
    if t.size < 2:
        raise DataValidationError("At least two sample points are required for PK math")
    if not (np.isfinite(t).all() and np.isfinite(c).all()):
        raise DataValidationError("time/concentration values must all be finite numbers")
    if np.any(t < 0):
        raise DataValidationError("time points must be non-negative")
    if np.any(np.diff(t) <= 0):
        bad = int(np.argmin(np.diff(t)))
        raise DataValidationError(
            f"time points must be strictly ascending; violation at index {bad}"
        )
    if np.any(c < 0):
        raise DataValidationError("concentrations must be non-negative")
    if require_positive_conc and np.any(c == 0):
        raise DataValidationError(
            "concentrations must be positive for log-trapezoid integration; "
            "pass require_positive_conc=False if zeros are intentional"
        )
    return t, c


def looks_continuous(y: Any, max_unique: int = 10) -> bool:
    """Heuristic: does ``y`` look like a numeric (regression) target?

    Stratification only makes sense for class labels, but the splitters
    also see regression targets.  A numeric target whose values are
    mostly unique — more distinct values than ``max(max_unique, n/2)`` —
    is treated as continuous so callers can skip stratification
    *quietly* instead of warning about "classes" with a single member.

    Balanced multiclass tables never trip this rule: with at least two
    members per class the distinct-value count stays at or below
    ``n / 2``.  Non-numeric targets are never continuous.
    """
    arr = np.asarray(y)
    if not np.issubdtype(arr.dtype, np.number):
        return False
    values = arr[~pd.isna(arr)] if arr.size else arr
    n_unique = int(pd.unique(values).size)
    return n_unique > max(max_unique, arr.size // 2)


def coerce_frame(data: Any) -> pd.DataFrame:
    """Coerce common inputs (DataFrame / dict of arrays) to a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        return pd.DataFrame(data)
    try:
        return pd.DataFrame(data)
    except Exception as exc:
        raise DataValidationError(f"Cannot coerce {type(data).__name__} to DataFrame") from exc
