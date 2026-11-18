"""Multiple-comparison corrections for a family of p-values.

Whenever the toolkit tests more than one hypothesis at once — one gap
per subgroup (:mod:`pharmaml.ml.subgroups`), one model versus the leader
(:mod:`pharmaml.ml.comparison`) — a fixed α no longer controls the
family-wise error: with ``m`` independent tests the chance of at least
one false positive climbs towards ``1 - (1 - α)^m``.  This module is the
one place that adjusts a family of p-values, so every caller reports the
same three methods:

* ``bonferroni`` — the union bound, ``min(m·p, 1)``.  The strictest and
  the one to quote when *any* false positive is unacceptable.
* ``holm`` — the step-down refinement of Bonferroni: sort ascending,
  scale the k-th smallest by ``m - k``, then enforce monotonicity.  Never
  worse than Bonferroni, and the right default when a strict
  family-wise guarantee is wanted.
* ``fdr_bh`` — Benjamini-Hochberg, which controls the *false discovery
  rate* (the expected share of false positives among the rejected
  hypotheses).  The pragmatic default for screening many candidates,
  where a few false positives are tolerable but power matters.

The adjusted values are *q-values*: monotone in the raw p-values, never
smaller than the raw value, and capped at 1.  A hypothesis is called
significant when its adjusted value clears α, which is the honest
family-wide verdict — a candidate that looked significant alone can be
demoted once its siblings are counted.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.multiplicity import correct_p_values
>>> raw = np.array([0.001, 0.02, 0.04, 0.5])
>>> [round(float(q), 4) for q in correct_p_values(raw, "bonferroni")]
[0.004, 0.08, 0.16, 1.0]
>>> [round(float(q), 4) for q in correct_p_values(raw, "fdr_bh")]
[0.004, 0.04, 0.0533, 0.5]
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import false_discovery_control

from pharmaml.exceptions import DataValidationError

__all__ = [
    "CORRECTION_METHODS",
    "correct_p_values",
]

#: The correction methods :func:`correct_p_values` accepts.
CORRECTION_METHODS: tuple[str, ...] = ("fdr_bh", "bonferroni", "holm")


def correct_p_values(p_values: Any, method: str = "fdr_bh") -> np.ndarray:
    """Adjust a family of p-values for multiplicity (all capped at 1).

    Parameters
    ----------
    p_values:
        The raw p-values, one per hypothesis in the family.
    method:
        ``'fdr_bh'`` (Benjamini-Hochberg), ``'bonferroni'`` or
        ``'holm'``.

    Returns
    -------
    numpy.ndarray
        The adjusted p-values (q-values), the same length and order as
        the input.

    Raises
    ------
    DataValidationError
        On an unknown method, an empty family, or any value outside
        ``[0, 1]`` (a NaN/inf p-value is a caller bug, not a quiet
        exclusion — callers that meet one drop it from the family
        first).

    Examples
    --------
    >>> from pharmaml.ml.multiplicity import correct_p_values
    >>> [round(float(q), 3) for q in correct_p_values([0.01, 0.02, 0.03], "holm")]
    [0.03, 0.04, 0.04]
    """
    if method not in CORRECTION_METHODS:
        raise DataValidationError(
            f"correction must be one of {', '.join(CORRECTION_METHODS)}, got {method!r}"
        )
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        raise DataValidationError("correct_p_values needs at least one p-value")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise DataValidationError(
            "p-values must all lie in [0, 1]; drop non-finite values from the "
            "family before correcting"
        )
    if method == "fdr_bh":
        return np.asarray(false_discovery_control(p, method="bh"), dtype=float)
    m = p.size
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    # Holm step-down: sort ascending, scale the k-th smallest by (m - k),
    # then enforce monotonicity so an adjusted value never falls below a
    # smaller one's.
    order = np.argsort(p)
    ranks = np.empty(m, dtype=float)
    ranks[order] = np.arange(m)
    adjusted = np.minimum(p * (m - ranks), 1.0)
    ordered = np.maximum.accumulate(adjusted[order])
    out = np.empty(m, dtype=float)
    out[order] = ordered
    return out
