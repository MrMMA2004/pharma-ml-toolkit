"""Pharmacodynamic models (Emax family, linear, inhibitory).

The Emax model family is the standard dose-response framework of
clinical pharmacology [1]_.

References
----------
.. [1] Holford, N. H. & Sheiner, L. B. "Understanding the
       dose-effect relationship: clinical application of
       pharmacokinetic-pharmacodynamic models." Clin.
       Pharmacokinet. 6 (1981): 429-453.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.utils.validation import validate_positive

__all__ = [
    "emax_effect",
    "sigmoid_emax_effect",
    "inhibitory_emax_effect",
    "linear_effect",
    "ec50_for_target_effect",
    "effect_table",
]


def emax_effect(
    concentration: float | Sequence[float],
    e0: float,
    emax: float,
    ec50: float,
) -> float | np.ndarray:
    """Simple (hyperbolic) Emax model: ``E = E0 + Emax·C/(EC50 + C)``.

    Parameters
    ----------
    concentration:
        Concentration(s); must be non-negative.
    e0:
        Baseline effect without drug.
    emax:
        Maximum achievable effect increment (negative for inhibitory
        direction when combined with a negative sign convention, though
        :func:`inhibitory_emax_effect` is the clearer formulation).
    ec50:
        Concentration producing 50% of Emax (must be > 0).

    Returns
    -------
    float or numpy.ndarray
        Predicted effect, matching the input's scalar/array shape.
    """
    validate_positive(ec50, "ec50")
    conc = np.asarray(concentration, dtype=float)
    if np.any(conc < 0):
        raise DataValidationError("concentrations must be non-negative")
    result = e0 + emax * conc / (ec50 + conc)
    return float(result) if np.ndim(concentration) == 0 else result


def sigmoid_emax_effect(
    concentration: float | Sequence[float],
    e0: float,
    emax: float,
    ec50: float,
    hill: float,
) -> float | np.ndarray:
    """Sigmoid (Hill) Emax model: ``E = E0 + Emax·C^h/(EC50^h + C^h)``.

    ``hill`` (the Hill coefficient) controls curve steepness:
    ``h = 1`` recovers the hyperbolic Emax model, ``h > 1`` produces
    steeper, threshold-like responses.
    """
    validate_positive(ec50, "ec50")
    validate_positive(hill, "hill")
    conc = np.asarray(concentration, dtype=float)
    if np.any(conc < 0):
        raise DataValidationError("concentrations must be non-negative")
    powered = np.power(conc, hill)
    result = e0 + emax * powered / (ec50**hill + powered)
    return float(result) if np.ndim(concentration) == 0 else result


def inhibitory_emax_effect(
    concentration: float | Sequence[float],
    e0: float,
    imax: float,
    ic50: float,
    hill: float = 1.0,
) -> float | np.ndarray:
    """Inhibitory (inverse) Emax model: ``E = E0 − Imax·C^h/(IC50^h + C^h)``.

    The standard PD model for antagonists / enzyme inhibitors where the
    effect *decreases* from baseline toward ``E0 − Imax``.
    """
    validate_positive(ic50, "ic50")
    validate_positive(hill, "hill")
    conc = np.asarray(concentration, dtype=float)
    if np.any(conc < 0):
        raise DataValidationError("concentrations must be non-negative")
    powered = np.power(conc, hill)
    result = e0 - imax * powered / (ic50**hill + powered)
    return float(result) if np.ndim(concentration) == 0 else result


def linear_effect(
    concentration: float | Sequence[float],
    slope: float,
    intercept: float = 0.0,
) -> float | np.ndarray:
    """Linear PD model: ``E = intercept + slope·C`` (small-dose region)."""
    conc = np.asarray(concentration, dtype=float)
    result = intercept + slope * conc
    return float(result) if np.ndim(concentration) == 0 else result


def ec50_for_target_effect(
    target_effect: float,
    ec50: float,
    e0: float,
    emax: float,
    hill: float = 1.0,
) -> float:
    """Inverse Emax: concentration reaching a target effect.

    Inverts ``E = E0 + Emax·C^h/(EC50^h + C^h)`` to solve for ``C``:

    ``C = EC50 · (f/(1−f))^(1/h)``  with  ``f = (E − E0)/Emax``.

    Useful for target-effect dosage design.
    """
    validate_positive(ec50, "ec50")
    validate_positive(hill, "hill")
    if emax == 0:
        raise DataValidationError(
            "emax must be non-zero to invert the Emax relationship "
            "(a zero Emax gives the same effect at every concentration)"
        )
    fraction = (target_effect - e0) / emax
    if not 0.0 < fraction < 1.0:
        raise DataValidationError(
            f"target effect {target_effect} is unreachable: the achievable "
            f"fraction of Emax must lie in (0, 1), got {fraction:.3f}"
        )
    return ec50 * (fraction / (1.0 - fraction)) ** (1.0 / hill)


def effect_table(
    concentrations: Sequence[float],
    e0: float,
    emax: float,
    ec50: float,
    hill: float = 1.0,
) -> dict[str, list[float]]:
    """Evaluate the sigmoid Emax model over a concentration grid.

    Returns a dictionary of parallel lists ``{"concentration": ...,
    "effect": ...}`` ready to be turned into a plotting/report table.
    """
    effects = sigmoid_emax_effect(
        np.asarray(concentrations, dtype=float), e0=e0, emax=emax, ec50=ec50, hill=hill
    )
    return {
        "concentration": [float(c) for c in concentrations],
        "effect": [float(e) for e in np.atleast_1d(effects)],
    }
