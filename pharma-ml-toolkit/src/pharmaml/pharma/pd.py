"""Pharmacodynamic models (Emax family, linear, inhibitory).

The Emax model family is the standard dose-response framework of
clinical pharmacology [1]_.

References
----------
.. [1] Holford, N. H. & Sheiner, L. B. "Understanding the
       dose-effect relationship: clinical application of
       pharmacokinetic-pharmacodynamic models." Clin.
       Pharmacokinet. 6 (1981): 429-453.

Examples
--------
>>> from pharmaml.pharma.pd import emax_effect, inhibitory_emax_effect
>>> emax_effect(0.0, e0=10.0, emax=5.0, ec50=2.0)     # C = 0 -> baseline
10.0
>>> emax_effect(2.0, e0=10.0, emax=5.0, ec50=2.0)     # C = EC50 -> E0 + Emax/2
12.5
>>> emax_effect([0.0, 2.0, 18.0], e0=10.0, emax=5.0, ec50=2.0)  # approaching 15
array([10. , 12.5, 14.5])
>>> inhibitory_emax_effect([0.0, 5.0], e0=100.0, imax=40.0, ic50=5.0)  # C=IC50
array([100.,  80.])

>>> from pharmaml.pharma.pd import linear_effect, sigmoid_emax_effect
>>> sigmoid_emax_effect(2.0, e0=10.0, emax=5.0, ec50=2.0, hill=3.0)  # EC50 pivot
12.5
>>> sigmoid_emax_effect(1.0, e0=10.0, emax=5.0, ec50=2.0, hill=2.0)  # steeper
11.0
>>> sigmoid_emax_effect(1.0, 10.0, 5.0, 2.0, hill=1.0) == emax_effect(1.0, 10.0, 5.0, 2.0)
True
>>> linear_effect([0.0, 1.0], slope=2.0)
array([0., 2.])

>>> from pharmaml.pharma.pd import ec50_for_target_effect, effect_table
>>> ec50_for_target_effect(12.5, ec50=2.0, e0=10.0, emax=5.0)  # inverse of Emax
2.0
>>> round(ec50_for_target_effect(14.0, ec50=2.0, e0=10.0, emax=5.0), 9)  # f = 0.8
8.0
>>> ec50_for_target_effect(10.0, ec50=2.0, e0=10.0, emax=5.0)   # E0 at C = 0
0.0
>>> effect_table([0.0, 2.0, 8.0], e0=10.0, emax=5.0, ec50=2.0)
{'concentration': [0.0, 2.0, 8.0], 'effect': [10.0, 12.5, 14.0]}
>>> emax_effect(-1.0, e0=10.0, emax=5.0, ec50=2.0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: concentrations must be non-negative
>>> emax_effect(float("nan"), e0=10.0, emax=5.0, ec50=2.0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: concentrations must be finite numbers (NaN/inf rejected)
>>> emax_effect(1.0, e0=10.0, emax=5.0, ec50=0.0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: ec50 must be > 0 (or >= 0 if allow_zero), got 0.0
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.utils.validation import validate_positive

__all__ = [
    "ec50_for_target_effect",
    "effect_table",
    "emax_effect",
    "inhibitory_emax_effect",
    "linear_effect",
    "sigmoid_emax_effect",
]


def _validated_concentration(
    concentration: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Asarray + domain checks shared by the Emax model family.

    Concentrations must be finite and non-negative; NaN or infinite
    values would otherwise flow through the formulas as silent NaN
    effects (``inf`` even evaluates to ``inf/inf = nan``).
    """
    conc = np.asarray(concentration, dtype=float)
    if not np.isfinite(conc).all():
        raise DataValidationError(
            "concentrations must be finite numbers (NaN/inf rejected)"
        )
    if np.any(conc < 0):
        raise DataValidationError("concentrations must be non-negative")
    return conc


def emax_effect(
    concentration: float | Sequence[float] | np.ndarray,
    e0: float,
    emax: float,
    ec50: float,
) -> float | np.ndarray:
    """Simple (hyperbolic) Emax model: ``E = E0 + Emax·C/(EC50 + C)``.

    Parameters
    ----------
    concentration:
        Concentration(s); must be non-negative and finite (NaN/inf
        values are rejected with a DataValidationError instead of
        silently returning NaN effects).
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
    conc = _validated_concentration(concentration)
    result = e0 + emax * conc / (ec50 + conc)
    return float(result) if np.ndim(concentration) == 0 else result


def sigmoid_emax_effect(
    concentration: float | Sequence[float] | np.ndarray,
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
    conc = _validated_concentration(concentration)
    powered = np.power(conc, hill)
    result = e0 + emax * powered / (ec50**hill + powered)
    return float(result) if np.ndim(concentration) == 0 else result


def inhibitory_emax_effect(
    concentration: float | Sequence[float] | np.ndarray,
    e0: float,
    imax: float,
    ic50: float,
    hill: float = 1.0,
) -> float | np.ndarray:
    """Inhibitory (inverse) Emax model: ``E = E0 − Imax·C^h/(IC50^h + C^h)``.

    The standard PD model for antagonists / enzyme inhibitors where the
    effect *decreases* from baseline toward ``E0 − Imax``.  As with the
    excitatory models, concentrations must be finite and non-negative.
    """
    validate_positive(ic50, "ic50")
    validate_positive(hill, "hill")
    conc = _validated_concentration(concentration)
    powered = np.power(conc, hill)
    result = e0 - imax * powered / (ic50**hill + powered)
    return float(result) if np.ndim(concentration) == 0 else result


def linear_effect(
    concentration: float | Sequence[float] | np.ndarray,
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

    Useful for target-effect dosage design.  ``f = 0`` (a target equal
    to the baseline ``E0``) is reached at ``C = 0``; ``f → 1`` needs
    ``C → ∞`` and is rejected.
    """
    validate_positive(ec50, "ec50")
    validate_positive(hill, "hill")
    if emax == 0:
        raise DataValidationError(
            "emax must be non-zero to invert the Emax relationship "
            "(a zero Emax gives the same effect at every concentration)"
        )
    fraction = (target_effect - e0) / emax
    if not 0.0 <= fraction < 1.0:
        raise DataValidationError(
            f"target effect {target_effect} is unreachable: the achievable "
            f"fraction of Emax must lie in [0, 1), got {fraction:.3f}"
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
