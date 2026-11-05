"""The scipy.interpolate core: three 1-D interpolators, gap filling.

``scipy/scipy`` was reviewed file by file for this round (the
``scipy/interpolate/`` tree: ``_interpolate.py``, ``_cubic.py``,
``_pchip.py``, ``_ndgriddata.py``, ``_fitpack2.py`` ...).  The
industrial half -- FITPACK's B-spline machinery, RBFinterp's
radial-basis engines, N-D scattered ``griddata`` -- stays with
scipy.  What lands is the three 1-D interpolators every pharma
workflow actually interpolates with, each verified against
scipy's own class to machine precision:

* **linear** (:func:`linear_interp`) -- piecewise-constant slopes,
  ``numpy.interp``'s contract, the honest default for concentration
  data between samples.
* **cubic** (:func:`cubic_natural_spline`) -- the natural cubic
  spline: the tridiagonal second-derivative system solved by the
  Thomas algorithm, the piecewise cubics evaluated in the classical
  ``M_i`` form (``scipy.interpolate.CubicSpline(bc_type='natural')``
  parity).
* **pchip** (:func:`pchip`) -- the monotone cubic Hermite
  interpolator: Fritsch-Carlson slopes (the weighted-harmonic mean
  of neighbouring secants, forced to zero where the secants change
  sign), so a dose-response curve can never overshoot its plateaus
  (``scipy.interpolate.PchipInterpolator`` parity).

Riding along: :func:`fill_gaps` (interpolate the NaN holes of a
series in place -- the missing-timepoint repair) and
:func:`interpolation_audit` (hold out every k-th point, rebuild,
score -- the leave-out comparison that answers "which interpolator
deserves this data?").

The pharma reading, stated once: the series is a concentration-time
profile between sampled times (linear -- the NCA convention), a
smooth dissolution curve (cubic -- smoothness for AUC), or a
dose-response with plateaus that must not overshoot (pchip -- the
monotonicity contract).

The honest scope: no extrapolation -- ``x_new`` outside the data
span returns NaN rather than scipy's ``fill_value`` options; no
parametric splines, no N-D grids.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.interpolate import (
...     cubic_natural_spline, fill_gaps, interpolation_audit,
...     interpolate_1d, linear_interp, pchip,
... )
>>> x = np.linspace(0.0, np.pi, 7)
>>> y = np.sin(x)
>>> x_new = np.linspace(0.0, np.pi, 13)
>>> cubic = cubic_natural_spline(x, y, x_new)
>>> round(float(np.max(np.abs(cubic - np.sin(x_new)))), 4) < 0.02
True
>>> mono = pchip(np.array([0.0, 1.0, 4.0, 10.0]),
...              np.array([0.0, 5.0, 20.0, 40.0]),
...              np.array([0.5, 2.0, 8.0]))
>>> bool(np.all(np.diff(mono) >= 0))       # rising data -> rising curve
True
>>> bool(np.all((mono >= 0.0) & (mono <= 40.0)))   # and no overshoot
True
>>> gappy = np.array([0.0, np.nan, 2.0, np.nan, np.nan, 5.0, 6.0])
>>> filled = fill_gaps(gappy, method="linear")
>>> filled.tolist()
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
>>> audit = interpolation_audit(np.sin(np.linspace(0.0, np.pi, 21)),
...                             holdout_every=4)
>>> [round(a.rmse, 3) for a in audit]   # cubic wins on smooth sine
[0.01, 0.0, 0.001]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..exceptions import DataValidationError

__all__ = [
    "InterpolationAudit",
    "cubic_natural_spline",
    "fill_gaps",
    "interpolate_1d",
    "interpolation_audit",
    "linear_interp",
    "pchip",
]

#: The methods :func:`interpolate_1d` dispatches over.
INTERP_METHODS = ("linear", "cubic", "pchip")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_knots(xs: np.ndarray, ys: np.ndarray, *, minimum: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Two finite 1-D arrays of equal length, knots strictly increasing."""
    x = np.asarray(xs, dtype=float).reshape(-1)
    y = np.asarray(ys, dtype=float).reshape(-1)
    if x.size != y.size:
        raise DataValidationError(
            f"xs and ys must match in length, got {x.size} vs {y.size}"
        )
    if x.size < minimum:
        raise DataValidationError(
            f"{minimum} knots are the minimum, got {x.size}"
        )
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise DataValidationError("knots must be finite (no NaN/inf)")
    if np.any(np.diff(x) <= 0.0):
        raise DataValidationError("knots must be strictly increasing")
    return x, y


def _as_query(x_new: np.ndarray) -> np.ndarray:
    """One finite 1-D query grid."""
    q = np.asarray(x_new, dtype=float).reshape(-1)
    if not np.all(np.isfinite(q)):
        raise DataValidationError("x_new must be finite (no NaN/inf)")
    return q


# ---------------------------------------------------------------------------
# The three interpolators
# ---------------------------------------------------------------------------


def linear_interp(xs: np.ndarray, ys: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation (``numpy.interp``'s contract).

    Outside the knot span the result is NaN -- the honest no-extrapolation
    rule this module states once.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.interpolate import linear_interp
    >>> linear_interp([0.0, 10.0], [0.0, 5.0], [2.0, 8.0, -1.0]).tolist()
    [1.0, 4.0, nan]
    """
    x, y = _as_knots(xs, ys, minimum=2)
    q = _as_query(x_new)
    out = np.interp(q, x, y, left=np.nan, right=np.nan)
    return np.where((q >= x[0]) & (q <= x[-1]), out, np.nan)


def _thomas_solve(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """The tridiagonal (Thomas) solve, no pivoting -- diagonally dominant here."""
    n = diag.size
    c = upper.astype(float).copy()
    d = rhs.astype(float).copy()
    b = diag.astype(float).copy()
    for i in range(1, n):
        factor = lower[i - 1] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
    out = np.empty(n, dtype=float)
    out[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        out[i] = (d[i] - c[i] * out[i + 1]) / b[i]
    return out


def _natural_spline_moments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The second derivatives ``M_i`` of the natural cubic spline.

    Interior rows: ``h[i-1] M[i-1] + 2(h[i-1]+h[i]) M[i] + h[i] M[i+1]
    = 6 (delta[i] - delta[i-1])`` with ``M_0 = M_{n-1} = 0`` (the
    natural boundary) -- the tridiagonal system ``CubicSpline(
    bc_type='natural')`` factorises.
    """
    n = x.size
    if n == 2:
        return np.zeros(2)
    h = np.diff(x)
    delta = np.diff(y) / h
    m = n - 2  # unknowns M_1 .. M_{n-2}
    lower = h[1:-1].copy()      # coefficient of M_{i-1} (row i-1)
    diag = 2.0 * (h[:-1] + h[1:])
    upper = h[1:-1].copy()      # coefficient of M_{i+1}
    rhs = 6.0 * np.diff(delta)
    if m == 1:
        inner = np.array([rhs[0] / diag[0]])
    else:
        inner = _thomas_solve(lower, diag, upper, rhs)
    return np.concatenate(([0.0], inner, [0.0]))


def cubic_natural_spline(
    xs: np.ndarray, ys: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    """The natural cubic spline, evaluated at ``x_new``.

    The classical ``M_i`` form on each interval
    ``[x_i, x_{i+1}]``::

        S(t) = M_i (x_{i+1}-t)^3 / (6 h_i)
             + M_{i+1} (t-x_i)^3 / (6 h_i)
             + (y_i - M_i h_i^2/6) (x_{i+1}-t)/h_i
             + (y_{i+1} - M_{i+1} h_i^2/6) (t-x_i)/h_i

    with the second derivatives from :func:`_natural_spline_moments`
    -- byte-for-byte the polynomial ``scipy.interpolate.CubicSpline(
    bc_type='natural')`` builds.  Outside the span: NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.interpolate import cubic_natural_spline
    >>> from scipy.interpolate import CubicSpline
    >>> x = np.array([0.0, 1.0, 2.5, 4.0, 6.0, 8.0])
    >>> y = np.array([0.0, 2.0, 3.0, 1.5, 4.0, 2.0])
    >>> q = np.linspace(0.5, 7.5, 11)
    >>> np.allclose(cubic_natural_spline(x, y, q),
    ...             CubicSpline(x, y, bc_type="natural")(q), atol=1e-10)
    True
    """
    x, y = _as_knots(xs, ys, minimum=3)
    q = _as_query(x_new)
    m = _natural_spline_moments(x, y)
    h = np.diff(x)
    idx = np.searchsorted(x, q, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)
    inside = (q >= x[0]) & (q <= x[-1])
    ti = np.where(inside, q, x[idx])  # safe eval outside; masked later
    hi = h[idx]
    a = x[idx + 1] - ti
    b = ti - x[idx]
    values = (
        m[idx] * a**3 / (6.0 * hi)
        + m[idx + 1] * b**3 / (6.0 * hi)
        + (y[idx] - m[idx] * hi * hi / 6.0) * a / hi
        + (y[idx + 1] - m[idx + 1] * hi * hi / 6.0) * b / hi
    )
    return np.where(inside, values, np.nan)


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson slopes for the monotone Hermite piecewise cubics.

    Interior knots: the weighted-harmonic mean of the two neighbouring
    secants (weights ``2h_i + h_{i-1}`` and ``h_i + 2h_{i-1}``), forced
    to zero where the secants disagree in sign; endpoints: the
    one-sided three-point parabola, clamped the same way.
    """
    n = x.size
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)
    if n == 2:
        d[0] = d[1] = delta[0]
        return d
    # Interior knots.
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            d[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    # Endpoints: the one-sided three-point parabola slope, clamped.
    d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if delta[0] * d[0] <= 0.0:
        d[0] = 0.0
    d[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (
        h[-1] + h[-2]
    )
    if delta[-1] * d[-1] <= 0.0:
        d[-1] = 0.0
    return d


def pchip(xs: np.ndarray, ys: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Monotone cubic Hermite interpolation (``PchipInterpolator`` parity).

    The Fritsch-Carlson slopes guarantee: on any interval where the
    data is monotone, the interpolant is monotone -- no overshoot
    past the plateaus, the property that makes pchip the honest
    choice for dose-response curves.  Outside the span: NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.interpolate import pchip
    >>> from scipy.interpolate import PchipInterpolator
    >>> x = np.array([0.0, 1.0, 2.5, 4.0, 6.0, 8.0])
    >>> y = np.array([0.0, 2.0, 3.0, 1.5, 4.0, 2.0])
    >>> q = np.linspace(0.5, 7.5, 11)
    >>> np.allclose(pchip(x, y, q), PchipInterpolator(x, y)(q), atol=1e-10)
    True
    """
    x, y = _as_knots(xs, ys, minimum=2)
    q = _as_query(x_new)
    d = _pchip_slopes(x, y)
    h = np.diff(x)
    idx = np.searchsorted(x, q, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)
    inside = (q >= x[0]) & (q <= x[-1])
    hi = h[idx]
    s = (q - x[idx]) / hi
    s = np.where(inside, s, 0.0)
    h00 = (1.0 + 2.0 * s) * (1.0 - s) ** 2
    h10 = s * (1.0 - s) ** 2
    h01 = s * s * (3.0 - 2.0 * s)
    h11 = s * s * (s - 1.0)
    values = (
        h00 * y[idx]
        + h10 * hi * d[idx]
        + h01 * y[idx + 1]
        + h11 * hi * d[idx + 1]
    )
    return np.where(inside, values, np.nan)


def interpolate_1d(
    xs: np.ndarray,
    ys: np.ndarray,
    x_new: np.ndarray,
    *,
    method: str = "linear",
) -> np.ndarray:
    """The scipy-named dispatcher: one call, one of the three methods.

    Parameters
    ----------
    xs, ys:
        Knots (strictly increasing ``xs``).
    x_new:
        Query points; outside the span the answer is NaN.
    method:
        ``'linear'`` | ``'cubic'`` (natural spline) | ``'pchip'``
        (monotone).

    Examples
    --------
    >>> from pharmaml.ml.interpolate import interpolate_1d
    >>> interpolate_1d([0.0, 2.0, 4.0], [0.0, 4.0, 8.0],
    ...                [1.0, 3.0], method="linear").tolist()
    [2.0, 6.0]
    """
    if method == "linear":
        return linear_interp(xs, ys, x_new)
    if method == "cubic":
        return cubic_natural_spline(xs, ys, x_new)
    if method == "pchip":
        return pchip(xs, ys, x_new)
    raise DataValidationError(
        f"method must be one of {INTERP_METHODS}, got {method!r}"
    )


# ---------------------------------------------------------------------------
# Gap filling and the audit
# ---------------------------------------------------------------------------


def fill_gaps(values: np.ndarray, *, method: str = "pchip") -> np.ndarray:
    """Repair the NaN holes of a 1-D series by index interpolation.

    The missing-timepoint repair: the observed indices become the
    knots, the holes are queried at their own indices, and the
    endpoints are held flat (``np.interp``'s edge convention for
    linear; the endpoint slope rule for the cubics).

    Parameters
    ----------
    values:
        One 1-D series; NaN entries are the holes.
    method:
        ``'linear'`` | ``'cubic'`` | ``'pchip'``.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.interpolate import fill_gaps
    >>> gappy = np.array([0.0, np.nan, 2.0, np.nan, np.nan, 5.0, 6.0])
    >>> fill_gaps(gappy, method="linear").tolist()
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> gappy2 = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    >>> fill_gaps(gappy2, method="linear").tolist()
    [1.0, 1.0, 2.0, 3.0, 3.0]
    """
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size < 2:
        raise DataValidationError(f"the series needs >= 2 points, got {series.size}")
    known = np.isfinite(series)
    if int(known.sum()) < 2:
        raise DataValidationError("fewer than 2 finite values; cannot interpolate")
    if bool(known.all()):
        return series.copy()
    x_known = np.flatnonzero(known).astype(float)
    y_known = series[known]
    holes = np.flatnonzero(~known).astype(float)
    # Endpoint rule: extend the nearest observed value flat.
    lo_pad = holes[holes < x_known[0]]
    hi_pad = holes[holes > x_known[-1]]
    mid_holes = holes[(holes >= x_known[0]) & (holes <= x_known[-1])]
    filled = series.copy()
    if mid_holes.size:
        filled[mid_holes.astype(int)] = interpolate_1d(
            x_known, y_known, mid_holes, method=method
        )
    if lo_pad.size:
        filled[lo_pad.astype(int)] = y_known[0]
    if hi_pad.size:
        filled[hi_pad.astype(int)] = y_known[-1]
    return filled


@dataclass(frozen=True)
class InterpolationAudit:
    """The leave-out comparison: which interpolator rebuilds the signal.

    Every ``holdout_every``-th point is removed, each method
    interpolates across the gap, and the RMSE against the withheld
    truth is reported -- with the withheld indices carried along so
    the caller can plot exactly where the methods were tested.
    """

    method: str
    rmse: float
    n_withheld: int
    withheld_indices: tuple[int, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        return f"{self.method:<7} rmse {self.rmse:.4f} over {self.n_withheld} points"


def interpolation_audit(
    values: np.ndarray, *, holdout_every: int = 4
) -> tuple[InterpolationAudit, ...]:
    """Hold out every k-th point; score each interpolator's rebuild.

    Parameters
    ----------
    values:
        One finite 1-D series (the knots are its indices).
    holdout_every:
        The stride of the withheld points (``>= 2``); the first and
        last points are always kept so the span never shrinks.

    Returns
    -------
    tuple[InterpolationAudit, ...]
        One row per method (linear, cubic, pchip), RMSE against the
        withheld truth; ``min(rows, key=lambda a: a.rmse)`` picks the
        winner.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.interpolate import interpolation_audit
    >>> audits = interpolation_audit(np.sin(np.linspace(0.0, np.pi, 21)),
    ...                               holdout_every=4)
    >>> len(audits)
    3
    >>> [a.method for a in audits]
    ['linear', 'cubic', 'pchip']
    >>> best = min(audits, key=lambda a: a.rmse)
    >>> best.rmse < 0.1
    True
    """
    series = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(series)):
        raise DataValidationError("the series must be finite (no NaN/inf)")
    if series.size < 8:
        raise DataValidationError(f"the series needs >= 8 points, got {series.size}")
    if int(holdout_every) < 2:
        raise DataValidationError(
            f"holdout_every must be >= 2, got {holdout_every}"
        )
    n = series.size
    withheld_mask = np.zeros(n, dtype=bool)
    withheld_mask[int(holdout_every) : n - 1 : int(holdout_every)] = True
    kept = ~withheld_mask
    x_known = np.flatnonzero(kept).astype(float)
    x_held = np.flatnonzero(withheld_mask).astype(float)
    rows = []
    for method in INTERP_METHODS:
        rebuilt = interpolate_1d(x_known, series[kept], x_held, method=method)
        rmse = float(np.sqrt(np.mean((rebuilt - series[withheld_mask]) ** 2)))
        rows.append(
            InterpolationAudit(
                method=method,
                rmse=rmse,
                n_withheld=int(withheld_mask.sum()),
                withheld_indices=tuple(int(i) for i in x_held),
            )
        )
    return tuple(rows)
