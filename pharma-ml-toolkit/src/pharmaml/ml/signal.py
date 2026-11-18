"""The scipy.signal core: peak finding and Savitzky-Golay smoothing.

``scipy/scipy`` was reviewed file by file for this round (the
``scipy/signal/`` tree: ``_peak_finding.py``, ``_savitzky_golay.py``,
``_filter_design.py``, ``_spectral_py.py`` ...).  The subpackage's
industrial half -- FIR/IIR filter design (``butter``, ``firwin``, the
bilinear transform), resampling polyphase banks, STFT machinery --
is C-backed signal engineering the toolkit deliberately leaves with
scipy.  What lands is the pair of closed-form algorithms that
scientific Python actually reaches for daily, both verified against
scipy's own outputs:

* :func:`find_peaks` -- local-maximum detection with the full
  prominence bookkeeping (the walk-out-to-higher-terrain algorithm
  ``scipy.signal._peak_finding_utils`` runs in C), the height and
  minimum-distance filters, plateau resolution to the
  middle-biased-left index, and half-height widths (``peak_widths``
  folded in -- the FWHM number every chromatography report quotes).
* :func:`savgol_filter` -- Savitzky-Golay smoothing: the local
  polynomial fit whose coefficients are one row of the Vandermonde
  pseudo-inverse, edges handled by scipy's ``interp`` convention
  (the same polynomial evaluated at the edge offsets), and the
  ``deriv`` estimate (the k-th derivative is ``k!·c_k`` of the same
  fit -- the cheap derivative engine ``savgol_filter(deriv=1)`` is
  normally used for).

The pharma reading, stated once: the signal is a chromatogram (an
LC-UV/LC-MS trace -- peak picking with prominence is *the* purity
workflow), a plate-reader spectrum (Savitzky-Golay is the smoothing
the instrument vendors ship), or a dissolution track -- and "how
many peaks, how tall, how wide" is the whole question.

The honest scope: no filter design, no polyphase resampling, no
spectrogram -- and :func:`find_peaks` resolves plateaus the way
scipy does (contiguous equal maxima collapse to one peak at the
middle, biased left) rather than offering ``plateau_size``
arbitration.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.signal import (
...     find_peaks, peak_table, savgol_filter, savgol_coeffs,
...     synthetic_chromatogram,
... )
>>> trace = synthetic_chromatogram(200, seed=42)
>>> result = find_peaks(trace, prominence=0.35)
>>> result.indices.size
3
>>> result.widths.shape == result.indices.shape   # FWHM, sample units
True
>>> noisy = np.sin(np.linspace(0.0, 3.0, 60)) + 0.25
>>> rng = np.random.default_rng(0)
>>> noisy = noisy + rng.normal(0.0, 0.25, noisy.size)
>>> smoothed = savgol_filter(noisy, 11, 2)
>>> bool(np.std(smoothed - np.sin(np.linspace(0.0, 3.0, 60))) <
...      np.std(noisy - np.sin(np.linspace(0.0, 3.0, 60))))
True
>>> float(np.max(np.abs(savgol_filter(np.arange(10.0) ** 2, 5, 2)
...                       - np.arange(10.0) ** 2))) < 1e-9
True
"""

from __future__ import annotations

import math

import numpy as np

from ..exceptions import DataValidationError

__all__ = [
    "PeakResult",
    "find_peaks",
    "peak_table",
    "savgol_coeffs",
    "savgol_filter",
    "synthetic_chromatogram",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_trace(values: np.ndarray, *, minimum: int = 5) -> np.ndarray:
    """One finite 1-D float trace, validated."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < minimum:
        raise DataValidationError(
            f"the trace needs at least {minimum} points, got {array.size}"
        )
    if not np.all(np.isfinite(array)):
        raise DataValidationError("the trace must be finite (no NaN/inf)")
    return array


def _resolve_bound(bound: object, name: str) -> tuple[float, float] | None:
    """A height/prominence bound: scalar floor or (min, max) pair."""
    if bound is None:
        return None
    if isinstance(bound, (int, float, np.floating, np.integer)):
        value = float(bound)  # type: ignore[arg-type]
        if not np.isfinite(value):
            raise DataValidationError(f"{name} must be finite, got {value}")
        return (value, np.inf)
    try:
        pair: tuple[float, float] = tuple(bound)  # type: ignore[arg-type, assignment]
    except TypeError as exc:
        raise DataValidationError(
            f"{name} must be a number or a (min, max) pair"
        ) from exc
    if len(pair) != 2:
        raise DataValidationError(
            f"{name} must be a number or a (min, max) pair"
        )
    low_f, high_f = float(pair[0]), float(pair[1])
    if low_f > high_f:
        raise DataValidationError(
            f"{name} lower bound {low_f} exceeds upper bound {high_f}"
        )
    return (low_f, high_f)


# ---------------------------------------------------------------------------
# Peak finding (scipy.signal.find_peaks + peak_widths, folded together)
# ---------------------------------------------------------------------------


class PeakResult:
    """The peak table one :func:`find_peaks` call produces.

    Attributes mirror scipy's property dict (``prominences``,
    ``left_bases``, ``right_bases`` -- the indices of the contour
    saddles) with ``peak_widths`` folded in: ``widths`` is the FWHM
    at ``rel_height`` 0.5, ``left_ips``/``right_ips`` the
    linearly-interpolated half-height crossings.
    """

    __slots__ = (
        "filters", "heights", "indices", "left_bases", "left_ips",
        "prominences", "right_bases", "right_ips", "widths",
    )

    def __init__(
        self,
        indices: np.ndarray,
        heights: np.ndarray,
        prominences: np.ndarray,
        left_bases: np.ndarray,
        right_bases: np.ndarray,
        widths: np.ndarray,
        left_ips: np.ndarray,
        right_ips: np.ndarray,
        filters: dict[str, object],
    ) -> None:
        self.indices = indices
        self.heights = heights
        self.prominences = prominences
        self.left_bases = left_bases
        self.right_bases = right_bases
        self.widths = widths
        self.left_ips = left_ips
        self.right_ips = right_ips
        self.filters = filters

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def fwhm(self) -> np.ndarray:
        """The full width at half maximum (scipy's ``widths``)."""
        return self.widths

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"PeakResult(n={self.indices.size}, "
            f"prominences={np.round(self.prominences, 4).tolist()})"
        )


def _local_maxima(x: np.ndarray) -> np.ndarray:
    """Indices of local maxima, plateaus resolved scipy's way.

    A strict rise followed by a strict fall is a peak; a rise into a
    plateau of equal values collapses to one peak at
    ``start + (end - start) // 2`` -- the middle, biased left, the
    rule ``scipy.signal._peak_finding._local_maxima_1d`` applies.
    """
    peaks: list[int] = []
    n = x.size
    i = 1
    while i < n - 1:
        if x[i] <= x[i - 1]:
            i += 1
            continue
        if x[i] > x[i + 1]:
            peaks.append(i)
            i += 1
            continue
        if x[i] < x[i + 1]:
            i += 1
            continue
        # x[i] > x[i-1] and x[i] == x[i+1]: a plateau starts here.
        j = i + 1
        while j < n - 1 and x[j] == x[j + 1]:
            j += 1
        if x[j] > x[j + 1]:
            peaks.append(i + (j - i) // 2)
        i = j + 1
    return np.asarray(peaks, dtype=int)


def _prominence(
    x: np.ndarray, peaks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The walk-out-to-higher-terrain algorithm, per peak.

    Left: walk from the peak outward while values stay ``<=`` the
    peak, tracking the running minimum; stop at the first strictly
    higher value or the boundary.  Right: symmetric.  The contour
    line is the higher of the two side minima, so prominence is
    ``peak - max(left_min, right_min)`` -- exactly scipy's
    ``_peak_finding_utils._calc_prominences``.
    """
    n = x.size
    prominences = np.empty(peaks.size, dtype=float)
    left_bases = np.empty(peaks.size, dtype=int)
    right_bases = np.empty(peaks.size, dtype=int)
    for slot, peak in enumerate(peaks):
        value = x[peak]
        # Left walk.
        left_min, left_base = x[peak], peak
        i = peak - 1
        while i >= 0 and x[i] <= value:
            if x[i] < left_min:
                left_min, left_base = x[i], i
            i -= 1
        # Right walk.
        right_min, right_base = x[peak], peak
        j = peak + 1
        while j < n and x[j] <= value:
            if x[j] < right_min:
                right_min, right_base = x[j], j
            j += 1
        prominences[slot] = value - max(left_min, right_min)
        left_bases[slot], right_bases[slot] = left_base, right_base
    return prominences, left_bases, right_bases


def _half_height_crossings(
    x: np.ndarray,
    peaks: np.ndarray,
    prominences: np.ndarray,
    left_bases: np.ndarray,
    right_bases: np.ndarray,
    *,
    rel_height: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolated crossings at ``height - rel_height*prominence``.

    The search is bounded by the prominence bases (scipy's
    ``peak_widths`` window); a crossing is linearly interpolated
    between the straddling samples, and a side with no in-window
    crossing falls back to its base index (the honest clamp: the
    width is then a lower bound).
    """
    left_ips = np.empty(peaks.size, dtype=float)
    right_ips = np.empty(peaks.size, dtype=float)
    for slot, peak in enumerate(peaks):
        level = x[peak] - rel_height * prominences[slot]
        lo, hi = int(left_bases[slot]), int(right_bases[slot])
        # Left crossing: walk from the peak toward the left base.
        i = peak
        while i > lo and x[i - 1] > level:
            i -= 1
        if i == lo and x[i] > level:
            left_ips[slot] = float(lo)
        else:
            frac = 0.0 if x[i] == x[i - 1] else (x[i] - level) / (x[i] - x[i - 1])
            left_ips[slot] = i - frac
        # Right crossing: walk from the peak toward the right base.
        j = peak
        while j < hi and x[j + 1] > level:
            j += 1
        if j == hi and x[j] > level:
            right_ips[slot] = float(hi)
        else:
            frac = 0.0 if x[j + 1] == x[j] else (level - x[j]) / (x[j + 1] - x[j])
            right_ips[slot] = j + frac
    return left_ips, right_ips


def find_peaks(
    values: np.ndarray,
    *,
    height: object = None,
    prominence: object = None,
    distance: int | None = None,
) -> PeakResult:
    """Find local maxima with prominence, height and distance filters.

    The algorithm is :func:`scipy.signal.find_peaks` in plain NumPy:
    local maxima (plateaus collapse to their middle, biased left),
    then the prominence of each (the walk out to strictly higher
    terrain on both sides, the contour line the higher of the two
    side minima), then the optional filters -- ``height`` (a scalar
    floor or ``(min, max)`` pair), ``prominence`` (likewise) and
    ``distance`` (a minimum separation in samples, enforced greedily
    by height so the *taller* peak survives a clash).

    Widths ride along at ``rel_height=0.5`` (the FWHM every
    chromatography report quotes); a side with no crossing inside
    its prominence window clamps to the base index, an honest lower
    bound.

    Parameters
    ----------
    values:
        One finite 1-D signal.
    height, prominence:
        ``None`` (no filter), a scalar minimum, or a
        ``(minimum, maximum)`` pair -- scipy's own bound convention.
    distance:
        Minimum spacing between kept peaks, in samples.

    Returns
    -------
    PeakResult
        Indices, heights, prominences, base indices, FWHM widths and
        the interpolated half-height crossings; ``filters`` echoes
        what was applied.

    Examples
    --------
    >>> from pharmaml.ml.signal import find_peaks
    >>> x = [0.0, 1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 0.0, 4.0, 1.0]
    >>> result = find_peaks(x)
    >>> result.indices.tolist()
    [2, 4, 6, 8]
    >>> result.prominences.tolist()
    [3.0, 3.0, 1.0, 3.0]
    >>> result.left_bases.tolist()
    [0, 0, 5, 7]
    >>> result.right_bases.tolist()
    [7, 7, 7, 9]
    >>> find_peaks([1.0, 2.0, 2.0, 1.0]).indices.tolist()   # plateau -> middle
    [1]
    >>> find_peaks([1.0, 2.0, 2.0, 2.0, 1.0]).indices.tolist()
    [2]
    >>> find_peaks([0.0, 5.0, 4.0, 4.5, 0.0, 6.0, 5.8, 0.0], distance=3
    ...            ).indices.tolist()
    [1, 5]
    >>> find_peaks([0.0, 5.0, 4.0, 4.5, 0.0, 6.0, 5.8, 0.0], height=4.6
    ...            ).indices.tolist()
    [1, 5]
    """
    x = _as_trace(values, minimum=3)
    height_bound = _resolve_bound(height, "height")
    prominence_bound = _resolve_bound(prominence, "prominence")
    if distance is not None and int(distance) < 1:
        raise DataValidationError(f"distance must be >= 1 sample, got {distance}")

    peaks = _local_maxima(x)
    filters: dict[str, object] = {
        "height": height_bound, "prominence": prominence_bound,
        "distance": None if distance is None else int(distance),
    }
    if peaks.size == 0:
        empty_i = np.empty(0, dtype=int)
        empty_f = np.empty(0, dtype=float)
        return PeakResult(empty_i, empty_f, empty_f, empty_i, empty_i,
                          empty_f, empty_f, empty_f, filters)

    prominences, left_bases, right_bases = _prominence(x, peaks)
    keep = np.ones(peaks.size, dtype=bool)
    if height_bound is not None:
        keep &= (x[peaks] >= height_bound[0]) & (x[peaks] <= height_bound[1])
    if prominence_bound is not None:
        keep &= (prominences >= prominence_bound[0]) & (
            prominences <= prominence_bound[1]
        )
    if distance is not None:
        # Greedy by height (tallest first): a peak is kept only if no
        # already-kept peak stands within `distance` samples.
        order = np.argsort(-x[peaks], kind="stable")
        dropped = np.zeros(peaks.size, dtype=bool)
        for a in range(order.size):
            i = order[a]
            if dropped[i]:
                continue
            for b in range(a + 1, order.size):
                j = order[b]
                if dropped[j]:
                    continue
                if abs(int(peaks[i]) - int(peaks[j])) < int(distance):
                    dropped[j] = True
        keep &= ~dropped
    peaks = peaks[keep]
    prominences = prominences[keep]
    left_bases = left_bases[keep]
    right_bases = right_bases[keep]
    left_ips, right_ips = _half_height_crossings(
        x, peaks, prominences, left_bases, right_bases
    )
    return PeakResult(
        indices=peaks,
        heights=x[peaks].astype(float),
        prominences=prominences,
        left_bases=left_bases,
        right_bases=right_bases,
        widths=right_ips - left_ips,
        left_ips=left_ips,
        right_ips=right_ips,
        filters=filters,
    )


def peak_table(result: PeakResult, *, n: int | None = None) -> str:
    """The chromatographer's peak table: index, height, prominence, FWHM.

    Parameters
    ----------
    result:
        A :func:`find_peaks` result.
    n:
        The trace length (for the location column as a fraction).

    Examples
    --------
    >>> from pharmaml.ml.signal import find_peaks, peak_table
    >>> print(peak_table(find_peaks([0.0, 1.0, 5.0, 0.5, 3.0, 0.0]), n=6))
    peak   index   height   prominence   FWHM   loc%
    1          2     5.00          5.00  1.18   40.0
    2          4     3.00          2.50  0.92   80.0
    """
    lines = ["peak   index   height   prominence   FWHM   loc%"]
    span = max(1, (n or int(result.indices.max()) + 1) - 1)
    for k in range(len(result)):
        location = 100.0 * float(result.indices[k]) / span
        lines.append(
            f"{k + 1:<5} {result.indices[k]:>6} "
            f"{result.heights[k]:>8.2f} {result.prominences[k]:>13.2f} "
            f"{result.widths[k]:>5.2f} {location:>6.1f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Savitzky-Golay smoothing (scipy.signal.savgol_filter)
# ---------------------------------------------------------------------------


def _savgol_design(window_length: int, polyorder: int, deriv: int) -> np.ndarray:
    """One row of the Vandermonde pseudo-inverse (the filter weights).

    The window positions ``t_j = j - m`` (``m`` the half-window) form
    the Vandermonde matrix ``V`` in ascending powers; the smoothing
    weights are row 0 of ``pinv(V)`` (the polynomial *value* at the
    centre), and the k-th derivative weights are ``k!`` times row k.
    """
    half = window_length // 2
    t = np.arange(window_length, dtype=float) - half
    V = np.vander(t, N=polyorder + 1, increasing=True)
    pseudo = np.linalg.pinv(V)
    row = pseudo[deriv] * float(math.factorial(deriv))
    return row


def savgol_coeffs(
    window_length: int, polyorder: int, *, deriv: int = 0
) -> np.ndarray:
    """The Savitzky-Golay convolution weights (scipy's ``savgol_coeffs``).

    The returned row follows scipy's **convolution** convention:
    ``np.convolve(x, coeffs)`` smooths the signal (the row is the
    correlation weights reversed -- invisible for the symmetric
    ``deriv=0`` row, the sign flip for antisymmetric derivative rows).

    Parameters
    ----------
    window_length:
        Odd window size (``2m + 1``).
    polyorder:
        Polynomial degree fitted inside each window
        (``polyorder < window_length``).
    deriv:
        Derivative order to estimate (``deriv <= polyorder``); the
        k-th derivative weights are ``k! * pinv(V)[k]``.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.signal import savgol_coeffs
    >>> np.round(savgol_coeffs(5, 2), 6)
    array([-0.085714,  0.342857,  0.485714,  0.342857, -0.085714])
    >>> float(np.round(savgol_coeffs(7, 2).sum(), 6))   # unity DC gain (deriv=0)
    1.0
    >>> deriv1 = savgol_coeffs(7, 3, deriv=1)
    >>> float(deriv1.sum().round(6))              # zero DC gain (deriv>0)
    0.0
    """
    _validate_savgol(window_length, polyorder, deriv)
    return _savgol_design(window_length, polyorder, deriv)[::-1].copy()


def _validate_savgol(window_length: int, polyorder: int, deriv: int) -> None:
    if window_length < 3 or window_length % 2 == 0:
        raise DataValidationError(
            f"window_length must be odd and >= 3, got {window_length}"
        )
    if polyorder < 0 or polyorder >= window_length:
        raise DataValidationError(
            f"polyorder must satisfy 0 <= polyorder < window_length, "
            f"got {polyorder} vs window {window_length}"
        )
    if deriv < 0 or deriv > polyorder:
        raise DataValidationError(
            f"deriv must satisfy 0 <= deriv <= polyorder, got {deriv}"
        )


def savgol_filter(
    values: np.ndarray, window_length: int, polyorder: int, *, deriv: int = 0
) -> np.ndarray:
    """Savitzky-Golay smoothing (or differentiation), scipy's ``interp`` edges.

    Each interior output is the fitted polynomial's value at the
    window centre -- a fixed convolution, so the whole interior is
    one dot product per sample.  The ``window_length // 2`` edge
    outputs on each side come from the *same* polynomial fitted to
    the truncated edge window, evaluated at the edge offsets --
    scipy's ``mode='interp'`` convention, exact on polynomials of
    degree ``<= polyorder``.

    Parameters
    ----------
    values:
        One finite 1-D signal (``len >= window_length``).
    window_length, polyorder, deriv:
        As :func:`savgol_coeffs`.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.signal import savgol_filter
    >>> y = np.arange(10.0) ** 2
    >>> float(np.max(np.abs(savgol_filter(y, 5, 2) - y))) < 1e-9  # quadratics
    True
    >>> noisy = np.array([0.0, 0.1, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.3,
    ...                    0.2, 0.3, 0.4, 0.5, 0.6])
    >>> smoothed = savgol_filter(noisy, 5, 2)
    >>> float(np.std(noised := noisy - np.linspace(0, 0.6, 15))) > float(
    ...     np.std(smoothed - np.linspace(0, 0.6, 15)))
    True
    """
    x = _as_trace(values, minimum=window_length)
    _validate_savgol(window_length, polyorder, deriv)
    n = x.size
    half = window_length // 2
    weights = _savgol_design(window_length, polyorder, deriv)
    out = np.empty(n, dtype=float)
    # Interior: fixed convolution (correlation with the weights).
    for i in range(half, n - half):
        out[i] = float(np.dot(x[i - half : i + half + 1], weights))
    # Edges: scipy's mode='interp' -- one polynomial per edge window,
    # its deriv-th DERIVATIVE evaluated at each edge offset t = i - half
    # (the value itself for deriv=0; the k-th derivative otherwise).
    t = np.arange(window_length, dtype=float) - half
    V = np.vander(t, N=polyorder + 1, increasing=True)
    pseudo = np.linalg.pinv(V)
    left_coeffs = pseudo @ x[:window_length]
    for i in range(half):
        out[i] = _poly_derivative_value(left_coeffs, t[i], deriv)
    right_coeffs = pseudo @ x[n - window_length :]
    for i in range(n - half, n):
        out[i] = _poly_derivative_value(
            right_coeffs, t[i - (n - window_length)], deriv
        )
    return out


def _poly_derivative_value(
    coeffs: np.ndarray, t: float, deriv: int
) -> float:
    """The ``deriv``-th derivative of the fitted polynomial at ``t``.

    ``d^k/dt^k (c_m t^m) = c_m * m!/(m-k)! * t^(m-k)`` -- the edge
    rule scipy's ``interp`` mode applies (the value itself at
    ``deriv=0``, which is why the quadratic-exact doctest works).
    """
    total = 0.0
    for m in range(deriv, coeffs.size):
        scale = 1.0
        for j in range(m - deriv + 1, m + 1):
            scale *= j
        total += float(coeffs[m]) * scale * (t ** (m - deriv))
    return total


# ---------------------------------------------------------------------------
# The demo signal
# ---------------------------------------------------------------------------


def synthetic_chromatogram(
    n: int = 200, *, seed: int = 42
) -> np.ndarray:
    """A seeded three-peak chromatogram (baseline + drift + noise).

    The demo trace for the whole module: three Gaussian peaks of
    decreasing width on a slowly drifting baseline, plus
    reproducible detector noise -- the shape an LC-UV run actually
    produces.

    Parameters
    ----------
    n:
        Number of samples.
    seed:
        Seed for the noise generator.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.signal import synthetic_chromatogram
    >>> trace = synthetic_chromatogram(200, seed=42)
    >>> trace.shape
    (200,)
    >>> round(float(trace.max()), 4)
    3.1979
    """
    if n < 20:
        raise DataValidationError(f"n must be >= 20, got {n}")
    t = np.linspace(0.0, 10.0, n)
    rng = np.random.default_rng(seed)
    peaks = (
        (2.2, 3.0, 0.18),
        (5.0, 2.2, 0.30),
        (7.8, 1.4, 0.45),
    )
    signal = np.zeros(n)
    for centre, height, width in peaks:
        signal += height * np.exp(-((t - centre) ** 2) / (2.0 * width**2))
    baseline = 0.15 + 0.02 * t
    noise = rng.normal(0.0, 0.08, n)
    return signal + baseline + noise
