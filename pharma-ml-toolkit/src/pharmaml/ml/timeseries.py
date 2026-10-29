"""The time-series core: correlation structure, decomposition,
changepoints, and the feature table.

The best-of-ml-python *Time Series Data* category (29 projects:
statsmodels, sktime, tsfresh, darts, kats, Prophet ...) was reviewed
entry by entry for this module.  The category's engines cluster into
four ideas, and the analysis half of each is closed-form mathematics:

* **The correlation structure** -- the ACF (which lag predicts which)
  and the PACF (the same, with the shorter lags' credit removed --
  Durbin-Levinson's recursion), the pair every ARIMA order-selection
  chapter starts from.
* **The decomposition** -- classical additive trend/seasonal/remainder
  (the centered moving-average trend, phase-averaged seasonal profile
  centred to sum zero, the remainder what is left), with the two
  strength numbers (how much of the detrended variance the seasonal
  profile explains, how much of the total the trend does).
* **The changepoints** -- offline binary segmentation on the CUSUM
  path: each segment's ``S_k = sum(x_i - x̄)`` standardised into a
  sup-|Brownian-bridge| statistic, judged against the
  Kolmogorov-distribution quantile (the K-S table's own critical
  values), split at the argmax, both halves recursed.
* **The feature table** -- tsfresh's contract in miniature: fixed
  non-overlapping windows, one row each, the dozen statistics
  (location, spread, extremes, slope, trend R², mean absolute change,
  above-mean count, lag-1 autocorrelation, binned entropy) that turn a
  series into a modelling matrix.

The pharma reading, stated once: the series is a stability-indication
assay track, a batch-release trend, a longitudinal PD reading -- and
"did the process shift?" is a changepoint question, "is the batch
seasonal?" a decomposition question, "which window features predict
failure?" a feature-table question.

The honest scope: statsmodels' ARIMA/SARIMAX likelihood machinery,
sktime's framework and Prophet's Bayesian curve fitter are *engines*
(iterative fitting over model spaces) the toolkit deliberately does
not vendor; forecasting itself is native next door in
:mod:`pharmaml.ml.sequences` (sliding windows, the commonsense
baseline, a from-scratch LSTM forecaster).

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.sequences import synthetic_series
>>> from pharmaml.ml.timeseries import (
...     autocorrelation, detect_changepoints, partial_autocorrelation,
...     seasonal_decomposition, window_features,
... )
>>> series = synthetic_series("sine", 240, seed=42)
>>> acf = autocorrelation(series, max_lag=30)
>>> round(float(acf.values[0]), 6)
1.0
>>> acf.peak_lag
24
>>> pacf = partial_autocorrelation(series, max_lag=30)
>>> round(float(pacf.values[0]), 6)
1.0
>>> decomp = seasonal_decomposition(series, period=24)
>>> decomp.seasonal.shape == series.shape
True
>>> float(np.abs(decomp.seasonal[:24].sum())) < 1e-9   # one period sums to 0
True
>>> decomp.seasonal_strength > 0.5                     # the signal is real
True
>>> table = window_features(series, window=24)
>>> len(table)
10
>>> 0.0 <= float(table["trend_r2"].min()) <= 1.0
True
>>> shifted = np.concatenate([np.zeros(100), np.full(100, 3.0)])
>>> changes = detect_changepoints(shifted)
>>> 95 <= changes.points[0].index <= 105
True
>>> changes.points[0].direction
'up'
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed kept for interface symmetry with the toolkit's pin.
RANDOM_STATE = 42

#: The significance level for the changepoint splits (Kolmogorov bar).
CHANGEPOINT_ALPHA = 0.05

#: The shortest segment binary segmentation will consider splitting.
CHANGEPOINT_MIN_SEGMENT = 8

#: Bins for the window entropy (tsfresh's binned-entropy convention).
ENTROPY_BINS = 10


def _as_series(values: np.ndarray, *, minimum: int = 4) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < minimum:
        raise DataValidationError(f"the series needs at least {minimum} points, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise DataValidationError("the series must be finite (no NaN/inf)")
    return array


# ---------------------------------------------------------------------------
# The correlation structure (ACF + PACF)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationResult:
    """The lag correlation table, with the significance band.

    ``values[k]`` is the correlation at lag ``k`` (``values[0] == 1`` by
    construction); ``band`` is the ±1.96/sqrt(n) white-noise bound --
    lags outside it are the ones worth reading.
    """

    values: np.ndarray
    band: float
    n: int

    @property
    def significant_lags(self) -> tuple[int, ...]:
        """Lags (beyond 0) whose correlation escapes the white-noise band."""
        return tuple(
            lag for lag in range(1, self.values.size) if abs(float(self.values[lag])) > self.band
        )

    @property
    def peak_lag(self) -> int:
        """The lag of the strongest positive correlation (beyond 0)."""
        if self.values.size < 2:
            return 0
        return int(np.argmax(self.values[1:]) + 1)


def autocorrelation(values: np.ndarray, max_lag: int = 24) -> CorrelationResult:
    """The autocorrelation function: which lag carries which memory.

    ``r_k = sum_{t>=k} (x_t - x̄)(x_{t-k} - x̄) / sum (x_t - x̄)²`` -- the
    biased (divisor n) estimator, the one whose white-noise band is
    the textbook ±1.96/sqrt(n).  ``max_lag`` is capped at n−2 so every
    lag has at least two pairs to estimate.

    Raises
    ------
    DataValidationError
        On a too-short series or a non-positive/too-large max_lag.
    """
    array = _as_series(values)
    if max_lag < 1:
        raise DataValidationError(f"max_lag must be >= 1, got {max_lag}")
    if max_lag > array.size - 2:
        raise DataValidationError(f"max_lag must be <= n-2 = {array.size - 2}, got {max_lag}")
    centered = array - array.mean()
    denominator = float(np.dot(centered, centered))
    if denominator < 1e-24:
        raise DataValidationError("the series is constant: autocorrelation is undefined")
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = float(np.dot(centered[lag:], centered[:-lag])) / denominator
    result = CorrelationResult(values=acf, band=1.96 / np.sqrt(array.size), n=int(array.size))
    _LOG.debug(
        "acf: n=%d max_lag=%d peak=%d significant=%s",
        array.size,
        max_lag,
        result.peak_lag,
        result.significant_lags[:6],
    )
    return result


def partial_autocorrelation(values: np.ndarray, max_lag: int = 24) -> CorrelationResult:
    """The partial ACF via Durbin-Levinson: lag-k credit, shorter lags removed.

    The recursion builds the order-k AR fit's last coefficient -- the
    PACF's textbook definition -- from the ACF alone: AR( p ) processes
    cut off to zero after lag p, which is exactly the order-selection
    rule this function exists to serve.

    Raises
    ------
    DataValidationError
        On a too-short/constant series or a non-positive/too-large
        max_lag.
    """
    array = _as_series(values)
    if max_lag < 1:
        raise DataValidationError(f"max_lag must be >= 1, got {max_lag}")
    if max_lag > array.size - 2:
        raise DataValidationError(f"max_lag must be <= n-2 = {array.size - 2}, got {max_lag}")
    acf = autocorrelation(array, max_lag=max_lag).values
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0
    # Durbin-Levinson over the last-coefficient diagonals
    previous = np.zeros(0)
    for k in range(1, max_lag + 1):
        if k == 1:
            phi_kk = float(acf[1])
            current = np.array([phi_kk])
        else:
            r = acf[1 : k + 1]
            num = float(r[k - 1] - np.dot(previous, r[: k - 1][::-1]))
            den = 1.0 - float(np.dot(previous, acf[1:k]))
            if abs(den) < 1e-12:
                raise DataValidationError(
                    "Durbin-Levinson hit a singular step: the series is degenerate"
                )
            phi_kk = num / den
            current = np.concatenate([previous - phi_kk * previous[::-1], [phi_kk]])
        pacf[k] = phi_kk
        previous = current
    result = CorrelationResult(values=pacf, band=1.96 / np.sqrt(array.size), n=int(array.size))
    _LOG.debug("pacf: n=%d max_lag=%d", array.size, max_lag)
    return result


# ---------------------------------------------------------------------------
# The classical additive decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecompositionResult:
    """The additive split: observed = trend + seasonal + remainder.

    ``trend`` carries NaN at the edges where the centered moving
    average has no support (half a period each side -- statsmodels'
    own contract); ``seasonal`` is the phase-averaged profile tiled
    over the whole series, centred so one period sums to exactly zero;
    ``remainder`` is what is left where both are defined.
    """

    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    remainder: np.ndarray
    period: int
    n_complete: int

    @property
    def seasonal_strength(self) -> float:
        """1 - Var(remainder)/Var(detrended): how much the profile explains."""
        mask = ~np.isnan(self.remainder)
        detrended = (self.observed - self.trend)[mask]
        if detrended.var() < 1e-24:
            return 0.0
        return float(1.0 - self.remainder[mask].var() / detrended.var())

    @property
    def trend_strength(self) -> float:
        """1 - Var(detrended)/Var(observed): how much the trend explains."""
        mask = ~np.isnan(self.trend)
        detrended = (self.observed - self.trend)[mask]
        observed = self.observed[mask]
        if observed.var() < 1e-24:
            return 0.0
        return float(1.0 - detrended.var() / observed.var())


def seasonal_decomposition(values: np.ndarray, period: int = 24) -> DecompositionResult:
    """Classical additive decomposition: centered-MA trend, phase-average seasonal.

    The recipe, exactly as the textbooks order it: (1) trend by a
    centered moving average over one period (the even-period two-pass
    ``period/2`` average included, so no phase leaks into the trend);
    (2) detrended values averaged *by phase* into a seasonal profile,
    then centred so one period sums to zero; (3) the profile tiled onto
    the whole index; (4) remainder = observed − trend − seasonal where
    both are defined.

    Raises
    ------
    DataValidationError
        On a too-short series for the period (two full periods are the
        floor for phase averaging) or a non-positive period.
    """
    array = _as_series(values, minimum=6)
    if period < 2:
        raise DataValidationError(f"period must be >= 2, got {period}")
    if array.size < 2 * period:
        raise DataValidationError(
            f"need at least two full periods ({2 * period} points), got {array.size}"
        )
    n = array.size
    trend = np.full(n, np.nan)
    if period % 2 == 1:
        half = period // 2
        for t in range(half, n - half):
            trend[t] = float(array[t - half : t + half + 1].mean())
    else:
        # even period: the two-pass half-weight average (statsmodels'
        # convention) keeps the window centered without phase bias --
        # half weight on x[t-half] and x[t+half], full weight between
        half = period // 2
        for t in range(half, n - half):
            inner = array[t - half + 1 : t + half]
            trend[t] = (0.5 * array[t - half] + inner.sum() + 0.5 * array[t + half]) / period
    mask = ~np.isnan(trend)
    detrended = array - trend
    valid_indices = np.where(mask)[0]
    profile = np.zeros(period)
    for phase in range(period):
        phase_values = detrended[valid_indices[valid_indices % period == phase]]
        profile[phase] = float(phase_values.mean()) if phase_values.size else 0.0
    profile -= profile.mean()
    seasonal = np.tile(profile, int(np.ceil(n / period)))[:n]
    remainder = np.where(mask, array - trend - seasonal, np.nan)
    result = DecompositionResult(
        observed=array,
        trend=trend,
        seasonal=seasonal,
        remainder=remainder,
        period=int(period),
        n_complete=int(mask.sum()),
    )
    _LOG.debug(
        "decomposition: n=%d period=%d complete=%d seasonal_strength=%.3f",
        n,
        period,
        result.n_complete,
        result.seasonal_strength,
    )
    return result


# ---------------------------------------------------------------------------
# The changepoints (two-sided CUSUM)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Changepoint:
    """One detected change: where, which way, how much."""

    index: int
    direction: str
    magnitude: float
    statistic: float


@dataclass(frozen=True)
class ChangepointResult:
    """The binary-segmentation CUSUM verdict for a series.

    ``sigma`` is the robust scale the statistic was standardised
    against (1.4826 x MAD, the outlier-safe convention);
    ``critical_value`` is the Kolmogorov-distribution quantile the
    splits were judged by; ``points`` are the detected changes in
    index order, each carrying the standardised statistic that earned
    it (comfortably above the critical value means a confident split).
    """

    points: tuple[Changepoint, ...]
    sigma: float
    critical_value: float
    alpha: float
    min_segment: int
    n: int


def _kolmogorov_quantile(alpha: float) -> float:
    """The sup|Brownian-bridge| quantile: ``P(sup|BB| > x) = alpha``.

    The classic series ``2·sum((-1)^(j-1)·exp(-2 j^2 x^2))`` (the
    Kolmogorov distribution tail), solved by bisection -- the same
    table the Kolmogorov-Smirnov test reads: 1.224 at alpha=0.10,
    1.358 at 0.05, 1.628 at 0.01.
    """
    if not (0.0 < alpha < 1.0):
        raise DataValidationError(f"alpha must be in (0, 1), got {alpha!r}")

    def tail(x: float) -> float:
        total = 0.0
        for j in range(1, 60):
            term = 2.0 * (-1.0) ** (j - 1) * np.exp(-2.0 * j * j * x * x)
            total += float(term)
            if abs(term) < 1e-18:
                break
        return total

    lo, hi = 0.5, 4.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tail(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return float(hi)


def detect_changepoints(
    values: np.ndarray,
    *,
    alpha: float = 0.05,
    min_segment: int = 8,
) -> ChangepointResult:
    """Offline mean-shift changepoints: binary segmentation on the CUSUM.

    On each segment, the CUSUM path ``S_k = sum_{i<=k}(x_i - x̄_seg)``
    is standardised by the segment's robust scale and length into a
    sup-|Brownian-bridge| statistic; a split is declared when its
    maximum exceeds the Kolmogorov-distribution quantile for ``alpha``
    (the same critical values the K-S test uses -- 1.358 at the
    default 0.05), located at the argmax.  The segment then splits
    and both halves recurse, stopping on segments shorter than
    ``min_segment``, constant segments, or statistics below the bar.

    Raises
    ------
    DataValidationError
        On a too-short series, alpha outside (0, 1), or a non-positive
        min_segment.
    """
    array = _as_series(values, minimum=6)
    if not (0.0 < alpha < 1.0):
        raise DataValidationError(f"alpha must be in (0, 1), got {alpha!r}")
    if min_segment < 2:
        raise DataValidationError(f"min_segment must be >= 2, got {min_segment}")
    critical_value = _kolmogorov_quantile(alpha)
    points: list[Changepoint] = []

    def recurse(segment: np.ndarray, offset: int) -> None:
        m = segment.size
        if m < 2 * min_segment:
            return
        centered = segment - segment.mean()
        path = np.cumsum(centered)
        mad = float(np.median(np.abs(centered)))
        sigma = 1.4826 * mad
        if sigma < 1e-12:
            return  # constant (or exactly alternating) segment: no shifts
        k_star = int(np.argmax(np.abs(path[: m - 1])))  # a split must leave
        # both sides non-empty; the last element is 0 by construction
        statistic = float(abs(path[k_star]) / (sigma * np.sqrt(m)))
        if statistic <= critical_value:
            return
        before = segment[: k_star + 1]
        after = segment[k_star + 1 :]
        magnitude = float(after.mean() - before.mean())
        direction = "up" if magnitude > 0 else "down"
        points.append(
            Changepoint(
                index=int(offset + k_star + 1),
                direction=direction,
                magnitude=magnitude,
                statistic=statistic,
            )
        )
        recurse(before, offset)
        recurse(after, offset + k_star + 1)

    recurse(array, 0)
    points.sort(key=lambda point: point.index)
    centered = array - array.mean()
    sigma = 1.4826 * float(np.median(np.abs(centered)))
    result = ChangepointResult(
        points=tuple(points),
        sigma=float(sigma),
        critical_value=float(critical_value),
        alpha=float(alpha),
        min_segment=int(min_segment),
        n=int(array.size),
    )
    _LOG.debug(
        "binary-cusum: n=%d sigma=%.4g critical=%.3f points=%s",
        array.size,
        sigma,
        critical_value,
        [(p.index, p.direction) for p in points],
    )
    return result


# ---------------------------------------------------------------------------
# The window feature table (tsfresh's contract in miniature)
# ---------------------------------------------------------------------------


def _binned_entropy(window: np.ndarray, bins: int) -> float:
    """Shannon entropy (bits) of the ``bins``-bin histogram of a window."""
    lo, hi = float(window.min()), float(window.max())
    if hi - lo < 1e-24:
        return 0.0
    counts, _ = np.histogram(window, bins=bins, range=(lo, hi))
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def window_features(values: np.ndarray, window: int = 24) -> pd.DataFrame:
    """Fixed non-overlapping windows, one feature row each (tsfresh-lite).

    The dozen statistics: location and spread (mean, std, min, max,
    median), the linear trend (slope per step, its R²), the path
    roughness (mean absolute step change), the above-mean count, the
    lag-1 autocorrelation and the ``ENTROPY_BINS``-bin entropy (bits).
    The final partial window is dropped -- a feature row from half a
    window would compare unfairly with the rest.

    Raises
    ------
    DataValidationError
        On a window shorter than 3, or a series shorter than one
        window.
    """
    array = _as_series(values, minimum=3)
    if window < 3:
        raise DataValidationError(f"window must be >= 3, got {window}")
    if array.size < window:
        raise DataValidationError(
            f"the series ({array.size} points) is shorter than one window ({window})"
        )
    n_windows = array.size // window
    rows: list[dict[str, float | int]] = []
    for w in range(n_windows):
        chunk = array[w * window : (w + 1) * window]
        t = np.arange(window, dtype=float)
        slope, intercept = np.polyfit(t, chunk, 1)
        fitted = slope * t + intercept
        ss_res = float(np.sum((chunk - fitted) ** 2))
        ss_tot = float(np.sum((chunk - chunk.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-24 else (1.0 if ss_res < 1e-24 else 0.0)
        centered = chunk - chunk.mean()
        denom = float(np.dot(centered, centered))
        autocorr1 = float(np.dot(centered[1:], centered[:-1]) / denom) if denom > 1e-24 else 0.0
        above_mean = int((chunk > chunk.mean()).sum())
        rows.append(
            {
                "window_start": w * window,
                "n": int(window),
                "mean": float(chunk.mean()),
                "std": float(chunk.std()),
                "min": float(chunk.min()),
                "max": float(chunk.max()),
                "median": float(np.median(chunk)),
                "slope": float(slope),
                "trend_r2": float(r2),
                "mean_abs_change": float(np.abs(np.diff(chunk)).mean()),
                "count_above_mean": above_mean,
                "autocorr_lag1": autocorr1,
                "entropy_bits": _binned_entropy(chunk, ENTROPY_BINS),
            }
        )
    table = pd.DataFrame.from_records(rows)
    _LOG.debug("window features: %d windows of %d from %d points", n_windows, window, array.size)
    return table


# ---------------------------------------------------------------------------
# The power spectrum (the scipy.signal.periodogram/welch absorption)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrumResult:
    """One-sided power spectral density, dB-scaled for reading.

    ``frequencies`` are cycles per sample (0 .. 0.5); ``powers`` the
    one-sided periodogram (2|X_k|²/n², doubled except DC and Nyquist);
    ``powers_db`` the 10·log10 form; ``peak_frequency`` and
    ``peak_period`` read the dominant oscillation off the table.
    """

    frequencies: np.ndarray
    powers: np.ndarray
    powers_db: np.ndarray
    n_segments: int
    method: str

    @property
    def peak_frequency(self) -> float:
        """The frequency of the largest non-DC bin (cycles/sample)."""
        if self.frequencies.size < 2:
            return 0.0
        index = int(np.argmax(self.powers[1:])) + 1
        return float(self.frequencies[index])

    @property
    def peak_period(self) -> float:
        """The dominant period in samples (inf when no peak above DC)."""
        frequency = self.peak_frequency
        if frequency <= 0:
            return float("inf")
        return 1.0 / frequency


def power_spectrum(values: np.ndarray, *, segment_length: int | None = None) -> SpectrumResult:
    """The one-sided power spectrum -- periodogram, or Welch-averaged.

    ``segment_length=None`` is the plain periodogram (one FFT over the
    whole series, Hann-windowed); an integer splits the series into
    half-overlapping Hann windows of that length, averages their
    periodograms and halves the DC/Nyquist bins -- Welch's method, the
    variance-reduced estimate ``scipy.signal.welch`` is called for,
    with the Hann taper both cases share.  The FFT itself is NumPy's.

    Raises
    ------
    DataValidationError
        On a too-short series or a segment length outside [4, n].

    Examples
    --------
    >>> from pharmaml.ml.sequences import synthetic_series
    >>> from pharmaml.ml.timeseries import power_spectrum
    >>> series = synthetic_series("sine", 240, seed=42)   # period-24 signal
    >>> spectrum = power_spectrum(series)
    >>> 23.0 <= spectrum.peak_period <= 25.0
    True
    >>> averaged = power_spectrum(series, segment_length=120)
    >>> 23.0 <= averaged.peak_period <= 25.0
    True
    >>> averaged.n_segments
    3
    """
    array = _as_series(values, minimum=8)
    if segment_length is None:
        window = np.hanning(array.size)
        tapered = array * window
        transform = np.fft.rfft(tapered)
        n = array.size
        powers = np.abs(transform) ** 2 / (np.sum(window**2) / 1.0)
        # one-sided scaling: double everything but DC (and Nyquist when even)
        powers[1:] *= 2.0
        if n % 2 == 0 and powers.size > 1:
            powers[-1] *= 0.5
        frequencies = np.fft.rfftfreq(n)
        result = SpectrumResult(
            frequencies=frequencies,
            powers=powers,
            powers_db=10.0 * np.log10(np.maximum(powers, 1e-24)),
            n_segments=1,
            method="periodogram",
        )
    else:
        if segment_length < 4 or segment_length > array.size:
            raise DataValidationError(
                f"segment_length must be in [4, n], got {segment_length} for n={array.size}"
            )
        step = segment_length // 2
        window = np.hanning(segment_length)
        window_norm = float(np.sum(window**2))
        segments = []
        start = 0
        while start + segment_length <= array.size:
            segments.append(array[start : start + segment_length] * window)
            start += step
        if not segments:  # pragma: no cover - guarded by the length check
            raise DataValidationError("segmentation produced no windows")
        stacked = np.vstack(segments)
        transforms = np.fft.rfft(stacked, axis=1)
        powers = np.mean(np.abs(transforms) ** 2, axis=0) / window_norm
        powers[1:] *= 2.0
        if segment_length % 2 == 0 and powers.size > 1:
            powers[-1] *= 0.5
        frequencies = np.fft.rfftfreq(segment_length)
        result = SpectrumResult(
            frequencies=frequencies,
            powers=powers,
            powers_db=10.0 * np.log10(np.maximum(powers, 1e-24)),
            n_segments=len(segments),
            method="welch",
        )
    _LOG.debug(
        "power spectrum (%s): %d bins, peak period %.2f",
        result.method,
        result.frequencies.size,
        result.peak_period,
    )
    return result
