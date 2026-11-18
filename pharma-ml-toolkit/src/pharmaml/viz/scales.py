"""matplotlib's normalization, ticker and scale math, absorbed natively
(100x-170).

The matplotlib repository was reviewed file by file; the pure-math half
of ``lib/matplotlib/colors.py``, ``ticker.py`` and ``scale.py`` lands
here as NumPy code with the upstream formulas verbatim:

* the **norm classes** -- :class:`Normalize` (``(x - vmin)/(vmax -
  vmin)``, no clipping leaves values outside [0, 1] for the
  under/over colors), :class:`LogNorm` (linear in log space;
  non-positive values become NaN -- matplotlib *masks* them, which
  takes the bad color downstream), :class:`SymLogNorm` (the exact
  piecewise mapping: linear ``x * linscale / (1 - 1/base)`` inside
  ``±linthresh``, ``sign(x)·linthresh·(linscale_adj -
  log_b(linthresh) + log_b|x|)`` outside), :class:`TwoSlopeNorm`
  (matplotlib's three-point ``np.interp`` through ``vcenter``) and
  :class:`BoundaryNorm` (``np.digitize`` with left-closed bins).
* the **scale transforms** -- :func:`symlog_transform` and its
  inverse (identical to ``SymmetricalLogScale``'s transform, the
  quoted constants), and :func:`logit_transform` (``log10(p/(1-p))``
  with upstream's ±1000 clip and the 1e-7 domain clamp).
* the **tick engine** -- :func:`nice_ticks` is matplotlib's
  ``MaxNLocator`` core: the ``[1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]``
  step staircase (extended with ``0.1·steps[:-1]`` and
  ``10·steps[1]``), ``scale_range``'s offset/scale factoring, the
  ``_Edge_integer`` float-precision-safe floor/ceil, and the
  smallest-qualifying-step walk with the ``min_n_ticks`` guard --
  verified against ``matplotlib.ticker.MaxNLocator`` tick for tick
  (``scratch/probe_vizlab_170.py``).  :func:`log_ticks` is
  ``LogLocator``'s decade/stride scheme, :func:`multiple_ticks` the
  ``MultipleLocator`` grid, and :func:`format_scalar_ticks` the
  ``ScalarFormatter`` significant-digits loop (the offset/scientific
  machinery is the one piece not carried -- stated in its docstring).

Examples
--------
>>> import numpy as np
>>> from pharmaml.viz.scales import (
...     BoundaryNorm, LogNorm, Normalize, SymLogNorm, TwoSlopeNorm,
...     format_scalar_ticks, logit_transform, log_ticks, multiple_ticks,
...     nice_ticks, symlog_transform,
... )
>>> norm = Normalize(0.0, 10.0)
>>> float(norm(5.0))
0.5
>>> norm([-5.0, 5.0, 15.0]).round(3).tolist()
[-0.5, 0.5, 1.5]
>>> float(norm.inverse(0.5))
5.0
>>> lognorm = LogNorm(1.0, 100.0)
>>> round(float(lognorm(10.0)[0]), 3)
0.5
>>> bool(np.isnan(lognorm(-1.0)))
True
>>> symlog = SymLogNorm(linthresh=1.0)
>>> [round(float(v), 4) for v in symlog([-10.0, 0.0, 10.0])]
[0.0, 0.5, 1.0]
>>> two = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=9.0)
>>> [round(float(v), 3) for v in two([-1.0, 0.0, 4.5, 9.0])]
[0.0, 0.5, 0.75, 1.0]
>>> bn = BoundaryNorm([0.0, 2.0, 5.0, 10.0], ncolors=3)
>>> bn([1.0, 3.0, 9.0]).tolist()
[0, 1, 2]
>>> nice_ticks(0.0, 1.0, nbins=5).round(6).tolist()
[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
>>> nice_ticks(3.0, 97.0, nbins=5).tolist()
[0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
>>> log_ticks(0.01, 100.0).round(10).tolist()
[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
>>> multiple_ticks(0.0, 10.0, base=2.5).tolist()
[0.0, 2.5, 5.0, 7.5, 10.0]
>>> format_scalar_ticks([0.1, 0.3, 0.5])
['0.1', '0.3', '0.5']
>>> format_scalar_ticks([1200.0, 1600.0, 2000.0])
['1200', '1600', '2000']
>>> [round(float(v), 4) for v in symlog_transform([-100.0, -1.0, 0.0, 1.0, 100.0], 1.0, 0.9)]
[-3.0, -1.0, 0.0, 1.0, 3.0]
>>> round(float(logit_transform(0.9)), 4)
0.9542
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "BoundaryNorm",
    "LogNorm",
    "Normalize",
    "SymLogNorm",
    "TwoSlopeNorm",
    "format_scalar_ticks",
    "log_ticks",
    "logit_inverse",
    "logit_transform",
    "multiple_ticks",
    "nice_ticks",
    "symlog_inverse",
    "symlog_transform",
]


class Normalize:
    """matplotlib's ``Normalize``: linear (vmin, vmax) -> (0, 1)."""

    def __init__(self, vmin: float | None = None, vmax: float | None = None,
                 clip: bool = False) -> None:
        self.vmin = None if vmin is None else float(vmin)
        self.vmax = None if vmax is None else float(vmax)
        self.clip = bool(clip)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(vmin={self.vmin}, vmax={self.vmax})"

    def autoscale(self, values: object) -> None:
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            self.vmin, self.vmax = float(finite.min()), float(finite.max())

    def __call__(self, values: object) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if self.vmin is None or self.vmax is None:
            self.autoscale(arr)
        assert self.vmin is not None and self.vmax is not None
        if self.vmin > self.vmax:
            raise ValueError("vmin must be less than or equal to vmax")
        if self.clip:
            arr = np.clip(arr, self.vmin, self.vmax)
        if self.vmin == self.vmax:
            return np.zeros_like(arr)
        return (arr - self.vmin) / (self.vmax - self.vmin)

    def inverse(self, values: object) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        assert self.vmin is not None and self.vmax is not None
        return self.vmin + arr * (self.vmax - self.vmin)


class LogNorm(Normalize):
    """matplotlib's ``LogNorm``: linear in log10 space, NaN for x <= 0.

    Upstream *masks* non-positive values (they render with the bad
    color); this module returns NaN, which every downstream consumer
    here already treats as bad.
    """

    def autoscale(self, values: object) -> None:
        arr = np.asarray(values, dtype=float)
        positive = arr[np.isfinite(arr) & (arr > 0)]
        if positive.size:
            self.vmin, self.vmax = float(positive.min()), float(positive.max())

    def __call__(self, values: object) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if self.vmin is None or self.vmax is None:
            self.autoscale(arr)
        assert self.vmin is not None and self.vmax is not None
        if self.vmin <= 0 or self.vmax <= 0:
            raise ValueError("LogNorm needs positive vmin and vmax")
        out = np.full(arr.shape, np.nan)
        positive = arr > 0
        out[positive] = (
            (np.log10(arr[positive]) - math.log10(self.vmin))
            / (math.log10(self.vmax) - math.log10(self.vmin))
        )
        if self.clip:
            out = np.clip(out, 0.0, 1.0)
        return np.atleast_1d(np.asarray(out, dtype=float))


class SymLogNorm(Normalize):
    """matplotlib's ``SymLogNorm``: linear near zero, log-scaled beyond.

    ``linscale`` stretches the linear region relative to the log
    decades (upstream default 1.0); the mapping matches
    ``SymmetricalLogTransform`` exactly.
    """

    def __init__(self, vmin: float | None = None, vmax: float | None = None,
                 clip: bool = False, *, linthresh: float = 2.0,
                 linscale: float = 1.0, base: float = 10.0) -> None:
        super().__init__(vmin, vmax, clip)
        if linthresh <= 0:
            raise ValueError("linthresh must be positive")
        self.linthresh = float(linthresh)
        self.linscale = float(linscale)
        self.base = float(base)

    def __call__(self, values: object) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(values, dtype=float))
        if self.vmin is None or self.vmax is None:
            self.autoscale(arr)
        assert self.vmin is not None and self.vmax is not None
        t = symlog_transform(arr, self.linthresh, self.linscale, self.base)
        tmin, tmax = symlog_transform(
            np.array([self.vmin, self.vmax]), self.linthresh, self.linscale, self.base
        )
        if tmax == tmin:
            return np.zeros_like(arr)
        out = (t - tmin) / (tmax - tmin)
        if self.clip:
            out = np.clip(out, 0.0, 1.0)
        return out


class TwoSlopeNorm(Normalize):
    """matplotlib's ``TwoSlopeNorm``: two linear ramps meeting at vcenter."""

    def __init__(self, vmin: float | None = None, vcenter: float = 0.0,
                 vmax: float | None = None, clip: bool = False) -> None:
        super().__init__(vmin, vmax, clip)
        self.vcenter = float(vcenter)

    def __call__(self, values: object) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(values, dtype=float))
        if self.vmin is None or self.vmax is None:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                self.vmin, self.vmax = float(finite.min()), float(finite.max())
        assert self.vmin is not None and self.vmax is not None
        if not self.vmin < self.vcenter < self.vmax:
            raise ValueError("vmin < vcenter < vmax is required")
        # upstream: np.interp over [vmin, vcenter, vmax] -> [0, 0.5, 1]
        lo = (arr - self.vmin) / (self.vcenter - self.vmin) * 0.5
        hi = 0.5 + (arr - self.vcenter) / (self.vmax - self.vcenter) * 0.5
        out = np.where(arr < self.vcenter, lo, hi)
        if self.clip:
            out = np.clip(out, 0.0, 1.0)
        return out


class BoundaryNorm:
    """matplotlib's ``BoundaryNorm``: values -> discrete color indices.

    Bins are left-closed ``[b[i], b[i+1])``; values below/above the
    boundary range take ``-1`` / ``ncolors`` (the under/over slots).
    """

    def __init__(self, boundaries: object, ncolors: int, clip: bool = False) -> None:
        self.boundaries = np.asarray(boundaries, dtype=float)
        if self.boundaries.ndim != 1 or len(self.boundaries) < 2:
            raise ValueError("boundaries must be a 1-D sequence of >= 2 values")
        if np.any(np.diff(self.boundaries) <= 0):
            raise ValueError("boundaries must be monotonically increasing")
        self.ncolors = int(ncolors)
        self.clip = bool(clip)
        self.vmin, self.vmax = float(self.boundaries[0]), float(self.boundaries[-1])
        if self.ncolors < len(self.boundaries) - 1:
            raise ValueError("ncolors must be at least the number of boundary bins")

    def __call__(self, values: object) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(values, dtype=float))
        if self.clip:
            arr = np.clip(arr, self.vmin, self.vmax)
        idx = np.digitize(arr, self.boundaries) - 1
        idx[arr < self.vmin] = -1
        idx[arr >= self.vmax] = self.ncolors
        return idx


# ---------------------------------------------------------------------------
# scale transforms (matplotlib/scale.py)
# ---------------------------------------------------------------------------


def symlog_transform(
    values: object, linthresh: float, linscale: float = 1.0, base: float = 10.0
) -> np.ndarray:
    """``SymmetricalLogTransform.transform_non_affine``, verbatim.

    ``x * linscale/(1 - 1/base)`` inside ``±linthresh``;
    ``sign(x)·linthresh·(linscale_adj - log_b(linthresh) + log_b|x|)``
    outside.
    """
    arr = np.asarray(values, dtype=float)
    if linthresh <= 0:
        raise ValueError("linthresh must be positive")
    linscale_adj = linscale / (1.0 - 1.0 / base)
    log_base = math.log(base)
    abs_a = np.abs(arr)
    inside = abs_a <= linthresh
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sign(arr) * linthresh * (
            linscale_adj - math.log(linthresh) / log_base + np.log(abs_a) / log_base
        )
    if np.all(inside):
        return arr * linscale_adj
    out = np.where(inside, arr * linscale_adj, out)
    return out


def symlog_inverse(
    values: object, linthresh: float, linscale: float = 1.0, base: float = 10.0
) -> np.ndarray:
    """``InvertedSymmetricalLogTransform``, the exact inverse above."""
    arr = np.asarray(values, dtype=float)
    linscale_adj = linscale / (1.0 - 1.0 / base)
    invlinthresh = linthresh * linscale_adj
    abs_a = np.abs(arr)
    inside = abs_a <= invlinthresh
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sign(arr) * linthresh * np.exp(
            (abs_a / linthresh - linscale_adj) * math.log(base)
        )
    return np.where(inside, arr / linscale_adj, out)


def logit_transform(values: object, clip: bool = True) -> np.ndarray:
    """matplotlib's ``LogitTransform``: ``log10(p / (1 - p))``.

    With ``clip`` (upstream default for the scale) values at/beyond
    the closed interval endpoints clamp to ±1000 -- upstream's exact
    constants; the domain itself is clamped to ``[1e-7, 1 - 1e-7]``
    by ``limit_range_for_scale``.
    """
    arr = np.asarray(values, dtype=float)
    p = np.clip(arr, 1e-7, 1.0 - 1e-7)
    out = np.log10(p / (1.0 - p))
    if clip:
        out = np.clip(out, -1000.0, 1000.0)
    return out


def logit_inverse(values: object) -> np.ndarray:
    """``LogisticTransform``: ``1 / (1 + 10**(-x))``."""
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.power(10.0, -arr))


# ---------------------------------------------------------------------------
# the tick engine (matplotlib/ticker.py)
# ---------------------------------------------------------------------------

_MAXN_STEPS = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
_AUTOLocator_STEPS = np.array([1.0, 2.0, 2.5, 5.0, 10.0])


def _scale_range(vmin: float, vmax: float, n: float = 1.0,
                 threshold: float = 100.0) -> tuple[float, float]:
    """matplotlib's ``scale_range``: factor out decade offset and scale."""
    dv = abs(vmax - vmin)
    meanv = (vmax + vmin) / 2.0
    ratio = abs(meanv) / dv if dv else 0.0
    if ratio < threshold:
        offset = 0.0
    else:
        offset = math.copysign(10.0 ** (math.log10(abs(meanv)) // 1), meanv)
    scale = 10.0 ** (math.log10(dv / n) // 1) if dv > 0 else 1.0
    return scale, offset


class _EdgeInteger:
    """``ticker._Edge_integer``: float-precision-safe floor/ceil in steps."""

    def __init__(self, step: float, offset: float = 0.0) -> None:
        self.step = step
        self._offset = offset

    def closeto(self, ms: float, edge: float) -> bool:
        if self._offset > 0:
            digits = np.log10(self._offset / self.step)
            tol = max(1e-10, 10.0 ** (digits - 12))
            tol = min(0.4999, tol)
        else:
            tol = 1e-10
        return abs(ms - edge) < tol

    def le(self, x: float) -> int:
        d, m = divmod(x, self.step)
        if self.closeto(m / self.step, 1.0):
            return int(d) + 1
        return int(d)

    def ge(self, x: float) -> int:
        d, m = divmod(x, self.step)
        if self.closeto(m / self.step, 0.0):
            return int(d)
        return int(d) + 1


def nice_ticks(
    vmin: float,
    vmax: float,
    nbins: int | str = 10,
    *,
    steps: object = None,
    prune: str | None = None,
    integer: bool = False,
    symmetric: bool = False,
    min_n_ticks: int = 2,
) -> np.ndarray:
    """matplotlib's ``MaxNLocator.tick_values`` -- the "nice ticks" engine.

    ``nbins`` is the target maximum (``'auto'`` behaves as 9 when no
    axis is attached); ``steps`` defaults to matplotlib's
    ``[1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]`` (``AutoLocator`` passes
    ``[1, 2, 2.5, 5, 10]``); ``prune`` in ``{'lower', 'upper',
    'both'}`` drops end ticks; ``integer`` restricts to integral
    steps; ``symmetric`` mirrors the range around zero.  Verified
    tick-for-tick against ``MaxNLocator`` in the verification probe.
    """
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmax - vmin < 1e-14:  # _nonsingular for degenerate ranges
        vmax, vmin = vmax + 1e-13, vmin - 1e-13
    if symmetric:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound
    if isinstance(nbins, str):
        if nbins != "auto":
            raise ValueError("nbins must be an int or 'auto'")
        nbins = 9
    nbins = max(int(nbins), 1)
    step_choices = _MAXN_STEPS if steps is None else np.asarray(steps, dtype=float)
    if step_choices[0] != 1.0 or step_choices[-1] != 10.0:
        step_choices = np.clip(np.concatenate([[1.0], step_choices, [10.0]]), 1.0, 10.0)
    extended = np.concatenate([0.1 * step_choices[:-1], step_choices, [10.0 * step_choices[1]]])
    scale, offset = _scale_range(vmin, vmax, nbins)
    _vmin, _vmax = vmin - offset, vmax - offset
    step_pool = extended * scale
    if integer:
        igood = (step_pool < 1) | (np.abs(step_pool - np.round(step_pool)) < 0.001)
        step_pool = step_pool[igood]
    raw_step = (_vmax - _vmin) / nbins
    large = step_pool >= raw_step
    if np.any(large):
        istep = int(np.nonzero(large)[0][0])
    else:
        istep = len(step_pool) - 1
    ticks: np.ndarray = np.array([])
    for step in step_pool[: istep + 1][::-1]:
        if integer and np.floor(_vmax) - np.ceil(_vmin) >= min_n_ticks - 1:
            step = max(1.0, step)
        best_vmin = (_vmin // step) * step
        edge = _EdgeInteger(step, offset)
        low = edge.le(_vmin - best_vmin)
        high = edge.ge(_vmax - best_vmin)
        ticks = np.arange(low, high + 1) * step + best_vmin
        inside = (ticks <= _vmax) & (ticks >= _vmin)
        if int(inside.sum()) >= min_n_ticks:
            break
    ticks = ticks + offset
    if prune == "lower":
        ticks = ticks[1:]
    elif prune == "upper":
        ticks = ticks[:-1]
    elif prune == "both":
        ticks = ticks[1:-1]
    return ticks


def multiple_ticks(vmin: float, vmax: float, base: float, offset: float = 0.0) -> np.ndarray:
    """matplotlib's ``MultipleLocator``: the arithmetic grid of ``base``."""
    if base <= 0:
        raise ValueError("base must be positive")
    step = float(base)
    edge = _EdgeInteger(step)
    low = edge.le(vmin - offset)
    high = edge.ge(vmax - offset)
    return np.arange(low, high + 1) * step + offset


def log_ticks(vmin: float, vmax: float, numticks: int = 9, base: float = 10.0) -> np.ndarray:
    """matplotlib 3.9's ``LogLocator.tick_values``: the decade formula.

    ``numdec = floor(log vmax) - ceil(log vmin)`` decades are available;
    the stride is ``numdec // numticks + 1`` (clipped back to
    ``max(1, numdec - 1)`` when it would swallow the whole range), and
    the decade grid deliberately runs one stride *outside* each limit
    -- upstream's own behavior (``log_ticks(0.01, 100)`` spans
    ``[1e-3, 1e3]``).  Verified tick-for-tick against the installed
    ``matplotlib.ticker.LogLocator`` across ranges and tick budgets.
    """
    if numticks < 1:
        raise ValueError("numticks must be >= 1")
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmin <= 0:
        raise ValueError("log_ticks needs positive limits (upstream refuses too)")
    log_base = math.log(base)
    log_vmin = math.log(vmin) / log_base
    log_vmax = math.log(vmax) / log_base
    numdec = math.floor(log_vmax) - math.ceil(log_vmin)
    stride = numdec // numticks + 1
    if stride >= numdec:
        stride = max(1, numdec - 1)
    decades = np.arange(math.floor(log_vmin) - stride, math.ceil(log_vmax) + 2 * stride, stride)
    return base**decades


def format_scalar_ticks(ticks: Any) -> list[str]:
    """matplotlib ``ScalarFormatter``'s significant-digit loop.

    The significant digits are chosen from the tick *range* (three
    significant digits on the smallest tick gap), then trimmed while
    the rounded ticks round-trip within ``1e-3 * 10^oom``.  The
    offset/scientific-notation half of upstream (the ``1e6 + 1.5``
    style axis annotations) is NOT carried -- plain per-tick strings
    only, stated here so nobody expects an offset line.
    """
    arr: np.ndarray = np.asarray([float(t) for t in ticks])
    if arr.size == 0:
        return []
    if arr.size < 2:
        arr = np.concatenate([arr, arr + 1.0])
    loc_range = float(np.ptp(arr))
    if loc_range == 0:
        loc_range = float(np.max(np.abs(arr))) or 1.0
    loc_range_oom = math.floor(math.log10(loc_range))
    sigfigs = max(0, 3 - loc_range_oom)
    thresh = 1e-3 * 10.0**loc_range_oom
    while sigfigs >= 0:
        rounded = np.round(arr, decimals=sigfigs).astype(float)
        if np.abs(arr - rounded).max() < thresh:
            sigfigs -= 1
        else:
            break
    sigfigs += 1
    return [f"%1.{sigfigs}f" % tick for tick in arr]
