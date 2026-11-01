r"""Seaborn-style statistical plots — the statistics in numpy, rendering lazy.

The seaborn round of the absorption pipeline reviewed the statistical chart
grammar — ``histplot``, ``kdeplot``, ``ecdfplot``, ``boxplot``, ``regplot``,
``clustermap``, ``PairGrid`` — not to vendor seaborn's drawing code but to
re-derive in pure numpy the statistics each figure stands on: the binning
rules (Sturges, Freedman & Diaconis 1981, sqrt, numpy's ``auto``), the
Scott/Silverman bandwidth factors, the empirical CDF, Tukey's five-number
summary with ``whis·IQR`` fences, the ``lstsq`` OLS line with seeded Efron
bootstrap intervals, UPGMA linkage seriation with the merge records, and the
m×m Pearson matrix a PairGrid annotates with *r*.

The honest scope, stated once: seaborn itself — the pandas-aware
grammar-of-graphics API, the dataset loaders, the categorical estimators and
every line of its drawing code — stays with seaborn and is deliberately not
vendored.  For the clustermap, ``pharmaml.ml.hierarchical`` turned out to be
the *HierarchicalSVR* regressor, not a linkage provider, and
:func:`pharmaml.ml.unsupervised.hierarchical_cluster` a k-cut without the
merge tree a dendrogram needs — so this module carries a compact UPGMA
(average linkage, Lance–Williams updates) instead.

Every ``plot_*`` function computes its statistics first (importing this
module never imports matplotlib), always returns that stats dict, renders
through the lazy :mod:`pharmaml.viz._compat` layer (headless ``Agg``),
saves byte-reproducible png/svg/pdf to ``path`` when given, and closes its
figure afterwards.

Examples
--------
>>> from pathlib import Path
>>> import numpy as np
>>> import pandas as pd
>>> from pharmaml.viz.statplots import (box_stats, ecdf_stats, histogram_stats,
...     kde_stats, plot_box, plot_clustermap, plot_ecdf, plot_histogram,
...     plot_pairgrid, plot_regression, regression_stats)

>>> small = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 9.0]
>>> hist = histogram_stats(small, bins="sturges")
>>> hist["n_bins"], hist["counts"].tolist(), round(hist["skew"], 4)
(4, [4, 3, 0, 1], 1.5829)
>>> (round(hist["mean"], 4), round(hist["std"], 4))
(3.3125, 2.3444)
>>> (hist["q1"], hist["median"], hist["q3"], hist["iqr"])
(1.875, 2.75, 3.625, 1.75)
>>> [histogram_stats(small, bins=r)["n_bins"] for r in ("fd", "sqrt", "auto")]
[5, 3, 5]
>>> histogram_stats(small, bins="rice")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: bins must be a positive integer or one of auto, fd, sturges, sqrt — got 'rice'

>>> kde = kde_stats([0.0, 1.0, 2.0, 3.0, 4.0], points=5)
>>> round(kde["bandwidth"], 4), kde["bandwidth_rule"], round(kde["integral"], 4)
(1.025, 'scott', 0.7112)
>>> [round(float(d), 4) for d in kde["density"]]
[0.1389, 0.1872, 0.1978, 0.1872, 0.1389]
>>> round(kde_stats([0.0, 1.0, 2.0, 3.0, 4.0], bandwidth="silverman")["bandwidth"], 4)
1.0857

>>> [round(float(p), 4) for p in ecdf_stats([3.0, 1.0, 2.0])["probabilities"]]
[0.3333, 0.6667, 1.0]

>>> box = box_stats([1.0, 2.0, 3.0, 4.0, 100.0])
>>> (box["median"], box["iqr"], box["whisker_hi"], box["outliers"].tolist())
(3.0, 2.0, 4.0, [100.0])

>>> xs = [1.0, 2.0, 3.0, 4.0, 5.0]
>>> ys = [2.1, 3.9, 6.2, 8.1, 9.7]
>>> reg = regression_stats(xs, ys, n_boot=200, seed=0)
>>> round(reg["slope"], 4), round(reg["intercept"], 4), round(reg["r2"], 4)
(1.94, 0.18, 0.9967)
>>> tuple(round(float(v), 4) for v in reg["slope_ci"])
(1.7254, 2.1251)
>>> reg["slope_ci"] == regression_stats(xs, ys, seed=0)["slope_ci"]
True
>>> regression_stats(xs, [1.0, 2.0])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: regression_stats: x (5) and y (2) must have the same length

>>> plot_histogram(small, bins="sturges", path="/tmp/pharmaml_doctest_hist.png")["counts"].tolist()
[4, 3, 0, 1]
>>> Path("/tmp/pharmaml_doctest_hist.png").unlink()
>>> plot_ecdf(small)["n"]
8
>>> plot_box([1.0, 2.0, 3.0, 4.0, 100.0])["outliers"].tolist()
[100.0]
>>> round(plot_regression(xs, ys, n_boot=100, seed=0)["slope"], 4)
1.94

>>> matrix = np.array([[0.0, 0.1, 5.0, 5.1], [0.2, 0.0, 5.2, 5.0],
...                    [0.1, 0.2, 5.1, 5.2], [8.0, 8.1, 1.0, 1.1],
...                    [8.2, 8.0, 1.2, 1.0], [8.1, 8.2, 1.1, 1.2]])
>>> stats = plot_clustermap(matrix, path="/tmp/pharmaml_doctest_clustermap.png")
>>> stats["row_order"].tolist(), stats["col_order"].tolist()
([1, 0, 2, 4, 3, 5], [0, 1, 2, 3])
>>> round(stats["row_merges"][-1][2], 4)
12.6512
>>> Path("/tmp/pharmaml_doctest_clustermap.png").unlink()
>>> plot_clustermap(matrix, ["a"], ["u", "v", "w", "z"])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: plot_clustermap: row_labels needs 6 label(s), got 1

>>> frame = pd.DataFrame({"logp": xs, "mw": [2.0 * v for v in xs], "tpsa": [-v for v in xs]})
>>> grid = plot_pairgrid(frame)
>>> grid["names"], grid["r"][0].round(3).tolist()
(['logp', 'mw', 'tpsa'], [1.0, 1.0, -1.0])
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.viz._compat import PALETTE, close_figure, new_figure, save_figure

__all__ = [
    "box_stats",
    "ecdf_stats",
    "histogram_stats",
    "kde_stats",
    "plot_box",
    "plot_clustermap",
    "plot_ecdf",
    "plot_histogram",
    "plot_pairgrid",
    "plot_regression",
    "regression_stats",
]

#: Selectable automatic-binning rules (the numpy/seaborn vocabulary).
_BIN_RULES: tuple[str, ...] = ("auto", "fd", "sturges", "sqrt")

#: Dark ink for box medians and dendrogram lines across this module.
_INK = PALETTE[2]

#: Grid points the regression line and its bootstrap band are evaluated on.
_REG_GRID = 50


# --- validation helpers ----------------------------------------------------


def _as_1d_float(values: Any, caller: str, min_n: int = 2) -> np.ndarray:
    """Validate ``values`` → a finite float64 1-D array (typed errors)."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"{caller}: values must be numeric — conversion failed: {exc}"
        ) from exc
    if array.ndim != 1:
        raise DataValidationError(
            f"{caller}: values must be 1-dimensional, got {array.ndim} dimension(s)"
        )
    if array.shape[0] < min_n:
        raise DataValidationError(
            f"{caller}: needs at least {min_n} value(s), got {array.shape[0]}"
        )
    if not np.isfinite(array).all():
        raise DataValidationError(
            f"{caller}: values contain NaN or infinite entries — drop or impute them"
        )
    return array


def _as_xy_pair(x: Any, y: Any, caller: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate two aligned 1-D numeric arrays of equal length."""
    x_arr = _as_1d_float(x, caller)
    y_arr = _as_1d_float(y, caller)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise DataValidationError(
            f"{caller}: x ({x_arr.shape[0]}) and y ({y_arr.shape[0]}) must have "
            "the same length"
        )
    return x_arr, y_arr


def _as_2d_matrix(data: Any, caller: str) -> tuple[np.ndarray, list[str]]:
    """Validate a DataFrame or 2-D array → (float64 matrix, column names)."""
    if isinstance(data, pd.DataFrame):
        names = [str(column) for column in data.columns]
        raw = data.to_numpy()
    else:
        raw = np.asarray(data)
        if raw.ndim != 2:
            raise DataValidationError(
                f"{caller}: data must be 2-dimensional, got {raw.ndim} dimension(s)"
            )
        names = [f"x{index}" for index in range(raw.shape[1])]
    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"{caller}: data must be numeric — conversion failed: {exc}"
        ) from exc
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise DataValidationError(f"{caller}: data is empty — got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise DataValidationError(
            f"{caller}: data contains NaN or infinite entries — drop or impute them"
        )
    return matrix, names


# --- statistics (pure numpy — matplotlib is never touched above this line) -


def _bin_count(values: np.ndarray, rule: str) -> int:
    """Bin count for one automatic rule (Sturges/FD/sqrt/auto).

    The Freedman–Diaconis width is ``2·IQR/n^(1/3)`` with a population-std
    fallback when the IQR collapses (heavy ties); a constant sample falls
    back to Sturges, exactly like numpy.
    """
    n = values.size
    span = float(values.max() - values.min())
    sturges = max(1, math.ceil(math.log2(n) + 1.0))
    if rule == "sturges":
        return sturges
    if rule == "sqrt":
        return max(1, math.ceil(math.sqrt(n)))
    q1, q3 = np.quantile(values, (0.25, 0.75), method="linear")
    spread = float(q3 - q1)
    if spread == 0.0:
        spread = float(values.std())
    width = 2.0 * spread / n ** (1.0 / 3.0) if spread > 0.0 else 0.0
    if width <= 0.0 or span <= 0.0:
        return sturges  # FD is undefined — numpy's fallback
    fd_bins = max(1, math.ceil(span / width))
    return fd_bins if rule == "fd" else max(fd_bins, sturges)  # numpy 'auto'


def histogram_stats(values: Any, bins: str | int = "auto") -> dict[str, Any]:
    """Histogram statistics with selectable automatic binning.

    ``bins`` is one of ``"auto"`` (numpy's max of FD and Sturges, the
    default), ``"fd"`` (Freedman & Diaconis 1981), ``"sturges"`` or
    ``"sqrt"``, or a fixed positive integer count.  Returns ``n``, ``rule``
    (``"fixed"`` for an integer), ``n_bins``, ``edges``, ``counts``,
    ``centers``, ``bin_width`` and the summary ``mean``, ``std``
    (population), ``min``, ``max``, ``q1``, ``median``, ``q3``, ``iqr``,
    ``skew`` (g1).  Raises :class:`~pharmaml.exceptions.DataValidationError`
    on malformed input or an unknown rule.
    """
    array = _as_1d_float(values, "histogram_stats")
    message = (
        f"bins must be a positive integer or one of {', '.join(_BIN_RULES)} "
        f"— got {bins!r}"
    )
    if isinstance(bins, str):
        if bins not in _BIN_RULES:
            raise DataValidationError(message)
        n_bins, rule = _bin_count(array, bins), bins
    elif isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if int(bins) < 1:
            raise DataValidationError(message)
        n_bins, rule = int(bins), "fixed"
    else:
        raise DataValidationError(message)

    lo, hi = float(array.min()), float(array.max())
    if lo == hi:  # constant sample: one honest bin around the value
        n_bins, edges = 1, np.array([lo - 0.5, hi + 0.5])
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(array, bins=edges)
    mean = float(array.mean())
    centered = array - mean
    m2 = float((centered**2).mean())
    q1, median, q3 = (
        float(v) for v in np.quantile(array, (0.25, 0.5, 0.75), method="linear")
    )
    return {
        "n": int(array.size), "rule": rule, "n_bins": int(n_bins),
        "edges": edges, "counts": counts.astype(np.int64),
        "centers": (edges[:-1] + edges[1:]) / 2.0,
        "bin_width": float(edges[1] - edges[0]), "mean": mean,
        "std": float(math.sqrt(m2)), "min": lo, "max": hi,
        "q1": q1, "median": median, "q3": q3, "iqr": q3 - q1,
        "skew": float((centered**3).mean() / m2**1.5) if m2 > 0 else 0.0,
    }


def kde_stats(
    values: Any, bandwidth: str | float = "scott", points: int = 64
) -> dict[str, Any]:
    """Gaussian kernel density estimate on an even grid over the data range.

    The bandwidth follows the ``scipy.stats.gaussian_kde`` factors on the
    population std — Scott's ``n^(-1/5)`` or Silverman's ``(3n/4)^(-1/5)``
    — or an absolute smoothing scale as a float.  Returns ``n``,
    ``bandwidth``, ``bandwidth_rule``, ``grid``, ``density`` and
    ``integral`` (trapezoid rule on the grid).  Raises
    :class:`~pharmaml.exceptions.DataValidationError` on constant input
    (the density would be a spike), an unknown rule or ``points < 2``.
    """
    array = _as_1d_float(values, "kde_stats")
    n, std = array.size, float(array.std())
    if std == 0.0:
        raise DataValidationError(
            "kde_stats refuses constant input (std=0): the density is a "
            "spike, not a curve — drop or jitter the values"
        )
    if isinstance(bandwidth, str):
        try:
            factor = {
                "scott": n ** (-1.0 / 5.0),
                "silverman": (3.0 * n / 4.0) ** (-1.0 / 5.0),
            }[bandwidth]
        except KeyError as exc:
            raise DataValidationError(
                f"bandwidth must be 'scott', 'silverman' or a positive float, "
                f"got {bandwidth!r}"
            ) from exc
        h, rule = float(factor * std), bandwidth
    elif isinstance(bandwidth, (int, float, np.integer, np.floating)) and not isinstance(
        bandwidth, bool
    ):
        h = float(bandwidth)
        if not math.isfinite(h) or h <= 0.0:
            raise DataValidationError(
                f"bandwidth must be > 0, got {h!r} — choose a positive "
                "smoothing scale or a named rule"
            )
        rule = "manual"
    else:
        raise DataValidationError(
            f"bandwidth must be 'scott', 'silverman' or a positive float, "
            f"got {bandwidth!r}"
        )
    points_ok = isinstance(points, (int, np.integer)) and not isinstance(points, bool)
    if not points_ok or int(points) < 2:
        raise DataValidationError(f"points must be an integer >= 2, got {points!r}")

    grid = np.linspace(float(array.min()), float(array.max()), int(points))
    # f(x) = 1/(n h) * sum_i phi((x - x_i)/h) — the Gaussian kernel sum
    z = (grid[:, None] - array[None, :]) / h
    density = np.exp(-0.5 * z**2).sum(axis=1) / (n * h * math.sqrt(2.0 * math.pi))
    dx = float(grid[1] - grid[0])
    return {
        "n": int(n), "bandwidth": float(h), "bandwidth_rule": rule,
        "grid": grid, "density": density,
        "integral": float(dx * (density.sum() - 0.5 * (density[0] + density[-1]))),
    }


def ecdf_stats(values: Any) -> dict[str, Any]:
    """Empirical CDF — the sorted sample with ``i/n`` probabilities.

    Returns ``n``, ``values`` (sorted) and ``probabilities``; raises
    :class:`~pharmaml.exceptions.DataValidationError` on malformed input.
    """
    array = _as_1d_float(values, "ecdf_stats", min_n=1)
    n = array.size
    return {
        "n": int(n), "values": np.sort(array),
        "probabilities": np.arange(1, n + 1, dtype=np.float64) / n,
    }


def box_stats(values: Any, whis: float = 1.5) -> dict[str, Any]:
    """Tukey boxplot statistics with linear-interpolation hinges.

    The hinges use numpy's ``method="linear"`` quantiles (the
    matplotlib/seaborn convention); fences sit at ``q1 - whis·IQR`` and
    ``q3 + whis·IQR``, whiskers at the most extreme data values inside the
    fences.  Returns ``n``, ``mean``, ``min``, ``q1``, ``median``, ``q3``,
    ``max``, ``iqr``, ``lo_fence``, ``hi_fence``, ``whisker_lo``,
    ``whisker_hi`` and ``outliers`` (sorted; empty when none).  Raises
    :class:`~pharmaml.exceptions.DataValidationError` on malformed input
    or a negative ``whis``.
    """
    array = _as_1d_float(values, "box_stats", min_n=1)
    if (
        isinstance(whis, bool)
        or not isinstance(whis, (int, float, np.integer, np.floating))
        or not math.isfinite(float(whis))
        or float(whis) < 0.0
    ):
        raise DataValidationError(f"whis must be a number >= 0, got {whis!r}")
    whis = float(whis)
    q1, median, q3 = (
        float(v) for v in np.quantile(array, (0.25, 0.5, 0.75), method="linear")
    )
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - whis * iqr, q3 + whis * iqr
    inside = array[(array >= lo_fence) & (array <= hi_fence)]
    return {
        "n": int(array.size), "mean": float(array.mean()), "min": float(array.min()),
        "q1": q1, "median": median, "q3": q3, "max": float(array.max()), "iqr": iqr,
        "lo_fence": float(lo_fence), "hi_fence": float(hi_fence),
        "whisker_lo": float(inside.min()) if inside.size else q1,
        "whisker_hi": float(inside.max()) if inside.size else q3,
        "outliers": np.sort(array[(array < lo_fence) | (array > hi_fence)]),
    }


def regression_stats(
    x: Any, y: Any, n_boot: int = 200, seed: int = 0
) -> dict[str, Any]:
    """Ordinary least squares fit with an optional bootstrap confidence band.

    The line comes from ``numpy.linalg.lstsq``; the 95% intervals are Efron
    percentile bootstrap over ``n_boot`` resamples of the ``(x, y)`` pairs
    (degenerate resamples — constant x — are skipped) drawn from a seeded
    ``numpy.random.default_rng`` so results are reproducible.  Returns
    ``n``, ``slope``, ``intercept``, ``r`` (Pearson, 0 when a variable is
    constant), ``r2``, ``fitted``, ``residuals``, ``x_grid``/``y_hat`` (the
    line over :data:`_REG_GRID` points spanning x) and ``slope_ci`` /
    ``intercept_ci`` / ``band_lo`` / ``band_hi`` (``None`` when ``n_boot=0``
    or every resample was degenerate).  Raises
    :class:`~pharmaml.exceptions.DataValidationError` on malformed input,
    mismatched lengths or constant ``x``.
    """
    x_arr, y_arr = _as_xy_pair(x, y, "regression_stats")
    n = x_arr.size
    if float(x_arr.std()) == 0.0:
        raise DataValidationError(
            "regression_stats: x is constant — a slope needs variation in x"
        )
    for name, value in (("n_boot", n_boot), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise DataValidationError(f"{name} must be an integer, got {value!r}")
    if int(n_boot) < 0:
        raise DataValidationError(f"n_boot must be >= 0, got {n_boot!r}")
    n_boot = int(n_boot)

    design = np.column_stack([x_arr, np.ones(n)])
    slope, intercept = (float(v) for v in np.linalg.lstsq(design, y_arr, rcond=None)[0])
    fitted = intercept + slope * x_arr
    r = float(np.corrcoef(x_arr, y_arr)[0, 1]) if float(y_arr.std()) > 0.0 else 0.0
    x_grid = np.linspace(float(x_arr.min()), float(x_arr.max()), _REG_GRID)

    slope_ci = intercept_ci = band_lo = band_hi = None
    if n_boot:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, n, size=(n_boot, n))
        xb, yb = x_arr[indices], y_arr[indices]
        mx, my = xb.mean(axis=1), yb.mean(axis=1)
        cov = ((xb - mx[:, None]) * (yb - my[:, None])).sum(axis=1)
        var = ((xb - mx[:, None]) ** 2).sum(axis=1)
        ok = var > 0.0
        if ok.any():
            boot_slopes = cov[ok] / var[ok]
            boot_intercepts = my[ok] - boot_slopes * mx[ok]
            lines = boot_intercepts[:, None] + boot_slopes[:, None] * x_grid[None, :]
            slope_ci = tuple(float(np.percentile(boot_slopes, q)) for q in (2.5, 97.5))
            intercept_ci = tuple(
                float(np.percentile(boot_intercepts, q)) for q in (2.5, 97.5)
            )
            band_lo, band_hi = (
                np.percentile(lines, 2.5, axis=0),
                np.percentile(lines, 97.5, axis=0),
            )
    return {
        "n": int(n), "slope": slope, "intercept": intercept, "r": r, "r2": r * r,
        "fitted": fitted, "residuals": y_arr - fitted, "x_grid": x_grid,
        "y_hat": intercept + slope * x_grid, "slope_ci": slope_ci,
        "intercept_ci": intercept_ci, "band_lo": band_lo, "band_hi": band_hi,
    }


# --- hierarchical ordering (compact UPGMA for the clustermap) --------------


def _upgma(matrix: np.ndarray) -> tuple[list[tuple[int, int, float, int]], list[int]]:
    """Average-linkage (UPGMA) agglomeration of the matrix rows.

    Lance–Williams updates over the full cluster-distance matrix (the
    textbook O(n^3) form — the same mathematics scipy's linkage runs at
    O(n^2); teaching scale).  Cluster ids below ``n`` are input rows, ids
    >= ``n`` are merged clusters.  Returns ``(merges, order)``: the
    scipy-like ``(left, right, height, size)`` records in agglomeration
    order and the left-to-right dendrogram leaf order.
    """
    n = matrix.shape[0]
    total = 2 * n - 1
    squared = np.sum(matrix**2, axis=1)
    dist_sq = squared[:, None] + squared[None, :] - 2.0 * (matrix @ matrix.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)
    dist = np.full((total, total), np.inf)
    dist[:n, :n] = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    sizes = [1] * n + [0] * (n - 1)
    children: dict[int, tuple[int, int]] = {}
    merges: list[tuple[int, int, float, int]] = []
    active = list(range(n))
    next_id = n
    while len(active) > 1:
        block = dist[np.ix_(active, active)]
        i, j = np.unravel_index(np.argmin(block), block.shape)
        a, b = active[i], active[j]  # row-major argmin = the first minimal pair
        merged = next_id
        next_id += 1
        for c in active:
            if c not in (a, b):
                dist[merged, c] = dist[c, merged] = (
                    sizes[a] * dist[a, c] + sizes[b] * dist[b, c]
                ) / (sizes[a] + sizes[b])
        dist[merged, merged] = math.inf
        sizes[merged] = sizes[a] + sizes[b]
        children[merged] = (a, b)
        merges.append((a, b, float(dist[a, b]), sizes[merged]))
        active = [c for c in active if c not in (a, b)]
        active.append(merged)

    order: list[int] = []
    stack = [total - 1]  # left-to-right leaf walk of the merge tree
    while stack:
        cluster_id = stack.pop()
        if cluster_id in children:
            left, right = children[cluster_id]
            stack.extend((right, left))
        else:
            order.append(cluster_id)
    return merges, order


def _clustermap_stats(matrix: Any) -> dict[str, Any]:
    """Matrix reordered by hierarchical linkage, with the merge records.

    Rows and columns are each seriated by :func:`_upgma` (average linkage
    over Euclidean row/column distances — see the module docstring for why
    the toolkit's ``hierarchical_cluster`` k-cut is not enough: a
    dendrogram needs the full merge tree).
    """
    array, _names = _as_2d_matrix(matrix, "clustermap")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise DataValidationError(
            f"clustermap: matrix must be at least 2x2, got {array.shape}"
        )
    row_merges, row_order = _upgma(array)
    col_merges, col_order = _upgma(array.T)
    return {
        "n_rows": int(array.shape[0]), "n_cols": int(array.shape[1]),
        "row_order": np.asarray(row_order, dtype=int),
        "col_order": np.asarray(col_order, dtype=int),
        "row_merges": row_merges, "col_merges": col_merges,
        "matrix": array[np.ix_(row_order, col_order)],
    }


def _pairgrid_stats(data: Any) -> dict[str, Any]:
    """Pearson correlation matrix (plus the matrix) for a PairGrid panel."""
    matrix, names = _as_2d_matrix(data, "pairgrid")
    if matrix.shape[1] < 2:
        raise DataValidationError(
            f"pairgrid: needs at least 2 columns, got {matrix.shape[1]}"
        )
    if matrix.shape[0] < 2:
        raise DataValidationError(
            f"pairgrid: needs at least 2 rows, got {matrix.shape[0]}"
        )
    if not np.all(matrix.std(axis=0) > 0.0):
        raise DataValidationError(
            "pairgrid: constant column(s) — Pearson r is undefined; "
            "drop or jitter the constant columns"
        )
    return {
        "n": int(matrix.shape[0]), "names": names,
        "r": np.corrcoef(matrix, rowvar=False), "matrix": matrix,
    }


# --- rendering (matplotlib imported lazily via pharmaml.viz._compat) -------


def _finish_render(figure: Any, path: str | Path | None) -> None:
    """Save to ``path`` (png/svg/pdf) when given and close the figure."""
    if path is not None:
        save_figure(figure, path)
    close_figure(figure)


def plot_histogram(
    values: Any,
    path: str | Path | None = None,
    *,
    bins: str | int = "auto",
    color: str = PALETTE[0],
) -> dict[str, Any]:
    """Histogram (seaborn ``histplot``) drawn from :func:`histogram_stats`.

    The bars honour the computed edges exactly.  Returns
    :func:`histogram_stats`' keys.  Raises
    :class:`~pharmaml.exceptions.DataValidationError` on malformed input or
    :class:`~pharmaml.exceptions.DependencyError` when matplotlib is missing.
    """
    results = histogram_stats(values, bins=bins)
    figure, ax = new_figure(figsize=(7.0, 4.5))
    ax.bar(
        results["centers"], results["counts"].astype(float),
        width=0.94 * results["bin_width"], color=color, edgecolor="white",
        linewidth=0.6, label="count",
    )
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.set_title(
        f"Histogram — n={results['n']}, mean={results['mean']:.3g}, "
        f"std={results['std']:.3g}"
    )
    _finish_render(figure, path)
    return results


def plot_ecdf(
    values: Any, path: str | Path | None = None, *, color: str = PALETTE[0]
) -> dict[str, Any]:
    """Empirical CDF as a post-step (seaborn's ``ecdfplot``).

    Returns :func:`ecdf_stats`' keys; raises
    :class:`~pharmaml.exceptions.DependencyError` when matplotlib is missing.
    """
    results = ecdf_stats(values)
    figure, ax = new_figure(figsize=(7.0, 4.5))
    ax.step(results["values"], results["probabilities"], where="post", color=color)
    ax.plot(
        results["values"], results["probabilities"], "o", ms=3, color=color, alpha=0.6
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("value")
    ax.set_ylabel("F(x)")
    ax.set_title(f"ECDF — n={results['n']}")
    _finish_render(figure, path)
    return results


def plot_box(
    values: Any,
    path: str | Path | None = None,
    *,
    whis: float = 1.5,
    color: str = PALETTE[0],
) -> dict[str, Any]:
    """Tukey box rendered from :func:`box_stats` via ``Axes.bxp``.

    The five-number summary, whiskers and fliers are exactly the computed
    statistics (``bxp`` draws a supplied summary — the contract
    matplotlib's own ``boxplot`` uses internally).  Returns
    :func:`box_stats`' keys.  Raises
    :class:`~pharmaml.exceptions.DependencyError` when matplotlib is missing.
    """
    results = box_stats(values, whis=whis)
    figure, ax = new_figure(figsize=(5.4, 4.6))
    summary = {
        "label": ["values"], "med": [results["median"]], "q1": [results["q1"]],
        "q3": [results["q3"]], "whislo": [results["whisker_lo"]],
        "whishi": [results["whisker_hi"]], "fliers": [results["outliers"]],
        "mean": [results["mean"]],
    }
    artists = ax.bxp([summary], widths=0.45, patch_artist=True, showmeans=True)
    artists["boxes"][0].set_facecolor(color)
    artists["boxes"][0].set_alpha(0.45)
    artists["medians"][0].set_color(_INK)
    ax.set_ylabel("value")
    ax.set_title(
        f"Box — median={results['median']:.3g}, IQR={results['iqr']:.3g}, "
        f"{results['outliers'].size} outlier(s)"
    )
    _finish_render(figure, path)
    return results


def plot_regression(
    x: Any,
    y: Any,
    path: str | Path | None = None,
    *,
    n_boot: int = 200,
    seed: int = 0,
    color: str = PALETTE[0],
) -> dict[str, Any]:
    """Scatter + OLS line + bootstrap band (seaborn's ``regplot``).

    Returns :func:`regression_stats`' keys; raises
    :class:`~pharmaml.exceptions.DependencyError` when matplotlib is missing.
    """
    results = regression_stats(x, y, n_boot=n_boot, seed=seed)
    x_arr, y_arr = _as_xy_pair(x, y, "plot_regression")
    figure, ax = new_figure(figsize=(7.0, 5.0))
    ax.scatter(x_arr, y_arr, s=22, color=color, alpha=0.8, label="observations")
    ax.plot(
        results["x_grid"], results["y_hat"], color=PALETTE[1], lw=2.0, label="OLS fit"
    )
    if results["band_lo"] is not None:
        ax.fill_between(
            results["x_grid"], results["band_lo"], results["band_hi"],
            color=PALETTE[1], alpha=0.18, label="95% bootstrap CI",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"OLS — ŷ = {results['slope']:.3g}·x + {results['intercept']:.3g} "
        f"(r² = {results['r2']:.3g})"
    )
    ax.legend(frameon=False, fontsize=9)
    _finish_render(figure, path)
    return results


def _draw_dendrogram(
    ax: Any, merges: list[tuple[int, int, float, int]], order: list[int], vertical: bool
) -> None:
    """Draw one UPGMA dendrogram — vertical above the heatmap (leaf rank on
    x, height growing up) or horizontal beside it (height on x, rank on y,
    both inverted so the root sits in the axis-origin corner)."""
    positions = {leaf: (float(rank), 0.0) for rank, leaf in enumerate(order)}
    for k, (a, b, height, _size) in enumerate(merges):
        xa, ya = positions[a]
        xb, yb = positions[b]
        if vertical:
            ax.plot([xa, xa, xb, xb], [ya, height, height, yb], color=_INK, lw=1.2)
        else:
            ax.plot([ya, height, height, yb], [xa, xa, xb, xb], color=_INK, lw=1.2)
        positions[len(order) + k] = ((xa + xb) / 2.0, float(height))
    extent = 1.05 * max((h for _, _, h, _ in merges), default=1.0)
    if vertical:
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(0.0, extent)
    else:
        ax.set_xlim(extent, 0.0)  # inverted: the root sits far left
        ax.set_ylim(len(order) - 0.5, -0.5)  # inverted: the first leaf on top
    ax.set_xticks([])
    ax.set_yticks([])


def _axis_labels(labels: Sequence[str] | None, n: int, prefix: str) -> list[str]:
    """Resolve heatmap tick labels — given names or generated ``r0..`` ones."""
    if labels is None:
        return [f"{prefix}{index}" for index in range(n)]
    if len(labels) != n:
        raise DataValidationError(
            f"plot_clustermap: {prefix}_labels needs {n} label(s), got {len(labels)}"
        )
    return [str(label) for label in labels]


def plot_clustermap(
    matrix: Any,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Heatmap with row/column dendrograms (seaborn's ``clustermap``).

    The matrix is reordered by the compact UPGMA linkage (see the module
    docstring); the 2x2 grid places the column dendrogram above the heatmap
    and the row dendrogram to its left, plus a colour bar.  Returns
    :func:`_clustermap_stats`' keys (``n_rows``, ``n_cols``,
    ``row_order``/``col_order``, ``row_merges``/``col_merges``, ``matrix``).
    Raises :class:`~pharmaml.exceptions.DataValidationError` on a malformed
    matrix or mismatched label lengths, or
    :class:`~pharmaml.exceptions.DependencyError` when matplotlib is missing.
    """
    results = _clustermap_stats(matrix)
    n_rows, n_cols = results["n_rows"], results["n_cols"]
    row_names = _axis_labels(row_labels, n_rows, "row")
    col_names = _axis_labels(col_labels, n_cols, "col")
    figure, axes = new_figure(
        2, 2, figsize=(8.5, 7.0), squeeze=False,
        gridspec_kw={"width_ratios": [1, 4], "height_ratios": [1, 4]},
    )
    _draw_dendrogram(axes[0][1], results["col_merges"], results["col_order"].tolist(), True)
    _draw_dendrogram(axes[1][0], results["row_merges"], results["row_order"].tolist(), False)
    ax = axes[1][1]
    image = ax.imshow(
        results["matrix"], aspect="auto", cmap="viridis", interpolation="nearest"
    )
    ax.set_xticks(range(n_cols), [col_names[i] for i in results["col_order"]])
    ax.set_yticks(range(n_rows), [row_names[i] for i in results["row_order"]])
    figure.colorbar(image, ax=ax, shrink=0.85)
    axes[0][0].axis("off")
    _finish_render(figure, path)
    return results


def plot_pairgrid(frame: Any, path: str | Path | None = None) -> dict[str, Any]:
    """Scatter matrix with Pearson r in the upper triangle (``PairGrid``).

    Diagonal cells are histograms, the lower triangle holds the
    (x = column variable, y = row variable) scatters, and each
    upper-triangle cell annotates the pairwise Pearson *r*.  Returns
    :func:`_pairgrid_stats`' keys (``n``, ``names``, ``r``).  Raises
    :class:`~pharmaml.exceptions.DataValidationError` on malformed input
    (fewer than 2 rows/columns, a constant column) or
    :class:`~pharmaml.exceptions.DependencyError` when matplotlib is missing.
    """
    results = _pairgrid_stats(frame)
    matrix, names, correlations = results["matrix"], results["names"], results["r"]
    m, n = len(names), matrix.shape[0]
    figure, axes = new_figure(m, m, figsize=(2.4 * m, 2.4 * m), squeeze=False)
    bins = min(20, max(4, math.ceil(math.sqrt(n))))
    for i in range(m):
        for j in range(m):
            ax = axes[i][j]
            if i == j:
                ax.hist(matrix[:, i], bins=bins, color=PALETTE[0], alpha=0.75)
            elif i > j:
                ax.scatter(matrix[:, j], matrix[:, i], s=10, alpha=0.75, color=PALETTE[0])
            else:
                ax.text(
                    0.5, 0.5, f"r = {correlations[i, j]:.2f}", ha="center", va="center",
                    transform=ax.transAxes, color=_INK,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            if i == m - 1:
                ax.set_xlabel(names[j])
            if j == 0:
                ax.set_ylabel(names[i])
    _finish_render(figure, path)
    return results
