"""Visual cheminformatics output — descriptor landscapes and similarity.

The archive's descriptor calculators (``others/Ml/Descriptor
calculator.py``, ``Morgan_calculator.py``, ...) wrote wide CSVs and
stopped there; this module turns them into figures:

* :func:`plot_descriptor_distributions` — a histogram grid over the
  numeric descriptor columns of any frame (also fits pIC50 tables);
* :func:`plot_similarity_heatmap` — a Tanimoto matrix heat-map with
  optional cell annotations.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = ["plot_descriptor_distributions", "plot_similarity_heatmap"]

_LOG = get_logger(__name__)


def plot_descriptor_distributions(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    max_columns: int = 9,
    bins: int = 25,
    title: str = "Descriptor distributions",
) -> Any:
    """Histogram grid over the numeric columns of a table.

    Parameters
    ----------
    frame:
        Any numeric table — descriptor frames, pIC50 conversions, or
        prediction outputs.
    columns:
        Subset of columns to show (default: every numeric column,
        capped at ``max_columns`` by column order).
    max_columns:
        Safety cap so a 200-descriptor frame renders as a readable grid.
    bins:
        Histogram bin count.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("frame must be a pandas DataFrame")
    if frame.empty:
        raise DataValidationError("frame is empty — nothing to plot")

    if columns is not None:
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise DataValidationError(
                f"column(s) not found: {missing}. "
                f"Available: {list(frame.columns)[:20]}"
            )
        numeric = [name for name in columns
                   if pd.api.types.is_numeric_dtype(frame[name])]
        skipped = len(columns) - len(numeric)
    else:
        numeric = [name for name in frame.columns
                   if pd.api.types.is_numeric_dtype(frame[name])]
        skipped = frame.shape[1] - len(numeric)
        numeric = numeric[:max_columns]

    if not numeric:
        raise DataValidationError(
            "no numeric columns to plot (non-numeric columns are skipped)"
        )
    if skipped:
        _LOG.info("Skipping %d non-numeric column(s) in distribution grid", skipped)

    n = len(numeric)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = new_figure(nrows=nrows, ncols=ncols,
                           figsize=(3.6 * ncols, 2.9 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes_flat, numeric):
        values = frame[name].dropna().to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            ax.text(0.5, 0.5, "no finite values", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(name, fontsize=9)
            continue
        ax.hist(finite, bins=bins, color=PALETTE[0], alpha=0.85, edgecolor="white",
                linewidth=0.4)
        ax.axvline(float(np.mean(finite)), color=PALETTE[1], lw=1.4,
                   label=f"mean {np.mean(finite):.4g}")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper right")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{title} — {n} numeric column(s)", y=1.02)
    return fig


def plot_similarity_heatmap(
    similarity: np.ndarray | pd.DataFrame,
    labels: list[str] | None = None,
    annotate: bool | None = None,
    title: str = "Tanimoto similarity",
) -> Any:
    """Heat-map of a pairwise similarity matrix.

    Parameters
    ----------
    similarity:
        Square matrix, or a DataFrame whose index/columns name the
        compounds (labels are taken from it automatically).
    labels:
        Compound labels (optional when a DataFrame is passed).
    annotate:
        Print similarity values inside the cells (default: only when
        the matrix is at most 20x20).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(similarity, pd.DataFrame):
        matrix = similarity.to_numpy(dtype=float)
        if labels is None:
            labels = [str(name) for name in similarity.index]
    else:
        matrix = np.asarray(similarity, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DataValidationError(
            f"similarity matrix must be square, got shape {matrix.shape}"
        )
    n = matrix.shape[0]
    if labels is None:
        labels = [f"c{i + 1}" for i in range(n)]
    if len(labels) != n:
        raise DataValidationError(f"{len(labels)} labels for a {n}x{n} matrix")
    if annotate is None:
        annotate = n <= 20

    fig, ax = new_figure(figsize=(1.6 + 0.62 * n, 1.6 + 0.52 * n))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="similarity")

    if annotate:
        threshold = 0.5
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7 if n > 10 else 8,
                        color="white" if matrix[i, j] > threshold else "#1F2937")

    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_title(title)
    ax.grid(False)
    return fig
