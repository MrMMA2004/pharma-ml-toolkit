"""Headless figures for molecular-dynamics metric series and correlations.

The plots consume :class:`~pharmaml.md.XvgSeries` objects directly and do
not convert units, resample, smooth, or infer axis units. Specify axis
labels matching the actual data. In particular, ``read_ligand_folder``
may normalize only some time axes; inspect the data before labeling them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.md import LigandResult, XvgSeries, metric_correlations
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = ["plot_md_correlation", "plot_md_series"]


def plot_md_series(
    series: XvgSeries | Mapping[str, XvgSeries],
    *,
    title: str = "MD metric series",
    x_label: str = "Time / index",
    y_label: str = "Value",
) -> Any:
    """Plot one MD metric series or a named comparison of series.

    The caller must ensure that compared series represent compatible metrics
    and units. Raw values are drawn without conversion or smoothing. A mapping
    supplies the legend labels, for example ``study.metric_series('ligand_rmsd')``.
    A single series uses its label or source, if present, as a legend label.
    The returned Matplotlib figure can be saved with ``save_figure`` and
    should be closed with ``close_figure`` when finished.

    Raises DataValidationError for empty mappings, non-series inputs, or
    non-finite time/value data.
    """
    if isinstance(series, XvgSeries):
        items = [(series.label or series.source, series)]
    elif isinstance(series, Mapping) and series:
        items = list(series.items())
    else:
        raise DataValidationError("series must be an XvgSeries or a non-empty mapping")
    for label, item in items:
        if not isinstance(label, str) or not isinstance(item, XvgSeries):
            raise DataValidationError("series mapping must contain string labels and XvgSeries values")
        if not np.isfinite(item.time).all() or not np.isfinite(item.values).all():
            raise DataValidationError(f"series {label!r} contains non-finite time or values")

    fig, ax = new_figure(figsize=(8.0, 4.5))
    for index, (label, item) in enumerate(items):
        ax.plot(item.time, item.values, color=PALETTE[index % len(PALETTE)], label=label or None)
    ax.set_title(title, loc="left")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if any(label for label, _ in items):
        ax.legend()
    fig.tight_layout()
    return fig


def plot_md_correlation(
    series: LigandResult | Mapping[str, XvgSeries],
    *,
    title: str = "MD metric correlations",
) -> Any:
    """Plot a Pearson correlation heatmap for aligned MD metric series.

    Accepts a ligand result or a mapping of metric names to series. At least
    two series and two observations are required. All time/index arrays must
    match exactly: this function never truncates, aligns or interpolates.
    Compare only measurements on the same meaningful axis; for example,
    RMSF indexed by residue should not be correlated with a time series.
    All values must be finite. Constant metrics have undefined correlations,
    displayed as ``N/A`` rather than zero. The result is descriptive, not
    evidence of causation or independent samples.
    """
    metrics = series.series if isinstance(series, LigandResult) else series
    if not isinstance(metrics, Mapping) or len(metrics) < 2:
        raise DataValidationError("at least two named MD series are required")
    for label, item in metrics.items():
        if not isinstance(label, str) or not isinstance(item, XvgSeries):
            raise DataValidationError("series mapping must contain string labels and XvgSeries values")
        if not np.isfinite(item.time).all() or not np.isfinite(item.values).all():
            raise DataValidationError(f"series {label!r} contains non-finite time or values")
    reference = next(iter(metrics.values())).time
    if len(reference) < 2:
        raise DataValidationError("at least two observations per series are required")
    if any(not np.array_equal(reference, item.time) for item in metrics.values()):
        raise DataValidationError("MD series axes must match exactly; align data before plotting")

    with np.errstate(divide="ignore", invalid="ignore"):
        result = metric_correlations(dict(metrics))
    assert result is not None  # At least two non-empty, equally sized series.
    labels, matrix = result
    fig, ax = new_figure(figsize=(max(6.0, len(labels) * 0.85), max(4.5, len(labels) * 0.75)))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.grid(False)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = matrix[row, col]
            text = f"{value:.2f}" if np.isfinite(value) else "N/A"
            ax.text(col, row, text, ha="center", va="center",
                    color="white" if np.isfinite(value) and abs(value) > 0.6 else "black")
    ax.set_title(title, loc="left")
    fig.colorbar(image, ax=ax, label="Pearson r")
    fig.tight_layout()
    return fig
