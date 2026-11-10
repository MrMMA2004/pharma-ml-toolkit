r"""Molecular-dynamics plots — the viz twins of the uploaded MD pipeline.

The uploaded ``MD_analysis.py`` script (productionized as
:mod:`pharmaml.md`) drew every parsed GROMACS export as a matplotlib
panel — the raw curve, an optional moving average, a mean reference
line — and closed each ligand with a cross-metric correlation heatmap.
Round 249 shipped the pipeline's *numbers* (:mod:`pharmaml.md`) and
gave the panels to the web Molecular-Dynamics tab; this module brings
the two headline panels back as first-class Python figures on the
shared :mod:`pharmaml.viz` contract, so scripts, notebooks and the HTML
reports tell the same story the web tab does — from the same
computed-once numbers:

* :func:`plot_md_series` — one parsed metric series as a line chart with
  a mean reference line and the uploaded smoothing available as an
  explicit overlay; title and axis labels come from
  :func:`pharmaml.md.xvg.classify_metric` when the series label is one
  of the ten standard GROMACS/MM-PBSA exports;
* :func:`plot_md_correlation` — the cross-metric Pearson heatmap over a
  ligand's ``{name: XvgSeries}`` mapping, computed by the pipeline's own
  :func:`pharmaml.md.study.metric_correlations` (min-length alignment —
  the uploaded script's rule) and annotated cell by cell, with NaN
  cells (a constant series' row) disclosed in the returned stats.

Honest scope, stated once: the statistics are numpy-first and only the
drawing touches matplotlib (lazy, headless ``Agg``); GROMACS itself,
MDAnalysis and mdtraj stay external — this module honours the
plain-text ``.xvg``/``.ndx`` contract those tools export, it does not
re-render their GUIs.  Every function saves byte-reproducible
png/svg/pdf to ``path`` when given and closes its figure afterwards.

Examples
--------
>>> import numpy as np
>>> from pharmaml.md.xvg import XvgSeries
>>> from pharmaml.viz.md_plots import plot_md_correlation, plot_md_series
>>> t = np.arange(0.0, 2000.0, 100.0)  # ps upstream, already in ns here
>>> drift = XvgSeries(time=t, values=0.1 + 0.0005 * t, label="ligand_rmsd")
>>> stats = plot_md_series(drift)
>>> (stats["n"], round(stats["mean"], 4), round(stats["std"], 4))
(20, 0.575, 0.2883)
>>> (stats["title"], stats["y_label"], stats["color"])
('Ligand RMSD', 'RMSD (Å)', '#7F1D1D')
>>> smoothed = plot_md_series(drift, smooth_window=5,
...                           path="/tmp/pharmaml_doctest_md.png")
>>> (smoothed["smooth_window"], len(smoothed["smoothed"]),
...  smoothed["path"] is not None)
(5, 20, True)
>>> import os
>>> os.path.exists("/tmp/pharmaml_doctest_md.png")
True
>>> os.unlink("/tmp/pharmaml_doctest_md.png")
>>> wiggle = XvgSeries(time=t, values=1.0 + np.sin(t / 300.0),
...                    label="hbnum_intermolecular")
>>> corr = plot_md_correlation({"hbnum_intermolecular": wiggle,
...                             "ligand_rmsd": drift})
>>> (corr["labels"], corr["n_series"], corr["n_points_aligned"])
(['hbnum_intermolecular', 'ligand_rmsd'], 2, 20)
>>> round(float(corr["r"][0, 1]), 4)
-0.704
>>> (corr["nan_cells"], corr["display_labels"])
(0, ['hbnum intermolecular', 'ligand rmsd'])
>>> plot_md_series(drift, smooth_window=1)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: smooth_window must be an integer >= 2 to average anything — got 1
>>> plot_md_correlation({"ligand_rmsd": drift})
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: plot_md_correlation needs at least two non-empty series — got 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.md.study import metric_correlations
from pharmaml.md.xvg import XvgSeries, classify_metric
from pharmaml.viz._compat import (
    PALETTE,
    close_figure,
    new_figure,
    save_figure,
)

__all__ = ["plot_md_correlation", "plot_md_series"]

#: The uploaded script's web-named metric colours, resolved to the house
#: palette (:mod:`pharmaml.viz._compat`.``PALETTE``) so MD figures share
#: the visual language of every other pharmaml figure.  Unknown colours
#: fall back to the palette's teal lead.
_METRIC_COLORS: dict[str, str] = {
    "red": "#7F1D1D",  # red-900
    "green": "#4D7C0F",  # lime-700
    "blue": "#0F766E",  # teal
    "purple": "#6D28D9",  # violet-700
    "orange": "#B45309",  # amber-700
    "brown": "#525252",  # neutral-600
}


def _finish_render(figure: Any, path: str | Path | None) -> None:
    """Save to ``path`` (png/svg/pdf) when given and close the figure."""
    if path is not None:
        save_figure(figure, path)
    close_figure(figure)


def _metric_style(series: XvgSeries) -> tuple[str, str, str, str]:
    """Resolve ``(title, x_label, y_label, color)`` for one series.

    A known standard export (the series ``label`` first — the study
    reader keys folders by the filename stem — then ``source``) donates
    its display metadata through :func:`classify_metric`; anything else
    gets honest generics.  An explicitly-passed colour wins later.
    """
    info = classify_metric(series.label) or classify_metric(series.source)
    if info is None:
        return (
            series.label or "MD series",
            "Time",
            "Value",
            PALETTE[0],
        )
    return (
        info.title,
        info.x_label or "Time",
        info.y_label or "Value",
        _METRIC_COLORS.get(info.color, PALETTE[0]),
    )


def plot_md_series(
    series: XvgSeries,
    path: str | Path | None = None,
    *,
    smooth_window: int | None = None,
    color: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Line chart of one parsed MD metric — the uploaded panel, typed.

    The raw curve is drawn in the metric's house colour, a dashed mean
    reference line rides the data, and ``smooth_window`` overlays the
    uploaded pipeline's centered moving average
    (:func:`pharmaml.md.xvg.moving_average`) when it is two or more.

    Parameters
    ----------
    series:
        One parsed export from :mod:`pharmaml.md.xvg` (or any
        :class:`~pharmaml.md.xvg.XvgSeries`).
    path:
        Save the figure to ``png``/``svg``/``pdf`` (picked from the
        extension) — byte-reproducible, like every pharmaml figure.
    smooth_window:
        Width of the centered moving-average overlay; ``None`` (the
        default) draws the raw curve only.  Must be an integer ``>= 2``
        (a one-point average is a no-op, not a smoothing) and no longer
        than the series.
    color:
        Override the resolved metric colour.
    title:
        Override the resolved chart title.

    Returns
    -------
    dict[str, Any]
        The series' honest numbers — ``n``, ``t_min``/``t_max``,
        ``v_min``/``v_max``, the NaN-aware ``mean``/``std``/``cv`` from
        :func:`pharmaml.md.xvg.metric_stats`, the resolved
        ``title``/``x_label``/``y_label``/``color``, the
        ``smooth_window`` and its ``smoothed`` array (``None`` when not
        requested) and the saved ``path``.

    Raises
    ------
    DataValidationError
        If ``series`` is not an :class:`~pharmaml.md.xvg.XvgSeries`,
        ``smooth_window`` is not an integer ``>= 2``, or it exceeds the
        series length.
    """
    if not isinstance(series, XvgSeries):
        raise DataValidationError(
            f"plot_md_series expects an XvgSeries — got {type(series).__name__}"
        )
    if smooth_window is not None:
        if (
            not isinstance(smooth_window, int)
            or isinstance(smooth_window, bool)
            or smooth_window < 2
        ):
            raise DataValidationError(
                "smooth_window must be an integer >= 2 to average anything — "
                f"got {smooth_window}"
            )
        if smooth_window > len(series):
            raise DataValidationError(
                f"smooth_window ({smooth_window}) exceeds the series "
                f"length ({len(series)})"
            )

    default_title, x_label, y_label, default_color = _metric_style(series)
    resolved_color = color or default_color
    resolved_title = title or default_title

    smoothed: np.ndarray | None = None
    if smooth_window is not None:
        smoothed = series.smoothed(window=smooth_window)

    stats = series.stats
    figure, ax = new_figure(figsize=(8.0, 4.5))
    ax.plot(series.time, series.values, color=resolved_color, lw=1.8,
            label="raw")
    if smoothed is not None:
        ax.plot(series.time, smoothed, color=PALETTE[1], lw=1.6, ls="--",
                label=f"moving average (w={smooth_window})")
    if stats["mean"] is not None:
        ax.axhline(stats["mean"], color=PALETTE[2], lw=1.2, ls=":",
                   label=f"mean = {stats['mean']:.3g}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(resolved_title)
    ax.legend(fontsize=9, loc="best")
    _finish_render(figure, path)

    return {
        "label": series.label,
        "n": len(series),
        "t_min": float(series.time[0]),
        "t_max": float(series.time[-1]),
        "v_min": float(np.nanmin(series.values)),
        "v_max": float(np.nanmax(series.values)),
        "mean": stats["mean"],
        "std": stats["std"],
        "cv": stats["cv"],
        "title": resolved_title,
        "x_label": x_label,
        "y_label": y_label,
        "color": resolved_color,
        "smooth_window": smooth_window,
        "smoothed": smoothed,
        "path": str(Path(path).resolve()) if path is not None else None,
    }


def plot_md_correlation(
    series_map: dict[str, XvgSeries],
    path: str | Path | None = None,
    *,
    cmap: str = "RdBu_r",
) -> dict[str, Any]:
    """Cross-metric Pearson heatmap — the uploaded script's closing panel.

    The matrix is :func:`pharmaml.md.study.metric_correlations`' own —
    the series are truncated to their shortest common length (the
    uploaded alignment rule) before ``np.corrcoef`` runs, so the figure
    cannot drift from the pipeline's numbers.  Cells are annotated with
    *r*; a constant series honestly produces a NaN row and the count is
    disclosed in the returned stats instead of being papered over.

    Parameters
    ----------
    series_map:
        ``{metric_name: XvgSeries}`` — one ligand's parsed exports, the
        exact shape :class:`~pharmaml.md.study.LigandResult.series`
        carries.
    path:
        Save the figure to ``png``/``svg``/``pdf`` (picked from the
        extension).
    cmap:
        A diverging matplotlib colormap name; the default ``"RdBu_r"``
        maps ``+1`` red / ``-1`` blue around the neutral white diagonal
        midline.

    Returns
    -------
    dict[str, Any]
        ``labels`` (the dict-key order the matrix rows/columns follow),
        ``display_labels`` (the same names, titled for the figure),
        ``r`` (the k×k Pearson matrix), ``n_series``,
        ``n_points_aligned`` (the shortest common length the alignment
        rule kept), ``nan_cells`` (NaN entries — a constant series
        correlates with nothing) and the saved ``path``.

    Raises
    ------
    DataValidationError
        If ``series_map`` is not a non-empty mapping of
        :class:`~pharmaml.md.xvg.XvgSeries`, fewer than two non-empty
        series survive, or their common length is under two — the
        conditions under which the pipeline itself refuses to score.
    """
    if not isinstance(series_map, dict) or not series_map:
        raise DataValidationError(
            "plot_md_correlation needs a non-empty {name: XvgSeries} mapping"
        )
    for key, value in series_map.items():
        if not isinstance(value, XvgSeries):
            raise DataValidationError(
                f"series_map[{key!r}] must be an XvgSeries — got "
                f"{type(value).__name__}"
            )
    usable = {key: s for key, s in series_map.items() if len(s) > 0}
    if len(usable) < 2:
        raise DataValidationError(
            "plot_md_correlation needs at least two non-empty series — "
            f"got {len(usable)}"
        )
    min_len = min(len(s) for s in usable.values())
    if min_len < 2:
        raise DataValidationError(
            "plot_md_correlation needs a common length >= 2 after the "
            f"min-length alignment — got {min_len}"
        )

    computed = metric_correlations(usable)
    if computed is None:  # pragma: no cover - the pre-checks above cover it
        raise DataValidationError(
            "plot_md_correlation: the pipeline could not score this mapping"
        )
    labels, r = computed
    r = np.asarray(r, dtype=float)
    nan_cells = int(np.isnan(r).sum())

    k = len(labels)
    display_labels = [label.replace("_", " ") for label in labels]
    side = max(6.0, 0.85 * k + 2.0)
    figure, ax = new_figure(figsize=(side, side * 0.92))
    image = ax.imshow(r, vmin=-1.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)
    annot_size = 9 if k <= 8 else 7
    for i in range(k):
        for j in range(k):
            value = r[i, j]
            text = "nan" if np.isnan(value) else f"{value:.2f}"
            ax.text(
                j, i, text, ha="center", va="center", fontsize=annot_size,
                color=("white" if abs(value) > 0.5 else "#1F2937"),
            )
    figure.colorbar(image, ax=ax, shrink=0.85, label="Pearson r")
    ax.set_title(
        f"MD metric correlations — {k} series x {min_len} aligned points"
    )
    _finish_render(figure, path)

    return {
        "labels": labels,
        "display_labels": display_labels,
        "r": r,
        "n_series": k,
        "n_points_aligned": int(min_len),
        "nan_cells": nan_cells,
        "path": str(Path(path).resolve()) if path is not None else None,
    }
