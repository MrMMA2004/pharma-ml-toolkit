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

* :func:`plot_md_series` — one parsed metric series (or a
  ``{name: XvgSeries}`` comparison across ligands) as a line chart with
  a mean reference line and the uploaded smoothing available as an
  explicit overlay; title and axis labels come from
  :func:`pharmaml.md.xvg.classify_metric` when the series label is one
  of the ten standard GROMACS/MM-PBSA exports, with explicit
  ``title``/``x_label``/``y_label`` overrides winning;
* :func:`plot_md_correlation` — the cross-metric Pearson heatmap over a
  ligand's series mapping (a plain mapping or a
  :class:`~pharmaml.md.study.LigandResult`), computed by the pipeline's
  own :func:`pharmaml.md.study.metric_correlations` (min-length
  alignment — the uploaded script's rule) and annotated cell by cell,
  with a constant series' NaN row disclosed as ``nan_cells`` instead of
  being papered over.

The honesty rules, stated once: the statistics are numpy-first and only
the drawing touches matplotlib (lazy, headless ``Agg``); NaN *values*
are tolerated and disclosed (a NaN pair's correlation is reported as a
NaN cell, never silently dropped) while ±inf is refused outright — an
infinite value poisons every statistic it touches; and the correlation
heatmap refuses series whose abscissas disagree over the aligned window
("axes must match") — truncating two misaligned time grids to their
common length would correlate numbers that were never observed at the
same moments, which is the one lie the pipeline's crude min-length rule
would otherwise let a *figure* tell.  GROMACS itself, MDAnalysis and
mdtraj stay external — this module honours the plain-text
``.xvg``/``.ndx`` contract those tools export, it does not re-render
their GUIs.

Both functions save byte-reproducible png/svg/pdf to ``path`` when
given, close their figures afterwards, and return the stats dict —
pass ``close=False`` (the Studio page's contract) to keep the figure
open: it then rides the returned dict as ``"figure"`` and the caller
owns closing it.

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
>>> open_fig = plot_md_series(drift, title="Open", close=False)
>>> open_fig["figure"] is not None, open_fig["closed"]
(True, False)
>>> from pharmaml.viz._compat import close_figure
>>> close_figure(open_fig["figure"])
>>> other = XvgSeries(time=np.arange(3.0) + 1.0, values=[3.0, 2.0, 1.0], label="b")
>>> shifted = XvgSeries(time=np.arange(2.0), values=[1.0, 2.0], label="a")
>>> plot_md_correlation({"a": shifted, "b": other})
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: plot_md_correlation: axes must match — series 'a' and 'b' disagree on the abscissa over the aligned window
>>> plot_md_series(drift, smooth_window=1)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: smooth_window must be an integer >= 2 to average anything — got 1
>>> plot_md_series(XvgSeries(time=np.arange(2.0), values=[1.0, float("inf")]))
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: plot_md_series: series '' contains ±inf time or values — an infinite value poisons every statistic
>>> plot_md_correlation({"a": XvgSeries(time=np.arange(2.0), values=[1.0, float("inf")]),
...                      "b": XvgSeries(time=np.arange(2.0), values=[2.0, 3.0])})
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: plot_md_correlation: series 'a' contains ±inf time or values — an infinite value poisons every statistic
>>> plot_md_correlation({"ligand_rmsd": drift})
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: plot_md_correlation needs at least two non-empty series — got 1
"""

from __future__ import annotations

from collections.abc import Mapping
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


def _finish_render(figure: Any, path: str | Path | None, close: bool) -> None:
    """Save to ``path`` (png/svg/pdf) when given; close unless told not to."""
    if path is not None:
        save_figure(figure, path)
    if close:
        close_figure(figure)


def _check_finite(series: XvgSeries, label: str, func: str) -> None:
    """Refuse ±inf (it poisons every statistic); NaN stays disclosed."""
    if bool(np.isinf(series.time).any()) or bool(np.isinf(series.values).any()):
        raise DataValidationError(
            f"{func}: series {label!r} contains ±inf time or values "
            "— an infinite value poisons every statistic"
        )


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


def _series_stats(series: XvgSeries) -> dict[str, Any]:
    """The honest per-series numbers every return shape carries."""
    stats = series.stats
    return {
        "n": len(series),
        "t_min": float(series.time[0]),
        "t_max": float(series.time[-1]),
        "v_min": float(np.nanmin(series.values)),
        "v_max": float(np.nanmax(series.values)),
        "mean": stats["mean"],
        "std": stats["std"],
        "cv": stats["cv"],
    }


def plot_md_series(
    series: XvgSeries | Mapping[str, XvgSeries],
    path: str | Path | None = None,
    *,
    smooth_window: int | None = None,
    color: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    close: bool = True,
) -> dict[str, Any]:
    """Line chart of parsed MD metrics — the uploaded panel, typed.

    One :class:`~pharmaml.md.xvg.XvgSeries` draws as the raw curve in
    the metric's house colour with a dashed mean reference line;
    ``smooth_window`` overlays the uploaded pipeline's centered moving
    average (:func:`pharmaml.md.xvg.moving_average`) when it is two or
    more.  A ``{name: XvgSeries}`` mapping (e.g.
    :meth:`pharmaml.md.study.MDStudy.metric_series`) draws one line per
    entry, palette-cycled and legend-labelled — the cross-ligand
    comparison panel; smoothing and the mean line are single-series
    features there.

    Parameters
    ----------
    series:
        One parsed export, or a named mapping of them for the
        comparison overlay.
    path:
        Save the figure to ``png``/``svg``/``pdf`` (picked from the
        extension) — byte-reproducible, like every pharmaml figure.
    smooth_window:
        Width of the centered moving-average overlay; ``None`` (the
        default) draws the raw curve only.  Must be an integer ``>= 2``
        (a one-point average is a no-op, not a smoothing) and no longer
        than the series.  Only valid for a single series.
    color:
        Override the resolved metric colour (single series only).
    title, x_label, y_label:
        Override the resolved chart title / axis labels.
    close:
        Close the figure after saving (the default).  Pass ``False``
        to keep it open — the returned dict then carries it as
        ``"figure"`` and the caller owns closing it.

    Returns
    -------
    dict[str, Any]
        For a single series: the honest numbers — ``n``,
        ``t_min``/``t_max``, ``v_min``/``v_max``, the NaN-aware
        ``mean``/``std``/``cv`` from :func:`pharmaml.md.xvg.metric_stats`,
        the resolved ``title``/``x_label``/``y_label``/``color``, the
        ``smooth_window`` and its ``smoothed`` array (``None`` when not
        requested), ``closed``, the ``figure`` (open only when
        ``close=False``) and the saved ``path``.  For a mapping:
        ``n_series``/``labels`` plus the same per-series numbers under
        ``series``.

    Raises
    ------
    DataValidationError
        If ``series`` is neither an :class:`~pharmaml.md.xvg.XvgSeries`
        nor a non-empty mapping of them, any member's time/values carry
        ±inf, ``smooth_window`` is not an integer ``>= 2`` or exceeds
        the series, or ``smooth_window``/``color`` ride a mapping.
    """
    if isinstance(series, XvgSeries):
        items = [(series.label or series.source, series)]
        single = True
    elif isinstance(series, Mapping) and series:
        items = [
            (str(name), value)
            for name, value in series.items()
        ]
        single = False
    else:
        raise DataValidationError(
            "plot_md_series expects an XvgSeries or a non-empty "
            "{name: XvgSeries} mapping — got " + type(series).__name__
        )
    for name, item in items:
        if not isinstance(item, XvgSeries):
            raise DataValidationError(
                f"plot_md_series: entry {name!r} must be an XvgSeries — got "
                f"{type(item).__name__}"
            )
        _check_finite(item, name, "plot_md_series")
    if not single and smooth_window is not None:
        raise DataValidationError(
            "smooth_window overlays one series; pass a single XvgSeries "
            "to smooth (a mapping draws raw comparison lines)"
        )
    if not single and color is not None:
        raise DataValidationError(
            "color styles one series; a mapping is palette-cycled per entry"
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
        if smooth_window > len(items[0][1]):
            raise DataValidationError(
                f"smooth_window ({smooth_window}) exceeds the series "
                f"length ({len(items[0][1])})"
            )

    if single:
        default_title, default_x, default_y, default_color = _metric_style(
            items[0][1]
        )
    else:
        default_title, default_x, default_y = (
            "MD metric series",
            "Time / index",
            "Value",
        )
        default_color = PALETTE[0]
    resolved_color = color or default_color
    resolved_title = title or default_title
    resolved_x = x_label or default_x
    resolved_y = y_label or default_y

    smoothed: np.ndarray | None = None
    if smooth_window is not None:
        smoothed = items[0][1].smoothed(window=smooth_window)

    figure, ax = new_figure(figsize=(8.0, 4.5))
    for index, (name, item) in enumerate(items):
        line_color = (
            resolved_color if single else PALETTE[index % len(PALETTE)]
        )
        ax.plot(item.time, item.values, color=line_color, lw=1.8,
                label=name or None)
    if smoothed is not None:
        ax.plot(items[0][1].time, smoothed, color=PALETTE[1], lw=1.6,
                ls="--", label=f"moving average (w={smooth_window})")
    if single and items[0][1].stats["mean"] is not None:
        ax.axhline(items[0][1].stats["mean"], color=PALETTE[2], lw=1.2,
                   ls=":", label=f"mean = {items[0][1].stats['mean']:.3g}")
    ax.set_xlabel(resolved_x)
    ax.set_ylabel(resolved_y)
    ax.set_title(resolved_title)
    ax.legend(fontsize=9, loc="best")
    _finish_render(figure, path, close)

    common: dict[str, Any] = {
        "title": resolved_title,
        "x_label": resolved_x,
        "y_label": resolved_y,
        "color": resolved_color if single else None,
        "smooth_window": smooth_window,
        "smoothed": smoothed,
        "closed": close,
        "figure": None if close else figure,
        "path": str(Path(path).resolve()) if path is not None else None,
    }
    if single:
        name, item = items[0]
        return {
            "label": item.label,
            **_series_stats(item),
            **common,
        }
    return {
        "n_series": len(items),
        "labels": [name for name, _ in items],
        "series": {name: _series_stats(item) for name, item in items},
        **common,
    }


def plot_md_correlation(
    series_map: Mapping[str, XvgSeries] | Any,
    path: str | Path | None = None,
    *,
    cmap: str = "RdBu_r",
    title: str | None = None,
    close: bool = True,
) -> dict[str, Any]:
    """Cross-metric Pearson heatmap — the uploaded script's closing panel.

    The matrix is :func:`pharmaml.md.study.metric_correlations`' own —
    the series are truncated to their shortest common length (the
    uploaded alignment rule) before ``np.corrcoef`` runs, so the figure
    cannot drift from the pipeline's numbers.  Cells are annotated with
    *r*; a constant series honestly produces a NaN row (masked in the
    image, annotated ``N/A``) and the count is disclosed in the
    returned stats instead of being papered over.  The one refusal the
    raw numbers don't make: series whose abscissas disagree over the
    aligned window are refused — a figure may not correlate
    observations that were never taken at the same moments.

    Parameters
    ----------
    series_map:
        ``{metric_name: XvgSeries}`` — one ligand's parsed exports, the
        exact shape :class:`~pharmaml.md.study.LigandResult.series`
        carries — or a :class:`~pharmaml.md.study.LigandResult` itself.
    path:
        Save the figure to ``png``/``svg``/``pdf`` (picked from the
        extension).
    cmap:
        A diverging matplotlib colormap name; the default ``"RdBu_r"``
        maps ``+1`` red / ``-1`` blue around the neutral white diagonal
        midline.
    title:
        Override the generated chart title.
    close:
        Close the figure after saving (the default).  Pass ``False``
        to keep it open — the returned dict then carries it as
        ``"figure"`` and the caller owns closing it.

    Returns
    -------
    dict[str, Any]
        ``labels`` (the dict-key order the matrix rows/columns follow),
        ``display_labels`` (the same names, titled for the figure),
        ``r`` (the k×k Pearson matrix), ``n_series``,
        ``n_points_aligned`` (the shortest common length the alignment
        rule kept), ``nan_cells`` (NaN entries — a constant series
        correlates with nothing), ``closed``, the ``figure`` (open only
        when ``close=False``) and the saved ``path``.

    Raises
    ------
    DataValidationError
        If ``series_map`` is neither a non-empty mapping of
        :class:`~pharmaml.md.xvg.XvgSeries` nor a LigandResult, fewer
        than two non-empty series survive, their common length is under
        two, or their abscissas disagree over the aligned window.
    """
    if hasattr(series_map, "series") and not isinstance(series_map, Mapping):
        series_map = series_map.series
    if not isinstance(series_map, Mapping) or not series_map:
        raise DataValidationError(
            "plot_md_correlation needs a non-empty {name: XvgSeries} mapping "
            "or a LigandResult"
        )
    for key, value in series_map.items():
        if not isinstance(value, XvgSeries):
            raise DataValidationError(
                f"series_map[{key!r}] must be an XvgSeries — got "
                f"{type(value).__name__}"
            )
        _check_finite(value, key, "plot_md_correlation")
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
    names = list(usable)
    reference = usable[names[0]].time[:min_len]
    for name in names[1:]:
        other = usable[name].time[:min_len]
        if not bool(np.allclose(reference, other, equal_nan=False)):
            raise DataValidationError(
                "plot_md_correlation: axes must match — series "
                f"{names[0]!r} and {name!r} disagree on the abscissa over "
                "the aligned window"
            )

    computed = metric_correlations(usable)
    if computed is None:  # pragma: no cover - the pre-checks above cover it
        raise DataValidationError(
            "plot_md_correlation: the pipeline could not score this mapping"
        )
    labels, r = computed
    r = np.asarray(r, dtype=float)
    nan_cells = int(np.isnan(r).sum())
    masked = np.ma.masked_invalid(r)

    k = len(labels)
    display_labels = [label.replace("_", " ") for label in labels]
    side = max(6.0, 0.85 * k + 2.0)
    figure, ax = new_figure(figsize=(side, side * 0.92))
    image = ax.imshow(masked, vmin=-1.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)
    annot_size = 9 if k <= 8 else 7
    for i in range(k):
        for j in range(k):
            value = r[i, j]
            if np.isnan(value):
                text, color = "N/A", "#6B7280"
            else:
                text = f"{value:.2f}"
                color = "white" if abs(value) > 0.5 else "#1F2937"
            ax.text(j, i, text, ha="center", va="center", fontsize=annot_size,
                    color=color)
    figure.colorbar(image, ax=ax, shrink=0.85, label="Pearson r")
    ax.set_title(
        title or f"MD metric correlations — {k} series x {min_len} aligned points"
    )
    _finish_render(figure, path, close)

    return {
        "labels": labels,
        "display_labels": display_labels,
        "r": r,
        "n_series": k,
        "n_points_aligned": int(min_len),
        "nan_cells": nan_cells,
        "closed": close,
        "figure": None if close else figure,
        "path": str(Path(path).resolve()) if path is not None else None,
    }
