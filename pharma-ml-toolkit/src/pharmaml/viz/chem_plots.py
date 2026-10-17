"""Visual cheminformatics output — descriptor landscapes and similarity.

The archive's descriptor calculators (``others/Ml/Descriptor
calculator.py``, ``Morgan_calculator.py``, ...) wrote wide CSVs and
stopped there; this module turns them into figures:

* :func:`plot_descriptor_distributions` — a histogram grid over the
  numeric descriptor columns of any frame (also fits pIC50 tables);
* :func:`plot_similarity_heatmap` — a Tanimoto matrix heat-map with
  optional cell annotations;
* :func:`plot_table_comparison` — the overlap breakdown of a
  :class:`~pharmaml.chem.tables.TableComparison` (the figure twin of
  the Explorer's "Ligand Sets" playground tab).

Examples
--------
>>> import pandas as pd
>>> from pharmaml.viz import close_figure
>>> from pharmaml.viz.chem_plots import plot_descriptor_distributions
>>> frame = pd.DataFrame({"mol_wt": [180.16, 300.2, 250.3, 400.4],
...                       "logp": [1.3, 2.0, 3.1, 4.2],
...                       "name": ["aspirin", "ibuprofen", "caffeine", "naproxen"]})
>>> fig = plot_descriptor_distributions(frame)      # the 'name' column is skipped
>>> [ax.get_title() for ax in fig.axes]
['mol_wt', 'logp']
>>> close_figure(fig)
>>> plot_descriptor_distributions([1, 2])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: frame must be a pandas DataFrame

>>> from pharmaml.viz.chem_plots import plot_similarity_heatmap
>>> sim = pd.DataFrame([[1.0, 0.5, 0.0], [0.5, 1.0, 0.25], [0.0, 0.25, 1.0]],
...                    index=["ethanol", "propanol", "phenol"])
>>> fig = plot_similarity_heatmap(sim)              # labels come from the frame
>>> (len(fig.axes), fig.axes[0].get_title())
(2, 'Tanimoto similarity')
>>> [t.get_text() for t in fig.axes[0].get_xticklabels()]
['ethanol', 'propanol', 'phenol']
>>> plot_similarity_heatmap([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: similarity matrix must be square, got shape (3, 2)
>>> plot_similarity_heatmap([[1.0, 0.0], [0.0, 1.0]], labels=["a", "b", "c"])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: 3 labels for a 2x2 matrix
>>> close_figure(fig)

>>> from pharmaml.chem.tables import compare_tables
>>> from pharmaml.viz.chem_plots import plot_table_comparison
>>> comp = compare_tables(pd.DataFrame({"ligand": ["aspirin", "ibuprofen", "caffeine"]}),
...                       pd.DataFrame({"ligand": ["Aspirin.pdbqt", "naproxen", "ibuprofen"]}),
...                       "ligand")                 # names are normalised first
>>> (comp.common, comp.only_in_a, comp.only_in_b)
(['aspirin', 'ibuprofen'], ['caffeine'], ['naproxen'])
>>> fig = plot_table_comparison(comp, labels=("set_A", "set_B"))
>>> (len(fig.axes), fig.axes[0].get_title(loc="left"))
(2, 'Ligand-set reconciliation')
>>> [t.get_text() for t in fig.axes[1].get_yticklabels()]
['common (both)', 'only in set_A', 'only in set_B']
>>> close_figure(fig)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = [
    "plot_descriptor_distributions",
    "plot_similarity_heatmap",
    "plot_table_comparison",
]

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


def plot_table_comparison(
    comparison: Any,
    labels: tuple[str, str] | None = None,
    title: str = "Ligand-set reconciliation",
) -> Any:
    """Overlap breakdown of a ligand-table :class:`TableComparison`.

    Renders the set reconciliation produced by
    :func:`pharmaml.chem.compare_tables` as a two-panel figure: a
    proportional 100% overlap bar (common / only in A / only in B) on
    top and per-bucket count bars below — the same visual language as
    the Explorer's "Ligand Sets" playground tab, so a Python campaign
    and the web UI tell one story.

    Parameters
    ----------
    comparison:
        A :class:`~pharmaml.chem.tables.TableComparison` (anything
        exposing ``common`` / ``only_in_a`` / ``only_in_b`` lists and a
        ``summary()`` mapping works).
    labels:
        Optional ``(table_a_name, table_b_name)`` pair used in the
        bucket names; defaults to ``("A", "B")``.
    title:
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``comparison`` lacks the comparison attributes, or both
        tables turned out empty (nothing to draw).
    """
    for attr in ("common", "only_in_a", "only_in_b"):
        if not hasattr(comparison, attr):
            raise DataValidationError(
                "comparison must be a pharmaml.chem.tables.TableComparison "
                f"(missing attribute {attr!r}); get one from compare_tables()"
            )
    common = list(comparison.common)
    only_a = list(comparison.only_in_a)
    only_b = list(comparison.only_in_b)
    counts = [len(common), len(only_a), len(only_b)]
    total = sum(counts)
    if total == 0:
        raise DataValidationError(
            "both tables are empty after normalisation — nothing to compare"
        )

    name_a, name_b = labels if labels is not None else ("A", "B")
    buckets = ["common (both)", f"only in {name_a}", f"only in {name_b}"]
    colors = [PALETTE[0], PALETTE[1], PALETTE[3]]  # teal / amber / red
    shares = [100.0 * count / total for count in counts]

    fig, (ax_share, ax_count) = new_figure(
        nrows=2, ncols=1, figsize=(9.0, 3.1),
        height_ratios=[1.0, 1.5],
        gridspec_kw={"hspace": 0.55},
    )

    # -- Top panel: proportional overlap bar (100% stacked) -------------
    left = 0.0
    for count, share, color, bucket in zip(counts, shares, colors, buckets):
        ax_share.barh(0, share, left=left, height=0.55, color=color,
                      edgecolor="white", linewidth=1.2, label=f"{bucket}: {count}")
        if share >= 7.0:  # label only segments wide enough to host text
            text_color = "white" if color != PALETTE[1] else "#1F2937"
            ax_share.text(left + share / 2.0, 0, f"{share:.0f}%",
                          ha="center", va="center", fontsize=9,
                          fontweight="bold", color=text_color)
        left += share
    ax_share.set_xlim(0.0, 100.0)
    ax_share.set_yticks([])
    ax_share.set_xlabel("share of unique ligand names (%)", fontsize=9)
    ax_share.set_title(title, fontsize=11, loc="left", pad=10)
    ax_share.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55),
                    ncol=3, fontsize=8, frameon=False)
    ax_share.tick_params(labelsize=8)

    # -- Bottom panel: per-bucket count bars -----------------------------
    positions = [2, 1, 0]  # common on top
    ax_count.barh(positions, counts, height=0.6, color=colors,
                  edgecolor="white", linewidth=0.8)
    ax_count.set_yticks(positions, buckets, fontsize=9)
    ax_count.set_xlim(0, max(counts) * 1.18)
    for y, count, share in zip(positions, counts, shares):
        ax_count.text(count + max(counts) * 0.025, y,
                      f"{count}  ({share:.1f}%)", va="center",
                      fontsize=9, color="#1F2937")
    ax_count.set_xlabel("number of unique ligand names", fontsize=9)
    ax_count.tick_params(labelsize=8)
    return fig
