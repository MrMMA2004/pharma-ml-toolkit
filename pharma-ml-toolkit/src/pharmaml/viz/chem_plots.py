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
  the Explorer's "Ligand Sets" playground tab);
* :func:`plot_selfies_robustness` — the SELFIES-vs-SMILES validity
  curves of a :func:`pharmaml.chem.selfies_robustness_experiment`
  run (the robustness property as one picture: random corruption of
  SELFIES symbols stays ~100% parseable while character corruption
  of SMILES collapses).

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
>>> plot_descriptor_distributions(pd.DataFrame([[1.0, 2.0]], columns=["v", "v"]))
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: duplicate column names: ['v'] — rename or drop the duplicates before plotting
>>> plot_descriptor_distributions(frame, bins=0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: bins must be a positive integer, got 0

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
    "plot_selfies_robustness",
    "plot_similarity_heatmap",
    "plot_table_comparison",
]

_LOG = get_logger(__name__)


def _check_bins_scalar(bins) -> None:
    """Typed guard for histogram ``bins`` scalars (0/-3/2.5 leak raw).

    Mirrors ``docking_plots._check_bins_scalar``: matplotlib accepts
    positive-int bins, bin-edge sequences and named strategies
    ("auto", "fd", ...); any OTHER scalar leaks a raw ValueError /
    TypeError from inside ``ax.hist`` — name it typed instead.
    """
    if isinstance(bins, (int, float, np.integer, np.floating)) and not (
        isinstance(bins, (int, np.integer)) and bins >= 1
    ):
        raise DataValidationError(f"bins must be a positive integer, got {bins!r}")


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
        prediction outputs.  Column names must be unique.
    columns:
        Subset of columns to show (default: every numeric column,
        capped at ``max_columns`` by column order).
    max_columns:
        Safety cap so a 200-descriptor frame renders as a readable grid.
    bins:
        Histogram bins — any :meth:`matplotlib.axes.Axes.hist` spec:
        a positive integer, a bin-edge sequence, or a named strategy
        such as ``"auto"``.  A non-positive integer scalar raises the
        typed error below instead of matplotlib's raw ``ValueError``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``frame`` is not a DataFrame, is empty, carries duplicated
        column names, the requested ``columns`` are missing or none of
        them is numeric, or ``bins`` is a non-positive integer.
    """
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("frame must be a pandas DataFrame")
    duplicated = frame.columns[frame.columns.duplicated()]
    if len(duplicated):
        # A duplicated column name turns frame[name] into a multi-column
        # DataFrame whose is_numeric_dtype check is False, so the grid
        # reported the MISLEADING "no numeric columns to plot" even when
        # every duplicate was numeric (100x-41-B corner).
        raise DataValidationError(
            f"duplicate column names: {list(duplicated)} — rename or drop "
            "the duplicates before plotting"
        )
    if frame.empty:
        raise DataValidationError("frame is empty — nothing to plot")
    _check_bins_scalar(bins)

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

    Raises
    ------
    DataValidationError
        If the matrix is not square, the label count does not match
        the matrix, or the values cannot be converted to floats (a
        string frame used to leak numpy's raw ``could not convert
        string to float``).
    """
    if isinstance(similarity, pd.DataFrame):
        try:
            matrix = similarity.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"similarity matrix values must be numeric ({exc})"
            ) from exc
        if labels is None:
            labels = [str(name) for name in similarity.index]
    else:
        try:
            matrix = np.asarray(similarity, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"similarity matrix values must be numeric ({exc})"
            ) from exc

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
        bucket names; defaults to ``("A", "B")``.  A pair of exactly
        two strings — any other arity raises the typed error below
        instead of an unpacking ``ValueError``.
    title:
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``comparison`` lacks the comparison attributes, both
        tables turned out empty (nothing to draw), or ``labels`` is not
        a two-item ``(table_a_name, table_b_name)`` pair.

    Notes
    -----
    The ``title`` is drawn left-anchored above the top panel
    (``set_title(..., loc="left")``); retrieve it with
    ``fig.axes[0].get_title(loc="left")`` — a plain ``get_title()``
    reads the centre slot and returns ``''``.
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
    if labels is not None and len(labels) != 2:
        # A 1- or 3-tuple used to die on the tuple unpacking below with
        # a raw ``ValueError: (not enough|too many) values to unpack``
        # (100x-41-B corner).
        raise DataValidationError(
            f"labels must be a (table_a_name, table_b_name) pair of two "
            f"strings, got {len(labels)} item(s): {list(labels)}"
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


def plot_selfies_robustness(
    report: Any,
    title: str = "SELFIES vs SMILES — validity under corruption",
) -> Any:
    """The robustness experiment's two validity curves, one figure.

    :func:`pharmaml.chem.selfies_robustness_experiment` measures how
    often a corrupted string still parses as a molecule; this figure
    draws the answer -- the SELFIES curve (random *symbol* corruption,
    flat at ~100%: the grammar clamps every mutation to a legal
    structure) against the SMILES curve (random *character* corruption,
    collapsing as k grows).  Levels sit on a categorical x-axis (the
    probed mutation counts), validity on a 0-100% y-axis.

    Parameters
    ----------
    report:
        A :class:`~pharmaml.chem.selfies_codec.SelfiesRobustnessReport`
        (duck-typed: anything exposing ``mutations``,
        ``selfies_validity`` and ``smiles_validity`` works).
    title:
        Figure title (left-anchored; retrieve with
        ``fig.axes[0].get_title(loc="left")``).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If the report carries no mutation levels, or its validity lists
        do not align with them.

    Example
    -------
    >>> from pharmaml.chem.selfies_codec import selfies_robustness_experiment
    >>> from pharmaml.viz import close_figure
    >>> from pharmaml.viz.chem_plots import plot_selfies_robustness
    >>> report = selfies_robustness_experiment(
    ...     ["c1ccccc1", "CCO"], mutations=(1, 4), trials=4, seed=42)
    >>> fig = plot_selfies_robustness(report)
    >>> (len(fig.axes), fig.axes[0].get_title(loc="left"))
    (1, 'SELFIES vs SMILES — validity under corruption — 2 molecules × 4 trials per level')
    >>> [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    ['SELFIES (symbol corruption)', 'SMILES (character corruption)']
    >>> fig.axes[0].get_xlabel()
    'random corruptions per string (k)'
    >>> close_figure(fig)
    >>> from types import SimpleNamespace
    >>> plot_selfies_robustness(SimpleNamespace(
    ...     mutations=[], selfies_validity=[], smiles_validity=[]))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: report carries no mutation levels — pass the SelfiesRobustnessReport of a selfies_robustness_experiment() run
    """
    mutations = list(getattr(report, "mutations", []) or [])
    selfies_validity = list(getattr(report, "selfies_validity", []) or [])
    smiles_validity = list(getattr(report, "smiles_validity", []) or [])
    if not mutations:
        raise DataValidationError(
            "report carries no mutation levels — pass the "
            "SelfiesRobustnessReport of a selfies_robustness_experiment() run"
        )
    if len(selfies_validity) != len(mutations) or len(smiles_validity) != len(
        mutations
    ):
        raise DataValidationError(
            f"validity lists ({len(selfies_validity)} / {len(smiles_validity)}) "
            f"do not align with the {len(mutations)} mutation levels"
        )
    n_molecules = getattr(report, "n_molecules", None)
    n_trials = getattr(report, "n_trials", None)

    fig, raw_axes = new_figure(figsize=(7.2, 4.8))
    axes = np.atleast_1d(raw_axes)
    ax = axes[0]
    positions = np.arange(len(mutations))
    ax.plot(
        positions,
        [100.0 * value for value in selfies_validity],
        color=PALETTE[0],
        marker="o",
        linewidth=1.8,
        label="SELFIES (symbol corruption)",
    )
    ax.plot(
        positions,
        [100.0 * value for value in smiles_validity],
        color=PALETTE[3],
        marker="s",
        linestyle="--",
        linewidth=1.6,
        label="SMILES (character corruption)",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([str(level) for level in mutations])
    ax.set_xlabel("random corruptions per string (k)", fontsize=9)
    ax.set_ylabel("strings still parsing as molecules (%)", fontsize=9)
    ax.set_ylim(-4.0, 104.0)
    subtitle = title
    if n_molecules is not None and n_trials is not None:
        subtitle += f" — {n_molecules} molecules × {n_trials} trials per level"
    ax.set_title(subtitle, fontsize=11, loc="left", pad=10)
    ax.legend(fontsize=8, frameon=False)
    return fig
