"""Visual molecular-docking output — energy landscapes and rankings.

The archive's docking readers exported Excel tables and stopped; this
module visualises them:

* :func:`plot_energy_distribution` — histogram of best binding
  energies across a ligand library with the median marked;
* :func:`plot_top_ligands` — ranked horizontal bars of the strongest
  binders, ready to drop into a report or presentation.

Both accept the summary frame produced by
:func:`~pharmaml.docking.parsing.scan_docking_directory` directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = ["plot_energy_distribution", "plot_top_ligands"]

#: Column scanned for the per-ligand best affinity.
_ENERGY_COLUMN = "best_lowest_energy"
#: Fallback column when the populated-cluster pick is unavailable.
_TOP_ENERGY_COLUMN = "top_lowest_energy"


def _energy_series(summary: pd.DataFrame, energy_column: str | None) -> tuple[pd.Series, str]:
    if not isinstance(summary, pd.DataFrame):
        raise DataValidationError("summary must be a pandas DataFrame (scan_docking_directory output)")
    if "folder" not in summary.columns:
        raise DataValidationError("summary frame needs a 'folder' column")

    column = energy_column or _ENERGY_COLUMN
    if column not in summary.columns and _TOP_ENERGY_COLUMN in summary.columns:
        column = _TOP_ENERGY_COLUMN
    if column not in summary.columns:
        raise DataValidationError(
            f"no energy column found (looked for '{_ENERGY_COLUMN}' / "
            f"'{_TOP_ENERGY_COLUMN}'); available: {list(summary.columns)}"
        )
    series = pd.to_numeric(summary[column], errors="coerce").dropna()
    folders = summary.loc[series.index, "folder"]
    if series.empty:
        raise DataValidationError(
            f"column {column!r} has no numeric values (check the scan status column)"
        )
    return folders, series


def plot_energy_distribution(
    summary: pd.DataFrame,
    energy_column: str | None = None,
    bins: int = 20,
    title: str = "Docking energy distribution",
) -> Any:
    """Histogram of per-ligand best binding energies (kcal/mol).

    Parameters
    ----------
    summary:
        Frame from :func:`~pharmaml.docking.parsing.scan_docking_directory`.
    energy_column:
        Override the affinity column (default: ``best_lowest_energy``,
        falling back to ``top_lowest_energy``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    folders, energies = _energy_series(summary, energy_column)
    values = energies.to_numpy(dtype=float)

    fig, ax = new_figure(figsize=(8.0, 5.0))
    ax.hist(values, bins=bins, color=PALETTE[0], alpha=0.85, edgecolor="white",
            linewidth=0.5)
    median = float(np.median(values))
    ax.axvline(median, color=PALETTE[1], lw=2.0,
               label=f"median = {median:.3g} kcal/mol")
    best_idx = int(np.argmin(values))
    ax.axvline(values[best_idx], color=PALETTE[3], lw=1.6, ls="--",
               label=f"best = {values[best_idx]:.3g} ({folders.iloc[best_idx]})")
    ax.set_xlabel("best binding energy (kcal/mol) — lower is stronger")
    ax.set_ylabel("ligands")
    ax.set_title(f"{title} — {len(values)} ligand(s)")
    ax.legend(fontsize=9, loc="best")
    return fig


def plot_top_ligands(
    summary: pd.DataFrame,
    top_n: int = 15,
    energy_column: str | None = None,
    title: str = "Top-ranked ligands by docking energy",
) -> Any:
    """Horizontal bars of the strongest-binding ligands.

    Bars are sorted so the strongest binder sits on top; values are
    annotated and the colour encodes energy (darker = stronger), which
    reads well both on screen and in exported png/svg/pdf files.
    """
    folders, energies = _energy_series(summary, energy_column)
    frame = pd.DataFrame({"ligand": [str(name) for name in folders],
                          "energy": energies.to_numpy(dtype=float)})
    frame = frame.sort_values("energy", ascending=True).head(
        top_n if top_n and top_n > 0 else len(frame)
    )
    frame = frame.iloc[::-1]  # strongest binder at the top of the axis

    fig, ax = new_figure(figsize=(8.6, 0.44 * len(frame) + 1.7))
    positions = np.arange(len(frame))
    ax.barh(positions, frame["energy"].to_numpy(), color=PALETTE[0], alpha=0.88,
            height=0.62, edgecolor="white")
    for pos, value in zip(positions, frame["energy"].to_numpy()):
        ax.text(value, pos, f" {value:.3g}", va="center", ha="left", fontsize=9)
    ax.set_yticks(positions, frame["ligand"].tolist())
    ax.set_xlabel("best binding energy (kcal/mol) — lower is stronger")
    ax.set_title(title)
    return fig
