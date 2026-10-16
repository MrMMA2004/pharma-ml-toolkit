"""Visual molecular-docking output — energy landscapes and rankings.

The archive's docking readers exported Excel tables and stopped; this
module visualises them:

* :func:`plot_energy_distribution` — histogram of best binding
  energies across a ligand library with the median marked;
* :func:`plot_top_ligands` — ranked horizontal bars of the strongest
  binders, ready to drop into a report or presentation;
* :func:`plot_receptor_atom_types` — the AutoDock type census of a
  prepared receptor (donors, acceptors, carbons, metals), the figure
  behind the ``types A×87 C×526 ...`` line ``gpfer.py`` hard-coded.

The first two accept the summary frame produced by
:func:`~pharmaml.docking.parsing.scan_docking_directory` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = ["plot_energy_distribution", "plot_receptor_atom_types", "plot_top_ligands"]

#: Column scanned for the per-ligand best affinity.
_ENERGY_COLUMN = "best_lowest_energy"
#: Fallback column when the populated-cluster pick is unavailable.
_TOP_ENERGY_COLUMN = "top_lowest_energy"


def _classify_atom_type(atom_type: str) -> str:
    """Group an AutoDock atom type into a legend-friendly chemistry class."""
    if atom_type.startswith("H"):
        return "polar hydrogen (donor)"
    if atom_type.startswith(("N", "O", "S")):
        return "heteroatom (acceptor/donor)"
    if atom_type in ("A", "C"):
        return "non-polar carbon"
    return "metal / ion"


def _receptor_type_counts(receptor) -> dict[str, int]:
    """Normalise the supported receptor inputs into a {type: count} dict."""
    from collections import Counter

    if hasattr(receptor, "atom_types") and hasattr(receptor, "output_pdbqt"):
        # ProteinReport from pharmaml.docking.prepare_protein
        counts = dict(receptor.atom_types or {})
    elif isinstance(receptor, dict):
        counts = {
            str(k): int(v) for k, v in receptor.items() if int(v) > 0
        }
    else:
        path = Path(str(receptor))
        if not path.is_file():
            raise DataValidationError(f"receptor PDBQT file not found: {path}")
        census: Counter[str] = Counter()
        for line in path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                if parts and parts[-1].isalpha():
                    census[parts[-1]] += 1
        counts = dict(census)
    if not counts:
        raise DataValidationError(
            "no AutoDock atom types found in the receptor "
            "(empty atom_types dict or PDBQT without ATOM/HETATM records)"
        )
    return counts


def plot_receptor_atom_types(
    receptor,
    title: str = "Receptor AutoDock atom types",
) -> Any:
    """Bar chart of the AutoDock atom-type census of a receptor.

    Parameters
    ----------
    receptor:
        Any of: a :class:`~pharmaml.docking.ProteinReport` (its
        ``atom_types`` dict is used), a plain ``{type: count}`` mapping,
        or the path to a PDBQT file whose ATOM/HETATM lines are counted
        on the fly.
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Bars are sorted by count and coloured by chemistry — polar
    hydrogens, heteroatoms, non-polar carbons, metals/ions — so a
    missing donor population (a classic mis-prepared receptor) is
    visible at a glance.
    """
    counts = _receptor_type_counts(receptor)
    frame = pd.DataFrame(
        {"type": list(counts), "count": list(counts.values())}
    ).sort_values("count", ascending=False)
    frame["group"] = [_classify_atom_type(t) for t in frame["type"]]

    group_colours = {
        "polar hydrogen (donor)": PALETTE[2],
        "heteroatom (acceptor/donor)": PALETTE[1],
        "non-polar carbon": PALETTE[0],
        "metal / ion": PALETTE[4],
    }

    fig, ax = new_figure(figsize=(0.72 * len(frame) + 2.4, 5.0))
    positions = np.arange(len(frame))
    bars = ax.bar(
        positions,
        frame["count"].to_numpy(),
        color=[group_colours[g] for g in frame["group"]],
        alpha=0.9,
        width=0.66,
        edgecolor="white",
    )
    for bar, value in zip(bars, frame["count"].to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(positions, frame["type"].tolist())
    ax.set_ylabel("atoms")
    total = int(frame["count"].sum())
    ax.set_title(f"{title} — {total} atoms, {len(frame)} type(s)")
    # one legend entry per chemistry class actually present
    seen_groups = frame["group"].drop_duplicates().tolist()
    handles = [
        dict(zip(frame["group"], bars))[g] for g in seen_groups
    ]
    ax.legend(handles, seen_groups, fontsize=9, loc="best", framealpha=0.9)
    return fig


def _energy_series(summary: pd.DataFrame, energy_column: str | None) -> tuple[pd.Series, pd.Series]:
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
        # energies are negative in practice: annotate OUTSIDE the bar tip
        # (ha="right" puts the label left of a negative bar end) instead
        # of on top of the coloured bar
        ax.text(
            value, pos, f" {value:.3g}", va="center",
            ha="right" if value < 0 else "left", fontsize=9,
        )
    ax.set_yticks(positions, frame["ligand"].tolist())
    ax.set_xlabel("best binding energy (kcal/mol) — lower is stronger")
    ax.set_title(title)
    return fig
