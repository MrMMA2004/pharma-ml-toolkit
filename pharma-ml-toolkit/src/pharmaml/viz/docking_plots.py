"""Visual molecular-docking output — energy landscapes and rankings.

The archive's docking readers exported Excel tables and stopped; this
module visualises them:

* :func:`plot_energy_distribution` — histogram of best binding
  energies across a ligand library with the median marked;
* :func:`plot_top_ligands` — ranked horizontal bars of the strongest
  binders, ready to drop into a report or presentation;
* :func:`plot_consensus_ranking` — the same ranked-bar reading for the
  0–1 consensus (the figure twin of ``consensus_breakdown``'s
  leaderboard);
* :func:`plot_receptor_atom_types` — the AutoDock type census of a
  prepared receptor (donors, acceptors, carbons, metals), the figure
  behind the ``types A×87 C×526 ...`` line ``gpfer.py`` hard-coded;
* :func:`plot_contact_map` — the residue × ligand-atom minimum-
  distance grid of a docked pose (the matrix view of
  :func:`~pharmaml.docking.contacts.compute_contacts`);
* :func:`plot_plif_heatmap` — the residue × interaction-class PLIF
  counts (or their 0/1 fingerprint twin), the figure behind a
  pose-comparison workflow.

The first two accept the summary frame produced by
:func:`~pharmaml.docking.parsing.scan_docking_directory` directly.

Examples
--------
>>> from pathlib import Path
>>> from pharmaml.viz import close_figure
>>> from pharmaml.viz.docking_plots import plot_receptor_atom_types
>>> fig = plot_receptor_atom_types({"HD": 190, "A": 130, "N": 40})  # plain dict
>>> fig.axes[0].get_title()
'Receptor AutoDock atom types — 360 atoms, 3 type(s)'
>>> [t.get_text() for t in fig.axes[0].get_xticklabels()]          # sorted by count
['HD', 'A', 'N']
>>> close_figure(fig)
>>> pdbqt = "/tmp/pharmaml_doctest_receptor.pdbqt"
>>> _ = Path(pdbqt).write_text(
...     "ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00  0.00           N\\n"
...     "ATOM      2  CA  MET A   1       2.000   2.000   3.000  1.00  0.00           C\\n"
...     "ATOM      3  HA  MET A   1       2.500   1.500   3.000  1.00  0.00          HD\\n")
>>> fig = plot_receptor_atom_types(pdbqt)          # ATOM/HETATM records counted
>>> fig.axes[0].get_title()
'Receptor AutoDock atom types — 3 atoms, 3 type(s)'
>>> close_figure(fig)
>>> Path(pdbqt).unlink()
>>> plot_receptor_atom_types("nope.pdbqt")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: receptor PDBQT file not found: nope.pdbqt

>>> import pandas as pd
>>> from pharmaml.viz.docking_plots import plot_energy_distribution
>>> summary = pd.DataFrame({"folder": ["lig1", "lig2", "lig3", "lig4"],
...                         "best_lowest_energy": [-7.1, -8.5, -6.3, -9.2]})
>>> fig = plot_energy_distribution(summary)
>>> (len(fig.axes), fig.axes[0].get_title())
(1, 'Docking energy distribution — 4 ligand(s)')
>>> [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
['median = -7.8 kcal/mol', 'best = -9.2 (lig4)']
>>> close_figure(fig)
>>> plot_energy_distribution(pd.DataFrame({"x": [1]}))
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: summary frame needs a 'folder' column

>>> from pharmaml.viz.docking_plots import plot_top_ligands
>>> fig = plot_top_ligands(summary, top_n=3)
>>> (fig.axes[0].get_title(), fig.axes[0].get_xlabel())
('Top-ranked ligands by docking energy', 'best binding energy (kcal/mol) — lower is stronger')
>>> [t.get_text() for t in fig.axes[0].get_yticklabels()]      # strongest on top
['lig1', 'lig2', 'lig4']
>>> close_figure(fig)
>>> plot_top_ligands(summary, top_n=0)   # 0/-1 no longer mean "all rows"
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: top_n must be a positive integer or None, got 0
>>> plot_energy_distribution(summary, energy_column="binding")   # explicit column
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: energy_column 'binding' not found in the summary frame; available: ['folder', 'best_lowest_energy']

>>> from pharmaml.docking.contacts import (
...     ContactAtom, compute_contacts, plif_matrix)
>>> ligand = [ContactAtom(1, "N1", "N", "LIG", "L", 1, 0.0, 0.0, 0.0),
...            ContactAtom(2, "C1", "C", "LIG", "L", 1, 3.0, 0.0, 0.0)]
>>> receptor = [ContactAtom(1, "OD1", "O", "ASP", "A", 101, 0.0, 3.2, 0.0),
...             ContactAtom(2, "CZ2", "C", "TRP", "A", 62, 6.0, 3.0, 0.0)]
>>> contacts = compute_contacts(ligand, receptor, cutoff=5.0)
>>> fig = plot_contact_map(contacts)
>>> fig.axes[0].get_title()
'Contact map — 2 residue(s) × 2 ligand atom(s), 3 contact(s)'
>>> [t.get_text() for t in fig.axes[0].get_xticklabels()]
['L:LIG1 N1', 'L:LIG1 C1']
>>> [t.get_text() for t in fig.axes[0].get_yticklabels()]
['A:ASP101', 'A:TRP62']
>>> close_figure(fig)
>>> fig = plot_contact_map(contacts, atom_level=True)
>>> fig.axes[0].get_title()
'Contact map — 2 receptor atom(s) × 2 ligand atom(s), 3 contact(s)'
>>> close_figure(fig)
>>> plot_contact_map([])      # an empty contact list is upstream's bug
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: contacts must be a non-empty list of ContactRecord (the contact map has no data -- check the cutoff or the pose)
>>> fig = plot_plif_heatmap(contacts)
>>> fig.axes[0].get_title()
'PLIF interaction fingerprint — 2 residue(s) × 4 classes, 3 contact(s)'
>>> [t.get_text() for t in fig.axes[0].get_xticklabels()]
['salt_bridge', 'hbond', 'hydrophobic', 'vdw']
>>> close_figure(fig)
>>> fig = plot_plif_heatmap(contacts, bits=True)   # the 0/1 fingerprint twin
>>> fig.axes[0].get_title()
'PLIF bits — 2 residue(s) × 4 classes, 3 contact(s)'
>>> close_figure(fig)
>>> plot_plif_heatmap(plif_matrix(contacts))
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: contacts must be a list of ContactRecord (got DataFrame)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.docking.contacts import (
    ContactRecord,
    contact_matrix,
    plif_bits,
    plif_matrix,
)
from pharmaml.exceptions import DataValidationError
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = [
    "plot_consensus_ranking",
    "plot_contact_map",
    "plot_energy_distribution",
    "plot_plif_heatmap",
    "plot_receptor_atom_types",
    "plot_top_ligands",
]

#: Column scanned for the per-ligand best affinity.
_ENERGY_COLUMN = "best_lowest_energy"
#: Fallback column when the populated-cluster pick is unavailable.
_TOP_ENERGY_COLUMN = "top_lowest_energy"


def _check_top_n(top_n: int | None) -> None:
    """Typed guard for the shared ``top_n`` contract.

    ``None`` -> all rows; a positive integer -> that many rows;
    ``0``/negative/non-int values (bools included — they are truth
       values, not counts) raise the typed error.  Pre-fix the two
       plotters disagreed: ``plot_top_ligands`` read 0/-1 as "all rows"
       while ``plot_feature_importance`` read them as "empty"/"all but
       one" (100x-41-B).
    """
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1
    ):
        raise DataValidationError(
            f"top_n must be a positive integer or None, got {top_n!r}"
        )


def _check_bins_scalar(bins) -> None:
    """Typed guard for histogram ``bins`` scalars (0/-3/2.5 leak raw)."""
    # matplotlib accepts positive-int bins, bin-edge sequences and named
    # strategies ("auto", "fd", ...); any OTHER scalar leaks a raw
    # ValueError/TypeError from inside ax.hist — name it typed instead.
    if isinstance(bins, (int, float, np.integer, np.floating)) and not (
        isinstance(bins, (int, np.integer)) and bins >= 1
    ):
        raise DataValidationError(f"bins must be a positive integer, got {bins!r}")


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
        raw = dict(receptor.atom_types or {})
    elif isinstance(receptor, dict):
        raw = dict(receptor)
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
        raw = dict(census)
    # One typed conversion for every input path: a junk dict value used
    # to leak the raw ``int() argument must be ...`` / ``invalid literal
    # for int()`` from the old inline comprehension (100x-41-B corner).
    counts: dict[str, int] = {}
    for key, value in raw.items():
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"receptor atom-type counts must be integers, got {key!r}: {value!r}"
            ) from exc
        if count > 0:
            counts[str(key)] = count
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

    # An EXPLICITLY requested column must exist: silently falling back to
    # the default hid the typo whenever top_lowest_energy happened to be
    # present (100x-41-B corner).  The None path keeps the documented
    # best_lowest_energy -> top_lowest_energy fallback below.
    if energy_column is not None and energy_column not in summary.columns:
        raise DataValidationError(
            f"energy_column {energy_column!r} not found in the summary frame; "
            f"available: {list(summary.columns)}"
        )
    column = energy_column or _ENERGY_COLUMN
    if column not in summary.columns and _TOP_ENERGY_COLUMN in summary.columns:
        column = _TOP_ENERGY_COLUMN
    if column not in summary.columns:
        raise DataValidationError(
            f"no energy column found (looked for '{_ENERGY_COLUMN}' / "
            f"'{_TOP_ENERGY_COLUMN}'); available: {list(summary.columns)}"
        )
    numeric = pd.to_numeric(summary[column], errors="coerce")
    mask = numeric.notna()
    # Boolean-mask selection, NOT label lookup: with a duplicated frame
    # index (the classic append-a-re-run concat) `.loc[series.index]`
    # EXPANDS the repeated labels, misaligning folders vs energies.
    series = numeric[mask]
    folders = summary.loc[mask, "folder"]
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
        Override the affinity column.  The default (``None``) uses
        ``best_lowest_energy``, falling back to ``top_lowest_energy``
        when only the fallback exists; an explicitly passed column must
        be present in the frame — a missing one raises
        :class:`~pharmaml.exceptions.DataValidationError` naming it
        instead of silently falling back.
    bins:
        Histogram bins — any :meth:`matplotlib.axes.Axes.hist` spec:
        a positive integer, a bin-edge sequence, or a named strategy
        such as ``"auto"``.  A non-positive integer scalar raises the
        typed error below instead of matplotlib's raw ``ValueError``.
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``summary`` is not a DataFrame, lacks the ``folder`` column
        or any energy column, the energy column holds no numeric
        values, an explicit ``energy_column`` is missing from the
        frame, or ``bins`` is a non-positive integer.
    """
    _check_bins_scalar(bins)
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
    top_n: int | None = 15,
    energy_column: str | None = None,
    title: str = "Top-ranked ligands by docking energy",
) -> Any:
    """Horizontal bars of the strongest-binding ligands.

    Bars are sorted so the strongest binder sits on top; values are
    annotated and the colour encodes energy (darker = stronger), which
    reads well both on screen and in exported png/svg/pdf files.

    Parameters
    ----------
    summary:
        Frame from :func:`~pharmaml.docking.parsing.scan_docking_directory`.
    top_n:
        How many ligands to show; ``None`` shows every ligand in the
        frame.  Values below 1 and non-integers raise
        :class:`~pharmaml.exceptions.DataValidationError` — the old
        ``0``/``-1`` silently meant "all rows" here while meaning
        "empty"/"all but one" in
        :func:`~pharmaml.viz.ml_plots.plot_feature_importance`.
    energy_column:
        Override the affinity column.  The default (``None``) uses
        ``best_lowest_energy``, falling back to ``top_lowest_energy``
        when only the fallback exists; an explicitly passed column must
        be present in the frame — a missing one raises
        :class:`~pharmaml.exceptions.DataValidationError` naming it
        instead of silently falling back.
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``top_n`` is not a positive integer or ``None``, the summary
        frame fails the energy-column contract above, or an explicit
        ``energy_column`` is missing from the frame.
    """
    _check_top_n(top_n)
    folders, energies = _energy_series(summary, energy_column)
    frame = pd.DataFrame({"ligand": [str(name) for name in folders],
                          "energy": energies.to_numpy(dtype=float)})
    # kind="stable": ties resolve first-declared-wins — the default
    # quicksort cut is an arbitrary block from the middle of the frame at scale
    frame = frame.sort_values("energy", ascending=True, kind="stable").head(
        len(frame) if top_n is None else int(top_n)
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


def plot_consensus_ranking(
    rescored: pd.DataFrame,
    identity_column: str | None = None,
    consensus_column: str = "consensus",
    top_n: int | None = 15,
    title: str = "Consensus ranking — strongest evidence wins",
) -> Any:
    """Horizontal bars of ligands ranked by the 0–1 consensus.

    The figure twin of :func:`~pharmaml.docking.consensus_breakdown`'s
    leaderboard: bars are sorted so the strongest consensus sits on top
    of the axis, the value is annotated outside the bar tip and the
    identity labels come from ``identity_column`` (the frame's first
    column by default — ``folder`` for campaign scans, ``ligand`` for
    enriched tables, ``rank`` for cluster tables).  Rows whose consensus
    is NaN (incomplete inputs) cannot rank and are dropped.

    Parameters
    ----------
    rescored:
        Frame carrying the consensus column — the output of
        ``consensus_rescore``/``consensus_breakdown`` joined back onto the
        summary table (or the ``docking rescore`` CLI's written table).
    identity_column:
        Column naming each row; ``None`` uses the frame's first column.
        An explicitly passed column must exist — a missing one raises
        the typed error naming it.
    consensus_column:
        The 0–1 score column (default ``"consensus"``).
    top_n:
        How many ligands to show; ``None`` shows every ligand in the
        frame.  Values below 1 and non-integers raise
        :class:`~pharmaml.exceptions.DataValidationError` (the same
        contract as :func:`plot_top_ligands`).
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``rescored`` is not a DataFrame, the consensus column is
        missing or holds no numeric values, the identity column is
        missing, or ``top_n`` is invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.viz import close_figure
    >>> from pharmaml.viz.docking_plots import plot_consensus_ranking
    >>> rescored = pd.DataFrame({"ligand": ["A", "B", "C"],
    ...                          "consensus": [0.2, 0.9, 0.5]})
    >>> fig = plot_consensus_ranking(rescored)
    >>> [t.get_text() for t in fig.axes[0].get_yticklabels()]   # strongest on top
    ['A', 'C', 'B']
    >>> fig.axes[0].get_xlabel()
    'consensus (0–1, weighted mean of the _norm units) — higher is stronger'
    >>> close_figure(fig)
    >>> plot_consensus_ranking(pd.DataFrame({"ligand": ["A"]}))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: consensus column 'consensus' not found in the rescored frame; available: ['ligand']
    """
    _check_top_n(top_n)
    if not isinstance(rescored, pd.DataFrame):
        raise DataValidationError(
            f"rescored must be a pandas DataFrame, got {type(rescored).__name__}"
        )
    if consensus_column not in rescored.columns:
        raise DataValidationError(
            f"consensus column {consensus_column!r} not found in the "
            f"rescored frame; available: {list(rescored.columns)}"
        )
    if identity_column is None:
        identity_column = str(rescored.columns[0])
    elif identity_column not in rescored.columns:
        raise DataValidationError(
            f"identity column {identity_column!r} not found in the "
            f"rescored frame; available: {list(rescored.columns)}"
        )
    scores = pd.to_numeric(rescored[consensus_column], errors="coerce")
    frame = pd.DataFrame({
        "ligand": [str(name) for name in rescored[identity_column]],
        "score": scores.to_numpy(dtype=float),
    }).dropna(subset=["score"])
    if frame.empty:
        raise DataValidationError(
            f"column {consensus_column!r} has no numeric consensus values "
            "to rank (all NaN or non-numeric)"
        )
    # kind="stable": ties resolve first-declared-wins, matching the
    # leaderboard's sort (and plot_top_ligands' tie contract).
    frame = frame.sort_values("score", ascending=False, kind="stable").head(
        len(frame) if top_n is None else int(top_n)
    )
    frame = frame.iloc[::-1]  # strongest consensus at the top of the axis

    fig, ax = new_figure(figsize=(8.6, 0.44 * len(frame) + 1.7))
    positions = np.arange(len(frame))
    ax.barh(positions, frame["score"].to_numpy(), color=PALETTE[0], alpha=0.88,
            height=0.62, edgecolor="white")
    for pos, value in zip(positions, frame["score"].to_numpy()):
        # consensus is 0-1: annotate OUTSIDE the bar tip (ha="left" puts
        # the label right of the positive bar end), never on top of it
        ax.text(
            value, pos, f" {value:.3g}", va="center",
            ha="right" if value < 0 else "left", fontsize=9,
        )
    ax.set_yticks(positions, frame["ligand"].tolist())
    ax.set_xlim(0.0, max(1.0, float(frame["score"].max()) * 1.12))
    ax.set_xlabel(
        "consensus (0–1, weighted mean of the _norm units) — higher is stronger"
    )
    ax.set_title(title)
    return fig


# -- contact-map / PLIF twins (docking.contacts views) --------------------------

def _check_contacts(contacts: Any, allow_empty: bool = False) -> list[ContactRecord]:
    """Validate the contacts argument type (the twins' shared gate)."""
    if not isinstance(contacts, list):
        raise DataValidationError(
            "contacts must be a list of ContactRecord "
            f"(got {type(contacts).__name__})"
        )
    if not contacts and not allow_empty:
        raise DataValidationError(
            "contacts must be a non-empty list of ContactRecord (the "
            "contact map has no data -- check the cutoff or the pose)"
        )
    if contacts and not isinstance(contacts[0], ContactRecord):
        raise DataValidationError(
            "contacts must be a list of ContactRecord "
            f"(got {type(contacts[0]).__name__})"
        )
    return contacts


def plot_contact_map(
    contacts: list[ContactRecord],
    atom_level: bool = False,
    title: str = "Contact map",
) -> Any:
    """Residue × ligand-atom minimum-distance grid of a docked pose.

    The matrix view of
    :func:`~pharmaml.docking.contacts.compute_contacts`: one cell per
    receptor residue (row, closest-contact order) × ligand atom
    (column, serial order), the cell painted with the minimum
    ligand-atom/receptor-atom distance and left blank where that pair
    never touches.  Close contacts paint dark (the house palette ramp),
    the no-contact cells stay white -- the classic contact-map reading
    of a binding pose.

    Parameters
    ----------
    contacts:
        Output of :func:`~pharmaml.docking.contacts.compute_contacts`;
        a non-empty list, or the typed error below (an empty map is
        upstream's bug -- cutoff too tight, pose outside the pocket).
    atom_level:
        ``False`` (default) aggregates rows by receptor residue (the
        readable map); ``True`` keeps every receptor atom its own row
        (the detailed map -- ``A:ASP101 OD1``-style labels).
    title:
        Chart title; the counts ride the suffix `` — R × L, N
    contact(s)``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``contacts`` is not a list of :class:`~pharmaml.docking.
        contacts.ContactRecord`, is empty, or ``atom_level`` is not a
        bool.
    """
    _check_contacts(contacts)
    if not isinstance(atom_level, bool):
        raise DataValidationError(
            f"atom_level must be a bool (got {atom_level!r})"
        )
    grid = contact_matrix(contacts, atom_level=atom_level)
    values = grid.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    fig, ax = new_figure(figsize=(1.1 + 0.55 * grid.shape[1],
                                  1.6 + 0.38 * grid.shape[0]))
    from matplotlib.colors import LinearSegmentedColormap
    ramp = LinearSegmentedColormap.from_list(
        "contact_ramp", ["#0F766E", "#14B8A6", "#D9F0EC"])
    ramp.set_bad("#FFFFFF")
    im = ax.imshow(masked, cmap=ramp, aspect="auto", vmin=2.5, vmax=5.0)
    ax.set_xticks(range(grid.shape[1]), list(grid.columns), rotation=45,
                  ha="right", fontsize=8)
    ax.set_yticks(range(grid.shape[0]), list(grid.index), fontsize=8)
    if grid.shape[0] > 22:
        ax.set_yticks(range(grid.shape[0]), list(grid.index), fontsize=6)
    row_name = "receptor atom" if atom_level else "receptor residue"
    unit = "receptor atom" if atom_level else "residue"
    ax.set_xlabel("ligand atom")
    ax.set_ylabel(row_name)
    n = len(contacts)
    ax.set_title(
        f"{title} — {grid.shape[0]} {unit}(s) × {grid.shape[1]} "
        f"ligand atom(s), {n} contact(s)"
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("contact distance (Å)")
    ax.grid(False)
    return fig


def plot_plif_heatmap(
    contacts: list[ContactRecord],
    bits: bool = False,
    title: str = "PLIF",
) -> Any:
    """Residue × interaction-class PLIF counts (or the 0/1 twin).

    The figure behind a pose-comparison workflow: rows are receptor
    residues (closest-contact order), columns the interaction classes
    of :data:`~pharmaml.docking.contacts.INTERACTION_CLASSES`, each
    cell the count of contacts of that class to that residue -- or, with
    ``bits=True``, the 1-when-any 0-otherwise bit vector a Tanimoto
    pose-similarity would run over.

    Parameters
    ----------
    contacts:
        Output of :func:`~pharmaml.docking.contacts.compute_contacts`;
        a non-empty list, or the typed error below.
    bits:
        ``False`` (default) paints the class counts; ``True`` paints
        the 0/1 fingerprint (the title switches to ``PLIF bits``).
    title:
        Chart title; the counts ride the suffix.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``contacts`` is not a list of :class:`~pharmaml.docking.
        contacts.ContactRecord`, is empty, or ``bits`` is not a bool.
    """
    _check_contacts(contacts)
    if not isinstance(bits, bool):
        raise DataValidationError(f"bits must be a bool (got {bits!r})")
    frame = plif_bits(contacts) if bits else plif_matrix(contacts)
    frame = frame.drop(columns=["total"])
    values = frame.to_numpy(dtype=float)

    fig, ax = new_figure(figsize=(1.4 + 1.05 * frame.shape[1],
                                  1.6 + 0.38 * frame.shape[0]))
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    if bits:
        cmap: object = ListedColormap(["#FFFFFF", "#B45309"])
    else:
        cmap = LinearSegmentedColormap.from_list(
            "plif_ramp", ["#FFFFFF", "#FDE9D9", "#B45309"])
    im = ax.imshow(values, cmap=cmap, aspect="auto",
                   vmin=0.0, vmax=max(1.0, float(values.max())))
    ax.set_xticks(range(frame.shape[1]), list(frame.columns), rotation=30,
                  ha="right", fontsize=9)
    ax.set_yticks(range(frame.shape[0]), list(frame.index), fontsize=8)
    if frame.shape[0] > 22:
        ax.set_yticks(range(frame.shape[0]), list(frame.index), fontsize=6)
    ax.set_xlabel("interaction class")
    ax.set_ylabel("receptor residue")
    kind = "bits" if bits else "interaction fingerprint"
    ax.set_title(
        f"{title} {kind} — {frame.shape[0]} residue(s) × "
        f"{frame.shape[1]} classes, {len(contacts)} contact(s)"
    )
    if not bits:  # annotate the counts (the bits speak for themselves)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if values[i, j] > 0:
                    ax.text(j, i, int(values[i, j]), ha="center",
                            va="center", fontsize=8,
                            color="#7C2D12" if values[i, j] < values.max() / 2
                            else "#FFFFFF")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("residue contacts (count)" if not bits else "class touched (0/1)")
    ax.grid(False)
    return fig
