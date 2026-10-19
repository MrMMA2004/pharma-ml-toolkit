"""Post-docking scoring: ligand efficiency and consensus rescoring.

Upgrades the intent behind ``Prettier_extracted_dlgs.py`` /
``dlg_extracter.py`` (ranking ligands across a docking campaign) with
two standard analysis hooks:

* :func:`ligand_efficiency` -- binding energy normalised by heavy-atom
  count, the classic flat-scoring correction [1]_;
* :func:`consensus_rescore` -- min-max normalisation of several
  evidence columns into one 0–1 consensus score;
* :func:`consensus_breakdown` -- the same normalisation, kept apart:
  one 0–1 unit column per evidence column plus the consensus, so a
  ranking can be *explained* (which evidence drove which ligand).

References
----------
.. [1] Hopkins, A. L. et al., J. Med. Chem. 47 (2004): 3385-3387.

Examples
--------
>>> from pharmaml.docking.scoring import ligand_efficiency
>>> ligand_efficiency(-12.4, 31)          # |dG| / N_heavy
0.4
>>> ligand_efficiency(-9.0, 30)
0.3
>>> ligand_efficiency(-8.0, 0)            # non-positive atom count
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: num_heavy_atoms must be a positive integer, got 0

>>> import pandas as pd
>>> from pharmaml.docking.scoring import ligand_efficiency_series
>>> docked = pd.DataFrame({"best_lowest_energy": [-11.0, -8.0],
...                        "num_heavy_atoms": [22, 16]})
>>> le = ligand_efficiency_series(docked)
>>> le.tolist()
[0.5, 0.5]
>>> le.name                                # the affinity column, by default
'best_lowest_energy'

>>> from pharmaml.docking.scoring import consensus_rescore
>>> summary = pd.DataFrame({"ligand": ["L1", "L2", "L3"],
...                         "best_lowest_energy": [-10.0, -8.0, -6.0],
...                         "cluster_size": [10, 30, 20]})
>>> consensus_rescore(summary, ["best_lowest_energy", "cluster_size"]).tolist()
[0.5, 0.75, 0.25]
>>> consensus_rescore(summary, ["best_lowest_energy", "cluster_size"],
...                   weights=[1.0, 3.0]).tolist()
[0.25, 0.875, 0.375]
>>> consensus_rescore(summary, ["best_lowest_energy", "cluster_size"],
...                   higher_is_better={"best_lowest_energy": True}).tolist()
[0.0, 0.75, 0.75]
>>> consensus_rescore(summary, ["cluster"])       # unknown column
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: missing columns ['cluster'] for rescoring
>>> consensus_rescore(summary, ["best_lowest_energy", "cluster_size"],
...                   weights=[1.0])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: got 1 weights for 2 columns

>>> from pharmaml.docking.scoring import consensus_breakdown
>>> breakdown = consensus_breakdown(
...     summary, ["best_lowest_energy", "cluster_size"])
>>> breakdown.round(2).to_dict("list")          # the explainable ranking
{'best_lowest_energy_norm': [1.0, 0.5, 0.0], 'cluster_size_norm': [0.0, 1.0, 0.5], 'consensus': [0.5, 0.75, 0.25]}
>>> breakdown["consensus"].tolist() == consensus_rescore(          # same core
...     summary, ["best_lowest_energy", "cluster_size"]).tolist()
True
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "consensus_breakdown",
    "consensus_rescore",
    "ligand_efficiency",
    "ligand_efficiency_series",
]


def _validated_atom_count(value: object, atoms_col: str) -> int:
    """Coerce one table cell to a positive integer atom count.

    Accepts ``int``/``np.integer`` and whole-number floats (pandas
    float64 columns holding counts); fractional floats, strings and
    NaN raise :class:`DataValidationError` naming the value, so a dirty
    ``atoms_col`` cannot silently truncate (``12.5`` -> ``12``) or crash
    with a raw ``ValueError``.
    """
    count: int | None = None
    if isinstance(value, (int, np.integer)) or isinstance(value, (float, np.floating)) and float(value).is_integer():
        count = int(value)
    if count is None or count <= 0:
        raise DataValidationError(
            f"column {atoms_col!r} must hold positive integer atom counts, "
            f"got {value!r}"
        )
    return count


def ligand_efficiency(affinity_kcal_mol: float, num_heavy_atoms: int) -> float:
    """Ligand efficiency ``LE = |ΔG| / N_heavy`` (kcal/mol per atom).

    Parameters
    ----------
    affinity_kcal_mol:
        Docking binding energy (kcal/mol, typically negative).
    num_heavy_atoms:
        Number of non-hydrogen atoms in the ligand.

    Raises
    ------
    DataValidationError
        If the atom count is not a positive integer.
    """
    if not isinstance(num_heavy_atoms, (int, np.integer)) or num_heavy_atoms <= 0:
        raise DataValidationError(
            f"num_heavy_atoms must be a positive integer, got {num_heavy_atoms!r}"
        )
    return abs(affinity_kcal_mol) / num_heavy_atoms


def ligand_efficiency_series(
    frame: pd.DataFrame,
    affinity_col: str = "best_lowest_energy",
    atoms_col: str = "num_heavy_atoms",
) -> pd.Series:
    """Vectorised ligand-efficiency column for a docking summary table.

    Parameters
    ----------
    frame:
        Docking summary table (one row per ligand); the result carries
        the frame's index, so it can be assigned straight back as a
        column.  An empty frame yields an empty float Series.
    affinity_col:
        Binding-energy column (kcal/mol).  Missing/non-numeric
        affinities produce NaN efficiencies and a warning instead of a
        raw ``TypeError``.
    atoms_col:
        Heavy-atom count column.  Values must be positive integers
        (whole-number floats such as ``30.0`` are accepted); anything
        else raises with the offending value named.

    Raises
    ------
    DataValidationError
        If either column is missing, or ``atoms_col`` holds values that
        are not positive integers (e.g. ``12.5``, ``"n/a"``, NaN).
    """
    for col in (affinity_col, atoms_col):
        if col not in frame.columns:
            raise DataValidationError(
                f"column {col!r} missing; available: {list(frame.columns)}"
            )
    affinities = pd.to_numeric(frame[affinity_col], errors="coerce")
    n_missing = int(affinities.isna().sum())
    if n_missing:
        _LOG.warning(
            "column %r has %d missing/non-numeric affinity value(s); "
            "their ligand efficiencies will be NaN",
            affinity_col,
            n_missing,
        )
    atoms = [_validated_atom_count(value, atoms_col) for value in frame[atoms_col]]
    return affinities.abs() / atoms


def _normalised_evidence(
    frame: pd.DataFrame,
    columns: list[str],
    weights: list[float] | None,
    higher_is_better: dict[str, bool] | None,
) -> tuple[list[pd.Series], list[float]]:
    """Validate the consensus inputs and normalise each column to 0–1.

    The shared core of :func:`consensus_rescore` and
    :func:`consensus_breakdown`: every check (empty list, missing
    columns, weight length/values, empty frame, all-NaN column) and the
    per-column min-max normalisation with direction flipping, NaN
    policy and the zero-variance neutral 0.5 -- so the two public
    functions can never drift apart.

    Returns
    -------
    tuple of (list of pandas.Series, list of float)
        One normalised unit Series per requested column (direction
        applied: higher is always better) and the effective weights
        (uniform when ``weights`` is None).
    """
    if not columns:
        raise DataValidationError(
            "columns list is empty -- pass the evidence columns to combine"
        )
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DataValidationError(f"missing columns {missing} for rescoring")
    if frame.empty:
        raise DataValidationError(
            "frame is empty -- consensus_rescore needs at least one row"
        )
    if weights is not None and len(weights) != len(columns):
        raise DataValidationError(
            f"got {len(weights)} weights for {len(columns)} columns"
        )
    if not weights:
        weights = [1.0] * len(columns)
    # Negative or all-zero weights divide by zero inside pandas (a silent
    # NaN Series, not an exception) -- reject them with a named error.
    if any(w < 0 for w in weights):
        raise DataValidationError(
            "consensus weights must be non-negative, got "
            f"{weights} (use higher_is_better to flip a column's direction)"
        )
    if sum(weights) <= 0:
        raise DataValidationError(
            f"consensus weights must not all be zero, got {weights}"
        )

    direction = higher_is_better or {}
    normalised: list[pd.Series] = []
    for col in list(columns):
        values = pd.to_numeric(frame[col], errors="coerce")
        invalid = ~np.isfinite(values)
        n_invalid = int(np.asarray(invalid).sum())
        if n_invalid == len(values):
            raise DataValidationError(
                f"column {col!r} has no numeric values to rescore "
                "(all NaN or non-numeric)"
            )
        if n_invalid:
            _LOG.warning(
                "column %r has %d NaN/non-numeric value(s); their "
                "consensus rows will be NaN",
                col,
                n_invalid,
            )
            values = values.where(np.isfinite(values))
        span = values.max() - values.min()
        if span == 0:
            _LOG.warning("column %r has zero variance; scoring it as 0.5", col)
            # The neutral 0.5 must honour the NaN policy too: rows whose
            # value in THIS column is NaN/inf stay NaN (the warning above
            # promises "their consensus rows will be NaN"), instead of
            # inheriting a bogus scored 0.5.
            normalised.append(
                pd.Series(0.5, index=frame.index).where(values.notna())
            )
            continue
        unit = (values - values.min()) / span
        if not direction.get(col, not col.endswith(("energy", "affinity"))):
            # Default: binding-energy/affinity columns (names ending in
            # "energy" or "affinity") are "lower is better".
            unit = 1.0 - unit
        normalised.append(unit)
    return normalised, weights


def _weighted_consensus(
    normalised: list[pd.Series], weights: list[float]
) -> pd.Series:
    """Weighted mean of the normalised units, clipped to [0, 1]."""
    score = sum(w * s for w, s in zip(weights, normalised)) / sum(weights)
    return score.clip(0.0, 1.0)


def consensus_rescore(
    frame: pd.DataFrame,
    columns: list[str],
    weights: list[float] | None = None,
    higher_is_better: dict[str, bool] | None = None,
) -> pd.Series:
    """Combine several docking evidence columns into a 0–1 consensus.

    Each column is min-max normalised across the table (direction
    configurable per column), then a weighted average produces the
    final score.  Typical inputs: lowest binding energy, cluster
    population, ligand efficiency.  Use :func:`consensus_breakdown`
    when the ranking itself must be explainable (it keeps the
    per-column normalised units next to the consensus).

    Parameters
    ----------
    frame:
        Docking summary table (one row per ligand).
    columns:
        Columns to combine.
    weights:
        Per-column weights (uniform when None).  Must match length.
    higher_is_better:
        Map column -> direction.  Columns whose names end in
        ``energy`` or ``affinity`` (e.g. ``best_lowest_energy``,
        ``best_affinity`` -- the outputs of :func:`parse_dlg_summary`
        and :func:`scan_vina_directory`) default to False (more
        negative = better); everything else (cluster sizes, ligand
        efficiency, ...) defaults to True.

    Returns
    -------
    pandas.Series
        Consensus score in [0, 1]; higher is always better.  Rows whose
        input columns hold NaN/inf keep NaN (with a warning naming the
        column) instead of silently poisoning the table.

    Raises
    ------
    DataValidationError
        On an empty ``columns`` list, missing columns, weight-length
        mismatch, invalid weights (negative or all-zero), an empty
        frame, or a column with no numeric values at all (all NaN or
        non-numeric after coercion).
    """
    normalised, effective = _normalised_evidence(
        frame, columns, weights, higher_is_better
    )
    return _weighted_consensus(normalised, effective)


def consensus_breakdown(
    frame: pd.DataFrame,
    columns: list[str],
    weights: list[float] | None = None,
    higher_is_better: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Explain a consensus ranking: per-column units plus the score.

    The introspection twin of :func:`consensus_rescore` -- identical
    validation and normalisation (the shared core), but instead of
    collapsing the evidence into one Series it returns a frame with one
    ``<column>_norm`` column per evidence column (each min-max
    normalised to 0–1, direction applied so higher is always better)
    plus the final ``consensus`` column.  Read a row across the ``_norm``
    columns to see *which* evidence drove a ligand's rank -- the answer
    to "why is this compound on top?".

    Parameters
    ----------
    frame:
        Docking summary table (one row per ligand).
    columns:
        Evidence columns to combine.
    weights:
        Per-column weights (uniform when None).  Must match length.
    higher_is_better:
        Map column -> direction, overriding the energy/affinity
        name heuristic (see :func:`consensus_rescore`).

    Returns
    -------
    pandas.DataFrame
        Columns ``<evidence>_norm`` (one per requested column, 0–1,
        higher = better) and ``consensus`` (the weighted mean, clipped
        to [0, 1]; higher = better).  The index matches ``frame``;
        rows whose inputs hold NaN/inf keep NaN exactly like
        :func:`consensus_rescore`.

    Raises
    ------
    DataValidationError
        Same contract as :func:`consensus_rescore` (shared validation
        core): empty ``columns``, missing columns, weight-length
        mismatch, negative or all-zero weights, empty frame, or an
        all-NaN evidence column.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.docking.scoring import consensus_breakdown
    >>> summary = pd.DataFrame({"ligand": ["L1", "L2", "L3"],
    ...                         "best_lowest_energy": [-10.0, -8.0, -6.0],
    ...                         "cluster_size": [10, 30, 20]})
    >>> breakdown = consensus_breakdown(summary,
    ...     ["best_lowest_energy", "cluster_size"], weights=[1.0, 3.0])
    >>> list(breakdown.columns)
    ['best_lowest_energy_norm', 'cluster_size_norm', 'consensus']
    >>> breakdown["consensus"].round(3).tolist()   # L2 wins on cluster size
    [0.25, 0.875, 0.375]
    >>> breakdown["cluster_size_norm"].tolist()  # the driving evidence
    [0.0, 1.0, 0.5]
    """
    normalised, effective = _normalised_evidence(
        frame, columns, weights, higher_is_better
    )
    breakdown = pd.DataFrame(index=frame.index)
    for col, unit in zip(columns, normalised):
        breakdown[f"{col}_norm"] = unit
    breakdown["consensus"] = _weighted_consensus(normalised, effective)
    return breakdown
