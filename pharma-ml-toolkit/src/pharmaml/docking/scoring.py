"""Post-docking scoring: ligand efficiency and consensus rescoring.

Upgrades the intent behind ``Prettier_extracted_dlgs.py`` /
``dlg_extracter.py`` (ranking ligands across a docking campaign) with
two standard analysis hooks:

* :func:`ligand_efficiency` -- binding energy normalised by heavy-atom
  count, the classic flat-scoring correction [1]_;
* :func:`consensus_rescore` -- min-max normalisation of several
  evidence columns into one 0–1 consensus score.

References
----------
.. [1] Hopkins, A. L. et al., J. Med. Chem. 47 (2004): 3385-3387.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = ["consensus_rescore", "ligand_efficiency", "ligand_efficiency_series"]


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
    population, ligand efficiency.

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

    score = sum(w * s for w, s in zip(weights, normalised)) / sum(weights)
    return score.clip(0.0, 1.0)
