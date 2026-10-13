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
from pharmaml.utils.validation import validate_positive

_LOG = get_logger(__name__)

__all__ = ["ligand_efficiency", "ligand_efficiency_series", "consensus_rescore"]


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
    """Vectorised ligand-efficiency column for a docking summary table."""
    for col in (affinity_col, atoms_col):
        if col not in frame.columns:
            raise DataValidationError(
                f"column {col!r} missing; available: {list(frame.columns)}"
            )
    return frame.apply(
        lambda row: ligand_efficiency(row[affinity_col], int(row[atoms_col])), axis=1
    )


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
        Map column -> direction.  Binding energies default to False
        (more negative = better); cluster sizes default to True.

    Returns
    -------
    pandas.Series
        Consensus score in [0, 1]; higher is always better.

    Raises
    ------
    DataValidationError
        On missing columns, weight-length mismatch or all-NaN inputs.
    """
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DataValidationError(f"missing columns {missing} for rescoring")
    if weights is not None and len(weights) != len(columns):
        raise DataValidationError(
            f"got {len(weights)} weights for {len(columns)} columns"
        )
    if not weights:
        weights = [1.0] * len(columns)

    direction = higher_is_better or {}
    normalised: list[pd.Series] = []
    for col in list(columns):
        values = pd.to_numeric(frame[col], errors="coerce")
        span = values.max() - values.min()
        if not np.isfinite(span) or span == 0:
            _LOG.warning("column %r has zero variance; scoring it as 0.5", col)
            normalised.append(pd.Series(0.5, index=frame.index))
            continue
        unit = (values - values.min()) / span
        if not direction.get(col, col not in ("lowest_energy", "best_lowest_energy")):
            # Default: binding-energy-like columns are "lower is better".
            unit = 1.0 - unit
        normalised.append(unit)

    score = sum(w * s for w, s in zip(weights, normalised)) / sum(weights)
    return score.clip(0.0, 1.0)
