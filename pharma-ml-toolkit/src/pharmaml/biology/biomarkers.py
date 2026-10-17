"""Biomarker discovery: gene-expression / drug-sensitivity associations.

This module answers the canonical cell-line omics question *"which
transcripts predict drug sensitivity?"*: correlate every gene's
expression across the panel against one drug's response vector, then
control the multiple-testing burden.

* :func:`sensitivity_associations` is THE biomarker-discovery call --
  per-gene Spearman (or Pearson) correlations with Benjamini-Hochberg
  FDR control via :func:`scipy.stats.false_discovery_control`,
* :func:`top_biomarkers` trims the association table to the strongest
  hits, and
* :func:`direction_of_association` translates a correlation sign into
  an English sentence, honouring the response metric's direction
  (LN_IC50-style: lower = more sensitive; pIC50-style: higher = more
  sensitive).

Examples
--------
>>> import pandas as pd
>>> from pharmaml.biology.biomarkers import (
...     direction_of_association, sensitivity_associations, top_biomarkers)
>>> expression = pd.DataFrame(
...     {"ABCB1": [1.0, 2.0, 3.0, 4.0, 5.0],
...      "CYP3A4": [2.0, 1.0, 3.0, 5.0, 4.0],
...      "GAPDH": [3.0, 3.0, 3.0, 3.0, 3.0],
...      "SRC": [5.0, 4.0, 3.0, 2.0, 1.0]},
...     index=["cell_a", "cell_b", "cell_c", "cell_d", "cell_e"],
... )
>>> response = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=expression.index, name="LN_IC50")
>>> associations = sensitivity_associations(expression, response)
>>> associations.columns.tolist()
['gene', 'rho', 'p_value', 'fdr', 'significant']
>>> associations["gene"].tolist()                 # ranked by |rho|, ties by gene name
['ABCB1', 'SRC', 'CYP3A4', 'GAPDH']
>>> associations.set_index("gene")["rho"].round(3).to_dict()
{'ABCB1': 1.0, 'SRC': -1.0, 'CYP3A4': 0.8, 'GAPDH': nan}
>>> associations.set_index("gene")["fdr"].round(4).to_dict()
{'ABCB1': 0.0, 'SRC': 0.0, 'CYP3A4': 0.1041, 'GAPDH': nan}
>>> associations["significant"].tolist()          # BH-controlled at the default alpha
[True, True, False, False]
>>> sensitivity_associations(expression, response, fdr_alpha=1.0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: fdr_alpha must lie strictly between 0 and 1, got 1.0

>>> top_biomarkers(associations, n=1)["gene"].tolist()
['ABCB1']
>>> top_biomarkers(associations, n=5)["gene"].tolist()   # only 2 significant: all rows
['ABCB1', 'SRC', 'CYP3A4', 'GAPDH']
>>> top_biomarkers(associations, n=0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: n must be >= 1, got 0
>>> sensitivity_associations(expression, response, method="pearson")["rho"].round(3).tolist()
[1.0, -1.0, 0.8, nan]

>>> direction_of_association(-0.9)               # LN_IC50-style: negative rho
'expression high -> more sensitive'
>>> direction_of_association(0.9)                # LN_IC50-style: positive rho
'expression high -> more resistant'
>>> direction_of_association(0.9, lower_is_sensitive=False)   # pIC50-style response
'expression high -> more sensitive'
>>> direction_of_association(float("nan"))
'no association'
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from pharmaml.biology.omics import correlate_with_response
from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.utils.validation import require_columns

_LOG = get_logger(__name__)

__all__ = [
    "direction_of_association",
    "sensitivity_associations",
    "top_biomarkers",
]

#: Columns every association table produced by
#: :func:`sensitivity_associations` carries.
_ASSOCIATION_COLUMNS = ["gene", "rho", "p_value", "fdr", "significant"]


def sensitivity_associations(
    expression_frame: pd.DataFrame,
    response_series,
    method: str = "spearman",
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Correlate every gene with one drug's response, FDR-controlled.

    The biomarker-discovery call: each gene's expression across the
    cell-line panel is correlated against ``response_series`` (see
    :func:`pharmaml.biology.omics.correlate_with_response` for the
    alignment and NaN rules), then the p-values are corrected for
    multiple testing with Benjamini-Hochberg
    (:func:`scipy.stats.false_discovery_control`) computed over the
    non-NaN p-values only.

    Parameters
    ----------
    expression_frame:
        Expression table (cell lines x genes).
    response_series:
        One drug's response values indexed by cell line.  Direction is
        whatever this metric's direction is -- see
        :func:`direction_of_association` for interpreting the sign.
    method:
        ``"spearman"`` (default) or ``"pearson"``.
    fdr_alpha:
        Significance level for the ``significant`` flag.

    Returns
    -------
    pandas.DataFrame
        Columns ``gene``, ``rho``, ``p_value``, ``fdr`` and
        ``significant`` (``fdr < fdr_alpha``; NaN p-values stay NaN and
        are never significant), sorted by ``|rho|`` descending with
        ties broken by gene name (deterministic).

    Raises
    ------
    DataValidationError
        If ``fdr_alpha`` is outside (0, 1) or the correlation inputs
        cannot be aligned (see :func:`correlate_with_response`).
    """
    if not 0.0 < fdr_alpha < 1.0:
        raise DataValidationError(
            f"fdr_alpha must lie strictly between 0 and 1, got {fdr_alpha}"
        )
    correlations = correlate_with_response(
        expression_frame, response_series, method=method
    )
    associations = correlations.copy()

    p_values = associations["p_value"].to_numpy(dtype=float)
    corrected = np.full(p_values.shape, np.nan, dtype=float)
    finite = ~np.isnan(p_values)
    if finite.any():
        corrected[finite] = false_discovery_control(
            p_values[finite], method="bh"
        )
    associations["fdr"] = corrected
    associations["significant"] = associations["fdr"] < fdr_alpha

    associations = associations.assign(
        _abs_rho=associations["rho"].abs(),
        _gene=associations["gene"],
    ).sort_values(
        ["_abs_rho", "_gene"], ascending=[False, True], na_position="last"
    )
    associations = associations.drop(columns=["_abs_rho", "_gene"]).reset_index(
        drop=True
    )
    n_significant = int(associations["significant"].sum())
    _LOG.info(
        "sensitivity_associations: %d gene(s), %d significant at FDR < %s (%s)",
        len(associations),
        n_significant,
        fdr_alpha,
        method,
    )
    return associations[_ASSOCIATION_COLUMNS]


def top_biomarkers(
    associations: pd.DataFrame, n: int = 10, only_significant: bool = True
) -> pd.DataFrame:
    """Trim an association table to its strongest hits.

    Parameters
    ----------
    associations:
        Output of :func:`sensitivity_associations` (requires the
        columns ``gene``, ``rho``, ``p_value``, ``fdr`` and
        ``significant``).
    n:
        Number of rows to return.
    only_significant:
        Keep only FDR-significant rows (default).  When fewer than
        ``n`` significant rows exist the full table is returned instead
        (still ranked by ``|rho|``) with a warning -- an empty or
        truncated result would silently hide the runner-up genes.
        This flag only matters for hand-crafted tables where significance
        and ``|rho|`` rankings disagree; :func:`sensitivity_associations`
        output already ranks its significant rows first.

    Returns
    -------
    pandas.DataFrame
        At most ``n`` rows of ``associations``.

    Raises
    ------
    DataValidationError
        If ``n`` is smaller than 1 or ``associations`` does not carry
        the association-table columns.
    """
    if n < 1:
        raise DataValidationError(f"n must be >= 1, got {n}")
    require_columns(associations, _ASSOCIATION_COLUMNS)

    subset = associations
    if only_significant:
        subset = associations[associations["significant"]]
        if len(subset) < n:
            _LOG.warning(
                "only %d significant association(s) -- returning the top %d "
                "of all %d association(s) ranked by |rho|",
                len(subset),
                min(n, len(associations)),
                len(associations),
            )
            subset = associations
    return subset.head(n)


def direction_of_association(rho: float, lower_is_sensitive: bool = True) -> str:
    """Translate a correlation sign into a sensitivity sentence.

    Parameters
    ----------
    rho:
        Spearman/Pearson correlation between a gene's expression and a
        drug's response metric (NaN counts as "no association").
    lower_is_sensitive:
        Direction of the response metric: ``True`` (default) for
        LN_IC50-style metrics where *lower* means more sensitive;
        ``False`` for pIC50-style metrics where *higher* means more
        sensitive.

    Returns
    -------
    str
        ``"expression high -> more sensitive"``,
        ``"expression high -> more resistant"`` or
        ``"no association"``.  With the default LN_IC50 direction a
        *negative* rho means high expression tracks low LN_IC50, i.e.
        sensitivity.
    """
    if rho is None:
        return "no association"
    try:
        value = float(rho)
    except (TypeError, ValueError):
        return "no association"
    if math.isnan(value) or value == 0.0:
        return "no association"
    sensitive_when_expression_high = (value < 0) if lower_is_sensitive else (value > 0)
    if sensitive_when_expression_high:
        return "expression high -> more sensitive"
    return "expression high -> more resistant"
