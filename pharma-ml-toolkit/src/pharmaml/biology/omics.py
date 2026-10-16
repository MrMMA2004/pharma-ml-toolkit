"""Gene-expression feature tables for cell-line panels.

The convention throughout this subpackage is **rows = cell lines,
columns = genes** -- the same orientation as the QSAR ``X`` matrices of
:mod:`pharmaml.ml`, so an expression frame can feed the ML pipeline
directly after :func:`normalize_expression` and
:func:`impute_expression`.

* :func:`normalize_expression` applies the standard ``log2(x + 1)``
  transform followed by a per-gene (column) z-score,
* :func:`impute_expression` fills missing values per gene with the
  gene's median or mean,
* :func:`most_variable_genes` ranks genes by MAD (median absolute
  deviation -- robust against outlier transcripts) for feature
  selection, and
* :func:`correlate_with_response` correlates every gene against one
  drug's response vector across the panel (Spearman by default,
  Pearson optional), NaN-safe and constant-gene-safe.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ConstantInputWarning

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "correlate_with_response",
    "impute_expression",
    "most_variable_genes",
    "normalize_expression",
]

#: Correlation methods understood by :func:`correlate_with_response`.
_CORRELATION_METHODS = {
    "spearman": stats.spearmanr,
    "pearson": stats.pearsonr,
}

#: Minimum number of jointly observed (cell line, value) pairs for a
#: correlation to be defined at all.
_MIN_CORRELATION_POINTS = 3


def _numeric_expression(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce every expression column to numeric dtype, warning per column.

    Non-numeric values become NaN after a warning that names the column
    and the count (round-11/16 semantics: warn, keep NaN).
    """
    coerced = frame.copy()
    for column in coerced.columns:
        series = coerced[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        lost = int((series.notna() & numeric.isna()).sum())
        if lost:
            _LOG.warning(
                "%d non-numeric value(s) in expression column %r coerced to NaN",
                lost,
                column,
            )
        coerced[column] = numeric
    return coerced


def normalize_expression(
    frame: pd.DataFrame,
    log_transform: bool = True,
    z_score: bool = True,
) -> pd.DataFrame:
    """Normalise an expression table: ``log2(x + 1)`` then per-gene z-score.

    Rows are cell lines, columns are genes (the ``X`` convention).  The
    log transform uses the ``+1`` offset so zero counts map to zero,
    and the z-score is computed **per gene** (column) over the non-NaN
    values with population standard deviation (ddof=0, matching
    ``scipy.stats.zscore``).

    Parameters
    ----------
    frame:
        Expression table (cell lines x genes).  Non-numeric columns are
        coerced to NaN with a warning naming the column and count.
    log_transform:
        Apply ``log2(x + 1)`` (default).  Negative values make this
        undefined and raise instead of producing complex numbers.
    z_score:
        Centre and scale each gene column (default).

    Returns
    -------
    pandas.DataFrame
        A new frame; the input is never mutated.  NaN entries pass
        through untouched, and a constant gene gets ``z = 0`` for all
        of its non-NaN entries (with a warning naming up to five such
        genes).

    Raises
    ------
    DataValidationError
        If the frame holds negative values while ``log_transform`` is
        enabled -- ``log2(x + 1)`` is undefined there.  The message
        counts the offending cells.
    """
    values = _numeric_expression(frame)

    if log_transform:
        negative = int((values < 0).sum().sum())
        if negative:
            raise DataValidationError(
                f"{negative} negative expression value(s) found -- "
                "log2(x + 1) is undefined for negatives; clip or drop them "
                "first (or pass log_transform=False)"
            )
        values = np.log2(values + 1.0)

    if not z_score:
        _LOG.info(
            "Normalised %d cell line(s) x %d gene(s)%s",
            values.shape[0],
            values.shape[1],
            " (log2 only)" if log_transform else "",
        )
        return values

    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0)
    constant = [str(gene) for gene in values.columns[stds == 0]]
    if constant:
        shown = ", ".join(constant[:5])
        extra = len(constant) - 5
        more = f" (+{extra} more)" if extra > 0 else ""
        _LOG.warning(
            "%d constant gene(s) after normalisation -- z-score set to 0: "
            "%s%s",
            len(constant),
            shown,
            more,
        )
    safe_stds = stds.where(stds != 0, 1.0)
    result = (values - means) / safe_stds
    result = result.where(values.notna(), np.nan)
    _LOG.info(
        "Normalised %d cell line(s) x %d gene(s) (log2=%s, z-score=True)",
        result.shape[0],
        result.shape[1],
        log_transform,
    )
    return result


def impute_expression(
    frame: pd.DataFrame, strategy: str = "median"
) -> pd.DataFrame:
    """Fill missing expression values with a per-gene statistic.

    Parameters
    ----------
    frame:
        Expression table (cell lines x genes).  Non-numeric columns are
        coerced to NaN with a warning naming the column and count.
    strategy:
        ``"median"`` (default) or ``"mean"`` -- computed per gene over
        the observed values.

    Returns
    -------
    pandas.DataFrame
        A new frame; the input is never mutated.  Genes that are
        entirely NaN stay NaN (there is nothing to impute from) and a
        warning names the count.

    Raises
    ------
    DataValidationError
        If ``strategy`` is neither ``"median"`` nor ``"mean"``.
    """
    if strategy not in ("median", "mean"):
        raise DataValidationError(
            f"strategy must be 'median' or 'mean', got {strategy!r}"
        )
    values = _numeric_expression(frame)

    all_missing = [str(gene) for gene in values.columns[values.isna().all()]]
    if all_missing:
        _LOG.warning(
            "%d gene(s) have no observed values at all -- left as NaN: %s",
            len(all_missing),
            ", ".join(all_missing[:5]) + (" ..." if len(all_missing) > 5 else ""),
        )

    filler = values.median(axis=0) if strategy == "median" else values.mean(axis=0)
    result = values.fillna(filler)
    n_filled = int(values.isna().sum().sum() - result.isna().sum().sum())
    _LOG.info(
        "Imputed %d missing value(s) with per-gene %s (%d gene(s) left all-NaN)",
        n_filled,
        strategy,
        len(all_missing),
    )
    return result


def most_variable_genes(frame: pd.DataFrame, n: int = 500) -> list[str]:
    """Return the ``n`` most variable genes by MAD.

    Variability is the median absolute deviation (MAD) of each gene's
    non-NaN values -- the robust workhorse for RNA-seq feature
    selection.  The result is deterministic: genes are sorted by MAD
    descending, ties broken by gene name ascending, so equal-MAD genes
    always come back in the same alphabetical order.

    Parameters
    ----------
    frame:
        Expression table (cell lines x genes).
    n:
        Number of genes to return.  Asking for more genes than the
        table holds returns all of them (with a warning) instead of
        failing.

    Returns
    -------
    list of str
        The top-``n`` gene names.

    Raises
    ------
    DataValidationError
        If ``n`` is smaller than 1.
    """
    if n < 1:
        raise DataValidationError(f"n must be >= 1, got {n}")
    values = _numeric_expression(frame)
    if n > values.shape[1]:
        _LOG.warning(
            "requested %d gene(s) but the table holds only %d -- returning "
            "all of them",
            n,
            values.shape[1],
        )
        n = values.shape[1]

    medians = values.median(axis=0)
    deviations = (values - medians).abs()
    mads = deviations.median(axis=0)

    ranked = pd.DataFrame(
        {"mad": mads, "gene": mads.index.astype(str)}
    ).sort_values(
        ["mad", "gene"], ascending=[False, True], na_position="last"
    )
    top = ranked["gene"].head(n).tolist()
    _LOG.info("Selected %d most variable gene(s) by MAD", len(top))
    return top


def _align_response(
    expression_frame: pd.DataFrame, response_series: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Align both inputs onto their shared cell-line labels.

    Returns the expression frame and the response subset to the same
    rows (identical label order).  Exact index matches (without NaN
    labels) pass straight through; otherwise both inputs are subset to
    the shared labels, with a warning whenever rows are dropped on
    either side.  Labels that are NaN never match (a row without an
    identifier cannot be paired) and count as dropped.  Duplicate
    labels on either side raise :class:`DataValidationError` -- the
    gene/response pairing would be ambiguous.  No overlap at all
    likewise raises.
    """
    for kind, index in (
        ("expression", expression_frame.index),
        ("response", response_series.index),
    ):
        if index.duplicated().any():
            labels = sorted({str(label) for label in index[index.duplicated()]})
            raise DataValidationError(
                f"{kind} index has duplicate cell-line label(s) {labels} -- "
                "the correlation pairing would be ambiguous; aggregate or "
                "deduplicate those rows first"
            )
    if (
        response_series.index.equals(expression_frame.index)
        and not expression_frame.index.hasnans
    ):
        return expression_frame, response_series
    # labels that are NaN cannot be paired -- drop them before matching
    frame_labels = expression_frame.index.dropna()
    response_labels = response_series.index.dropna()
    shared = frame_labels.intersection(response_labels)
    if len(shared) == 0:
        raise DataValidationError(
            "no overlap between expression rows and response values -- "
            f"expression index holds {list(expression_frame.index[:3])!r} ... "
            f"while the response holds {list(response_series.index[:3])!r} ..."
        )
    if len(shared) < len(expression_frame) or len(shared) < len(response_series):
        _LOG.warning(
            "response alignment uses %d shared cell line(s) (%d expression "
            "row(s) and %d response value(s) unmatched)",
            len(shared),
            len(expression_frame) - len(shared),
            len(response_series) - len(shared),
        )
    return expression_frame.loc[shared], response_series.loc[shared]


def correlate_with_response(
    expression_frame: pd.DataFrame,
    response_series,
    method: str = "spearman",
) -> pd.DataFrame:
    """Correlate every gene's expression with one drug's response.

    Parameters
    ----------
    expression_frame:
        Expression table (cell lines x genes).
    response_series:
        One drug's response values indexed by cell line (e.g. a column
        of :func:`pharmaml.biology.response_matrix`).  Aligned with the
        expression rows by index: partial overlap warns about the
        dropped rows and correlates on the shared cell lines, labels
        that are NaN never match, and no overlap raises.
    method:
        ``"spearman"`` (default) or ``"pearson"``.

    Returns
    -------
    pandas.DataFrame
        One row per gene with columns ``gene``, ``rho`` and
        ``p_value``.  Genes with fewer than three jointly observed
        (non-NaN) values keep their row with ``rho``/``p_value`` NaN --
        a single warning names how many genes were affected; constant
        genes likewise stay NaN (scipy's correlation is undefined
        there).

    Raises
    ------
    DataValidationError
        If ``method`` is unknown, no cell line is shared between the
        two inputs, or either index holds duplicate cell-line labels
        (the pairing would be ambiguous).
    """
    if method not in _CORRELATION_METHODS:
        raise DataValidationError(
            f"method must be one of {sorted(_CORRELATION_METHODS)}, got "
            f"{method!r}"
        )
    correlate = _CORRELATION_METHODS[method]

    values = _numeric_expression(expression_frame)
    response = (
        response_series
        if isinstance(response_series, pd.Series)
        else pd.Series(response_series)
    )
    response = pd.to_numeric(response, errors="coerce")
    values, response = _align_response(values, response)
    y = response.to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    short_genes = 0
    for gene in values.columns:
        x = values[gene].to_numpy(dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if int(mask.sum()) < _MIN_CORRELATION_POINTS:
            rows.append({"gene": str(gene), "rho": np.nan, "p_value": np.nan})
            short_genes += 1
            continue
        with warnings.catch_warnings():
            # scipy warns (ConstantInputWarning) for the degenerate inputs
            # we already handle with our own NaN policy and log records.
            warnings.simplefilter("ignore", ConstantInputWarning)
            result = correlate(x[mask], y[mask])
        rho = float(result.statistic)
        p_value = float(result.pvalue)
        if not np.isfinite(rho):
            rho = np.nan
        if not np.isfinite(p_value):
            p_value = np.nan
        rows.append({"gene": str(gene), "rho": rho, "p_value": p_value})

    if short_genes:
        _LOG.warning(
            "%d gene(s) had fewer than %d joint non-NaN value(s) with the "
            "response -- rho/p_value set to NaN",
            short_genes,
            _MIN_CORRELATION_POINTS,
        )
    result_frame = pd.DataFrame(rows, columns=["gene", "rho", "p_value"])
    _LOG.info(
        "Correlated %d gene(s) with the response vector (%s)",
        len(result_frame),
        method,
    )
    return result_frame
