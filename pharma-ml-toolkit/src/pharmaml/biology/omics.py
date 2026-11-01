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
  Pearson optional), NaN-safe and constant-gene-safe, and
* :func:`single_sample_enrichment` / :func:`enrichment_rank_report`
  score gene sets per sample -- ssGSEA (Barbie et al. 2009) and GSVA
  (Hänzelmann et al. 2013) in the toolkit's own NumPy, no R and no
  enrichment package behind them.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from pharmaml.biology.omics import (
...     correlate_with_response, impute_expression,
...     most_variable_genes, normalize_expression,
... )

>>> frame = pd.DataFrame({"G1": [1.0, 3.0], "G2": [7.0, 7.0]}, index=["cell_a", "cell_b"])
>>> normalised = normalize_expression(frame)
>>> normalised["G1"].round(3).tolist()   # log2(x+1) then z: 1, 2 -> -1, +1
[-1.0, 1.0]
>>> normalised["G2"].tolist()            # constant gene -> z of 0, not NaN
[0.0, 0.0]

>>> impute_expression(pd.DataFrame({"G1": [1.0, np.nan, 3.0]}))["G1"].tolist()
[1.0, 2.0, 3.0]
>>> impute_expression(frame, strategy="mode")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: strategy must be 'median' or 'mean', got 'mode'

>>> variability = pd.DataFrame(
...     {"ACTB": [1.0, 1.0, 1.0], "BRAF": [1.0, 5.0, 9.0], "CDK4": [0.0, 2.0, 4.0]}
... )
>>> most_variable_genes(variability, n=2)   # MAD ranking: BRAF 4, CDK4 2, ACTB 0
['BRAF', 'CDK4']
>>> most_variable_genes(variability, n=0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: n must be >= 1, got 0

>>> expression = pd.DataFrame(
...     {"G1": [1.0, 2.0, 3.0], "G2": [3.0, 2.0, 1.0], "G3": [1.0, np.nan, 1.0]},
...     index=["cell_a", "cell_b", "cell_c"],
... )
>>> response = pd.Series([1.0, 2.0, 3.0], index=["cell_a", "cell_b", "cell_c"])
>>> corr = correlate_with_response(expression, response)
>>> corr.set_index("gene")["rho"].round(3).to_dict()
{'G1': 1.0, 'G2': -1.0, 'G3': nan}

>>> enrichment_frame = pd.DataFrame(
...     {"G1": [3.0, 1.0], "G2": [2.0, 2.0], "G3": [1.0, 3.0], "G4": [0.0, 4.0]},
...     index=["cell_a", "cell_b"],
... )
>>> gene_sets = {"SET_A": {"G1", "G2"}, "SET_B": {"G3"}}
>>> single_sample_enrichment(enrichment_frame, gene_sets).round(3).to_dict("index")
{'cell_a': {'SET_A': 2.025, 'SET_B': -0.667}, 'cell_b': {'SET_A': -1.957, 'SET_B': 0.667}}
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
    "enrichment_rank_report",
    "impute_expression",
    "most_variable_genes",
    "normalize_expression",
    "single_sample_enrichment",
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
    and the count (round-11/16 semantics: warn, keep NaN).  Duplicated
    gene symbols (duplicate DataFrame columns) raise
    :class:`DataValidationError` naming the symbols -- pandas operations
    on duplicate columns would otherwise crash with a raw ``TypeError``
    deep inside the numeric pipeline.  Every public omics entry point
    funnels through here, so the check covers them all.
    """
    duplicated = frame.columns[frame.columns.duplicated()]
    if not duplicated.empty:
        symbols = sorted({str(symbol) for symbol in duplicated})
        raise DataValidationError(
            f"duplicated gene symbol(s) in the expression columns: {symbols} -- "
            "each gene must appear exactly once; aggregate or rename the "
            "duplicates before calling the omics functions"
        )
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
        counts the offending cells.  Duplicated gene symbols (duplicate
        DataFrame columns) likewise raise, naming the symbols.
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
        If ``strategy`` is neither ``"median"`` nor ``"mean"``, or the
        frame holds duplicated gene symbols (duplicate DataFrame
        columns).
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
        If ``n`` is smaller than 1, or the frame holds duplicated gene
        symbols (duplicate DataFrame columns).
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
        (the pairing would be ambiguous).  Duplicated gene symbols in
        ``expression_frame`` (duplicate DataFrame columns) likewise
        raise, naming the symbols.
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


# ---------------------------------------------------------------------------
# Single-sample enrichment (ssGSEA / GSVA, the toolkit's own NumPy)
# ---------------------------------------------------------------------------

#: Enrichment methods understood by :func:`single_sample_enrichment`.
_ENRICHMENT_METHODS = ("ssgsea", "gsva")

#: Default weight exponent of the ssGSEA statistic (Barbie et al. 2009).
_SSGSEA_ALPHA = 0.25

#: Abramowitz & Stegun 7.1.26 rational-approximation coefficients for erf
#: (max absolute error 1.5e-7 -- the Gaussian kernel stays pure NumPy).
_ERF_P = 0.3275911
_ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Vectorised erf via the A&S 7.1.26 rational approximation."""
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + _ERF_P * ax)
    a1, a2, a3, a4, a5 = _ERF_A
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    return sign * (1.0 - poly * np.exp(-ax * ax))


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """The standard normal CDF, vectorised, NumPy-only."""
    return 0.5 * (1.0 + _erf_approx(x / np.sqrt(2.0)))


def _ssgsea_score(x: np.ndarray, in_set: np.ndarray, alpha: float) -> float:
    """One sample's ssGSEA enrichment for one gene set (Barbie et al. 2009).

    Genes are ranked by expression (descending, stable ties); the
    statistic is the sum over all rank positions of the *weighted*
    empirical CDF of the set's genes (weights ``|x|^alpha``, normalised
    by the set's total weight) minus the unweighted CDF of the rest.
    """
    n = x.size
    order = np.argsort(-x, kind="stable")
    weights = np.abs(x) ** alpha
    set_sorted = in_set[order]
    set_weights = np.where(set_sorted, weights[order], 0.0)
    total_set = float(set_weights.sum())
    cdf_set = np.cumsum(set_weights) / total_set
    cdf_rest = np.cumsum(~set_sorted) / float(n - int(set_sorted.sum()))
    return float(np.sum(cdf_set - cdf_rest))


def _gsva_score(x: np.ndarray, in_set: np.ndarray) -> float:
    """One sample's GSVA enrichment for one gene set (Hänzelmann 2013).

    Genes are ranked by expression (descending, stable ties) and both
    the set's and the background's CDFs are estimated over the rank
    positions with a Gaussian kernel whose bandwidth follows Silverman's
    rule of thumb on the rank spacing (``1.06·sd·N^(-1/5)``); the score
    is the signed Kolmogorov-Smirnov deviation (``max(F_set - F_rest)``
    minus ``max(F_rest - F_set)``) at the gene positions, normalised by
    the two-sample KS null standard deviation ``sqrt(N / (n_set *
    n_rest))`` so scores stay comparable across set sizes.
    """
    n = x.size
    order = np.argsort(-x, kind="stable")
    positions = np.arange(1, n + 1, dtype=float)
    set_sorted = in_set[order]
    n_set = int(set_sorted.sum())
    n_rest = n - n_set
    bandwidth = 1.06 * float(np.std(positions)) * n ** (-0.2)
    kernel_set = _normal_cdf((positions[:, None] - positions[set_sorted][None, :]) / bandwidth)
    kernel_rest = _normal_cdf((positions[:, None] - positions[~set_sorted][None, :]) / bandwidth)
    difference = kernel_set.mean(axis=1) - kernel_rest.mean(axis=1)
    d_plus = float(difference.max())
    d_minus = float((-difference).max())
    ks_std = float(np.sqrt(n / (n_set * n_rest)))
    return (d_plus - d_minus) / ks_std


def _resolve_gene_sets(
    gene_sets: dict[str, set[str] | list[str] | tuple[str, ...]], columns: pd.Index
) -> dict[str, np.ndarray]:
    """Map every gene set onto a boolean column-membership vector."""
    known = {str(gene) for gene in columns}
    resolved: dict[str, np.ndarray] = {}
    unmapped = 0
    empty_sets: list[str] = []
    for name, members in gene_sets.items():
        member_set = {str(gene) for gene in members}
        missing = member_set - known
        unmapped += len(missing)
        membership = np.array([str(gene) in member_set for gene in columns], dtype=bool)
        resolved[str(name)] = membership
        if not membership.any():
            empty_sets.append(str(name))
    if unmapped:
        _LOG.warning(
            "%d gene(s) named in the gene sets are not columns of the frame -- "
            "skipped (the sets score on their profiled members only)",
            unmapped,
        )
    if empty_sets:
        _LOG.warning(
            "%d gene set(s) hold no gene of the frame at all -- scored NaN: %s",
            len(empty_sets),
            ", ".join(sorted(empty_sets)[:5]),
        )
    return resolved


def single_sample_enrichment(
    frame: pd.DataFrame,
    gene_sets: dict[str, set[str] | list[str] | tuple[str, ...]],
    method: str = "ssgsea",
    alpha: float = _SSGSEA_ALPHA,
) -> pd.DataFrame:
    """Per-sample gene-set enrichment: ssGSEA or GSVA, pure NumPy.

    Rows are cell lines (samples), columns are genes -- the omics
    convention of this module.  For every sample the genes are ranked by
    expression (descending; ties keep the frame's column order, so the
    result is deterministic), and each gene set is scored by either

    ``method="ssgsea"``
        Barbie et al. (2009): the sum over all rank positions of the
        weighted CDF of the set's genes (weights ``|x|^alpha``) minus
        the CDF of the rest -- positive when the set's genes sit among
        the sample's most-expressed genes.  This is the raw integral
        statistic of the paper's supplement; the per-sample score
        re-normalisation some implementations apply afterwards is
        deliberately not applied (the units stay honest).

    ``method="gsva"``
        Hänzelmann et al. (2013): Gaussian-kernel smoothed CDFs over
        the sorted expression ranks (Silverman-rule bandwidth), and the
        signed Kolmogorov-Smirnov-type maximum deviation between the
        set's and the background's smoothed distributions, normalised
        by the two-sample KS null standard deviation.

    Both scores run gene by gene in the toolkit's own NumPy -- no R, no
    GSVA package, no randomness.  The cost note: GSVA is ``O(N²)`` per
    (sample, set) in the number of genes; at screening scale subsample
    the frame (e.g. :func:`most_variable_genes`) first.

    Parameters
    ----------
    frame:
        Expression table (cell lines x genes).  Non-numeric columns are
        coerced to NaN with a warning naming the column and count.
    gene_sets:
        Named gene sets (any iterable of gene symbols per name; the
        caller's key order becomes the output's column order).  Genes
        missing from the frame are skipped with a warning.
    method:
        ``"ssgsea"`` (default) or ``"gsva"``.
    alpha:
        The ssGSEA weight exponent (default 0.25, the paper's).  Must
        be positive; ignored by ``"gsva"``.

    Returns
    -------
    pandas.DataFrame
        Samples x gene sets.  A set that covers every finite gene of a
        sample (no background left) or whose members all carry NaN
        scores NaN for that sample, with one warning per call.

    Raises
    ------
    DataValidationError
        If ``method`` is unknown, ``alpha`` is not positive, the frame
        has no genes, or the frame holds duplicated gene symbols
        (duplicate DataFrame columns).

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.biology.omics import single_sample_enrichment
    >>> frame = pd.DataFrame(
    ...     {"G1": [3.0, 1.0], "G2": [2.0, 2.0], "G3": [1.0, 3.0], "G4": [0.0, 4.0]},
    ...     index=["s1", "s2"],
    ... )
    >>> gene_sets = {"SET_A": {"G1", "G2"}, "SET_B": {"G3"}, "SET_AB": {"G1", "G2", "G3"}}
    >>> scores = single_sample_enrichment(frame, gene_sets)
    >>> scores.round(3).to_dict("index")
    {'s1': {'SET_A': 2.025, 'SET_B': -0.667, 'SET_AB': 2.09}, 's2': {'SET_A': -1.957, 'SET_B': 0.667, 'SET_AB': -1.91}}
    >>> gsva = single_sample_enrichment(frame, gene_sets, method="gsva")
    >>> gsva.round(3).to_dict("index")
    {'s1': {'SET_A': 0.92, 'SET_B': -0.25, 'SET_AB': 0.751}, 's2': {'SET_A': -0.92, 'SET_B': 0.25, 'SET_AB': -0.751}}
    >>> single_sample_enrichment(frame, gene_sets, method="mean")
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: method must be one of ['gsva', 'ssgsea'], got 'mean'
    """
    if method not in _ENRICHMENT_METHODS:
        raise DataValidationError(
            f"method must be one of {sorted(_ENRICHMENT_METHODS)}, got {method!r}"
        )
    if alpha <= 0:
        raise DataValidationError(f"alpha must be > 0, got {alpha}")
    values = _numeric_expression(frame)
    if values.shape[1] < 1:
        raise DataValidationError("the frame holds no gene columns to enrich against")
    resolved = _resolve_gene_sets(gene_sets, values.columns)

    n_nan_cells = int(values.isna().any(axis=1).sum())
    if n_nan_cells:
        _LOG.warning(
            "%d sample row(s) hold NaN expression value(s) -- those genes sit out "
            "of the ranking (and out of every set) for that sample",
            n_nan_cells,
        )
    scores: dict[str, dict[str, float]] = {}
    degenerate = 0
    for sample in values.index:
        row = values.loc[sample]
        finite = np.isfinite(row.to_numpy(dtype=float))
        x = row.to_numpy(dtype=float)[finite]
        sample_scores: dict[str, float] = {}
        for name, membership in resolved.items():
            in_set = membership[finite]
            n_set = int(in_set.sum())
            if x.size < 2 or n_set < 1 or n_set >= x.size:
                sample_scores[name] = float("nan")
                if n_set >= x.size and x.size >= 2:
                    degenerate += 1
                continue
            if method == "ssgsea":
                sample_scores[name] = _ssgsea_score(x, in_set, alpha)
            else:
                sample_scores[name] = _gsva_score(x, in_set)
        scores[str(sample)] = sample_scores
    if degenerate:
        _LOG.warning(
            "%d (sample, gene set) pair(s) cover every finite gene of their sample "
            "-- no background left, scored NaN",
            degenerate,
        )
    result = pd.DataFrame.from_dict(scores, orient="index")
    result = result.reindex(columns=[str(name) for name in gene_sets])
    _LOG.info(
        "Enriched %d gene set(s) across %d sample(s) (%s)",
        result.shape[1],
        result.shape[0],
        method,
    )
    return result


def enrichment_rank_report(
    frame: pd.DataFrame,
    gene_sets: dict[str, set[str] | list[str] | tuple[str, ...]],
    method: str = "ssgsea",
    alpha: float = _SSGSEA_ALPHA,
    top: int = 3,
) -> pd.DataFrame:
    """Each sample's top enriched gene sets, with their scores.

    Wraps :func:`single_sample_enrichment` and trims it to the
    leaderboard: the ``top`` highest-scoring gene sets per sample
    (ties broken by gene-set name ascending -- deterministic), with the
    per-sample rank attached.

    Parameters
    ----------
    frame:
        Expression table (cell lines x genes).
    gene_sets:
        Named gene sets, as for :func:`single_sample_enrichment`.
    method:
        ``"ssgsea"`` (default) or ``"gsva"``.
    alpha:
        The ssGSEA weight exponent (ignored by ``"gsva"``).
    top:
        How many leading sets to report per sample (>= 1).

    Returns
    -------
    pandas.DataFrame
        Columns ``sample``, ``gene_set``, ``score`` and ``rank``; only
        samples with at least one finite score appear.  Asking for more
        sets than were scored returns however many exist.

    Raises
    ------
    DataValidationError
        On the same conditions as :func:`single_sample_enrichment`, or
        ``top < 1``.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.biology.omics import enrichment_rank_report
    >>> frame = pd.DataFrame(
    ...     {"G1": [3.0, 1.0], "G2": [2.0, 2.0], "G3": [1.0, 3.0], "G4": [0.0, 4.0]},
    ...     index=["s1", "s2"],
    ... )
    >>> gene_sets = {"SET_A": {"G1", "G2"}, "SET_B": {"G3"}, "SET_AB": {"G1", "G2", "G3"}}
    >>> report = enrichment_rank_report(frame, gene_sets, top=2)
    >>> report[report["sample"] == "s1"][["gene_set", "score"]].round(3).to_dict("records")
    [{'gene_set': 'SET_AB', 'score': 2.09}, {'gene_set': 'SET_A', 'score': 2.025}]
    >>> report["gene_set"].tolist()
    ['SET_AB', 'SET_A', 'SET_B', 'SET_AB']
    """
    if top < 1:
        raise DataValidationError(f"top must be >= 1, got {top}")
    scores = single_sample_enrichment(frame, gene_sets, method=method, alpha=alpha)
    rows: list[dict[str, object]] = []
    for sample in scores.index:
        series = scores.loc[sample].dropna()
        if series.empty:
            continue
        ranked = (
            pd.DataFrame(
                {"gene_set": series.index.astype(str), "score": series.to_numpy(dtype=float)}
            )
            .sort_values(["score", "gene_set"], ascending=[False, True])
            .head(top)
        )
        for position, (gene_set, score) in enumerate(
            zip(ranked["gene_set"], ranked["score"], strict=True)
        ):
            rows.append(
                {
                    "sample": str(sample),
                    "gene_set": str(gene_set),
                    "score": float(score),
                    "rank": position + 1,
                }
            )
    report = pd.DataFrame(rows, columns=["sample", "gene_set", "score", "rank"])
    _LOG.info(
        "Ranked the top %d gene set(s) for %d sample(s) (%s)",
        top,
        int(report["sample"].nunique()) if not report.empty else 0,
        method,
    )
    return report
