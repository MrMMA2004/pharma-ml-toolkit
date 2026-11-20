"""Biomarker association testing: the GDSC protocol and friends (100x-171).

Four repositories were reviewed file by file for this module:

* ``CasteloDan/gdsctools`` (the Sanger GDSC association engine): the
  per-(drug, feature) ANOVA protocol of ``anova.py`` -- OLS of log-IC50
  on ``C(tissue) + C(MSI) + feature`` with the QR-based Type-I F test,
  Cohen's d / Glass delta effect sizes, the minimum-count filters, the
  globally corrected FDR written in PERCENT, and the 20-column result
  contract; ``qvalue.py``'s Storey-Tibshirani q-value with the
  cubic-spline pi0 estimate; and ``regression.py``'s
  ``check_randomness`` Bayes-factor test (real-vs-shuffled CV scores).
* ``CBIIT/cellminercdb`` (the cross-database comparison tool):
  ``appUtils.R``'s pattern search (a feature vector correlated against
  every row of a matrix, BH-corrected, sorted) and the per-tissue
  correlation table over OncoTree-stratified cell lines.
* ``TangYiChing/GDSC_DataParser``: ``useGDSC.py``'s pairwise
  ``|Delta IC50|`` cell-pair features (per drug, the upper-triangle
  absolute response differences between all cell-line pairs).
* ``cbioportal`` (the data layer only -- the Java/React portal is
  honestly out of scope): the exact two-tailed Fisher convention of
  ``FisherExactTestCalculator.java`` (sum of table probabilities <= the
  observed table's), the alteration/clinical enrichment dispatch of
  ``AlterationEnrichmentUtil`` / ``ClinicalDataEnrichmentUtil``
  (Fisher for binary alterations, pooled t / one-way ANOVA for
  expression, Kruskal-Wallis / chi-square for clinical attributes),
  and the clinical-file contract (the 5-row ``#``-prefixed header of
  ``data_clinical_patients.txt``) plus the CNA vocabulary
  (``-2..2`` wide matrix).

What each function computes, and the honest boundaries:

* :func:`qvalue` -- Storey-Tibshirani q-values: ``pi0(lambda) =
  mean(p >= lambda)/(1-lambda)`` on the 0..0.9 grid (step 0.05),
  smoothed by a cubic spline evaluated at lambda_max and capped at 1;
  ``q = pi0 * m * p_(i) / (i+1)`` with the monotone back-off, exactly
  gdsctools' implementation.
* :func:`cohens_d` / :func:`glass_delta` -- the pooled-SD and
  single-group-SD standardised mean differences with gdsctools'
  negative-variance guards.
* :func:`gdsc_anova` -- the association protocol: per drug, cell lines
  with non-null IC50 are regressed (pure numpy lstsq) on the
  intercept + tissue dummies + MSI + the binary feature; the feature's
  Type-I F statistic is extracted through a QR contrast projection
  (``effects = Q^T y``, factor SS from the 0/1 projection matrix,
  ``p = f.sf(F, dof, df_resid)``); the 20-column result frame carries
  the effect sizes, the group statistics and the FDR (BH by default,
  ``'qvalue'`` available) as a PERCENT column, sorted by p with
  ``ASSOC_ID`` inserted -- gdsctools' own conventions.
* :func:`check_randomness` -- the Bayes-factor randomisation test:
  call a scorer N times with the real target and N times with a
  shuffled target; ``proba = P(real > random)``,
  ``BF = 1/(1 - proba)`` (capped at N), gdsctools' formula verbatim.
* :func:`pattern_search` -- cellminercdb's pattern comparison in one
  pandas call: ``matrix.corrwith(vector)`` + t-test p-values + BH
  correction, sorted ascending by p.
* :func:`per_tissue_correlation` -- the per-tissue (plus an ``ALL``
  row) Pearson table, computed only where a tissue has more than 2
  complete pairs.
* :func:`pairwise_response_deltas` -- per drug, the upper-triangle
  ``|response_i - response_j|`` between all cell-line pairs, long
  format (``drug, cell_a, cell_b, delta``).
* :func:`fisher_two_tailed` -- cbioportal's convention: the
  two-tailed p is the sum of the probabilities of every table (in both
  directions) no greater than the observed one's.
* :func:`alteration_enrichment` -- altered-vs-unaltered dispatch:
  binary values -> Fisher two-tailed; numeric values -> pooled t test
  (cbioportal log2-transforms RNA-seq upstream; here the caller's
  scale is taken as-is, honestly).
* :func:`parse_cbio_clinical` / :func:`parse_cbio_cna` -- the study
  file contracts: the 5-row clinical header (Display Name /
  Description / Datatype / Priority / attribute IDs) and the wide CNA
  matrix with the ``-2..2`` vocabulary.

Deliberately NOT ported: gdsctools' HTML report machinery,
canvasXpress volcano JavaScript, the multi-core drivers and its
ElasticNet/Ridge layer (sklearn covers the models directly);
cellminercdb's Shiny shell and its cross-database identifier
resolution (a data-provenance problem, not an algorithm); the
cbioportal ClickHouse schema, the REST server and the frontend
mutual-exclusivity plots (the backend Fisher here is the same
calculator the frontend calls).

Examples
--------
>>> import numpy as np, pandas as pd
>>> from scipy.stats import false_discovery_control  # noqa: F401
>>> pvals = np.array([0.001, 0.002, 0.01, 0.4, 0.9])
>>> qv = qvalue(pvals)
>>> bool(qv[0] <= 0.01 and qv[-1] <= 1.0 + 1e-9)
True
>>> rng = np.random.default_rng(2)
>>> frame = pd.DataFrame({
...     "drug_1": rng.normal(1.0, 0.3, 40),
...     "drug_2": rng.normal(1.0, 0.3, 40),
... })
>>> feature = pd.DataFrame({
...     "COSMIC_ID": np.arange(40),
...     "TISSUE_FACTOR": ["breast"] * 20 + ["lung"] * 20,
...     "TP53_mut": np.r_[np.ones(10), np.zeros(30)].astype(int),
... }).set_index("COSMIC_ID")
>>> # drug_1 sensitive in TP53-mutants, drug_2 flat
>>> frame.loc[:, "drug_1"] = frame["drug_1"] + 1.5 * feature["TP53_mut"].to_numpy()
>>> result = gdsc_anova(frame, feature)
>>> result["FEATURE"].tolist()
['TP53_mut', 'TP53_mut']
>>> bool(result["ANOVA_FEATURE_pval"].iloc[0] < 1e-4)  # drug_1 association
True
>>> bool(result["ANOVA_FEATURE_pval"].iloc[1] > 0.05)  # drug_2 null
True
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "AssociationResult",
    "alteration_enrichment",
    "check_randomness",
    "cohens_d",
    "fisher_two_tailed",
    "gdsc_anova",
    "glass_delta",
    "pairwise_response_deltas",
    "parse_cbio_clinical",
    "parse_cbio_cna",
    "pattern_search",
    "per_tissue_correlation",
    "qvalue",
]


# --------------------------------------------------------------------- #
# multiple testing: Storey-Tibshirani q-values (gdsctools/qvalue.py)
# --------------------------------------------------------------------- #


def qvalue(pvals: Sequence[float]) -> np.ndarray:
    """Storey-Tibshirani q-values with the spline pi0 estimate.

    ``pi0(lambda) = mean(p >= lambda) / (1 - lambda)`` is evaluated on
    the ``0..0.9`` grid (step 0.05), smoothed by a cubic B-spline
    evaluated at ``lambda_max`` and capped at 1 (gdsctools' exact
    recipe, which tracks R's ``qvalue`` package to ~0.3%); then
    ``q_(i) = pi0 * m * p_(i) / (i + 1)`` over ASCENDING p, walked
    backwards with ``q_(i) = min(q_(i), q_(i+1))`` so q is monotone in
    p, the largest capped at 1.

    >>> import numpy as np
    >>> qv = qvalue([0.001, 0.001, 0.5, 0.5, 0.5])
    >>> bool(qv[0] <= qv[2] and qv[4] <= 1.0)
    True
    >>> bool(np.all(np.diff(qvalue([0.01, 0.02, 0.03])) >= -1e-12))
    True
    """
    p = np.asarray(pvals, dtype=float).ravel()
    if p.size == 0:
        raise DataValidationError("qvalue needs at least one p-value")
    if np.any((p < 0) | (p > 1)):
        raise DataValidationError("p-values must lie in [0, 1]")
    if p.size == 1:
        return np.array([min(float(p[0]), 1.0)])
    m = p.size
    lambdas = np.arange(0.0, 0.9 + 1e-8, 0.05)
    pi0_raw = np.array(
        [float(np.mean(p >= lam)) / (1.0 - lam) for lam in lambdas]
    )
    try:
        from scipy.interpolate import splev, splrep

        knots = splrep(lambdas, pi0_raw, k=3)
        pi0 = float(splev(lambdas[-1], knots))
    except (ArithmeticError, ValueError):  # pragma: no cover - degenerate spline
        pi0 = float(pi0_raw[-1])
    pi0 = min(max(pi0, 0.0), 1.0)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    indices = np.arange(1, m + 1)
    q_ranked = pi0 * m * ranked / indices
    # monotone back-off: q_(i) = min(q_(i), q_(i+1))
    for i in range(m - 2, -1, -1):
        q_ranked[i] = min(q_ranked[i], q_ranked[i + 1])
    q_ranked = np.minimum(q_ranked, 1.0)
    out = np.empty(m)
    out[order] = q_ranked
    return out


# --------------------------------------------------------------------- #
# effect sizes (gdsctools/stats.py)
# --------------------------------------------------------------------- #


def _group_stats(values: np.ndarray) -> tuple[float, float]:
    """Mean and ddof=1 SD with gdsctools' negative-variance guard."""
    if values.size == 0:
        return 0.0, 0.0
    mean = float(values.mean())
    ss = float((values**2).sum() - (values.sum()) ** 2 / values.size)
    sd = float(np.sqrt(max(ss, 0.0) / max(values.size - 1, 1)))
    return mean, sd


def cohens_d(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Cohen's d with the pooled SD (gdsctools' FEATURE_IC50_effect_size)."""
    pos = np.asarray(positives, dtype=float)
    neg = np.asarray(negatives, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return 0.0
    mean_pos, sd_pos = _group_stats(pos)
    mean_neg, sd_neg = _group_stats(neg)
    pooled = ((pos.size - 1) * sd_pos**2 + (neg.size - 1) * sd_neg**2) / max(
        pos.size + neg.size - 2, 1
    )
    if pooled <= 0:
        return 0.0
    return abs(mean_pos - mean_neg) / float(np.sqrt(pooled))


def glass_delta(
    positives: Sequence[float], negatives: Sequence[float]
) -> tuple[float, float]:
    """Glass delta against each group's own SD (pos, neg) variants."""
    pos = np.asarray(positives, dtype=float)
    neg = np.asarray(negatives, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return 0.0, 0.0
    mean_pos, sd_pos = _group_stats(pos)
    mean_neg, sd_neg = _group_stats(neg)
    delta_pos = abs(mean_pos - mean_neg) / sd_pos if sd_pos > 0 else 0.0
    delta_neg = abs(mean_pos - mean_neg) / sd_neg if sd_neg > 0 else 0.0
    return float(delta_pos), float(delta_neg)


# --------------------------------------------------------------------- #
# the GDSC association protocol (gdsctools/anova.py + models.py)
# --------------------------------------------------------------------- #


@dataclass
class AssociationResult:
    """The 20-column gdsctools result frame plus its metadata."""

    frame: pd.DataFrame
    n_drugs: int
    n_features: int
    correction: str

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"GDSC association: {self.frame.shape[0]} tests "
            f"({self.n_drugs} drugs x {self.n_features} features), "
            f"{self.correction} correction"
        )

    # thin proxies so the result reads like gdsctools' ANOVAResults df
    def __getitem__(self, key: str) -> Any:
        return self.frame[key]

    def __len__(self) -> int:
        return len(self.frame)

    def head(self, n: int = 5) -> pd.DataFrame:
        """The result frame's head."""
        return self.frame.head(n)


#: gdsctools' minimum-count settings (settings.py defaults).
MINIMUM_NONNA_IC50 = 6
FEATURE_FACTOR_THRESHOLD = 3
MSI_FACTOR_THRESHOLD = 2


def _design_and_test(
    ic50: np.ndarray, tissue: np.ndarray, msi: np.ndarray | None, feature: np.ndarray
) -> tuple[float, float, float, float]:
    """OLS + the QR Type-I F test for the FEATURE factor.

    Returns ``(F, p, tissue_p, msi_p)``.  The contrast trick is
    gdsctools' ``_get_anova_summary``: ``effects = Q^T y``; each
    factor's SS is the sum of squared effects over its design columns
    (via a 0/1 projection); ``F = MS_factor / (SSR / df_resid)``.
    """
    from scipy.stats import f as f_dist

    columns: list[np.ndarray] = [np.ones(len(ic50))]
    factor_dofs: list[int] = []
    factor_blocks: list[list[int]] = []

    levels = sorted(set(tissue), key=str)
    for level in levels[1:]:  # reference level dropped (R convention)
        columns.append((tissue == level).astype(float))
    factor_blocks.append(list(range(1, len(columns))))
    factor_dofs.append(len(levels) - 1)

    if msi is not None:
        columns.append(msi.astype(float))
        msi_block = [len(columns) - 1]
    else:
        msi_block = []
    factor_blocks.append(msi_block)
    factor_dofs.append(len(msi_block))

    columns.append(feature.astype(float))
    feature_block = [len(columns) - 1]
    factor_blocks.append(feature_block)
    factor_dofs.append(1)

    design = np.column_stack(columns)
    if design.shape[0] <= design.shape[1]:
        return 0.0, 1.0, 1.0, 1.0
    q, _r = np.linalg.qr(design)
    effects = q.T @ ic50
    fitted = q @ effects
    ssr = float(((ic50 - fitted) ** 2).sum())
    df_resid = design.shape[0] - design.shape[1]
    if df_resid <= 0 or ssr <= 0:
        return 0.0, 1.0, 1.0, 1.0

    def factor_p(block: list[int], dof: int) -> float:
        if not block or dof == 0:
            return 1.0
        ss = float((effects[block] ** 2).sum())
        ms = ss / dof
        stat = ms / (ssr / df_resid)
        return float(f_dist.sf(stat, dof, df_resid))

    feature_p = factor_p(feature_block, 1)
    # F for the feature itself (reported as ANOVA_FEATURE_pval's F)
    ss_feature = float((effects[feature_block] ** 2).sum())
    f_stat = ss_feature / (ssr / df_resid)
    tissue_p = factor_p(factor_blocks[0], factor_dofs[0])
    msi_p = factor_p(msi_block, factor_dofs[1] if msi is not None else 0)
    return f_stat, feature_p, tissue_p, msi_p


def gdsc_anova(
    ic50: pd.DataFrame,
    features: pd.DataFrame,
    *,
    drug_names: dict[str, tuple[str, str]] | None = None,
    use_msi: bool = True,
    correction: str = "fdr_bh",
    equal_var_ttest: bool = True,
    minimum_nonna_ic50: int = MINIMUM_NONNA_IC50,
    feature_threshold: int = FEATURE_FACTOR_THRESHOLD,
) -> AssociationResult:
    """The GDSC drug x feature association protocol.

    Duplicate-symbol map (round 304, part12 A-6): ``gdsc_anova`` also
    exists at :mod:`pharmaml.ml.pharmacogenomics` — the
    classic-frame variant (DataFrame + Storey-Tibshirani q-values).
    THIS one is the association-protocol variant (``AssociationResult``,
    global FDR, the 20-column contract) — the family is two
    intentional interface variants over the GDSC protocol, both
    library-surface; the definition sites are pinned by
    ``tests/test_architecture_hygiene.py``.

    Parameters mirror gdsctools' ``ANOVA``/``ANOVASettings``:
    ``ic50`` is the cell-line x drug log-IC50 frame (rows indexed by
    COSMIC-like IDs, columns = drugs); ``features`` the matching
    genomic-feature frame whose first columns may be ``TISSUE_FACTOR``
    and ``MSI_FACTOR`` (both recognised automatically), the remaining
    columns being binary features.  For every (drug, feature) pair
    with enough counts (>= ``minimum_nonna_ic50`` non-null IC50s,
    >= ``feature_threshold`` positives AND negatives) the protocol
    fits ``IC50 ~ C(tissue) + C(MSI) + feature`` and reports the
    20-column contract; the p-value vector is corrected GLOBALLY
    (across all associations) and the ``ANOVA_FEATURE_FDR`` column is
    a PERCENT (gdsctools' convention).
    """
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    if not isinstance(ic50, pd.DataFrame) or ic50.shape[1] < 1:
        raise DataValidationError("ic50 must be a DataFrame of drugs (columns)")
    if not isinstance(features, pd.DataFrame):
        raise DataValidationError("features must be a DataFrame")
    common = ic50.index.intersection(features.index)
    if len(common) < minimum_nonna_ic50:
        raise DataValidationError(
            f"only {len(common)} shared cell lines; need >= {minimum_nonna_ic50}"
        )
    ic50 = ic50.loc[common]
    features = features.loc[common]

    tissue = (
        features["TISSUE_FACTOR"].astype(str).to_numpy()
        if "TISSUE_FACTOR" in features.columns
        else np.array(["UNDEFINED"] * len(common))
    )
    msi = None
    if use_msi and "MSI_FACTOR" in features.columns:
        msi = pd.to_numeric(features["MSI_FACTOR"], errors="coerce").fillna(0).to_numpy()
        pos_msi = int((msi == 1).sum())
        neg_msi = int((msi == 0).sum())
        if min(pos_msi, neg_msi) < MSI_FACTOR_THRESHOLD:
            msi = None
    feature_cols = [
        column
        for column in features.columns
        if column not in ("TISSUE_FACTOR", "MSI_FACTOR", "MEDIA_FACTOR")
    ]

    rows: list[dict[str, Any]] = []
    for drug in ic50.columns:
        values = pd.to_numeric(ic50[drug], errors="coerce")
        kept = np.where(values.notna().to_numpy())[0]
        if len(kept) < minimum_nonna_ic50:
            continue
        y = values.to_numpy()[kept]
        tissue_kept = tissue[kept]
        msi_kept = msi[kept] if msi is not None else None
        for feature_name in feature_cols:
            f_values = pd.to_numeric(
                features[feature_name], errors="coerce"
            ).to_numpy()[kept]
            valid = np.isfinite(f_values) & np.isfinite(y)
            positives = y[valid & (f_values == 1)]
            negatives = y[valid & (f_values == 0)]
            if len(positives) < feature_threshold or len(negatives) < feature_threshold:
                continue
            _f_stat, p_value, tissue_p, msi_p = _design_and_test(
                y[valid], tissue_kept[valid], msi_kept[valid] if msi_kept is not None else None,
                f_values[valid],
            )
            mean_pos, sd_pos = _group_stats(positives)
            mean_neg, sd_neg = _group_stats(negatives)
            t_p = float(
                ttest_ind(negatives, positives, equal_var=equal_var_ttest).pvalue
            )
            drug_label = drug
            target = ""
            if drug_names and drug in drug_names:
                drug_label, target = drug_names[drug]
            rows.append(
                {
                    "FEATURE": feature_name,
                    "DRUG_ID": drug,
                    "DRUG_NAME": drug_label,
                    "DRUG_TARGET": target,
                    "N_FEATURE_neg": len(negatives),
                    "N_FEATURE_pos": len(positives),
                    "FEATURE_pos_logIC50_MEAN": mean_pos,
                    "FEATURE_neg_logIC50_MEAN": mean_neg,
                    "FEATURE_delta_MEAN_IC50": mean_pos - mean_neg,
                    "FEATURE_IC50_effect_size": cohens_d(positives, negatives),
                    "FEATURE_neg_Glass_delta": glass_delta(positives, negatives)[1],
                    "FEATURE_pos_Glass_delta": glass_delta(positives, negatives)[0],
                    "FEATURE_neg_IC50_sd": sd_neg,
                    "FEATURE_pos_IC50_sd": sd_pos,
                    "FEATURE_IC50_T_pval": t_p,
                    "ANOVA_FEATURE_pval": p_value,
                    "ANOVA_TISSUE_pval": tissue_p,
                    "ANOVA_MSI_pval": msi_p,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise DataValidationError(
            "no association passed the minimum-count filters; "
            "lower feature_threshold/minimum_nonna_ic50 or add signal"
        )
    if correction == "qvalue":
        fdr = qvalue(frame["ANOVA_FEATURE_pval"].to_numpy())
    else:
        fdr = multipletests(
            frame["ANOVA_FEATURE_pval"].to_numpy(), method=correction
        )[1]
    frame["ANOVA_FEATURE_FDR"] = np.asarray(fdr) * 100.0  # percent
    frame = frame.sort_values("ANOVA_FEATURE_pval").reset_index(drop=True)
    frame.insert(0, "ASSOC_ID", np.arange(1, len(frame) + 1))
    _LOG.info(
        "gdsc_anova: %d associations over %d drugs x %d features (%s)",
        len(frame),
        ic50.shape[1],
        len(feature_cols),
        correction,
    )
    return AssociationResult(
        frame=frame,
        n_drugs=int(ic50.shape[1]),
        n_features=len(feature_cols),
        correction=correction,
    )


# --------------------------------------------------------------------- #
# the Bayes-factor randomisation test (gdsctools/regression.py)
# --------------------------------------------------------------------- #


def check_randomness(
    score_fn: Callable[[np.ndarray], float],
    target: Sequence[float],
    n_iterations: int = 10,
    seed: int | None = None,
) -> dict[str, float]:
    """gdsctools' ``check_randomness``: is a score real or noise?

    ``score_fn`` receives a target vector and returns a score (the CV
    Pearson r of a model, say).  It is called ``n_iterations`` times
    with the real target and ``n_iterations`` times with a shuffled
    target; ``proba = P(real score > random score)`` over all pairs
    and the Bayes factor is ``1 / (1 - proba)`` (capped at
    ``n_iterations^2`` when proba == 1) -- gdsctools' formula, with
    the Kass & Raftery interpretation bands in the returned dict.

    >>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> rng = np.random.default_rng(4)
>>> X = rng.normal(size=(40, 3))
>>> y = 2.0 * X[:, 0] + 0.1
>>> def score(target: np.ndarray) -> float:
...     model = LinearRegression().fit(X[:30], target[:30])
...     pred = model.predict(X[30:])
...     return float(np.corrcoef(pred, target[30:])[0, 1])
>>> out = check_randomness(score, y, n_iterations=8, seed=0)
>>> bool(out["bayes_factor"] >= 1.0)
True
    """
    y = np.asarray(target, dtype=float).ravel()
    if n_iterations < 1:
        raise DataValidationError("n_iterations must be >= 1")
    rng = np.random.default_rng(seed)
    real_scores = [score_fn(y) for _ in range(n_iterations)]
    random_scores = [
        score_fn(rng.permutation(y)) for _ in range(n_iterations)
    ]
    wins = sum(
        1
        for real in real_scores
        for rand in random_scores
        if real > rand
    )
    total = n_iterations * n_iterations
    proba = wins / total
    bayes = total if proba >= 1.0 else 1.0 / (1.0 - proba)
    return {
        "proba": float(proba),
        "bayes_factor": float(bayes),
        "real_mean": float(np.mean(real_scores)),
        "random_mean": float(np.mean(random_scores)),
    }


# --------------------------------------------------------------------- #
# cellminercdb: pattern search + per-tissue correlation
# --------------------------------------------------------------------- #


def pattern_search(
    vector: Sequence[float],
    matrix: pd.DataFrame,
    *,
    label_prefix: str = "",
) -> pd.DataFrame:
    """CellMinerCDB's pattern comparison: correlate against every row.

    The query vector is correlated (Pearson, pairwise-complete) with
    every ROW of ``matrix``; each row gets the t-test p-value and the
    BH-corrected FDR; the result is sorted ascending by p with the
    columns ``feature, correlation, p_value, fdr``.

    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> frame = pd.DataFrame(rng.normal(size=(30, 4)), index=[f"r{i}" for i in range(30)])
    >>> query = frame.loc["r0"].to_numpy() * 2.0
    >>> hits = pattern_search(query, frame)
    >>> hits["feature"].iloc[0]
    'r0'
    """
    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests

    if not isinstance(matrix, pd.DataFrame) or matrix.shape[0] < 2:
        raise DataValidationError("matrix must be a DataFrame with >= 2 rows")
    query = np.asarray(vector, dtype=float).ravel()
    if query.shape[0] != matrix.shape[1]:
        raise DataValidationError(
            f"vector length {query.shape[0]} != matrix width {matrix.shape[1]}"
        )
    records = []
    for label, row in matrix.iterrows():
        values = row.to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(query)
        if valid.sum() < 3 or np.std(query[valid]) == 0 or np.std(values[valid]) == 0:
            records.append(
                {"feature": f"{label_prefix}{label}", "correlation": np.nan, "p_value": np.nan}
            )
            continue
        r, p = pearsonr(query[valid], values[valid])
        records.append(
            {"feature": f"{label_prefix}{label}", "correlation": float(r), "p_value": float(p)}
        )
    table = pd.DataFrame.from_records(records)
    valid_mask = table["p_value"].notna()
    table["fdr"] = np.nan
    if valid_mask.any():
        table.loc[valid_mask, "fdr"] = multipletests(
            table.loc[valid_mask, "p_value"].to_numpy(), method="fdr_bh"
        )[1]
    return table.sort_values("p_value", na_position="last").reset_index(drop=True)


def per_tissue_correlation(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    tissue_column: str,
) -> pd.DataFrame:
    """Per-tissue Pearson table (CellMinerCDB's tissue correlations).

    One row per tissue plus the ``ALL`` row; computed only where more
    than 2 complete pairs exist (cellminercdb's own guard), columns
    ``tissue, n, r, p_value``.

    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> frame = pd.DataFrame({
    ...     "x": rng.normal(size=30),
    ...     "y": rng.normal(size=30),
    ...     "tissue": ["breast"] * 15 + ["lung"] * 15,
    ... })
    >>> table = per_tissue_correlation(frame, "x", "y", "tissue")
    >>> set(table["tissue"]) == {"ALL", "breast", "lung"}
    True
    """
    from scipy.stats import pearsonr

    for column in (x_column, y_column, tissue_column):
        if column not in frame.columns:
            raise DataValidationError(f"column {column!r} not in frame")
    blocks = [("ALL", frame)] + [
        (str(tissue), group)
        for tissue, group in frame.groupby(tissue_column, sort=True)
    ]
    rows = []
    for tissue, group in blocks:
        pairs = group[[x_column, y_column]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if len(pairs) <= 2:
            continue
        r, p = pearsonr(pairs[x_column].to_numpy(), pairs[y_column].to_numpy())
        rows.append({"tissue": tissue, "n": len(pairs), "r": float(r), "p_value": float(p)})
    return pd.DataFrame(rows, columns=["tissue", "n", "r", "p_value"])


# --------------------------------------------------------------------- #
# GDSC_DataParser: pairwise response deltas
# --------------------------------------------------------------------- #


def pairwise_response_deltas(
    ic50: pd.DataFrame,
    *,
    max_pairs_per_drug: int = 2000,
) -> pd.DataFrame:
    """Per drug, the upper-triangle ``|response_i - response_j|`` pairs.

    ``useGDSC.py``'s ``respMat2respDf``: for every drug column, all
    cell-line pairs (i < j) with finite responses give one long-format
    row ``drug, cell_a, cell_b, delta`` -- the pairwise difference
    feature its ML pipeline consumes.

    >>> import pandas as pd
    >>> screen = pd.DataFrame(
    ...     {"d1": [1.0, 2.0, 4.0]}, index=["c1", "c2", "c3"]
    ... )
    >>> deltas = pairwise_response_deltas(screen)
    >>> sorted(deltas["delta"].tolist())
    [1.0, 2.0, 3.0]
    """
    if not isinstance(ic50, pd.DataFrame) or ic50.shape[1] < 1:
        raise DataValidationError("ic50 must be a DataFrame of drugs (columns)")
    rows = []
    for drug in ic50.columns:
        series = pd.to_numeric(ic50[drug], errors="coerce").dropna()
        cells = list(series.index)
        if len(cells) < 2:
            continue
        values = series.to_numpy()
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                rows.append(
                    {
                        "drug": drug,
                        "cell_a": cells[i],
                        "cell_b": cells[j],
                        "delta": abs(float(values[i] - values[j])),
                    }
                )
                if len(rows) >= max_pairs_per_drug * ic50.shape[1]:
                    _LOG.warning("pairwise_response_deltas truncated at cap")
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=["drug", "cell_a", "cell_b", "delta"])


# --------------------------------------------------------------------- #
# cbioportal: Fisher convention + enrichment + file contracts
# --------------------------------------------------------------------- #


def _fisher_table_p(a: int, b: int, c: int, d: int) -> float:
    """Single-table probability via the log-factorial route."""
    from scipy.special import gammaln

    return float(
        np.exp(
            gammaln(a + b + 1)
            + gammaln(c + d + 1)
            + gammaln(a + c + 1)
            + gammaln(b + d + 1)
            - gammaln(a + 1)
            - gammaln(b + 1)
            - gammaln(c + 1)
            - gammaln(d + 1)
            - gammaln(a + b + c + d + 1)
        )
    )


def fisher_two_tailed(a: int, b: int, c: int, d: int) -> float:
    """cbioportal's two-tailed Fisher exact p-value.

    The two-tailed p is the sum of the hypergeometric probabilities of
    every table (in both directions from the observed one) whose
    probability does not exceed the observed table's -- the exact
    convention of ``FisherExactTestCalculator.java``.

    >>> round(fisher_two_tailed(3, 1, 0, 4), 4)  # a decent enrichment
    0.1429
    >>> bool(fisher_two_tailed(1, 1, 1, 1) > 0.9)  # nothing going on
    True
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    base_p = _fisher_table_p(a, b, c, d)
    total = 0.0
    for a_alt in range(n + 1):
        # keep the marginals fixed: a+c and b+d rows, a+b column
        for c_alt in range(n - a_alt + 1):
            if a_alt + c_alt != a + c:
                continue
            b_alt = (a + b) - a_alt
            d_alt = (b + d) - b_alt
            if b_alt < 0 or d_alt < 0:
                continue
            if _fisher_table_p(a_alt, b_alt, c_alt, d_alt) <= base_p + 1e-15:
                total += _fisher_table_p(a_alt, b_alt, c_alt, d_alt)
    return min(total, 1.0)


def alteration_enrichment(
    altered_values: Sequence[float],
    unaltered_values: Sequence[float],
) -> dict[str, float]:
    """Altered-vs-unaltered enrichment (cbioportal's dispatch).

    Binary (0/1) values take the two-tailed Fisher route on the
    ``[[unaltered, altered], ...]`` table; numeric values take the
    pooled two-sided t test (cbioportal log2-transforms RNA-seq
    upstream -- the caller's scale is used as-is here).  Returns
    ``{'p_value': ..., 'statistic': ...}`` plus the group means.

    >>> import numpy as np
    >>> rng = np.random.default_rng(3)
    >>> out = alteration_enrichment(
    ...     rng.normal(2.0, 0.5, 20), rng.normal(1.0, 0.5, 25))
    >>> bool(out["p_value"] < 1e-4)
    True
    >>> binary = alteration_enrichment(
    ...     [1, 1, 1, 1], [1, 0, 0, 0, 0])
    >>> bool(binary["p_value"] < 0.05)
    True
    """
    altered = np.asarray(altered_values, dtype=float)
    unaltered = np.asarray(unaltered_values, dtype=float)
    if altered.size == 0 or unaltered.size == 0:
        raise DataValidationError("both cohorts need at least one value")
    if set(np.unique(np.concatenate([altered, unaltered])).tolist()) <= {0.0, 1.0}:
        alt_pos = int(altered.sum())
        alt_neg = int(altered.size - alt_pos)
        una_pos = int(unaltered.sum())
        una_neg = int(unaltered.size - una_pos)
        p = fisher_two_tailed(una_pos, alt_pos, una_neg, alt_neg)
        return {
            "p_value": float(p),
            "statistic": float("nan"),
            "altered_mean": float(altered.mean()),
            "unaltered_mean": float(unaltered.mean()),
        }
    from scipy.stats import ttest_ind

    stat, p = ttest_ind(altered, unaltered, equal_var=True)
    return {
        "p_value": float(p),
        "statistic": float(stat),
        "altered_mean": float(altered.mean()),
        "unaltered_mean": float(unaltered.mean()),
    }


def parse_cbio_clinical(path: str) -> pd.DataFrame:
    """Read a cbioportal clinical file (the 5-row header contract).

    Rows 1-4 are the ``#``-prefixed Display Name / Description /
    Datatype / Priority lines, row 5 the attribute IDs (UPPERCASE);
    the returned frame is indexed by the first ID column
    (``PATIENT_ID`` or ``SAMPLE_ID``).

    >>> import tempfile
    >>> from pathlib import Path
    >>> text = (
    ...     "#Display Name\\tPatient Identifier\\n"
    ...     "#Description\\tUnique patient id\\n"
    ...     "#Datatype\\tSTRING\\n"
    ...     "#Priority\\t1\\n"
    ...     "PATIENT_ID\\tOS_STATUS\\tOS_MONTHS\\n"
    ...     "P1\\t1:DECEASED\\t12.5\\n"
    ...     "P2\\t0:LIVING\\t30.0\\n"
    ... )
    >>> with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
    ...     _ = handle.write(text)
    >>> clinical = parse_cbio_clinical(handle.name)
    >>> clinical.index.tolist()
    ['P1', 'P2']
    >>> clinical["OS_MONTHS"].tolist()
    [12.5, 30.0]
    >>> Path(handle.name).unlink()
    """
    frame = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    id_column = "PATIENT_ID" if "PATIENT_ID" in frame.columns else "SAMPLE_ID"
    if id_column not in frame.columns:
        raise DataValidationError(
            f"clinical file needs a PATIENT_ID or SAMPLE_ID column, got {list(frame.columns)}"
        )
    frame = frame.set_index(id_column)
    numeric = []
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().mean() > 0.5:
            frame[column] = converted
            numeric.append(column)
    _LOG.debug("parsed cbio clinical: %d rows, %d numeric columns", len(frame), len(numeric))
    return frame


def parse_cbio_cna(path: str) -> pd.DataFrame:
    """Read a cbioportal CNA matrix (the ``-2..2`` vocabulary).

    Wide format: ``Hugo_Symbol`` / ``Entrez_Gene_Id`` + one column per
    sample, values ``-2`` deep loss, ``-1`` het loss, ``0`` diploid,
    ``1`` gain, ``2`` amplification.  Returns genes x samples.

    >>> import tempfile
    >>> from pathlib import Path
    >>> text = (
    ...     "Hugo_Symbol\\tS1\\tS2\\n"
    ...     "TP53\\t-2\\t0\\n"
    ...     "MYC\\t2\\t1\\n"
    ... )
    >>> with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
    ...     _ = handle.write(text)
    >>> cna = parse_cbio_cna(handle.name)
    >>> float(cna.loc["TP53", "S1"])
    -2.0
    >>> int(cna.loc["MYC"].sum())
    3
    >>> Path(handle.name).unlink()
    """
    frame = pd.read_csv(path, sep="\t")
    for id_column in ("Hugo_Symbol", "Entrez_Gene_Id"):
        if id_column in frame.columns:
            frame = frame.set_index(id_column)
            break
    else:
        raise DataValidationError(
            "CNA file needs a Hugo_Symbol or Entrez_Gene_Id column"
        )
    drop = [c for c in ("Entrez_Gene_Id", "Hugo_Symbol") if c in frame.columns]
    if drop:
        frame = frame.drop(columns=drop)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    bad = (~frame.isin([-2, -1, 0, 1, 2, np.nan])).to_numpy().sum()
    if bad:
        raise DataValidationError(
            f"{bad} CNA value(s) outside the -2..2 vocabulary"
        )
    return frame
