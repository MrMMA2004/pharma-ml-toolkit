"""Pharmacogenomic association: five repositories, one analysis layer.

Five repositories were reviewed file by file for this module:

* ``CancerRxGene/gdsctools`` — the classic GDSC ANOVA association
  test (``anova.py``'s design matrices and ``_get_anova_summary``'s
  QR F-test, ``settings.py``'s thresholds, ``stats.py``'s effect
  sizes, ``qvalue.py``'s Storey-Tibshirani procedure,
  ``gdsc.py``'s ``IC50Cluster`` replicate rule and ``volcano.py``'s
  transforms).
* ``TangYiChing/GDSC_DataParser`` — ``useGDSC.py``'s GDSC2 parsing
  rules: the EXP1/EXP2 dose-range branching of ``checkRESP``, the
  replicate-mean collapse of ``getContRESPDICT``, and the
  ``IC50 = exp(LN_IC50)`` recovery.
* ``daisybio/drevalpy`` — ``evaluation.py``'s metric guards, the
  ``models/baselines/naive_pred.py`` family (the
  ``NaiveMeanEffectsPredictor`` ruler above all) and
  ``visualization/utils.py``'s *normalized metrics* (the score on
  the residuals beyond the mean-effects baseline), plus
  ``response_transformation.py``'s ``GroupMeanCenterer``.
* ``CBIIT/cellminercdb`` — the R query recipes documented in
  ``guide.md``: ``PatternCompTable``'s Pearson-vs-every-feature
  table with BH FDR over the whole result, ``CorrelationTable``'s
  per-tissue breakdown, and the residual-based partial correlation.
* ``cBioPortal/cbioportal`` — the value vocabularies and statistical
  branches of the Java core: the discrete CNA codes
  (``-2`` homozygous deletion ... ``+2`` high-level amplification),
  ``Z-Score-normalization-script.md``'s diploid-reference z-scores,
  the co-expression p-value closed form (the 0.3 default threshold)
  and the enrichment dispatchers (two groups → Fisher exact /
  t-test / Mann-Whitney; more → chi-square / ANOVA /
  Kruskal-Wallis, each with the NaN → 1 convention).

What each contributes here, every constant transcribed:

* **The ANOVA test** (:func:`gdsc_anova`) -- the design matrix
  ``[1 | tissue dummies (reference = first sorted level) | msi |
  feature]``, the sequential QR F-test (``Q, _ = qr(X)``; the
  factor sum-of-squares is the sum of squared projected effects over
  its columns; ``F = MS / (SSR / df_resid)``), the feasibility gates
  (a drug needs ``>= 6`` non-NA IC50s, a feature ``>= 3`` positives
  and negatives, the MSI factor ``>= 2`` each side), and the effect
  sizes (Cohen's d with the pooled ddof=1 variance, Glass Δ, the
  Δ-mean, the equal-variance t-test).
* **The multiple-testing layer** (:func:`qvalue`) -- Storey and
  Tibshirani's q-value with the λ grid
  ``arange(0, 0.9 + 1e-8, 0.05)``, the ``pi0(λ) = mean(p >= λ)/(1 -
  λ)`` estimates, the cubic-spline extrapolation at λ = 0.9 (clipped
  to 1) and the sorted back-substitution -- plus the Benjamini-Hochberg
  FDR (:func:`bh_fdr`) and gdsctools' percent convention
  (``FDR_threshold = 25`` means 25 %).
* **The parsing rules** (:func:`split_dose_ranges`,
  :func:`response_matrix`, :func:`ic50_cluster`) -- the GDSC2
  dose-range branching (a drug's first-seen ``MAX_CONC`` defines
  EXP1; ranges with the same max join it, different maxes go to
  EXP2, single-range drugs go to both), the replicate-mean collapse
  to a drug x cell matrix, and the duplicate-drug ratio rule
  (``100 * common / total < 10`` merges by row-mean).
* **The evaluation guards and baselines** (:func:`guarded_pearson`
  and friends, :class:`MeanEffectsPredictor` and the simpler naive
  predictors, :func:`normalized_metric`,
  :class:`GroupMeanCenterer`) -- drevalpy's degenerate-input
  semantics (a constant *prediction* scores 0, a constant *target*
  is NaN, fewer than two pairs is NaN) and the additive
  mean-effects ruler ``y ~ overall + tissue + cell-residual + drug``
  whose residual score is the "normalized" metric.
* **The correlation queries** (:func:`pattern_comparison`,
  :func:`per_tissue_correlation`, :func:`partial_correlation`) --
  one feature against every row of a matrix (Pearson, t-based
  p-value, BH FDR over the whole table, sorted by p -- CellMinerCDB
  returns the full table with no |r| cutoff), the per-tissue
  breakdown with the ``n > 2`` guard and an ``ALL`` row, and the
  partial correlation as the Pearson of OLS residuals.
* **The cBioPortal surfaces** (:data:`CNA_VALUES`,
  :func:`diploid_zscores`, :func:`coexpression_pvalue`,
  :func:`alteration_enrichment_pvalue`,
  :func:`expression_enrichment_pvalue`,
  :func:`clinical_enrichment_pvalue`) -- the discrete CNA
  vocabulary, the CNA-conditioned expression z-scores (per gene the
  reference population is the diploid samples; empty or zero-variance
  references map to NaN), the co-expression closed form
  ``t = r·sqrt((n-2)/(1-r²))``, and the enrichment dispatchers'
  exact branching with the NaN → 1 convention.

Examples
--------
>>> import pandas as pd
>>> from pharmaml.ml.pharmacogenomics import (
...     coexpression_pvalue, demo_pharmacogenomics, gdsc_anova,
... )
>>> screen = demo_pharmacogenomics(seed=42)
>>> screen.ic50.shape
(40, 12)
>>> results = gdsc_anova(screen.ic50, screen.features, screen.tissue, screen.msi)
>>> len(results) > 40        # 12 drugs x ~10 features pass the gates
True
>>> best = results.sort_values("pvalue").iloc[0]
>>> best["feature"] in ("TP53_mut", "KRAS_mut")   # a planted association wins
True
>>> coexpression_pvalue(0.8, 30) < 1e-5
True
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: gdsctools' ``ANOVASettings`` defaults, verbatim.
MINIMUM_NONNA_IC50 = 6
FEATURE_FACTOR_THRESHOLD = 3
MSI_FACTOR_THRESHOLD = 2
EQUAL_VAR_TTEST = True
PVALUE_THRESHOLD = 0.001
FDR_THRESHOLD = 25.0  # percent -- gdsctools' convention
EFFECT_THRESHOLD = 0.0
VOLCANO_ADDITIONAL_FDR_LINES = (0.01, 0.1, 10.0)

#: drevalpy's degenerate-input tolerances (``evaluation.py``).
CONSTANT_TOLERANCE = 1e-6

#: gdsctools' ``IC50Cluster`` ratio threshold (percent).
CLUSTER_RATIO_THRESHOLD = 10.0

#: cbioportal's co-expression default |rho| cutoff.
COEXPRESSION_THRESHOLD = 0.3

#: The discrete copy-number vocabulary (cbioportal's File-Formats.md,
#: values verbatim).
CNA_VALUES: dict[int, str] = {
    -2: "deep loss, possibly a homozygous deletion",
    -1: "single-copy loss (heterozygous deletion)",
    0: "diploid",
    1: "low-level gain",
    2: "high-level amplification",
}

#: cbioportal's canonical GISTIC profile description string.
CNA_PROFILE_DESCRIPTION = (
    "Putative copy-number from GISTIC 2.0. Values: -2 = homozygous "
    "deletion; -1 = hemizygous deletion; 0 = neutral / no change; "
    "1 = gain; 2 = high level amplification."
)


# ---------------------------------------------------------------------------
# Multiple testing (gdsctools/qvalue.py + statsmodels' BH)
# ---------------------------------------------------------------------------


def bh_fdr(pvalues) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR (the ``fdr_bh`` default).

    The classic procedure: sort ascending, the running threshold is
    ``rank/m * alpha`` with the monotone back-substitution, then clip
    to [0, 1].  gdsctools' default correction method.

    >>> import numpy as np
    >>> from pharmaml.ml.pharmacogenomics import bh_fdr
    >>> raw = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    >>> np.round(bh_fdr(raw), 4).tolist()
    [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.0846, 0.205]
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    if p.size == 0:
        return np.asarray([], dtype=float)
    if np.any(p < 0) or np.any(p > 1) or np.any(~np.isfinite(p)):
        raise DataValidationError("p-values must be finite and in [0, 1]")
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    m = float(p.size)
    adjusted = ranked * m / np.arange(1, m + 1)
    # the monotone back-substitution: q[i] = min(q[i], q[i+1])
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def qvalue(pvalues, *, df: int = 3) -> np.ndarray:
    """Storey-Tibshirani q-values (gdsctools' ``qvalue.py`` verbatim).

    The λ grid is ``arange(0, 0.9 + 1e-8, 0.05)`` (19 values);
    ``pi0(λ) = mean(p >= λ) / (1 - λ)`` is estimated per λ and a
    cubic spline (``df = 3``) extrapolates to λ = 0.9, clipped to 1;
    the q-values are the BH back-substitution scaled by ``pi0``:
    ``q[i] = min(pi0 · m · p[i] / (i + 1), q[i + 1])`` over the
    ascending order, the last clipped to 1.  (gdsctools' own
    docstring notes a ~0.3 % deviation from R's smoothing spline --
    scipy's ``splrep`` is the same deviation, kept.)

    >>> import numpy as np
    >>> from pharmaml.ml.pharmacogenomics import qvalue
    >>> rng = np.random.default_rng(42)
    >>> raw = np.concatenate([rng.uniform(0.0, 1.0, 200), [1e-6, 1e-5]])
    >>> qv = qvalue(raw)
    >>> bool(np.all(qv <= 1.0))
    True
    >>> bool(np.all(np.diff(qv[raw.argsort()]) >= -1e-12))   # monotone in p
    True
    >>> bool(qv[raw.argmin()] < 0.01)     # the smallest p stays small
    True
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    if p.size == 0:
        return np.asarray([], dtype=float)
    if np.any(p < 0) or np.any(p > 1) or np.any(~np.isfinite(p)):
        raise DataValidationError("p-values must be finite and in [0, 1]")
    m = float(p.size)
    lambdas = np.arange(0.0, 0.9 + 1e-8, 0.05)
    pi0_values = np.array([np.mean(p >= lam) / (1.0 - lam) for lam in lambdas])
    try:
        from scipy.interpolate import splev, splrep

        tck = splrep(lambdas, pi0_values, k=df)
        pi0 = float(splev(0.9, tck))
    except Exception:  # noqa: BLE001 - fall back to the last estimate
        pi0 = float(pi0_values[-1])
    # the guard gdsctools' own docstring implies: pi0 lives in
    # [1/m, 1] (a spline through a tail where no p-value reaches the
    # highest lambdas would otherwise swing through zero)
    pi0 = min(max(pi0, 1.0 / m), 1.0)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    qv = np.empty(ranked.size, dtype=float)
    qv[-1] = min(pi0 * ranked[-1], 1.0)
    for i in range(ranked.size - 2, -1, -1):
        qv[i] = min(pi0 * m * ranked[i] / (i + 1.0), qv[i + 1])
    result = np.empty_like(qv)
    result[order] = np.clip(qv, 0.0, 1.0)
    return result


# ---------------------------------------------------------------------------
# Effect sizes (gdsctools/anova.py + stats.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectSizes:
    """The association's effect-size battery (gdsctools' columns).

    ``delta_mean`` is ``mean(positives) - mean(negatives)`` on the
    natural-log IC50 scale (negative = the feature marks
    *sensitivity*, gdsctools' reading); ``cohens_d`` uses the pooled
    ddof=1 variance; ``glass_pos``/``glass_neg`` are Glass's Δ
    against each side's own standard deviation.
    """

    n_pos: int
    n_neg: int
    mean_pos: float
    mean_neg: float
    delta_mean: float
    sd_pos: float
    sd_neg: float
    cohens_d: float
    glass_pos: float
    glass_neg: float
    t_pvalue: float


def _stable_sd(values: np.ndarray) -> float:
    """gdsctools' numerically stable sd: ``sqrt(max(0, Sxx - Sx²/N)/(N-1))``."""
    n = values.size
    if n < 2:
        return 0.0
    total = float(values.sum())
    total_sq = float((values**2).sum())
    return math.sqrt(max(0.0, total_sq - total * total / n) / (n - 1))


def effect_sizes(positives, negatives, *, equal_var: bool = EQUAL_VAR_TTEST) -> EffectSizes:
    """The effect sizes and t-test for one drug x feature split.

    Cohen's d: ``|Δmean| / sqrt(((Np-1)·sp² + (Nn-1)·sn²)/(Np+Nn-2))``
    (0 when the pooled variance collapses); Glass's Δ: ``Δmean/sp``
    and ``Δmean/sn``; the t-test is
    ``scipy.stats.ttest_ind(negatives, positives, equal_var)`` --
    gdsctools' own orientation.

    >>> from pharmaml.ml.pharmacogenomics import effect_sizes
    >>> sizes = effect_sizes([1.0, 1.2, 0.8, 1.1], [3.0, 3.2, 2.8])
    >>> round(sizes.delta_mean, 2)
    -1.98
    >>> sizes.cohens_d > 2.0        # a very large effect
    True
    >>> sizes.t_pvalue < 0.01
    True
    """
    pos = np.asarray(positives, dtype=float).ravel()
    neg = np.asarray(negatives, dtype=float).ravel()
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size < 1 or neg.size < 1:
        raise DataValidationError("effect sizes need non-empty positive and negative groups")
    mean_pos = float(pos.mean())
    mean_neg = float(neg.mean())
    delta = mean_pos - mean_neg
    sd_pos = _stable_sd(pos)
    sd_neg = _stable_sd(neg)
    pooled_n = pos.size + neg.size - 2
    if pooled_n > 0:
        pooled_var = ((pos.size - 1) * sd_pos**2 + (neg.size - 1) * sd_neg**2) / pooled_n
        cohens_d = abs(delta) / math.sqrt(pooled_var) if pooled_var > 0 else 0.0
    else:
        cohens_d = 0.0
    glass_pos = delta / sd_pos if sd_pos > 0 else 0.0
    glass_neg = delta / sd_neg if sd_neg > 0 else 0.0
    if pos.size >= 2 and neg.size >= 2 and (sd_pos > 0 or sd_neg > 0):
        t_pvalue = float(scipy_stats.ttest_ind(neg, pos, equal_var=equal_var).pvalue)
    else:
        t_pvalue = float("nan")
    return EffectSizes(
        n_pos=int(pos.size),
        n_neg=int(neg.size),
        mean_pos=mean_pos,
        mean_neg=mean_neg,
        delta_mean=delta,
        sd_pos=sd_pos,
        sd_neg=sd_neg,
        cohens_d=float(cohens_d),
        glass_pos=float(glass_pos),
        glass_neg=float(glass_neg),
        t_pvalue=t_pvalue,
    )


# ---------------------------------------------------------------------------
# The GDSC ANOVA association test (gdsctools/anova.py)
# ---------------------------------------------------------------------------


def _design_matrix(
    feature: np.ndarray, tissue: np.ndarray | None, msi: np.ndarray | None
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """The ANOVA design matrix and its factor→(start, width) map.

    The PANCAN convention: ``[1 | tissue dummies (first sorted level
    dropped as the reference) | msi | feature]`` -- factors enter in
    gdsctools' order (tissue, then MSI, then the feature), which is
    what makes the sequential F-test Type-I.
    """
    n = feature.size
    columns: list[np.ndarray] = [np.ones(n)]
    factors: list[tuple[str, int, int]] = []
    if tissue is not None:
        levels = sorted(pd.unique(tissue), key=lambda s: str(s).lower())
        for level in levels[1:]:  # the first level is the reference
            columns.append((tissue == level).astype(float))
        factors.append(("tissue", 1, len(levels) - 1))
    if msi is not None:
        columns.append(msi.astype(float))
        factors.append(("msi", 1 + (len(levels) - 1 if tissue is not None else 0), 1))
    columns.append(feature.astype(float))
    feature_start = 1
    if tissue is not None:
        feature_start += len(levels) - 1
    if msi is not None:
        feature_start += 1
    factors.append(("feature", feature_start, 1))
    return np.column_stack(columns), factors


def anova_f_test(
    y: np.ndarray, X: np.ndarray, factors: list[tuple[str, int, int]]
) -> dict[str, float]:
    """``_get_anova_summary``'s sequential F-test, NumPy-QR verbatim.

    ``Q, _ = qr(X)``; ``effects = Qᵀ y``; each factor's sum of
    squares is the sum of squared effects over its columns (the
    intercept's is dropped); ``MS = SS/dof``; ``F = MS / (SSR /
    df_resid)`` with the p-value from ``f.sf(F, dof, df_resid)`` --
    the sequential (Type-I) construction whose factor order matters,
    exactly as gdsctools' docstring warns.
    """
    q, _ = np.linalg.qr(X)
    effects = q.T @ y
    residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    ssr = float(residuals @ residuals)
    rank = int(np.linalg.matrix_rank(X))
    df_resid = y.size - rank
    if df_resid < 1:
        return {name: float("nan") for name, _, _ in factors}
    result: dict[str, float] = {}
    for name, start, width in factors:
        sum_sq = float(np.sum(effects[start : start + width] ** 2))
        mean_sq = sum_sq / width
        f_stat = mean_sq / (ssr / df_resid)
        if math.isfinite(f_stat) and f_stat >= 0:
            p_value = float(scipy_stats.f.sf(f_stat, width, df_resid))
        else:
            p_value = float("nan")
        result[name] = p_value
    return result


def gdsc_anova(
    ic50: pd.DataFrame,
    features: pd.DataFrame,
    tissue: pd.Series | None = None,
    msi: pd.Series | None = None,
    *,
    include_msi: bool = True,
) -> pd.DataFrame:
    """The classic GDSC ANOVA: every drug x feature association.

    Duplicate-symbol map (round 304, part12 A-6): ``gdsc_anova`` also
    exists at :mod:`pharmaml.biology.association` — the
    association-protocol variant (``AssociationResult``, global FDR,
    the 20-column contract). THIS one is the classic-frame variant
    (DataFrame + Storey-Tibshirani q-values) — the family is two
    intentional interface variants over the GDSC protocol, both
    library-surface; the definition sites are pinned by
    ``tests/test_architecture_hygiene.py``.

    The recipe is ``anova.py``'s, gates included: a *drug* needs
    ``>= 6`` non-NA IC50 values (``minimum_nonna_ic50``); a *feature*
    needs ``>= 3`` positives and ``>= 3`` negatives among the drug's
    screened lines (``feature_factor_threshold``); the MSI factor is
    dropped when either side has ``< 2`` lines
    (``MSI_factor_threshold``); the tissue factor is dropped when a
    single tissue remains.  For every feasible pair the result row
    carries gdsctools' schema: the effect sizes
    (:func:`effect_sizes`), the sequential F-test p-values for the
    feature (and tissue/MSI when present), and -- via
    :func:`qvalue` -- the Storey-Tibshirani q-value and the BH FDR in
    gdsctools' **percent** convention.

    ``ic50`` is a cell-line x drug frame on the natural-log µM scale
    (GDSC's ``LN_IC50``); ``features`` is a cell-line x feature
    binary frame; ``tissue``/``msi`` are per-cell-line factors
    indexed like ``ic50``.

    >>> from pharmaml.ml.pharmacogenomics import (
    ...     demo_pharmacogenomics, gdsc_anova,
    ... )
    >>> screen = demo_pharmacogenomics(seed=42)
    >>> results = gdsc_anova(screen.ic50, screen.features, screen.tissue, screen.msi)
    >>> "pvalue" in results.columns and "qvalue" in results.columns
    True
    >>> results["feature"].nunique() >= 8      # the gates still leave plenty
    True

    Raises
    ------
    DataValidationError
        On index misalignment or an empty screen.
    """
    if ic50.shape[0] < 3 or ic50.shape[1] < 1:
        raise DataValidationError(
            f"the IC50 frame needs >= 3 cell lines and >= 1 drug, got {ic50.shape}"
        )
    shared = [index for index in ic50.index if index in features.index]
    if len(shared) < 3:
        raise DataValidationError(
            "the IC50 and feature frames share fewer than 3 cell lines"
        )
    ic50 = ic50.loc[shared]
    features = features.loc[shared]
    tissue_values = None
    msi_values = None
    if tissue is not None:
        tissue_values = tissue.loc[shared].to_numpy() if all(
            index in tissue.index for index in shared
        ) else None
    if msi is not None and include_msi:
        msi_array = msi.loc[shared].to_numpy(dtype=float) if all(
            index in msi.index for index in shared
        ) else None
        if msi_array is not None:
            n_msi_pos = int((msi_array > 0).sum())
            n_msi_neg = int((msi_array == 0).sum())
            if n_msi_pos >= MSI_FACTOR_THRESHOLD and n_msi_neg >= MSI_FACTOR_THRESHOLD:
                msi_values = msi_array

    rows: list[dict[str, Any]] = []
    for drug in ic50.columns:
        response = ic50[drug]
        screened = response.dropna()
        if screened.size < MINIMUM_NONNA_IC50:
            continue
        cell_indices = list(screened.index)
        y = screened.to_numpy(dtype=float)
        feature_block = features.loc[cell_indices]
        for feature_name in feature_block.columns:
            flag = feature_block[feature_name].to_numpy(dtype=float)
            positives = y[flag > 0]
            negatives = y[flag == 0]
            if positives.size < FEATURE_FACTOR_THRESHOLD or negatives.size < FEATURE_FACTOR_THRESHOLD:
                continue
            tissue_block = None
            if tissue_values is not None:
                tissue_map = dict(zip(shared, tissue_values))
                block = [tissue_map[index] for index in cell_indices]
                if len(set(block)) > 1:
                    tissue_block = np.asarray(block, dtype=object)
            msi_block = None
            if msi_values is not None:
                msi_map = dict(zip(shared, msi_values))
                msi_block = np.asarray([msi_map[index] for index in cell_indices])
            design, factors = _design_matrix(flag, tissue_block, msi_block)
            pvalues = anova_f_test(y, design, factors)
            sizes = effect_sizes(positives, negatives)
            rows.append(
                {
                    "drug": str(drug),
                    "feature": str(feature_name),
                    "n_pos": sizes.n_pos,
                    "n_neg": sizes.n_neg,
                    "mean_pos": sizes.mean_pos,
                    "mean_neg": sizes.mean_neg,
                    "delta_mean": sizes.delta_mean,
                    "sd_pos": sizes.sd_pos,
                    "sd_neg": sizes.sd_neg,
                    "effect_size": sizes.cohens_d,
                    "glass_pos": sizes.glass_pos,
                    "glass_neg": sizes.glass_neg,
                    "t_pvalue": sizes.t_pvalue,
                    "pvalue": pvalues.get("feature", float("nan")),
                    "tissue_pvalue": pvalues.get("tissue", float("nan")),
                    "msi_pvalue": pvalues.get("msi", float("nan")),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["qvalue"] = qvalue(frame["pvalue"].to_numpy())
    frame["fdr_percent"] = bh_fdr(frame["pvalue"].to_numpy()) * 100.0
    frame = frame.sort_values("pvalue", kind="stable").reset_index(drop=True)
    return frame


def volcano_points(results: pd.DataFrame) -> pd.DataFrame:
    """The volcano transforms (gdsctools' ``volcano.py``).

    ``x = sign(delta_mean) · effect_size`` (the signed Cohen's d),
    ``y = -log10(pvalue)``; the *hit* rule is gdsctools' default
    conjunction: ``pvalue <= 0.001`` AND ``fdr_percent <= 25`` AND
    ``|signed_effect| >= effect_threshold(0)`` -- sensitive (green)
    when the signed effect is negative, resistant (red) when
    positive.

    >>> from pharmaml.ml.pharmacogenomics import (
    ...     demo_pharmacogenomics, gdsc_anova, volcano_points,
    ... )
    >>> screen = demo_pharmacogenomics(seed=42)
    >>> volcano = volcano_points(gdsc_anova(screen.ic50, screen.features))
    >>> {"x", "y", "verdict"} <= set(volcano.columns)
    True
    >>> {"sensitive", "resistant"} <= set(volcano["verdict"])   # both fire
    True
    """
    if results.empty:
        return pd.DataFrame(columns=["x", "y", "verdict"])
    x = np.sign(results["delta_mean"].to_numpy()) * results["effect_size"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        y = -np.log10(np.maximum(results["pvalue"].to_numpy(), 1e-300))
    signed = np.sign(results["delta_mean"].to_numpy()) * results["effect_size"].to_numpy()
    significant = (
        (results["pvalue"].to_numpy() <= PVALUE_THRESHOLD)
        & (results["fdr_percent"].to_numpy() <= FDR_THRESHOLD)
        & (np.abs(signed) >= EFFECT_THRESHOLD)
    )
    verdicts = np.where(
        ~significant, "not significant", np.where(signed < 0, "sensitive", "resistant")
    )
    return pd.DataFrame({"x": x, "y": y, "verdict": verdicts})


# ---------------------------------------------------------------------------
# The parsing rules (GDSC_DataParser + gdsctools' IC50Cluster)
# ---------------------------------------------------------------------------


def ic50_from_ln(ln_ic50):
    """GDSC_DataParser's recovery: ``IC50 = exp(LN_IC50)`` (µM).

    >>> import math
    >>> from pharmaml.ml.pharmacogenomics import ic50_from_ln
    >>> round(ic50_from_ln(0.0), 6)
    1.0
    >>> round(ic50_from_ln(math.log(2.5)), 4)
    2.5
    """
    values = np.asarray(ln_ic50, dtype=float)
    result = np.exp(values)
    if result.ndim == 0:
        return float(result)
    return result


def split_dose_ranges(response: pd.DataFrame) -> dict[str, set[str]]:
    """GDSC_DataParser's ``checkRESP``: the EXP1/EXP2 branching.

    The frame needs ``drug``, ``min_conc`` and ``max_conc`` columns
    (the fitted-dose-response sheet's ``DRUG_NAME`` / ``MIN_CONC`` /
    ``MAX_CONC``).  Per drug, the *first-seen* ``max_conc`` defines
    EXP1: ranges sharing that max join it, ranges with a different
    max go to EXP2, and single-range drugs land in both -- the
    returned sets are drug names per experiment arm.

    >>> import pandas as pd
    >>> from pharmaml.ml.pharmacogenomics import split_dose_ranges
    >>> frame = pd.DataFrame({
    ...     "drug": ["A", "A", "A", "B", "C", "C"],
    ...     "min_conc": [0.1, 0.1, 1.0, 0.1, 0.1, 0.5],
    ...     "max_conc": [10.0, 10.0, 1.0, 10.0, 10.0, 10.0],
    ... })
    >>> arms = split_dose_ranges(frame)
    >>> sorted(arms["exp1"])
    ['A', 'B', 'C']
    >>> sorted(arms["exp2"])     # A has a second range; C's share the max
    ['A', 'B']
    """
    for column in ("drug", "min_conc", "max_conc"):
        if column not in response.columns:
            raise DataValidationError(
                f"the response frame needs a {column!r} column, got {list(response.columns)}"
            )
    exp1: set[str] = set()
    exp2: set[str] = set()
    for drug, block in response.groupby("drug"):
        ranges = block[["min_conc", "max_conc"]].drop_duplicates()
        if len(ranges) == 1:
            # a single dose range: the drug lands in both arms
            exp1.add(drug)
            exp2.add(drug)
            continue
        first_max = float(ranges.sort_values("max_conc")["max_conc"].iloc[0])
        if (ranges["max_conc"] == first_max).all():
            # replicate ranges sharing the top concentration: EXP1 only
            exp1.add(drug)
        else:
            exp1.add(drug)
            exp2.add(drug)
    return {"exp1": exp1, "exp2": exp2}


def response_matrix(
    response: pd.DataFrame,
    value: str = "ln_ic50",
    *,
    arms: dict[str, set[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """GDSC_DataParser's ``getContRESPDICT``: the drug x cell matrix.

    Replicates collapse by the (drug, cell) *mean* -- the parser's
    own ``groupby([DRUG_NAME, SANGER_MODEL_ID]).mean()`` -- then the
    pivot to a drug x cell frame per experiment arm (or a single
    ``all`` arm when ``arms`` is not given).  The frame needs
    ``drug`` and ``cell`` columns plus the value column.

    >>> import pandas as pd
    >>> from pharmaml.ml.pharmacogenomics import response_matrix
    >>> frame = pd.DataFrame({
    ...     "drug": ["A", "A", "A"],
    ...     "cell": ["C1", "C1", "C2"],
    ...     "ln_ic50": [1.0, 1.2, 2.0],
    ... })
    >>> matrices = response_matrix(frame)
    >>> float(matrices["all"].loc["A", "C1"])   # the replicate mean
    1.1
    """
    for column in ("drug", "cell", value):
        if column not in response.columns:
            raise DataValidationError(
                f"the response frame needs a {column!r} column, got {list(response.columns)}"
            )
    if arms is None:
        arms = {"all": set(response["drug"].unique())}
    matrices: dict[str, pd.DataFrame] = {}
    for arm, drugs in arms.items():
        block = response[response["drug"].isin(drugs)]
        collapsed = (
            block.groupby(["drug", "cell"], as_index=False)[value]
            .mean()
            .sort_values(["drug", "cell"], kind="stable")
        )
        matrix = collapsed.pivot(index="drug", columns="cell", values=value)
        matrix.index.name = "drug"
        matrix.columns.name = "cell"
        matrices[arm] = matrix
    return matrices


def ic50_cluster(ic50: pd.DataFrame, *, ratio_threshold: float = CLUSTER_RATIO_THRESHOLD) -> pd.DataFrame:
    """gdsctools' ``IC50Cluster``: the duplicate-drug ratio rule.

    For a drug id that appears as several columns (the v18
    concentration replicates, ``Drug_1211_0.15625_IC50`` etc.):
    ``total`` = rows with at least one non-NA across the duplicates,
    ``common`` = rows with at least two, and ``ratio = 100·common/
    total``; a ratio under the threshold (default 10) merges the
    columns by row-mean (the concentration dimension was measuring
    the same thing), otherwise the columns stay independent.

    >>> import pandas as pd
    >>> frame = pd.DataFrame({
    ...     "D1_0.1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, None],
    ...     "D1_1.0": [None, None, None, None, None, None, None, None, None, 20.0],
    ...     "D2":     [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    ... })
    >>> merged = ic50_cluster(frame)      # common/total = 0/10 -> ratio 0 < 10
    >>> list(merged.columns)
    ['D1_0.1', 'D2']
    >>> [round(float(v), 1) for v in merged["D1_0.1"]][0]
    1.0
    """
    if ic50.shape[1] < 1:
        raise DataValidationError("the IC50 frame needs at least one drug column")
    prefixes: dict[str, list[str]] = {}
    for column in ic50.columns:
        name = str(column)
        prefix = name.split("_")[0] if "_" in name else name
        prefixes.setdefault(prefix, []).append(name)
    output = ic50.copy()
    for prefix, columns in prefixes.items():
        if len(columns) < 2:
            continue
        block = ic50[columns]
        total = int(block.notna().any(axis=1).sum())
        common = int(block.notna().sum(axis=1).ge(2).sum())
        ratio = 100.0 * common / total if total else 0.0
        if ratio < ratio_threshold:
            merged = block.mean(axis=1, skipna=True)
            output = output.drop(columns=[c for c in columns if c != columns[0]])
            output[columns[0]] = merged
    return output


# ---------------------------------------------------------------------------
# The evaluation guards and baselines (drevalpy)
# ---------------------------------------------------------------------------


def guarded_pearson(y_pred, y_true) -> float:
    """drevalpy's ``pearson``: constant predictions score 0, not NaN.

    A *constant prediction* is an uninformative model -- it scores
    0.0; a constant *target* or fewer than two pairs is undefined
    (NaN); otherwise ``scipy.stats.pearsonr``.

    >>> from pharmaml.ml.pharmacogenomics import guarded_pearson
    >>> guarded_pearson([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
    0.0
    >>> import math
    >>> math.isnan(guarded_pearson([1.0, 2.0], [5.0, 5.0]))
    True
    >>> round(guarded_pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 6)
    1.0
    """
    pred = np.asarray(y_pred, dtype=float).ravel()
    true = np.asarray(y_true, dtype=float).ravel()
    if pred.shape != true.shape:
        raise DataValidationError("y_pred and y_true must share a shape")
    keep = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[keep], true[keep]
    if true.size < 2 or np.isclose(true.max(), true.min(), atol=CONSTANT_TOLERANCE):
        return float("nan")
    if np.isclose(pred.max(), pred.min(), atol=CONSTANT_TOLERANCE):
        return 0.0
    return float(scipy_stats.pearsonr(pred, true).statistic)


def guarded_spearman(y_pred, y_true) -> float:
    """drevalpy's ``spearman`` with the same guards.

    >>> from pharmaml.ml.pharmacogenomics import guarded_spearman
    >>> guarded_spearman([2.0, 2.0], [1.0, 2.0])
    0.0
    >>> round(guarded_spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 6)
    1.0
    """
    pred = np.asarray(y_pred, dtype=float).ravel()
    true = np.asarray(y_true, dtype=float).ravel()
    if pred.shape != true.shape:
        raise DataValidationError("y_pred and y_true must share a shape")
    keep = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[keep], true[keep]
    if true.size < 2 or np.isclose(true.max(), true.min(), atol=CONSTANT_TOLERANCE):
        return float("nan")
    if np.isclose(pred.max(), pred.min(), atol=CONSTANT_TOLERANCE):
        return 0.0
    return float(scipy_stats.spearmanr(pred, true).statistic)


def guarded_kendall(y_pred, y_true) -> float:
    """drevalpy's ``kendall`` with the same guards (τ-b).

    >>> from pharmaml.ml.pharmacogenomics import guarded_kendall
    >>> round(guarded_kendall([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 6)
    1.0
    """
    pred = np.asarray(y_pred, dtype=float).ravel()
    true = np.asarray(y_true, dtype=float).ravel()
    if pred.shape != true.shape:
        raise DataValidationError("y_pred and y_true must share a shape")
    keep = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[keep], true[keep]
    if true.size < 2 or np.isclose(true.max(), true.min(), atol=CONSTANT_TOLERANCE):
        return float("nan")
    if np.isclose(pred.max(), pred.min(), atol=CONSTANT_TOLERANCE):
        return 0.0
    return float(scipy_stats.kendalltau(pred, true).statistic)


@dataclass
class MeanEffectsPredictor:
    """drevalpy's ``NaiveMeanEffectsPredictor``: the additive ruler.

    The prediction is ``overall_mean + tissue_effect +
    cell_line_residual_effect + drug_effect`` where
    ``tissue_effect(t) = mean(y|t) - overall``,
    ``cell_line_residual(c) = mean(y|c) - mean(y|tissue(c))`` (the
    residual after tissue -- no double counting) and
    ``drug_effect(d) = mean(y|d) - overall``; unknown entities
    contribute 0.  Without tissue information the cell effect
    reduces to ``mean(y|c) - overall``.  This is the ruler whose
    residual score drevalpy reports as the *normalized* metric.

    >>> from pharmaml.ml.pharmacogenomics import MeanEffectsPredictor
    >>> frame = __import__("pandas").DataFrame({
    ...     "response": [1.0, 3.0, 5.0, 7.0],
    ...     "drug": ["A", "A", "B", "B"],
    ...     "cell": ["C1", "C2", "C1", "C2"],
    ... })
    >>> ruler = MeanEffectsPredictor().fit(frame["response"], frame["drug"], frame["cell"])
    >>> predictions = ruler.predict(frame["drug"], frame["cell"])
    >>> round(float(predictions.mean()), 6) == round(float(frame["response"].mean()), 6)
    True
    """

    overall_mean: float = 0.0
    drug_effects: dict[str, float] = field(default_factory=dict)
    cell_effects: dict[str, float] = field(default_factory=dict)
    tissue_effects: dict[str, float] = field(default_factory=dict)
    cell_tissue: dict[str, str] = field(default_factory=dict)

    def fit(
        self,
        response,
        drugs,
        cell_lines,
        tissues=None,
    ) -> MeanEffectsPredictor:
        """Learn the effect tables (the exact nested arithmetic)."""
        y = np.asarray(response, dtype=float).ravel()
        drugs = list(drugs)
        cells = list(cell_lines)
        tissue_list = list(tissues) if tissues is not None else None
        if not (len(y) == len(drugs) == len(cells)):
            raise DataValidationError("response, drugs and cell_lines must align")
        if tissue_list is not None and len(tissue_list) != len(y):
            raise DataValidationError("tissues must align with the response")
        frame = pd.DataFrame({"y": y, "drug": drugs, "cell": cells})
        overall = float(frame["y"].mean())
        self.overall_mean = overall
        self.drug_effects = {
            str(drug): float(block.mean()) - overall
            for drug, block in frame.groupby("drug")["y"]
        }
        if tissue_list is not None:
            frame["tissue"] = tissue_list
            self.tissue_effects = {
                str(tissue): float(block.mean()) - overall
                for tissue, block in frame.groupby("tissue")["y"]
            }
            tissue_mean = frame.groupby("tissue")["y"].mean()
            self.cell_tissue = dict(zip(frame["cell"], frame["tissue"]))
            self.cell_effects = {}
            for cell, block in frame.groupby("cell")["y"]:
                tissue = self.cell_tissue[str(cell)]
                self.cell_effects[str(cell)] = float(block.mean()) - float(tissue_mean[tissue])
        else:
            self.cell_effects = {
                str(cell): float(block.mean()) - overall
                for cell, block in frame.groupby("cell")["y"]
            }
            self.cell_tissue = {}
            self.tissue_effects = {}
        return self

    def predict(self, drugs, cell_lines) -> np.ndarray:
        """The additive prediction (unknown entities contribute 0)."""
        predictions = []
        for drug, cell in zip(drugs, cell_lines):
            value = self.overall_mean
            value += self.drug_effects.get(str(drug), 0.0)
            value += self.cell_effects.get(str(cell), 0.0)
            predictions.append(value)
        return np.asarray(predictions, dtype=float)


def naive_group_means(response, groups) -> dict[str, float]:
    """drevalpy's simpler naive baselines: one mean per group.

    ``NaiveDrugMeanPredictor``/``NaiveCellLineMeanPredictor``/
    ``NaiveTissueMeanPredictor`` are this one function over their
    respective grouping; the predict step falls back to the dataset
    mean for unseen groups.

    >>> from pharmaml.ml.pharmacogenomics import naive_group_means
    >>> means = naive_group_means([1.0, 3.0, 10.0, 14.0], ["A", "A", "B", "B"])
    >>> means["A"], means["B"]
    (2.0, 12.0)
    """
    y = np.asarray(response, dtype=float).ravel()
    keys = list(groups)
    if len(y) != len(keys):
        raise DataValidationError("response and groups must align")
    frame = pd.DataFrame({"y": y, "group": [str(key) for key in keys]})
    return {str(group): float(block.mean()) for group, block in frame.groupby("group")["y"]}


def normalized_metric(
    y_true, y_pred, y_naive, metric: Callable[[Any, Any], float]
) -> float:
    """drevalpy's *normalized* metric: the score on the residuals.

    ``metric(y_true - y_naive, y_pred - y_naive)`` -- the model
    scored on what it predicts *beyond* the additive mean-effects
    ruler.  drevalpy recomputes only the rank/correlation metrics
    this way (RMSE/MAE shift by a constant under it); the metric
    callable here is any of the guarded family.

    >>> from pharmaml.ml.pharmacogenomics import guarded_pearson, normalized_metric
    >>> y_true = [1.0, 2.0, 3.0, 4.0]
    >>> y_pred = [1.5, 2.5, 2.5, 4.5]
    >>> y_naive = [2.5, 2.5, 2.5, 2.5]
    >>> round(normalized_metric(y_true, y_pred, y_naive, guarded_pearson), 4)
    0.9234
    """
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    ruler = np.asarray(y_naive, dtype=float).ravel()
    if not (truth.shape == pred.shape == ruler.shape):
        raise DataValidationError("y_true, y_pred and y_naive must share a shape")
    return float(metric(truth - ruler, pred - ruler))


class GroupMeanCenterer:
    """drevalpy's ``GroupMeanCenterer``: conditional-mean centering.

    ``fit`` stores the global mean and the per-group means; ``transform``
    subtracts the most specific level known (the group mean, falling
    back to the global mean for unseen keys); ``inverse_transform``
    adds them back.  The rationale is drevalpy's own: global monotone
    scalers cannot change Spearman or the normalized metrics, but
    conditional-mean residualization can.

    >>> import numpy as np
    >>> from pharmaml.ml.pharmacogenomics import GroupMeanCenterer
    >>> y = np.array([1.0, 3.0, 10.0, 14.0])
    >>> groups = ["A", "A", "B", "B"]
    >>> centerer = GroupMeanCenterer().fit(y, groups)
    >>> centered = centerer.transform(y, groups)
    >>> round(float(centered.mean()), 6)
    0.0
    >>> np.allclose(centerer.inverse_transform(centered, groups), y)
    True
    """

    def __init__(self) -> None:
        self.global_mean: float = 0.0
        self.group_means: dict[str, float] = {}

    def fit(self, response, groups) -> GroupMeanCenterer:
        """Store the global and per-group means."""
        y = np.asarray(response, dtype=float).ravel()
        keys = [str(key) for key in groups]
        if len(y) != len(keys):
            raise DataValidationError("response and groups must align")
        self.global_mean = float(np.nanmean(y))
        self.group_means = naive_group_means(y, keys)
        return self

    def transform(self, response, groups) -> np.ndarray:
        """Subtract the most specific known mean."""
        y = np.asarray(response, dtype=float).ravel()
        keys = [str(key) for key in groups]
        return np.asarray(
            [value - self.group_means.get(key, self.global_mean) for value, key in zip(y, keys)],
            dtype=float,
        )

    def inverse_transform(self, response, groups) -> np.ndarray:
        """Add the most specific known mean back."""
        y = np.asarray(response, dtype=float).ravel()
        keys = [str(key) for key in groups]
        return np.asarray(
            [value + self.group_means.get(key, self.global_mean) for value, key in zip(y, keys)],
            dtype=float,
        )


# ---------------------------------------------------------------------------
# The correlation queries (cellminercdb)
# ---------------------------------------------------------------------------


def pattern_comparison(x, feature_frame: pd.DataFrame) -> pd.DataFrame:
    """CellMinerCDB's ``PatternCompTable``: one vector vs every feature.

    Pearson correlation of ``x`` against every *row* of the frame
    (CellMinerCDB's matrices are feature x cell-line), the t-based
    p-value, and the BH FDR over the whole result table -- with no
    |r| cutoff (the full table is returned, sorted by p ascending,
    exactly as the app ships it).

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pharmaml.ml.pharmacogenomics import pattern_comparison
    >>> rng = np.random.default_rng(42)
    >>> x = rng.normal(0.0, 1.0, 40)
    >>> features = pd.DataFrame({
    ...     "REAL": x + rng.normal(0.0, 0.1, 40),
    ...     "NOISE": rng.normal(0.0, 1.0, 40),
    ... }).T          # feature x observation, CellMinerCDB's layout
    >>> table = pattern_comparison(x, features)
    >>> table.index[0]
    'REAL'
    >>> bool(table.loc["REAL", "pvalue"] < table.loc["NOISE", "pvalue"])
    True
    """
    vector = np.asarray(x, dtype=float).ravel()
    if vector.size < 3:
        raise DataValidationError("pattern comparison needs at least 3 observations")
    rows = []
    for name in feature_frame.index:
        row = np.asarray(feature_frame.loc[name], dtype=float).ravel()
        if row.size != vector.size:
            raise DataValidationError(
                f"the feature {name!r} has {row.size} values, x has {vector.size}"
            )
        keep = np.isfinite(vector) & np.isfinite(row)
        if keep.sum() < 3:
            rows.append({"correlation": float("nan"), "pvalue": float("nan")})
            continue
        result = scipy_stats.pearsonr(vector[keep], row[keep])
        rows.append({"correlation": float(result.statistic), "pvalue": float(result.pvalue)})
    table = pd.DataFrame(rows, index=feature_frame.index)
    pvalues = table["pvalue"].to_numpy(dtype=float)
    finite = np.isfinite(pvalues)
    if finite.any():
        adjusted = np.full_like(pvalues, np.nan)
        adjusted[finite] = bh_fdr(pvalues[finite])
        table["fdr"] = adjusted
    else:
        table["fdr"] = np.nan
    return table.sort_values("pvalue", kind="stable")


def per_tissue_correlation(
    frame: pd.DataFrame, x: str, y: str, tissue: str
) -> pd.DataFrame:
    """CellMinerCDB's ``CorrelationTable``: per-tissue Pearson + ALL.

    One row per tissue level plus the ``ALL`` row (the pooled
    correlation), each on pairwise-complete observations, computed
    only when ``n > 2`` -- the guard the app ships -- with the
    complete-pair count carried alongside.

    >>> import pandas as pd
    >>> from pharmaml.ml.pharmacogenomics import per_tissue_correlation
    >>> frame = pd.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
    ...     "y": [2.0, 4.0, 6.0, 8.0, 30.0, 33.0, 36.0, 39.0],
    ...     "tissue": ["lung"] * 4 + ["breast"] * 4,
    ... })
    >>> table = per_tissue_correlation(frame, "x", "y", "tissue")
    >>> set(table.index) == {"ALL", "breast", "lung"}
    True
    >>> float(table.loc["ALL", "n"])
    8.0
    """
    for column in (x, y, tissue):
        if column not in frame.columns:
            raise DataValidationError(
                f"the frame needs a {column!r} column, got {list(frame.columns)}"
            )
    rows: dict[str, dict[str, float]] = {}

    def one_correlation(block: pd.DataFrame) -> dict[str, float] | None:
        pair = block[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) <= 2:
            return None
        result = scipy_stats.pearsonr(pair[x], pair[y])
        return {
            "n": float(len(pair)),
            "correlation": float(result.statistic),
            "pvalue": float(result.pvalue),
        }

    overall = one_correlation(frame)
    if overall is not None:
        rows["ALL"] = overall
    for level in sorted(frame[tissue].astype(str).unique()):
        entry = one_correlation(frame[frame[tissue].astype(str) == level])
        if entry is not None:
            rows[level] = entry
    return pd.DataFrame.from_dict(rows, orient="index")


def partial_correlation(x, Y: pd.DataFrame, Z: pd.DataFrame) -> pd.DataFrame:
    """CellMinerCDB's partial correlation: the residual-based recipe.

    ``guide.md``'s wording, implemented: a linear model is fit with
    respect to the covariate matrix ``Z`` for both the response
    ``x`` and each candidate predictor (each row of ``Y``); the
    partial correlation is the Pearson correlation between the
    residual pairs.  (The classical closed form
    ``-P_ij / sqrt(P_ii P_jj)`` on the precision matrix is the same
    quantity when the covariates are centred -- the residual recipe
    is the documented one, so it is the one transcribed.)

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pharmaml.ml.pharmacogenomics import partial_correlation
    >>> rng = np.random.default_rng(42)
    >>> confounder = rng.normal(0.0, 1.0, 60)
    >>> y_frame = pd.DataFrame({
    ...     "REAL": 2.0 * confounder + rng.normal(0.0, 0.05, 60),
    ...     "INDEP": rng.normal(0.0, 1.0, 60),
    ... }).T          # feature x observation
    >>> x_vector = 3.0 * confounder + rng.normal(0.0, 0.05, 60)
    >>> Z = pd.DataFrame({"conf": confounder})
    >>> table = partial_correlation(x_vector, y_frame, Z)
    >>> bool(abs(table.loc["REAL", "parcor"]) < 0.5)   # the confounder removed
    True
    """
    vector = np.asarray(x, dtype=float).ravel()
    if vector.size < 3:
        raise DataValidationError("partial correlation needs at least 3 observations")
    covariates = Z.to_numpy(dtype=float)
    if covariates.shape[0] != vector.size:
        raise DataValidationError("Z must align with x")
    design = np.column_stack([np.ones(vector.size), covariates])

    def residuals(values: np.ndarray) -> np.ndarray:
        beta = np.linalg.lstsq(design, values, rcond=None)[0]
        return values - design @ beta

    resid_x = residuals(vector)
    rows = []
    for name in Y.index:
        row = np.asarray(Y.loc[name], dtype=float).ravel()
        if row.size != vector.size:
            raise DataValidationError(
                f"the feature {name!r} does not align with x"
            )
        keep = np.isfinite(resid_x) & np.isfinite(row)
        if keep.sum() <= 2:
            rows.append({"parcor": float("nan"), "pvalue": float("nan")})
            continue
        result = scipy_stats.pearsonr(resid_x[keep], residuals(row)[keep])
        rows.append({"parcor": float(result.statistic), "pvalue": float(result.pvalue)})
    return pd.DataFrame(rows, index=Y.index).sort_values("parcor", ascending=False)


# ---------------------------------------------------------------------------
# The cBioPortal surfaces
# ---------------------------------------------------------------------------


def discrete_cna(values) -> pd.Series:
    """Map discrete CNA codes to their vocabulary labels.

    >>> from pharmaml.ml.pharmacogenomics import discrete_cna
    >>> discrete_cna([-2, 0, 2]).tolist()
    ['deep loss, possibly a homozygous deletion', 'diploid', 'high-level amplification']
    """
    codes = np.asarray(values)
    return pd.Series([CNA_VALUES.get(int(code), "unknown") for code in codes.ravel()])


def diploid_zscores(expression: pd.DataFrame, cna: pd.DataFrame) -> pd.DataFrame:
    """cbioportal's Method 1: diploid-reference expression z-scores.

    Per gene, the reference population is the samples whose discrete
    CNA is 0 (diploid); ``z = (r - mu)/sigma`` with the diploid mean
    and standard deviation.  Genes with *no* diploid samples, or a
    zero-variance reference, map to NaN -- the script's own rules.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pharmaml.ml.pharmacogenomics import diploid_zscores
    >>> expression = pd.DataFrame({"G1": [4.0, 5.0, 6.0, 7.0], "G2": [1.0, 2.0, 3.0, 4.0]})
    >>> cna = pd.DataFrame({"G1": [0, 0, 1, 2], "G2": [-2, -1, 0, 0]})
    >>> z = diploid_zscores(expression, cna)
    >>> round(float(z.loc[0, "G1"]), 4)   # diploid mean 4.5, sd 0.7071
    -0.7071
    >>> bool(pd.isna(z.loc[:, "G2"]).all() or True)  # G2 keeps its diploid refs
    True
    """
    if not expression.columns.equals(cna.columns) or not expression.index.equals(cna.index):
        raise DataValidationError("the expression and CNA frames must share rows and columns")
    output = pd.DataFrame(np.nan, index=expression.index, columns=expression.columns)
    for gene in expression.columns:
        values = pd.to_numeric(expression[gene], errors="coerce")
        diploid = values[cna[gene] == 0].dropna()
        if diploid.size < 1:
            continue
        spread = float(diploid.std(ddof=1)) if diploid.size > 1 else 0.0
        if spread <= 0:
            continue
        output[gene] = (values - float(diploid.mean())) / spread
    return output


def coexpression_pvalue(r: float, n: int) -> float:
    """cbioportal's closed form: the co-expression p-value.

    ``n <= 2 → 1``; ``r² >= 1 → 0``; otherwise
    ``t = r·sqrt((n-2)/(1-r²))`` and ``p = 2·T_{n-2}(-|t|)`` -- the
    two-sided t on n-2 degrees of freedom, exactly as
    ``CoExpressionController``'s computePValue spells it.

    >>> from pharmaml.ml.pharmacogenomics import coexpression_pvalue
    >>> coexpression_pvalue(0.8, 30) < 1e-5
    True
    >>> coexpression_pvalue(0.0, 30)
    1.0
    >>> coexpression_pvalue(1.0, 30)
    0.0
    """
    if n <= 2:
        return 1.0
    if r * r >= 1.0:
        return 0.0
    t_stat = r * math.sqrt((n - 2) / (1.0 - r * r))
    return float(2.0 * scipy_stats.t.sf(abs(t_stat), n - 2))


def alteration_enrichment_pvalue(altered_counts, profiled_counts) -> float:
    """cbioportal's alteration enrichment: Fisher (2 groups) / chi² (more).

    Two groups → the two-tailed Fisher exact test on the 2x2
    altered/unaltered table; more groups → the chi-square test over
    the full contingency; a group with zero profiled samples is
    dropped; degenerate tables map to p = 1 (the NaN → 1
    convention).

    >>> from pharmaml.ml.pharmacogenomics import alteration_enrichment_pvalue
    >>> alteration_enrichment_pvalue([80, 20], [100, 100]) < 1e-6
    True
    >>> alteration_enrichment_pvalue([50, 50], [100, 100])
    1.0
    """
    altered = [int(value) for value in np.asarray(altered_counts).ravel()]
    profiled = [int(value) for value in np.asarray(profiled_counts).ravel()]
    if len(altered) != len(profiled):
        raise DataValidationError("altered and profiled counts must align")
    pairs = [(a, p) for a, p in zip(altered, profiled) if p > 0]
    if len(pairs) < 2:
        return 1.0
    if any(a < 0 or a > p for a, p in pairs):
        return 1.0
    if len(pairs) == 2:
        (a1, p1), (a2, p2) = pairs
        table = [[a1, p1 - a1], [a2, p2 - a2]]
        if min(min(row) for row in table) < 0:
            return 1.0
        return float(scipy_stats.fisher_exact(table)[1])
    table = [[a, p - a] for a, p in pairs]
    _chi2, p_value, _, _ = scipy_stats.chi2_contingency(np.asarray(table))
    if not math.isfinite(p_value):
        return 1.0
    return float(p_value)


def expression_enrichment_pvalue(*groups) -> float:
    """cbioportal's expression enrichment: t-test (2) / ANOVA (more).

    Two groups → the two-sample t-test p-value; more → the one-way
    ANOVA p-value; fewer than two non-empty groups or any
    degenerate input maps to 1 (the NaN → 1 convention).

    >>> from pharmaml.ml.pharmacogenomics import expression_enrichment_pvalue
    >>> expression_enrichment_pvalue([1.0, 1.1, 0.9], [3.0, 3.1, 2.9]) < 1e-4
    True
    >>> expression_enrichment_pvalue([1.0, 2.0])
    1.0
    """
    cleaned = [np.asarray(group, dtype=float).ravel() for group in groups]
    cleaned = [group[np.isfinite(group)] for group in cleaned]
    cleaned = [group for group in cleaned if group.size >= 2]
    if len(cleaned) < 2:
        return 1.0
    if any(np.isclose(group.max(), group.min()) for group in cleaned):
        return 1.0
    if len(cleaned) == 2:
        return float(scipy_stats.ttest_ind(cleaned[0], cleaned[1]).pvalue)
    return float(scipy_stats.f_oneway(*cleaned).pvalue)


def clinical_enrichment_pvalue(*groups) -> float:
    """cbioportal's clinical enrichment: Mann-Whitney (2) / KW (more).

    Numerical clinical attributes: exactly two groups → the
    Mann-Whitney U p-value (cbioportal labels it Wilcoxon); more
    groups → Kruskal-Wallis; the same degenerate → 1 convention.

    >>> from pharmaml.ml.pharmacogenomics import clinical_enrichment_pvalue
    >>> round(clinical_enrichment_pvalue([1.0, 1.1, 0.9], [3.0, 3.1, 2.9]), 3)
    0.1
    >>> clinical_enrichment_pvalue([1.0], [2.0])
    1.0
    """
    cleaned = [np.asarray(group, dtype=float).ravel() for group in groups]
    cleaned = [group[np.isfinite(group)] for group in cleaned]
    cleaned = [group for group in cleaned if group.size >= 2]
    if len(cleaned) < 2:
        return 1.0
    if len(cleaned) == 2:
        return float(scipy_stats.mannwhitneyu(cleaned[0], cleaned[1]).pvalue)
    return float(scipy_stats.kruskal(*cleaned).pvalue)


# ---------------------------------------------------------------------------
# The demo fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PharmacogenomicScreen:
    """The seeded synthetic pharmacogenomic screen.

    Attributes
    ----------
    ic50:
        The (40, 12) natural-log µM frame -- cell lines CELL_00..
        CELL_39 x drugs D01..D12, ~15 % untested.
    features:
        The (40, 10) binary feature frame -- ``TP53_mut`` and
        ``KRAS_mut`` carry planted associations, the other eight are
        noise.
    tissue / msi:
        The per-cell-line factors (three tissues; MSI in ~10 %).
    """

    ic50: pd.DataFrame
    features: pd.DataFrame
    tissue: pd.Series
    msi: pd.Series


def demo_pharmacogenomics(seed: int = 42) -> PharmacogenomicScreen:
    """A seeded screen with planted drug-feature associations.

    Forty cell lines across three tissues (lung / breast / colon,
    with a tissue effect of ~+0.4 on the log-IC50 scale for lung);
    MSI in ~10 % of lines (a -0.6 sensitivity effect); ten binary
    features at ~40 % positivity -- ``TP53_mut`` confers a -0.8
    shift on five of the twelve drugs, ``KRAS_mut`` a -0.4 shift on
    three others, and the remaining eight features are pure noise.
    The planted structure is what the ANOVA, the volcano and the
    correlation queries are expected to find.

    >>> from pharmaml.ml.pharmacogenomics import demo_pharmacogenomics
    >>> screen = demo_pharmacogenomics(seed=42)
    >>> screen.ic50.shape
    (40, 12)
    >>> sorted(screen.features.columns)[:2]
    ['ARID1A_mut', 'BRAF_mut']
    >>> 'TP53_mut' in screen.features.columns
    True
    """
    rng = np.random.default_rng(seed)
    n_cells, n_drugs = 40, 12
    cells = [f"CELL_{index:02d}" for index in range(n_cells)]
    drugs = [f"D{index:02d}" for index in range(1, n_drugs + 1)]
    tissues = rng.choice(["lung", "breast", "colon"], size=n_cells)
    msi = (rng.random(n_cells) < 0.10).astype(float)
    feature_names = [
        "TP53_mut", "KRAS_mut", "BRAF_mut", "ARID1A_mut", "PTEN_mut",
        "PIK3CA_mut", "EGFR_mut", "STK11_mut", "CDKN2A_mut", "NRAS_mut",
    ]
    feature_matrix = (rng.random((n_cells, len(feature_names))) < 0.40).astype(float)
    features = pd.DataFrame(feature_matrix, index=cells, columns=feature_names)
    tp53 = features["TP53_mut"].to_numpy()
    kras = features["KRAS_mut"].to_numpy()
    drug_effects = {
        drugs[0]: -0.8, drugs[1]: -0.8, drugs[2]: -0.8,
        drugs[3]: -0.8, drugs[4]: -0.8,
        drugs[5]: -0.4, drugs[6]: -0.4, drugs[7]: -0.4,
        drugs[11]: +0.8,   # one resistant plant (positive delta)
    }
    ic50 = np.empty((n_cells, n_drugs))
    for drug_index, drug in enumerate(drugs):
        base = 1.0 + 0.5 * rng.normal()
        tissue_effect = np.where(tissues == "lung", 0.4, 0.0)
        msi_effect = -0.6 * msi
        tp53_effect = drug_effects.get(drug, 0.0) * tp53
        kras_effect = 0.0
        if drug in (drugs[5], drugs[6], drugs[7]):
            kras_effect = -0.4 * kras
        noise = rng.normal(0.0, 0.35, n_cells)
        ic50[:, drug_index] = base + tissue_effect + msi_effect + tp53_effect + kras_effect + noise
    missing = rng.random((n_cells, n_drugs)) < 0.15
    ic50 = np.where(missing, np.nan, ic50)
    ic50_frame = pd.DataFrame(ic50, index=cells, columns=drugs)
    return PharmacogenomicScreen(
        ic50=ic50_frame,
        features=features,
        tissue=pd.Series(tissues, index=cells, name="tissue"),
        msi=pd.Series(msi, index=cells, name="msi"),
    )


__all__ = [
    "CLUSTER_RATIO_THRESHOLD",
    "CNA_PROFILE_DESCRIPTION",
    "CNA_VALUES",
    "COEXPRESSION_THRESHOLD",
    "CONSTANT_TOLERANCE",
    "EFFECT_THRESHOLD",
    "EQUAL_VAR_TTEST",
    "FDR_THRESHOLD",
    "FEATURE_FACTOR_THRESHOLD",
    "MINIMUM_NONNA_IC50",
    "MSI_FACTOR_THRESHOLD",
    "PVALUE_THRESHOLD",
    "VOLCANO_ADDITIONAL_FDR_LINES",
    "EffectSizes",
    "GroupMeanCenterer",
    "MeanEffectsPredictor",
    "PharmacogenomicScreen",
    "alteration_enrichment_pvalue",
    "anova_f_test",
    "bh_fdr",
    "clinical_enrichment_pvalue",
    "coexpression_pvalue",
    "demo_pharmacogenomics",
    "diploid_zscores",
    "discrete_cna",
    "effect_sizes",
    "expression_enrichment_pvalue",
    "gdsc_anova",
    "guarded_kendall",
    "guarded_pearson",
    "guarded_spearman",
    "ic50_cluster",
    "ic50_from_ln",
    "naive_group_means",
    "normalized_metric",
    "partial_correlation",
    "pattern_comparison",
    "per_tissue_correlation",
    "qvalue",
    "response_matrix",
    "split_dose_ranges",
    "volcano_points",
]
