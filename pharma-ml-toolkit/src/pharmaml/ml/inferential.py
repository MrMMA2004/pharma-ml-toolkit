"""Inferential statistics scipy never grew -- the pingouin gaps.

``raphaelvallat/pingouin`` wraps R's favourite tests in a DataFrame
API, and most of what it adds over scipy/statsmodels is a specific,
formulable list.  This module absorbs the pieces the toolkit's
existing stats surfaces (t-tests, MWU, normality, rank correlations in
:mod:`pharmaml.ml.stats`; Type II ANOVA, OLS/GLM in
:mod:`pharmaml.ml.formula`) genuinely lack:

* :func:`rm_anova` -- one-way **repeated-measures** ANOVA (the
  subject × within-factor SS decomposition), with Mauchly-adjacent
  sphericity checking via the **Greenhouse-Geisser ε** correction.
* :func:`welch_anova` -- the Welch F on weighted group means (no
  homoscedasticity assumed), which scipy does not ship.
* :func:`games_howell` / :func:`tukey_hsd` -- the post-hoc pairs with
  the studentized-range distribution.
* :func:`bayesfactor_ttest` -- Rouder's JZS Bayes factor (r = 0.707
  default), the number a Bayesian t-test report leads with.
* :func:`circ_mean`, :func:`circ_rayleigh`, :func:`circ_corrcc` --
  the circular-statistics trio (scipy has none of it).
* :func:`partial_corr` -- correlation conditioned on covariates via
  the precision-matrix identity.
* :func:`cronbach_alpha` -- reliability of a multi-item scale.

The honest scope, stated once: mixed two-way designs, the full Mauchly
W statistic, mediation, ICC variants, robust (Shepherd's π / percentage
bend) correlations and mpmath-backed one-sided correlation Bayes
factors are out of scope; pingouin's own omega-squared is a comment in
its source, not code -- the effect sizes here are the ones it ships
(partial η², η²).

Examples
--------
>>> import pandas as pd, numpy as np
>>> from pharmaml.ml.inferential import rm_anova, welch_anova, games_howell
>>> from pharmaml.ml.inferential import bayesfactor_ttest, cronbach_alpha
>>> rm = pd.DataFrame({
...     "subject": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
...     "time":    ["t0", "t1", "t2"] * 4,
...     "score":   [8.0, 9.5, 11.0, 7.0, 7.5, 10.0, 9.0, 10.5, 12.0, 6.0, 7.0, 8.5],
... })
>>> table = rm_anova(rm, dv="score", within="time", subject="subject")
>>> round(float(table.loc[0, "F"]), 2)
93.0
>>> groups = [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [2.0, 3.0, 4.5]]
>>> round(float(welch_anova(groups)["F"].iloc[0]), 2)
12.63
>>> round(bayesfactor_ttest(3.0, 20), 2)
5.59
>>> alpha = cronbach_alpha(pd.DataFrame({
...     "i1": [3.0, 4.0, 4.0, 2.0], "i2": [3.0, 5.0, 4.0, 1.0],
...     "i3": [2.0, 4.0, 5.0, 2.0],
... }))
>>> round(alpha, 3)
0.914
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import integrate
from scipy import stats as sps

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

__all__ = [
    "bayesfactor_ttest",
    "circ_corrcc",
    "circ_mean",
    "circ_rayleigh",
    "cronbach_alpha",
    "games_howell",
    "partial_corr",
    "rm_anova",
    "tukey_hsd",
    "welch_anova",
]

_log = get_logger(__name__)


def _check_frames(**frames: pd.DataFrame | None) -> None:
    pass  # placeholder for symmetric validation if ever needed


# --------------------------------------------------------------------------- #
# Repeated-measures ANOVA (+ Greenhouse-Geisser)
# --------------------------------------------------------------------------- #
def rm_anova(
    data: pd.DataFrame,
    *,
    dv: str,
    within: str,
    subject: str,
    correction: str = "auto",
) -> pd.DataFrame:
    """One-way repeated-measures ANOVA with the GG ε correction.

    The decomposition: ``SS_effect = r·Σ(Ȳ_level − Ȳ)²``,
    ``SS_subject = r_levels... Σ(Ȳ_subject − Ȳ)²`` (scaled by the
    number of levels), ``SS_error = SS_total − SS_effect − SS_subject``;
    ``F = MS_effect / MS_error`` on ``df = (r−1, (n−1)(r−1))``.
    Sphericity is checked through the Greenhouse-Geisser ε
    (``ε = tr(M)²/(d·tr(M²))`` on the pooled within-subject
    covariance); with ``correction="auto"`` the corrected p is used
    when ``ε < 1`` and there are at least 3 levels (pingouin's rule
    minus the full Mauchly W).

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.ml.inferential import rm_anova
    >>> rm = pd.DataFrame({
    ...     "subject": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    ...     "time":    ["t0", "t1", "t2"] * 4,
    ...     "score":   [8.0, 9.5, 11.0, 7.0, 7.5, 10.0, 9.0, 10.5, 12.0, 6.0, 7.0, 8.5],
    ... })
    >>> table = rm_anova(rm, dv="score", within="time", subject="subject")
    >>> list(table.columns[:4])
    ['Source', 'ddof1', 'ddof2', 'F']
    >>> bool(table.loc[0, "p-unc"] < 0.01)
    True
    >>> round(float(table.loc[0, "ng2"]), 3)
    0.314
    """
    for column in (dv, within, subject):
        if column not in data.columns:
            raise DataValidationError(f"column {column!r} not in the data")
    wide = data.pivot(index=subject, columns=within, values=dv).dropna()
    n, r = wide.shape
    if n < 2 or r < 2:
        raise DataValidationError("rm_anova needs >= 2 subjects and >= 2 levels")
    values = wide.to_numpy(dtype=float)
    grand = values.mean()
    level_means = values.mean(axis=0)
    subject_means = values.mean(axis=1)
    ss_effect = n * float(((level_means - grand) ** 2).sum())
    ss_subject = r * float(((subject_means - grand) ** 2).sum())
    ss_total = float(((values - grand) ** 2).sum())
    ss_error = ss_total - ss_effect - ss_subject
    ddof1, ddof2 = r - 1, (n - 1) * (r - 1)
    ms_effect = ss_effect / ddof1
    ms_error = ss_error / max(ddof2, 1)
    f_value = ms_effect / ms_error if ms_error > 0 else float("inf")
    p_value = float(sps.f.sf(f_value, ddof1, ddof2))

    # Greenhouse-Geisser epsilon from the within-subject covariance
    within_residuals = values - level_means[None, :]
    cov = np.cov(within_residuals, rowvar=False, ddof=1) if n > 1 else np.zeros((r, r))
    d = r - 1
    if d >= 1:
        trace = float(np.trace(cov))
        trace_sq = float(np.trace(cov @ cov))
        epsilon = (trace ** 2) / (d * trace_sq) if trace_sq > 0 else 1.0
    else:
        epsilon = 1.0
    epsilon = float(min(epsilon, 1.0))
    if correction == "auto":
        use_gg = epsilon < 1.0 and r >= 3
    elif correction == "gg":
        use_gg = True
    elif correction == "none":
        use_gg = False
    else:
        raise DataValidationError("correction must be 'auto', 'gg' or 'none'")
    if use_gg:
        ddof1_c = max(epsilon * ddof1, 1.0)
        ddof2_c = max(epsilon * ddof2, 1.0)
        p_value = float(sps.f.sf(f_value, ddof1_c, ddof2_c))
    else:
        ddof1_c, ddof2_c = float(ddof1), float(ddof2)

    ng2 = ss_effect / (ss_effect + ss_total) if (ss_effect + ss_total) > 0 else 0.0
    return pd.DataFrame([{
        "Source": within,
        "ddof1": ddof1,
        "ddof2": ddof2,
        "F": f_value,
        "p-unc": p_value,
        "GG-epsilon": epsilon,
        "ddof1-corr": ddof1_c,
        "ddof2-corr": ddof2_c,
        "ng2": ng2,
        "p-GG-corr": p_value if use_gg else float("nan"),
    }])


# --------------------------------------------------------------------------- #
# Welch ANOVA + post-hocs
# --------------------------------------------------------------------------- #
def welch_anova(groups: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Welch's one-way ANOVA (unequal variances, unbalanced n).

    Weighted means ``ȳ_w = Σwᵢȳᵢ/Σwᵢ`` with ``wᵢ = nᵢ/sᵢ²``;
    ``Λ = 3Σ((1/(nᵢ−1))(1−wᵢ/Σw)²)/(r²−1)`` and
    ``F = (SS_eff/(r−1))/(1 + 2Λ(r−2)/3)`` on ``df = (r−1, 1/Λ)``
    -- pingouin's exact assembly of the Welch 1951 statistic.

    Examples
    --------
    >>> from pharmaml.ml.inferential import welch_anova
    >>> table = welch_anova([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> round(float(table["F"].iloc[0]), 1)
    13.5
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    if len(arrays) < 2 or any(a.size < 2 for a in arrays):
        raise DataValidationError("welch_anova needs >= 2 groups of >= 2 values")
    r = len(arrays)
    n = np.array([a.size for a in arrays])
    means = np.array([a.mean() for a in arrays])
    variances = np.array([a.var(ddof=1) for a in arrays])
    weights = n / variances
    total_w = weights.sum()
    weighted_mean = float((weights * means).sum() / total_w)
    ss_effect = float((weights * (means - weighted_mean) ** 2).sum())
    lam = 3.0 * float(
        ((1.0 / (n - 1)) * (1.0 - weights / total_w) ** 2).sum()
    ) / (r * r - 1)
    f_value = (ss_effect / (r - 1)) / (1.0 + 2.0 * lam * (r - 2) / 3.0)
    ddof2 = 1.0 / lam if lam > 0 else float("inf")
    p_value = float(sps.f.sf(f_value, r - 1, ddof2))
    return pd.DataFrame([{
        "Source": "welch",
        "ddof1": r - 1,
        "ddof2": ddof2,
        "F": f_value,
        "p-unc": p_value,
    }])


def _pairwise_studentized(
    groups: Sequence[Sequence[float]],
    method: str,
) -> pd.DataFrame:
    arrays = [np.asarray(g, dtype=float) for g in groups]
    if len(arrays) < 2:
        raise DataValidationError("need >= 2 groups")
    r = len(arrays)
    n = np.array([a.size for a in arrays])
    N = int(n.sum())
    means = np.array([a.mean() for a in arrays])
    variances = np.array([a.var(ddof=1) for a in arrays])
    rows: list[dict[str, float | str]] = []
    for i in range(r):
        for j in range(i + 1, r):
            if method == "tukey":
                ms_within = float(
                    sum((n[k] - 1) * variances[k] for k in range(r))
                ) / max(N - r, 1)
                se = math.sqrt(ms_within / n[i] + ms_within / n[j])
                ddof = N - r
            else:
                se_sq = variances[i] / n[i] + variances[j] / n[j]
                se = math.sqrt(se_sq)
                ddof = se_sq ** 2 / (
                    (variances[i] / n[i]) ** 2 / (n[i] - 1)
                    + (variances[j] / n[j]) ** 2 / (n[j] - 1)
                )
            diff = float(means[i] - means[j])
            t_stat = diff / se if se > 0 else 0.0
            p_value = float(
                sps.studentized_range.sf(math.sqrt(2.0) * abs(t_stat), r, ddof)
            )
            rows.append({
                "A": f"g{i}", "B": f"g{j}", "mean(A)": float(means[i]),
                "mean(B)": float(means[j]), "diff": diff,
                "se": se, "T": t_stat, "p-tukey": p_value,
            })
    return pd.DataFrame(rows)


def tukey_hsd(groups: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Tukey HSD pairwise comparisons (equal-variance studentized range).

    Examples
    --------
    >>> from pharmaml.ml.inferential import tukey_hsd
    >>> table = tukey_hsd([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 2.0]])
    >>> len(table)
    3
    >>> bool((table["p-tukey"] < 0.05).any())
    True
    """
    return _pairwise_studentized(groups, "tukey")


def games_howell(groups: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Games-Howell pairwise comparisons (unequal variances).

    Welch-Satterthwaite degrees of freedom per pair; the p-value from
    the studentized-range distribution -- the post-hoc that pairs with
    :func:`welch_anova`.

    Examples
    --------
    >>> from pharmaml.ml.inferential import games_howell
    >>> table = games_howell([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 2.0]])
    >>> len(table)
    3
    """
    return _pairwise_studentized(groups, "games-howell")


# --------------------------------------------------------------------------- #
# Bayes factors
# --------------------------------------------------------------------------- #
def bayesfactor_ttest(
    t: float,
    df: float,
    *,
    r: float = 0.707,
    paired: bool = False,
) -> float:
    """Rouder's JZS Bayes factor BF₁₀ for a t statistic.

    The Cauchy prior scale is ``r`` (0.707 = √2/2, the "medium" default);
    ``t`` and ``df`` are the classical t statistic and its degrees of
    freedom.  BF₁₀ > 1 favours the alternative; the integral is
    evaluated with adaptive quadrature exactly as pingouin does it.

    Examples
    --------
    >>> from pharmaml.ml.inferential import bayesfactor_ttest
    >>> 0.0 < bayesfactor_ttest(0.0, 20) < 1   # t=0: the Occam factor
    True
    >>> round(bayesfactor_ttest(2.5, 20), 2)
    2.78
    """
    if df <= 0:
        raise DataValidationError("df must be positive")
    v = float(df)

    def posterior(g: float) -> float:
        if g <= 0:
            return 0.0
        lik_alt = (1.0 + t * t / (v * (1.0 + g))) ** (-(v + 1.0) / 2.0)
        prior = (2.0 * math.pi) ** -0.5 * g ** -1.5 * math.exp(-1.0 / (2.0 * g))
        return (1.0 + g) ** -0.5 * lik_alt * prior

    evidence_alt, _ = integrate.quad(posterior, 0.0, np.inf)
    evidence_null = (1.0 + t * t / v) ** (-(v + 1.0) / 2.0)
    if evidence_null <= 0:
        return float("inf")
    return float(evidence_alt / evidence_null)


# --------------------------------------------------------------------------- #
# Partial correlation
# --------------------------------------------------------------------------- #
def partial_corr(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    covar: str | Sequence[str],
    method: str = "pearson",
) -> float:
    """Correlation of ``x`` and ``y`` controlling for ``covar``.

    The precision-matrix identity: partial correlation is
    ``−D·V⁻¹·D`` off the diagonal, with ``D = diag(1/√diag(V⁻¹))``
    over the covariance (ranks for Spearman) of the involved columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.ml.inferential import partial_corr
    >>> rng = pd.DataFrame({"z": [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> frame = pd.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "y": [1.2, 2.1, 2.8, 4.2, 5.1],
    ...     "z": [1.0, 1.5, 2.0, 2.5, 3.0],
    ... })
    >>> round(partial_corr(frame, x="x", y="y", covar="z"), 3)
    0.995
    """
    covars = [covar] if isinstance(covar, str) else list(covar)
    columns = [x, y] + covars
    for column in columns:
        if column not in data.columns:
            raise DataValidationError(f"column {column!r} not in the data")
    block = data[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(block) < len(columns) + 2:
        raise DataValidationError("partial_corr needs more rows than variables")
    values = block.to_numpy(dtype=float)
    if method == "spearman":
        values = np.apply_along_axis(sps.rankdata, 0, values)
    elif method != "pearson":
        raise DataValidationError("method must be 'pearson' or 'spearman'")
    cov = np.cov(values, rowvar=False)
    precision = np.linalg.pinv(cov)
    scale = np.diag(1.0 / np.sqrt(np.diag(precision)))
    partial = -scale @ precision @ scale
    return float(partial[0, 1])


# --------------------------------------------------------------------------- #
# Circular statistics
# --------------------------------------------------------------------------- #
def circ_mean(angles: Sequence[float]) -> float:
    """Circular mean (the atan2 of the resultant's components).

    Examples
    --------
    >>> from pharmaml.ml.inferential import circ_mean
    >>> round(circ_mean([0.0, 0.2, 6.1]), 3)   # 6.1 rad ≈ -0.183
    0.006
    """
    alpha = np.asarray(angles, dtype=float)
    if alpha.size == 0:
        raise DataValidationError("circ_mean needs >= 1 angle")
    return float(math.atan2(np.sin(alpha).sum(), np.cos(alpha).sum()))


def circ_rayleigh(
    angles: Sequence[float], *, directed: bool = False
) -> tuple[float, float]:
    """Rayleigh test: ``(R_bar, p)`` for uniformity of directions.

    ``r̄ = |Σe^{iα}|/n``; for the non-directed test the statistic is
    ``z = n·r̄²`` with Zar's p approximation
    ``exp(√(1 + 4n + 4(n² − R²)) − (1 + 2n))`` where ``R = n·r̄``.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.inferential import circ_rayleigh
    >>> rng = np.random.default_rng(42)
    >>> r_bar, p_uniform = circ_rayleigh(rng.uniform(0, 2*np.pi, 200))
    >>> bool(p_uniform > 0.05)
    True
    >>> r_bar_clumped, p_clumped = circ_rayleigh([0.1, 0.2, 0.15, 6.2, 0.05])
    >>> bool(p_clumped < 0.01)
    True
    """
    alpha = np.asarray(angles, dtype=float)
    n = alpha.size
    if n < 3:
        raise DataValidationError("circ_rayleigh needs >= 3 angles")
    r_bar = float(
        math.hypot(np.sin(alpha).sum(), np.cos(alpha).sum()) / n
    )
    if directed:
        r_stat = r_bar
        z = r_stat * r_stat
    else:
        R = n * r_bar
        z = R * R / n
        root = math.sqrt(max(1.0 + 4.0 * n + 4.0 * (n * n - R * R), 0.0))
        p_value = float(math.exp(root - (1.0 + 2.0 * n)))
        return r_bar, min(max(p_value, 0.0), 1.0)
    p_value = float(math.exp(-z))
    return r_bar, min(max(p_value, 0.0), 1.0)


def circ_corrcc(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Circular-circular correlation (Fisher & Lee 1983) with its p.

    ``r = Σsin(x−x̄)·sin(y−ȳ)/√(Σsin²(x−x̄)·Σsin²(y−ȳ))``; the test
    statistic ``t = r·√(n·mean(sin²)·mean(sin²)/mean(sin·sin))`` on
    ``df = n − 2``.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.inferential import circ_corrcc
    >>> theta = np.linspace(0.0, 4.0, 30)
    >>> y = theta + 0.05 * np.sin(3.0 * theta)
    >>> r, p = circ_corrcc(theta, y)
    >>> round(r, 3)
    1.0
    >>> round(p, 4)
    0.0002
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape or xa.size < 4:
        raise DataValidationError("circ_corrcc needs equal-length angle arrays")
    mx, my = circ_mean(xa.tolist()), circ_mean(ya.tolist())
    sx = np.sin(xa - mx)
    sy = np.sin(ya - my)
    denom = math.sqrt(float((sx * sx).sum()) * float((sy * sy).sum()))
    if denom <= 0:
        raise DataValidationError("circular variance is zero on one side")
    r_value = float((sx * sy).sum()) / denom
    n = xa.size
    mean_ss = float(np.mean(sx * sx)) * float(np.mean(sy * sy))
    mean_cross = abs(float(np.mean(sx * sy)))
    if mean_cross <= 0:
        return r_value, 1.0
    t_stat = r_value * math.sqrt(n * mean_ss / mean_cross)
    p_value = 2.0 * float(sps.t.sf(abs(t_stat), n - 2))
    return r_value, float(min(max(p_value, 0.0), 1.0))


# --------------------------------------------------------------------------- #
# Reliability
# --------------------------------------------------------------------------- #
def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's α for a multi-item scale.

    ``α = k/(k−1) · (1 − tr(C)/ΣΣC)`` over the item covariance matrix
    ``C`` -- the reliability a questionnaire round reports first.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.ml.inferential import cronbach_alpha
    >>> alpha = cronbach_alpha(pd.DataFrame({
    ...     "i1": [3.0, 4.0, 4.0, 2.0], "i2": [3.0, 5.0, 4.0, 1.0],
    ...     "i3": [2.0, 4.0, 5.0, 2.0],
    ... }))
    >>> round(alpha, 3)
    0.914
    """
    values = items.apply(pd.to_numeric, errors="coerce").dropna()
    k = values.shape[1]
    if k < 2 or values.shape[0] < 2:
        raise DataValidationError("cronbach_alpha needs >= 2 items and >= 2 rows")
    cov = np.cov(values.to_numpy(dtype=float), rowvar=False, ddof=1)
    trace = float(np.trace(cov))
    total = float(cov.sum())
    if total <= 0:
        raise DataValidationError("degenerate item covariance")
    return float(k / (k - 1.0) * (1.0 - trace / total))
