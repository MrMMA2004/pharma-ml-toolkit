"""The Pharmacometrics_Portfolio absorption: popPK diagnostics in NumPy.

``dcmgithb/Pharmacometrics_Portfolio`` (reviewed file by file — the
instruction of this round) is a Quarto portfolio of five simulated
analyses around nlmixr2/rxode2; its *estimation* engines (SAEM,
FOCE-I) are exactly what this toolkit does not vendor — the honest
boundary, the same stance as the MCMC and torch precedents.  What
transfers natively is the entire diagnostic and simulation layer the
portfolio's ``R/helpers.R`` and ``python/`` carry, and it lands here:

* the **shrinkage pair** — :func:`eta_shrinkage`
  (``1 − SD(η)/ω``) and :func:`epsilon_shrinkage`
  (``1 − SD(IWRES)``) with the EMA ~25 % / FDA ~30 % interpretation
  bands the portfolio colour-codes;
* the **condition number** — :func:`condition_number`
  (``sqrt(λ_max/λ_min)`` of the parameter covariance, the > 1000
  near-singularity flag);
* the **GOF quantities** — :func:`log_residuals` and
  :func:`gof_summary` (RMSE-log and MPE %, the portfolio's project-04
  numbers), plus :func:`aic_bic` for exposure-metric selection
  (AUC vs Cmax vs Cavg — the project-02 story);
* the **VPC layer** — :func:`vpc_percentiles` (time-bin 5/50/95
  quantiles, the portfolio's manual fallback recipe: quantile cut
  points, per-bin quantiles) and :func:`pcvpc_correction` (divide
  each observation by its bin's median population prediction — the
  prediction-corrected VPC);
* the **bootstrap layer** — :func:`bootstrap_case_resample`
  (subject-level resampling with replacement, the portfolio's
  ``bootstrap_nlmixr`` stripped of the refit) and
  :func:`bootstrap_summary` (percentile CIs, the EMA/FDA guidance
  the portfolio cites);
* the **sensitivity layer** — :func:`one_at_a_time` (multiplicative
  factors × any callable → the tornado table of project 04);
* the **trial-simulation operating characteristics** —
  :func:`power_with_ci` (power with the binomial Monte-Carlo CI
  ``p ± 1.96·sqrt(p(1−p)/n_rep)``) and
  :func:`operating_characteristics` (bias, RMSE, the ±30 %
  coverage of project 05).

Every function is deterministic (seeded where randomness enters) and
carries doctests pinning the portfolio's own arithmetic.

Examples
--------
>>> from pharmaml.pharma.poppk import eta_shrinkage, condition_number
>>> round(float(eta_shrinkage([[0.1], [-0.1]] * 5, [0.2])), 3)   # SD(eta)=0.105
0.473
>>> round(condition_number([[2.0, 0.0], [0.0, 0.5]]), 3)
2.0
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "aic_bic",
    "bootstrap_case_resample",
    "bootstrap_summary",
    "condition_number",
    "epsilon_shrinkage",
    "eta_shrinkage",
    "gof_summary",
    "log_residuals",
    "one_at_a_time",
    "operating_characteristics",
    "pcvpc_correction",
    "power_with_ci",
    "vpc_percentiles",
]


# ---------------------------------------------------------------------------
# the shrinkage + conditioning pair
# ---------------------------------------------------------------------------


def eta_shrinkage(etas: Sequence[Sequence[float]] | np.ndarray, omega: Sequence[float]) -> np.ndarray:
    """Per-effect η-shrinkage: ``1 − SD(η_i) / ω_i``.

    The portfolio's ``shrinkage_table``: high shrinkage means the data
    carry little information about that random effect (EMA guidance
    flags ~25 %, FDA ~30 %).  ``etas`` is one row per subject (or per
    observation) with one column per effect; ``omega`` the
    random-effect standard deviations the model assumed.

    Examples
    --------
    >>> import numpy as np
    >>> etas = np.tile([[0.1, -0.2], [-0.1, 0.2]], (10, 1))   # 20 rows
    >>> np.round(eta_shrinkage(etas, [0.2, 0.4]), 3).tolist()
    [0.487, 0.487]
    """

    matrix = np.asarray(etas, dtype=float)
    omegas = np.asarray(omega, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != omegas.size:
        raise DataValidationError("eta_shrinkage: etas must be (n, effects) matching omega")
    if matrix.shape[0] < 2:
        raise DataValidationError("eta_shrinkage: need at least two eta rows")
    if np.any(omegas <= 0):
        raise DataValidationError("eta_shrinkage: omega entries must be positive")
    return np.clip(1.0 - matrix.std(axis=0, ddof=1) / omegas, 0.0, 1.0)


def epsilon_shrinkage(iwres: Sequence[float] | np.ndarray) -> float:
    """ε-shrinkage: ``1 − SD(IWRES)``.

    The residual shrinkage the portfolio computes on individual
    weighted residuals (a correctly specified model has IWRES ~ N(0,
    1), so SD < 1 means the residuals shrank towards their mean).

    Examples
    --------
    >>> import numpy as np
    >>> iwres = np.linspace(-0.8, 0.8, 41)     # SD ~ 0.486
    >>> 0.45 < float(epsilon_shrinkage(iwres)) < 0.55
    True
    """

    values = np.asarray(iwres, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise DataValidationError("epsilon_shrinkage: need a 1-D vector of residuals")
    return float(np.clip(1.0 - values.std(ddof=1), 0.0, 1.0))


def condition_number(covariance: Sequence[Sequence[float]] | np.ndarray) -> float:
    """Square-root condition number of a parameter covariance matrix.

    ``sqrt(λ_max / λ_min)`` over the positive eigenvalues — the
    portfolio's near-singularity flag (> 1000 warns that the
    parameter estimates are poorly determined).

    Examples
    --------
    >>> round(condition_number([[4.0, 0.0], [0.0, 1.0]]), 3)
    2.0
    """

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise DataValidationError("condition_number: need a square covariance matrix")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    positive = eigenvalues[eigenvalues > 1e-12]
    if positive.size == 0:
        raise DataValidationError("condition_number: no positive eigenvalues")
    return float(np.sqrt(positive.max() / positive.min()))


# ---------------------------------------------------------------------------
# the GOF quantities
# ---------------------------------------------------------------------------


def log_residuals(observed: Sequence[float], predicted: Sequence[float]) -> np.ndarray:
    """``log(obs) − log(pred)`` per point (the portfolio's P04 residuals).

    Examples
    --------
    >>> log_residuals([1.0, math.e], [math.e, 1.0]).round(3).tolist()
    [-1.0, 1.0]
    """

    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape or obs.size == 0:
        raise DataValidationError("log_residuals: arrays must align and be non-empty")
    if np.any(obs <= 0) or np.any(pred <= 0):
        raise DataValidationError("log_residuals: values must be positive")
    return np.log(obs) - np.log(pred)


def gof_summary(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    """RMSE(log) and MPE % — the portfolio's P04 goodness-of-fit stats.

    ``RMSE_log = sqrt(mean(residual²))`` on log residuals;
    ``MPE = mean((pred − obs)/obs · 100)`` (mean prediction error,
    percent).

    Examples
    --------
    >>> stats = gof_summary([1.0, 2.0], [1.1, 1.9])
    >>> round(stats["rmse_log"], 4), round(stats["mpe_percent"], 2)
    (0.0765, 2.5)
    """

    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape or obs.size == 0:
        raise DataValidationError("gof_summary: arrays must align and be non-empty")
    if np.any(obs <= 0) or np.any(pred <= 0):
        raise DataValidationError("gof_summary: values must be positive")
    residuals = np.log(obs) - np.log(pred)
    return {
        "n": float(obs.size),
        "rmse_log": float(np.sqrt(np.mean(residuals**2))),
        "mpe_percent": float(np.mean((pred - obs) / obs) * 100.0),
    }


def aic_bic(rss: float, n: int, k: int) -> dict[str, float]:
    """AIC and BIC from a residual sum of squares (metric selection).

    ``AIC = n·ln(RSS/n) + 2k``, ``BIC = n·ln(RSS/n) + k·ln(n)`` — the
    portfolio's project-02 comparison of AUC / Cmax / Cavg-driven PD
    models picks the lowest-AIC exposure metric.

    Examples
    --------
    >>> scores = aic_bic(2.5, 80, 3)
    >>> round(scores["aic"], 2), round(scores["bic"], 2)
    (-271.26, -264.11)
    """

    if n < 1 or k < 0:
        raise DataValidationError("aic_bic: n >= 1 and k >= 0 required")
    if rss <= 0:
        raise DataValidationError(f"aic_bic: rss must be positive, got {rss}")
    log_term = n * math.log(rss / n)
    return {"aic": log_term + 2.0 * k, "bic": log_term + k * math.log(n)}


# ---------------------------------------------------------------------------
# the VPC layer
# ---------------------------------------------------------------------------


def vpc_percentiles(
    frame: pd.DataFrame,
    time: str = "time",
    value: str = "dv",
    bins: int = 10,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> pd.DataFrame:
    """Observed percentile profile over quantile time bins.

    The portfolio's manual VPC fallback, verbatim in pandas: cut the
    time axis at its own ``bins``-quantile break points, then report
    each bin's requested quantiles of the dependent variable with the
    bin's midpoint time — the ribbon a VPC plot draws.

    Examples
    --------
    >>> frame = pd.DataFrame({"time": [0.0, 1.0, 2.0, 3.0, 4.0] * 4,
    ...                       "dv": [1.0, 2.0, 3.0, 2.0, 1.0] * 4})
    >>> profile = vpc_percentiles(frame, bins=2, quantiles=(0.5, 0.95))
    >>> profile["bin"].tolist()
    [0, 1]
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError("vpc_percentiles: need a non-empty DataFrame")
    if time not in frame.columns or value not in frame.columns:
        raise DataValidationError(f"vpc_percentiles: columns {time!r}/{value!r} missing")
    if bins < 1:
        raise DataValidationError(f"bins must be >= 1, got {bins}")
    breaks = np.quantile(frame[time].to_numpy(dtype=float), np.linspace(0.0, 1.0, bins + 1))
    breaks[0] -= 1e-9
    assignment = np.clip(np.searchsorted(breaks, frame[time].to_numpy(dtype=float), side="right") - 1, 0, bins - 1)
    quantiles = tuple(float(q) for q in quantiles)
    rows = []
    for bin_index in range(bins):
        mask = assignment == bin_index
        if not mask.any():
            continue
        times = frame.loc[mask, time].to_numpy(dtype=float)
        values = frame.loc[mask, value].to_numpy(dtype=float)
        row: dict[str, float] = {
            "bin": bin_index,
            "time_mid": float(0.5 * (times.min() + times.max())),
            "n": int(mask.sum()),
        }
        for q in quantiles:
            row[f"q{q:g}"] = float(np.quantile(values, q))
        rows.append(row)
    return pd.DataFrame(rows)


def pcvpc_correction(
    frame: pd.DataFrame,
    time: str = "time",
    value: str = "dv",
    prediction: str = "pred",
    bins: int = 10,
) -> pd.DataFrame:
    """Prediction-corrected observations for a pcVPC.

    Each observation is divided by its time bin's median *population*
    prediction — the correction that removes covariate/dose effects so
    the percentile ribbon reflects pure variability (the portfolio's
    ``pcvpc`` docstring recipe).  Returns the frame with a
    ``dv_corrected`` column.

    Examples
    --------
    >>> frame = pd.DataFrame({
    ...     "time": [0.0, 1.0, 2.0, 3.0] * 3,
    ...     "dv":   [2.0, 4.0, 6.0, 8.0] * 3,
    ...     "pred": [1.0, 2.0, 3.0, 4.0] * 3,
    ... })
    >>> corrected = pcvpc_correction(frame, bins=2)
    >>> sorted(corrected["dv_corrected"].round(2).unique().tolist())
    [1.33, 1.71, 2.29, 2.67]
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError("pcvpc_correction: need a non-empty DataFrame")
    for column in (time, value, prediction):
        if column not in frame.columns:
            raise DataValidationError(f"pcvpc_correction: column {column!r} missing")
    breaks = np.quantile(frame[time].to_numpy(dtype=float), np.linspace(0.0, 1.0, bins + 1))
    breaks[0] -= 1e-9
    assignment = np.clip(np.searchsorted(breaks, frame[time].to_numpy(dtype=float), side="right") - 1, 0, bins - 1)
    corrected = frame.copy()
    medians = np.empty(len(frame))
    for bin_index in range(bins):
        mask = assignment == bin_index
        if not mask.any():
            continue
        median = float(np.median(frame.loc[mask, prediction].to_numpy(dtype=float)))
        medians[mask] = median
    if np.any(medians <= 0):
        raise DataValidationError("pcvpc_correction: a time bin's median prediction is <= 0")
    corrected["dv_corrected"] = frame[value].to_numpy(dtype=float) / medians
    return corrected


# ---------------------------------------------------------------------------
# the bootstrap layer
# ---------------------------------------------------------------------------


def bootstrap_case_resample(
    subjects: Sequence[int | str],
    n_boot: int = 200,
    seed: int = 123,
) -> list[list[int | str]]:
    """Subject-level case-resampling indices (the bootstrap design).

    Draws ``n_boot`` resamples of the subject list with replacement —
    the portfolio's ``bootstrap_nlmixr`` resampling step (subjects
    may repeat inside a replicate; each replicate has the original
    subject count).

    Examples
    --------
    >>> resamples = bootstrap_case_resample(["a", "b", "c"], n_boot=10, seed=1)
    >>> len(resamples), set(len(r) for r in resamples)
    (10, {3})
    """

    if n_boot < 1:
        raise DataValidationError(f"n_boot must be >= 1, got {n_boot}")
    unique = list(dict.fromkeys(subjects))  # order-stable dedup
    if not unique:
        raise DataValidationError("bootstrap_case_resample: need at least one subject")
    rng = np.random.default_rng(seed)
    resamples: list[list[int | str]] = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(unique), size=len(unique))
        resamples.append([unique[int(index)] for index in picks])
    return resamples


def bootstrap_summary(
    values: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap CI with mean / median / SE.

    The portfolio's ``summarize_bootstrap``: percentile CI at
    ``confidence``, Boot_Mean / Boot_Median, Boot_SE = sd, and the
    replicate count (NaNs skipped, non-converged replicates in the
    portfolio's sense).

    Examples
    --------
    >>> summary = bootstrap_summary([1.0, 1.1, 0.9, 1.2, 1.05])
    >>> round(summary["median"], 3), round(summary["n"], 1)
    (1.05, 5.0)
    """

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise DataValidationError("bootstrap_summary: need at least two finite values")
    if not 0.0 < confidence < 1.0:
        raise DataValidationError(f"confidence must be in (0, 1), got {confidence}")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(array, [alpha, 1.0 - alpha])
    return {
        "n": float(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "se": float(array.std(ddof=1)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


# ---------------------------------------------------------------------------
# the sensitivity layer
# ---------------------------------------------------------------------------


def one_at_a_time(
    base: dict[str, float],
    factors: Sequence[float],
    simulate: Callable[[dict[str, float]], float],
    parameters: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One-at-a-time sensitivity: the tornado table.

    For each parameter (all of ``base`` by default) and each
    multiplicative ``factor``, re-run ``simulate`` on the perturbed
    parameter dict and record the metric — the portfolio's P04
    sensitivity block (AUC or Cmax of a re-solved profile).  The
    tidy table carries ``parameter``, ``factor``, ``value`` and
    ``pct_change`` (against the factor-1.0 run).

    Examples
    --------
    >>> table = one_at_a_time(
    ...     {"cl": 3.0, "v": 30.0}, [0.5, 1.0, 1.5],
    ...     lambda p: p["v"] / p["cl"],          # half-life as the metric
    ...     parameters=["cl"],
    ... )
    >>> table["pct_change"].round(1).tolist()
    [100.0, 0.0, -33.3]
    """

    if not base:
        raise DataValidationError("one_at_a_time: base parameters must be non-empty")
    if not list(factors):
        raise DataValidationError("one_at_a_time: factors must be non-empty")
    names = list(parameters) if parameters is not None else list(base)
    reference: dict[str, float] = {}
    for name in names:
        if name not in base:
            raise DataValidationError(f"one_at_a_time: unknown parameter {name!r}")
    rows: list[dict[str, Any]] = []
    for factor in factors:
        for name in names:
            perturbed = dict(base)
            perturbed[name] = base[name] * float(factor)
            value = float(simulate(perturbed))
            if float(factor) == 1.0:
                reference[name] = value
            rows.append(
                {"parameter": name, "factor": float(factor), "value": value, "pct_change": float("nan")}
            )
    for row in rows:
        ref = reference[row["parameter"]]
        row["pct_change"] = (row["value"] - ref) / ref * 100.0 if ref != 0 else float("nan")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the trial-simulation operating characteristics
# ---------------------------------------------------------------------------


def power_with_ci(rejections: Sequence[bool] | np.ndarray) -> dict[str, float]:
    """Power with the binomial Monte-Carlo confidence interval.

    ``p = mean(rejections)``, ``CI = p ± 1.96·sqrt(p(1−p)/n)`` — the
    portfolio's project-05 power aggregation over its 1000
    replicates (clipped to [0, 1]).

    Examples
    --------
    >>> stats = power_with_ci([True] * 8 + [False] * 2)
    >>> round(stats["power"], 2), round(stats["mc_se"], 3)
    (0.8, 0.126)
    """

    flags = np.asarray(list(rejections), dtype=bool)
    if flags.size < 1:
        raise DataValidationError("power_with_ci: need at least one replicate")
    p = float(flags.mean())
    mc_se = math.sqrt(p * (1.0 - p) / flags.size)
    return {
        "n": float(flags.size),
        "power": p,
        "mc_se": mc_se,
        "ci_lower": max(p - 1.96 * mc_se, 0.0),
        "ci_upper": min(p + 1.96 * mc_se, 1.0),
    }


def operating_characteristics(
    estimates: Sequence[float],
    truth: float,
    tolerance: float = 0.30,
) -> dict[str, float]:
    """Bias, RMSE and coverage of a simulated estimator.

    ``bias = mean((est − truth)/truth)``, ``RMSE = sqrt(mean((est −
    truth)²))`` and ``coverage = mean(|est − truth|/truth ≤
    tolerance)`` — the portfolio's project-05 EC₅₀ operating
    characteristics (the ±30 % criterion).

    Examples
    --------
    >>> oc = operating_characteristics([2.9, 3.1, 3.0], truth=3.0)
    >>> round(oc["bias"], 4), round(oc["coverage"], 3)
    (0.0, 1.0)
    """

    values = np.asarray(list(estimates), dtype=float)
    if values.size < 1:
        raise DataValidationError("operating_characteristics: need at least one estimate")
    if truth == 0:
        raise DataValidationError("operating_characteristics: truth must be non-zero")
    relative = (values - truth) / truth
    return {
        "n": float(values.size),
        "bias": float(relative.mean()),
        "rmse": float(math.sqrt(np.mean((values - truth) ** 2))),
        "coverage": float(np.mean(np.abs(relative) <= tolerance)),
    }
