"""Pharmacometric diagnostics: VPC, bootstrap, operating characteristics.

Absorbs the diagnostics layer of the pharmacometrics portfolio
(``R/helpers.R`` + project 05's operating characteristics) over the
toolkit's own simulation machinery:

* :func:`vpc_statistics` — the visual predictive check table: time
  bins, observed percentiles and the simulated 5th/50th/95th
  percentiles from Monte-Carlo profiles;
* :func:`prediction_corrected_vpc` — the prediction-corrected VPC
  (each concentration divided by the median population prediction in
  its time bin — the correction that lets studies with different
  doses/covariates pool into one VPC);
* :func:`case_bootstrap_subjects` — subject-level case resampling
  (the EMA 2007 / FDA 2019 standard for population-PK uncertainty);
* :func:`bootstrap_summary` — percentile bootstrap CIs with
  mean/median/SE (the portfolio's ``summarize_bootstrap``);
* :func:`operating_characteristics` — the trial-simulation scorecard:
  power with its Monte-Carlo error, bias, RMSE and coverage.

Examples
--------
>>> import numpy as np
>>> from pharmaml.pkpd.diagnostics import (
...     bootstrap_summary, case_bootstrap_subjects,
...     operating_characteristics, prediction_corrected_vpc,
...     vpc_statistics)

A VPC on a decay with 20 % noise — the simulated 50th percentile
tracks the observed median:

>>> rng = np.random.default_rng(3)
>>> times = np.repeat(np.linspace(1.0, 10.0, 10), 3)
>>> observed = 5.0 * np.exp(-0.3 * times) * rng.normal(1.0, 0.08, times.size)
>>> sim_times = np.tile(np.linspace(1.0, 10.0, 10), 200)
>>> sims = 5.0 * np.exp(-0.3 * sim_times) * rng.normal(1.0, 0.08, sim_times.size)
>>> table = vpc_statistics(observed_times=times, observed_values=observed,
...                        simulated_times=sim_times, simulated_values=sims,
...                        n_bins=4)
>>> len(table)
4
>>> mid = table[0]                     # first bin's simulated median
>>> bool(abs(mid["sim_p50"] - 5.0 * np.exp(-0.3 * mid["t_mid"])) < 0.2)
True

Prediction correction divides by the bin-median population prediction
(2× the dose ⇒ the same corrected curve):

>>> pop_pred = 5.0 * np.exp(-0.3 * sim_times)
>>> corrected = prediction_corrected_vpc(sims, pop_pred, sim_times, n_bins=4)
>>> round(float(np.median(corrected)), 2)
1.0

Case resampling keeps the subject count (duplicates are the point):

>>> resampled = case_bootstrap_subjects(list(range(1, 6)), n_boot=3, seed=11)
>>> [len(rep) for rep in resampled]
[5, 5, 5]
>>> len({int(i) for i in resampled[0]}) <= 5
True

Bootstrap percentile summary over the values 1..9:

>>> summary = bootstrap_summary(list(range(1, 10)))
>>> (round(summary["ci_lower"], 2), round(summary["ci_upper"], 2))
(1.2, 8.8)

Operating characteristics of a trial simulation: 80/100 replicates
significant, estimates centred on the truth:

>>> oc = operating_characteristics(
...     estimates=np.full(100, 2.5), true_value=2.5,
...     rejected=np.concatenate([np.ones(80), np.zeros(20)]))
>>> (round(oc["power"], 2), round(oc["bias"], 4), round(oc["rmse"], 4),
...  round(oc["coverage_30pct"], 2))
(0.8, 0.0, 0.0, 1.0)
>>> round(oc["power_ci_lower"], 3), round(oc["power_ci_upper"], 3)
(0.722, 0.878)
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from pharmaml.exceptions import DataValidationError

__all__ = [
    "bootstrap_summary",
    "case_bootstrap_subjects",
    "operating_characteristics",
    "prediction_corrected_vpc",
    "vpc_statistics",
]

_DEFAULT_PERCENTILES = (5.0, 50.0, 95.0)


def _bin_edges(times: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-spaced bin edges over the observed time range."""
    unique = np.unique(times)
    if unique.size < 2:
        raise DataValidationError("need at least two distinct times to bin")
    if n_bins < 1:
        raise DataValidationError(f"n_bins must be >= 1, got {n_bins}")
    probabilities = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(unique, probabilities))
    if edges.size < 2:
        raise DataValidationError("binning collapsed to a single bin")
    return edges


def vpc_statistics(
    *,
    observed_times: np.ndarray | list[float],
    observed_values: np.ndarray | list[float],
    simulated_times: np.ndarray | list[float],
    simulated_values: np.ndarray | list[float],
    n_bins: int = 8,
    percentiles: tuple[float, ...] = _DEFAULT_PERCENTILES,
) -> list[dict[str, float]]:
    """One VPC row per time bin: observed vs simulated percentiles.

    Bins are quantile-spaced over the observed time range (the
    ``vpc`` R package's default); each row carries the bin's midpoint,
    the observed percentile triple and count, and the simulated
    percentiles from the pooled Monte-Carlo replicates.
    """
    obs_t = np.asarray(observed_times, dtype=float).reshape(-1)
    obs_v = np.asarray(observed_values, dtype=float).reshape(-1)
    sim_t = np.asarray(simulated_times, dtype=float).reshape(-1)
    sim_v = np.asarray(simulated_values, dtype=float).reshape(-1)
    if obs_t.size != obs_v.size or obs_t.size == 0:
        raise DataValidationError("observed times/values must match, non-empty")
    if sim_t.size != sim_v.size or sim_t.size == 0:
        raise DataValidationError("simulated times/values must match, non-empty")
    edges = _bin_edges(obs_t, n_bins)
    rows: list[dict[str, float]] = []
    for left, right in pairwise(edges):
        in_obs = (obs_t >= left) & (obs_t <= right if right == edges[-1] else obs_t < right)
        in_sim = (sim_t >= left) & (sim_t <= right if right == edges[-1] else sim_t < right)
        row: dict[str, float] = {
            "t_mid": 0.5 * (float(left) + float(right)),
            "n_obs": float(np.count_nonzero(in_obs)),
        }
        if np.any(in_obs):
            for level in percentiles:
                key = "obs_p" + ("50" if level == 50.0 else str(int(level)))
                row[key] = float(np.percentile(obs_v[in_obs], level))
        if np.any(in_sim):
            for level in percentiles:
                key = "sim_p" + ("50" if level == 50.0 else str(int(level)))
                row[key] = float(np.percentile(sim_v[in_sim], level))
        rows.append(row)
    return rows


def prediction_corrected_vpc(
    simulated_values: np.ndarray | list[float],
    population_prediction: np.ndarray | list[float],
    simulated_times: np.ndarray | list[float],
    *,
    n_bins: int = 8,
    observed_values: np.ndarray | list[float] | None = None,
    observed_times: np.ndarray | list[float] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Divide concentrations by the bin-median population prediction.

    The prediction-corrected VPC (Bergstrand et al. 2011; the
    portfolio's ``pcvpc``): ``pc = value / median_bin(pred)`` for every
    simulated (and optionally observed) point, removing the structural
    model's dose/covariate effects so groups pool cleanly.  Returns
    the corrected simulated values, or ``(sim, obs)`` when observed
    data is supplied.
    """
    sim_v = np.asarray(simulated_values, dtype=float).reshape(-1)
    pred = np.asarray(population_prediction, dtype=float).reshape(-1)
    sim_t = np.asarray(simulated_times, dtype=float).reshape(-1)
    if not (sim_v.size == pred.size == sim_t.size) or sim_v.size == 0:
        raise DataValidationError("simulated values/prediction/times must match")
    edges = _bin_edges(sim_t, n_bins)
    corrected = np.empty_like(sim_v)
    for left, right in pairwise(edges):
        inside = (sim_t >= left) & (sim_t <= right if right == edges[-1] else sim_t < right)
        if not np.any(inside):
            continue
        bin_median = float(np.median(pred[inside]))
        corrected[inside] = sim_v[inside] / bin_median if bin_median != 0 else np.nan
    if observed_values is None or observed_times is None:
        return corrected
    obs_t = np.asarray(observed_times, dtype=float).reshape(-1)
    obs_v = np.asarray(observed_values, dtype=float).reshape(-1)
    if obs_t.size != obs_v.size:
        raise DataValidationError("observed times/values must match")
    obs_corrected = np.empty_like(obs_v)
    for left, right in pairwise(edges):
        inside = (obs_t >= left) & (obs_t <= right if right == edges[-1] else obs_t < right)
        if not np.any(inside):
            continue
        sim_inside = (sim_t >= left) & (
            sim_t <= right if right == edges[-1] else sim_t < right
        )
        bin_median = (
            float(np.median(pred[sim_inside])) if np.any(sim_inside) else math.nan
        )
        obs_corrected[inside] = obs_v[inside] / bin_median if bin_median else np.nan
    return corrected, obs_corrected


def case_bootstrap_subjects(
    subject_ids: list[int] | list[str],
    *,
    n_boot: int,
    seed: int = 0,
) -> list[list[int | str]]:
    """Subject-level case resampling (with replacement).

    Each replicate keeps the study size — some subjects appear twice
    or more, others drop — the resampling scheme EMA/FDA population-PK
    guidance prescribes for parameter uncertainty.
    """
    if n_boot < 1:
        raise DataValidationError(f"n_boot must be >= 1, got {n_boot}")
    if not subject_ids:
        raise DataValidationError("subject_ids must be non-empty")
    rng = np.random.default_rng(seed)
    n = len(subject_ids)
    replicates: list[list[int | str]] = []
    for _ in range(n_boot):
        draw = rng.integers(0, n, size=n)
        replicates.append([subject_ids[int(index)] for index in draw])
    return replicates


def bootstrap_summary(
    values: np.ndarray | list[float],
    *,
    level: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap summary (mean, median, SE, CI).

    The portfolio's ``summarize_bootstrap``: percentile interval at
    ``level``, boot mean/median/SE and the replicate count.
    """
    sample = np.asarray(values, dtype=float).reshape(-1)
    if sample.size < 2:
        raise DataValidationError("bootstrap_summary needs at least two values")
    if not 0.0 < level < 1.0:
        raise DataValidationError(f"level must be in (0, 1), got {level}")
    alpha = (1.0 - level) / 2.0
    return {
        "boot_mean": float(np.mean(sample)),
        "boot_median": float(np.median(sample)),
        "boot_se": float(np.std(sample, ddof=1)),
        "ci_lower": float(np.quantile(sample, alpha)),
        "ci_upper": float(np.quantile(sample, 1.0 - alpha)),
        "n_replicates": float(sample.size),
    }


def operating_characteristics(
    *,
    estimates: np.ndarray | list[float],
    true_value: float,
    rejected: np.ndarray | list[bool] | list[int],
) -> dict[str, float]:
    """Trial-simulation scorecard (the portfolio's project 05).

    ``power`` = fraction of significant replicates with its
    Monte-Carlo error interval ``±1.96·√(p(1−p)/N)``; ``bias`` and
    ``rmse`` of the estimates against ``true_value``; ``coverage`` =
    the share within 30 % of the truth.
    """
    est = np.asarray(estimates, dtype=float).reshape(-1)
    rej = np.asarray(rejected).reshape(-1).astype(bool)
    if est.size == 0 or est.size != rej.size:
        raise DataValidationError("estimates and rejected must match, non-empty")
    n = float(est.size)
    power = float(np.mean(rej))
    mc_error = 1.96 * math.sqrt(power * (1.0 - power) / n) if n else float("nan")
    error = est - true_value
    return {
        "n_replicates": n,
        "power": power,
        "power_ci_lower": max(0.0, power - mc_error),
        "power_ci_upper": min(1.0, power + mc_error),
        "bias": float(np.mean(error) / true_value) if true_value else float("nan"),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "coverage_30pct": float(np.mean(np.abs(error) / abs(true_value) <= 0.30))
        if true_value
        else float("nan"),
    }
