"""Local surrogate explanations — the lime and DALEX concepts on numpy.

Absorbed in round 100x-174 from two repositories reviewed file by file:

* **lime** (marcotcr/lime, v0.2.0.1) — the tabular LIME algorithm
  exactly as implemented in ``lime/lime_tabular.py`` +
  ``lime/lime_base.py``: quartile discretization of the training
  columns (``np.percentile(col, [25, 50, 75])`` cut points, dedup);
  perturbation samples drawn from the **training bin-frequency
  distribution** with the binary matrix marking "same bin as the
  instance"; sampled bins mapped back to continuous values inside the
  bin; the kernel ``sqrt(exp(−d²/w²))`` (LIME's own square root — half
  the exponent of the paper's kernel, an effective width of
  ``w·√2`` with ``w = 0.75·sqrt(n_features)``) on euclidean distances
  of the binary matrix; feature selection ``'auto'`` (forward
  selection with weighted R² when ``num_features ≤ 6``, else
  highest-weights ranking through a Ridge α=0.01 pass); the final
  weighted Ridge α=1 surrogate with ``local_exp`` sorted by
  ``|coef|``.
* **DALEX** (ModelOriented/DALEX, v2.5.4 — the R package is a façade
  over ``ingredients``/``iBreakDown``, but the sibling
  ``python/dalex`` package reimplements everything natively and is
  what was transcribed): ``model_performance``'s loss table (per
  observation residuals with mse / rmse / mae / mad / r²);
  ``model_profile``'s partial dependence (the 101-point uniform or
  quantile grid of ``calculate_variable_split``, the group-by-mean
  aggregation, per-instance ICE curves); and ``predict_parts``'s
  **break-down** attributions (baseline = mean prediction, the 1-D
  changes ranked by |diff|, the greedy path walk whose step
  contributions close the sum from the baseline to the instance's
  prediction).

Deduplicated on purpose: DALEX's permutation ``model_parts`` is
already this package's
:func:`pharmaml.ml.evaluation.permutation_feature_importance`; lime's
tree-surrogate sibling and DALEX's oscillation/shap variants are
honest boundaries.  LIME's truncated-normal bin sampling is
approximated by clipped normals (no SciPy dependency at the call
site) — a documented deviation.

Examples
--------
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> rng = np.random.default_rng(7)
>>> X = rng.normal(size=(400, 3))
>>> y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.05 * X[:, 2] + 1.0
>>> model = LinearRegression().fit(X, y)
>>> explanation = lime_tabular_explain(model, X, X[41], num_features=3,
...                                    num_samples=1000, seed=11)
>>> weights = dict(explanation.local_exp)
>>> abs(weights['x1']) > abs(weights['x2']) > abs(weights['x3'])
True
>>> explanation.score > 0.5                          # the local fit
True
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BreakDownAttribution",
    "LimeExplanation",
    "PartialDependenceProfile",
    "break_down_attribution",
    "lime_tabular_explain",
    "loss_table",
    "partial_dependence",
]


# ---------------------------------------------------------------------------
# LIME — the tabular surrogate (marcotcr/lime)
# ---------------------------------------------------------------------------


@dataclass
class LimeExplanation:
    """One LIME result — ``explanation.Explanation`` reshaped.

    Attributes
    ----------
    prediction:
        The model's own prediction on the instance.
    intercept / local_pred:
        The weighted-Ridge surrogate's intercept and its prediction on
        the instance row (all-ones in discretized mode).
    local_exp:
        ``(feature name, coefficient)`` pairs sorted by ``|coef|``
        descending — LIME's ``exp.local_exp``.
    score:
        The weighted R² of the surrogate on the perturbation sample.
    """

    prediction: float
    intercept: float
    local_pred: float
    local_exp: list[tuple[str, float]]
    score: float


def _quartile_bins(column: np.ndarray) -> np.ndarray:
    """LIME's ``QuartileDiscretizer.bins`` — deduplicated quartile cuts."""
    cuts = np.percentile(column, [25.0, 50.0, 75.0])
    return np.unique(cuts)


def _bin_index(column: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """LIME's bin assignment — ``searchsorted`` (upper-bin semantics)."""
    return np.asarray(np.searchsorted(cuts, column), dtype=int)


def _bin_stats(column: np.ndarray, bins: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Per-bin ``(mean, std, lo, hi)`` over the *training* column."""
    stats: list[tuple[float, float, float, float]] = []
    lo = float(column.min())
    hi = float(column.max())
    edges = np.concatenate([[lo], bins, [hi]])
    for k in range(len(edges) - 1):
        in_bin = (column >= edges[k]) & (column <= edges[k + 1]) if k == len(edges) - 2 else (
            (column >= edges[k]) & (column < edges[k + 1])
        )
        values = column[in_bin]
        if values.size:
            stats.append((float(values.mean()), float(values.std()) + 1e-11,
                          float(edges[k]), float(edges[k + 1])))
        else:
            stats.append((float(edges[k]), 1e-11, float(edges[k]), float(edges[k + 1])))
    return stats


def lime_tabular_explain(
    model: Any,
    data: np.ndarray | pd.DataFrame,
    instance: np.ndarray | pd.Series,
    *,
    feature_names: list[str] | None = None,
    num_features: int = 10,
    num_samples: int = 5000,
    kernel_width: float | None = None,
    seed: int | None = None,
) -> LimeExplanation:
    """Explain one prediction with a local weighted-Ridge surrogate — LIME.

    The algorithm follows ``LimeTabularExplainer.explain_instance`` with
    the default quartile discretizer: every column's training
    distribution is cut into quartile bins; ``num_samples`` perturbation
    rows are drawn from the training **bin frequencies** and mapped to
    continuous values inside their bins; the binary "same bin as the
    instance" matrix carries the distances through lime's kernel
    ``sqrt(exp(−d²/w²))`` with ``w = 0.75·sqrt(n_columns)``; the
    features are selected lime's ``'auto'`` way (forward selection for
    ``num_features ≤ 6``, highest-weights otherwise) and a weighted
    Ridge α=1 explains the model's predictions around the instance.

    Parameters
    ----------
    model:
        Any fitted scikit-learn style predictor with ``predict`` (or
        ``predict_proba`` — the classifier path takes label 1).
    data:
        The training matrix (LIME bins and bin frequencies come from
        it).
    instance:
        The row being explained.
    feature_names:
        Column labels (``x1 … xn`` by default).
    num_features / num_samples:
        LIME's defaults are 10 and 5000.
    seed:
        Reproducibility of the perturbation draw.

    Returns
    -------
    LimeExplanation
        See :class:`LimeExplanation`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> rng = np.random.default_rng(3)
    >>> X = rng.normal(size=(300, 4))
    >>> y = 4.0 * X[:, 0] + 0.01 * X[:, 3]
    >>> model = LinearRegression().fit(X, y)
    >>> exp = lime_tabular_explain(model, X, X[5], num_features=2,
    ...                            num_samples=800, seed=1)
    >>> names = [name for name, _ in exp.local_exp]
    >>> names[0] == 'x1'                    # the informative feature leads
    True
    >>> abs(dict(exp.local_exp)['x1']) > 1.0    # its bin effect (beta=4)
    True
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {X.shape}")
    row = np.asarray(instance, dtype=float).ravel()
    if row.shape[0] != X.shape[1]:
        raise ValueError(
            f"instance has {row.shape[0]} features, data has {X.shape[1]}"
        )
    if num_features < 1 or num_samples < 10:
        raise ValueError("num_features >= 1 and num_samples >= 10 required")
    names = feature_names or [f"x{j + 1}" for j in range(X.shape[1])]
    if len(names) != X.shape[1]:
        raise ValueError("feature_names length does not match the data width")
    rng = np.random.default_rng(seed)
    width = kernel_width if kernel_width is not None else 0.75 * math.sqrt(X.shape[1])

    # -- discretize every column on the training quartiles ---------------
    bins = [_quartile_bins(X[:, j]) for j in range(X.shape[1])]
    train_bins = [_bin_index(X[:, j], bins[j]) for j in range(X.shape[1])]
    stats = [_bin_stats(X[:, j], bins[j]) for j in range(X.shape[1])]
    instance_bins = np.array(
        [int(_bin_index(np.asarray([row[j]]), bins[j])[0]) for j in range(X.shape[1])]
    )

    # -- sample bin ids from the training frequencies ---------------------
    n_bins = [len(stats[j]) for j in range(X.shape[1])]
    sampled_bins = np.zeros((num_samples, X.shape[1]), dtype=int)
    binary = np.zeros((num_samples, X.shape[1]), dtype=float)
    binary[0, :] = 1.0
    for j in range(X.shape[1]):
        values, counts = np.unique(train_bins[j], return_counts=True)
        freqs = counts / counts.sum()
        sampled_bins[:, j] = rng.choice(values, size=num_samples, p=freqs)
        binary[:, j] = (sampled_bins[:, j] == instance_bins[j]).astype(float)
    binary[0, :] = 1.0  # LIME pins the instance row itself

    # -- map bins to continuous values inside each bin --------------------
    # LIME draws a truncated normal per bin; without SciPy at the call
    # site this uses the clipped-normal approximation (documented).
    samples = np.empty_like(binary)
    for j in range(X.shape[1]):
        for k in range(n_bins[j]):
            mask = sampled_bins[:, j] == k
            if not mask.any():
                continue
            mean, std, lo, hi = stats[j][k]
            draw = rng.normal(mean, std, size=int(mask.sum()))
            samples[mask, j] = np.clip(draw, lo, hi)
        samples[0, j] = row[j]

    # -- the model's predictions on the perturbations ---------------------
    predict = getattr(model, "predict_proba", None) or model.predict
    raw = predict(samples)
    target = np.asarray(raw)[:, 1] if np.asarray(raw).ndim == 2 else np.asarray(raw).ravel()
    prediction = float(target[0])

    # -- lime's kernel on the binary matrix distances ---------------------
    diffs = binary - binary[0][None, :]
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    weights = np.sqrt(np.exp(-(distances ** 2) / (width ** 2)))

    def _fit(columns: list[int], w: np.ndarray) -> tuple[float, np.ndarray, float]:
        from sklearn.linear_model import Ridge

        ridge = Ridge(alpha=1.0).fit(
            binary[:, columns], target, sample_weight=w
        )
        pred = ridge.predict(binary[:, columns])
        resid = target - pred
        denom = float(w.sum())
        ss_res = float((w * resid ** 2).sum())
        ss_tot = float((w * (target - np.average(target, weights=w)) ** 2).sum())
        score = 1.0 - ss_res / ss_tot if ss_tot > 0 and denom > 0 else 0.0
        return float(ridge.intercept_), ridge.coef_.ravel(), score

    # -- feature selection: lime's 'auto' ---------------------------------
    if num_features <= 6 and num_features < X.shape[1]:
        # forward selection: greedily add the column with the best
        # weighted R2 (lime_base.forward_selection)
        chosen: list[int] = []
        for _ in range(num_features):
            best_gain, best_col = -math.inf, None
            for j in range(X.shape[1]):
                if j in chosen:
                    continue
                trial = chosen + [j]
                _int, _coef, score = _fit(trial, weights)
                if score > best_gain + 1e-12:
                    best_gain, best_col = score, j
            if best_col is None:
                break
            chosen.append(best_col)
        intercept, coef, score = _fit(chosen, weights)
    elif num_features < X.shape[1]:
        from sklearn.linear_model import Ridge

        ranker = Ridge(alpha=0.01).fit(binary, target, sample_weight=weights)
        magnitude = np.abs(ranker.coef_.ravel() * binary[0])
        chosen = list(np.argsort(-magnitude)[:num_features])
        intercept, coef, score = _fit(chosen, weights)
    else:
        chosen = list(range(X.shape[1]))
        intercept, coef, score = _fit(chosen, weights)

    local_exp = sorted(
        ((names[j], float(c)) for j, c in zip(chosen, coef)),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    local_pred = float(intercept + float(np.sum(coef * binary[0, chosen])))
    return LimeExplanation(
        prediction=prediction,
        intercept=intercept,
        local_pred=local_pred,
        local_exp=local_exp,
        score=score,
    )


# ---------------------------------------------------------------------------
# DALEX — performance loss table, partial dependence, break-down
# ---------------------------------------------------------------------------


@dataclass
class PartialDependenceProfile:
    """A PD profile — DALEX's ``model_profile`` result.

    Attributes
    ----------
    grid:
        The 101-point grid (uniform or quantile).
    mean:
        The partial-dependence curve (the group-by-mean of
        predictions).
    ice:
        Per-instance curves (Individual Conditional Expectation), one
        row per sampled observation.
    feature:
        The profiled column's name.
    """

    feature: str
    grid: np.ndarray
    mean: np.ndarray
    ice: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def partial_dependence(
    model: Any,
    data: np.ndarray | pd.DataFrame,
    feature: int | str,
    *,
    grid_points: int = 101,
    grid_type: str = "uniform",
    n: int = 300,
    ice: bool = False,
) -> PartialDependenceProfile:
    """Partial dependence — DALEX ``model_profile(type='partial')``.

    The grid follows python-dalex's ``calculate_variable_split``:
    ``uniform`` spans ``linspace(min, max, grid_points)`` and
    ``quantile`` takes the unique column quantiles; the profile is the
    mean prediction over ``n`` sampled rows with the feature clamped
    to each grid value, and ``ice=True`` keeps the per-instance curves
    (ceteris paribus profiles) too.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> rng = np.random.default_rng(5)
    >>> X = rng.normal(size=(200, 2))
    >>> y = 2.5 * X[:, 0] + 1.0
    >>> model = LinearRegression().fit(X, y)
    >>> prof = partial_dependence(model, X, 0, grid_points=11, n=100)
    >>> slope = np.polyfit(prof.grid, prof.mean, 1)[0]
    >>> bool(abs(slope - 2.5) < 1e-6)          # a linear model's PD is its line
    True
    >>> prof2 = partial_dependence(model, X, 0, grid_type='quantile',
    ...                            grid_points=11)
    >>> bool(np.all(np.diff(prof2.grid) > 0))
    True
    """
    if grid_points < 2:
        raise ValueError("grid_points must be >= 2")
    if grid_type not in ("uniform", "quantile"):
        raise ValueError("grid_type must be 'uniform' or 'quantile'")
    if n < 1:
        raise ValueError("n must be >= 1")
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {X.shape}")
    column_index, name = _resolve_column(X, data, feature)
    column = X[:, column_index]
    if grid_type == "uniform":
        axis_grid = np.linspace(float(column.min()), float(column.max()), grid_points)
    else:
        axis_grid = np.unique(
            np.quantile(column, np.linspace(0.0, 1.0, grid_points))
        )
    n = min(n, X.shape[0])
    rows = X[:n].copy()
    curves = np.empty((n, axis_grid.size), dtype=float)
    predict = getattr(model, "predict_proba", None) or model.predict
    for k, value in enumerate(axis_grid):
        rows[:, column_index] = value
        raw = predict(rows)
        curves[:, k] = np.asarray(raw)[:, 1] if np.asarray(raw).ndim == 2 else np.asarray(raw).ravel()
    return PartialDependenceProfile(
        feature=name,
        grid=axis_grid,
        mean=curves.mean(axis=0),
        ice=curves if ice else np.zeros((0, 0)),
    )


def _resolve_column(
    X: np.ndarray, data: np.ndarray | pd.DataFrame, feature: int | str
) -> tuple[int, str]:
    if isinstance(feature, int):
        if not 0 <= feature < X.shape[1]:
            raise ValueError(f"feature index {feature} out of range")
        if isinstance(data, pd.DataFrame):
            return feature, str(data.columns[feature])
        return feature, f"x{feature + 1}"
    if isinstance(data, pd.DataFrame) and feature in data.columns:
        return list(data.columns).index(feature), str(feature)
    raise ValueError(f"feature {feature!r} is not a column of the data")


def loss_table(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Per-observation losses — DALEX ``model_performance`` for regression.

    mse / rmse / mae / mad / r² over a table of
    ``predicted · observed · diff`` rows (python-dalex's
    ``_model_performance`` measures, r² defined as
    ``1 − mse/mse(mean(y), y)``).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> y = np.array([1.0, 2.0, 3.0, 4.0])
    >>> table = loss_table(y, np.array([1.5, 2.0, 2.5, 4.5]))
    >>> round(float(table.mse.iloc[0]), 4)
    0.1875
    >>> round(float(table.mad.iloc[0]), 4)
    0.5
    """
    truth = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must be the same length")
    diff = truth - predicted
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    mad = float(np.median(np.abs(diff)))
    base = float(np.mean((truth - truth.mean()) ** 2))
    r2 = 1.0 - mse / base if base > 0 else math.nan
    return pd.DataFrame(
        {
            "predicted": [predicted],
            "observed": [truth],
            "diff": [diff],
            "mse": [mse],
            "rmse": [math.sqrt(mse)],
            "mae": [mae],
            "mad": [mad],
            "r2": [r2],
        }
    )


@dataclass
class BreakDownAttribution:
    """One break-down result — DALEX ``predict_parts(type='break_down')``.

    The step contributions close exactly from the baseline to the
    instance's own prediction.
    """

    baseline: float
    prediction: float
    contributions: list[tuple[str, float]]

    @property
    def closes(self) -> float:
        """baseline + Σ contributions − prediction (≈ 0 by construction)."""
        total = self.baseline + sum(value for _, value in self.contributions)
        return total - self.prediction


def break_down_attribution(
    model: Any,
    data: np.ndarray | pd.DataFrame,
    observation: np.ndarray | pd.Series,
    *,
    feature_names: list[str] | None = None,
    keep_order: bool = False,
) -> BreakDownAttribution:
    """Additive attribution of one prediction — DALEX's ``break_down``.

    python-dalex's ``_break_down`` algorithm: the baseline is the mean
    prediction over ``data``; each feature's "1-D change" (the mean
    prediction with that single column clamped to the observation's
    value) ranks the path by decreasing ``|diff|`` (unless
    ``keep_order``); walking the path and clamping one column at a
    time, each step's contribution is the change in the mean
    prediction — the contributions sum from the baseline to the
    observation's own prediction.  For a linear model this reproduces
    ``Σ β_j·(x_j − x̄_j)`` exactly.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> rng = np.random.default_rng(9)
    >>> X = rng.normal(size=(500, 3)) * np.array([2.0, 1.0, 0.5])
    >>> y = 3.0 * X[:, 0] - 1.0 * X[:, 1] + 2.0 * X[:, 2]
    >>> model = LinearRegression().fit(X, y)
    >>> obs = np.array([1.0, 0.0, 0.0])
    >>> bd = break_down_attribution(model, X, obs, keep_order=True)
    >>> means = X.mean(axis=0)
    >>> contrib = dict(bd.contributions)
    >>> bool(abs(contrib['x1'] - 3.0 * (1.0 - means[0])) < 1e-6)
    True
    >>> bool(abs(bd.closes) < 1e-6)
    True
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {X.shape}")
    row = np.asarray(observation, dtype=float).ravel()
    if row.shape[0] != X.shape[1]:
        raise ValueError(
            f"observation has {row.shape[0]} features, data has {X.shape[1]}"
        )
    names = feature_names or [f"x{j + 1}" for j in range(X.shape[1])]
    predict = getattr(model, "predict_proba", None) or model.predict

    def mean_prediction(matrix: np.ndarray) -> float:
        raw = predict(matrix)
        values = (
            np.asarray(raw)[:, 1] if np.asarray(raw).ndim == 2 else np.asarray(raw).ravel()
        )
        return float(values.mean())

    baseline = mean_prediction(X)
    # 1-D changes (the ranking statistic)
    one_d = []
    for j in range(X.shape[1]):
        clamped = X.copy()
        clamped[:, j] = row[j]
        one_d.append(mean_prediction(clamped) - baseline)
    order = (
        list(range(X.shape[1]))
        if keep_order
        else list(np.argsort(-np.abs(one_d)))
    )
    current = X.copy()
    previous = baseline
    contributions: list[tuple[str, float]] = []
    for j in order:
        current = current.copy()
        current[:, j] = row[j]
        value = mean_prediction(current)
        contributions.append((names[j], value - previous))
        previous = value
    obs_matrix = row[None, :]
    prediction = float(
        np.asarray(predict(obs_matrix)).ravel()[0]
        if np.asarray(predict(obs_matrix)).ndim == 1
        else np.asarray(predict(obs_matrix))[0, 1]
    )
    return BreakDownAttribution(
        baseline=baseline,
        prediction=prediction,
        contributions=contributions,
    )
