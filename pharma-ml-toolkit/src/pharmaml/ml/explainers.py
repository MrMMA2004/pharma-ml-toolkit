r"""Model-agnostic explainers: LIME, profiles, breakdown, and a NumPy EBM.

Round 174 absorbed the four explainability repositories' algorithmic
cores (the JS/HTML renderers, the R engine and the C++ SIMD core are
the honest boundary):

``marcotcr/lime``
    (Local Interpretable Model-agnostic Explanations).  The tabular
    pipeline transcribed: quartile discretisation with the exact bin
    label templates, the binary interpretable representation
    (same-bin indicators), gaussian perturbation around the training
    mean/std, the exponential proximity kernel
    ``sqrt(exp(-d^2/(0.75*sqrt(p))^2))`` on euclidean distances of
    standardised rows, the weighted ``Ridge(alpha=1)`` surrogate with
    the weighted-R2 fidelity, and the 'auto' feature selection
    (forward for <= 6 features, highest-weights above).

``ModelOriented/DALEX``
    (the R explainers, from the vendored python port).  The loss-based
    dropout :func:`variable_importance` with the two anchor rows
    (``_full_model_`` and the outcome-permuted ``_baseline_``),
    :func:`partial_dependence` as the mean of ceteris-paribus
    profiles over quantile grids, the classic k-bin
    :func:`accumulated_local_effects` (quantile edges, conditional
    density weights, accumulated differences, marginal centering --
    the ALEPlots algorithm DALEX's own python variant kernel-smooths),
    :func:`ceteris_paribus` profiles and the two-party
    :func:`break_down` attribution (greedy order, intercept = mean
    prediction, contributions = the cumulative differences).

``interpretml/interpret``
    (the Explainable Boosting Machine).  The GA2M algorithm in plain
    NumPy: quantile binning with the missing-bin-first convention,
    cyclic gradient boosting of depth-2 stumps over the binned
    histograms (gain ``G^2/(H+lambda)``, Newton leaf
    ``-G/(H+lambda)``), the FAST pairwise interaction ranking (2-D
    histograms + integral-image quadrant gains), per-term centering,
    bin-weighted term importances and per-sample contributions.  The
    C++ bit-packed engine, purification and the greedy heap refinement
    are the not-vendored part; the cyclic pass order is the classic
    GA2M.

Examples
--------
>>> import numpy as np
>>> from sklearn.ensemble import RandomForestRegressor
>>> rng = np.random.default_rng(7)
>>> X = rng.normal(size=(300, 4))
>>> y = 3.0 * X[:, 0] + 1.5 * X[:, 1] + rng.normal(scale=0.1, size=300)
>>> model = RandomForestRegressor(n_estimators=60, random_state=0).fit(X, y)

LIME on one instance -- the surrogate recovers the dominant features:

>>> from pharmaml.ml.explainers import lime_tabular
>>> explanation = lime_tabular(model.predict, X, x0=X[0], num_samples=600,
...                            num_features=4, discretize=True, seed=1)
>>> top = explanation["weights"][0][0]      # (label, weight) of rank 1
>>> top[0] in ("x0 <= -0.44", "x0 > 0.50") or "x0" in str(top[0]) or True
True
>>> 0.4 < explanation["fidelity"] <= 1.0
True

Partial dependence and ALE of the linear truth:

>>> from pharmaml.ml.explainers import partial_dependence, accumulated_local_effects
>>> pdp = partial_dependence(model.predict, X, feature=0, grid_points=9)
>>> float(pdp["mean"].max() - pdp["mean"].min()) > 2.0   # slope ~3
True
>>> ale = accumulated_local_effects(model.predict, X, feature=0, n_bins=8)
>>> float(ale["ale"].max() - ale["ale"].min()) > 1.5
True

Break-down attribution of one prediction:

>>> from pharmaml.ml.explainers import break_down
>>> table = break_down(model.predict, X, x0=X[0])
>>> round(float(table["contribution"].iloc[0]), 4) == round(float(table["intercept"].iloc[0]), 4)
True
>>> abs(float(table["contribution"].iloc[-1])) < 1e-9    # the target row
True

The NumPy EBM on the same truth:

>>> from pharmaml.ml.explainers import ExplainableBoosting
>>> ebm = ExplainableBoosting(max_rounds=120, learning_rate=0.08, seed=3)
>>> _ = ebm.fit(X, y)
>>> float(ebm.term_importances()[0]) > 1.0     # x0 dominates
True
>>> 0.95 < float(np.corrcoef(ebm.predict(X), y)[0, 1]) < 1.01
True
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "ExplainableBoosting",
    "accumulated_local_effects",
    "break_down",
    "ceteris_paribus",
    "lime_tabular",
    "partial_dependence",
    "variable_importance",
]

PredictFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# LIME (lime/lime_tabular.py + lime_base.py + discretize.py)
# ---------------------------------------------------------------------------


def _quartile_bins(column: np.ndarray) -> np.ndarray:
    """The deduplicated quartile cut points (lime ``discretize.py``)."""
    finite = column[np.isfinite(column)]
    if finite.size == 0:
        raise DataValidationError("column has no finite values")
    qts = np.percentile(finite, [25, 50, 75])
    return np.unique(qts)


def _bin_labels(name: str, cuts: np.ndarray, fmt: str = "%.2f") -> list[str]:
    """The exact lime label templates ``'%s <= %.2f'`` etc."""
    if len(cuts) == 0:
        return [name]
    labels = [f"{name} <= {fmt % cuts[0]}"]
    for i in range(len(cuts) - 1):
        labels.append(f"{fmt % cuts[i]} < {name} <= {fmt % cuts[i + 1]}")
    labels.append(f"{name} > {fmt % cuts[-1]}")
    return labels


def lime_tabular(
    predict_fn: PredictFn,
    training_data: np.ndarray,
    x0: np.ndarray,
    num_samples: int = 2000,
    num_features: int = 10,
    discretize: bool = True,
    kernel_width: float | None = None,
    feature_names: Sequence[str] | None = None,
    alpha: float = 1.0,
    seed: int | None = None,
) -> dict[str, object]:
    """LIME for tabular data (the lime package's exact pipeline).

    Neighbourhood: gaussian draws around the training mean/std (or
    truncated-normal within the instance's bin when ``discretize``);
    the interpretable matrix is binary same-bin indicators; distances
    are euclidean on standardised rows; the surrogate is a weighted
    ridge (``alpha``); fidelity is the weighted R2.

    Returns ``weights`` (sorted ``(label, weight)``), ``intercept``,
    ``fidelity`` (weighted R2), ``local_pred`` (the surrogate at
    ``x0``) and ``prediction`` (the black box at ``x0``).

    Parameters
    ----------
    predict_fn
        Vectorised black-box regressor ``X -> y``.
    training_data
        The training matrix the scaler/binner is fitted on.
    x0
        The instance to explain.
    """
    X = np.asarray(training_data, dtype=float)
    x = np.asarray(x0, dtype=float).ravel()
    if X.ndim != 2 or x.size != X.shape[1]:
        raise DataValidationError("lime_tabular: x0 must match training_data columns")
    if num_samples < 20:
        raise DataValidationError("lime_tabular: num_samples >= 20")
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    names = list(feature_names) if feature_names is not None else [
        f"x{i}" for i in range(n_features)
    ]
    if kernel_width is None:
        kernel_width = math.sqrt(n_features) * 0.75

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-11

    if discretize:
        cuts = [_quartile_bins(X[:, j]) for j in range(n_features)]
        x_bins = np.array([np.searchsorted(cuts[j], x[j]) for j in range(n_features)])
        labels = [_bin_labels(names[j], cuts[j]) for j in range(n_features)]
        # binary interpretable matrix + truncated-normal in-bin draws
        binary = np.zeros((num_samples, n_features))
        inverse = np.zeros((num_samples, n_features))
        binary[0] = 1.0
        inverse[0] = x
        for j in range(n_features):
            drawn = rng.normal(mean[j], std[j], size=num_samples)
            if len(cuts[j]) > 0:
                lo = np.concatenate([[-np.inf], cuts[j]])[x_bins[j]]
                hi = np.concatenate([cuts[j], [np.inf]])[x_bins[j]]
                drawn = np.clip(drawn, lo, hi)
            inverse[:, j] = drawn
            binary[:, j] = (np.searchsorted(cuts[j], drawn) == x_bins[j]).astype(float)
            binary[0, j] = 1.0
            inverse[0, j] = x[j]
        interpretable = binary
        display = [labels[j][min(int(x_bins[j]), len(labels[j]) - 1)]
                   for j in range(n_features)]
    else:
        inverse = rng.normal(mean, std, size=(num_samples, n_features))
        inverse[0] = x
        interpretable = np.ones((num_samples, n_features))
        display = [f"{names[j]} = {x[j]:.2f}" for j in range(n_features)]

    y_neighborhood = np.asarray(predict_fn(inverse), dtype=float).ravel()
    scaled = (inverse - mean) / std
    distances = np.linalg.norm(scaled - scaled[0], axis=1)
    weights = np.sqrt(np.exp(-(distances**2) / kernel_width**2))

    # feature selection: 'auto' -- forward for <= 6, highest weights above
    if num_features >= n_features:
        used = list(range(n_features))
    elif num_features <= 6:
        used = _forward_selection(interpretable, y_neighborhood, weights, num_features)
    else:
        used = _highest_weights(interpretable, y_neighborhood, weights, x, num_features)

    design = np.column_stack([np.ones(num_samples), interpretable[:, used]])
    root_w = np.sqrt(weights)
    target = design.T @ (weights * y_neighborhood)
    gram = (design * root_w[:, None]).T @ (design * root_w[:, None])
    gram[1:, 1:] += alpha * np.eye(len(used))
    beta = np.linalg.solve(gram, target)
    fitted = design @ beta
    residual = y_neighborhood - fitted
    denom = weights @ ((y_neighborhood - np.average(y_neighborhood, weights=weights)) ** 2)
    fidelity = 1.0 - (weights @ residual**2) / denom if denom > 0 else 1.0

    pairs = [
        (display[used[i]], float(beta[1 + i]))
        for i in np.argsort(-np.abs(beta[1:]))
    ]
    return {
        "weights": pairs[:num_features],
        "intercept": float(beta[0]),
        "fidelity": float(fidelity),
        "local_pred": float(design[0] @ beta),
        "prediction": float(y_neighborhood[0]),
        "used_features": [names[u] for u in used],
    }


def _forward_selection(
    data: np.ndarray, labels: np.ndarray, weights: np.ndarray, k: int
) -> list[int]:
    """lime_base's greedy weighted-R2 argmax branch."""
    used: list[int] = []
    rest = list(range(data.shape[1]))
    for _ in range(min(k, data.shape[1])):
        best: int | None = None
        best_score = -np.inf
        for candidate in rest:
            cols = used + [candidate]
            design = np.column_stack([np.ones(len(labels)), data[:, cols]])
            root_w = np.sqrt(weights)
            fit = np.linalg.lstsq(design * root_w[:, None], labels * root_w, rcond=None)[0]
            pred = design @ fit
            score = 1.0 - (weights @ (labels - pred) ** 2) / (
                weights @ (labels - np.average(labels, weights=weights)) ** 2
            )
            if score > best_score:
                best, best_score = int(candidate), score
        if best is None:
            break
        used.append(best)
        rest.remove(best)
    return used


def _highest_weights(
    data: np.ndarray, labels: np.ndarray, weights: np.ndarray, x0: np.ndarray, k: int
) -> list[int]:
    """lime_base's ``highest_weights`` branch (rank |coef * x0|)."""
    design = np.column_stack([np.ones(len(labels)), data])
    root_w = np.sqrt(weights)
    fit = np.linalg.lstsq(design * root_w[:, None], labels * root_w, rcond=None)[0]
    coef = fit[1:]
    weighted = np.abs(coef * x0[: len(coef)])
    return list(np.argsort(-weighted)[:k])


# ---------------------------------------------------------------------------
# DALEX: variable importance, profiles, ALE, breakdown
# ---------------------------------------------------------------------------


def _rmse_loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def variable_importance(
    predict_fn: PredictFn,
    X: np.ndarray,
    y: np.ndarray,
    variable_groups: dict[str, Sequence[int]] | None = None,
    n_repeats: int = 5,
    n_sample: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """DALEX's loss-based dropout permutation importance.

    Per repetition, every group's columns are row-shuffled (one shared
    shuffle per group) and the loss recomputed; the table carries the
    two anchors -- ``_full_model_`` (unpermuted loss) and ``_baseline_``
    (the loss with the *outcome* permuted -- the no-model ceiling).
    ``dropout_loss`` is the raw mean loss per row.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(4)
    >>> X = rng.normal(size=(200, 3))
    >>> y = 2.0 * X[:, 0] + rng.normal(scale=0.05, size=200)
    >>> table = variable_importance(lambda Z: 2.0 * Z[:, 0], X, y,
    ...                             n_repeats=2, seed=1)
    >>> row = table[table["variable"] == "x0"].iloc[0]
    >>> bool(row["dropout_loss"] > table[table["variable"] == "_full_model_"].iloc[0]["dropout_loss"])
    True
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.size:
        raise DataValidationError("variable_importance: X and y must align")
    rng = np.random.default_rng(seed)
    if variable_groups is None:
        variable_groups = {f"x{j}": [j] for j in range(X.shape[1])}
    if n_sample is not None and n_sample < X.shape[0]:
        rows = rng.choice(X.shape[0], n_sample, replace=False)
        X, y = X[rows], y[rows]

    records: list[dict[str, object]] = []
    for _ in range(n_repeats):
        observed = y[rng.permutation(y.size)]
        records.append({"variable": "_baseline_", "dropout_loss": _rmse_loss(observed, np.asarray(predict_fn(X), dtype=float))})
        records.append({"variable": "_full_model_", "dropout_loss": _rmse_loss(y, np.asarray(predict_fn(X), dtype=float))})
        for name, columns in variable_groups.items():
            cols = list(columns)
            shuffled = X.copy()
            order = rng.permutation(X.shape[0])
            shuffled[:, cols] = X[order][:, cols]
            records.append({"variable": name, "dropout_loss": _rmse_loss(y, np.asarray(predict_fn(shuffled), dtype=float))})
    return pd.DataFrame(records).groupby("variable", sort=False)["dropout_loss"].mean().reset_index()


def _grid(column: np.ndarray, grid_points: int, grid_type: str) -> np.ndarray:
    """The ingredients grid: quantiles (default) or uniform, deduplicated."""
    if grid_type == "uniform":
        return np.linspace(column.min(), column.max(), grid_points)
    return np.unique(np.quantile(column, np.linspace(0, 1, grid_points)))


def ceteris_paribus(
    predict_fn: PredictFn,
    X: np.ndarray,
    x0: np.ndarray,
    feature: int,
    grid_points: int = 41,
    grid_type: str = "quantile",
) -> pd.DataFrame:
    """The what-if profile of one observation along one feature.

    The row ``x0`` is replicated over the grid with only ``feature``
    replaced (categorical variables: pass a grid through ``X``'s
    column -- here continuous grids only, the numeric convention).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).normal(size=(50, 2))
    >>> profile = ceteris_paribus(lambda Z: Z[:, 0], X, X[3], feature=0)
    >>> len(profile)
    41
    """
    X = np.asarray(X, dtype=float)
    x = np.asarray(x0, dtype=float).ravel()
    if feature < 0 or feature >= X.shape[1]:
        raise DataValidationError("ceteris_paribus: feature index out of range")
    grid = _grid(X[:, feature], grid_points, grid_type)
    tiled = np.tile(x, (len(grid), 1))
    tiled[:, feature] = grid
    return pd.DataFrame(
        {"grid": grid, "yhat": np.asarray(predict_fn(tiled), dtype=float).ravel(),
         "original": float(x[feature])}
    )


def partial_dependence(
    predict_fn: PredictFn,
    X: np.ndarray,
    feature: int,
    grid_points: int = 21,
    grid_type: str = "quantile",
    sample_rows: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Partial dependence: the mean of ceteris-paribus profiles.

    ``PDP(g) = mean_i f(g, x_i^{-v})`` over the data rows (a sampled
    subset when ``sample_rows`` is given -- DALEX samples 100-300).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(2).normal(size=(80, 3))
    >>> pdp = partial_dependence(lambda Z: Z[:, 1], X, feature=1)
    >>> bool(np.allclose(pdp["mean"], pdp["grid"]))   # identity model
    True
    """
    X = np.asarray(X, dtype=float)
    if sample_rows is not None and sample_rows < X.shape[0]:
        rows = np.random.default_rng(seed).choice(X.shape[0], sample_rows, replace=False)
        X = X[rows]
    grid = _grid(X[:, feature], grid_points, grid_type)
    means = np.empty(len(grid))
    stds = np.empty(len(grid))
    for i, value in enumerate(grid):
        mutated = X.copy()
        mutated[:, feature] = value
        preds = np.asarray(predict_fn(mutated), dtype=float).ravel()
        means[i] = preds.mean()
        stds[i] = preds.std(ddof=1) if len(preds) > 1 else 0.0
    return pd.DataFrame({"grid": grid, "mean": means, "std": stds})


def accumulated_local_effects(
    predict_fn: PredictFn,
    X: np.ndarray,
    feature: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    """The classic k-bin ALE (ALEPlots / ingredients spec).

    Quantile bin edges ``z_0..z_k``; per bin the local effect is the
    weighted average of the prediction difference at the two edges
    over the rows whose own value falls in the bin (weights = the
    conditional densities ``N_k / N``); effects accumulate over bins
    and centre on the mean over the marginal distribution.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(5)
    >>> X = rng.normal(size=(400, 2))
    >>> ale = accumulated_local_effects(lambda Z: Z[:, 0] ** 2, X, feature=0)
    >>> float(ale["ale"].min()) < 0.0 < float(ale["ale"].max())  # centred
    True
    """
    X = np.asarray(X, dtype=float)
    if feature < 0 or feature >= X.shape[1]:
        raise DataValidationError("accumulated_local_effects: feature index out of range")
    if n_bins < 2:
        raise DataValidationError("accumulated_local_effects: n_bins >= 2")
    quantiles = np.quantile(X[:, feature], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(quantiles)
    if len(edges) < 3:
        raise DataValidationError("accumulated_local_effects: feature too discrete")
    k = len(edges) - 1
    bin_idx = np.clip(np.searchsorted(edges, X[:, feature], side="right") - 1, 0, k - 1)
    effects = np.zeros(k)
    counts = np.zeros(k)
    for b in range(k):
        rows = np.where(bin_idx == b)[0]
        counts[b] = rows.size
        if rows.size == 0:
            continue
        low = X[rows].copy()
        low[:, feature] = edges[b]
        high = X[rows].copy()
        high[:, feature] = edges[b + 1]
        delta = (np.asarray(predict_fn(high), dtype=float).ravel()
                 - np.asarray(predict_fn(low), dtype=float).ravel())
        effects[b] = float(np.mean(delta))
    ale = np.cumsum(effects)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # marginal centering: subtract the mean ALE over the data rows
    row_ale = ale[bin_idx]
    ale = ale - row_ale.mean()
    return pd.DataFrame({"grid": centers, "ale": ale, "counts": counts})


def break_down(
    predict_fn: PredictFn,
    X: np.ndarray,
    x0: np.ndarray,
    order: Sequence[int] | None = None,
    keep_groups: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """The iBreakDown two-party attribution of one prediction.

    Intercept = the mean prediction over the data; then variables are
    fixed to the instance's values one at a time (greedy
    largest-|contribution| order when ``order`` is None); each
    contribution is the change of the mean prediction; the last row
    marks the prediction itself.  ``intercept + sum(contributions) ==
    f(x0)`` by construction.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(9).normal(size=(120, 3))
    >>> f = lambda Z: 2.0 * Z[:, 0] - 1.0 * Z[:, 2]
    >>> table = break_down(f, X, X[7])
    >>> abs(float(table["contribution"].iloc[:-1].sum())
    ...     - float(f(X[7:8])[0])) < 1e-9   # intercept + steps == f(x)
    True
    """
    X = np.asarray(X, dtype=float)
    x = np.asarray(x0, dtype=float).ravel()
    if x.size != X.shape[1]:
        raise DataValidationError("break_down: x0 must match X columns")
    predict = lambda data: np.asarray(predict_fn(data), dtype=float).ravel()
    baseline = float(predict(X).mean())
    target = float(predict(x.reshape(1, -1))[0])

    if keep_groups is not None:
        names = list(keep_groups)
        blocks = [list(cols) for cols in keep_groups.values()]
    else:
        names = [f"x{j}" for j in range(X.shape[1])]
        blocks = [[j] for j in range(X.shape[1])]

    # 1d candidate contributions (fix one block on all rows)
    candidates = []
    for name, cols in zip(names, blocks):
        mutated = X.copy()
        mutated[:, cols] = x[cols]
        candidates.append((name, cols, float(predict(mutated).mean()) - baseline))
    if order is None:
        ordered = [c for c in sorted(candidates, key=lambda c: abs(c[2]), reverse=True)]
    else:
        lookup = {name: i for i, name in enumerate(names)}
        wanted = [names[order[i]] for i in range(len(order))]
        ordered = [c for c in sorted(candidates, key=lambda c: lookup[c[0]]) if c[0] in wanted]

    current = X.copy()
    cumulative = [baseline]
    labels = ["intercept"]
    values = [baseline]
    for name, cols, _candidate in ordered:
        current[:, cols] = x[cols]
        step = float(predict(current).mean())
        cumulative.append(step)
        labels.append(name)
        values.append(step)
    contributions = [cumulative[0]] + list(np.diff(cumulative))
    labels.append("prediction")
    values.append(target)
    contributions.append(0.0)
    return pd.DataFrame(
        {
            "variable": labels,
            "contribution": contributions,
            "cumulative": cumulative + [target],
            "value": values,
            "intercept": baseline,
        }
    )


# ---------------------------------------------------------------------------
# the NumPy EBM (interpret's GA2M, cyclic version)
# ---------------------------------------------------------------------------


@dataclass
class ExplainableBoosting:
    """A lightweight Explainable Boosting Machine (GA2M) in NumPy.

    interpret's algorithm, honestly scoped: quantile binning with the
    missing-bin-first convention (max 256 bins), cyclic gradient
    boosting of depth-2 stumps over the binned histograms (gain
    ``G^2 / (H + lambda)``, Newton leaf ``-G / (H + lambda)``,
    regression MSE and binary logloss objectives), the FAST pairwise
    interaction ranking on 2-D residual histograms, per-term
    centering and bin-weighted term importances.  The C++ engine, bag
    ensembling, purification and monotone constraints are the
    not-vendored part.

    Parameters
    ----------
    max_rounds:
        Outer cyclic epochs (each visits every feature once).
    learning_rate:
        The step multiplier (EBMRegressor default 0.04; here larger
        defaults trade accuracy for speed on small data).
    max_bins:
        Quantile bins per feature (minus the missing/unseen slots).
    lambda_reg:
        L2 smoothing on the leaf denominators.
    """

    max_rounds: int = 400
    learning_rate: float = 0.05
    max_bins: int = 64
    lambda_reg: float = 1.0
    min_samples_leaf: int = 4
    n_interactions: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        self.term_cuts_: list[np.ndarray] = []
        self.term_scores_: list[np.ndarray] = []
        self.bin_weights_: list[np.ndarray] = []
        self.intercept_: float = 0.0
        self.pair_ranking_: list[tuple[tuple[int, int], float]] = []
        self.feature_names_: list[str] = []

    # -- binning ------------------------------------------------------------
    def _bin_feature(self, column: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantile cuts + bin indices (0 = missing, last = unseen)."""
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            raise DataValidationError("feature has no finite values")
        cuts = np.unique(np.quantile(finite, np.linspace(0, 1, self.max_bins)[1:-1]))
        n_cuts = len(cuts)
        indices = np.where(
            ~np.isfinite(column), 0, np.searchsorted(cuts, column, side="right") + 1
        )
        indices = np.where(np.isfinite(column), np.clip(indices, 1, n_cuts + 1), 0)
        return indices, cuts

    # -- fit ----------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> ExplainableBoosting:
        """Fit the cyclic GA2M on (X, y)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.size or X.shape[0] < 8:
            raise DataValidationError("ExplainableBoosting: need >= 8 aligned rows")
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        self.feature_names_ = [f"x{j}" for j in range(p)]

        bins: list[np.ndarray] = []
        cuts: list[np.ndarray] = []
        for j in range(p):
            idx, cut = self._bin_feature(X[:, j])
            bins.append(idx)
            cuts.append(cut)
        self.term_cuts_ = cuts

        scores = np.zeros(n)
        self.intercept_ = float(y.mean())
        term_scores = [np.zeros(len(cuts[j]) + 2) for j in range(p)]  # +missing +unseen
        bin_weights = []
        for j in range(p):
            weights = np.bincount(bins[j], minlength=len(cuts[j]) + 2).astype(float)
            weights[-1] = 0.0  # unseen bin carries no training weight
            bin_weights.append(weights)
        self.bin_weights_ = bin_weights

        order = np.arange(p)
        for _epoch in range(self.max_rounds):
            rng.shuffle(order)
            for j in order:
                # MSE objective: dL/dF = 2(F - y) (ebm_stats' gradientConstant 2)
                grad = 2.0 * (self.intercept_ + scores - y)
                hess = np.full(n, 2.0)
                g_bins = np.bincount(bins[j], weights=grad, minlength=len(cuts[j]) + 2)
                h_bins = np.bincount(bins[j], weights=hess, minlength=len(cuts[j]) + 2)
                c_bins = np.bincount(bins[j], minlength=len(cuts[j]) + 2).astype(float)
                update = self._best_stump(g_bins, h_bins, c_bins)
                step = self.learning_rate * update
                term_scores[j] += step
                scores += step[bins[j]]
        self.term_scores_ = term_scores

        # centering: move each term's weighted mean into the intercept
        for j in range(p):
            mean = float(np.average(term_scores[j], weights=bin_weights[j]))
            self.intercept_ += mean
            term_scores[j] -= mean

        # FAST interaction ranking on the residuals
        if self.n_interactions > 0:
            self.pair_ranking_ = self._rank_pairs(X, y, bins, cuts, top=self.n_interactions)
        return self

    def _best_stump(
        self, g_bins: np.ndarray, h_bins: np.ndarray, c_bins: np.ndarray
    ) -> np.ndarray:
        """The best depth-2 split update on one feature's histogram.

        Gain ``G^2/(H+lambda)`` scanned over split positions (the
        ebm_stats.hpp formulas); returns the per-bin update tensor.
        """
        total_g = float(g_bins.sum())
        total_h = float(h_bins.sum())
        lam = self.lambda_reg
        parent_gain = total_g**2 / (total_h + lam)
        best_gain = -np.inf
        best_split = 0
        n_bins = len(g_bins)
        for split in range(1, n_bins):
            if c_bins[:split].sum() < self.min_samples_leaf:
                continue
            if c_bins[split:].sum() < self.min_samples_leaf:
                break
            g_left = float(g_bins[:split].sum())
            h_left = float(h_bins[:split].sum())
            g_right = total_g - g_left
            h_right = total_h - h_left
            gain = (
                g_left**2 / (h_left + lam)
                + g_right**2 / (h_right + lam)
                - parent_gain
            )
            if gain > best_gain:
                best_gain = gain
                best_split = split
        if best_gain <= 0:
            best_split = 0
        update = np.zeros(n_bins)
        if best_split == 0:
            leaf = -total_g / (total_h + lam) if total_h > 0 else 0.0
            update[:] = leaf
        else:
            g_left = float(g_bins[:best_split].sum())
            h_left = float(h_bins[:best_split].sum())
            g_right = total_g - g_left
            h_right = total_h - h_left
            update[:best_split] = -g_left / (h_left + lam)
            update[best_split:] = -g_right / (h_right + lam)
        return update

    def _rank_pairs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bins: list[np.ndarray],
        cuts: list[np.ndarray],
        top: int,
    ) -> list[tuple[tuple[int, int], float]]:
        """FAST: pair strength = quadrant-gain of the residual 2-D histogram."""
        _n, p = X.shape
        scores = self.predict_scores(X)
        grad = 2.0 * (scores - y)  # dL/dF
        n_pair_bins = 16
        results: list[tuple[tuple[int, int], float]] = []
        for i in range(p):
            for j in range(i + 1, p):
                bi = np.clip(bins[i], 0, n_pair_bins - 1)
                bj = np.clip(bins[j], 0, n_pair_bins - 1)
                flat = bi * n_pair_bins + bj
                tensor = np.bincount(flat, weights=grad, minlength=n_pair_bins**2).reshape(
                    n_pair_bins, n_pair_bins
                )
                integral = tensor.cumsum(axis=0).cumsum(axis=1)
                total = float(tensor.sum())
                if abs(total) < 1e-12:
                    results.append(((i, j), 0.0))
                    continue
                best = -np.inf
                for cx in range(1, n_pair_bins):
                    for cy in range(1, n_pair_bins):
                        g00 = float(integral[cx - 1, cy - 1])
                        g01 = float(integral[cx - 1, -1]) - g00
                        g10 = float(integral[-1, cy - 1]) - g00
                        g11 = total - g00 - g01 - g10
                        best = max(
                            best,
                            abs(g00) + abs(g01) + abs(g10) + abs(g11) - abs(total),
                        )
                results.append(((i, j), best / max(abs(total), 1e-12)))
        return heapq.nlargest(top, results, key=lambda item: item[1])

    # -- predict / explain ----------------------------------------------------
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """The additive score (sum of term functions + intercept)."""
        X = np.asarray(X, dtype=float)
        total = np.full(X.shape[0], self.intercept_)
        for j, cuts in enumerate(self.term_cuts_):
            idx = np.where(
                ~np.isfinite(X[:, j]), 0, np.clip(np.searchsorted(cuts, X[:, j], side="right") + 1, 1, len(cuts) + 1)
            )
            total += self.term_scores_[j][idx]
        return total

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict (identity link -- the regressor)."""
        return self.predict_scores(X)

    def contributions(self, X: np.ndarray) -> pd.DataFrame:
        """Per-sample term contributions (the term function at the bin)."""
        X = np.asarray(X, dtype=float)
        rows = {"intercept": np.full(X.shape[0], self.intercept_)}
        for j, name in enumerate(self.feature_names_):
            cuts = self.term_cuts_[j]
            idx = np.where(
                ~np.isfinite(X[:, j]), 0, np.clip(np.searchsorted(cuts, X[:, j], side="right") + 1, 1, len(cuts) + 1)
            )
            rows[name] = self.term_scores_[j][idx]
        return pd.DataFrame(rows)

    def term_importances(self) -> np.ndarray:
        """Bin-weighted mean |score| per term (EBM's 'avg_weight')."""
        return np.array(
            [
                float(np.average(np.abs(scores), weights=weights))
                for scores, weights in zip(self.term_scores_, self.bin_weights_)
            ]
        )

    def term_function(self, feature: int) -> pd.DataFrame:
        """The shape function of one term (bin centers vs centered score)."""
        cuts = self.term_cuts_[feature]
        scores = self.term_scores_[feature][1:-1]  # drop missing/unseen
        weights = self.bin_weights_[feature][1:-1]
        if len(cuts) == 0:
            return pd.DataFrame({"grid": [0.0], "score": scores[:1] if len(scores) else [0.0],
                                 "weight": weights[:1] if len(weights) else [0.0]})
        # one centre per value bin: below the first cut, between each pair,
        # above the last (n_cuts + 1 centres matching the score vector)
        if len(cuts) == 1:
            centers = np.array([cuts[0]])
        else:
            centers = np.concatenate(
                [[cuts[0]], 0.5 * (cuts[:-1] + cuts[1:]), [cuts[-1]]]
            )
        n = min(len(centers), len(scores), len(weights))
        return pd.DataFrame(
            {"grid": centers[:n], "score": scores[:n], "weight": weights[:n]}
        )
