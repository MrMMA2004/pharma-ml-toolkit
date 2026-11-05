"""LIME tabular explanations, pure NumPy/sklearn (100x-174).

``marcotcr/lime`` was reviewed file by file for this module: the core
engine ``lime/lime_base.py`` (the exponential kernel, the four feature
selection strategies and the weighted Ridge fit), ``lime_tabular.py``
(the three-way data flow -- interpretable binary matrix, original-space
``inverse`` rows for the black box, scaled matrix for distances),
``discretize.py`` (quartile/decile bins and the truncated-normal
``undiscretize``), ``explanation.py`` (the per-label ``local_exp`` /
``intercept`` / ``score`` / ``local_pred`` contract) and
``submodular_pick.py`` (SP-LIME's greedy coverage).  The behaviour
constants are preserved verbatim so explanations stay numerically
comparable with reference LIME: the kernel
``sqrt(exp(-d^2/sigma^2))``, tabular ``sigma = sqrt(p) * 0.75``,
``num_samples = 5000``, final regressor ``Ridge(alpha=1)``
sample-weighted by the kernel, the ``num_features <= 6 -> forward
selection`` switch, quartile bins at the 25/50/75 percentiles, and the
regression label-duplication convention (label 1 positive weights,
label 0 the negated copy).

What this port keeps and what it drops, stated once:

* **kept** -- the whole tabular pipeline for regression AND
  classification, quartile/decile discretization with the
  truncated-normal sampler, all four selection modes including
  ``'lasso_path'`` (sklearn's ``lars_path``), ``top_labels``,
  the ``Explanation`` data contract with ``as_list`` / ``as_map``, and
  SP-LIME's submodular pick.
* **dropped, honestly** -- the text and image explainers (removal
  perturbation on tokenised strings; superpixel segmentation needs
  scikit-image), the d3/JS notebook HTML rendering (the toolkit's
  figures live in :mod:`pharmaml.viz.ml_plots`), the Latin-hypercube
  ``pyDOE2`` sampling route (``'gaussian'`` only), sparse CSR inputs
  and the keras-shaped recurrent wrapper.  SP-LIME's use of the
  *global* numpy RNG for its subsample shuffle is treated as the bug
  it is: this port shuffles with the explainer's own seeded generator.

Where this sits beside the existing explainability surfaces:
:mod:`pharmaml.ml.explain` explains fitted *tree* models with exact
TreeSHAP; :func:`pharmaml.ml.permutation_feature_importance` ranks
global importance for any estimator.  LIME is the third leg: a
*local, surrogate* explanation of one prediction of any black box
without touching its internals.

Examples
--------
>>> import numpy as np
>>> from sklearn.ensemble import RandomForestClassifier
>>> rng = np.random.default_rng(7)
>>> X = rng.normal(size=(80, 4))
>>> y = (X[:, 0] + 2.0 * X[:, 2] > 0).astype(int)
>>> forest = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)
>>> explainer = LimeTabularExplainer(X, feature_names=["a","b","c","d"],
...                                  random_state=0)
>>> exp = explainer.explain_instance(X[3], forest.predict_proba,
...                                  labels=(1,), num_features=2,
...                                  num_samples=400)
>>> items = exp.as_list(1)
>>> len(items) == 2 and all(isinstance(w, float) for _, w in items)
True
>>> bool(exp.score[1] >= 0.0)   # weighted local R^2 is well defined
True
>>> mapping = exp.as_map()
>>> sorted(mapping) == [1]
True
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pharmaml.exceptions import ModelError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "LimeError",
    "LimeExplanation",
    "LimeTabularExplainer",
    "lime_base_kernel",
    "submodular_pick",
]


class LimeError(ModelError):
    """Raised for malformed LIME inputs (mirrors lime's own LimeError)."""


def lime_base_kernel(d: Any, kernel_width: float) -> Any:
    """LIME's exponential kernel ``sqrt(exp(-d^2 / sigma^2))``.

    Identical to ``lime_base``/``lime_tabular``/``lime_text``/
    ``lime_image`` (all four define the same one-liner): a Gaussian of
    bandwidth ``sigma * sqrt(2)`` clipped to ``(0, 1]``.

    >>> import numpy as np
    >>> w = lime_base_kernel(np.array([0.0, 3.0, 10.0]), 3.0)
    >>> bool(abs(float(w[0]) - 1.0) < 1e-12)  # zero distance -> weight 1
    True
    >>> bool(w[2] < w[1] < w[0])              # monotonically decreasing
    True
    """
    return np.sqrt(np.exp(-(np.asarray(d) ** 2) / kernel_width**2))


# --------------------------------------------------------------------- #
# discretization (lime/discretize.py)
# --------------------------------------------------------------------- #


class _Discretizer:
    """Quartile/decile binner with truncated-normal ``undiscretize``.

    ``BaseDiscretizer``'s mechanics: per column the cut points are the
    unique percentiles (25/50/75 for quartiles, 10..90 for deciles),
    ``discretize`` is ``np.searchsorted(cuts, x)`` per column, and the
    sampled bin ids map back to continuous values by a normal
    ``N(bin_mean, bin_std)`` truncated to ``[bin_min, bin_max]`` --
    lime's sampler, not a uniform draw.
    """

    def __init__(
        self,
        data: np.ndarray,
        feature_names: Sequence[str],
        strategy: str = "quartile",
    ) -> None:
        if strategy not in ("quartile", "decile"):
            raise LimeError(f"unknown discretizer strategy {strategy!r}")
        pct = [25, 50, 75] if strategy == "quartile" else list(range(10, 100, 10))
        self.names: list[list[str]] = []
        self.cuts: list[np.ndarray] = []
        self._means: list[np.ndarray] = []
        self._stds: list[np.ndarray] = []
        self._mins: list[np.ndarray] = []
        self._maxs: list[np.ndarray] = []
        for column in range(data.shape[1]):
            values = data[:, column]
            qts = np.unique(np.percentile(values, pct))
            self.cuts.append(qts)
            raw_name = (
                feature_names[column] if column < len(feature_names) else f"x{column}"
            )
            edges = [-np.inf, *qts, np.inf]
            self.names.append(
                [
                    (
                        f"{raw_name} <= {qts[0]:.2f}"
                        if i == 0
                        else (
                            f"{raw_name} > {qts[-1]:.2f}"
                            if i == len(edges) - 2
                            else f"{qts[i - 1]:.2f} < {raw_name} <= {qts[i]:.2f}"
                        )
                    )
                    for i in range(len(edges) - 1)
                ]
            )
            ids = np.searchsorted(qts, values)
            n_bins = len(qts) + 1
            means = np.zeros(n_bins)
            stds = np.zeros(n_bins)
            for b in range(n_bins):
                mask = ids == b
                means[b] = float(values[mask].mean()) if mask.any() else 0.0
                stds[b] = float(values[mask].std()) + 1e-11  # lime's epsilon
            self._means.append(means)
            self._stds.append(stds)
            self._mins.append(np.array([values.min(), *qts]))
            self._maxs.append(np.array([*qts, values.max()]))

    def discretize(self, row: np.ndarray) -> np.ndarray:
        """Map one raw row to per-column bin ids (``searchsorted``)."""
        return np.array(
            [np.searchsorted(self.cuts[c], row[c]) for c in range(len(self.cuts))],
            dtype=float,
        )

    def discretize_frame(self, data: np.ndarray) -> np.ndarray:
        """Map a whole matrix to bin ids."""
        return np.column_stack(
            [np.searchsorted(self.cuts[c], data[:, c]) for c in range(len(self.cuts))]
        )

    def undiscretize(self, sampled: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample continuous values inside each sampled bin.

        Per lime: ``truncnorm`` at the bin's own mean/std, clipped to
        the bin's min/max, degenerate bins keep the edge.
        """
        from scipy.stats import truncnorm

        out = np.array(sampled, dtype=float, copy=True)
        for c in range(len(self.cuts)):
            means, stds = self._means[c], self._stds[c]
            mins, maxs = self._mins[c], self._maxs[c]
            ids = sampled[:, c].astype(int)
            lo = np.clip((mins[ids] - means[ids]) / stds[ids], -1e6, 1e6)
            hi = np.clip((maxs[ids] - means[ids]) / stds[ids], -1e6, 1e6)
            bad = hi <= lo
            if bad.any():  # degenerate bin: keep the lower edge
                out[bad, c] = np.clip(mins[ids[bad]], mins[ids[bad]], maxs[ids[bad]])
                lo[bad] = hi[bad] = 0.0
            keep = ~bad
            if keep.any():
                draws = truncnorm.rvs(
                    lo[keep], hi[keep], loc=means[ids[keep]], scale=stds[ids[keep]],
                    random_state=rng,
                )
                out[keep, c] = draws
        return out


# --------------------------------------------------------------------- #
# the core engine (lime/lime_base.py)
# --------------------------------------------------------------------- #


def _fit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> Any:
    """A sample-weighted Ridge fit (lime's ``Ridge(alpha=...)``)."""
    from sklearn.linear_model import Ridge

    return Ridge(alpha=alpha, fit_intercept=True).fit(matrix, target, sample_weight=weights)


def _weighted_r2(
    model: Any, matrix: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> float:
    score = model.score(matrix, target, sample_weight=weights)
    return float(score)


def _forward_selection(
    data: np.ndarray, labels: np.ndarray, weights: np.ndarray, num_features: int
) -> np.ndarray:
    """Greedy addition with no early stop -- exactly lime's loop."""
    used: list[int] = []
    for _ in range(min(num_features, data.shape[1])):
        best, best_score = 0, -np.inf
        for feature in range(data.shape[1]):
            if feature in used:
                continue
            model = _fit_ridge(
                data[:, used + [feature]], labels, weights, alpha=0.0
            )
            score = _weighted_r2(model, data[:, used + [feature]], labels, weights)
            if score > best_score:
                best, best_score = feature, score
        used.append(best)
    return np.array(used)


def _highest_weights(
    data: np.ndarray, labels: np.ndarray, weights: np.ndarray, num_features: int
) -> np.ndarray:
    """Rank by |Ridge(alpha=0.01) coef * instance value| (lime's route)."""
    model = _fit_ridge(data, labels, weights, alpha=0.01)
    ranked = np.argsort(np.abs(model.coef_ * data[0]))[::-1]
    return ranked[:num_features]


def _lasso_path(
    data: np.ndarray, labels: np.ndarray, weights: np.ndarray, num_features: int
) -> np.ndarray:
    """Walk lars_path backwards for the first step with <= num_features."""
    from sklearn.linear_model import lars_path

    centered = data - np.average(data, axis=0, weights=weights)
    scaled_x = centered * np.sqrt(weights)[:, None]
    centered_y = labels - np.average(labels, weights=weights)
    scaled_y = centered_y * np.sqrt(weights)
    _, _, coefs = lars_path(scaled_x, scaled_y, method="lasso")
    nonzero = np.array(range(data.shape[1]))
    for step in range(len(coefs.T) - 1, 0, -1):
        nonzero = np.nonzero(coefs.T[step])[0]
        if len(nonzero) <= num_features:
            return nonzero
    return nonzero


def _select_features(
    data: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    num_features: int,
    method: str,
) -> np.ndarray:
    if method == "none":
        return np.array(range(data.shape[1]))
    if method == "forward_selection":
        return _forward_selection(data, labels, weights, num_features)
    if method == "highest_weights":
        return _highest_weights(data, labels, weights, num_features)
    if method == "lasso_path":
        return _lasso_path(data, labels, weights, num_features)
    if method == "auto":
        if num_features <= 6:
            return _forward_selection(data, labels, weights, num_features)
        return _highest_weights(data, labels, weights, num_features)
    raise LimeError(f"unknown feature_selection {method!r}")


def explain_instance_with_data(
    neighborhood_data: np.ndarray,
    neighborhood_labels: np.ndarray,
    distances: np.ndarray,
    label: int,
    num_features: int,
    kernel_width: float,
    feature_selection: str = "auto",
    model_regressor: Any = None,
) -> tuple[float, list[tuple[int, float]], float, float]:
    """The ``LimeBase`` contract, verbatim.

    Row 0 of ``neighborhood_data`` must be the original instance.  Fits
    a kernel-weighted ``Ridge(alpha=1)`` on the selected features and
    returns ``(intercept, [(feature, weight)] sorted by |weight|,
    weighted R^2, local prediction)``.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(30, 3))
    >>> y = (2.0 * X[:, 0] - X[:, 2] + 0.1)[:, np.newaxis]
    >>> dists = np.abs(X).sum(axis=1)
    >>> intercept, exp, score, local = explain_instance_with_data(
    ...     X, y, dists, 0, 3, kernel_width=3.0, feature_selection="none")
    >>> bool(abs(exp[0][1] - 2.0) < 0.1)   # feature 0's weight recovered
    True
    """
    weights = np.asarray(lime_base_kernel(distances, kernel_width), dtype=float)
    labels_column = np.asarray(neighborhood_labels)[:, label]
    used = _select_features(
        neighborhood_data, labels_column, weights, num_features, feature_selection
    )
    easy_model = (
        model_regressor
        if model_regressor is not None
        else _fit_ridge(neighborhood_data[:, used], labels_column, weights, alpha=1.0)
    )
    if model_regressor is not None:
        easy_model.fit(
            neighborhood_data[:, used], labels_column, sample_weight=weights
        )
    score = _weighted_r2(
        easy_model, neighborhood_data[:, used], labels_column, weights
    )
    local_pred = float(
        easy_model.predict(neighborhood_data[0, used].reshape(1, -1))[0]
    )
    exp = sorted(
        zip([int(u) for u in used], [float(w) for w in easy_model.coef_]),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    intercept = float(easy_model.intercept_)
    _LOG.debug("lime local fit: %d features, weighted R^2 %.3f", len(used), score)
    return intercept, exp, score, local_pred


# --------------------------------------------------------------------- #
# the tabular explainer (lime/lime_tabular.py)
# --------------------------------------------------------------------- #

#: lime's tabular kernel width default: ``sqrt(n_features) * 0.75``.
KERNEL_WIDTH_FACTOR = 0.75


@dataclass
class LimeExplanation:
    """The ``Explanation`` data contract (per-label dictionaries).

    ``local_exp[label]`` is the ``[(feature_id, weight), ...]`` list
    sorted by |weight|; ``intercept``/``score``/``local_pred`` are the
    surrogate's per-label scalars; ``predict_proba`` is the black
    box's own prediction on the explained row (classification only).
    """

    mode: str = "classification"
    local_exp: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    intercept: dict[int, float] = field(default_factory=dict)
    score: dict[int, float] = field(default_factory=dict)
    local_pred: dict[int, float] = field(default_factory=dict)
    predict_proba: Any = None
    predicted_value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    class_names: list[str] | None = None
    feature_names: list[str] | None = None
    discretized_names: list[str] | None = None

    def available_labels(self) -> list[int]:
        """Labels this explanation carries (classification only)."""
        return sorted(self.local_exp)

    def as_list(self, label: int = 1) -> list[tuple[str, float]]:
        """Map feature ids to human-readable ``(description, weight)``."""
        pairs = self.local_exp.get(label, [])
        out: list[tuple[str, float]] = []
        for fid, weight in pairs:
            if self.discretized_names is not None and fid < len(self.discretized_names):
                name = self.discretized_names[fid]
            elif self.feature_names is not None and fid < len(self.feature_names):
                name = self.feature_names[fid]
            else:
                name = f"x{fid}"
            out.append((name, float(weight)))
        return out

    def as_map(self) -> dict[int, list[tuple[int, float]]]:
        """The raw ``local_exp`` dictionary."""
        return self.local_exp


class LimeTabularExplainer:
    """LIME for tabular data, lime's defaults intact.

    Parameters mirror ``LimeTabularExplainer``: the training matrix
    (continuous columns standardised behind the scenes, categorical
    ones sampled from their training distribution), ``kernel_width``
    defaulting to ``sqrt(p) * 0.75``, quartile discretization of
    continuous features ON by default (every column then becomes a
    binary "same bin as the instance" indicator), and one seeded
    generator driving every random draw.
    """

    def __init__(
        self,
        training_data: Any,
        feature_names: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
        kernel_width: float | None = None,
        discretize_continuous: bool = True,
        discretizer: str = "quartile",
        random_state: int | None = None,
    ) -> None:
        data = np.asarray(training_data, dtype=float)
        if data.ndim != 2 or data.shape[0] < 2:
            raise LimeError("training_data must be a 2-D matrix with >= 2 rows")
        self.training = data
        self.n_features = data.shape[1]
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(self.n_features)]
        )
        self.class_names = list(class_names) if class_names is not None else None
        self._discretized_training: np.ndarray | None = None
        self.categorical_features = (
            list(categorical_features) if categorical_features is not None else []
        )
        self.kernel_width = (
            float(kernel_width)
            if kernel_width is not None
            else float(np.sqrt(self.n_features) * KERNEL_WIDTH_FACTOR)
        )
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        self._means = data.mean(axis=0)
        stds = data.std(axis=0)
        self._stds = np.where(stds > 0, stds, 1.0)

        self.discretizer_instance: _Discretizer | None = None
        self.discretized_names: list[str] | None = None
        to_discretize = [
            c for c in range(self.n_features) if c not in self.categorical_features
        ]
        if discretize_continuous and to_discretize:
            self.discretizer_instance = _Discretizer(
                data[:, to_discretize],
                [self.feature_names[c] for c in to_discretize],
                strategy=discretizer,
            )
            self.categorical_features = list(range(self.n_features))
            discretized = self.discretizer_instance.discretize_frame(
                data[:, to_discretize]
            )
            full_bins = np.zeros_like(data)
            for position, column in enumerate(to_discretize):
                full_bins[:, column] = discretized[:, position]
            self._discretized_training = full_bins
            instance_bins = self.discretizer_instance.discretize(
                data[0]
            )  # placeholder, real use in explain
            self.discretized_names = ["" for _ in range(self.n_features)]
            _ = instance_bins
        else:
            self._discretized_training = None

        # categorical value/frequency tables (discretized columns incl.)
        source = (
            self._discretized_training
            if self._discretized_training is not None
            else data
        )
        self._cat_values: dict[int, np.ndarray] = {}
        self._cat_freqs: dict[int, np.ndarray] = {}
        for column in self.categorical_features:
            counts: dict[float, int] = {}
            for value in source[:, column]:
                counts[float(value)] = counts.get(float(value), 0) + 1
            values = np.array(sorted(counts))
            freqs = np.array([counts[v] for v in values], dtype=float)
            self._cat_values[column] = values
            self._cat_freqs[column] = freqs / freqs.sum()
            # lime neutralises the scaler for categorical columns
            self._means[column] = 0.0
            self._stds[column] = 1.0

    # -- the three-way data flow ------------------------------------- #

    def _data_inverse(
        self, data_row: np.ndarray, num_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perturbation: binary ``data`` + original-space ``inverse``."""
        rng = np.random.default_rng(self.random_state)
        data = rng.normal(0.0, 1.0, size=(num_samples, self.n_features))
        inverse = data * self._stds + self._means
        first_row = np.array(data_row, dtype=float)

        if self.discretizer_instance is not None:
            first_row = self.discretizer_instance.discretize(first_row)

        for column in self.categorical_features:
            values = self._cat_values[column]
            freqs = self._cat_freqs[column]
            draws = rng.choice(values, size=num_samples, replace=True, p=freqs)
            binary = (draws == first_row[column]).astype(float)
            binary[0] = 1.0
            draws[0] = first_row[column]
            data[:, column] = binary
            inverse[:, column] = draws

        if self.discretizer_instance is not None:
            inverse[1:] = self.discretizer_instance.undiscretize(inverse[1:], rng)
        else:
            for column in range(self.n_features):
                if column not in self.categorical_features:
                    inverse[0, column] = first_row[column]
        data[0, :] = 0.0
        for column in self.categorical_features:
            data[0, column] = 1.0
        if self.discretizer_instance is not None:
            inverse[0] = np.asarray(data_row, dtype=float)
        return data, inverse

    def _row_description(self, data_row: np.ndarray) -> list[str]:
        """Feature descriptions for one explained row."""
        names: list[str] = []
        for column in range(self.n_features):
            if self.discretizer_instance is not None:
                position = (
                    column
                    if column not in self.categorical_features
                    else -1
                )
                if self.discretizer_instance is not None and column in range(
                    self.n_features
                ):
                    bins = self.discretizer_instance.discretize(
                        np.asarray(data_row, dtype=float)
                    )
                    position = column
                    # discretizer was built over ALL non-categorical cols
                    # in order; since discretize_continuous covered every
                    # column, position == column.
                    names.append(
                        self.discretizer_instance.names[position][int(bins[column])]
                    )
            elif column in self.categorical_features:
                names.append(
                    f"{self.feature_names[column]}={data_row[column]:g}"
                )
            else:
                names.append(self.feature_names[column])
        return names

    # -- the public entry point -------------------------------------- #

    def explain_instance(
        self,
        data_row: Any,
        predict_fn: Callable[[np.ndarray], Any],
        labels: Sequence[int] = (1,),
        top_labels: int | None = None,
        num_features: int = 10,
        num_samples: int = 5000,
        feature_selection: str = "auto",
    ) -> LimeExplanation:
        """Explain one row's prediction with a local weighted Ridge.

        ``predict_fn`` receives the ORIGINAL-space ``inverse`` rows (a
        ``(num_samples, p)`` matrix) and returns either a 1-D array
        (regression) or a ``(num_samples, n_classes)`` matrix
        (classification -- probability columns, like
        ``predict_proba``).
        """
        row = np.asarray(data_row, dtype=float).ravel()
        if row.shape[0] != self.n_features:
            raise LimeError(
                f"data_row has {row.shape[0]} values, expected {self.n_features}"
            )
        data, inverse = self._data_inverse(row, num_samples)
        scaled = (data - self._means) / self._stds
        distances = np.sqrt(((scaled - scaled[0]) ** 2).sum(axis=1))

        yss = np.asarray(predict_fn(inverse))
        explanation = LimeExplanation(
            class_names=self.class_names, feature_names=self.feature_names
        )
        if self.discretizer_instance is not None:
            explanation.discretized_names = self._row_description(row)

        if yss.ndim == 2 and yss.shape[1] > 1:
            explanation.mode = "classification"
            explanation.predict_proba = yss[0]
            if top_labels is not None:
                ordered = np.argsort(yss[0])[-top_labels:]
                label_list = [int(i) for i in ordered[::-1]]
            else:
                label_list = [int(i) for i in labels]
        elif yss.ndim <= 2:
            values = yss.ravel()
            explanation.mode = "regression"
            explanation.predicted_value = float(values[0])
            explanation.min_value = float(values.min())
            explanation.max_value = float(values.max())
            yss = values[:, np.newaxis]
            label_list = [0]
        else:
            raise LimeError("predict_fn must return a 1-D or 2-D array")

        for label in label_list:
            intercept, exp, score, local_pred = explain_instance_with_data(
                scaled,
                yss,
                distances,
                label,
                num_features,
                kernel_width=self.kernel_width,
                feature_selection=feature_selection,
            )
            explanation.intercept[label] = intercept
            explanation.local_exp[label] = exp
            explanation.score[label] = score
            explanation.local_pred[label] = local_pred

        if explanation.mode == "regression":
            # lime's convention: label 1 keeps the weights, label 0 is
            # the negated copy, so both directions are available.
            explanation.local_exp[1] = explanation.local_exp[0]
            explanation.intercept[1] = explanation.intercept[0]
            explanation.score[1] = explanation.score[0]
            explanation.local_pred[1] = explanation.local_pred[0]
            explanation.local_exp[0] = [
                (i, -w) for i, w in explanation.local_exp[1]
            ]
        _LOG.info(
            "lime explanation: mode=%s, %d features, weighted R^2 %.3f",
            explanation.mode,
            len(explanation.local_exp[label_list[0]]),
            explanation.score[label_list[0]],
        )
        return explanation


# --------------------------------------------------------------------- #
# SP-LIME (lime/submodular_pick.py)
# --------------------------------------------------------------------- #


def submodular_pick(
    explainer: LimeTabularExplainer,
    data: Any,
    predict_fn: Callable[[np.ndarray], Any],
    sample_size: int = 100,
    num_exps_desired: int = 5,
    num_features: int = 10,
    top_labels: int | None = 1,
    num_samples: int = 800,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Greedy submodular coverage over candidate LIME explanations.

    SP-LIME's exact objective: build the explanation matrix ``W`` of
    summed weights per feature NAME, take global importances
    ``I_j = (sum_i |W_ij|)^0.5``, and greedily add the row whose
    explanation covers the most unused high-importance features.  The
    upstream shuffle used numpy's global RNG; this port seeds its own.

    Returns ``{"indices": [...], "explanations": [...],
    "importance": {...}}``.

    >>> import numpy as np
>>> from sklearn.ensemble import RandomForestClassifier
>>> rng = np.random.default_rng(3)
>>> X = rng.normal(size=(60, 5))
>>> y = (X[:, 1] > 0).astype(int)
>>> forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
>>> ex = LimeTabularExplainer(X, feature_names=[f"f{i}" for i in range(5)],
...                           random_state=1)
>>> picked = submodular_pick(ex, X[:10], forest.predict_proba,
...                          sample_size=6, num_exps_desired=2,
...                          num_features=3, num_samples=150,
...                          random_state=1)
>>> len(picked["indices"]) == 2 and 0 <= min(picked["indices"])
True
    """
    rng = np.random.default_rng(random_state)
    matrix = np.asarray(data, dtype=float)
    order = rng.permutation(matrix.shape[0])
    chosen = order[: min(sample_size, matrix.shape[0])]

    explanations = [
        explainer.explain_instance(
            matrix[i], predict_fn, num_features=num_features,
            top_labels=top_labels, num_samples=num_samples,
        )
        for i in chosen
    ]

    feature_dict: dict[str, int] = {}
    rows: list[dict[str, float]] = []
    for exp in explanations:
        row: dict[str, float] = {}
        for label in ([1] if exp.mode == "regression" else exp.available_labels()):
            for name, weight in exp.as_list(label):
                if name not in feature_dict:
                    feature_dict[name] = len(feature_dict)
                row[name] = row.get(name, 0.0) + weight
        rows.append(row)

    width = len(feature_dict)
    W = np.zeros((len(rows), width))
    for i, row in enumerate(rows):
        for name, weight in row.items():
            W[i, feature_dict[name]] += weight
    importance = np.sqrt(np.abs(W).sum(axis=0))
    covered = np.abs(W) > 0

    remaining = list(range(len(rows)))
    selected: list[int] = []
    for _ in range(min(num_exps_desired, len(rows))):
        best_index, best_gain = None, -1.0
        for i in remaining:
            union = covered[selected + [i]].any(axis=0)
            gain = float(np.dot(union.astype(float), importance))
            if gain >= best_gain:
                best_gain, best_index = gain, i
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)

    return {
        "indices": [int(chosen[i]) for i in selected],
        "explanations": [explanations[i] for i in selected],
        "importance": {
            name: float(importance[j])
            for name, j in sorted(feature_dict.items(), key=lambda kv: kv[1])
        },
    }
