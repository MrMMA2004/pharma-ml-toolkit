"""SHAP explanations for fitted tree models (100x-154).

The last unabsorbed block of ``D2.py`` (its lines 472-528): the
``shap.TreeExplainer`` beeswarm and the top-20 mean-|SHAP| bar the
upload drew for its stacking pipeline.  Every earlier audit of the
upload left this as the standing "shap is not installed" note; this
module closes it as a first-class, optional-gated surface, exactly
the shape the round-33 audit predicted ("would stay optional-gated"):

* :func:`shap_summary` computes the SHAP values of a fitted TREE
  model (LightGBM / XGBoost / CatBoost / every sklearn tree family)
  and returns the typed :class:`ShapSummary` record -- feature names,
  per-feature mean |SHAP|, the full values matrix, and the feature
  values the beeswarm colours by.
* :func:`pharmaml.viz.ml_plots.plot_shap_importance` draws the
  top-N mean-|SHAP| bar (the upload's own top-20).
* :func:`pharmaml.viz.ml_plots.plot_shap_beeswarm` draws the beeswarm
  in PURE matplotlib -- every (sample, feature) point jittered onto
  its feature's row, coloured by the feature's own value -- so the
  plotting layer never needs shap installed, only the computation
  does.

The honest scope, stated once: ``TreeExplainer`` only, the upload's
own choice -- for non-tree models the typed error names the shipped
model-agnostic sibling (:func:`pharmaml.ml.permutation_feature_importance`,
which works for ANY fitted estimator) instead of silently falling
back to a slow ``KernelExplainer`` approximation.  Multiclass
classification is likewise refused with a typed error (a per-class
SHAP summary is a different figure; the permutation sibling covers
the model-agnostic question today).

Install with::

    pip install 'pharmaml[explain]'

Examples
--------
>>> import numpy as np, pandas as pd
>>> from sklearn.ensemble import GradientBoostingRegressor
>>> rng = np.random.default_rng(42)
>>> frame = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
>>> target = 3.0 * frame["a"] - 1.5 * frame["b"]
>>> model = GradientBoostingRegressor(random_state=42).fit(frame, target)
>>> summary = shap_summary(model, frame)
>>> summary.ranking(top_n=2)          # a dominates, b is negative-shifted
[('a', ...), ('b', ...)]
>>> bool(summary.n_samples == 60)
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import ModelError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: The optional dependency this module gates on, and the extra that
#: installs it (mirrors ml/models.py's ``_REQUIREMENT_EXTRAS`` table).
SHAP_REQUIREMENT = "shap"
SHAP_EXTRA = "explain"


@dataclass(frozen=True, slots=True)
class ShapSummary:
    """SHAP values of a fitted tree model over a feature frame.

    The typed upgrade of ``D2.py``'s inline ``TreeExplainer`` call:
    the record keeps the feature names, the per-feature mean |SHAP|
    (the bar chart's numbers), the FULL values matrix (the beeswarm's
    points), and the feature values the beeswarm colours by, so
    callers can print, serialise, or plot without re-deriving
    anything.

    Attributes
    ----------
    names:
        Feature labels (column names or positional ``x0, x1, ...``).
    mean_abs:
        Mean |SHAP value| per feature -- the average magnitude of the
        feature's impact on the model output.
    values:
        The full SHAP values matrix, ``(n_samples, n_features)`` (for
        binary classification: the positive class's values).
    feature_values:
        The feature matrix the values were computed on (the beeswarm
        colours each point by its feature's own value).
    base_value:
        The explainer's expected model output (the beeswarm's centre
        of gravity), when the explainer exposes one.
    model_name:
        The estimator's class name (provenance for the caption).
    task:
        ``"regression"`` or ``"classification"``.
    n_samples:
        Rows in the explained frame.
    """

    names: list[str]
    mean_abs: np.ndarray
    values: np.ndarray
    feature_values: np.ndarray
    base_value: float | None
    model_name: str
    task: str
    n_samples: int

    def as_frame(self) -> pd.DataFrame:
        """The summary as a DataFrame (feature, mean_abs_shap columns)."""
        return pd.DataFrame(
            {
                "feature": self.names,
                "mean_abs_shap": np.asarray(self.mean_abs, dtype=float),
            }
        )

    def ranking(self, top_n: int | None = None) -> list[tuple[str, float]]:
        """``[(feature, mean_abs_shap), ...]`` descending.

        Ties resolve first-declared-wins (a stable argsort on the
        negated means), matching the viz layer's convention.
        """
        means = np.asarray(self.mean_abs, dtype=float)
        order = np.argsort(-means, kind="stable")
        keep = len(means) if top_n is None else min(int(top_n), len(means))
        return [(self.names[int(i)], float(means[int(i)])) for i in order[:keep]]

    def __str__(self) -> str:  # pragma: no cover - presentation only
        top = self.ranking(top_n=5)
        rendered = ", ".join(f"{name}={value:.4f}" for name, value in top)
        return (
            f"SHAP summary ({self.model_name}, {self.task}, "
            f"n={self.n_samples}): {rendered}"
        )


def shap_summary(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    *,
    feature_names: list[str] | None = None,
) -> ShapSummary:
    """SHAP values of a fitted tree model over ``X``.

    ``D2.py``'s ``shap.TreeExplainer`` block, upgraded to the house
    contract: typed guards, the record, and an honest refusal (never
    a silent fallback) for anything the tree explainer cannot handle.

    Parameters
    ----------
    model:
        A FITTED tree estimator (LightGBM / XGBoost / CatBoost /
        sklearn's tree families).  Pipelines are refused with the
        typed error -- explain the final step directly.
    X:
        The feature rows to explain (a DataFrame keeps its column
        names; an ndarray gets ``x0, x1, ...``).
    feature_names:
        Labels for the columns; ``None`` reads ``X.columns`` when
        ``X`` is a DataFrame, else the positional names.

    Returns
    -------
    ShapSummary

    Raises
    ------
    DependencyError
        When shap is not installed -- the message carries the exact
        ``pip install 'pharmaml[explain]'`` command.
    ModelError
        When ``model`` is not fitted, not a tree model the explainer
        supports, a pipeline, or a multiclass classifier (each error
        names the permutation-importance sibling that works instead).

    Examples
--------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(40, 2))
    >>> y = X[:, 0] * 2.0
    >>> model = RandomForestRegressor(random_state=42).fit(X, y)
    >>> summary = shap_summary(model, X)
    >>> summary.ranking(top_n=1)[0][0]
    'x0'
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - needs shap absent
        from pharmaml.exceptions import DependencyError

        raise DependencyError(
            "shap_summary requires the optional package 'shap'. "
            "Install with:  pip install 'pharmaml[explain]'"
        ) from exc

    if isinstance(model, __import__("sklearn").pipeline.Pipeline):
        raise ModelError(
            "shap_summary explains one tree estimator, not a Pipeline -- "
            "pass the fitted final step (pipeline.steps[-1][1]) directly"
        )

    if not hasattr(model, "fit"):
        raise ModelError(
            f"model must be a fitted estimator, got {type(model).__name__}"
        )

    # input coercion: 2-D numeric frame/array, honest names
    if isinstance(X, pd.DataFrame):
        names = [str(column) for column in X.columns]
        matrix = X.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2:
            raise ModelError(
                f"X must be 2-dimensional (rows x features), got shape {matrix.shape}"
            )
        names = [f"x{i}" for i in range(matrix.shape[1])]
    if feature_names is not None:
        if len(feature_names) != matrix.shape[1]:
            raise ModelError(
                f"feature_names carries {len(feature_names)} label(s) for "
                f"{matrix.shape[1]} column(s)"
            )
        names = [str(name) for name in feature_names]
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ModelError(
            f"X must carry at least one row and one feature, got shape {matrix.shape}"
        )

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(matrix)
    except Exception as exc:
        raise ModelError(
            f"TreeExplainer could not explain {type(model).__name__} "
            f"({exc}).  TreeExplainer covers LightGBM, XGBoost, CatBoost "
            "and sklearn's tree families -- for any OTHER estimator use "
            "the model-agnostic pharmaml.ml.permutation_feature_importance."
        ) from exc

    # output shape: regression/binary -> (n, features); multiclass ->
    # (n, features, classes); legacy lists wrap the same arrays.
    if isinstance(values, list):
        values = np.asarray(values[1] if len(values) == 2 else values[0])
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        if values.shape[2] == 2:
            values = values[:, :, 1]  # the positive class (binary read)
        else:
            raise ModelError(
                "shap_summary covers regression and binary classification; "
                f"this model explains {values.shape[2]} classes (a per-class "
                "SHAP summary is a different figure -- "
                "pharmaml.ml.permutation_feature_importance answers the "
                "model-agnostic question today)"
            )
    if values.ndim != 2 or values.shape != matrix.shape:
        raise ModelError(
            f"unexpected SHAP values shape {values.shape} for X of shape "
            f"{matrix.shape}"
        )

    base_value = getattr(explainer, "expected_value", None)
    if base_value is not None:
        try:
            base_value = float(np.asarray(base_value).reshape(-1)[0])
        except (TypeError, ValueError):
            base_value = None

    task = "classification" if getattr(model, "classes_", None) is not None else "regression"
    _LOG.debug(
        "shap_summary: %d row(s) x %d feature(s) over %s",
        matrix.shape[0],
        matrix.shape[1],
        type(model).__name__,
    )
    return ShapSummary(
        names=names,
        mean_abs=np.abs(values).mean(axis=0),
        values=values,
        feature_values=matrix,
        base_value=base_value,
        model_name=type(model).__name__,
        task=task,
        n_samples=int(matrix.shape[0]),
    )


__all__ = [
    "SHAP_EXTRA",
    "SHAP_REQUIREMENT",
    "ShapSummary",
    "shap_summary",
]
