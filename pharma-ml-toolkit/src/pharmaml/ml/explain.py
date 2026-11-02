"""SHAP explanations for fitted tree models (100x-154, 100x-160).

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
* :func:`shap_interactions` (100x-160) computes the exact pairwise
  interaction cube -- ``shap.TreeExplainer.shap_interaction_values``,
  the SHAP repo's surface the 100x-154 round left unabsorbed -- and
  returns the typed :class:`ShapInteractions` record (diagonal = main
  effects, symmetric off-diagonal = pairwise interactions).
* :func:`group_difference` (100x-160) splits any :class:`ShapSummary`
  into two cohorts and returns the per-feature mean-SHAP difference
  with a seeded bootstrap confidence interval -- shap's
  ``group_difference`` recipe in pure numpy, no shap import needed.
* :func:`pharmaml.viz.ml_plots.plot_shap_importance` draws the
  top-N mean-|SHAP| bar (the upload's own top-20).
* :func:`pharmaml.viz.ml_plots.plot_shap_beeswarm` draws the beeswarm
  in PURE matplotlib -- every (sample, feature) point jittered onto
  its feature's row, coloured by the feature's own value -- so the
  plotting layer never needs shap installed, only the computation
  does.
* :func:`pharmaml.viz.ml_plots.plot_shap_waterfall` /
  :func:`plot_shap_dependence` / :func:`plot_shap_heatmap` /
  :func:`plot_shap_force` / :func:`plot_shap_group_difference`
  (100x-160) draw the five remaining signature figures of the SHAP
  repo's plot gallery in the same pure-matplotlib dialect.

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
        sklearn's tree families).  (100x-169) a
        :class:`~pharmaml.ml.training.TrainingResult` is unwrapped to
        its ``.model`` (the summary's ``model_name`` becomes the
        registry name), and a sklearn ``Pipeline`` is unwrapped to its
        final step -- the values are computed over the pipeline's OWN
        preprocessed rows, exactly what it predicts on.  Pipelines
        whose preprocessing changes the feature count are refused
        with the typed error naming the manual alternative.
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
        supports, a width-changing pipeline, or a multiclass
        classifier (each error names the permutation-importance
        sibling that works instead).

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

    (100x-169) the registry's impute+scale Pipeline is explained
    whole -- the final step over its own preprocessed rows:

    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipe = make_pipeline(
    ...     StandardScaler(), RandomForestRegressor(random_state=42)
    ... ).fit(X, y)
    >>> summary = shap_summary(pipe, X)
    >>> summary.model_name
    'RandomForestRegressor'
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

    # (100x-169) the GUI hands its TrainingResult over whole (and the
    # registry wraps every estimator in the impute+scale Pipeline) --
    # unwrap both instead of refusing: the TreeExplainer runs on the
    # FINAL step over the PREPROCESSED rows, exactly the rows the
    # pipeline predicts on, so the values explain the fitted model the
    # caller actually has.
    name_hint: str | None = None
    try:
        from pharmaml.ml.training import TrainingResult
    except ImportError:  # pragma: no cover - circular-import paranoia
        TrainingResult = None  # type: ignore[misc, assignment]
    if TrainingResult is not None and isinstance(model, TrainingResult):
        name_hint = str(model.name)
        model = model.model

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

    # (100x-169) Pipeline unwrap: explain the final estimator over its
    # own preprocessed output.  Width-changing preprocessing is
    # refused honestly (the names/values pairing would be a lie), as is
    # a transform failure -- both name the manual unwrap alternative.
    from sklearn.pipeline import Pipeline as _SkPipeline

    if isinstance(model, _SkPipeline):
        if len(model.steps) == 0:
            raise ModelError(
                "an empty Pipeline carries no estimator to explain"
            )
        final = model.steps[-1][1]
        prefix = _SkPipeline(model.steps[:-1])
        source: Any = X if isinstance(X, pd.DataFrame) else matrix
        try:
            transformed = np.asarray(prefix.transform(source), dtype=float)
        except Exception as exc:  # the transformer zoo; re-raised typed below
            raise ModelError(
                "the Pipeline's preprocessing steps could not transform X "
                f"({exc}).  Explain the final step directly: "
                "shap_summary(pipeline.steps[-1][1], "
                "pipeline[:-1].transform(X))"
            ) from exc
        if (
            transformed.ndim != 2
            or transformed.shape[0] != matrix.shape[0]
            or transformed.shape[1] != matrix.shape[1]
        ):
            raise ModelError(
                "shap_summary explains pipelines whose preprocessing keeps "
                f"the feature count fixed: X has {matrix.shape[1]} column(s) "
                f"but the pipeline emits {transformed.shape[1] if transformed.ndim == 2 else 'non-2D'} "
                "-- explain the final step over its transformed rows "
                "directly instead"
            )
        matrix = transformed
        model = final

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
        model_name=name_hint or type(model).__name__,
        task=task,
        n_samples=int(matrix.shape[0]),
    )


@dataclass(frozen=True, slots=True)
class ShapInteractions:
    """Pairwise SHAP interaction values of a fitted tree model (100x-160).

    The typed wrap of ``shap.TreeExplainer.shap_interaction_values`` --
    the SHAP repo's exact pairwise interaction cube, the surface the
    100x-154 round left unabsorbed: ``values`` carries shape
    ``(n_samples, n_features, n_features)`` where the DIAGONAL holds
    each feature's main effect and the symmetric OFF-DIAGONAL holds the
    pairwise interactions (``values[:, i, j] == values[:, j, i]``; the
    pair's contribution enters a prediction once through ``i`` and once
    through ``j``, so the cube's full sum plus the base value
    reconstructs the model output, exactly like the plain values
    matrix does).

    Attributes
    ----------
    names:
        Feature labels (column names or positional ``x0, x1, ...``).
    values:
        The interaction cube, ``(n_samples, n_features, n_features)``;
        diagonal = main effects, symmetric off-diagonal = pairwise
        interactions.
    feature_values:
        The feature matrix the cube was computed on (provenance for
        downstream figures).
    model_name:
        The estimator's class name (provenance for the caption).
    n_samples:
        Rows in the explained frame.
    """

    names: list[str]
    values: np.ndarray
    feature_values: np.ndarray
    model_name: str
    n_samples: int

    def _pair_means(self) -> tuple[list[tuple[str, str]], np.ndarray]:
        """Upper-triangle pairs (diagonal included) and their mean |value|."""
        values = np.asarray(self.values, dtype=float)
        pairs: list[tuple[str, str]] = []
        means: list[float] = []
        for i in range(len(self.names)):
            for j in range(i, len(self.names)):
                pairs.append((self.names[i], self.names[j]))
                means.append(float(np.abs(values[:, i, j]).mean()))
        return pairs, np.asarray(means, dtype=float)

    def as_frame(self) -> pd.DataFrame:
        """The cube's upper triangle as a DataFrame.

        One row per pair INCLUDING the diagonal (``a == b`` is the
        feature's main effect) -- ``f * (f + 1) / 2`` rows with columns
        ``feature_a`` / ``feature_b`` / ``mean_abs_interaction``.
        """
        pairs, means = self._pair_means()
        return pd.DataFrame(
            {
                "feature_a": [pair[0] for pair in pairs],
                "feature_b": [pair[1] for pair in pairs],
                "mean_abs_interaction": means,
            }
        )

    def ranking(self, top_n: int | None = None) -> list[tuple[str, str, float]]:
        """``[(feature_a, feature_b, mean_abs_interaction), ...]`` descending.

        Sorted over the upper triangle (diagonal included -- the main
        effects compete with the interactions); ties resolve
        first-declared-wins, matching the viz layer's convention.
        """
        pairs, means = self._pair_means()
        order = np.argsort(-means, kind="stable")
        keep = len(means) if top_n is None else min(int(top_n), len(means))
        return [
            (pairs[int(i)][0], pairs[int(i)][1], float(means[int(i)]))
            for i in order[:keep]
        ]

    def __str__(self) -> str:  # pragma: no cover - presentation only
        top = self.ranking(top_n=5)
        rendered = ", ".join(f"{a} x {b}={value:.4f}" for a, b, value in top)
        return (
            f"SHAP interactions ({self.model_name}, n={self.n_samples}): "
            f"{rendered}"
        )


def shap_interactions(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    *,
    feature_names: list[str] | None = None,
) -> ShapInteractions:
    """Pairwise SHAP interaction values of a fitted tree model over ``X``.

    The SHAP repo's ``shap_interaction_values`` surface (its
    ``explainers/_tree.py``), absorbed with the same guard chain
    :func:`shap_summary` established in 100x-154: the exact pairwise
    interaction cube from ``shap.TreeExplainer``, never an
    approximation.  The honest scope is identical too --
    ``TreeExplainer`` only, mirroring ``shap_summary``'s gate; for
    non-tree models the typed error names the model-agnostic
    permutation sibling instead of silently falling back to a slow
    KernelSHAP-style approximation.

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
    ShapInteractions
        The ``(n_samples, n_features, n_features)`` cube -- diagonal
        = main effects, symmetric off-diagonal = pairwise interactions.

    Raises
    ------
    DependencyError
        When shap is not installed -- the message carries the exact
        ``pip install 'pharmaml[explain]'`` command.
    ModelError
        When ``model`` is not fitted, not a tree model the explainer
        supports, a pipeline, a multiclass classifier, or the returned
        cube does not match ``X``'s shape.

    Examples
    --------
    The record's contract needs no shap import -- hand-build the cube:

    >>> import numpy as np
    >>> cube = np.zeros((3, 2, 2))
    >>> cube[:, 0, 0] = 0.75                # a's main effect (diagonal)
    >>> cube[:, 1, 1] = 0.25                # b's main effect
    >>> cube[:, 0, 1] = cube[:, 1, 0] = 0.5  # the symmetric a-b interaction
    >>> record = ShapInteractions(
    ...     names=["a", "b"], values=cube,
    ...     feature_values=np.arange(6.0).reshape(3, 2),
    ...     model_name="GBR", n_samples=3)
    >>> record.ranking()                     # main effect first here
    [('a', 'a', 0.75), ('a', 'b', 0.5), ('b', 'b', 0.25)]
    >>> len(record.as_frame())               # f*(f+1)/2 upper-triangle rows
    3
    >>> list(record.as_frame()["feature_b"])
    ['a', 'b', 'b']
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - needs shap absent
        from pharmaml.exceptions import DependencyError

        raise DependencyError(
            "shap_interactions requires the optional package 'shap'. "
            "Install with:  pip install 'pharmaml[explain]'"
        ) from exc

    if isinstance(model, __import__("sklearn").pipeline.Pipeline):
        raise ModelError(
            "shap_interactions explains one tree estimator, not a Pipeline "
            "-- pass the fitted final step (pipeline.steps[-1][1]) directly"
        )

    if not hasattr(model, "fit"):
        raise ModelError(
            f"model must be a fitted estimator, got {type(model).__name__}"
        )

    # input coercion: the same 2-D numeric frame/array contract as
    # shap_summary, honest names included.
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
        values = explainer.shap_interaction_values(matrix)
    except Exception as exc:
        raise ModelError(
            f"TreeExplainer could not explain {type(model).__name__} "
            f"({exc}).  TreeExplainer covers LightGBM, XGBoost, CatBoost "
            "and sklearn's tree families -- for any OTHER estimator use "
            "the model-agnostic pharmaml.ml.permutation_feature_importance."
        ) from exc

    # output shape: regression/binary -> (n, features, features);
    # multiclass -> one cube per class (a list, or a trailing class
    # axis); both multiclass forms are refused, never silently
    # class-zero'd.
    if isinstance(values, list):
        if len(values) == 2:
            values = np.asarray(values[1], dtype=float)  # positive class
        elif len(values) > 2:
            raise ModelError(
                "shap_interactions covers regression and binary "
                "classification; this model explains "
                f"{len(values)} classes (a per-class interaction cube is a "
                "different figure -- "
                "pharmaml.ml.permutation_feature_importance answers the "
                "model-agnostic question today)"
            )
        else:
            values = np.asarray(values[0], dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim == 4:
        if values.shape[-1] == 2:
            values = values[..., 1]  # the positive class (binary read)
        else:
            raise ModelError(
                "shap_interactions covers regression and binary "
                "classification; this model explains "
                f"{values.shape[-1]} classes (a per-class interaction cube "
                "is a different figure -- "
                "pharmaml.ml.permutation_feature_importance answers the "
                "model-agnostic question today)"
            )
    expected = (matrix.shape[0], matrix.shape[1], matrix.shape[1])
    if values.ndim != 3 or values.shape != expected:
        raise ModelError(
            f"unexpected SHAP interaction values shape {values.shape} for X "
            f"of shape {matrix.shape} (expected {expected})"
        )

    _LOG.debug(
        "shap_interactions: %d row(s) x %d feature(s) x %d feature(s) over %s",
        matrix.shape[0],
        matrix.shape[1],
        matrix.shape[1],
        type(model).__name__,
    )
    return ShapInteractions(
        names=names,
        values=values,
        feature_values=matrix,
        model_name=type(model).__name__,
        n_samples=int(matrix.shape[0]),
    )


@dataclass(frozen=True, slots=True)
class GroupDifference:
    """Cohort mean-SHAP differences with a bootstrap interval (100x-160).

    The typed wrap of shap's ``group_difference`` plot helper (its
    ``plots/_group_difference.py``): for every feature, the difference
    of the cohort's mean SHAP value from the rest's, with a seeded
    bootstrap confidence interval -- the fairness/cohort decomposition
    the pharma questions actually ask (active-vs-inactive cohorts,
    scaffold classes, high/low-potency splits).

    Attributes
    ----------
    names:
        Feature labels, carried over from the explained summary.
    differences:
        Per-feature ``mean(SHAP | group) - mean(SHAP | rest)``.
    low, high:
        The bootstrap confidence bounds of each difference (default:
        the 5th/95th percentiles of 200 seeded resamples).
    n_group, n_rest:
        Row counts of the two cohorts.
    label:
        The cohort's display name (rides the figure caption).
    """

    names: list[str]
    differences: np.ndarray
    low: np.ndarray
    high: np.ndarray
    n_group: int
    n_rest: int
    label: str

    def as_frame(self) -> pd.DataFrame:
        """The differences as a DataFrame (feature/difference/low/high)."""
        return pd.DataFrame(
            {
                "feature": self.names,
                "difference": np.asarray(self.differences, dtype=float),
                "low": np.asarray(self.low, dtype=float),
                "high": np.asarray(self.high, dtype=float),
            }
        )

    def ranking(self, top_n: int | None = None) -> list[tuple[str, float]]:
        """``[(feature, difference), ...]`` by |difference| descending.

        Ties resolve first-declared-wins, matching the module's other
        rankings.
        """
        differences = np.asarray(self.differences, dtype=float)
        order = np.argsort(-np.abs(differences), kind="stable")
        keep = len(differences) if top_n is None else min(int(top_n), len(differences))
        return [(self.names[int(i)], float(differences[int(i)])) for i in order[:keep]]

    def __str__(self) -> str:  # pragma: no cover - presentation only
        top = self.ranking(top_n=5)
        rendered = ", ".join(f"{name}={value:+.4f}" for name, value in top)
        return (
            f"SHAP group difference ({self.label}: group={self.n_group}, "
            f"rest={self.n_rest}): {rendered}"
        )


def group_difference(
    summary: Any,
    group_mask: Any,
    *,
    label: str = "group",
    n_resamples: int = 200,
    confidence: float = 90.0,
    seed: int = 42,
) -> GroupDifference:
    """Cohort difference of a SHAP summary, with a bootstrap interval.

    PURE numpy -- no shap import anywhere, the computation works off
    any :class:`ShapSummary` record (the plotting layer's freedom,
    extended to the cohort question).  For every feature the point
    estimate is ``mean(SHAP | group) - mean(SHAP | rest)``; the
    confidence bounds are the ``[(100 - confidence) / 2, (100 +
    confidence) / 2]`` percentiles (default 5th/95th) of ``n_resamples``
    seeded bootstrap resamples -- each resample redraws ``len(group)``
    rows with replacement from the group's rows and ``len(rest)`` from
    the rest's, exactly shap's ``group_difference`` recipe.

    Parameters
    ----------
    summary:
        A :class:`ShapSummary` record (only ``names`` and ``values``
        are read).
    group_mask:
        Boolean array / Series / sequence of length ``n_samples`` --
        ``True`` marks the cohort, ``False`` the rest.
    label:
        The cohort's display name (rides the record and the figure).
    n_resamples:
        Bootstrap resample count (default 200, shap's own).
    confidence:
        Interval coverage in percent, within ``(0, 100]`` (default 90).
    seed:
        Seed for the bootstrap draws (the house determinism rule).

    Returns
    -------
    GroupDifference

    Raises
    ------
    ModelError
        If ``summary`` is not a record with ``names``/``values``, the
        mask is not boolean / wrong-length / selects every or no row,
        or ``n_resamples``/``confidence`` fall outside their ranges.

    Examples
    --------
    A hand-built summary whose cohort difference is known exactly:

    >>> import numpy as np
    >>> values = np.array([
    ...     [1.0, 0.25], [0.5, -0.25], [-0.5, 0.5], [-1.0, 0.0]])
    >>> record = ShapSummary(
    ...     names=["a", "b"], mean_abs=np.abs(values).mean(axis=0),
    ...     values=values, feature_values=values,
    ...     base_value=None, model_name="GBR", task="regression", n_samples=4)
    >>> mask = np.array([True, True, False, False])
    >>> result = group_difference(record, mask, label="high potency")
    >>> result.differences.tolist()
    [1.5, -0.25]
    >>> bool((result.low <= result.differences).all())   # the CI brackets
    True
    >>> bool((result.differences <= result.high).all())
    True
    >>> result.ranking(top_n=1)
    [('a', 1.5)]
    >>> result.n_group, result.n_rest, result.label
    (2, 2, 'high potency')
    """
    if not (hasattr(summary, "names") and hasattr(summary, "values")):
        raise ModelError(
            "group_difference needs the ShapSummary record (names/values) "
            f"-- the record any shap_summary call returns, got "
            f"{type(summary).__name__}"
        )
    names = [str(name) for name in summary.names]
    values = np.asarray(summary.values, dtype=float)
    if values.ndim != 2:
        raise ModelError(
            f"summary.values must be 2-dimensional (n_samples x n_features), "
            f"got shape {values.shape}"
        )
    if len(names) != values.shape[1]:
        raise ModelError(
            f"names carries {len(names)} label(s) for {values.shape[1]} column(s)"
        )
    if values.shape[0] < 2:
        raise ModelError(
            "group_difference needs at least two rows to split into cohorts, "
            f"got {values.shape[0]}"
        )

    mask = np.asarray(group_mask)
    if mask.dtype != bool:
        raise ModelError(
            f"group_mask must be boolean (True/False) entries, got dtype "
            f"{mask.dtype}"
        )
    mask = mask.reshape(-1)
    if mask.shape[0] != values.shape[0]:
        raise ModelError(
            f"group_mask carries {mask.shape[0]} entries for "
            f"{values.shape[0]} row(s)"
        )
    if not mask.any() or mask.all():
        raise ModelError(
            "group_mask must leave rows on BOTH sides -- both cohorts need "
            "at least one row (the mask is all True or all False)"
        )
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) \
            or int(n_resamples) < 1:
        raise ModelError(
            f"n_resamples must be a positive integer, got {n_resamples!r}"
        )
    if not 0.0 < float(confidence) <= 100.0:
        raise ModelError(
            f"confidence must lie within (0, 100] percent, got {confidence!r}"
        )

    group_rows = values[mask]
    rest_rows = values[~mask]
    differences = group_rows.mean(axis=0) - rest_rows.mean(axis=0)

    # shap's group_difference recipe: seeded bootstrap resamples, each
    # redrawing len(group) rows with replacement from the group and
    # len(rest) from the rest, percentile interval at the ends.
    rng = np.random.default_rng(seed)
    n_group = int(mask.sum())
    n_rest = int(values.shape[0] - n_group)
    group_draws = rng.integers(0, n_group, size=(int(n_resamples), n_group))
    rest_draws = rng.integers(0, n_rest, size=(int(n_resamples), n_rest))
    bootstrap = (
        group_rows[group_draws].mean(axis=1) - rest_rows[rest_draws].mean(axis=1)
    )
    alpha = (100.0 - float(confidence)) / 2.0
    low = np.percentile(bootstrap, alpha, axis=0)
    high = np.percentile(bootstrap, 100.0 - alpha, axis=0)

    _LOG.debug(
        "group_difference: %d group row(s) vs %d rest row(s), %d resample(s)",
        n_group,
        n_rest,
        int(n_resamples),
    )
    return GroupDifference(
        names=names,
        differences=differences,
        low=np.asarray(low, dtype=float),
        high=np.asarray(high, dtype=float),
        n_group=n_group,
        n_rest=n_rest,
        label=str(label),
    )


__all__ = [
    "SHAP_EXTRA",
    "SHAP_REQUIREMENT",
    "GroupDifference",
    "ShapInteractions",
    "ShapSummary",
    "group_difference",
    "shap_interactions",
    "shap_summary",
]
