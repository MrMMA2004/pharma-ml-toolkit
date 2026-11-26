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

    python -m pip install -e '.[explain]'

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
        ``python -m pip install -e '.[explain]'`` command.
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
            "From the toolkit package root, run:  python -m pip install -e '.[explain]'"
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
        ``python -m pip install -e '.[explain]'`` command.
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
            "From the toolkit package root, run:  python -m pip install -e '.[explain]'"
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

# ---------------------------------------------------------------------------
# Round 187 (scratch/gh187_review.md, section "2. shap/shap"): the shap
# explainer family the earlier rounds left behind, ported from the SHAP
# repository as PURE numpy -- the algorithms, not the numba/C plumbing:
#
# * :func:`tree_shap_summary` -- the EXACT path-dependent TreeSHAP (Lundberg
#   et al. 2018) for sklearn's tree ensembles; the port of
#   ``shap/cext/tree_shap.h`` (the ``PathElement`` record, the polynomial
#   ``extend_path`` push, the hot/cold ``unwind_path`` bookkeeping and the
#   max-depth path-buffer guard).
# * :func:`linear_shap` -- ``shap/explainers/_linear.py``: the interventional
#   linear SHAP plus the correlation (covariance-shrinkage) transform.
# * :func:`permutation_shap` -- ``shap/explainers/_permutation.py``'s
#   antithetic permutation sampling of marginal contributions.
# * :func:`exact_shap` -- ``shap/explainers/_exact.py``'s all-coalitions sweep.
# * :func:`shap_links` -- ``shap/links.py``'s identity/logit link pair.
# * :func:`tree_shap_vs_kernel` -- the honesty check against the kernel-style
#   coalition regression (shap_summary's estimator, reimplemented inline so
#   the comparison never needs the optional shap package).
#
# Every explainer runs offline: numpy + the fitted sklearn estimator, no
# shap import anywhere.  Errors reuse this module's typed :class:`ModelError`.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PathFeature:
    """One decision-path record of the TreeSHAP recursion.

    The plain-numpy twin of ``shap/cext/tree_shap.h``'s ``PathElement``:
    ``feature_index`` is the split feature the path entered through,
    ``zero_fraction`` the fraction of the node's training weight that does
    NOT follow the explained row down this branch, ``one_fraction`` the
    fraction that DOES, and ``pweight`` the permutation weight carried by
    the polynomial recursion (paths with ``i`` ones among the first ``i``
    elements).
    """

    feature_index: int
    zero_fraction: float
    one_fraction: float
    pweight: float


def _extend_path(
    path: list[_PathFeature],
    unique_depth: int,
    zero_fraction: float,
    one_fraction: float,
    feature_index: int,
) -> None:
    """Extend the decision path with one zero/one fraction pair.

    ``shap/cext/tree_shap.h`` ``extend_path`` verbatim: the new element
    takes slot ``unique_depth`` (pweight 1 for the empty path, else 0) and
    the permutation weights are re-pushed down the path -- the O(depth)
    polynomial update that makes TreeSHAP polynomial instead of
    exponential.
    """
    path[unique_depth] = _PathFeature(
        feature_index, zero_fraction, one_fraction, 1.0 if unique_depth == 0 else 0.0
    )
    for i in range(unique_depth - 1, -1, -1):
        path[i + 1].pweight += one_fraction * path[i].pweight * (i + 1) / (unique_depth + 1)
        path[i].pweight = zero_fraction * path[i].pweight * (unique_depth - i) / (unique_depth + 1)


def _unwind_path(path: list[_PathFeature], unique_depth: int, path_index: int) -> None:
    """Undo one previous extension (a repeated split on the same feature).

    ``shap/cext/tree_shap.h`` ``unwind_path`` verbatim: the pweights are
    recomputed first (they stay POSITIONAL over the path slots), then the
    element identities shift left over the removed slot.
    """
    one_fraction = path[path_index].one_fraction
    zero_fraction = path[path_index].zero_fraction
    next_one_portion = path[unique_depth].pweight
    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.0:
            tmp = path[i].pweight
            path[i].pweight = next_one_portion * (unique_depth + 1) / ((i + 1) * one_fraction)
            next_one_portion = (
                tmp - path[i].pweight * zero_fraction * (unique_depth - i) / (unique_depth + 1)
            )
        else:
            path[i].pweight = path[i].pweight * (unique_depth + 1) / (
                zero_fraction * (unique_depth - i)
            )
    for i in range(path_index, unique_depth):
        path[i].feature_index = path[i + 1].feature_index
        path[i].zero_fraction = path[i + 1].zero_fraction
        path[i].one_fraction = path[i + 1].one_fraction


def _unwound_path_sum(path: list[_PathFeature], unique_depth: int, path_index: int) -> float:
    """The total permutation weight if slot ``path_index`` were unwound.

    ``shap/cext/tree_shap.h`` ``unwound_path_sum`` verbatim -- this is the
    leaf-side weight ``w`` of the SHAP formula.
    """
    one_fraction = path[path_index].one_fraction
    zero_fraction = path[path_index].zero_fraction
    next_one_portion = path[unique_depth].pweight
    total = 0.0
    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.0:
            tmp = next_one_portion / ((i + 1) * one_fraction)
            total += tmp
            next_one_portion = path[i].pweight - tmp * zero_fraction * (unique_depth - i)
        else:
            total += path[i].pweight / (zero_fraction * (unique_depth - i))
    return total * (unique_depth + 1)


@dataclass(slots=True)
class _ShapTree:
    """A scalar-output sklearn tree prepared for the path-dependent recursion.

    The port of shap's ``SingleTree`` sklearn branch: the tree's arrays are
    COPIED (the recursion rewrites ``values`` in place), classifier columns
    are row-normalised to per-node class fractions before the requested
    output column is selected, ``children_default`` follows sklearn's
    ``missing_go_to_left``, and zero training weights get the shap #3574
    epsilon so the unwind arithmetic can never divide by zero.
    """

    children_left: np.ndarray
    children_right: np.ndarray
    children_default: np.ndarray
    features: np.ndarray
    thresholds: np.ndarray
    values: np.ndarray
    node_sample_weight: np.ndarray
    max_depth: int


def _compute_expectations(tree: _ShapTree, i: int = 0, depth: int = 0) -> int:
    """Rewrite node values as the training-weighted expectation below them.

    ``shap/cext/tree_shap.h`` ``compute_expectations`` verbatim: after this
    pass ``tree.values[0]`` is the path-dependent bias term and the return
    value is the tree's depth (the path-buffer guard's budget).
    """
    if tree.children_right[i] >= 0:
        left = int(tree.children_left[i])
        right = int(tree.children_right[i])
        depth_left = _compute_expectations(tree, left, depth + 1)
        depth_right = _compute_expectations(tree, right, depth + 1)
        left_weight = tree.node_sample_weight[left]
        right_weight = tree.node_sample_weight[right]
        if left_weight == 0.0 and right_weight == 0.0:
            tree.values[i] = 0.0
        else:
            tree.values[i] = (
                left_weight * tree.values[left] + right_weight * tree.values[right]
            ) / (left_weight + right_weight)
        return max(depth_left, depth_right) + 1
    return 0


def _build_shap_tree(
    tree: Any, *, scaling: float, normalize: bool, output_index: int
) -> _ShapTree:
    """Prepare one sklearn ``tree_`` for the recursion (shap's ``SingleTree``).

    ``scaling`` folds the ensemble weights in (1/n_estimators for the
    forests, the learning rate for GradientBoosting, 1 for a lone tree);
    ``normalize`` row-normalises classifier class-count columns to per-node
    fractions before ``output_index`` is selected.
    """
    children_left = np.asarray(tree.children_left, dtype=np.int64)
    children_right = np.asarray(tree.children_right, dtype=np.int64)
    children_default = children_left.copy()
    if hasattr(tree, "missing_go_to_left"):
        children_default = np.where(
            np.asarray(tree.missing_go_to_left, dtype=bool), children_left, children_right
        ).astype(np.int64)
    values = np.asarray(tree.value, dtype=float).reshape(tree.value.shape[0], -1)
    if normalize:
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    if not 0 <= output_index < values.shape[1]:
        raise ModelError(
            f"output_index {output_index} out of range: the tree's value table carries "
            f"{values.shape[1]} column(s)"
        )
    node_sample_weight = np.asarray(tree.weighted_n_node_samples, dtype=float).copy()
    zero_weight = node_sample_weight == 0.0
    if zero_weight.any():
        # shap #3574's fix: zero-weight nodes would produce 0/0 in unwind_path.
        node_sample_weight[zero_weight] = 1e-6
    shap_tree = _ShapTree(
        children_left=children_left,
        children_right=children_right,
        children_default=children_default,
        features=np.asarray(tree.feature, dtype=np.int64),
        thresholds=np.asarray(tree.threshold, dtype=float),
        values=values[:, output_index] * scaling,
        node_sample_weight=node_sample_weight,
        max_depth=0,
    )
    shap_tree.max_depth = _compute_expectations(shap_tree)
    if shap_tree.max_depth > 350:  # pragma: no cover - teaching-scale guard
        raise ModelError(
            f"tree_shap_summary is a teaching-scale pure-numpy port: its recursion needs a "
            f"tree depth up to ~350, got {shap_tree.max_depth} (use the shap package's "
            "C-extension TreeExplainer for production-scale trees)"
        )
    return shap_tree


def _tree_shap_recursive(
    tree: _ShapTree,
    x: np.ndarray,
    x_missing: np.ndarray,
    phi: np.ndarray,
    node_index: int,
    unique_depth: int,
    parent_path: list[_PathFeature],
    parent_zero_fraction: float,
    parent_one_fraction: float,
    parent_feature_index: int,
    condition_fraction: float,
    maxd: int,
) -> None:
    """The path-dependent TreeSHAP recursion for ONE row.

    ``shap/cext/tree_shap.h`` ``tree_shap_recursive`` with the conditioning
    plumbing dropped (this module explains main effects only -- the
    interaction cube remains :func:`shap_interactions`' surface): the
    parent path is value-copied (the C++ ``std::copy``), extended with this
    node's fractions, split features already on the path are unwound and
    re-merged, and the hot/cold children recurse with the zero/one routing
    the paper's Algorithm 2 specifies.
    """
    if condition_fraction == 0.0:
        return
    if unique_depth + 1 >= maxd:  # the C++ path-buffer guard (maxd = depth + 2)
        raise ModelError(  # pragma: no cover - unreachable for well-formed trees
            "the TreeSHAP decision path outgrew its max-depth buffer"
        )
    # the child's path: a value copy of the parent's slots plus fresh scratch
    unique_path = [
        _PathFeature(el.feature_index, el.zero_fraction, el.one_fraction, el.pweight)
        for el in parent_path[:unique_depth]
    ]
    unique_path.append(_PathFeature(-1, 0.0, 0.0, 0.0))
    _extend_path(
        unique_path, unique_depth, parent_zero_fraction, parent_one_fraction,
        parent_feature_index,
    )

    if tree.children_right[node_index] < 0:
        # leaf node: the paper's phi update over every path element
        for i in range(1, unique_depth + 1):
            w = _unwound_path_sum(unique_path, unique_depth, i)
            el = unique_path[i]
            phi[el.feature_index] += (
                w
                * (el.one_fraction - el.zero_fraction)
                * tree.values[node_index]
                * condition_fraction
            )
        return

    # internal node: find which branch is "hot" (meaning x would follow it)
    split_index = int(tree.features[node_index])
    left = int(tree.children_left[node_index])
    right = int(tree.children_right[node_index])
    if x_missing[split_index]:
        hot_index = int(tree.children_default[node_index])
    elif x[split_index] <= tree.thresholds[node_index]:
        hot_index = left
    else:
        hot_index = right
    cold_index = right if hot_index == left else left
    weight = tree.node_sample_weight[node_index]
    hot_zero_fraction = tree.node_sample_weight[hot_index] / weight
    cold_zero_fraction = tree.node_sample_weight[cold_index] / weight
    incoming_zero_fraction = 1.0
    incoming_one_fraction = 1.0

    # see if we have already split on this feature; if so unwind that split
    # so it can be redone for this node (the path-merging step)
    path_index = 0
    while path_index <= unique_depth and unique_path[path_index].feature_index != split_index:
        path_index += 1
    if path_index != unique_depth + 1:
        incoming_zero_fraction = unique_path[path_index].zero_fraction
        incoming_one_fraction = unique_path[path_index].one_fraction
        _unwind_path(unique_path, unique_depth, path_index)
        unique_depth -= 1

    _tree_shap_recursive(
        tree, x, x_missing, phi, hot_index, unique_depth + 1, unique_path,
        hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
        split_index, condition_fraction, maxd,
    )
    _tree_shap_recursive(
        tree, x, x_missing, phi, cold_index, unique_depth + 1, unique_path,
        cold_zero_fraction * incoming_zero_fraction, 0.0,
        split_index, condition_fraction, maxd,
    )


def _tree_shap_row(tree: _ShapTree, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Path-dependent TreeSHAP for ONE row on ONE prepared tree.

    Returns ``(phi, base)`` -- the per-feature SHAP vector and the tree's
    root expectation (the path-dependent bias term).  Their sum is the
    tree's own output for the row, exactly (the property the tests pin).
    """
    phi = np.zeros(x.shape[0], dtype=float)
    x_missing = np.isnan(x)
    _tree_shap_recursive(
        tree, x, x_missing, phi, 0, 0, [], 1.0, 1.0, -1, 1.0, tree.max_depth + 2
    )
    return phi, float(tree.values[0])


def _gb_init_raw(model: Any, n_features: int, n_outputs: int) -> np.ndarray:
    """The GradientBoosting init stage's raw prediction (folded into the base).

    sklearn's own ``_raw_predict_init`` answers exactly for every init
    estimator (constant / prior / zero); the manual fallbacks cover the
    same estimators should that private helper ever move.
    """
    anchor = np.zeros((1, n_features))
    helper = getattr(model, "_raw_predict_init", None)
    if callable(helper):
        try:
            return np.asarray(helper(anchor), dtype=float).reshape(-1)[:n_outputs]
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - older inits
            pass  # fall through to the manual constant / prior forms below
    init = getattr(model, "init_", None)
    constant = getattr(init, "constant_", None)
    if constant is not None:
        return np.asarray(constant, dtype=float).reshape(-1)[:n_outputs]
    prior = getattr(init, "class_prior_", None)
    if prior is not None and n_outputs == 1:
        positive = float(np.asarray(prior, dtype=float)[1])
        return np.array([np.log(positive / (1.0 - positive))])
    return np.zeros(n_outputs)  # pragma: no cover - exotic init estimators


@dataclass(frozen=True, slots=True)
class TreeShapSummary:
    """Exact path-dependent TreeSHAP values of a fitted sklearn tree ensemble.

    The typed record mirrors :class:`ShapSummary` field-for-field (the viz
    layer renders both), but ``values`` is the EXACT path-dependent TreeSHAP
    matrix (Lundberg et al. 2018) computed in pure numpy -- no shap import,
    no sampling, no approximation -- and ``base_value`` is the ensemble's
    root expectation.  The exactness contract:
    ``values.sum(axis=1) + base_value`` reconstructs the model output the
    row scores carry (``predict`` for regressors, the summed per-tree class
    probability for forest classifiers, ``decision_function``'s margin for
    GradientBoosting classifiers) up to float rounding -- the property the
    doctests and tests pin at 1e-6.

    Attributes
    ----------
    names:
        Feature labels (column names or positional ``x0, x1, ...``).
    mean_abs:
        Mean |SHAP value| per feature -- the bar chart's numbers.
    values:
        The full SHAP values matrix, ``(n_samples, n_features)``.
    feature_values:
        The feature matrix the values were computed on.
    base_value:
        The path-dependent expected value (the root's training
        expectation, plus the GradientBoosting init stage's raw score).
    model_name:
        The estimator's class name (provenance for the caption).
    task:
        ``"regression"`` or ``"classification"``.
    n_samples:
        Rows in the explained frame (after any ``max_rows`` cap).
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
        """``[(feature, mean_abs_shap), ...]`` descending, stable on ties."""
        means = np.asarray(self.mean_abs, dtype=float)
        order = np.argsort(-means, kind="stable")
        keep = len(means) if top_n is None else min(int(top_n), len(means))
        return [(self.names[int(i)], float(means[int(i)])) for i in order[:keep]]

    def __str__(self) -> str:  # pragma: no cover - presentation only
        top = self.ranking(top_n=5)
        rendered = ", ".join(f"{name}={value:.4f}" for name, value in top)
        return (
            f"TreeSHAP summary ({self.model_name}, {self.task}, "
            f"n={self.n_samples}): {rendered}"
        )


def tree_shap_summary(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    *,
    max_rows: int | None = None,
    output_index: int = 0,
) -> TreeShapSummary:
    """Exact path-dependent TreeSHAP values for sklearn tree ensembles.

    The port of the SHAP repo's ``TreeExplainer`` with
    ``feature_perturbation="tree_path_dependent"`` (its C core lives in
    ``shap/cext/tree_shap.h``): the per-tree polynomial push recursion is
    reproduced line-for-line in numpy, so the values are EXACT -- no
    coalitions are sampled.  The ensemble wiring follows shap's own sklearn
    loader: forest trees carry ``1 / n_estimators``, GradientBoosting
    trees carry the ``learning_rate`` and the init stage's raw prediction
    is folded into the base value.

    Per-tree SHAP vectors sum (linearity of Shapley values), and each
    tree's vector plus its root expectation reconstructs that tree's own
    leaf output for the row -- so the record's
    ``values.sum(axis=1) + base_value`` equals the model output EXACTLY.

    Parameters
    ----------
    model:
        A FITTED sklearn tree estimator: ``DecisionTree{Regressor,}``,
        ``RandomForest{Regressor,Classifier}``,
        ``ExtraTrees{Regressor,Classifier}`` or
        ``GradientBoosting{Regressor,Classifier}``.
    X:
        The feature rows to explain (a DataFrame keeps its column names;
        an ndarray gets ``x0, x1, ...``).
    max_rows:
        Cap on the explained rows (the deterministic head of ``X``) --
        the teaching-scale guard for wide sweeps.
    output_index:
        Which raw-score column to explain.  Regressors expose one column;
        forest/tree classifiers index ``classes_`` (binary positive class
        = 1); GradientBoosting classifiers index the raw margin's class
        columns (binary has the single class-1 margin column, so 0).

    Returns
    -------
    TreeShapSummary

    Raises
    ------
    ModelError
        When ``model`` is not a fitted sklearn tree ensemble, ``X`` is not
        a 2-D numeric table, ``max_rows`` is not a positive integer, or
        ``output_index`` falls outside the model's raw-score columns.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> rng = np.random.default_rng(7)
    >>> X = rng.normal(size=(40, 4))
    >>> y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
    >>> model = GradientBoostingRegressor(random_state=7, n_estimators=10).fit(X, y)
    >>> record = tree_shap_summary(model, X)
    >>> record.values.shape
    (40, 4)
    >>> residual = float(np.abs(
    ...     record.values.sum(axis=1) + record.base_value - model.predict(X)
    ... ).max())
    >>> residual < 1e-6      # the exactness property: phi sums to f(x) - base
    True
    >>> record.ranking(top_n=1)[0][0]
    'x0'
    """
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
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ModelError(
            f"X must carry at least one row and one feature, got shape {matrix.shape}"
        )
    if max_rows is not None:
        if (
            isinstance(max_rows, bool)
            or not isinstance(max_rows, (int, np.integer))
            or int(max_rows) < 1
        ):
            raise ModelError(f"max_rows must be a positive integer, got {max_rows!r}")
        matrix = matrix[: int(max_rows)]
    if isinstance(output_index, bool) or not isinstance(output_index, (int, np.integer)):
        raise ModelError(f"output_index must be an integer, got {output_index!r}")
    output_index = int(output_index)

    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    # (sklearn tree_, scaling, normalize) per tree -- shap's loader wiring.
    specs: list[tuple[Any, float, bool]] = []
    init_raw: np.ndarray | None = None
    task = "regression"
    if isinstance(model, GradientBoostingRegressor):
        specs = [
            (estimator.tree_, float(model.learning_rate), False)
            for estimator in model.estimators_[:, 0]
        ]
        init_raw = _gb_init_raw(model, matrix.shape[1], 1)
    elif isinstance(model, GradientBoostingClassifier):
        n_outputs = int(model.estimators_.shape[1])
        if not 0 <= output_index < n_outputs:
            raise ModelError(
                f"output_index {output_index} out of range: this GradientBoostingClassifier "
                f"explains {n_outputs} class column(s) of its raw margin"
            )
        init_raw = _gb_init_raw(model, matrix.shape[1], n_outputs)
        specs = [
            (estimator.tree_, float(model.learning_rate), False)
            for estimator in model.estimators_[:, output_index]
        ]
        task = "classification"
    elif isinstance(model, (RandomForestRegressor, ExtraTreesRegressor)):
        scaling = 1.0 / len(model.estimators_)  # output is the average of trees
        specs = [(estimator.tree_, scaling, False) for estimator in model.estimators_]
    elif isinstance(model, (RandomForestClassifier, ExtraTreesClassifier)):
        scaling = 1.0 / len(model.estimators_)
        specs = [(estimator.tree_, scaling, True) for estimator in model.estimators_]
        task = "classification"
    elif isinstance(model, DecisionTreeRegressor):
        specs = [(model.tree_, 1.0, False)]
    elif isinstance(model, DecisionTreeClassifier):
        specs = [(model.tree_, 1.0, True)]
        task = "classification"
    else:
        raise ModelError(
            f"tree_shap_summary ports the path-dependent TreeSHAP for sklearn's tree "
            f"ensembles (DecisionTree / RandomForest / ExtraTrees / GradientBoosting), "
            f"got {type(model).__name__} -- other tree libraries (LightGBM / XGBoost / "
            "CatBoost) stay with shap_summary; other model families use this module's "
            "model-agnostic permutation_shap / exact_shap"
        )

    n_outputs = 1 if task == "regression" else len(model.classes_)
    if not 0 <= output_index < n_outputs:
        raise ModelError(
            f"output_index {output_index} out of range: this {type(model).__name__} "
            f"exposes {n_outputs} raw output column(s)"
        )

    n_rows, n_features = matrix.shape
    values = np.zeros((n_rows, n_features), dtype=float)
    base = float(init_raw[output_index]) if init_raw is not None else 0.0
    # GradientBoosting's class/output pick already happened through
    # ``estimators_[:, output_index]`` (and ``init_raw[output_index]``);
    # each boosted tree's value table carries exactly ONE column, so the
    # tree-level pick is 0 there.  The forest classifiers pick the class
    # column INSIDE each tree's class-count table (the normalize path).
    tree_output_index = 0 if init_raw is not None else output_index
    for tree_, _scaling, normalize in specs:
        shap_tree = _build_shap_tree(
            tree_, scaling=_scaling, normalize=normalize, output_index=tree_output_index
        )
        base += float(shap_tree.values[0])  # the root expectation, ONCE per tree
        for row_index in range(n_rows):
            phi_row, _tree_base = _tree_shap_row(shap_tree, matrix[row_index])
            values[row_index] += phi_row  # the scaling rides inside tree.values

    _LOG.debug(
        "tree_shap_summary: %d row(s) x %d feature(s) over %d tree(s) of %s",
        n_rows,
        n_features,
        len(specs),
        type(model).__name__,
    )
    return TreeShapSummary(
        names=names,
        mean_abs=np.abs(values).mean(axis=0),
        values=values,
        feature_values=matrix,
        base_value=base,
        model_name=type(model).__name__,
        task=task,
        n_samples=int(n_rows),
    )


# ---------------------------------------------------------------------------
# The model-agnostic machinery: the scorer contract, the background-marginal
# expectation v(S), then the linear / permutation / exact explainers, the
# links, and the tree-vs-kernel honesty check.
# ---------------------------------------------------------------------------


def _score_matrix(model: Any, matrix: np.ndarray, output_index: int) -> np.ndarray:
    """Evaluate the module's model-agnostic scorer over ``matrix`` rows.

    Accepts, in order: a sklearn-style ``predict_proba`` (classification --
    the ``output_index``-th class probability), a ``predict`` (regression
    or raw scores), or a raw callable ``matrix -> scores`` -- the estimator
    contract the rest of this module documents.  Returns a 1-D score per
    row.
    """
    predict_proba = getattr(model, "predict_proba", None)
    predict = getattr(model, "predict", None)
    if callable(predict_proba):
        scores = np.asarray(predict_proba(matrix), dtype=float)
    elif callable(predict):
        scores = np.asarray(predict(matrix), dtype=float)
    elif callable(model):
        scores = np.asarray(model(matrix), dtype=float)
    else:
        raise ModelError(
            f"the model must expose predict / predict_proba or be a callable "
            f"matrix -> scores, got {type(model).__name__}"
        )
    if scores.ndim == 1:
        return scores
    if scores.ndim == 2:
        if scores.shape[1] == 1:
            return scores[:, 0]
        if 0 <= output_index < scores.shape[1]:
            return scores[:, output_index]
        raise ModelError(
            f"output_index {output_index} out of range for "
            f"{scores.shape[1]} output column(s)"
        )
    raise ModelError(f"the scorer must return 1-D or 2-D scores, got shape {scores.shape}")


def _masked_scores(
    model: Any,
    background: np.ndarray,
    x: np.ndarray,
    masks: np.ndarray,
    output_index: int,
) -> np.ndarray:
    """The background-marginal expectation ``v(S)`` for each boolean mask row.

    One masked evaluation per (coalition, background row): features IN the
    coalition keep the explained row's values, the rest are replaced by the
    background row's -- then the mean over the background is the
    interventional conditional expectation the SHAP paper defines.
    """
    if masks.shape[0] == 0:
        return np.zeros(0, dtype=float)
    blocked = np.where(masks[:, None, :], x[None, None, :], background[None, :, :])
    scores = _score_matrix(model, blocked.reshape(-1, background.shape[1]), output_index)
    return scores.reshape(masks.shape[0], background.shape[0]).mean(axis=1)


def _coerce_background_and_row(
    X: pd.DataFrame | np.ndarray, row: Any
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The shared input coercion of the row-level explainers."""
    if isinstance(X, pd.DataFrame):
        names = [str(column) for column in X.columns]
        background = X.to_numpy(dtype=float)
    else:
        background = np.asarray(X, dtype=float)
        if background.ndim != 2:
            raise ModelError(
                f"X must be 2-dimensional (rows x features), got shape {background.shape}"
            )
        names = [f"x{i}" for i in range(background.shape[1])]
    if background.shape[0] == 0 or background.shape[1] == 0:
        raise ModelError(
            f"X must carry at least one row and one feature, got shape {background.shape}"
        )
    x = np.asarray(row, dtype=float).reshape(-1)
    if x.shape[0] != background.shape[1]:
        raise ModelError(
            f"row carries {x.shape[0]} value(s) for {background.shape[1]} feature(s)"
        )
    return background, x, names


def _task_of(model: Any) -> str:
    """``"classification"`` for probabilistic models, else ``"regression"``."""
    return "classification" if callable(getattr(model, "predict_proba", None)) else "regression"


def _model_label(model: Any) -> str:
    """The provenance name for a record (the class, or ``callable``)."""
    name = getattr(model, "__name__", None)
    return str(name) if name else type(model).__name__


def _parse_linear_model(model: Any, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Pull ``(coef, intercept)`` out of a sklearn linear model or a pair.

    shap/explainers/_linear.py ``_parse_model``: either a ``(coef,
    intercept)`` tuple or an estimator exposing ``coef_``/``intercept_``;
    the coef comes back as ``(n_output_rows, n_features)`` for the caller's
    ``output_index`` pick.
    """
    if isinstance(model, tuple) and len(model) == 2:
        coef = np.asarray(model[0], dtype=float)
        intercept = np.asarray(model[1], dtype=float).reshape(-1)
    elif hasattr(model, "coef_") and hasattr(model, "intercept_"):
        coef = np.asarray(model.coef_, dtype=float)
        intercept = np.asarray(model.intercept_, dtype=float).reshape(-1)
    else:
        raise ModelError(
            f"linear_shap needs a linear model exposing coef_/intercept_ (or a "
            f"(coef, intercept) tuple), got {type(model).__name__}"
        )
    coef = coef.reshape(-1, n_features) if coef.ndim > 1 else coef.reshape(1, -1)
    if coef.shape[1] != n_features:
        raise ModelError(
            f"the model's coef_ carries {coef.shape[1]} weight(s) for "
            f"{n_features} feature(s)"
        )
    return coef, intercept


def _estimate_correlation_transforms(
    coef: np.ndarray, cov: np.ndarray, *, nsamples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """shap/_linear.py ``_estimate_transforms``: the correlation-case matrices.

    For each sampled feature permutation the block-matrix-inversion walk
    accumulates the conditional-expectation deltas; averaging over the
    permutations yields ``mean_transform``/``x_transform`` so that
    ``phi = X @ x_transform.T - mean_transform @ mean`` is the
    correlation-aware (conditional) SHAP value.  Faithful port; the one
    deviation is the SEEDED rng (the house determinism rule) where shap
    shuffles numpy's global random state.
    """
    n_features = len(coef)
    mean_transform = np.zeros((n_features, n_features))
    x_transform = np.zeros((n_features, n_features))
    for _ in range(int(nsamples)):
        inds = rng.permutation(n_features)
        cov_inv_ss = np.zeros((0, 0))
        cov_si = np.zeros((n_features, 0))
        for j in range(n_features):
            i = int(inds[j])
            # use the last Si as the new S
            cov_s = cov_si
            cov_inv_s = cov_inv_ss
            cov_si = cov[:, inds[: j + 1]]
            # compute the new cov_inv_SiSi from cov_inv_SS (block inverse)
            d = cov_si[i, :-1].T
            t = np.matmul(cov_inv_s, d)
            u = cov[i, i] - np.matmul(t.T, d)
            cov_inv_ss = np.zeros((j + 1, j + 1))
            if j > 0:
                cov_inv_ss[:-1, :-1] = cov_inv_s + np.outer(t, t) / u
                cov_inv_ss[:-1, -1] = cov_inv_ss[-1, :-1] = -t / u
            cov_inv_ss[-1, -1] = 1.0 / u

            # + coef @ (Q(bar(Sui)) - Q(bar(S)))      (the mean-side delta)
            mean_transform[i, i] += coef[i]
            coef_r_si = np.matmul(
                coef[inds[j + 1 :]],
                np.matmul(cov_si, cov_inv_ss)[inds[j + 1 :]],
            )
            mean_transform[i, inds[: j + 1]] += coef_r_si
            coef_r_s = np.matmul(
                coef[inds[j:]],
                np.matmul(cov_s, cov_inv_s)[inds[j:]],
            )
            mean_transform[i, inds[:j]] -= coef_r_s

            # - coef @ (Q(Sui) - Q(S))                (the x-side delta)
            x_transform[i, i] += coef[i]
            x_transform[i, inds[: j + 1]] += coef_r_si
            x_transform[i, inds[:j]] -= coef_r_s
    return mean_transform / nsamples, x_transform / nsamples


def linear_shap(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    *,
    feature_weights: np.ndarray | None = None,
    output_index: int = 0,
    feature_perturbation: str = "interventional",
) -> ShapSummary:
    """SHAP values of a linear model over ``X`` (shap's ``LinearExplainer``).

    ``X`` is BOTH the background distribution (its per-feature mean and
    covariance) and the rows to explain.  Two perturbation modes:

    * ``"interventional"`` (default) -- features treated as independent,
      the EXACT interventional SHAP of a linear model:
      ``phi = (X - mean) * coef`` with
      ``base = intercept + coef @ mean``.
    * ``"correlation"`` -- shap's correlation-dependent mode: the
      conditional-expectation deltas are accumulated over seeded sampled
      permutations via block matrix inversions (the
      ``_estimate_transforms`` port), sharing credit across correlated
      features.  Zero-variance columns are dropped from the transform (the
      ``valid_inds`` step) and a near-singular covariance gets shap's
      1e-6 ridge.

    ``feature_weights`` is a house shim: optional per-feature multipliers
    on the coefficients (a what-if down-weighting).  The same weights fold
    into the base value, so the record still satisfies
    ``values.sum(axis=1) + base_value == weighted prediction`` exactly.

    Which mode ran is recorded in the summary's ``model_name`` provenance
    (``"LinearRegression (correlation)"``), the ShapSummary record having
    no separate note field.

    Parameters
    ----------
    model:
        A FITTED linear model exposing ``coef_``/``intercept_`` (or a raw
        ``(coef, intercept)`` tuple).
    X:
        Background + explained rows (a DataFrame keeps its column names).
    feature_weights:
        Optional per-feature coefficient multipliers (length n_features,
        finite).
    output_index:
        The coef row to explain for multi-output models (binary sklearn
        classifiers carry a single row, so 0).
    feature_perturbation:
        ``"interventional"`` or ``"correlation"``.

    Returns
    -------
    ShapSummary

    Raises
    ------
    ModelError
        On non-linear models, shape mismatches, non-finite
        ``feature_weights``, an out-of-range ``output_index``, or the
        correlation mode with fewer than two background rows.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> rng = np.random.default_rng(11)
    >>> X = rng.normal(size=(40, 4))
    >>> y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + X[:, 2] + 1.0
    >>> model = LinearRegression().fit(X, y)
    >>> record = linear_shap(model, X)
    >>> residual = float(np.abs(
    ...     record.values.sum(axis=1) + record.base_value - model.predict(X)
    ... ).max())
    >>> residual < 1e-8      # interventional linear SHAP is exact
    True
    >>> correlated = linear_shap(model, X, feature_perturbation="correlation")
    >>> residual = float(np.abs(
    ...     correlated.values.sum(axis=1) + correlated.base_value - model.predict(X)
    ... ).max())
    >>> residual < 1e-6      # the correlation deltas telescope to f(x) - E[f]
    True
    >>> correlated.model_name
    'LinearRegression (correlation)'
    """
    if feature_perturbation not in ("interventional", "correlation"):
        raise ModelError(
            f"feature_perturbation must be 'interventional' or 'correlation', got "
            f"{feature_perturbation!r}"
        )
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
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ModelError(
            f"X must carry at least one row and one feature, got shape {matrix.shape}"
        )
    coef_rows, intercepts = _parse_linear_model(model, matrix.shape[1])
    if coef_rows.shape[0] == 1:
        coef = coef_rows[0]
        intercept = float(intercepts[0])
    else:
        if not 0 <= int(output_index) < coef_rows.shape[0]:
            raise ModelError(
                f"output_index {output_index} out of range for "
                f"{coef_rows.shape[0]} coef row(s)"
            )
        coef = coef_rows[int(output_index)]
        intercept = float(intercepts[int(output_index)]) if intercepts.size > 1 else float(intercepts[0])
    if feature_weights is not None:
        weights = np.asarray(feature_weights, dtype=float).reshape(-1)
        if weights.shape[0] != matrix.shape[1]:
            raise ModelError(
                f"feature_weights carries {weights.shape[0]} weight(s) for "
                f"{matrix.shape[1]} feature(s)"
            )
        if not np.isfinite(weights).all():
            raise ModelError("feature_weights must be finite")
        coef = coef * weights
    if feature_perturbation == "interventional":
        mean = matrix.mean(axis=0)
        values = (matrix - mean) * coef
        base = float(intercept + coef @ mean)
    else:
        if matrix.shape[0] < 2:
            raise ModelError(
                "the correlation perturbation needs at least two background rows to "
                f"estimate a covariance, got {matrix.shape[0]}"
            )
        mean = matrix.mean(axis=0)
        cov = np.atleast_2d(np.asarray(np.cov(matrix, rowvar=False), dtype=float))
        valid = np.flatnonzero(np.diag(cov) > 1e-8)
        if valid.size == 0:  # pragma: no cover - constant columns only
            raise ModelError("every feature is constant; there is nothing to explain")
        mean_valid = mean[valid]
        cov_valid = cov[np.ix_(valid, valid)]
        eigenvalues = np.linalg.eigvalsh(cov_valid)
        if float(eigenvalues.min()) < 1e-7:
            cov_valid = cov_valid + np.eye(valid.size) * 1e-6
        mean_transform, x_transform = _estimate_correlation_transforms(
            coef[valid], cov_valid, nsamples=1000, rng=np.random.default_rng(0)
        )
        values = np.zeros_like(matrix)
        values[:, valid] = matrix[:, valid] @ x_transform.T - mean_transform @ mean_valid
        base = float(intercept + coef @ mean)

    _LOG.debug(
        "linear_shap: %d row(s) x %d feature(s), %s perturbation",
        matrix.shape[0],
        matrix.shape[1],
        feature_perturbation,
    )
    return ShapSummary(
        names=names,
        mean_abs=np.abs(values).mean(axis=0),
        values=values,
        feature_values=matrix,
        base_value=base,
        model_name=f"{_model_label(model)} ({feature_perturbation})",
        task=_task_of(model),
        n_samples=int(matrix.shape[0]),
    )


def permutation_shap(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    row: Any,
    *,
    n_permutations: int = 16,
    seed: int | None = None,
    output_index: int = 0,
) -> ShapSummary:
    """Antithetic-permutation SHAP values of ONE row (shap's ``PermutationExplainer``).

    For each of ``n_permutations`` seeded feature permutations the walk
    through the permutation AND its reverse (the antithetic pair of
    ``shap/explainers/_permutation.py``) accumulates every feature's
    marginal contribution ``v(S + j) - v(S)``, where ``v(S)`` is the
    background-marginal expectation over ``X``'s rows.  The pair averages
    into the returned per-feature vector; ``base_value`` is ``v(empty)``,
    the mean background score.

    For an ADDITIVE scorer (a linear model, say) every permutation's
    contributions are already the exact Shapley values, so the record
    satisfies ``values.sum() + base_value == score(row)`` to float
    precision; for genuinely interactive models this is a Monte-Carlo
    estimate and a seeded ``seed`` keeps it reproducible.

    Parameters
    ----------
    model:
        A fitted estimator (``predict``/``predict_proba``) or a raw
        callable ``matrix -> scores``.
    X:
        The background rows the marginal expectation averages over.
    row:
        The single row to explain (length n_features, 1-D or ``(1, M)``).
    n_permutations:
        Antithetic pairs to sample (default 16).
    seed:
        Seed for the permutation draws; ``None`` leaves them to fresh
        entropy (pass a seed for the house reproducibility rule).
    output_index:
        The score column for multi-output scorers.

    Returns
    -------
    ShapSummary
        A one-row record (``n_samples == 1``).

    Raises
    ------
    ModelError
        On non-2-D ``X``, a wrong-length ``row``, or ``n_permutations < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> rng = np.random.default_rng(3)
    >>> X = rng.normal(size=(40, 3))
    >>> y = 2.0 * X[:, 0] - X[:, 1]
    >>> model = LinearRegression().fit(X, y)
    >>> record = permutation_shap(model, X, X[0], n_permutations=8, seed=0)
    >>> record.n_samples
    1
    >>> residual = float(abs(
    ...     record.values.sum() + record.base_value - model.predict(X[:1])[0]
    ... ))
    >>> residual < 1e-8      # a linear scorer's contributions are permutation-exact
    True
    """
    background, x, names = _coerce_background_and_row(X, row)
    if (
        isinstance(n_permutations, bool)
        or not isinstance(n_permutations, (int, np.integer))
        or int(n_permutations) < 1
    ):
        raise ModelError(f"n_permutations must be a positive integer, got {n_permutations!r}")
    n_features = background.shape[1]
    rng = np.random.default_rng(seed)
    phi = np.zeros(n_features, dtype=float)
    empty_mask = np.zeros((1, n_features), dtype=bool)
    base = float(_masked_scores(model, background, x, empty_mask, output_index)[0])
    for _ in range(int(n_permutations)):
        order = rng.permutation(n_features)
        for walk in (order, order[::-1]):  # the antithetic pair
            masks = np.zeros((n_features + 1, n_features), dtype=bool)
            for step in range(n_features):
                masks[step + 1] = masks[step]
                masks[step + 1, int(walk[step])] = True
            scores = _masked_scores(model, background, x, masks, output_index)
            for step in range(n_features):
                phi[int(walk[step])] += scores[step + 1] - scores[step]
    phi /= 2 * int(n_permutations)

    _LOG.debug(
        "permutation_shap: 1 row x %d feature(s), %d antithetic pair(s)",
        n_features,
        int(n_permutations),
    )
    return ShapSummary(
        names=names,
        mean_abs=np.abs(phi),
        values=phi.reshape(1, -1),
        feature_values=x.reshape(1, -1),
        base_value=base,
        model_name=_model_label(model),
        task=_task_of(model),
        n_samples=1,
    )


def _factorial(n: int) -> float:
    """``n!`` as a float (no stdlib import; the counts here stay tiny)."""
    result = 1.0
    for step in range(2, n + 1):
        result *= step
    return result


def exact_shap(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    row: Any,
    *,
    output_index: int = 0,
) -> ShapSummary:
    """Exact Shapley values from ALL 2^M coalitions (shap's ``ExactExplainer``).

    The exponential-exact sweep of ``shap/explainers/_exact.py`` on a
    teaching scale: every coalition's background-marginal score ``v(S)``
    is evaluated, and each feature's value is the classic Shapley sum

    ``phi_j = sum over S not containing j of  |S|! (M-|S|-1)! / M! * (v(S u j) - v(S))``.

    ``base_value`` is ``v(empty)``.  The record is exact -- no sampling
    anywhere -- so ``values.sum() + base_value`` equals the row's score up
    to float rounding.

    Parameters
    ----------
    model:
        A fitted estimator (``predict``/``predict_proba``) or a raw
        callable ``matrix -> scores``.
    X:
        The background rows the marginal expectation averages over.
    row:
        The single row to explain (length n_features, 1-D or ``(1, M)``).
    output_index:
        The score column for multi-output scorers.

    Returns
    -------
    ShapSummary
        A one-row record (``n_samples == 1``).

    Raises
    ------
    ModelError
        On non-2-D ``X``, a wrong-length ``row``, or more than 12 features
        (2^13 = 8192 masked evaluations per background row -- this module's
        other explainers sample instead).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> rng = np.random.default_rng(4)
    >>> X = rng.normal(size=(24, 3))
    >>> y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
    >>> model = LinearRegression().fit(X, y)
    >>> record = exact_shap(model, X, X[0])
    >>> closed = model.coef_ * (X[0] - X.mean(axis=0))
    >>> bool(np.allclose(record.values[0], closed, atol=1e-8))
    True
    >>> residual = float(abs(
    ...     record.values.sum() + record.base_value - model.predict(X[:1])[0]
    ... ))
    >>> residual < 1e-8      # efficiency: phi sums to v(full) - v(empty)
    True
    """
    background, x, names = _coerce_background_and_row(X, row)
    n_features = background.shape[1]
    if n_features > 12:
        raise ModelError(
            f"exact_shap enumerates ALL 2^M coalitions: {n_features} features would need "
            f"{2 ** n_features} masked evaluations per background row.  Keep it at 12 or "
            "fewer (teaching scale) -- this module's permutation_shap samples the same "
            "Shapley values for wider feature sets"
        )
    masks = ((np.arange(2 ** n_features)[:, None] >> np.arange(n_features)) & 1).astype(bool)
    scores = _masked_scores(model, background, x, masks, output_index)
    base = float(scores[0])
    weights = np.array(
        [
            _factorial(int(masks[index].sum()))
            * _factorial(n_features - 1 - int(masks[index].sum()))
            / _factorial(n_features)
            for index in range(2 ** n_features)
        ],
        dtype=float,
    )
    phi = np.zeros(n_features, dtype=float)
    for feature in range(n_features):
        bit = 1 << feature
        without = np.flatnonzero(~masks[:, feature])
        phi[feature] = float(
            np.sum(weights[without] * (scores[without + bit] - scores[without]))
        )

    _LOG.debug(
        "exact_shap: 1 row x %d feature(s), %d coalition(s)",
        n_features,
        2 ** n_features,
    )
    return ShapSummary(
        names=names,
        mean_abs=np.abs(phi),
        values=phi.reshape(1, -1),
        feature_values=x.reshape(1, -1),
        base_value=base,
        model_name=_model_label(model),
        task=_task_of(model),
        n_samples=1,
    )


def _identity_link(x: Any) -> Any:
    """A no-op link (shap/links.py ``identity``)."""
    return x


def _sigmoid_link(x: Any) -> Any:
    """Log-odds -> probability, the numerically-stable tanh form of expit."""
    arr = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.tanh(0.5 * arr))


def _logit_link(x: Any) -> Any:
    """Probability -> log-odds, guarded against exactly-0/1 inputs.

    shap/links.py ``logit`` clips to the float-epsilon neighbourhood of
    the open interval before the logarithm, so 0 and 1 stay finite.
    """
    arr = np.asarray(x, dtype=float)
    eps = float(np.finfo(float).eps)
    clipped = np.clip(arr, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def shap_links() -> dict[str, tuple[Any, Any]]:
    """The identity/logit link pairs of ``shap/links.py``, as callables.

    Returns
    -------
    dict
        ``{"identity": (f, f_inv), "logit": (sigmoid, logit)}`` -- each
        pair is ``(link, inverse_link)``; the logit link is guarded
        against exactly-0/1 probabilities and the sigmoid is the stable
        tanh form.

    Examples
    --------
    >>> import numpy as np
    >>> links = shap_links()
    >>> sorted(links)
    ['identity', 'logit']
    >>> links["identity"][0](3.5)          # the identity link round-trips
    3.5
    >>> probability = 0.85
    >>> log_odds = links["logit"][1](probability)
    >>> float(abs(links["logit"][0](log_odds) - probability)) < 1e-12
    True
    >>> bool(np.isfinite(links["logit"][1](1.0)))   # the 0/1 guard
    True
    """
    return {
        "identity": (_identity_link, _identity_link),
        "logit": (_sigmoid_link, _logit_link),
    }


def _binom(n: int, k: int) -> int:
    """The binomial coefficient (pure python; the counts stay tiny here)."""
    k = min(k, n - k)
    result = 1
    for step in range(k):
        result = result * (n - step) // (step + 1)
    return result


def _kernel_shap_values(
    model: Any,
    background: np.ndarray,
    x: np.ndarray,
    n_coalitions: int,
    seed: int,
    output_index: int,
) -> np.ndarray:
    """Kernel-style SHAP values for one row (shap_summary's estimator).

    The SHAP kernel's weighted least squares over sampled coalitions:
    coalition sizes uniform over ``1 .. M-1`` (shap's ``_kernel.py``
    sampler), the kernel weights
    ``w(s) = (M - 1) / (C(M, s) s (M - s))``, and the equality-constrained
    solve (efficiency ``sum(phi) = v(full) - v(empty)``) via the KKT
    system.
    """
    n_features = background.shape[1]
    rng = np.random.default_rng(seed)
    empty_mask = np.zeros((1, n_features), dtype=bool)
    full_mask = np.ones((1, n_features), dtype=bool)
    base = float(_masked_scores(model, background, x, empty_mask, output_index)[0])
    if n_features == 1:
        full = float(_masked_scores(model, background, x, full_mask, output_index)[0])
        return np.array([full - base])
    masks = np.zeros((int(n_coalitions), n_features), dtype=bool)
    for row_index in range(int(n_coalitions)):
        size = int(rng.integers(1, n_features))
        masks[row_index, rng.choice(n_features, size=size, replace=False)] = True
    values = _masked_scores(model, background, x, masks, output_index) - base
    sizes = masks.sum(axis=1).astype(float)
    counts = np.array([_binom(n_features, int(size)) for size in sizes], dtype=float)
    weights = (n_features - 1.0) / (
        counts * sizes * np.maximum(1.0, n_features - sizes)
    )
    weighted = masks.astype(float) * weights[:, None]
    normal = masks.astype(float).T @ weighted + np.eye(n_features) * 1e-12
    rhs = masks.astype(float).T @ (weights * values)
    kkt = np.zeros((n_features + 1, n_features + 1))
    kkt[:n_features, :n_features] = normal
    kkt[:n_features, n_features] = 1.0
    kkt[n_features, :n_features] = 1.0
    full_value = float(_masked_scores(model, background, x, full_mask, output_index)[0])
    target = np.concatenate([rhs, [full_value - base]])
    try:
        solution = np.linalg.solve(kkt, target)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate samplings
        solution = np.linalg.lstsq(kkt, target, rcond=None)[0]
    return solution[:n_features]


def tree_shap_vs_kernel(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    row: Any,
    *,
    n_coalitions: int = 64,
    seed: int = 0,
) -> dict[str, float]:
    """The honesty check: exact TreeSHAP against the kernel-style estimator.

    Computes the SAME row twice -- once with
    :func:`tree_shap_summary`'s exact path-dependent recursion, once with
    the kernel-style weighted-coalition regression
    (:func:`shap_summary`'s estimator, reimplemented inline so the check
    never needs the optional shap package) -- and reports the per-feature
    disagreement.

    The two estimators answer DIFFERENT perturbations in general
    (path-dependent conditioning vs background-marginalisation), so they
    agree only where the theory says they must: ADDITIVE models over
    independent features, with ``X`` the frame the tree was trained on
    (then both share the background expectation).  That agreement is the
    caption's point -- on interactive models the gap is itself the honest
    finding, not a bug.

    Parameters
    ----------
    model:
        A FITTED sklearn tree ensemble (the
        :func:`tree_shap_summary` families).
    X:
        The frame that doubles as background distribution (pass the
        training frame for the additive-case agreement).
    row:
        The single row to check (length n_features).
    n_coalitions:
        Sampled coalitions for the kernel side (default 64).
    seed:
        Seed for the kernel's coalition sampler.

    Returns
    -------
    dict
        ``{"max_abs_diff": ..., "mean_abs_diff": ...}`` -- the largest and
        the mean absolute per-feature difference between the two SHAP
        vectors.

    Raises
    ------
    ModelError
        Whatever :func:`tree_shap_summary` raises for unsupported models,
        or a non-2-D ``X`` / wrong-length ``row``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> rng = np.random.default_rng(5)
    >>> X = rng.normal(size=(40, 4))
    >>> y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]   # an additive target
    >>> model = GradientBoostingRegressor(
    ...     random_state=5, max_depth=1, n_estimators=20
    ... ).fit(X, y)
    >>> checks = tree_shap_vs_kernel(model, X, X[0], n_coalitions=32, seed=0)
    >>> sorted(checks)
    ['max_abs_diff', 'mean_abs_diff']
    >>> bool(np.isfinite(checks["max_abs_diff"]) and checks["max_abs_diff"] < 1e-6)
    True
    """
    background, x, _names = _coerce_background_and_row(X, row)
    if (
        isinstance(n_coalitions, bool)
        or not isinstance(n_coalitions, (int, np.integer))
        or int(n_coalitions) < 1
    ):
        raise ModelError(f"n_coalitions must be a positive integer, got {n_coalitions!r}")
    tree_record = tree_shap_summary(model, x.reshape(1, -1))
    tree_phi = np.asarray(tree_record.values[0], dtype=float)
    kernel_phi = _kernel_shap_values(
        model, background, x, int(n_coalitions), int(seed), output_index=0
    )
    diffs = np.abs(tree_phi - kernel_phi)
    _LOG.debug(
        "tree_shap_vs_kernel: max %.3e / mean %.3e over %d feature(s)",
        float(diffs.max()),
        float(diffs.mean()),
        diffs.shape[0],
    )
    return {"max_abs_diff": float(diffs.max()), "mean_abs_diff": float(diffs.mean())}


__all__ += [
    "TreeShapSummary",
    "exact_shap",
    "linear_shap",
    "permutation_shap",
    "shap_links",
    "tree_shap_summary",
    "tree_shap_vs_kernel",
]
