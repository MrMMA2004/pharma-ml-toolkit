"""Training, hyperparameter search and model comparison.

The flagship here is :func:`compare_models` -- a cross-validated
leaderboard across the whole registry.  In the legacy codebase the
same comparison was performed by running 30+ separate scripts and
collating their console output by hand; now one call returns a tidy
DataFrame ranked by the metric of choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from pharmaml.exceptions import ModelError
from pharmaml.logger import get_logger
from pharmaml.ml.models import MODEL_REGISTRY, build_model, resolve_model_name
from pharmaml.ml.preprocessing import build_preprocessor, has_categorical_columns

_LOG = get_logger(__name__)

__all__ = ["TrainingResult", "compare_models", "train_model"]


@dataclass(slots=True)
class TrainingResult:
    """Outcome of a single training run.

    Attributes
    ----------
    model:
        The fitted estimator (a Pipeline when preprocessing is inside).
    name:
        Registry name of the trained algorithm.
    task:
        ``'classification'`` or ``'regression'``.
    best_params:
        Hyperparameters chosen by the search (empty when ``tune=False``).
    cv_scores:
        Per-fold scores of the primary metric.
    cv_mean / cv_std:
        Mean/std of ``cv_scores``.
    fit_seconds:
        Wall-clock training time.
    """

    model: Any
    name: str
    task: str
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_scores: np.ndarray | None = None
    cv_mean: float | None = None
    cv_std: float | None = None
    fit_seconds: float = 0.0


_PRIMARY_METRIC = {
    "classification": "roc_auc_ovo",
    "regression": "neg_root_mean_squared_error",
}

#: Multiclass-safe fallback for classifiers without ``predict_proba``
#: (e.g. ``ridge_classifier``): ``roc_auc_*`` needs probability (or at
#: least decision-function) scores, so those models are scored by
#: balanced accuracy instead of silently producing NaN folds.
_CLASSIFICATION_FALLBACK_METRIC = "balanced_accuracy"


def _select_scorer(pipeline: Pipeline, task: str) -> str:
    """Pick the CV metric for ``pipeline``.

    Classification estimators exposing ``predict_proba`` (logistic
    regression, forests, SVM with ``probability=True``, ...) are scored
    by one-vs-one ROC-AUC, which works for **binary and multiclass**
    targets alike.  Estimators without probabilities fall back to
    balanced accuracy, a label-only metric that also supports any
    number of classes.
    """
    if task == "classification" and not hasattr(pipeline, "predict_proba"):
        _LOG.info(
            "Estimator lacks predict_proba; scoring CV folds with %s "
            "instead of %s",
            _CLASSIFICATION_FALLBACK_METRIC,
            _PRIMARY_METRIC["classification"],
        )
        return _CLASSIFICATION_FALLBACK_METRIC
    return _PRIMARY_METRIC[task]


def train_model(
    name: str,
    task: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    tune: bool = True,
    param_grid: dict[str, list[Any]] | None = None,
    cv: int = 5,
    n_jobs: int = -1,
    preprocess: bool = True,
    random_state: int = 42,
) -> TrainingResult:
    """Train (and optionally grid-search) one registry model.

    Parameters
    ----------
    name:
        Registry key, e.g. ``'random_forest'``.
    task:
        ``'classification'`` or ``'regression'``.
    X_train, y_train:
        Training data.
    tune:
        Run :class:`~sklearn.model_selection.GridSearchCV` with the
        registry default grid (or ``param_grid``).  When False the
        model is fitted once with default/registry parameters.
    param_grid:
        Overrides the registry grid; keys are raw estimator parameter
        names (the ``clf__``/``reg__`` prefix is added automatically).
    cv:
        Number of CV folds for tuning/scoring.
    n_jobs:
        Parallel workers for the search.
    preprocess:
        Wrap the estimator with the standard impute+scale pipeline.
    random_state:
        Reproducibility seed (kept for interface symmetry; estimators
        already receive it from the registry factories).

    Returns
    -------
    TrainingResult

    Raises
    ------
    ModelError
        When ``cv < 2`` — the same entry guard ``compare_models`` has;
        pre-fix ``--no-tune`` silently ignored the value and the tuned
        path crashed deep inside GridSearchCV with raw sklearn jargon.
        Also when ``tune`` is enabled and the training data has fewer
        rows than ``cv`` folds: no fold can be built at all, so the
        request fails up front with the row count and fold count named
        instead of letting sklearn's splitter ``ValueError`` penetrate
        raw (27-A).  With ``tune=False`` no cross-validation runs, so a
        dataset smaller than ``cv`` stays usable.
    """
    import time

    if isinstance(cv, int) and cv < 2:
        raise ModelError(f"cv must be >= 2 folds, got {cv}")

    estimator = build_model(name, task)
    key = resolve_model_name(name)
    spec = MODEL_REGISTRY[key]

    encode = has_categorical_columns(X_train)
    if encode:
        _LOG.info(
            "Feature table contains categorical column(s); enabling one-hot "
            "encoding inside the preprocessing pipeline."
        )
    steps = [
        ("prep", build_preprocessor(encode_categorical=encode))
    ] if preprocess else []
    step_name = "clf" if task == "classification" else "reg"
    pipeline = Pipeline(steps + [(step_name, estimator)])

    grid = param_grid if param_grid is not None else spec.param_grid
    prefixed = {f"{step_name}__{k}": v for k, v in grid.items()}

    started = time.perf_counter()
    if tune and prefixed:
        n_samples = int(X_train.shape[0])
        if isinstance(cv, int) and n_samples < cv:
            # Fail up front instead of letting KFold's raw ValueError
            # ("Cannot have number of splits n_splits=5 greater than the
            # number of samples") penetrate to the user (27-A).
            raise ModelError(
                f"cv={cv} folds exceeds the number of training rows "
                f"(n_samples={n_samples}) -- no fold can be built; lower "
                "cv or supply more data"
            )
        scorer = _select_scorer(pipeline, task)
        _LOG.info("Tuning %s (%s) over %d parameter combinations",
                  name, task, int(np.prod([len(v) for v in prefixed.values()])))
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=prefixed,
            cv=cv,
            scoring=scorer,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        best_params = {k.removeprefix(f"{step_name}__"): v for k, v in search.best_params_.items()}
        # Per-fold scores of the winning combination.
        best_index = search.best_index_
        cv_scores = np.array(
            [search.cv_results_[f"split{i}_test_score"][best_index] for i in range(cv)]
        )
    else:
        pipeline.fit(X_train, y_train)
        fitted = pipeline
        best_params = {}
        cv_scores = None
    elapsed = time.perf_counter() - started

    cv_mean: float | None
    cv_std: float | None
    if cv_scores is not None:
        cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    else:
        cv_mean = cv_std = None

    return TrainingResult(
        model=fitted,
        name=spec.name,
        task=task,
        best_params=best_params,
        cv_scores=cv_scores,
        cv_mean=cv_mean,
        cv_std=cv_std,
        fit_seconds=elapsed,
    )


def compare_models(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    task: str,
    models: list[str] | None = None,
    cv: int = 5,
    n_jobs: int = 1,
    scoring: str | None = None,
) -> pd.DataFrame:
    """Cross-validated leaderboard over (a subset of) the registry.

    Parameters
    ----------
    X, y:
        Full dataset (splitting happens inside CV folds).
    task:
        ``'classification'`` or ``'regression'``.
    models:
        Registry names to compare; defaults to every *available*
        model supporting the task.  Aliases that resolve to the same
        registry key (``'rf'`` and ``'random_forest'``) collapse into a
        single row.
    cv:
        Fold count (at least 2).
    n_jobs:
        Workers per estimator evaluation.
    scoring:
        Primary metric; defaults to ROC-AUC (one-vs-one) for
        classification and negative RMSE for regression.  When the
        default is used and *any* compared classifier lacks
        ``predict_proba`` (e.g. ``ridge_classifier``), the whole board
        falls back to balanced accuracy so every row stays comparable
        — both metrics support binary **and multiclass** targets.
        A string must be a scikit-learn scorer name (see
        :func:`sklearn.metrics.get_scorer_names`).

    Returns
    -------
    pandas.DataFrame
        One row per model with mean/std CV metric and mean fit time,
        sorted by the primary metric (descending).  A model that fails
        to cross-validate keeps its row with a NaN score and sinks to
        the bottom instead of killing the leaderboard.

    Raises
    ------
    ModelError
        If any requested model is unavailable or task-incompatible,
        the models list is empty, ``cv < 2``, ``scoring`` is not a
        scikit-learn scorer name, or the dataset has fewer rows than
        ``cv`` folds — the last guard keeps a doomed request from
        degrading into a 100%-NaN leaderboard where every model fails
        with the identical splitter error (27-A).
    """
    if task not in {"classification", "regression"}:
        raise ModelError(f"task must be 'classification' or 'regression', got {task!r}")
    if isinstance(cv, int) and cv < 2:
        raise ModelError(f"cv must be >= 2 folds, got {cv}")
    n_samples = int(X.shape[0])
    if isinstance(cv, int) and n_samples < cv:
        # cross_validate runs for every model here; with n_samples < cv
        # every single row would swallow the same raw splitter error
        # below and surface as an all-NaN board instead of an
        # actionable message.
        raise ModelError(
            f"cv={cv} folds exceeds the number of samples "
            f"(n_samples={n_samples}) -- no fold can be built; lower cv "
            "or supply more data"
        )
    # Guard the metric name up front: a typo would otherwise fail inside
    # every per-model try/except below and surface as an all-NaN
    # leaderboard instead of an actionable error.
    if isinstance(scoring, str) and scoring not in get_scorer_names():
        raise ModelError(
            f"Unknown scoring {scoring!r}; valid metric names are listed by "
            "sklearn.metrics.get_scorer_names() (e.g. 'roc_auc', "
            "'balanced_accuracy', 'neg_root_mean_squared_error')"
        )

    if models is None:
        models = [
            spec.name
            for spec in MODEL_REGISTRY.values()
            if spec.available and spec.task in {task, "both"}
        ]
    else:
        resolved: list[str] = []
        for name in models:
            # resolve_model_name's own ModelError already lists the
            # registered names — propagate it verbatim.
            key = resolve_model_name(name)
            spec = MODEL_REGISTRY[key]
            resolved.append(spec.name)
            if not spec.available:
                raise ModelError(
                    f"Model {name!r} requires optional package {spec.requires!r}; "
                    "install pharmaml[boosting] or drop it from the list"
                )
            if spec.task not in {task, "both"}:
                raise ModelError(f"Model {name!r} does not support {task}")
        # Aliases may resolve to the same registry key ('rf' and
        # 'random_forest'); collapse duplicates so the leaderboard keeps
        # one row per distinct model.
        models = list(dict.fromkeys(resolved))

    if not models:
        raise ModelError(
            "models list is empty — pass registry names (see "
            "pharmaml.ml.list_models()) or None to compare every available "
            "model for the task"
        )

    encode = has_categorical_columns(X)
    if encode:
        _LOG.info(
            "Feature table contains categorical column(s); enabling one-hot "
            "encoding inside the comparison pipelines."
        )

    # All models in one leaderboard must be scored by the same metric:
    # when *any* member lacks predict_proba (ridge_classifier ...) the
    # whole board falls back to balanced accuracy (multiclass-safe).
    pipelines: dict[str, Pipeline] = {
        name: Pipeline(
            [("prep", build_preprocessor(encode_categorical=encode)), ("est", build_model(name, task))]
        )
        for name in models
    }
    metric = scoring or _select_scorer(pipelines[models[0]], task)
    if (
        scoring is None
        and task == "classification"
        and any(not hasattr(p, "predict_proba") for p in pipelines.values())
    ):
        metric = _CLASSIFICATION_FALLBACK_METRIC
        _LOG.info(
            "Leaderboard contains classifier(s) without predict_proba; "
            "scoring all models with %s", metric
        )

    rows: list[dict[str, Any]] = []
    for name in models:
        pipeline = pipelines[name]
        try:
            result = cross_validate(
                pipeline, X, y, cv=cv, scoring=metric, n_jobs=n_jobs
            )
            scores = result["test_score"]
            if np.isnan(scores).any():
                # cross_validate can return NaN folds without raising
                # (e.g. a single-class target breaks proba-based scorers)
                # — make the silent poisoning visible in the log.
                _LOG.warning(
                    "Model %s produced NaN CV scores for metric %r "
                    "(degenerate folds — e.g. a single-class target); "
                    "keeping the row as NaN",
                    name,
                    metric,
                )
            rows.append(
                {
                    "model": name,
                    f"{metric}_mean": float(np.mean(scores)),
                    f"{metric}_std": float(np.std(scores)),
                    "fit_time_s": float(np.mean(result["fit_time"])),
                    "folds": cv,
                }
            )
        except Exception as exc:  # noqa: BLE001 — keep the leaderboard alive on failures
            _LOG.warning("Model %s failed during comparison: %s", name, exc)
            rows.append({"model": name, f"{metric}_mean": np.nan, "folds": cv})

    frame = pd.DataFrame(rows)
    column = f"{metric}_mean"
    return frame.sort_values(column, ascending=False).reset_index(drop=True)
