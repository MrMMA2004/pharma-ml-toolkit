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
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from pharmaml.exceptions import ModelError
from pharmaml.logger import get_logger
from pharmaml.ml.models import MODEL_REGISTRY, build_model
from pharmaml.ml.preprocessing import build_preprocessor

_LOG = get_logger(__name__)

__all__ = ["train_model", "TrainingResult", "compare_models"]


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
    """
    import time

    estimator = build_model(name, task)
    spec = MODEL_REGISTRY[name.replace(" ", "_")]

    steps = [("prep", build_preprocessor())] if preprocess else []
    step_name = "clf" if task == "classification" else "reg"
    pipeline = Pipeline(steps + [(step_name, estimator)])

    grid = param_grid if param_grid is not None else spec.param_grid
    prefixed = {f"{step_name}__{k}": v for k, v in grid.items()}

    started = time.perf_counter()
    if tune and prefixed:
        scorer = _PRIMARY_METRIC[task]
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

    if cv_scores is not None:
        mean, std = float(cv_scores.mean()), float(cv_scores.std())
    else:
        mean = std = None

    return TrainingResult(
        model=fitted,
        name=spec.name,
        task=task,
        best_params=best_params,
        cv_scores=cv_scores,
        cv_mean=mean,
        cv_std=std,
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
        model supporting the task.
    cv:
        Fold count.
    n_jobs:
        Workers per estimator evaluation.
    scoring:
        Primary metric; defaults to ROC-AUC (one-vs-one) for
        classification and negative RMSE for regression.

    Returns
    -------
    pandas.DataFrame
        One row per model with mean/std CV metric, fit time and
        registry availability notes, sorted by the primary metric.

    Raises
    ------
    ModelError
        If any requested model is unavailable or task-incompatible.
    """
    if task not in {"classification", "regression"}:
        raise ModelError(f"task must be 'classification' or 'regression', got {task!r}")

    if models is None:
        models = [
            spec.name
            for spec in MODEL_REGISTRY.values()
            if spec.available and spec.task in {task, "both"}
        ]
    else:
        for name in models:
            spec = MODEL_REGISTRY.get(name)
            if spec is None:
                raise ModelError(f"Unknown model {name!r}")
            if not spec.available:
                raise ModelError(
                    f"Model {name!r} requires optional package {spec.requires!r}; "
                    "install pharmaml[boosting] or drop it from the list"
                )
            if spec.task not in {task, "both"}:
                raise ModelError(f"Model {name!r} does not support {task}")

    metric = scoring or _PRIMARY_METRIC[task]
    rows: list[dict[str, Any]] = []
    for name in models:
        try:
            estimator = build_model(name, task)
            pipeline = Pipeline(
                [("prep", build_preprocessor()), ("est", estimator)]
            )
            result = cross_validate(
                pipeline, X, y, cv=cv, scoring=metric, n_jobs=n_jobs
            )
            scores = result["test_score"]
            rows.append(
                {
                    "model": name,
                    f"{metric}_mean": float(np.mean(scores)),
                    f"{metric}_std": float(np.std(scores)),
                    "fit_time_s": float(np.mean(result["fit_time"])),
                    "folds": cv,
                }
            )
        except Exception as exc:  # keep the leaderboard alive on failures
            _LOG.warning("Model %s failed during comparison: %s", name, exc)
            rows.append({"model": name, f"{metric}_mean": np.nan, "folds": cv})

    frame = pd.DataFrame(rows)
    column = f"{metric}_mean"
    return frame.sort_values(column, ascending=False).reset_index(drop=True)
