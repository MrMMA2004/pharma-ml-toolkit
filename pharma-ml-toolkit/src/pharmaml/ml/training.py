"""Training, hyperparameter search and model comparison.

The flagship here is :func:`compare_models` -- a cross-validated
leaderboard across the whole registry.  In the legacy codebase the
same comparison was performed by running 30+ separate scripts and
collating their console output by hand; now one call returns a tidy
DataFrame ranked by the metric of choice.

Examples
--------
>>> import pandas as pd
>>> from pharmaml.ml.training import train_model
>>> X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
>>> y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
>>> result = train_model("ridge", "regression", X, y, tune=False)
>>> (result.name, result.task, result.best_params)
('ridge', 'regression', {})
>>> result.cv_scores is None and result.cv_mean is None    # no search ran
True
>>> round(float(result.model.predict(pd.DataFrame({"x": [10.0]}))[0]), 6)
9.928571
>>> list(result.model.named_steps)      # impute+scale prep wraps the estimator
['prep', 'reg']
>>> train_model("ridge", "regression", X, y, tune=False, cv=1)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: cv must be >= 2 folds, got 1
>>> train_model("ridge", "banana", X, y, tune=False)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: task must be 'classification' or 'regression', got 'banana'
>>> train_model("ridge", "classification", X, y, tune=False)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: Model 'ridge' supports 'regression', not 'classification'

>>> from pharmaml.ml.training import compare_models
>>> X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
>>> y = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])   # y = 2x + 1
>>> board = compare_models(X, y, task="regression",
...                        models=["ridge", "linear_regression"], cv=2)
>>> board["model"].tolist()            # the exact fit outranks the shrunken ridge
['linear_regression', 'ridge']
>>> abs(float(board.iloc[0, 1])) < 1e-9      # ~zero RMSE for the exact fit
True
>>> round(float(board.iloc[1, 1]), 6)        # ridge's L2 shrinkage costs RMSE
-1.661325
"""

from __future__ import annotations

from collections.abc import Callable
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
        Per-fold scores of the primary metric.  When EVERY fold is
        NaN (degenerate folds, e.g. a class rarer than ``cv``), the
        scores are still returned and ``cv_mean``/``cv_std`` stay NaN —
        a warning naming the fold scores is logged (100x-49-A).
    cv_mean / cv_std:
        Mean/std of ``cv_scores`` — NaN when every fold is NaN (see
        ``cv_scores``; the run is kept, not failed).
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
    model_params: dict[str, Any] | None = None,
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
    model_params:
        Manual hyperparameters, pinned as-is on the estimator before
        anything is fitted (keys are raw estimator parameter names —
        unknown keys raise :class:`~pharmaml.exceptions.ModelError`
        naming the valid ones).  Pinned keys are *excluded* from any
        grid search, so ``tune=True`` searches only the remaining
        grid slots around the user's fixed choices; with ``tune=False``
        the estimator is fitted exactly at these values and they are
        reported back as ``best_params``.  ``None``/empty keeps the
        historical behaviour untouched.

    Returns
    -------
    TrainingResult
        ``cv_scores`` carries the per-fold scores of the winning
        parameter combination.  When every fold is NaN (degenerate
        folds — e.g. a class rarer than ``cv``, where GridSearchCV's
        default ``error_score=NaN`` swallows the per-fold failures),
        the result is STILL returned with ``cv_mean``/``cv_std`` NaN
        and a module-logger warning naming the fold scores — matching
        the contract ``compare_models`` already documents for its NaN
        rows.

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

    Notes
    -----
    ``y_train`` is NOT validated for NaN or ±inf here (nothing in
    this module ever guarded it — the historical contract): the
    underlying sklearn fit rejects such targets with its own raw
    ``Input y contains NaN`` / ``Input y contains infinity`` errors.
    :func:`~pharmaml.ml.preprocessing.make_train_test_split` rejects
    NaN targets up front (typed) but lets ±inf through, so clean the
    target before training.
    """
    import time

    if isinstance(cv, int) and cv < 2:
        raise ModelError(f"cv must be >= 2 folds, got {cv}")

    estimator = build_model(name, task)
    if model_params:
        valid = set(estimator.get_params())
        unknown = sorted(set(model_params) - valid)
        if unknown:
            raise ModelError(
                f"model_params for {name!r} name unknown parameter(s) "
                f"{', '.join(unknown)}; valid parameters: "
                f"{', '.join(sorted(valid))}"
            )
        estimator.set_params(**model_params)
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
    if model_params:
        # Manual values pin their slots: a running grid search tunes
        # only the keys the user left at their defaults.
        grid = {k: v for k, v in grid.items() if k not in model_params}
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
        if model_params:
            # the winner carries the pinned values too — report the
            # estimator's ACTUAL parameters, search-found and pinned.
            best_params.update(model_params)
        # Per-fold scores of the winning combination.
        best_index = search.best_index_
        cv_scores = np.array(
            [search.cv_results_[f"split{i}_test_score"][best_index] for i in range(cv)]
        )
    else:
        pipeline.fit(X_train, y_train)
        fitted = pipeline
        best_params = dict(model_params) if model_params else {}
        cv_scores = None
    elapsed = time.perf_counter() - started

    cv_mean: float | None
    cv_std: float | None
    if cv_scores is not None:
        if np.isnan(cv_scores).all():
            # GridSearchCV's default error_score=NaN swallows per-fold
            # failures (a class rarer than the cv folds is the classic)
            # into silent NaN fold scores; compare_models warns for the
            # same condition -- keep returning (non-breaking) but make
            # the NaN cv_mean/cv_std visible in the log (100x-41-C G1).
            _LOG.warning(
                "cv_scores for %s are all NaN (fold scores: %s) — degenerate "
                "folds (e.g. a class rarer than cv=%d); cv_mean/cv_std stay NaN",
                name,
                cv_scores.tolist(),
                cv,
            )
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
    on_model: Callable[[dict[str, Any]], None] | None = None,
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
        :func:`sklearn.metrics.get_scorer_names`).  Note that scorers
        undefined on single-sample folds (``r2`` under leave-one-out
        ``cv == n_samples``) yield NaN rows — sklearn's own
        ``UndefinedMetricWarning`` contract, kept as-is.
    on_model:
        Optional progress callback invoked after each model finishes
        cross-validating (successfully or not) with the payload
        ``{'model': name, 'index': i, 'total': n}`` — one call per
        leaderboard row, in comparison order.  Designed for GUI
        progress bars and ETA displays; it must never raise.

    Returns
    -------
    pandas.DataFrame
        One row per model with mean/std CV metric and mean fit time,
        sorted by the primary metric (descending).  A model that fails
        to cross-validate keeps its row with a NaN score and sinks to
        the bottom instead of killing the leaderboard.
        The board carries ``model``, ``<metric>_mean``,
        ``<metric>_std``, ``fit_time_s`` and ``folds`` columns;
        ``fit_time_s`` is a wall-clock mean that varies between runs
        and machines, so only the metric columns (and the ranking they
        induce) are reproducible.

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
    for position, name in enumerate(models, start=1):
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
        if on_model is not None:
            # one call per completed row (success or failure) — the
            # progress-bar heartbeat for the longest job in the tab.
            on_model({"model": name, "index": position, "total": len(models)})

    frame = pd.DataFrame(rows)
    column = f"{metric}_mean"
    return frame.sort_values(column, ascending=False).reset_index(drop=True)


def benchmark_feature_families(
    smiles: pd.Series | list[str],
    y: pd.Series | list | np.ndarray,
    task: str,
    families: list[str] | tuple[str, ...] | None = None,
    model: str | None = None,
    cv: int = 3,
    n_jobs: int = 1,
    scoring: str | None = None,
    on_family: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Cross-validated leaderboard ACROSS feature families, one model fixed.

    The library port of ``fak_ml.ipynb`` cells 40-47 (the notebook's
    ``test_fp`` loop: one LGBMRegressor per datamol fingerprint type,
    r2/RMSE collected into a comparison chart).  Here the benchmark
    runs over the toolkit's own :data:`~pharmaml.chem.features.FEATURE_FAMILIES`
    vocabulary with ONE fixed registry model, so the board isolates the
    FEATURISATION effect the way the notebook's loop isolated the
    fingerprint effect.

    Parameters
    ----------
    smiles:
        SMILES strings, one per molecule (row ``i`` aligns with ``y[i]``
        positionally).
    y:
        Targets, positional alignment with ``smiles``.  Molecules the
        featuriser drops take their target row with them.
    task:
        ``'classification'`` or ``'regression'``.
    families:
        Feature families to benchmark; defaults to every family EXCEPT
        ``shape_3d`` and ``rism`` (the two 3D-geometry families embed
        conformers per molecule -- minutes on a real library, opt-in
        only; the notebook's own loop covered 2D fingerprints only).
    model:
        Registry name of the benchmark model; defaults to
        ``lightgbm`` when installed (the notebook's own choice),
        otherwise ``ridge`` (regression) / ``logistic_regression``
        (classification) -- the fastest always-available twins.
    cv:
        Fold count (at least 2).
    n_jobs:
        Workers for ``cross_validate``.
    scoring:
        Primary metric; same defaults and fallbacks as
        :func:`compare_models` (ROC-AUC ovo for classification with
        balanced-accuracy fallback for proba-less estimators, negative
        RMSE for regression).
    on_family:
        Optional progress callback invoked after each family finishes
        (successfully or not) with the payload
        ``{'family': name, 'index': i, 'total': n}`` -- the GUI
        progress heartbeat.  It must never raise.

    Returns
    -------
    pandas.DataFrame
        One row per family with ``family``, ``n_features``,
        ``n_samples``, ``<metric>_mean``, ``<metric>_std``,
        ``fit_time_s`` and ``folds``, sorted by the primary metric
        (descending).  A family that fails to featurise or
        cross-validate keeps its row with a NaN score and sinks to the
        bottom instead of killing the board.  Only the metric columns
        are reproducible (``fit_time_s`` is wall-clock).

    Raises
    ------
    ModelError
        If ``task`` is invalid, ``cv < 2``, a family name is outside
        the feature vocabulary, ``model`` is unknown or task
        incompatible, ``scoring`` is not a scikit-learn scorer name, or
        the dataset has fewer aligned rows than ``cv`` folds — the
        last guard (the compare_models 27-A twin) keeps a doomed
        request from degrading into a 100%-NaN board where every
        family fails with the identical splitter error; a family that
        drops molecules below ``cv`` only AFTER featurisation (bad
        SMILES) keeps its NaN row instead, per the swallow contract
        below.

    Examples
    --------
    >>> from pharmaml.ml.training import benchmark_feature_families
    >>> smiles = ["CCO", "CCCCO", "CCCCCCO", "CCCCCCCCO",
    ...           "CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "O=C(O)C1CCCO1"]
    >>> y = [2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 4.5, 3.0]
    >>> board = benchmark_feature_families(smiles, y, task="regression",
    ...                                    families=("estate",), model="ridge", cv=2)
    >>> list(board.columns)
    ['family', 'n_features', 'n_samples', 'neg_root_mean_squared_error_mean', 'neg_root_mean_squared_error_std', 'fit_time_s', 'folds']
    >>> board["family"].tolist()
    ['estate']
    >>> int(board["n_features"].iloc[0]), int(board["n_samples"].iloc[0])
    (79, 8)
    """
    if task not in {"classification", "regression"}:
        raise ModelError(f"task must be 'classification' or 'regression', got {task!r}")
    if isinstance(cv, int) and cv < 2:
        raise ModelError(f"cv must be >= 2 folds, got {cv}")
    if isinstance(scoring, str) and scoring not in get_scorer_names():
        raise ModelError(
            f"Unknown scoring {scoring!r}; valid metric names are listed by "
            "sklearn.metrics.get_scorer_names() (e.g. 'roc_auc', "
            "'balanced_accuracy', 'neg_root_mean_squared_error')"
        )

    # Lazy imports keep this module's import path rdkit-free (the same
    # rule the features module holds for lightgbm/sklearn).
    from pharmaml.chem.features import FEATURE_FAMILIES, compute_feature_frame

    if families is None:
        # The notebook's own loop covered 2D fingerprints only; the two
        # 3D-geometry families (conformer embedding per molecule) stay
        # opt-in -- minutes on a real library.
        families = [
            name for name in FEATURE_FAMILIES if name not in {"shape_3d", "rism"}
        ]
    family_list = list(families)
    if not family_list:
        raise ModelError("families is empty -- pass at least one feature family name")
    unknown = sorted(set(family_list) - set(FEATURE_FAMILIES))
    if unknown:
        raise ModelError(
            f"unknown feature family {unknown[0]!r}; valid families are: "
            + ", ".join(FEATURE_FAMILIES)
        )

    if model is None:
        lightgbm_spec = MODEL_REGISTRY["lightgbm"]
        model = "lightgbm" if lightgbm_spec.available else (
            "ridge" if task == "regression" else "logistic_regression"
        )
        _LOG.info("benchmark_feature_families default model: %s", model)

    y_series = pd.Series(y) if not isinstance(y, pd.Series) else y.reset_index(drop=True)
    if len(smiles) != len(y_series):
        raise ModelError(
            f"smiles and y must be row-aligned (got {len(smiles)} SMILES "
            f"and {len(y_series)} targets)"
        )
    if isinstance(cv, int) and len(y_series) < cv:
        # cross_validate runs for every family here; with fewer aligned
        # rows than folds every single row would swallow the same raw
        # splitter error inside the per-family try/except below and
        # surface as an all-NaN board instead of an actionable message
        # (the compare_models 27-A guard, hoisted in 100x-150 -- the
        # audit caught the docstring promising this raise while the
        # swallow made it unreachable).
        raise ModelError(
            f"cv={cv} folds exceeds the number of aligned samples "
            f"(n_samples={len(y_series)}) -- no fold can be built; lower "
            "cv or supply more data"
        )

    encode = False  # every feature family emits numeric columns by contract
    pipeline = Pipeline(
        [
            ("prep", build_preprocessor(encode_categorical=encode)),
            ("est", build_model(model, task)),  # unknown/task-mismatch -> ModelError
        ]
    )
    metric = scoring or _select_scorer(pipeline, task)

    rows: list[dict[str, Any]] = []
    for position, family in enumerate(family_list, start=1):
        try:
            frame, _report = compute_feature_frame(smiles, families=(family,))
            aligned_y = y_series.iloc[list(frame.index)]
            if len(aligned_y) < cv:
                raise ModelError(
                    f"cv={cv} folds exceeds the number of aligned samples "
                    f"(n_samples={len(aligned_y)}) for family {family!r}"
                )
            result = cross_validate(
                pipeline, frame.to_numpy(), aligned_y.to_numpy(),
                cv=cv, scoring=metric, n_jobs=n_jobs,
            )
            scores = result["test_score"]
            if np.isnan(scores).any():
                _LOG.warning(
                    "Family %s produced NaN CV scores for metric %r "
                    "(degenerate folds); keeping the row as NaN",
                    family, metric,
                )
            rows.append(
                {
                    "family": family,
                    "n_features": int(frame.shape[1]),
                    "n_samples": int(frame.shape[0]),
                    f"{metric}_mean": float(np.mean(scores)),
                    f"{metric}_std": float(np.std(scores)),
                    "fit_time_s": float(np.mean(result["fit_time"])),
                    "folds": cv,
                }
            )
        except Exception as exc:  # noqa: BLE001 — keep the board alive on failures
            _LOG.warning("Family %s failed during benchmarking: %s", family, exc)
            rows.append({"family": family, f"{metric}_mean": np.nan, "folds": cv})
        if on_family is not None:
            on_family({"family": family, "index": position, "total": len(family_list)})

    board = pd.DataFrame(rows)
    column = f"{metric}_mean"
    return board.sort_values(column, ascending=False).reset_index(drop=True)
