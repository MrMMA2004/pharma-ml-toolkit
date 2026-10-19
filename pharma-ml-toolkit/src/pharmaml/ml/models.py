"""Unified model registry.

The legacy codebase kept one ~150-line script per algorithm (30+ files
of near-duplicate load/preprocess/tune/evaluate boilerplate).  This
module collapses them into a registry: each entry provides the
estimator factory, the task(s) it supports, a tuned default parameter
grid and an availability probe.  One generic trainer then serves every
algorithm.

Optional backends (XGBoost, LightGBM, CatBoost, PyTorch deep nets) are
imported lazily and reported as unavailable rather than crashing the
registry import.

Examples
--------
>>> from pharmaml.ml.models import MODEL_REGISTRY
>>> spec = MODEL_REGISTRY["random_forest"]
>>> spec.task
'both'
>>> sorted(spec.param_grid)
['max_depth', 'min_samples_split', 'n_estimators']
>>> spec.available            # sklearn-backed: no optional dependency
True
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from pharmaml.exceptions import DependencyError, ModelError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

TaskKind = Literal["classification", "regression"]


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Registry entry for one algorithm family.

    Attributes
    ----------
    name:
        Registry key (kebab-case, e.g. ``'random_forest'``).
    task:
        ``'classification'``, ``'regression'`` or ``'both'``.
    factory:
        Zero-argument callable returning a fresh, unfitted estimator
        (random_state already threaded in where the estimator accepts
        one).  ``None`` for dual-task entries whose per-task estimator
        is built by :data:`_TASK_FACTORIES` inside :func:`build_model`.
    param_grid:
        Compact default grid for hyperparameter search.
    requires:
        Optional third-party module needed for the factory.
    notes:
        Short human description used by ``list_models`` output.
    """

    name: str
    task: str
    factory: Callable[..., Any] | None
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    requires: str | None = None
    notes: str = ""

    @property
    def available(self) -> bool:
        """True when the optional dependency (if any) is importable."""
        return self.requires is None or _module_available(self.requires)


def _rf_cls() -> RandomForestClassifier:
    return RandomForestClassifier(random_state=42, n_jobs=-1)


def _rf_reg() -> RandomForestRegressor:
    return RandomForestRegressor(random_state=42, n_jobs=-1)


def _svm_cls() -> SVC:
    return SVC(random_state=42, probability=True)


def _mlp_cls() -> MLPClassifier:
    return MLPClassifier(random_state=42, max_iter=1000)


def _mlp_reg() -> MLPRegressor:
    return MLPRegressor(random_state=42, max_iter=1000)


def _xgb_cls():
    from xgboost import XGBClassifier

    return XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)


def _xgb_reg():
    from xgboost import XGBRegressor

    return XGBRegressor(random_state=42, n_jobs=-1)


def _lgbm_cls():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)


def _lgbm_reg():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)


def _cat_cls():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(random_state=42, verbose=False)


def _cat_reg():
    from catboost import CatBoostRegressor

    return CatBoostRegressor(random_state=42, verbose=False)


def _dnn_cls():
    from pharmaml.ml.neural import DeepMLPClassifier

    return DeepMLPClassifier(random_state=42)


def _dnn_reg():
    from pharmaml.ml.neural import DeepMLPRegressor

    return DeepMLPRegressor(random_state=42)


def _res_dnn_cls():
    from pharmaml.ml.neural import ResidualMLPClassifier

    return ResidualMLPClassifier(random_state=42)


def _res_dnn_reg():
    from pharmaml.ml.neural import ResidualMLPRegressor

    return ResidualMLPRegressor(random_state=42)


#: Optional requirement -> pip extra that installs it (used by the
#: DependencyError hint in :func:`build_model` so every backend names
#: its own extra instead of a hard-coded one).
_REQUIREMENT_EXTRAS = {
    "xgboost": "boosting",
    "lightgbm": "boosting",
    "catboost": "boosting",
    "torch": "deep",
}


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        task="classification",
        factory=lambda: LogisticRegression(max_iter=5000, random_state=42),
        param_grid={"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"]},
        notes="Linear baseline (upgrades Logistic regression.py)",
    ),
    "ridge_classifier": ModelSpec(
        name="ridge_classifier",
        task="classification",
        factory=lambda: RidgeClassifier(random_state=42),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        notes="L2 linear classifier (upgrades Ridge.py)",
    ),
    "lasso_logistic": ModelSpec(
        name="lasso_logistic",
        task="classification",
        factory=lambda: LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=5000, random_state=42
        ),
        param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
        notes="L1 sparse logistic classifier (upgrades Classification/Lasso.py)",
    ),
    "elastic_net_logistic": ModelSpec(
        name="elastic_net_logistic",
        task="classification",
        factory=lambda: LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            max_iter=5000,
            random_state=42,
        ),
        param_grid={"C": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.2, 0.5, 0.8]},
        notes=(
            "Elastic-net logistic classifier, saga solver "
            "(upgrades Classification/Elastic net.py)"
        ),
    ),
    "lasso": ModelSpec(
        name="lasso",
        task="regression",
        factory=lambda: Lasso(random_state=42, max_iter=20000),
        param_grid={"alpha": [0.001, 0.01, 0.1, 1.0]},
        notes="L1 sparse linear regression (upgrades Lasso.py)",
    ),
    "ridge": ModelSpec(
        name="ridge",
        task="regression",
        factory=lambda: Ridge(random_state=42),
        param_grid={"alpha": [0.1, 1.0, 10.0, 100.0]},
        notes="L2 linear regression (upgrades Ridge.py)",
    ),
    "elastic_net": ModelSpec(
        name="elastic_net",
        task="regression",
        factory=lambda: ElasticNet(random_state=42, max_iter=20000),
        param_grid={"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
        notes="L1/L2 mixture (upgrades Elastic net.py)",
    ),
    "linear_regression": ModelSpec(
        name="linear_regression",
        task="regression",
        factory=lambda: LinearRegression(),
        param_grid={},
        notes="OLS baseline (upgrades Linear regression.py)",
    ),
    "huber_regression": ModelSpec(
        name="huber_regression",
        task="regression",
        factory=lambda: HuberRegressor(),
        param_grid={"epsilon": [1.1, 1.35, 1.75, 2.0], "alpha": [0.0001, 0.001, 0.01]},
        notes="Robust regression with Huber loss (upgrades Huber.py)",
    ),
    "knn": ModelSpec(
        name="knn",
        task="both",
        factory=None,  # per-task dispatch via _TASK_FACTORIES in build_model
        param_grid={"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
        notes="k-nearest neighbours (upgrades KNN.py)",
    ),
    "decision_tree": ModelSpec(
        name="decision_tree",
        task="both",
        factory=None,
        param_grid={"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
        notes="CART tree (upgrades Decision tree.py)",
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        task="both",
        factory=None,
        param_grid={
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        notes="Bagged ensemble (upgrades RF.py)",
    ),
    "svm": ModelSpec(
        name="svm",
        task="both",
        factory=None,
        param_grid={"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
        notes="Kernel machines (upgrades SVM.py / SVM reg.py)",
    ),
    "adaboost": ModelSpec(
        name="adaboost",
        task="both",
        factory=None,
        param_grid={"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1, 1.0]},
        notes="Boosted stumps (upgrades ADAboost.py)",
    ),
    "ann": ModelSpec(
        name="ann",
        task="both",
        factory=None,
        param_grid={
            "hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
            "alpha": [1e-4, 1e-3, 1e-2],
        },
        notes="MLP neural net (upgrades ANN.py / ANN class.py)",
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        task="both",
        factory=None,
        requires="xgboost",
        param_grid={"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
        notes="Gradient boosting (upgrades XGboost.py)",
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        task="both",
        factory=None,
        requires="lightgbm",
        param_grid={"n_estimators": [200, 400], "num_leaves": [31, 63]},
        notes="LightGBM (upgrades LightGBM.py)",
    ),
    "catboost": ModelSpec(
        name="catboost",
        task="both",
        factory=None,
        requires="catboost",
        param_grid={"iterations": [200, 400], "depth": [6, 8]},
        notes="CatBoost (upgrades CATboost.py)",
    ),
    "dnn": ModelSpec(
        name="dnn",
        task="both",
        factory=None,
        requires="torch",
        param_grid={
            "hidden_layers": [(64, 32), (128, 64, 32)],
            "dropout": [0.0, 0.1],
            "learning_rate": [1e-3, 3e-4],
        },
        notes="deep feed-forward network (PyTorch) — needs `pip install 'pharmaml[deep]'`",
    ),
    "dnn_residual": ModelSpec(
        name="dnn_residual",
        task="both",
        factory=None,
        requires="torch",
        param_grid={
            "hidden_layers": [(64, 32), (128, 64, 32)],
            "dropout": [0.0, 0.1],
            "learning_rate": [1e-3, 3e-4],
        },
        notes="residual deep network (PyTorch) — needs `pip install 'pharmaml[deep]'`",
    ),
}


#: Friendly short names accepted anywhere a registry key is expected
#: (``--model rf`` on the CLI, ``train_model("logistic", ...)`` ...).
MODEL_ALIASES: dict[str, str] = {
    "logistic": "logistic_regression",
    "logreg": "logistic_regression",
    "rf": "random_forest",
    "randomforest": "random_forest",
    "dt": "decision_tree",
    "tree": "decision_tree",
    "gbm": "lightgbm",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "cb": "catboost",
    "cat": "catboost",
    "catboost": "catboost",
    "knn": "knn",
    "svm": "svm",
    "svc": "svm",
    "ann": "ann",
    "mlp": "ann",
    "neural_net": "ann",
    "ada": "adaboost",
    "adaboost": "adaboost",
    "linreg": "linear_regression",
    "ols": "linear_regression",
    "huber": "huber_regression",
    "huberreg": "huber_regression",
    "ridge": "ridge",
    "ridge_cls": "ridge_classifier",
    "lasso": "lasso",
    "enet": "elastic_net",
    "elasticnet": "elastic_net",
    "dnn": "dnn",
    "deep_mlp": "dnn",
    "dnn_residual": "dnn_residual",
    "resnet": "dnn_residual",
    "residual_dnn": "dnn_residual",
}


def resolve_model_name(name: str) -> str:
    """Normalise a user-supplied model name to a registry key.

    Applies case/whitespace/punctuation clean-up, then the
    :data:`MODEL_ALIASES` short-name map, so ``'Random-Forest'``,
    ``'rf'`` and ``'random_forest'`` all resolve to the same key.

    Raises
    ------
    ModelError
        When the cleaned name is neither a registry key nor an alias.
    """
    lowered = str(name).strip().lower()
    underscored = lowered.replace(" ", "_").replace("-", "_")
    for candidate in (underscored, MODEL_ALIASES.get(lowered), MODEL_ALIASES.get(underscored)):
        if candidate in MODEL_REGISTRY:
            return candidate
    known = ", ".join(sorted(MODEL_REGISTRY))
    raise ModelError(f"Unknown model {name!r}. Registered models: {known}")


_TASK_FACTORIES: dict[str, dict[str, Callable[[], Any]]] = {
    "knn": {
        "classification": KNeighborsClassifier,
        "regression": KNeighborsRegressor,
    },
    "decision_tree": {
        "classification": lambda: DecisionTreeClassifier(random_state=42),
        "regression": lambda: DecisionTreeRegressor(random_state=42),
    },
    "random_forest": {
        "classification": _rf_cls,
        "regression": _rf_reg,
    },
    "svm": {
        "classification": _svm_cls,
        "regression": lambda: SVR(),
    },
    "adaboost": {
        "classification": lambda: AdaBoostClassifier(random_state=42),
        "regression": lambda: AdaBoostRegressor(random_state=42),
    },
    "ann": {
        "classification": _mlp_cls,
        "regression": _mlp_reg,
    },
    "xgboost": {
        "classification": _xgb_cls,
        "regression": _xgb_reg,
    },
    "lightgbm": {
        "classification": _lgbm_cls,
        "regression": _lgbm_reg,
    },
    "catboost": {
        "classification": _cat_cls,
        "regression": _cat_reg,
    },
    "dnn": {
        "classification": _dnn_cls,
        "regression": _dnn_reg,
    },
    "dnn_residual": {
        "classification": _res_dnn_cls,
        "regression": _res_dnn_reg,
    },
}


def list_models(task: str | None = None, only_available: bool = True) -> list[dict[str, Any]]:
    """List registry entries with availability information.

    ``task`` filters by kind: ``'classification'``, ``'regression'`` or
    ``'both'`` (models usable for either task); ``None`` (the default)
    lists every entry.

    Returns a list of dicts with keys ``name``, ``task``, ``available``,
    ``requires`` and ``notes`` -- ready for pretty printing or the CLI.

    Raises
    ------
    ModelError
        When ``task`` is not one of the accepted kinds.  A typo'd value
        such as ``'clasification'`` used to silently return only the
        dual-task models instead of complaining.
    """
    if task is not None and task not in {"classification", "regression", "both"}:
        raise ModelError(
            f"task must be 'classification', 'regression' or 'both', got {task!r}"
        )
    entries = []
    for spec in MODEL_REGISTRY.values():
        if task and spec.task != "both" and spec.task != task:
            continue
        if only_available and not spec.available:
            continue
        entries.append(
            {
                "name": spec.name,
                "task": spec.task,
                "available": spec.available,
                "requires": spec.requires,
                "notes": spec.notes,
            }
        )
    return entries


def _accepted_param_names(estimator: Any) -> set[str]:
    """Parameter names :func:`build_model` accepts in ``params=``.

    ``get_params()`` alone under-reports for estimators like CatBoost,
    whose ``get_params()`` returns only the *explicitly set* constructor
    kwargs — every real CatBoost hyperparameter ('iterations', 'depth',
    'learning_rate', ...) would look unknown and be rejected with a
    spurious :class:`ModelError`.  The constructor signature fills the
    gap, but only when ``__init__`` declares no ``**kwargs``: a
    kwargs-catching constructor accepts arbitrary names, so its named
    parameters say nothing about the accepted set (there
    :meth:`set_params` stays the rejector, and :func:`build_model`
    wraps its rejections as the typed error).
    """
    accepted = set(estimator.get_params())
    try:
        parameters = inspect.signature(type(estimator).__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic constructors
        return accepted
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return accepted
    accepted.update(
        name
        for name, param in parameters.items()
        if name != "self"
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    return accepted


def build_model(
    name: str,
    task: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Instantiate an unfitted estimator from the registry.

    Parameters
    ----------
    name:
        Registry key, e.g. ``'random_forest'``.
    task:
        ``'classification'`` or ``'regression'``.
    params:
        Estimator keyword overrides (e.g. ``{'n_estimators': 500}``).
        Names are validated against ``get_params()`` ∪ the constructor
        signature, so real CatBoost hyperparameters ('iterations',
        'depth', ...) are accepted despite CatBoost's partial
        ``get_params()``; genuinely unknown names raise
        :class:`ModelError`, as do ``set_params`` rejections.

    Returns
    -------
    sklearn-compatible estimator

    Raises
    ------
    ModelError
        Unknown name or task; ``DependencyError`` when the backend is
        not installed.
    """
    key = resolve_model_name(name)
    if task not in {"classification", "regression"}:
        raise ModelError(f"task must be 'classification' or 'regression', got {task!r}")

    spec = MODEL_REGISTRY[key]
    if task != "both" and spec.task != "both" and spec.task != task:
        raise ModelError(f"Model {key!r} supports {spec.task!r}, not {task!r}")
    requires = spec.requires
    # ``requires is None`` entries are always available (see
    # ModelSpec.available), so this branch always has a concrete module name.
    if requires is not None and not spec.available:
        extra = _REQUIREMENT_EXTRAS.get(requires, requires)
        raise DependencyError(
            f"Model {key!r} requires the optional package '{requires}'. "
            f"Install with:  pip install 'pharmaml[{extra}]'"
        )

    if spec.name in _TASK_FACTORIES:
        estimator = _TASK_FACTORIES[spec.name][task]()
    else:
        factory = spec.factory
        if factory is None:
            # Dual-task entries carry factory=None and are dispatched through
            # _TASK_FACTORIES above; a factory-less entry outside that table
            # would be a registry bug, not a caller error.
            raise ModelError(
                f"Model {key!r} has no estimator factory registered for task {task!r}"
            )
        estimator = factory()

    if params:
        accepted = _accepted_param_names(estimator)
        unknown = set(params) - accepted
        if unknown:
            raise ModelError(
                f"Unknown parameters {sorted(unknown)} for {key!r}. "
                f"Valid: {sorted(accepted)[:20]}..."
            )
        try:
            estimator.set_params(**params)
        except Exception as exc:
            # Estimators with lenient constructors reject bad *values*
            # (or unknown names on **kwargs classes) inside set_params;
            # surface that as the documented typed error instead of a
            # raw backend exception (ValueError / CatBoostError / ...).
            raise ModelError(
                f"Invalid parameters {sorted(params)} for {key!r}: {exc}"
            ) from exc
    return estimator


def build_stacking(
    task: str,
    base_models: list[str],
    meta_model: str | None = None,
    passthrough: bool = False,
) -> Any:
    """Build a stacking ensemble (upgrades Stacking.py / Stacking reg.py).

    Parameters
    ----------
    task:
        ``'classification'`` or ``'regression'``.
    base_models:
        Registry names of the level-0 learners.
    meta_model:
        Registry name of the level-1 blender.  Defaults to
        ``'logistic_regression'`` for classification and ``'ridge'``
        for regression.
    passthrough:
        Append original features to meta-features.

    Raises
    ------
    ModelError
        When ``base_models`` is empty, a member name is unknown, or a
        member does not support ``task``.  Aliases in ``base_models``
        are resolved and duplicates collapsed, so ``['rf',
        'random_forest']`` builds one random-forest learner, not two
        identical ones.
    """
    if not base_models:
        raise ModelError(
            "base_models list is empty -- pass at least one registry name "
            "(see pharmaml.ml.list_models())"
        )
    if meta_model is None:
        meta_model = "logistic_regression" if task == "classification" else "ridge"
    keys = list(dict.fromkeys(resolve_model_name(name) for name in base_models))
    estimators = [(key, build_model(key, task)) for key in keys]
    meta = build_model(meta_model, task)
    if task == "classification":
        return StackingClassifier(
            estimators=estimators, final_estimator=meta, passthrough=passthrough, cv=5
        )
    return StackingRegressor(
        estimators=estimators, final_estimator=meta, passthrough=passthrough, cv=5
    )


def build_voting(
    task: str,
    base_models: list[str],
    voting: str = "soft",
    weights: list[float] | None = None,
) -> Any:
    """Build a voting ensemble (upgrades Voting.py / Voting reg.py).

    ``voting='soft'`` (probability averaging) requires all members to
    expose ``predict_proba``; ``voting='hard'`` (majority vote) works
    for every estimator and for **any number of classes**.  For
    regression the voting rule is ignored (``VotingRegressor``
    averages predictions), but the value must still be ``'soft'`` or
    ``'hard'``.  Aliases in ``base_models`` are resolved and
    duplicates collapsed, so ``['rf', 'random_forest']`` casts one
    vote, not two.

    Raises
    ------
    ModelError
        When ``base_models`` is empty, a member name is unknown,
        ``voting`` is neither ``'soft'`` nor ``'hard'``, ``weights``
        does not match the number of unique members, or a soft-voting
        member lacks ``predict_proba``.
    """
    if not base_models:
        raise ModelError(
            "base_models list is empty -- pass at least one registry name "
            "(see pharmaml.ml.list_models())"
        )
    if voting not in {"soft", "hard"}:
        raise ModelError(f"voting must be 'soft' or 'hard', got {voting!r}")
    keys = list(dict.fromkeys(resolve_model_name(name) for name in base_models))
    if weights is not None and len(weights) != len(keys):
        raise ModelError(
            f"weights has {len(weights)} entries for {len(keys)} unique base "
            "model(s) -- alias duplicates are collapsed"
        )
    estimators = [(key, build_model(key, task)) for key in keys]
    if task == "classification":
        if voting == "soft":
            unsupported = [
                key for key, est in estimators if not hasattr(est, "predict_proba")
            ]
            if unsupported:
                raise ModelError(
                    f"voting='soft' needs predict_proba on every member; "
                    f"{unsupported} lack it. Use voting='hard' (works for "
                    "binary and multiclass) or drop those models."
                )
        return VotingClassifier(estimators=estimators, voting=voting, weights=weights)
    return VotingRegressor(estimators=estimators, weights=weights)
