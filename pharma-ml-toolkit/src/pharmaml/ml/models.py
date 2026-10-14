"""Unified model registry.

The legacy codebase kept one ~150-line script per algorithm (30+ files
of near-duplicate load/preprocess/tune/evaluate boilerplate).  This
module collapses them into a registry: each entry provides the
estimator factory, the task(s) it supports, a tuned default parameter
grid and an availability probe.  One generic trainer then serves every
algorithm.

Optional backends (XGBoost, LightGBM, CatBoost, Keras-style MLP is
sklearn-based here) are imported lazily and reported as unavailable
rather than crashing the registry import.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

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
        (random_state already threaded in).
    param_grid:
        Compact default grid for hyperparameter search.
    requires:
        Optional third-party module needed for the factory.
    notes:
        Short human description used by ``list_models`` output.
    """

    name: str
    task: str
    factory: Callable[..., Any]
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
    from xgboost import XGBClassifier  # noqa: PLC0415

    return XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)


def _xgb_reg():
    from xgboost import XGBRegressor  # noqa: PLC0415

    return XGBRegressor(random_state=42, n_jobs=-1)


def _lgbm_cls():
    from lightgbm import LGBMClassifier  # noqa: PLC0415

    return LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)


def _lgbm_reg():
    from lightgbm import LGBMRegressor  # noqa: PLC0415

    return LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)


def _cat_cls():
    from catboost import CatBoostClassifier  # noqa: PLC0415

    return CatBoostClassifier(random_state=42, verbose=False)


def _cat_reg():
    from catboost import CatBoostRegressor  # noqa: PLC0415

    return CatBoostRegressor(random_state=42, verbose=False)


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
    "knn": ModelSpec(
        name="knn",
        task="both",
        factory=None,  # replaced below -- see _build for dispatch
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
    "ridge": "ridge",
    "ridge_cls": "ridge_classifier",
    "lasso": "lasso",
    "enet": "elastic_net",
    "elasticnet": "elastic_net",
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
        "classification": AdaBoostClassifier,
        "regression": AdaBoostRegressor,
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
}


def list_models(task: str | None = None, only_available: bool = True) -> list[dict[str, Any]]:
    """List registry entries with availability information.

    Returns a list of dicts with keys ``name``, ``task``, ``available``,
    ``requires`` and ``notes`` -- ready for pretty printing or the CLI.
    """
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
    if not spec.available:
        raise DependencyError(
            f"Model {key!r} requires the optional package '{spec.requires}'. "
            f"Install with:  pip install 'pharmaml[boosting]'"
        )

    if spec.name in _TASK_FACTORIES:
        estimator = _TASK_FACTORIES[spec.name][task]()
    else:
        estimator = spec.factory()

    if params:
        unknown = set(params) - set(estimator.get_params())
        if unknown:
            raise ModelError(
                f"Unknown parameters {sorted(unknown)} for {key!r}. "
                f"Valid: {sorted(estimator.get_params())[:20]}..."
            )
        estimator.set_params(**params)
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
    """
    if meta_model is None:
        meta_model = "logistic_regression" if task == "classification" else "ridge"
    estimators = [(name, build_model(name, task)) for name in base_models]
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
    for every estimator and for **any number of classes**.
    """
    estimators = [(name, build_model(name, task)) for name in base_models]
    if task == "classification":
        if voting == "soft":
            unsupported = [
                name for name, est in estimators if not hasattr(est, "predict_proba")
            ]
            if unsupported:
                raise ModelError(
                    f"voting='soft' needs predict_proba on every member; "
                    f"{unsupported} lack it. Use voting='hard' (works for "
                    "binary and multiclass) or drop those models."
                )
        return VotingClassifier(estimators=estimators, voting=voting, weights=weights)
    return VotingRegressor(estimators=estimators, weights=weights)
