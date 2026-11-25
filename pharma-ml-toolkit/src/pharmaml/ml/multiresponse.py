"""Generic multi-response (multi-output) regression -- part17 §5.

The scientific grounding, stated once up front: most real pharma ML
endpoints are not one column of ``y`` but several -- solubility AND
permeability AND logP from the same descriptor block, IC50 across a
panel of targets, or a formulation's three quality responses from the
same factor matrix.  scikit-learn answers with three honest mechanisms
(the cross-decomposition and multioutput guides the mandate cites):

* **native** -- the estimator's own ``fit`` accepts a 2-D ``Y``
  (``PLSRegression``/PLS2, ``MultiTaskElasticNet`` /
  ``MultiTaskLasso``, ``Ridge``, ``LinearRegression``,
  ``KNeighborsRegressor``, the tree ensembles, ``MLPRegressor``).
* **wrapper** -- :class:`~sklearn.multioutput.MultiOutputRegressor`
  clones one single-output base per target (per-target independence;
  the ONLY honest route for estimators such as
  ``GradientBoostingRegressor``/``SVR`` whose ``fit`` enforces a
  1-D ``y``).
* **chain** -- :class:`~sklearn.multioutput.RegressorChain` feeds each
  target's predictions as extra features for the next (order matters;
  error propagation is the price of the coupling).

This module's registry says WHICH mechanism each strategy rides and
never pretends a 1-D-only estimator is native.  What lands:

* :data:`STRATEGIES` -- the extensible registry (one
  :class:`MultiResponseStrategy` per key: mechanism, the honest
  capability flags, the estimator factory, the notes).
* :class:`PerTargetMultiOutput` -- the masked multi-task wrapper for
  the per-target missing-value strategy: each target's clone is fit on
  that target's own non-missing rows (the mandate's "a compatible
  masked multi-task approach when implemented and tested"), riding the
  SAME per-target independence ``MultiOutputRegressor`` expresses.
* :func:`train_multi_response` -- the full contract: feature/target
  validation (>=2 numeric targets), missing-target strategies
  (``complete_case`` always; ``per_target`` where the mechanism
  supports it -- refused with a typed error elsewhere, never silent),
  the leakage-safe order (split FIRST, then preprocessing and
  per-target scaling fit on the train partition only), group-aware
  split, k-fold cross-validation over the SAME full procedure, and
  per-target evaluation (R2 / RMSE / MAE / MSE / explained variance
  per target -- never hidden behind the aggregate), plus the uniform
  and user-weighted aggregates and the target correlation matrix.
* :class:`MultiResponseBundle` -- the persistence carrier: the fitted
  estimator, the feature pipeline, the per-target scalers, and the
  complete training contract (§5.6's list verbatim) as a JSON-safe
  dict; pickles through the session model store and the joblib
  artifact lane unchanged.
* :func:`predict_multi_response` -- multi-column predictions in the
  STABLE target order with per-target inverse scaling back to raw
  units.
* :func:`demo_multiresponse_dataset` -- the seeded SYNTHETIC demo
  frame (240 x 5 features, 3 correlated responses, labelled
  synthetic wherever it travels; the seed is the only generative
  dial).

The honest scope: no per-target loss aggregation inside the fit (the
aggregate lives in the evaluation, not the objective), no native
cross-target covariance modelling beyond what PLS2/MultiTask penalties
themselves carry, and ``per_target`` missing handling is a wrapper
mechanism lane (a native/chain estimator needs the full Y block --
complete_case is that lane's honest option).  The multi-target metrics
are computed per column on the raw-unit predictions after inverse
scaling, so every number reads in the target's own units.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskLasso,
    Ridge,
)
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

__all__ = [
    "METRIC_NAMES",
    "MISSING_STRATEGIES",
    "STRATEGIES",
    "MultiResponseBundle",
    "MultiResponseStrategy",
    "PerTargetMultiOutput",
    "demo_multiresponse_dataset",
    "environment_metadata",
    "predict_multi_response",
    "strategy_registry_payload",
    "train_multi_response",
]

# the residual/prediction sample bound (the house precedent: bounded
# rows, the bound reported, never silent).
MAX_SAMPLE_ROWS = 500


# ---------------------------------------------------------------------------
# the base-estimator factories (the wrapper/chain strategies' legs)
# ---------------------------------------------------------------------------


def _ridge_factory(params: dict[str, Any], seed: int):
    return Ridge(alpha=float(params.get("alpha", 1.0)), random_state=seed)


def _pls2_factory(params: dict[str, Any], seed: int):
    return PLSRegression(
        n_components=int(params.get("n_components", 2)),
        max_iter=int(params.get("max_iter", 500)),
    )


def _mten_factory(params: dict[str, Any], seed: int):
    return MultiTaskElasticNet(
        alpha=float(params.get("alpha", 1.0)),
        l1_ratio=float(params.get("l1_ratio", 0.5)),
        random_state=seed,
    )


def _mtlasso_factory(params: dict[str, Any], seed: int):
    return MultiTaskLasso(
        alpha=float(params.get("alpha", 1.0)),
        random_state=seed,
    )


def _linear_factory(params: dict[str, Any], seed: int):
    return LinearRegression()


def _knn_factory(params: dict[str, Any], seed: int):
    return KNeighborsRegressor(n_neighbors=int(params.get("n_neighbors", 5)))


def _tree_factory(params: dict[str, Any], seed: int):
    return DecisionTreeRegressor(max_depth=params.get("max_depth"), random_state=seed)


def _rf_factory(params: dict[str, Any], seed: int):
    return RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=params.get("max_depth"),
        random_state=seed,
    )


def _extra_trees_factory(params: dict[str, Any], seed: int):
    return ExtraTreesRegressor(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=params.get("max_depth"),
        random_state=seed,
    )


def _mlp_factory(params: dict[str, Any], seed: int):
    hidden = params.get("hidden_layer_sizes", (64, 32))
    if isinstance(hidden, (list, tuple)):
        hidden = tuple(int(h) for h in hidden)
    return MLPRegressor(
        hidden_layer_sizes=hidden,
        max_iter=int(params.get("max_iter", 300)),
        random_state=seed,
    )


def _svr_factory(params: dict[str, Any], seed: int):
    return SVR(C=float(params.get("C", 1.0)), kernel=str(params.get("kernel", "rbf")))


def _gbr_factory(params: dict[str, Any], seed: int):
    # used as a WRAPPER base only (GBR's fit enforces 1-D y -- the
    # honest reason the wrapper lane exists)
    return GradientBoostingRegressor(
        n_estimators=int(params.get("n_estimators", 100)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        random_state=seed,
    )


_BASE_FACTORIES: dict[str, Callable[[dict[str, Any], int], Any]] = {
    "ridge": _ridge_factory,
    "linear_regression": _linear_factory,
    "knn": _knn_factory,
    "decision_tree": _tree_factory,
    "random_forest": _rf_factory,
    "svr": _svr_factory,
    "gradient_boosting": _gbr_factory,
}


@dataclass(frozen=True)
class MultiResponseStrategy:
    """One registry entry -- the mechanism is stated, never implied."""

    key: str
    label: str
    mechanism: Literal["native", "wrapper", "chain"]
    factory: Callable[[dict[str, Any], int], Any]
    sample_weight_support: bool
    notes: str
    base_keys: tuple[str, ...] | None = None  # wrapper/chain base choices
    default_base: str | None = None

    def build(self, params: dict[str, Any], seed: int) -> Any:
        params = dict(params or {})
        if self.mechanism in ("wrapper", "chain"):
            base_key = str(params.pop("base", self.default_base))
            if base_key not in _BASE_FACTORIES:
                raise ValueError(
                    f"unknown base estimator {base_key!r} for strategy "
                    f"{self.key!r}; choose one of {sorted(_BASE_FACTORIES)}"
                )
            base = _BASE_FACTORIES[base_key](params, seed)
            if self.mechanism == "wrapper":
                return MultiOutputRegressor(estimator=base, n_jobs=None)
            order = params.pop("order", None)
            return RegressorChain(base_estimator=base, order=order)
        return self.factory(params, seed)


STRATEGIES: dict[str, MultiResponseStrategy] = {
    s.key: s  # type: ignore[misc]
    for s in [
        MultiResponseStrategy(
            key="pls2",
            label="PLSRegression (PLS2)",
            mechanism="native",
            factory=_pls2_factory,
            sample_weight_support=False,
            notes=(
                "the cross-decomposition classic: latent components model "
                "the X<->Y blocks JOINTLY, the only native strategy here "
                "that couples the targets through a shared latent space; "
                "n_components must stay <= min(n_samples, n_features) and "
                "the fit itself carries no sample_weight"
            ),
        ),
        MultiResponseStrategy(
            key="multi_task_elastic_net",
            label="MultiTaskElasticNet",
            mechanism="native",
            factory=_mten_factory,
            sample_weight_support=False,
            notes=(
                "the natively 2-D elastic net: the l1_ratio penalty mixes "
                "the shared block sparsity with per-target noise sharing; "
                "needs standardized features to behave; no sample_weight"
            ),
        ),
        MultiResponseStrategy(
            key="multi_task_lasso",
            label="MultiTaskLasso",
            mechanism="native",
            factory=_mtlasso_factory,
            sample_weight_support=False,
            notes=(
                "the multi-task l1: one sparsity pattern across all targets "
                "simultaneously; no sample_weight"
            ),
        ),
        MultiResponseStrategy(
            key="ridge",
            label="Ridge",
            mechanism="native",
            factory=_ridge_factory,
            sample_weight_support=True,
            notes=(
                "the closed-form baseline: natively 2-D Y, one shared "
                "coefficient fit over all targets; supports sample_weight"
            ),
        ),
        MultiResponseStrategy(
            key="linear_regression",
            label="LinearRegression (OLS)",
            mechanism="native",
            factory=_linear_factory,
            sample_weight_support=True,
            notes=(
                "ordinary least squares over the stacked target block -- "
                "multi-target by construction; supports sample_weight"
            ),
        ),
        MultiResponseStrategy(
            key="knn",
            label="KNeighborsRegressor",
            mechanism="native",
            factory=_knn_factory,
            sample_weight_support=True,
            notes=(
                "nearest-neighbour regression is natively multi-output "
                "(the neighbours' target VECTORS are averaged); supports "
                "sample_weight"
            ),
        ),
        MultiResponseStrategy(
            key="decision_tree",
            label="DecisionTreeRegressor",
            mechanism="native",
            factory=_tree_factory,
            sample_weight_support=True,
            notes="a single tree splits on joint node impurity across all targets",
        ),
        MultiResponseStrategy(
            key="random_forest",
            label="RandomForestRegressor",
            mechanism="native",
            factory=_rf_factory,
            sample_weight_support=True,
            notes=(
                "the ensemble averages multi-output leaf vectors natively; "
                "supports sample_weight"
            ),
        ),
        MultiResponseStrategy(
            key="extra_trees",
            label="ExtraTreesRegressor",
            mechanism="native",
            factory=_extra_trees_factory,
            sample_weight_support=True,
            notes="extremely randomized trees over the joint target block",
        ),
        MultiResponseStrategy(
            key="mlp_regressor",
            label="MLPRegressor",
            mechanism="native",
            factory=_mlp_factory,
            sample_weight_support=False,
            notes=(
                "the multi-layer perceptron with a 2-D output layer -- "
                "native multi-output; no sample_weight in its fit"
            ),
        ),
        MultiResponseStrategy(
            key="multi_output",
            label="MultiOutputRegressor (wrapper)",
            mechanism="wrapper",
            factory=lambda params, seed: None,  # built through .build()
            sample_weight_support=True,
            notes=(
                "one clone of the base estimator per target -- per-target "
                "independence, the ONLY honest route for 1-D-only "
                "estimators (SVR, GradientBoosting); the wrapper's fit "
                "forwards sample_weight to each clone; and the ONLY "
                "mechanism whose per-target structure admits the "
                "per_target missing-value strategy"
            ),
            base_keys=tuple(sorted(_BASE_FACTORIES)),
            default_base="ridge",
        ),
        MultiResponseStrategy(
            key="regressor_chain",
            label="RegressorChain (chain)",
            mechanism="chain",
            factory=lambda params, seed: None,  # built through .build()
            sample_weight_support=False,
            notes=(
                "each target is predicted in chain order with the earlier "
                "targets' PREDICTIONS appended as features -- order matters "
                "(explicit 'order' param), error propagation is the price "
                "of the coupling, and the fit accepts no sample_weight "
                "in the pinned scikit-learn"
            ),
            base_keys=tuple(sorted(_BASE_FACTORIES)),
            default_base="ridge",
        ),
    ]
}

MISSING_STRATEGIES: dict[str, dict[str, Any]] = {
    "complete_case": {
        "label": "Complete-case analysis",
        "description": (
            "rows missing ANY target column are dropped once before the "
            "split; the usable-row count per target is reported BEFORE "
            "the drop, so the loss is visible, never silent"
        ),
        "mechanisms": ["native", "wrapper", "chain"],
    },
    "per_target": {
        "label": "Per-target fitting (masked multi-task)",
        "description": (
            "each target's clone is fit on that target's own non-missing "
            "rows (the PerTargetMultiOutput masked multi-task wrapper); "
            "test rows are likewise evaluated per target on that target's "
            "non-missing rows -- one target's gaps never erase another "
            "target's rows"
        ),
        "mechanisms": ["wrapper"],
    },
}

METRIC_NAMES = ("r2", "rmse", "mae", "mse", "explained_variance")


class PerTargetMultiOutput(BaseEstimator):
    """The masked multi-task wrapper -- per-target clones on per-target rows.

    Exactly ``MultiOutputRegressor``'s per-target independence, extended
    to per-target missing masks: target ``j``'s clone is fit on the rows
    where ``Y[:, j]`` is present (and predicts every row).  This is the
    mandate's "compatible masked multi-task approach, implemented and
    tested" -- the ONLY lane that can honour "do not silently drop large
    numbers of samples because one of many target columns is missing"
    at the FIT level (the native/chain mechanisms need the complete Y
    block; complete_case is their honest option).
    """

    def __init__(self, base_estimator: Any) -> None:
        self.base_estimator = base_estimator

    def fit(self, X: Any, Y: Any, sample_weight: Any = None) -> PerTargetMultiOutput:
        Y = np.asarray(Y, dtype=float)
        self.n_targets_ = int(Y.shape[1])
        self.estimators_ = []
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        for j in range(self.n_targets_):
            mask = np.isfinite(Y[:, j])
            est = clone(self.base_estimator)
            if np.all(mask):
                if sw is not None:
                    est.fit(X, Y[:, j], sample_weight=sw)
                else:
                    est.fit(X, Y[:, j])
            else:
                Xj = np.asarray(X, dtype=float)[mask]
                swj = None if sw is None else sw[mask]
                if swj is not None:
                    est.fit(Xj, Y[mask, j], sample_weight=swj)
                else:
                    est.fit(Xj, Y[mask, j])
            self.estimators_.append(est)
        return self

    def predict(self, X: Any) -> Any:
        X = np.asarray(X, dtype=float)
        cols = [np.asarray(est.predict(X)).ravel() for est in self.estimators_]
        return np.column_stack(cols)


@dataclass
class MultiResponseBundle:
    """The persistence carrier (§5.6): everything a reload needs.

    ``contract`` is the JSON-safe dict the wire travels on; the fitted
    objects (estimator, feature pipeline, per-target scalers) pickle
    through the session model store and the joblib artifact lane.
    """

    estimator: Any
    feature_pipeline: Any
    target_scalers: list[Any]  # one per target, or empty (no scaling)
    target_names: list[str]
    feature_names: list[str]
    contract: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# the metric helpers
# ---------------------------------------------------------------------------


def _metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "explained_variance": float(explained_variance_score(actual, predicted)),
    }


def _aggregate(
    values: dict[str, dict[str, Any]], weights: np.ndarray
) -> dict[str, dict[str, float]]:
    uniform: dict[str, float] = {}
    weighted: dict[str, float] = {}
    for metric in METRIC_NAMES:
        per = np.array([values[t][metric] for t in values], dtype=float)
        uniform[metric] = float(per.mean())
        weighted[metric] = float(np.dot(per, weights) / weights.sum())
    return {"uniform": uniform, "weighted": weighted}


def demo_multiresponse_dataset(seed: int = 42, n_rows: int = 240) -> pd.DataFrame:
    """The seeded SYNTHETIC demo frame -- 5 features, 3 correlated responses.

    Deliberately correlated structure (y1 shares x2 with y3, y1/y2 share
    x1) so the target correlation matrix has honest off-diagonals to
    show; labelled synthetic wherever it travels.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n_rows)
    x2 = rng.normal(0.0, 1.0, n_rows)
    x3 = rng.normal(0.0, 1.0, n_rows)
    x4 = rng.normal(0.0, 1.0, n_rows)
    x5 = rng.normal(0.0, 1.0, n_rows)
    y1 = 2.0 * x1 - 0.5 * x2 + 0.1 * x5 + rng.normal(0.0, 0.10, n_rows)
    y2 = 1.0 * x1 + 1.5 * x3 + rng.normal(0.0, 0.15, n_rows)
    y3 = -0.7 * x2 + 0.9 * x4 + rng.normal(0.0, 0.12, n_rows)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5,
            "response_a": y1,
            "response_b": y2,
            "response_c": y3,
        }
    )


DEMO_FEATURES = ["x1", "x2", "x3", "x4", "x5"]
DEMO_TARGETS = ["response_a", "response_b", "response_c"]


def strategy_registry_payload() -> dict[str, Any]:
    """The registry vocabulary (the route's GET /registry body)."""
    strategies = []
    for key, spec in STRATEGIES.items():
        strategies.append(
            {
                "key": key,
                "label": spec.label,
                "mechanism": spec.mechanism,
                "native_multi_output": spec.mechanism == "native",
                "sample_weight_support": spec.sample_weight_support,
                "notes": spec.notes,
                "base_keys": list(spec.base_keys or ()),
                "default_base": spec.default_base,
            }
        )
    return {
        "strategies": strategies,
        "missing_strategies": MISSING_STRATEGIES,
        "metrics": list(METRIC_NAMES),
        "max_sample_rows": MAX_SAMPLE_ROWS,
        "demo_dataset": {
            "features": list(DEMO_FEATURES),
            "targets": list(DEMO_TARGETS),
            "n_rows": 240,
            "synthetic": True,
            "seed": 42,
            "note": (
                "the seeded synthetic demonstration frame -- labelled "
                "synthetic wherever it travels; every number it produces "
                "is a model estimate over generated data, never an "
                "experimental fact"
            ),
        },
        "scope_statement": (
            "the mechanism is stated per strategy, never implied: native "
            "means the estimator's own fit accepts the 2-D Y block "
            "(PLS2, MultiTaskElasticNet/Lasso, Ridge, OLS, KNN, the "
            "trees, MLP); wrapper means MultiOutputRegressor per-target "
            "clones (the only honest route for SVR/GBR whose fit enforces "
            "1-D y); chain means RegressorChain's predicted-target "
            "feature feed. per_target missing handling rides the wrapper "
            "mechanism only -- a native/chain estimator needs the "
            "complete Y block, complete_case is that lane's honest "
            "option. metrics are computed per target on raw-unit "
            "predictions after inverse scaling; the uniform/weighted "
            "aggregates are reported ALONGSIDE, never instead."
        ),
    }


def environment_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------


def _validate_columns(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not target_columns or len(target_columns) < 2:
        raise ValueError("multi-response training needs >= 2 target columns")
    if not feature_columns:
        raise ValueError("multi-response training needs >= 1 feature column")
    overlap = set(feature_columns) & set(target_columns)
    if overlap:
        raise ValueError(
            f"columns cannot be both feature and target: {sorted(overlap)}"
        )
    unknown = sorted(
        (set(feature_columns) | set(target_columns)) - set(map(str, frame.columns))
    )
    if unknown:
        raise ValueError(f"columns not in the dataset: {unknown}")
    if len(set(target_columns)) != len(target_columns):
        raise ValueError("target columns must be unique")
    X = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    for c in feature_columns:
        if int(X[c].isna().sum()) - int(frame[c].isna().sum()) > 0:
            raise ValueError(f"non-numeric feature column: {c}")
    Y = frame[target_columns].apply(pd.to_numeric, errors="coerce")
    for c in target_columns:
        if int(Y[c].isna().sum()) - int(frame[c].isna().sum()) > 0:
            raise ValueError(f"non-numeric target column: {c}")
    return X, Y


def _make_feature_pipeline(preprocess: bool) -> Pipeline:
    if preprocess:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
    return Pipeline(
        [("identity", StandardScaler(with_mean=False, with_std=False))]
    )


def _fit_all_parts(
    X_raw: pd.DataFrame,
    Y: pd.DataFrame,
    strategy_key: str,
    params: dict[str, Any],
    preprocess: bool,
    target_scaling: bool,
    seed: int,
    sample_weight: np.ndarray | None,
    missing: str,
):
    """Fit the full procedure on ONE partition (the leakage-safe unit).

    Feature pipeline and per-target scalers are fit on THIS partition's
    data only (per-target scalers on each target's own non-missing rows
    when the missing lane is per_target); the estimator sees the
    transformed X and the (optionally scaled) Y.
    """
    spec = STRATEGIES[strategy_key]
    feature_pipeline = _make_feature_pipeline(preprocess)
    X_tr = feature_pipeline.fit_transform(X_raw)
    target_scalers: list[Any] = []
    if target_scaling:
        cols = []
        for col in Y.columns:
            scaler = StandardScaler()
            values = Y[[col]].to_numpy(dtype=float)
            if missing == "per_target":
                mask = np.isfinite(values.ravel())
                scaler.fit(values[mask])
            else:
                scaler.fit(values)
            cols.append(scaler.transform(values).ravel())
            target_scalers.append(scaler)
        Y_fit = np.column_stack(cols)
    else:
        Y_fit = Y.to_numpy(dtype=float)
    if missing == "per_target":
        base_params = {k: v for k, v in params.items() if k != "base"}
        base_key = str(params.get("base", spec.default_base))
        base = _BASE_FACTORIES[base_key](base_params, seed)
        estimator = PerTargetMultiOutput(base_estimator=base)
        estimator.fit(X_tr, Y_fit, sample_weight=sample_weight)
    else:
        estimator = spec.build(params, seed)
        if sample_weight is not None and spec.sample_weight_support:
            estimator.fit(X_tr, Y_fit, sample_weight=sample_weight)
        else:
            estimator.fit(X_tr, Y_fit)
    return estimator, feature_pipeline, target_scalers


def _predict_parts(
    estimator: Any,
    feature_pipeline: Any,
    target_scalers: list[Any],
    X_raw: pd.DataFrame,
    target_columns: list[str],
) -> pd.DataFrame:
    X = feature_pipeline.transform(X_raw)
    Y_pred = np.asarray(estimator.predict(X), dtype=float)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)
    out = pd.DataFrame(Y_pred, index=X_raw.index, columns=list(target_columns))
    if target_scalers:
        for j, col in enumerate(target_columns):
            out[col] = target_scalers[j].inverse_transform(
                out[[col]].to_numpy(dtype=float)
            ).ravel()
    return out


def train_multi_response(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    strategy_key: str = "ridge",
    params: dict[str, Any] | None = None,
    preprocess: bool = True,
    target_scaling: bool = False,
    test_size: float = 0.25,
    cv_folds: int = 5,
    missing: str = "complete_case",
    group_column: str | None = None,
    sample_weight_column: str | None = None,
    target_weights: list[float] | None = None,
    random_state: int = 42,
    target_units: dict[str, str] | None = None,
    environment_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The full §5.4/§5.5 training round trip over a real frame.

    Returns the result dict the route serialises (metrics, residuals,
    CV, usable counts, split report, target correlation matrix, and the
    §5.6 persistence contract).  Raises ``ValueError`` with the honest
    typed message on every contract violation (the route maps it to a
    422, never a silent default).
    """
    started = time.time()
    params = dict(params or {})
    if strategy_key not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy_key!r}")
    spec = STRATEGIES[strategy_key]
    if missing not in MISSING_STRATEGIES:
        raise ValueError(f"unknown missing-value strategy {missing!r}")
    if missing == "per_target" and spec.mechanism != "wrapper":
        raise ValueError(
            "per_target missing handling rides the multi_output wrapper "
            f"mechanism only; {strategy_key!r} is {spec.mechanism} and "
            "needs the complete Y block (complete_case is its lane)"
        )
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2")
    X_all, Y_all = _validate_columns(frame, feature_columns, target_columns)
    target_columns = list(target_columns)
    if target_weights is not None:
        if len(target_weights) != len(target_columns):
            raise ValueError("target_weights must have one entry per target column")
        if any(w < 0 for w in target_weights):
            raise ValueError("target_weights must be non-negative")
        if sum(target_weights) <= 0:
            raise ValueError("target_weights must not all be zero")
    weights = np.array(
        target_weights if target_weights is not None else [1.0] * len(target_columns),
        dtype=float,
    )

    # usable counts BEFORE any drop (the mandate's visibility rule)
    missing_counts = {c: int(Y_all[c].isna().sum()) for c in target_columns}
    usable_counts = {c: int(Y_all[c].notna().sum()) for c in target_columns}

    sample_weight: np.ndarray | None = None
    if sample_weight_column is not None:
        if sample_weight_column not in frame.columns:
            raise ValueError(
                f"weights column not in the dataset: {sample_weight_column}"
            )
        if not spec.sample_weight_support:
            raise ValueError(
                f"strategy {strategy_key!r} does not support sample weights "
                "(its estimator's fit accepts none -- the registry says so)"
            )
        sw = pd.to_numeric(frame[sample_weight_column], errors="coerce")
        if sw.isna().any() or (sw < 0).any():
            raise ValueError(
                "the weights column must be numeric, non-negative, complete"
            )
        sample_weight = sw.to_numpy(dtype=float)

    # the split FIRST (leakage discipline: nothing downstream sees the
    # test rows before its own partition-fit)
    if group_column is not None:
        if group_column not in frame.columns:
            raise ValueError(f"group column not in the dataset: {group_column}")
        if group_column in feature_columns:
            raise ValueError("the group column cannot also be a feature column")
        groups_all = frame[group_column].to_numpy()
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_pos, test_pos = next(splitter.split(X_all, groups=groups_all))
        split_strategy = f"group_shuffle_split(group={group_column})"
        n_groups_train = len(set(groups_all[train_pos]))
        n_groups_test = len(set(groups_all[test_pos]))
        group_overlap = sorted(
            set(map(str, groups_all[train_pos])) & set(map(str, groups_all[test_pos]))
        )
        groups_train_series = pd.Series(
            groups_all[train_pos], index=X_all.iloc[train_pos].index
        )
    else:
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(len(X_all))
        n_test = max(1, round(test_size * len(X_all)))
        test_pos, train_pos = shuffled[:n_test], shuffled[n_test:]
        split_strategy = "shuffle_split"
        n_groups_train = None
        n_groups_test = None
        group_overlap = []
        groups_train_series = None
    X_train_raw = X_all.iloc[train_pos]
    X_test_raw = X_all.iloc[test_pos]
    Y_train_all = Y_all.iloc[train_pos]
    Y_test_all = Y_all.iloc[test_pos]

    # missing-target strategy on the TRAIN partition
    if missing == "complete_case":
        complete = Y_train_all.notna().all(axis=1)
        dropped_train_rows = int((~complete).sum())
        Y_train = Y_train_all[complete]
        X_train_fit = X_train_raw[complete]
        groups_fit = (
            groups_train_series[complete.to_numpy()]
            if groups_train_series is not None
            else None
        )
        sw_fit = (
            sample_weight[train_pos][complete.to_numpy()]
            if sample_weight is not None
            else None
        )
    else:  # per_target -- the masked wrapper lane
        dropped_train_rows = 0
        Y_train = Y_train_all
        X_train_fit = X_train_raw
        groups_fit = groups_train_series
        sw_fit = sample_weight[train_pos] if sample_weight is not None else None

    fitted = _fit_all_parts(
        X_train_fit,
        Y_train,
        strategy_key,
        params,
        preprocess,
        target_scaling,
        random_state,
        sw_fit,
        missing,
    )
    estimator, feature_pipeline, target_scalers = fitted

    # test predictions -- per-target evaluation on that target's own
    # non-missing test rows (the mandate's anti-silent-drop rule)
    Y_pred_test = _predict_parts(
        estimator, feature_pipeline, target_scalers, X_test_raw, target_columns
    )
    per_target: dict[str, dict[str, Any]] = {}
    residual_rows: dict[str, dict[str, Any]] = {}
    for col in target_columns:
        mask = Y_test_all[col].notna()
        actual = Y_test_all.loc[mask, col].to_numpy(dtype=float)
        predicted = Y_pred_test.loc[mask, col].to_numpy(dtype=float)
        per_target[col] = {
            **_metric_values(actual, predicted),
            "n_test_rows": int(mask.sum()),
        }
        n_res = min(len(actual), MAX_SAMPLE_ROWS)
        residual_rows[col] = {
            "actual": [float(v) for v in actual[:n_res]],
            "predicted": [float(v) for v in predicted[:n_res]],
            "residual": [
                float(a - p) for a, p in zip(actual[:n_res], predicted[:n_res])
            ],
            "n_rows": len(actual),
            "n_sampled": int(n_res),
        }
    aggregates = _aggregate(per_target, weights)

    # target correlation matrix on the complete-case Y block
    complete_all = Y_all.notna().all(axis=1)
    if int(complete_all.sum()) >= 2:
        corr = Y_all[complete_all].corr(method="pearson")
        correlation_matrix: dict[str, dict[str, Any]] = {
            col: {
                other: (
                    None
                    if not np.isfinite(corr.loc[col, other])
                    else float(corr.loc[col, other])
                )
                for other in target_columns
            }
            for col in target_columns
        }
    else:
        correlation_matrix = {col: {o: None for o in target_columns} for col in target_columns}

    # cross-validation over the SAME full procedure (per-fold pipeline
    # and scalars fit on the fold-train only -- the leakage rule holds
    # inside every fold)
    if groups_fit is not None:
        gkf = GroupKFold(n_splits=min(cv_folds, int(groups_fit.nunique())))
        splitter_cv = gkf.split(X_train_fit, groups=groups_fit.to_numpy())
    else:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        splitter_cv = kf.split(X_train_fit)
    fold_metrics: dict[str, list[float]] = {c: [] for c in target_columns}
    fold_records: list[dict[str, Any]] = []
    for fold_idx, (tr_i, te_i) in enumerate(splitter_cv):
        Xf_tr, Xf_te = X_train_fit.iloc[tr_i], X_train_fit.iloc[te_i]
        Yf_tr, Yf_te = Y_train.iloc[tr_i], Y_train.iloc[te_i]
        fold_fit = _fit_all_parts(
            Xf_tr,
            Yf_tr,
            strategy_key,
            params,
            preprocess,
            target_scaling,
            random_state,
            None,
            missing,
        )
        Yf_pred = _predict_parts(
            fold_fit[0], fold_fit[1], fold_fit[2], Xf_te, target_columns
        )
        fold_row: dict[str, Any] = {"fold": fold_idx}
        for col in target_columns:
            m = Yf_te[col].notna()
            vals = _metric_values(
                Yf_te.loc[m, col].to_numpy(dtype=float),
                Yf_pred.loc[m, col].to_numpy(dtype=float),
            )
            fold_metrics[col].append(vals["r2"])
            fold_row[col] = vals["r2"]
        fold_records.append(fold_row)
    cv_summary: dict[str, Any] = {
        "folds": len(fold_records),
        "per_target": {},
        "aggregate": {},
        "folds_table": fold_records,
    }
    for col in target_columns:
        arr = np.array(fold_metrics[col], dtype=float)
        cv_summary["per_target"][col] = {
            "r2_mean": float(arr.mean()),
            "r2_std": float(arr.std()),
            "r2_folds": [float(v) for v in arr],
        }
    cv_summary["aggregate"] = {
        "r2_uniform": float(
            np.mean([cv_summary["per_target"][c]["r2_mean"] for c in target_columns])
        ),
        "r2_weighted": float(
            np.dot(
                np.array([cv_summary["per_target"][c]["r2_mean"] for c in target_columns]),
                weights,
            )
            / weights.sum()
        ),
    }

    fit_seconds = time.time() - started

    contract = {
        "target_names": list(target_columns),
        "target_order": list(target_columns),
        "target_units": dict(target_units or {}),
        "target_scaler": (
            {
                col: {
                    "kind": "StandardScaler",
                    "mean": float(target_scalers[j].mean_[0]),
                    "scale": float(target_scalers[j].scale_[0]),
                }
                for j, col in enumerate(target_columns)
            }
            if target_scaling and target_scalers
            else {"kind": "none"}
        ),
        "feature_names": list(feature_columns),
        "preprocessing": {
            "impute": "median",
            "feature_scaling": "StandardScaler" if preprocess else "none",
        },
        "estimator_strategy": {
            "key": strategy_key,
            "label": spec.label,
            "mechanism": spec.mechanism,
            "params": dict(params),
        },
        "hyperparameters": dict(params),
        "split": {
            "strategy": split_strategy,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "group_column": group_column,
        },
        "missing_strategy": missing,
        "sample_weight_column": sample_weight_column,
        "target_weights": [float(w) for w in weights.tolist()],
        "random_seed": int(random_state),
        "cv_folds": int(cv_folds),
        "per_target_metrics": {c: dict(per_target[c]) for c in target_columns},
        "aggregate_metrics": aggregates,
        "n_train_rows": len(X_train_fit),
        "n_test_rows": len(X_test_raw),
        "usable_rows_per_target": usable_counts,
        "missing_counts_per_target": missing_counts,
        "environment": {
            **environment_metadata(),
            **(environment_extra or {}),
        },
    }
    bundle = MultiResponseBundle(
        estimator=estimator,
        feature_pipeline=feature_pipeline,
        target_scalers=target_scalers,
        target_names=list(target_columns),
        feature_names=list(feature_columns),
        contract=contract,
    )
    return {
        "bundle": bundle,
        "contract": contract,
        "per_target_metrics": per_target,
        "aggregate_metrics": aggregates,
        "target_weights": [float(w) for w in weights.tolist()],
        "residuals": {"per_target": residual_rows, "max_rows": MAX_SAMPLE_ROWS},
        "target_correlation_matrix": correlation_matrix,
        "cv": cv_summary,
        "data": {
            "n_rows": len(frame),
            "usable_rows_per_target": usable_counts,
            "missing_counts_per_target": missing_counts,
            "dropped_train_rows": dropped_train_rows,
            "split": {
                "strategy": split_strategy,
                "train_rows": len(X_train_fit),
                "test_rows": len(X_test_raw),
                "n_train_groups": n_groups_train,
                "n_test_groups": n_groups_test,
                "group_overlap": group_overlap,
                "test_size": float(test_size),
                "random_state": int(random_state),
            },
        },
        "fit_seconds": float(fit_seconds),
    }


def predict_multi_response(
    bundle: MultiResponseBundle, rows: pd.DataFrame
) -> dict[str, Any]:
    """Multi-column predictions in the STABLE target order (§5.6)."""
    missing_cols = sorted(set(bundle.feature_names) - set(map(str, rows.columns)))
    if missing_cols:
        raise ValueError(f"rows missing feature columns: {missing_cols}")
    X_raw = rows[bundle.feature_names].apply(pd.to_numeric, errors="coerce")
    preds = _predict_parts(
        bundle.estimator,
        bundle.feature_pipeline,
        bundle.target_scalers,
        X_raw,
        bundle.target_names,
    )
    values = preds.to_numpy(dtype=float)
    return {
        "target_names": list(bundle.target_names),
        "target_units": dict(bundle.contract.get("target_units") or {}),
        "predictions": [
            {t: float(p) for t, p in zip(bundle.target_names, row)}
            for row in values.tolist()
        ],
        "n_rows": len(preds),
    }
