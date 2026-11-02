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

from sklearn.base import ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
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


#: Estimators whose training pipeline must NOT standardise features:
#: ``MultinomialNB`` demands non-negative inputs (counts), and a
#: ``StandardScaler`` would push half of any count table negative --
#: ``train_model`` / ``compare_models`` /
#: ``benchmark_feature_families`` check this set when they build the
#: "prep" step, so the registry entry alone is enough to keep such
#: models trainable through the standard flows (PDSH 05.05's own
#: caveat, enforced by the toolkit instead of by a raw sklearn crash).
NO_SCALE_MODELS: frozenset[str] = frozenset({"multinomial_nb"})

#: Default members of the registered ``voting`` / ``stacking`` ensembles
#: -- the same RF + SVM + KNN trio the rar's own Voting.py and
#: Stacking.py stacked.  Both builders accept any registry names, so
#: custom ensembles are one ``build_voting(task, [...])`` call away.
_ENSEMBLE_BASE_MODELS: tuple[str, ...] = ("random_forest", "svm", "knn")


def _rf_cls() -> RandomForestClassifier:
    return RandomForestClassifier(random_state=42, n_jobs=-1)


def _rf_reg() -> RandomForestRegressor:
    return RandomForestRegressor(random_state=42, n_jobs=-1)


def _et_cls() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(random_state=42, n_jobs=-1)


def _et_reg() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(random_state=42, n_jobs=-1)


def _gb_cls() -> GradientBoostingClassifier:
    # Friedman's classic: shallow depth-3 trees fit the residual
    # gradient, one learning_rate-shrunk contribution at a time.
    # n_jobs is not a GradientBoosting* parameter (fit is sequential
    # by construction) -- only random_state is pinned.
    return GradientBoostingClassifier(random_state=42)


def _gb_reg() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=42)


def _pls_reg() -> PLSRegression:
    # PLS1 defaults: two latent components over internally z-scored
    # columns (scale=True, its default) -- mirroring the standardisation
    # the web playground applies before its own NIPALS solve. The
    # regression-only habitat: PLSRegression predicts continuous targets
    # (sklearn retired its PLS discriminant flavour decades ago).
    return PLSRegression(n_components=2)


def _svm_cls() -> SVC:
    return SVC(random_state=42, probability=True)


def _mlp_cls() -> MLPClassifier:
    return MLPClassifier(random_state=42, max_iter=1000)


def _mlp_reg() -> MLPRegressor:
    return MLPRegressor(random_state=42, max_iter=1000)


def _xgb_cls():
    from xgboost import XGBClassifier

    # XGBClassifier is the one registry family that refuses string class
    # labels ("Expected: [0 1], got: ['active' 'inactive']") -- every
    # other family takes any label dtype. The wrapper encodes y at fit
    # time (a sorted LabelEncoder) and decodes predictions back to the
    # original label space, so string targets flow through
    # train_model/evaluate_model/persistence unchanged. Parameter
    # routing is transparent: ``clf__n_estimators`` keeps working
    # because get_params/set_params forward to the wrapped estimator;
    # ClassifierMixin marks the pipeline as a classifier so the ROC-AUC
    # scorer and predict_proba routing recognise it.
    return _IntegerLabelClassifier(
        XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
    )


class _IntegerLabelClassifier(ClassifierMixin):
    """Wrap classifiers that require integer class labels (xgboost).

    Fits a :class:`~sklearn.preprocessing.LabelEncoder` on ``y``, fits
    the wrapped estimator on the encoded labels, and decodes
    ``predict`` output back to the original label space. Hyperparameter
    grids address the wrapped estimator directly (``clf__n_estimators``
    keeps working) because ``get_params`` / ``set_params`` forward
    transparently::

        >>> from pharmaml.ml.models import _IntegerLabelClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> import pandas as pd
        >>> w = _IntegerLabelClassifier(LogisticRegression())
        >>> X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> w.fit(X, ["lo", "lo", "lo", "hi", "hi", "hi"]).classes_.tolist()
        ['hi', 'lo']
        >>> w.predict(X).tolist()
        ['lo', 'lo', 'lo', 'hi', 'hi', 'hi']
        >>> w.get_params()["penalty"]
        'l2'
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self._encoder: Any = None
        self._inner: Any = None
        self.classes_: Any = None

    # -- sklearn parameter plumbing (transparent forwarding) ----------
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"estimator": self.estimator}
        if deep and hasattr(self.estimator, "get_params"):
            params.update(self.estimator.get_params(deep=True))
        return params

    def set_params(self, **params: Any) -> _IntegerLabelClassifier:
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        if params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**params)
        return self

    # -- the ClassifierMixin surface ---------------------------------
    def fit(self, X: Any, y: Any) -> _IntegerLabelClassifier:
        from sklearn.base import clone
        from sklearn.preprocessing import LabelEncoder

        self._encoder = LabelEncoder().fit(y)
        self._inner = clone(self.estimator)
        self._inner.fit(X, self._encoder.transform(y))
        # decoded label space, sorted -- predict_proba columns follow it
        self.classes_ = self._encoder.classes_
        return self

    def predict(self, X: Any) -> Any:
        encoded = self._inner.predict(X).ravel()
        return self._encoder.inverse_transform(encoded)

    def predict_proba(self, X: Any) -> Any:
        return self._inner.predict_proba(X)

    def __sklearn_is_fitted__(self) -> bool:
        return self._inner is not None and hasattr(self._inner, "classes_")


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


def _np_dnn_cls():
    from pharmaml.ml.numpy_neural import NumpyMLPClassifier

    return NumpyMLPClassifier(random_state=42)


def _np_dnn_reg():
    from pharmaml.ml.numpy_neural import NumpyMLPRegressor

    return NumpyMLPRegressor(random_state=42)


def _gp_reg() -> GaussianProcessRegressor:
    # The classic QSAR surrogate kernel: amplitude-scaled RBF smoothness
    # plus white observation noise.  The marginal-likelihood gradient
    # ascent tunes length_scale / amplitude / noise_level at fit time
    # (ONE deterministic optimiser run from these starting points --
    # n_restarts_optimizer=0), and normalize_y=True recentres/rescales
    # the target so the kernel hyperparameters live on a sane scale for
    # raw pIC50-style columns.  Predictions come with their own sigma
    # via predict(..., return_std=True) -- surfaced by
    # pharmaml.ml.predict_with_uncertainty.
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=42, n_restarts_optimizer=0
    )


def _gp_cls() -> GaussianProcessClassifier:
    # The classification half of the GP family: a latent GP squashed
    # through the logistic likelihood, posterior approximated by
    # Laplace's method (Newton mode-finding, exactly GPML algorithm
    # 3.1).  Binary labels use the single latent; multiclass falls back
    # to sklearn's internal one-vs-rest ensemble of binary Laplace GPs.
    # No white-noise term: the logistic likelihood already absorbs the
    # label noise a WhiteKernel would model on the regression side.
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    return GaussianProcessClassifier(
        kernel=kernel, random_state=42, n_restarts_optimizer=0
    )


def _hsvr():
    from pharmaml.ml.hierarchical import HierarchicalSVR

    return HierarchicalSVR(random_state=42)


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
    "sgd_classifier": ModelSpec(
        name="sgd_classifier",
        task="classification",
        factory=lambda: SGDClassifier(max_iter=1000, random_state=42),
        param_grid={
            "loss": ["hinge", "log_loss", "modified_huber"],
            "penalty": ["l2", "l1", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.01],
        },
        notes=(
            "Linear SGD classifier (xxx.ipynb cell 26; loss='log_loss' "
            "and 'modified_huber' enable predict_proba, plain 'hinge' "
            "does not -- the leaderboard's balanced-accuracy fallback "
            "and the voting guard both know.  The notebook's "
            "early_stopping=True is deliberately NOT the factory default: "
            "it reserves a validation split that raises on screens with "
            "fewer than 2*n_classes rows; enable it via manual "
            "hyperparameters when your dataset is large enough)"
        ),
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
    "pls": ModelSpec(
        name="pls",
        task="regression",
        factory=lambda: _pls_reg(),
        param_grid={"n_components": [1, 2, 3, 5, 8]},
        notes=(
            "partial least squares regression (projection to latent "
            "structures, NIPALS): latent directions chosen to maximise "
            "X-y covariance rather than X variance alone -- the QSAR "
            "staple for wide, collinear descriptor blocks where OLS "
            "strains and observations are scarce"
        ),
    ),
    "gaussian_process": ModelSpec(
        name="gaussian_process",
        task="regression",
        factory=lambda: _gp_reg(),
        param_grid={
            "kernel": [
                ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
                ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
                + WhiteKernel(noise_level=0.1),
                ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=0.2)
                + WhiteKernel(noise_level=0.1),
            ]
        },
        notes=(
            "Gaussian process regression (kriging): kernel hyperparameters "
            "tuned by marginal likelihood at fit time, every prediction "
            "carrying its own sigma (predict_with_uncertainty) -- the "
            "surrogate standard for scarce observations and active learning"
        ),
    ),
    "gaussian_process_classifier": ModelSpec(
        name="gaussian_process_classifier",
        task="classification",
        factory=lambda: _gp_cls(),
        param_grid={
            "kernel": [
                ConstantKernel(1.0) * RBF(length_scale=1.0),
                ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
                ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=0.2),
            ]
        },
        notes=(
            "Gaussian process classification (Laplace approximation): a "
            "latent GP squashed through the logistic likelihood, posterior "
            "mode found by Newton iteration -- calibrated probabilities "
            "with honest uncertainty away from the data, the active-"
            "learning classifier that knows what it does not know"
        ),
    ),
    "knn": ModelSpec(
        name="knn",
        task="both",
        factory=None,  # per-task dispatch via _TASK_FACTORIES in build_model
        param_grid={"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
        notes="k-nearest neighbours (upgrades KNN.py)",
    ),
    # -- naive Bayes (PDSH 05.05): the probabilistic baselines the
    # handbook trains first -- fast, closed-form, and embarrassingly
    # strong on wide sparse tables (exactly what fingerprint blocks
    # are).  multinomial_nb bypasses the StandardScaler pipeline
    # (counts must stay non-negative -- see _NO_SCALE_MODELS).
    "gaussian_nb": ModelSpec(
        name="gaussian_nb",
        task="classification",
        factory=lambda: GaussianNB(),
        param_grid={"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        notes=(
            "Gaussian naive Bayes (PDSH 05.05) -- per-class Gaussian "
            "marginals; the fast probabilistic baseline"
        ),
    ),
    "multinomial_nb": ModelSpec(
        name="multinomial_nb",
        task="classification",
        factory=lambda: MultinomialNB(),
        param_grid={"alpha": [0.1, 0.5, 1.0, 2.0]},
        notes=(
            "multinomial naive Bayes (PDSH 05.05) -- count/non-negative "
            "features (fingerprint counts, MACCS keys); trained WITHOUT "
            "feature scaling so counts stay counts"
        ),
    ),
    "bernoulli_nb": ModelSpec(
        name="bernoulli_nb",
        task="classification",
        factory=lambda: BernoulliNB(),
        param_grid={"alpha": [0.1, 0.5, 1.0, 2.0], "binarize": [0.0, 0.5, 1.0]},
        notes=(
            "Bernoulli naive Bayes (PDSH 05.05) -- binary/indicator "
            "features (presence bits); threshold via binarize"
        ),
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
    "extra_trees": ModelSpec(
        name="extra_trees",
        task="both",
        factory=None,
        param_grid={
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        notes=(
            "extremely randomized trees — random thresholds on the full "
            "sample (bootstrap=False): the extra variance reduction when "
            "bagging alone plateaus"
        ),
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
    "gradient_boosting": ModelSpec(
        name="gradient_boosting",
        task="both",
        factory=None,
        param_grid={
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3, 4],
        },
        notes=(
            "Friedman gradient boosting (the sklearn classic behind "
            "XGboost.py / LightGBM.py): shallow depth-3 trees fit the "
            "residual gradient, learning-rate shrunk"
        ),
    ),
    "ann": ModelSpec(
        name="ann",
        task="both",
        factory=None,
        param_grid={
            "hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
            "alpha": [1e-4, 1e-3, 1e-2],
        },
        notes=(
            "MLP neural net (upgrades ANN.py / ANN class.py); batch "
            "normalisation: use the dnn/dnn_residual estimators "
            "(batchnorm=True) — sklearn's MLP has no BN layers"
        ),
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
        notes=(
            "deep feed-forward network (PyTorch, batchnorm=True available) — "
            "needs `pip install 'pharmaml[deep]'`"
        ),
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
        notes=(
            "residual deep network (PyTorch, batchnorm=True available) — "
            "needs `pip install 'pharmaml[deep]'`"
        ),
    ),
    "dnn_numpy": ModelSpec(
        name="dnn_numpy",
        task="both",
        factory=None,
        param_grid={
            "hidden_layers": [(64, 32), (128, 64, 32)],
            "optimizer": ["adam", "adamw"],
            "learning_rate": [1e-3, 3e-4],
        },
        notes=(
            "deep feed-forward network on the toolkit's own autograd "
            "engine — numpy only (no torch), optimizer selectable: sgd, "
            "adagrad, rmsprop, adam (default), adamw"
        ),
    ),
    "hsvr": ModelSpec(
        name="hsvr",
        task="regression",
        factory=lambda: _hsvr(),
        param_grid={
            "n_clusters": [2, 3, 4],
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto"],
        },
        notes=(
            "hierarchical SVR: k-means local specialists blended with a "
            "global SVR (regression only)"
        ),
    ),
    "voting": ModelSpec(
        name="voting",
        task="both",
        factory=None,
        param_grid={
            # only knobs BOTH task estimators accept (VotingRegressor
            # has no 'voting' rule -- it always averages); the voting
            # rule itself lives on the playground's ensemble card and
            # build_voting's ``voting=`` argument.
            "weights": [(1, 1, 1), (2, 1, 1), (1, 2, 1)],
        },
        notes=(
            "voting ensemble of RF + SVM + KNN (upgrades Voting.py / "
            "Voting reg.py; members configurable via build_voting)"
        ),
    ),
    "stacking": ModelSpec(
        name="stacking",
        task="both",
        factory=None,
        param_grid={
            "passthrough": [False, True],
            "cv": [3, 5],
        },
        notes=(
            "stacked ensemble: RF + SVM + KNN level-0, logistic/ridge "
            "meta-learner level-1 (upgrades Stacking.py / Stacking reg.py; "
            "members configurable via build_stacking)"
        ),
    ),
}


#: Friendly short names accepted anywhere a registry key is expected
#: (``--model rf`` on the CLI, ``train_model("logistic", ...)`` ...).
MODEL_ALIASES: dict[str, str] = {
    "logistic": "logistic_regression",
    "logreg": "logistic_regression",
    "rf": "random_forest",
    "randomforest": "random_forest",
    "et": "extra_trees",
    "extratrees": "extra_trees",
    "ert": "extra_trees",
    "extremely_randomized_trees": "extra_trees",
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
    "voting": "voting",
    "vote": "voting",
    "stacking": "stacking",
    "stack": "stacking",
    "neural_net": "ann",
    "ada": "adaboost",
    "adaboost": "adaboost",
    "gb": "gradient_boosting",
    "gradientboosting": "gradient_boosting",
    "friedman_gbm": "gradient_boosting",
    "sk_gbm": "gradient_boosting",
    "linreg": "linear_regression",
    "ols": "linear_regression",
    "huber": "huber_regression",
    "huberreg": "huber_regression",
    "ridge": "ridge",
    "ridge_cls": "ridge_classifier",
    "sgd": "sgd_classifier",
    "sgdclassifier": "sgd_classifier",
    "lasso": "lasso",
    "enet": "elastic_net",
    "elasticnet": "elastic_net",
    "dnn": "dnn",
    "deep_mlp": "dnn",
    "dnn_residual": "dnn_residual",
    "resnet": "dnn_residual",
    "residual_dnn": "dnn_residual",
    "dnn_numpy": "dnn_numpy",
    "numpy_dnn": "dnn_numpy",
    "autograd_mlp": "dnn_numpy",
    "np_dnn": "dnn_numpy",
    "hsvr": "hsvr",
    "hierarchical_svr": "hsvr",
    "hierarchicalsvr": "hsvr",
    "h_svr": "hsvr",
    "pls": "pls",
    "pls_regression": "pls",
    "plsr": "pls",
    "partial_least_squares": "pls",
    "gp": "gaussian_process",
    "gpr": "gaussian_process",
    "gaussianprocess": "gaussian_process",
    "kriging": "gaussian_process",
    "gpc": "gaussian_process_classifier",
    "gp_classifier": "gaussian_process_classifier",
    "gp_classification": "gaussian_process_classifier",
    "gaussianprocessclassifier": "gaussian_process_classifier",
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
    "extra_trees": {
        "classification": _et_cls,
        "regression": _et_reg,
    },
    "svm": {
        "classification": _svm_cls,
        "regression": lambda: SVR(),
    },
    "adaboost": {
        "classification": lambda: AdaBoostClassifier(random_state=42),
        "regression": lambda: AdaBoostRegressor(random_state=42),
    },
    "gradient_boosting": {
        "classification": _gb_cls,
        "regression": _gb_reg,
    },
    "ann": {
        "classification": _mlp_cls,
        "regression": _mlp_reg,
    },
    "voting": {
        "classification": lambda: build_voting(
            "classification", list(_ENSEMBLE_BASE_MODELS), voting="soft"
        ),
        "regression": lambda: build_voting(
            "regression", list(_ENSEMBLE_BASE_MODELS)
        ),
    },
    "stacking": {
        "classification": lambda: build_stacking(
            "classification", list(_ENSEMBLE_BASE_MODELS)
        ),
        "regression": lambda: build_stacking(
            "regression", list(_ENSEMBLE_BASE_MODELS)
        ),
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
    "dnn_numpy": {
        "classification": _np_dnn_cls,
        "regression": _np_dnn_reg,
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
