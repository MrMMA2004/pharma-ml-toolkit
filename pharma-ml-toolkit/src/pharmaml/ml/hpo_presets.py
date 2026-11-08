"""The hyperparameter recommendations layer (Part-3 §18.2/§18.3).

The search *engines* already exist -- :mod:`~pharmaml.ml.hpo` carries the
TPE / successive-halving / Hyperband math and :mod:`~pharmaml.ml.tuner`
the keras-tuner-style ``HyperParameters`` container plus the searchers --
and the registry grid (``MODEL_REGISTRY[token].param_grid``) is
deliberately tiny: three points per slot, sized for a GridSearchCV
sweep, not for optimization.  §18 asks for the layer that sits on top:
comprehensive, *resource-aware* search ranges with the advice that makes
them usable.  This module is that layer.

* **The tables (§18.3)** -- hand-authored ``ParamAdvice`` rows for the
  directive's eight families (logistic regression, random forest,
  SVM/SVR, XGBoost, LightGBM, CatBoost, k-NN, and the two neural
  families the toolkit actually ships: the sklearn ``ann`` and the
  PyTorch ``dnn``), each parameter carrying its range, its log/linear
  sampling prior, an essential/key/extended priority, a one-line
  purpose, and its dependency structure.  Every other grid-bearing
  registry family gets the honest fallback: advice derived from its own
  declared ``param_grid`` (the grid was authored from the upstream
  documentation in the first place).
* **The presets (§18.2)** -- ``conservative`` (essential parameters
  only, middle-half ranges), ``balanced`` (essential + key parameters,
  full ranges, trimmed categorical lists), ``exploratory`` (everything,
  including the combinations that need validation).  The narrowing is
  deterministic and snapped to the parameter's own step, so a preset is
  reproducible from the table.
* **The adaptation (§18.2)** -- ``recommend_search`` adapts the chosen
  preset to the dataset (size, dimensionality, class imbalance) and the
  compute budget, and *every* adaptation leaves a human-readable note:
  small datasets cap estimator budgets, imbalance keeps ``balanced``
  class weighting searchable, low resources cap epochs and drop the
  batch sizes that would thrash, high dimensionality re-orders the
  ``max_features`` prior.
* **The validation (§18.2)** -- ``validate_space`` / ``validate_values``
  enforce the compatibility rules the directive calls out: the
  solver-penalty matrix (``l1`` needs ``saga``/``liblinear``,
  ``elasticnet`` only ``saga``), the kernel-conditioned parameters
  (``degree`` only matters for polynomial kernels, ``epsilon`` is
  SVR-only, ``class_weight`` is classification-only), and the plain
  sanity of every range.  The bridge re-checks each drawn trial and
  fails it loudly instead of silently training an invalid combination.
* **The bridge** -- ``hypermodel_from_advice`` compiles a recommendation
  into a genuine :data:`~pharmaml.ml.tuner.HyperModel`, so the presets
  plug straight into ``RandomSearch`` / ``TPESearch`` / ``Hyperband``
  with no copy-paste of ranges.
* **The interchange (§18.5)** -- ``recommendation_to_json`` /
  ``recommendation_from_json`` round-trip a recommendation losslessly
  (schema-checked on load), and ``estimate_evaluations`` answers the
  "how many evaluations is this budget" question before execution.

The honest scope, stated once: parameters the underlying estimator does
not accept are never advised -- XGBoost's imbalance knob is
``scale_pos_weight`` (not searched here, noted when imbalance is
reported), CatBoost's is ``auto_class_weights`` (same), the PyTorch
``dnn`` family fixes its optimizer internally (AdamW; ``weight_decay``
IS searched as its regularization surrogate), and the ``ann`` family
inherits sklearn's ``'auto'`` batch semantics.  §18.3's "adapt to the
actual implementation" is the rule this module follows.

Examples
--------
>>> from pharmaml.ml.hpo_presets import (
...     estimate_evaluations, preset_styles, recommend_search,
...     validate_space,
... )
>>> preset_styles()
('conservative', 'balanced', 'exploratory')
>>> rec = recommend_search("random_forest", "classification", preset="conservative")
>>> [p.name for p in rec.params]
['n_estimators', 'max_depth']
>>> (rec.params[0].high <= 800, rec.params[0].purpose)
(True, 'number of trees in the forest -- more is smoother but costlier')

Adaptation to a small, imbalanced dataset leaves visible notes:

>>> small = recommend_search(
...     "random_forest", "classification", preset="exploratory",
...     n_samples=180, class_counts=[150, 30],
... )
>>> small.params[0].high            # the n_estimators budget, capped
300.0
>>> any("imbalance" in n for n in small.notes)
True

Compatibility validation flags the combinations §18.2 worries about:

>>> issues = validate_space(
...     "logistic_regression", "classification", preset="exploratory"
... )
>>> any("lbfgs" in i.message for i in issues if i.parameter == "penalty")
True

The budget estimate before running anything:

>>> est = estimate_evaluations(rec, strategy="grid")
>>> (est.count, est.strategy)
(8, 'grid')

>>> from pharmaml.ml.hpo_presets import recommendation_from_json, recommendation_to_json
>>> text = recommendation_to_json(small)
>>> roundtrip = recommendation_from_json(text)
>>> (roundtrip.model, roundtrip.task, roundtrip.preset)
('random_forest', 'classification', 'exploratory')
>>> [p.name for p in roundtrip.params] == [p.name for p in small.params]
True
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.logger import get_logger

__all__ = [
    "EvaluationEstimate",
    "ModelAdvice",
    "ParamAdvice",
    "SearchRecommendation",
    "SpaceIssue",
    "advice_families",
    "estimate_evaluations",
    "hypermodel_from_advice",
    "preset_styles",
    "recommend_search",
    "recommendation_from_json",
    "recommendation_to_json",
    "validate_space",
    "validate_values",
]

_LOG = get_logger(__name__)

#: The three §18.2 search presets, in increasing cost order.
PresetStyle = Literal["conservative", "balanced", "exploratory"]
PRESET_STYLES: tuple[str, ...] = ("conservative", "balanced", "exploratory")

#: The three resource tiers the adaptation layer understands.
ResourceLevel = Literal["low", "medium", "high"]
RESOURCE_LEVELS: tuple[str, ...] = ("low", "medium", "high")

#: Parameter priorities: essential params ride every preset, key params
#: join from ``balanced`` up, extended params only ``exploratory``.
PRIORITIES: tuple[str, ...] = ("essential", "key", "extended", "fixed")

#: (family-token, task) pairs with hand-authored §18.3 tables.  The
#: ``svm`` token covers both SVC and SVR (task-aware rows); ``ann`` is
#: the sklearn MLP, ``dnn`` the PyTorch DeepMLP (``deep`` aliases it).
_TABLE_TOKENS: frozenset[str] = frozenset(
    {
        "logistic_regression",
        "random_forest",
        "svm",
        "xgboost",
        "lightgbm",
        "catboost",
        "knn",
        "ann",
        "dnn",
        "hsvr",
    }
)

#: The estimator-budget slots the adaptation layer caps.
_BUDGET_SLOTS: frozenset[str] = frozenset(
    {"n_estimators", "iterations", "max_iter", "epochs"}
)


# ---------------------------------------------------------------------------
# The advice records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamAdvice:
    """One searchable hyperparameter: its range, prior, role, and deps.

    ``kind`` is one of ``'int' | 'float' | 'choice' | 'boolean' |
    'fixed'``; numeric kinds carry ``low``/``high``/``step`` (and
    ``log=True`` for the log-domain prior on scale parameters), choice
    kinds carry ``choices`` (``None`` allowed -- ``class_weight=None``
    is a real choice), and fixed kinds carry ``value`` (documented,
    drawn verbatim, never searched).  ``priority`` follows
    :data:`PRIORITIES`; ``depends_on`` names the parameter or condition
    this one is conditional on; ``purpose`` is the one-line §18.2
    explanation.
    """

    name: str
    kind: str
    priority: str
    purpose: str
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: tuple[Any, ...] = ()
    value: Any = None
    depends_on: str | None = None

    @property
    def searchable(self) -> bool:
        """Fixed parameters are documentation, not search."""
        return self.kind != "fixed"

    def narrowed(self, style: str) -> ParamAdvice:
        """This parameter under a §18.2 preset (deterministic).

        Conservative keeps the middle half of a numeric range (the
        middle half *in the log domain* for log parameters -- one
        narrower decade band) and the first two categorical values;
        balanced keeps full numeric ranges but trims categorical lists
        to their first two-thirds; exploratory is the identity.  Integer
        and stepped bounds are snapped to the parameter's own step.
        """
        if style == "exploratory" or self.kind in {"fixed", "boolean"}:
            return self
        if self.kind in {"int", "float"}:
            low, high = float(self.low), float(self.high)
            if style == "balanced":
                return self
            if self.log:
                ratio = high / low
                lo, hi = low * ratio**0.25, low * ratio**0.75
            else:
                width = high - low
                lo, hi = low + 0.25 * width, low + 0.75 * width
            if self.kind == "int":
                lo_i, hi_i = math.ceil(lo), math.floor(hi)
                if self.step:
                    step = int(self.step)
                    lo_i = int(low) + math.ceil((lo_i - low) / step) * step
                    hi_i = int(low) + math.floor((hi_i - low) / step) * step
                if hi_i <= lo_i:
                    return self
                return replace(self, low=float(lo_i), high=float(hi_i))
            if self.step:
                step = float(self.step)
                lo = low + math.ceil((lo - low) / step) * step
                hi = low + math.floor((hi - low) / step) * step
                if hi <= lo:
                    return self
            return replace(self, low=lo, high=hi)
        if self.kind == "choice":
            if style == "conservative":
                keep = self.choices[: min(2, len(self.choices))]
            else:
                keep = self.choices[: max(2, math.ceil(len(self.choices) * 2 / 3))]
            return replace(self, choices=tuple(keep) or self.choices)
        return self


@dataclass(frozen=True)
class ModelAdvice:
    """The full §18.3 table for one (family, task) pair."""

    model: str
    task: str
    params: tuple[ParamAdvice, ...]
    notes: tuple[str, ...] = ()

    def by_name(self, name: str) -> ParamAdvice | None:
        for param in self.params:
            if param.name == name:
                return param
        return None


@dataclass(frozen=True)
class SearchRecommendation:
    """A preset applied and adapted to a concrete dataset/budget.

    ``notes`` carries every adaptation that fired (§18.2's "explain"
    requirement): each note is human-readable and deterministic.
    """

    model: str
    task: str
    preset: str
    params: tuple[ParamAdvice, ...]
    notes: tuple[str, ...] = ()
    resources: str = "medium"
    n_samples: int | None = None
    n_features: int | None = None

    @property
    def searchable(self) -> tuple[ParamAdvice, ...]:
        return tuple(p for p in self.params if p.searchable)

    def summary(self) -> str:
        """The one-paragraph budget review (§18.5's pre-execution view)."""
        lines = [
            (
                f"{self.model} ({self.task}) -- {self.preset} preset, "
                f"{len(self.searchable)} searchable parameters:"
            )
        ]
        for p in self.searchable:
            if p.kind == "choice":
                span = f"choices {list(p.choices)}"
            elif p.kind == "boolean":
                span = "true/false"
            elif p.log:
                span = f"log [{p.low:g}, {p.high:g}]"
            else:
                span = f"[{p.low:g}, {p.high:g}]" + (
                    f" step {p.step:g}" if p.step else ""
                )
            lines.append(f"  {p.name} ({p.priority}): {span}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SpaceIssue:
    """One compatibility or sanity finding (§18.2 validation).

    ``severity`` is ``'error'`` (the configuration cannot run) or
    ``'warning'`` (it runs, but a drawn trial may fail or a parameter
    has no effect).
    """

    severity: str
    parameter: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


@dataclass(frozen=True)
class EvaluationEstimate:
    """The pre-execution budget answer (§18.5)."""

    count: int
    strategy: str
    assumptions: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# The §18.3 tables
# ---------------------------------------------------------------------------


def _int(
    name: str,
    low: int,
    high: int,
    priority: str,
    purpose: str,
    step: int | None = None,
    depends_on: str | None = None,
) -> ParamAdvice:
    return ParamAdvice(
        name=name,
        kind="int",
        priority=priority,
        purpose=purpose,
        low=float(low),
        high=float(high),
        step=float(step) if step else None,
        depends_on=depends_on,
    )


def _float(
    name: str,
    low: float,
    high: float,
    priority: str,
    purpose: str,
    log: bool = False,
    step: float | None = None,
    depends_on: str | None = None,
) -> ParamAdvice:
    return ParamAdvice(
        name=name,
        kind="float",
        priority=priority,
        purpose=purpose,
        low=float(low),
        high=float(high),
        step=float(step) if step else None,
        log=log,
        depends_on=depends_on,
    )


def _choice(
    name: str,
    choices: Sequence[Any],
    priority: str,
    purpose: str,
    depends_on: str | None = None,
) -> ParamAdvice:
    return ParamAdvice(
        name=name,
        kind="choice",
        priority=priority,
        purpose=purpose,
        choices=tuple(choices),
        depends_on=depends_on,
    )


def _bool(name: str, priority: str, purpose: str) -> ParamAdvice:
    return ParamAdvice(name=name, kind="boolean", priority=priority, purpose=purpose)


def _fixed(name: str, value: Any, purpose: str) -> ParamAdvice:
    return ParamAdvice(
        name=name, kind="fixed", priority="fixed", purpose=purpose, value=value
    )


def _logistic_table(task: str) -> tuple[ParamAdvice, ...]:
    rows = [
        _float(
            "C",
            1e-4,
            1e2,
            "essential",
            "inverse regularization strength -- larger fits harder, "
            "smaller regularizes more",
            log=True,
        ),
        _choice(
            "penalty",
            ("l2", "l1"),
            "essential",
            "the norm penalizing coefficients -- l1 sparsifies, l2 shrinks",
        ),
        _choice(
            "solver",
            ("saga", "liblinear", "lbfgs"),
            "essential",
            "the optimizer -- saga handles every penalty, liblinear is "
            "the small-data workhorse, lbfgs is fastest for l2-only "
            "(listed in compatibility order so preset trims stay valid)",
        ),
    ]
    if task == "classification":
        rows.append(
            _choice(
                "class_weight",
                (None, "balanced"),
                "key",
                "reweights classes -- 'balanced' counters minority imbalance",
            )
        )
    rows.append(
        _int(
            "max_iter",
            500,
            3000,
            "extended",
            "solver iterations -- raise when convergence warnings persist",
            step=250,
        )
    )
    return tuple(rows)


def _random_forest_table(task: str) -> tuple[ParamAdvice, ...]:
    rows = [
        _int(
            "n_estimators",
            100,
            800,
            "essential",
            "number of trees in the forest -- more is smoother but costlier",
            step=100,
        ),
        _choice(
            "max_depth",
            (None, 5, 10, 20, 40),
            "essential",
            "tree depth cap -- None grows fully (overfit risk on small data)",
        ),
        _choice(
            "max_features",
            ("sqrt", "log2", 0.3, 0.5, 0.8, 1.0),
            "key",
            "features considered per split -- sqrt/log2 decorrelate trees",
        ),
        _int(
            "min_samples_split",
            2,
            20,
            "key",
            "minimum samples to split a node -- higher prunes harder",
        ),
        _int(
            "min_samples_leaf",
            1,
            10,
            "extended",
            "minimum samples per leaf -- higher smooths the fit",
        ),
    ]
    if task == "classification":
        rows.append(
            _choice(
                "class_weight",
                (None, "balanced"),
                "key",
                "reweights classes -- 'balanced' counters minority imbalance",
            )
        )
    return tuple(rows)


def _svm_table(task: str) -> tuple[ParamAdvice, ...]:
    rows = [
        _float(
            "C",
            1e-3,
            1e3,
            "essential",
            "margin slack budget -- larger fits harder, smaller regularizes more",
            log=True,
        ),
        _choice(
            "kernel",
            ("rbf", "linear", "poly"),
            "essential",
            "the similarity kernel -- rbf is the default, linear for high "
            "dimensions, poly for explicit interactions",
        ),
        _choice(
            "gamma",
            ("scale", "auto", 0.001, 0.01, 0.1),
            "key",
            "rbf/poly bandwidth -- 'scale'/'auto' adapt to the feature "
            "count; the floats cover §18.3's log-uniform 1e-5..1 spirit "
            "discretely (customize the list for a finer prior)",
            depends_on="kernel != linear",
        ),
    ]
    if task == "regression":
        rows.append(
            _float(
                "epsilon",
                0.01,
                0.5,
                "key",
                "the SVR insensitive-tube width -- wider tubes tolerate more error",
            )
        )
        rows.append(
            _int(
                "degree",
                2,
                5,
                "extended",
                "polynomial degree (poly kernel only)",
                depends_on="kernel == poly",
            )
        )
    else:
        rows.append(
            _choice(
                "class_weight",
                (None, "balanced"),
                "extended",
                "reweights classes -- 'balanced' counters minority imbalance",
            )
        )
    return tuple(rows)


def _xgboost_table() -> tuple[ParamAdvice, ...]:
    return (
        _int(
            "n_estimators",
            100,
            1000,
            "essential",
            "boosting rounds -- more fits harder (early stopping is the "
            "honest cap)",
            step=100,
        ),
        _int(
            "max_depth",
            2,
            10,
            "essential",
            "per-tree depth -- the primary complexity knob",
        ),
        _float(
            "learning_rate",
            0.01,
            0.3,
            "essential",
            "the shrinkage step -- smaller needs more rounds",
            log=True,
        ),
        _float(
            "subsample",
            0.6,
            1.0,
            "key",
            "row subsampling per tree -- below 1.0 decorrelates rounds",
        ),
        _float(
            "colsample_bytree",
            0.6,
            1.0,
            "key",
            "column subsampling per tree",
        ),
        _int(
            "min_child_weight",
            1,
            10,
            "extended",
            "minimum leaf weight -- higher prunes harder",
        ),
        _float(
            "reg_alpha",
            1e-8,
            10.0,
            "extended",
            "L1 penalty on leaf weights",
            log=True,
        ),
        _float(
            "reg_lambda",
            0.1,
            100.0,
            "extended",
            "L2 penalty on leaf weights",
            log=True,
        ),
    )


def _lightgbm_table(task: str) -> tuple[ParamAdvice, ...]:
    rows = [
        _int(
            "n_estimators",
            100,
            1000,
            "essential",
            "boosting rounds",
            step=100,
        ),
        _int(
            "num_leaves",
            7,
            127,
            "essential",
            "leaves per tree -- the LightGBM complexity knob (keep well "
            "below 2^max_depth)",
        ),
        _int(
            "max_depth",
            -1,
            12,
            "key",
            "depth cap (-1 = unlimited, defers to num_leaves)",
        ),
        _float(
            "learning_rate",
            0.01,
            0.2,
            "essential",
            "the shrinkage step",
            log=True,
        ),
        _int(
            "min_child_samples",
            5,
            50,
            "extended",
            "minimum samples per leaf",
        ),
        _float(
            "subsample",
            0.6,
            1.0,
            "extended",
            "row subsampling (bagging) fraction",
        ),
        _float(
            "colsample_bytree",
            0.6,
            1.0,
            "extended",
            "column subsampling fraction",
        ),
    ]
    if task == "classification":
        rows.append(
            _choice(
                "class_weight",
                (None, "balanced"),
                "key",
                "reweights classes -- 'balanced' counters minority imbalance",
            )
        )
    return tuple(rows)


def _catboost_table(task: str) -> tuple[ParamAdvice, ...]:
    if task == "classification":
        loss = ("Logloss", "CrossEntropy")
        loss_purpose = (
            "the classification objective -- Logloss for binary, "
            "CrossEntropy for multiclass probability"
        )
    else:
        loss = ("RMSE", "MAE")
        loss_purpose = "the regression objective"
    return (
        _int(
            "iterations",
            200,
            1500,
            "essential",
            "boosting rounds",
            step=100,
        ),
        _int(
            "depth",
            4,
            10,
            "essential",
            "per-tree depth (symmetric trees)",
        ),
        _float(
            "learning_rate",
            0.01,
            0.3,
            "essential",
            "the shrinkage step",
            log=True,
        ),
        _choice(
            "loss_function",
            loss,
            "key",
            loss_purpose,
            depends_on="task",
        ),
        _float(
            "l2_leaf_reg",
            1.0,
            20.0,
            "key",
            "L2 penalty on leaf weights",
        ),
        _float(
            "random_strength",
            0.0,
            2.0,
            "extended",
            "score randomization -- a CatBoost-specific regularizer",
        ),
    )


def _knn_table() -> tuple[ParamAdvice, ...]:
    return (
        _int(
            "n_neighbors",
            3,
            50,
            "essential",
            "neighbors polled -- small k is noisy, large k blurs",
        ),
        _choice(
            "weights",
            ("uniform", "distance"),
            "key",
            "vote weighting -- 'distance' favors closer neighbors",
        ),
        _choice(
            "metric",
            ("minkowski", "euclidean", "manhattan"),
            "key",
            "the distance definition",
        ),
        _int(
            "p",
            1,
            2,
            "extended",
            "the minkowski power (1=manhattan, 2=euclidean)",
            depends_on="metric == minkowski",
        ),
    )


def _ann_table() -> tuple[ParamAdvice, ...]:
    return (
        _int(
            "n_layers",
            1,
            3,
            "essential",
            "hidden layer count",
        ),
        _int(
            "units",
            16,
            128,
            "essential",
            "neurons per hidden layer",
            step=16,
        ),
        _float(
            "alpha",
            1e-6,
            1e-2,
            "key",
            "L2 penalty on the weights",
            log=True,
        ),
        _float(
            "learning_rate_init",
            1e-4,
            1e-2,
            "key",
            "the initial SGD step",
            log=True,
        ),
        _int(
            "max_iter",
            200,
            1000,
            "key",
            "the epoch budget (the resource slot)",
            step=100,
        ),
        _choice(
            "batch_size",
            ("auto", 32, 64, 128),
            "extended",
            "minibatch size -- 'auto' is min(200, n_samples); sklearn's "
            "semantics differ from the deep-learning families",
        ),
    )


def _dnn_table() -> tuple[ParamAdvice, ...]:
    return (
        _int(
            "n_layers",
            1,
            5,
            "essential",
            "hidden layer count",
        ),
        _int(
            "units",
            16,
            128,
            "essential",
            "neurons per hidden layer",
            step=16,
        ),
        _float(
            "learning_rate",
            1e-5,
            1e-2,
            "essential",
            "the optimizer step size",
            log=True,
        ),
        _int(
            "batch_size",
            16,
            128,
            "key",
            "minibatch size -- smaller batches cost more steps per epoch",
        ),
        _int(
            "epochs",
            50,
            300,
            "key",
            "the epoch budget -- early stopping bounds the effective count",
            step=10,
        ),
        _int(
            "patience",
            5,
            30,
            "key",
            "early-stopping patience in epochs",
        ),
        _float(
            "dropout",
            0.0,
            0.5,
            "key",
            "dropout rate between layers",
            step=0.05,
        ),
        _choice(
            "activation",
            ("relu", "gelu", "tanh"),
            "extended",
            "the hidden activation",
        ),
        _float(
            "weight_decay",
            1e-7,
            1e-3,
            "extended",
            "AdamW's decoupled L2 -- the optimizer is fixed (AdamW), its "
            "regularization is not",
            log=True,
        ),
        _bool(
            "batchnorm",
            "extended",
            "batch normalization between layers",
        ),
        _fixed(
            "optimizer",
            "adamw",
            "the DeepMLP family fixes AdamW internally -- its surrogate "
            "(weight_decay) is searched instead",
        ),
    )


def _hsvr_table() -> tuple[ParamAdvice, ...]:
    return (
        _int(
            "n_clusters",
            2,
            5,
            "essential",
            "number of local models the hierarchy fits",
        ),
        _float(
            "C",
            0.1,
            10.0,
            "essential",
            "the per-cluster SVR slack budget",
            log=True,
        ),
        _choice(
            "gamma",
            ("scale", "auto"),
            "essential",
            "the per-cluster SVR bandwidth",
        ),
        _float(
            "alpha",
            0.0,
            1.0,
            "key",
            "local-vs-global blending (0=global only, 1=local only)",
            step=0.1,
        ),
    )


#: The hand-authored tables, keyed (token, task).
_TABLES: dict[tuple[str, str], tuple[ParamAdvice, ...]] = {
    ("logistic_regression", "classification"): _logistic_table("classification"),
}
_TABLES.update(
    {
        (token, task): table(task)
        for token, table in (
            ("random_forest", _random_forest_table),
            ("svm", _svm_table),
            ("xgboost", lambda task: _xgboost_table()),
            ("lightgbm", _lightgbm_table),
            ("catboost", _catboost_table),
            ("knn", lambda task: _knn_table()),
            ("ann", lambda task: _ann_table()),
            ("dnn", lambda task: _dnn_table()),
        )
        for task in ("classification", "regression")
    }
)
_TABLES[("hsvr", "regression")] = _hsvr_table()

#: sklearn's LogisticRegression solver support (the compatibility
#: matrix §18.2 asks to validate before optimization starts).
_SOLVER_PENALTY: dict[str, frozenset[str]] = {
    "l2": frozenset({"lbfgs", "liblinear", "saga"}),
    "l1": frozenset({"liblinear", "saga"}),
    "elasticnet": frozenset({"saga"}),
}


def _registry_fallback(token: str) -> tuple[ParamAdvice, ...]:
    """Advice derived from a registry family's own declared grid.

    (§18.3 "Additional Models": for any other implemented estimator,
    generate model-specific ranges from its documentation -- the grid
    WAS authored from the upstream docs, so deriving the advice from it
    is the honest version of that clause.)
    """
    from pharmaml.ml.models import MODEL_REGISTRY

    grid = MODEL_REGISTRY[token].param_grid
    rows = [
        _choice(
            slot,
            values,
            "essential",
            f"registry-declared values for {token!r} (authored from the "
            "upstream documentation)",
        )
        for slot, values in sorted(grid.items())
    ]
    return tuple(rows)


def _resolve_family(model: str, task: str) -> tuple[str, str]:
    """Resolve a user token/alias to the advice table's (token, task)."""
    if task not in {"regression", "classification"}:
        raise ModelError(
            f"task must be 'regression' or 'classification', got {task!r}"
        )
    token = model.strip().lower()
    if token in {"deep", "deep_mlp", "mlp"}:
        token = "dnn"
    if token in _TABLE_TOKENS:
        if (token, task) not in _TABLES:
            tasks = sorted(t for (_, t) in _TABLES if _ == token)
            raise ModelError(
                f"Model {token!r} supports {tasks[0]!r}, not {task!r}"
            )
        return token, task
    from pharmaml.ml.models import MODEL_REGISTRY, resolve_model_name

    key = resolve_model_name(token)
    spec = MODEL_REGISTRY[key]
    if spec.task != "both" and spec.task != task:
        raise ModelError(f"Model {key!r} supports {spec.task!r}, not {task!r}")
    if not spec.param_grid:
        raise ModelError(f"the model {key!r} declares no param_grid to search")
    return key, task


def advice_families() -> list[str]:
    """The tokens with hand-authored §18.3 tables.

    Examples
    --------
    >>> from pharmaml.ml.hpo_presets import advice_families
    >>> sorted(advice_families())[:3]
    ['ann', 'catboost', 'dnn']
    """
    return sorted(_TABLE_TOKENS)


# ---------------------------------------------------------------------------
# Presets and adaptation
# ---------------------------------------------------------------------------


def preset_styles() -> tuple[str, ...]:
    """The three §18.2 preset styles, in increasing cost order."""
    return PRESET_STYLES


def _apply_preset(
    table: tuple[ParamAdvice, ...], style: str
) -> tuple[ParamAdvice, ...]:
    if style == "exploratory":
        return table
    allowed = {"essential"} if style == "conservative" else {"essential", "key"}
    kept = [p for p in table if p.kind == "fixed" or p.priority in allowed]
    return tuple(p.narrowed(style) for p in kept)


def _cap_int(param: ParamAdvice, cap: int) -> tuple[ParamAdvice, bool]:
    """Clamp an int parameter's high bound to ``cap`` (low is untouched).

    A cap below the declared low leaves the parameter unchanged -- the
    honest refusal (``max_iter``'s floor of 500 is not "capped" by 300).
    """
    low, high = int(param.low), int(param.high)
    if cap < low or cap >= high:
        return param, False
    return replace(param, high=float(cap)), True


def recommend_search(
    model: str,
    task: str,
    *,
    preset: str = "balanced",
    n_samples: int | None = None,
    n_features: int | None = None,
    class_counts: Sequence[int] | None = None,
    resources: str = "medium",
) -> SearchRecommendation:
    """The §18.2 recommendation: preset + dataset/budget adaptation.

    Parameters
    ----------
    model:
        Registry token or alias (``'rf'``, ``'random_forest'``,
        ``'deep'`` for the PyTorch family, ``'ann'`` for sklearn's MLP).
    task:
        ``'regression'`` or ``'classification'``.
    preset:
        One of :data:`PRESET_STYLES` -- conservative / balanced /
        exploratory.
    n_samples, n_features:
        The dataset's shape -- small samples cap estimator budgets,
        high dimensionality re-orders the feature-fraction prior.
    class_counts:
        Per-class example counts -- a >= 4:1 imbalance keeps
        ``balanced`` weighting searchable (or notes the family's honest
        alternative when it has no weight parameter).
    resources:
        ``'low' | 'medium' | 'high'`` -- the compute tier.

    Returns
    -------
    SearchRecommendation
        With every adaptation that fired recorded in ``notes``.

    Raises
    ------
    ModelError
        Unknown model/task, a bad preset or resource tier.

    Examples
    --------
    >>> rec = recommend_search("knn", "regression", preset="exploratory")
    >>> [p.name for p in rec.params]
    ['n_neighbors', 'weights', 'metric', 'p']
    """
    if preset not in PRESET_STYLES:
        raise ModelError(f"preset must be one of {PRESET_STYLES}, got {preset!r}")
    if resources not in RESOURCE_LEVELS:
        raise ModelError(
            f"resources must be one of {RESOURCE_LEVELS}, got {resources!r}"
        )
    token, task = _resolve_family(model, task)
    table = _TABLES.get((token, task)) or _registry_fallback(token)
    # Adaptation runs on the FULL table first; the preset narrows the
    # adapted envelope afterwards (a small dataset caps the budget, the
    # preset then decides how much of the capped range to search).
    params = list(table)
    notes: list[str] = []
    names = {p.name for p in params}

    # -- dataset-size adaptation ------------------------------------
    if n_samples is not None and n_samples > 0:
        if n_samples < 500:
            capped_any = False
            for i, p in enumerate(params):
                if p.kind == "int" and p.name in _BUDGET_SLOTS:
                    params[i], changed = _cap_int(p, 300)
                    capped_any = capped_any or changed
            if "n_neighbors" in names:
                for i, p in enumerate(params):
                    if p.name == "n_neighbors":
                        cap = max(3, min(int(p.high), n_samples // 2))
                        if cap < p.high:
                            params[i] = replace(p, high=float(cap))
                            capped_any = True
                        break
            if capped_any:
                notes.append(
                    f"small dataset (n={n_samples}): estimator budgets "
                    f"capped at 300"
                    + (" and n_neighbors at n/2" if "n_neighbors" in names else "")
                )
        elif n_samples >= 10_000 and resources != "low":
            notes.append(
                f"large dataset (n={n_samples}): full budgets apply -- "
                f"prefer RandomizedSearch/TPESearch over a full grid"
            )

    # -- resource-tier adaptation ------------------------------------
    if resources == "low":
        capped_any = False
        for i, p in enumerate(params):
            if p.kind == "int" and p.name in _BUDGET_SLOTS:
                params[i], changed = _cap_int(p, 300)
                capped_any = capped_any or changed
            if p.name == "epochs":
                params[i], changed = _cap_int(p, 100)
                capped_any = capped_any or changed
            if p.kind == "int" and p.name == "batch_size" and p.low < 32:
                params[i] = replace(p, low=32.0)
                capped_any = True
            if p.kind == "choice" and p.name == "batch_size":
                kept = tuple(c for c in p.choices if c == "auto" or c >= 32)
                if kept and kept != p.choices:
                    params[i] = replace(p, choices=kept)
                    capped_any = True
        if capped_any:
            notes.append(
                "low resources: estimator budgets capped at 300, deep "
                "epochs at 100, batch sizes below 32 dropped"
            )

    # -- dimensionality adaptation ------------------------------------
    if n_features is not None and n_features >= 100:
        hints = []
        if "max_features" in names:
            hints.append(
                "keep max_features at sqrt/log2 (the lists are ordered that way)"
            )
        if "gamma" in names:
            hints.append("prefer gamma='scale' (it adapts to feature count)")
        if "n_neighbors" in names:
            hints.append(
                "distances concentrate in high dimensions -- consider PCA first"
            )
        if hints:
            notes.append(f"high dimensionality (p={n_features}): " + "; ".join(hints))

    # -- class-imbalance adaptation ------------------------------------
    if class_counts:
        counts = [int(c) for c in class_counts if int(c) > 0]
        if len(counts) >= 2:
            ratio = max(counts) / min(counts)
            if ratio >= 4.0:
                if "class_weight" in names:
                    for i, p in enumerate(params):
                        if p.name == "class_weight":
                            choices = p.choices
                            if "balanced" not in choices:
                                choices = tuple(choices) + ("balanced",)
                            # imbalance elevates the knob to essential: it
                            # rides every preset once the data demands it
                            params[i] = replace(
                                p, choices=choices, priority="essential"
                            )
                            break
                    notes.append(
                        f"class imbalance {ratio:.1f}:1 -- 'balanced' "
                        f"weighting kept searchable (elevated to essential)"
                    )
                elif token == "xgboost":
                    notes.append(
                        f"class imbalance {ratio:.1f}:1 -- this family's knob "
                        f"is scale_pos_weight (not searched here); set it to "
                        f"~{ratio:.0f} manually for binary targets"
                    )
                elif token == "catboost":
                    notes.append(
                        f"class imbalance {ratio:.1f}:1 -- this family's knob "
                        f"is auto_class_weights='Balanced' (not searched here)"
                    )
                else:
                    notes.append(
                        f"class imbalance {ratio:.1f}:1 -- this family has no "
                        f"class_weight parameter; prefer stratified CV and "
                        f"imbalance-aware scoring"
                    )
                if preset == "conservative":
                    notes.append(
                        "conservative preset + imbalance: consider 'balanced' "
                        "as the near-default choice"
                    )

    # -- the preset narrows the ADAPTED envelope, last ----------------
    params = list(_apply_preset(tuple(params), preset))

    return SearchRecommendation(
        model=token,
        task=task,
        preset=preset,
        params=tuple(params),
        notes=tuple(notes),
        resources=resources,
        n_samples=n_samples,
        n_features=n_features,
    )


# ---------------------------------------------------------------------------
# Validation (§18.2 "validate parameter compatibility")
# ---------------------------------------------------------------------------


def _table_for(model: str, task: str) -> tuple[ParamAdvice, ...]:
    token, task = _resolve_family(model, task)
    return _TABLES.get((token, task)) or _registry_fallback(token)


def _collect_choices(params: Sequence[ParamAdvice], name: str) -> list[Any]:
    for p in params:
        if p.name == name and p.kind == "choice":
            return list(p.choices)
    return []


def validate_space(
    model: str,
    task: str,
    params: Sequence[ParamAdvice] | None = None,
    *,
    preset: str | None = None,
) -> list[SpaceIssue]:
    """Compatibility-check a search space before optimization starts.

    With ``params=None`` the §18.3 table for ``(model, task)`` is used;
    with ``preset`` given, the preset-narrowed table is checked instead
    (pass exactly one of the two).

    Returns
    -------
    list[SpaceIssue]
        ``severity='error'`` issues mean the space cannot run as
        configured; ``'warning'`` issues mean a drawn trial may fail or
        a parameter may have no effect.

    Examples
    --------
    >>> issues = validate_space("svm", "regression")
    >>> [i.severity for i in issues]
    []
    """
    if params is not None and preset is not None:
        raise ModelError("pass params or preset, not both")
    if params is None:
        if preset is not None:
            params = recommend_search(model, task, preset=preset).params
        else:
            params = _table_for(model, task)
    known = {p.name: p for p in _table_for(model, task)}
    issues: list[SpaceIssue] = []
    seen: set[str] = set()
    for param in params:
        if param.name in seen:
            issues.append(
                SpaceIssue("error", param.name, "duplicate parameter in the space")
            )
        seen.add(param.name)
        if param.name not in known:
            issues.append(
                SpaceIssue(
                    "error",
                    param.name,
                    f"not a supported parameter of {model!r} -- never expose "
                    f"unsupported parameters (§18.3)",
                )
            )
            continue
        if param.kind in {"int", "float"}:
            if param.low is None or param.high is None:
                issues.append(
                    SpaceIssue("error", param.name, "numeric parameter missing bounds")
                )
            elif param.low >= param.high:
                issues.append(
                    SpaceIssue(
                        "error",
                        param.name,
                        f"invalid range [{param.low}, {param.high}]",
                    )
                )
            elif param.log and param.low <= 0:
                issues.append(
                    SpaceIssue(
                        "error",
                        param.name,
                        "log sampling requires positive bounds",
                    )
                )
        if param.kind == "choice" and not param.choices:
            issues.append(
                SpaceIssue("error", param.name, "categorical parameter with no values")
            )
    # -- the compatibility matrices ---------------------------------
    names = seen & known.keys()
    token = model.strip().lower()
    if token == "deep":
        token = "dnn"
    if {"solver", "penalty"} <= names and token in {
        "logistic_regression",
        "elastic_net_logistic",
        "lasso_logistic",
    }:
        solvers = _collect_choices(params, "solver")
        penalties = _collect_choices(params, "penalty")
        for penalty in penalties:
            allowed = _SOLVER_PENALTY.get(str(penalty))
            if allowed is None:
                continue
            bad = [s for s in solvers if str(s) not in allowed]
            if bad:
                issues.append(
                    SpaceIssue(
                        "warning",
                        "penalty",
                        f"penalty={penalty!r} is incompatible with solver "
                        f"{bad} -- trials drawing that pair will fail "
                        f"validation (sklearn's matrix: l1 needs "
                        f"liblinear/saga, elasticnet only saga)",
                    )
                )
    if "degree" in names:
        kernels = _collect_choices(params, "kernel")
        if kernels and "poly" not in {str(k) for k in kernels}:
            issues.append(
                SpaceIssue(
                    "warning",
                    "degree",
                    "degree has no effect unless kernel='poly' is in the "
                    "searched choices",
                )
            )
    if "gamma" in names and "kernel" in names:
        kernels = {str(k) for k in _collect_choices(params, "kernel")}
        if kernels == {"linear"}:
            issues.append(
                SpaceIssue("warning", "gamma", "gamma is ignored by the linear kernel")
            )
    if "p" in names:
        metrics = _collect_choices(params, "metric")
        if metrics and "minkowski" not in {str(m) for m in metrics}:
            issues.append(
                SpaceIssue(
                    "warning",
                    "p",
                    "p has no effect unless metric='minkowski' is searched",
                )
            )
    if "epsilon" in seen and task != "regression":
        issues.append(
            SpaceIssue("error", "epsilon", "epsilon is an SVR-only parameter")
        )
    if "class_weight" in seen and task != "classification":
        issues.append(
            SpaceIssue("error", "class_weight", "class_weight is classification-only")
        )
    if "loss_function" in names:
        losses = {str(v) for v in _collect_choices(params, "loss_function")}
        cls_losses = {"Logloss", "CrossEntropy"}
        if task == "classification" and not losses <= cls_losses:
            issues.append(
                SpaceIssue(
                    "error",
                    "loss_function",
                    f"classification loss_function must be one of "
                    f"{sorted(cls_losses)}, got {sorted(losses)}",
                )
            )
        if task == "regression" and losses & cls_losses:
            issues.append(
                SpaceIssue(
                    "error",
                    "loss_function",
                    f"regression loss_function cannot be "
                    f"{sorted(losses & cls_losses)}",
                )
            )
    return issues


def validate_values(
    model: str, task: str, values: Mapping[str, Any]
) -> list[SpaceIssue]:
    """Compatibility-check one drawn parameter combination.

    This is the per-trial guard the bridge calls before constructing an
    estimator (§18.2's "validate before starting"): unknown names,
    invalid solver/penalty pairs, kernel-conditioned no-ops, and
    task-only parameters out of place.

    Examples
    --------
    >>> issues = validate_values(
    ...     "logistic_regression", "classification",
    ...     {"C": 1.0, "penalty": "l1", "solver": "lbfgs"},
    ... )
    >>> issues[0].is_error
    True
    >>> validate_values(
    ...     "svm", "classification", {"C": 1.0, "class_weight": None}
    ... )
    []
    """
    known = {p.name for p in _table_for(model, task)}
    issues: list[SpaceIssue] = []
    for name in values:
        if name not in known:
            issues.append(
                SpaceIssue(
                    "error",
                    name,
                    f"not a supported parameter of {model!r}",
                )
            )
    token = model.strip().lower()
    if token == "deep":
        token = "dnn"
    if token in {"logistic_regression", "elastic_net_logistic", "lasso_logistic"}:
        penalty, solver = values.get("penalty"), values.get("solver")
        if penalty is not None and solver is not None:
            allowed = _SOLVER_PENALTY.get(str(penalty))
            if allowed is not None and str(solver) not in allowed:
                issues.append(
                    SpaceIssue(
                        "error",
                        "solver",
                        f"solver {solver!r} does not support penalty "
                        f"{penalty!r} (sklearn's matrix)",
                    )
                )
    if token == "svm":
        kernel = values.get("kernel")
        if "degree" in values and kernel is not None and str(kernel) != "poly":
            issues.append(
                SpaceIssue("warning", "degree", "no effect with kernel != 'poly'")
            )
        if "gamma" in values and kernel is not None and str(kernel) == "linear":
            issues.append(
                SpaceIssue("warning", "gamma", "ignored by the linear kernel")
            )
    if "epsilon" in values and task != "regression":
        issues.append(
            SpaceIssue("error", "epsilon", "epsilon is an SVR-only parameter")
        )
    if "class_weight" in values and task != "classification":
        issues.append(
            SpaceIssue("error", "class_weight", "class_weight is classification-only")
        )
    return issues


# ---------------------------------------------------------------------------
# The budget estimate (§18.5)
# ---------------------------------------------------------------------------


def estimate_evaluations(
    recommendation: SearchRecommendation,
    *,
    strategy: str = "grid",
    n_trials: int = 25,
) -> EvaluationEstimate:
    """Estimate the number of evaluations a budget implies.

    ``grid`` multiplies the per-parameter point counts (choices count
    their values; integers are inherently discrete grids --
    ``high - low + 1``, or ``range/step`` points when a step is given;
    stepped floats count ``range/step`` points; continuous floats are
    counted as a conventional 20-bin grid, and the assumption is
    recorded).  ``random`` / ``tpe`` simply count the
    trials -- the estimate IS the budget.

    Examples
    --------
    >>> rec = recommend_search("knn", "regression", preset="conservative")
    >>> estimate_evaluations(rec, strategy="grid").count
    24
    >>> estimate_evaluations(rec, strategy="tpe", n_trials=40).count
    40
    """
    if strategy not in {"grid", "random", "tpe"}:
        raise ModelError(
            f"strategy must be 'grid', 'random' or 'tpe', got {strategy!r}"
        )
    if strategy != "grid":
        return EvaluationEstimate(n_trials, strategy)
    count = 1
    assumptions: list[str] = []
    for p in recommendation.searchable:
        if p.kind == "choice":
            count *= max(1, len(p.choices))
        elif p.kind == "boolean":
            count *= 2
        elif p.kind == "int" and not p.step:
            count *= max(1, int(p.high) - int(p.low) + 1)
        elif p.step:
            points = round((p.high - p.low) / p.step) + 1
            count *= max(1, points)
        else:
            count *= 20
            if not assumptions:
                assumptions.append(
                    "continuous parameters counted as a 20-bin grid"
                )
    return EvaluationEstimate(count, "grid", tuple(assumptions))


# ---------------------------------------------------------------------------
# JSON interchange (§18.5 import/export)
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1


def recommendation_to_json(recommendation: SearchRecommendation) -> str:
    """Serialize a recommendation (§18.5's export-as-JSON)."""
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "model": recommendation.model,
        "task": recommendation.task,
        "preset": recommendation.preset,
        "resources": recommendation.resources,
        "n_samples": recommendation.n_samples,
        "n_features": recommendation.n_features,
        "notes": list(recommendation.notes),
        "params": [
            {
                "name": p.name,
                "kind": p.kind,
                "priority": p.priority,
                "purpose": p.purpose,
                "low": p.low,
                "high": p.high,
                "step": p.step,
                "log": p.log,
                "choices": list(p.choices),
                "value": p.value,
                "depends_on": p.depends_on,
            }
            for p in recommendation.params
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def recommendation_from_json(text: str) -> SearchRecommendation:
    """Load a recommendation exported by :func:`recommendation_to_json`.

    Raises
    ------
    DataValidationError
        On any schema violation -- wrong version, missing keys, bad
        kinds/priorities, or malformed ranges (the load-time mirror of
        §18.5's "validate search configurations before execution").

    Examples
    --------
    >>> rec = recommend_search("knn", "regression")
    >>> loaded = recommendation_from_json(recommendation_to_json(rec))
    >>> loaded == rec
    True
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError("the JSON root must be an object")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise DataValidationError(
            f"schema_version {payload.get('schema_version')!r} is not "
            f"supported (expected {_SCHEMA_VERSION})"
        )
    for key in ("model", "task", "preset", "params"):
        if key not in payload:
            raise DataValidationError(f"missing required key {key!r}")
    if payload["preset"] not in PRESET_STYLES:
        raise DataValidationError(
            f"preset {payload['preset']!r} is not one of {PRESET_STYLES}"
        )
    raw_params = payload["params"]
    if not isinstance(raw_params, list) or not raw_params:
        raise DataValidationError("'params' must be a non-empty list")
    params: list[ParamAdvice] = []
    for i, row in enumerate(raw_params):
        if not isinstance(row, dict) or "name" not in row or "kind" not in row:
            raise DataValidationError(
                f"params[{i}] must be an object with name/kind"
            )
        kind = row.get("kind")
        if kind not in {"int", "float", "choice", "boolean", "fixed"}:
            raise DataValidationError(f"params[{i}].kind {kind!r} is unknown")
        priority = row.get("priority", "extended")
        if priority not in PRIORITIES:
            raise DataValidationError(
                f"params[{i}].priority {priority!r} is not one of {PRIORITIES}"
            )
        param = ParamAdvice(
            name=str(row["name"]),
            kind=kind,
            priority=priority,
            purpose=str(row.get("purpose", "")),
            low=row.get("low"),
            high=row.get("high"),
            step=row.get("step"),
            log=bool(row.get("log", False)),
            choices=tuple(row.get("choices", ())),
            value=row.get("value"),
            depends_on=row.get("depends_on"),
        )
        if kind in {"int", "float"}:
            if param.low is None or param.high is None:
                raise DataValidationError(
                    f"params[{i}] ({param.name}): numeric kind needs low/high"
                )
            if param.low >= param.high:
                raise DataValidationError(
                    f"params[{i}] ({param.name}): low must be < high"
                )
            if param.log and param.low <= 0:
                raise DataValidationError(
                    f"params[{i}] ({param.name}): log sampling needs positive bounds"
                )
        if kind == "choice" and not param.choices:
            raise DataValidationError(
                f"params[{i}] ({param.name}): choice kind needs choices"
            )
        params.append(param)
    return SearchRecommendation(
        model=str(payload["model"]),
        task=str(payload["task"]),
        preset=payload["preset"],
        params=tuple(params),
        notes=tuple(payload.get("notes", ())),
        resources=payload.get("resources", "medium"),
        n_samples=payload.get("n_samples"),
        n_features=payload.get("n_features"),
    )


# ---------------------------------------------------------------------------
# The bridge: a recommendation as a tunable HyperModel
# ---------------------------------------------------------------------------


def hypermodel_from_advice(
    model: str,
    task: str,
    *,
    preset: str = "balanced",
    **adapt_kwargs: Any,
) -> Any:
    """Compile a recommendation into a tuner-searchable hypermodel.

    The builder draws every searchable parameter from its recommended
    range/prior, re-validates the drawn combination (§18.2's "validate
    before starting", enforced per trial), and constructs the estimator:
    registry families via ``build_model`` + ``set_params``, the ``ann``
    family as a fresh sklearn MLP, the ``dnn`` family as a fresh PyTorch
    ``DeepMLP``, ``hsvr`` as the hierarchical SVR.  Collection mode
    (no trial values) returns the low bound of every numeric range and
    the first choice of every list -- the deterministic display point.

    Parameters
    ----------
    model, task, preset:
        As :func:`recommend_search`; ``**adapt_kwargs`` forwards the
        dataset/budget adaptation (``n_samples``, ``class_counts``, ...).

    Returns
    -------
    HyperModel
        A ``(hp) -> estimator`` callable named per the module
        convention (``build_<token>_<suffix>_advised``).

    Raises
    ------
    ModelError
        When the recommendation's space carries an ``error``-severity
        validation issue, or a drawn trial violates compatibility.

    Examples
    --------
    >>> from pharmaml.ml.tuner import HyperParameters
    >>> builder = hypermodel_from_advice(
    ...     "random_forest", "classification", preset="conservative"
    ... )
    >>> builder.__name__
    'build_random_forest_classifier_advised'
    >>> hp = HyperParameters()
    >>> estimator = builder(hp)
    >>> sorted(spec.name for spec in hp.space)
    ['max_depth', 'n_estimators']
    >>> estimator.n_estimators    # collection mode: the range's low bound
    300
    """
    rec = recommend_search(model, task, preset=preset, **adapt_kwargs)
    errors = [i for i in validate_space(rec.model, rec.task, rec.params) if i.is_error]
    if errors:
        raise ModelError(
            "the recommended space has validation errors: "
            + "; ".join(f"{i.parameter}: {i.message}" for i in errors)
        )
    token, resolved_task = rec.model, rec.task
    suffix = "regressor" if resolved_task == "regression" else "classifier"

    def build(hp: Any) -> Any:
        drawn: dict[str, Any] = {}
        for p in rec.params:
            if p.kind == "int":
                drawn[p.name] = hp.Int(
                    p.name,
                    int(p.low),
                    int(p.high),
                    step=int(p.step) if p.step else None,
                )
            elif p.kind == "float":
                drawn[p.name] = hp.Float(
                    p.name,
                    float(p.low),
                    float(p.high),
                    step=float(p.step) if p.step else None,
                    log=p.log,
                )
            elif p.kind == "choice":
                drawn[p.name] = hp.Choice(p.name, list(p.choices))
            elif p.kind == "boolean":
                drawn[p.name] = hp.Boolean(p.name)
            else:
                drawn[p.name] = hp.Fixed(p.name, p.value)
        issues = validate_values(token, resolved_task, drawn)
        hard = [i for i in issues if i.is_error]
        if hard:
            raise ModelError(
                "incompatible trial parameters: "
                + "; ".join(f"{i.parameter}: {i.message}" for i in hard)
            )
        return _construct_estimator(token, resolved_task, drawn)

    build.__name__ = f"build_{token}_{suffix}_advised"
    build.__qualname__ = build.__name__
    build.__doc__ = (
        f"(Part-3 §18) advised hypermodel for {token!r} ({resolved_task}, "
        f"{preset} preset): {len(rec.searchable)} searchable parameters "
        f"from the §18.3 recommendation tables."
    )
    return build


def _construct_estimator(token: str, task: str, drawn: Mapping[str, Any]) -> Any:
    """Materialize one estimator from a drawn parameter combination."""
    if token == "ann":
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        layers = int(drawn.get("n_layers", 1))
        units = int(drawn.get("units", 32))
        kwargs: dict[str, Any] = {
            "hidden_layer_sizes": tuple(units for _ in range(layers)),
            "random_state": 42,
        }
        for slot in ("alpha", "learning_rate_init", "max_iter", "batch_size"):
            if slot in drawn:
                kwargs[slot] = drawn[slot]
        return (MLPRegressor if task == "regression" else MLPClassifier)(**kwargs)
    if token == "dnn":
        from pharmaml.ml.neural import DeepMLPClassifier, DeepMLPRegressor

        layers = int(drawn.get("n_layers", 2))
        units = int(drawn.get("units", 64))
        kwargs = {
            "hidden_layers": tuple(units for _ in range(layers)),
            "random_state": 42,
        }
        for slot in (
            "dropout",
            "learning_rate",
            "activation",
            "batchnorm",
            "epochs",
            "batch_size",
            "weight_decay",
            "patience",
        ):
            if slot in drawn:
                kwargs[slot] = drawn[slot]
        return (DeepMLPRegressor if task == "regression" else DeepMLPClassifier)(
            **kwargs
        )
    if token == "hsvr":
        from pharmaml.ml.hierarchical import HierarchicalSVR

        kwargs = {"random_state": 42}
        for slot in ("n_clusters", "C", "gamma", "alpha"):
            if slot in drawn:
                kwargs[slot] = drawn[slot]
        return HierarchicalSVR(**kwargs)
    from pharmaml.ml.models import build_model

    estimator = build_model(token, task)
    estimator.set_params(**dict(drawn))
    return estimator
