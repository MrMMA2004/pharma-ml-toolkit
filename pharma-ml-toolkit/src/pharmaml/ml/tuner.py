"""Keras-tuner-style hyperparameter search for scikit-compatible models.

Re-implements the programming model of `Keras Tuner
<https://keras.io/keras_tuner/>`_ — the ``HyperParameters`` space
object whose ``Int`` / ``Float`` / ``Choice`` / ``Boolean`` / ``Fixed``
methods are *called inside a model-building function* to draw values,
and the ``RandomSearch`` / ``Hyperband`` tuners that drive it — over
ANY scikit-learn-compatible estimator, with no TensorFlow requirement
(the toolkit's deep estimators are PyTorch; the legacy ANN is sklearn's
MLP).  The same ``hypermodel(hp) -> estimator`` contract works for all
of them::

    from pharmaml.ml.tuner import HyperParameters, RandomSearch
    from pharmaml.ml.tuner import build_ann_regressor

    tuner = RandomSearch(build_ann_regressor, objective="r2",
                         max_trials=12, cv=3, seed=42)
    results = tuner.search(X, y)
    results.best_params            # {'n_layers': 2, 'units': 64, ...}
    results.best_score             # mean cross-validated r2
    results.best_estimator         # refitted on the full data

Search strategy: every trial draws one value per registered
hyperparameter from a seeded NumPy generator (``Int``/``Float`` honour
``step``; ``Choice`` samples uniformly; ``Boolean`` is a two-way
``Choice``), builds the estimator, and scores it with
:func:`sklearn.model_selection.cross_val_score`.  Higher scores win —
sklearn scorer convention (``'r2'``, ``'accuracy'``, ``roc_auc'`, ...);
custom callables ``score = f(estimator, X, y)`` are accepted with the
same higher-is-better contract.  A trial whose estimator raises is
recorded as failed (``score = nan``) and the search continues; a search
in which every trial fails raises :class:`~pharmaml.exceptions.ModelError`.

:func:`Hyperband` implements the classic successive-halving schedule
(Li et al., 2013): brackets of configs start at a small resource
budget, and only the top ``1/eta`` survive each rung.  The resource is
a hyperparameter the hypermodel declares by name (``epochs`` for the
deep estimators, ``max_iter`` for the sklearn ANN) whose value the
tuner overrides per rung.

Determinism: sampling, k-fold splitting and estimator seeds all derive
from ``seed``, so two searches with the same seed and data produce
identical results.

Examples
--------
>>> from pharmaml.ml.tuner import HyperParameters, RandomSearch, build_ann_regressor

The space is collected by calling the hypermodel once with a fresh
``HyperParameters`` (values unset -> defaults returned, specs recorded):

>>> hp = HyperParameters()
>>> estimator = build_ann_regressor(hp)
>>> sorted(spec.name for spec in hp.space)
['alpha', 'learning_rate_init', 'max_iter', 'n_layers', 'units']
>>> estimator.max_iter               # the declared default
400

A trial-time ``HyperParameters`` returns the oracle's values instead:

>>> hp = HyperParameters(values={"n_layers": 2, "units": 64, "alpha": 1e-3,
...                              "learning_rate_init": 1e-3, "max_iter": 300})
>>> build_ann_regressor(hp).hidden_layer_sizes
(64, 64)

>>> import numpy as np
>>> rng = np.random.default_rng(7)
>>> X = rng.uniform(0, 1, (80, 3))
>>> y = 3.0 * X[:, 0] + 1.0
>>> tuner = RandomSearch(build_ann_regressor, objective="r2",
...                      max_trials=3, cv=2, seed=11)
>>> results = tuner.search(X, y)
>>> len(results.trials)              # one TrialResult per trial
3
>>> results.best_score > 0.9         # a linear surface is learnable
True
>>> twin = RandomSearch(build_ann_regressor, objective="r2",
...                     max_trials=3, cv=2, seed=11).search(X, y)
>>> twin.best_params == results.best_params   # deterministic
True
>>> results.summary().splitlines()[0]
'RandomSearch: 3 trial(s), objective=r2'

>>> RandomSearch(build_ann_regressor, objective="not_a_scorer").search(X, y)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: unknown objective 'not_a_scorer' — pass a sklearn scorer name (e.g. 'r2', 'neg_mean_squared_error', 'accuracy', 'roc_auc') or a callable(estimator, X, y) -> float
>>> RandomSearch(build_ann_regressor, max_trials=0).search(X, y)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: max_trials must be an integer >= 1, got 0
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from pharmaml.exceptions import ModelError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "HyperParameters",
    "Hyperband",
    "RandomSearch",
    "TrialResult",
    "TunerResults",
    "build_ann_classifier",
    "build_ann_regressor",
    "build_deep_classifier",
    "build_deep_regressor",
    "build_hsvr",
    "make_cv_splitter",
]


#: A hypermodel: takes a :class:`HyperParameters`, returns an unfitted
#: scikit-learn-compatible estimator.
HyperModel = Callable[["HyperParameters"], Any]


@dataclass(frozen=True, slots=True)
class _HyperParameterSpec:
    """One registered hyperparameter slot (keras-tuner flavour)."""

    name: str
    kind: str  # 'int' | 'float' | 'choice' | 'boolean' | 'fixed'
    values: tuple[Any, ...] = ()  # discrete grid (int/choice/boolean/fixed)
    low: float = 0.0  # continuous bounds (float)
    high: float = 0.0
    step: float | None = None
    default: Any = None

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw one value from this slot's distribution."""
        if self.kind == "float":
            if self.step:
                n_steps = round((self.high - self.low) / self.step)
                return float(self.low + self.step * int(rng.integers(0, n_steps + 1)))
            return float(rng.uniform(self.low, self.high))
        index = int(rng.integers(0, len(self.values)))
        value = self.values[index]
        if self.kind == "int":
            return int(value)
        if self.kind == "boolean":
            return bool(value)
        return value


class HyperParameters:
    """The keras-tuner container of hyperparameter definitions.

    Call ``hp.Int(...)`` / ``hp.Float(...)`` / ``hp.Choice(...)`` /
    ``hp.Boolean(...)`` / ``hp.Fixed(...)`` *inside* the hypermodel:
    each call registers the slot (or verifies it is already registered)
    and returns the value chosen for this trial — the default when no
    ``values`` mapping was supplied (the space-collection pass), the
    oracle's value during a trial.

    Parameters
    ----------
    values:
        Optional name -> value mapping for trial mode; unregistered
        names passed by the caller are ignored (the oracle only samples
        slots the hypermodel declares).
    """

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values: dict[str, Any] = dict(values or {})
        self.space: list[_HyperParameterSpec] = []
        self._by_name: dict[str, _HyperParameterSpec] = {}

    # -- registration helpers ------------------------------------------
    def _register(self, spec: _HyperParameterSpec) -> Any:
        """Register ``spec`` (or verify an identical re-registration)."""
        existing = self._by_name.get(spec.name)
        if existing is not None and existing != spec:
            raise ModelError(
                f"hyperparameter {spec.name!r} re-registered with a different "
                f"spec: {existing!r} vs {spec!r}"
            )
        if existing is None:
            self.space.append(spec)
            self._by_name[spec.name] = spec
        if spec.name in self.values:
            return self.values[spec.name]
        return spec.default

    # -- keras-tuner API -------------------------------------------------
    def Int(
        self,
        name: str,
        min_value: int,
        max_value: int,
        step: int | None = None,
        default: int | None = None,
    ) -> int:
        """Register an integer slot and return this trial's value.

        Raises
        ------
        ModelError
            When ``min_value > max_value`` or ``step`` is not positive.
        """
        min_value, max_value = int(min_value), int(max_value)
        if min_value > max_value:
            raise ModelError(
                f"Int({name!r}): min_value {min_value} > max_value {max_value}"
            )
        if step is not None and step <= 0:
            raise ModelError(f"Int({name!r}): step must be > 0, got {step}")
        grid = (
            tuple(range(min_value, max_value + 1, step))
            if step
            else tuple(range(min_value, max_value + 1))
        )
        if not grid:  # pragma: no cover - step always divides into >= 1 value
            grid = (min_value,)
        return self._register(
            _HyperParameterSpec(
                name=name,
                kind="int",
                values=grid,
                default=default if default is not None else grid[0],
            )
        )

    def Float(
        self,
        name: str,
        min_value: float,
        max_value: float,
        step: float | None = None,
        default: float | None = None,
    ) -> float:
        """Register a float slot and return this trial's value.

        Raises
        ------
        ModelError
            When ``min_value > max_value`` or ``step`` is not positive.
        """
        min_value, max_value = float(min_value), float(max_value)
        if min_value > max_value:
            raise ModelError(
                f"Float({name!r}): min_value {min_value} > max_value {max_value}"
            )
        if step is not None and step <= 0:
            raise ModelError(f"Float({name!r}): step must be > 0, got {step}")
        return self._register(
            _HyperParameterSpec(
                name=name,
                kind="float",
                low=min_value,
                high=max_value,
                step=step,
                default=default if default is not None else min_value,
            )
        )

    def Choice(
        self,
        name: str,
        values: Sequence[Any],
        default: Any = None,
    ) -> Any:
        """Register a categorical slot and return this trial's value.

        Raises
        ------
        ModelError
            When ``values`` is empty.
        """
        if len(values) == 0:
            raise ModelError(f"Choice({name!r}): values must not be empty")
        return self._register(
            _HyperParameterSpec(
                name=name,
                kind="choice",
                values=tuple(values),
                default=default if default is not None else values[0],
            )
        )

    def Boolean(self, name: str, default: bool = False) -> bool:
        """Register a boolean slot and return this trial's value."""
        return self._register(
            _HyperParameterSpec(
                name=name,
                kind="boolean",
                values=(False, True),
                default=bool(default),
            )
        )

    def Fixed(self, name: str, value: Any) -> Any:
        """Register a constant slot (always returns ``value``)."""
        return self._register(
            _HyperParameterSpec(name=name, kind="fixed", values=(value,), default=value)
        )


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One evaluated trial.

    Attributes
    ----------
    trial_id:
        1-based trial number (Hyperband rung evaluations get their own
        ids too).
    hyperparameters:
        The values the oracle assigned (resource overrides included).
    score:
        Mean cross-validated objective; ``nan`` for a failed trial.
    score_std:
        Standard deviation across the CV folds.
    resource:
        The resource budget this trial ran at (Hyperband only; ``None``
        for RandomSearch).
    error:
        The exception message when the trial failed, else ``None``.
    """

    trial_id: int
    hyperparameters: dict[str, Any]
    score: float
    score_std: float
    resource: float | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        """True when the trial raised instead of scoring."""
        return self.error is not None or not math.isfinite(self.score)


class TunerResults:
    """The outcome of one :meth:`RandomSearch.search` / Hyperband run.

    Attributes
    ----------
    trials:
        Every :class:`TrialResult`, in evaluation order.
    objective:
        The objective label (scorer name or ``'custom'``).
    """

    def __init__(
        self, trials: list[TrialResult], objective: str, label: str = "RandomSearch"
    ) -> None:
        self.trials = trials
        self.objective = objective
        self._label = label
        #: Refit winner, cached by the tuner's search() (None until then).
        self._best_estimator: Any = None
        scored = [t for t in trials if not t.failed]
        if not scored:
            raise ModelError(
                "every trial failed — the last error was: "
                f"{trials[-1].error if trials else 'no trials ran'}"
            )
        self._best = max(scored, key=lambda t: t.score)

    @property
    def best_trial(self) -> TrialResult:
        """The highest-scoring (finite) trial."""
        return self._best

    @property
    def best_params(self) -> dict[str, Any]:
        """The winning hyperparameters (resource override included)."""
        return dict(self._best.hyperparameters)

    @property
    def best_score(self) -> float:
        """The winning mean cross-validated objective."""
        return self._best.score

    @property
    def best_estimator(self) -> Any:
        """The winning estimator, refitted on the full search data.

        Rebuilt from ``best_params`` and fitted inside
        :meth:`RandomSearch.search`; cached afterwards.
        """
        cached = self._best_estimator
        if cached is None:
            raise ModelError(
                "best_estimator is only available after search() completed "
                "with at least one scoring trial"
            )
        return cached

    def summary(self) -> str:
        """A human-readable leaderboard (top 10 trials by score)."""
        lines = [
            f"{self._label}: {len(self.trials)} trial(s), objective={self.objective}"
        ]
        ranked = sorted(
            (t for t in self.trials if not t.failed),
            key=lambda t: t.score,
            reverse=True,
        )[:10]
        for trial in ranked:
            resource = "" if trial.resource is None else f", resource={trial.resource:g}"
            lines.append(
                f"  trial {trial.trial_id}: score {trial.score:.4f} "
                f"+/- {trial.score_std:.4f}{resource} | {trial.hyperparameters}"
            )
        failed = sum(1 for t in self.trials if t.failed)
        if failed:
            lines.append(f"  ({failed} trial(s) failed)")
        return "\n".join(lines)

    def trials_frame(self) -> pd.DataFrame:
        """The trials as a DataFrame (one row per trial, params as columns)."""
        rows = []
        for trial in self.trials:
            row: dict[str, Any] = {
                "trial_id": trial.trial_id,
                "score": trial.score,
                "score_std": trial.score_std,
                "resource": trial.resource,
                "failed": trial.failed,
            }
            row.update(trial.hyperparameters)
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"<TunerResults: {len(self.trials)} trials, objective={self.objective!r}, "
            f"best_score={self.best_score:.4f}>"
        )


# ---------------------------------------------------------------------------
# Shared tuner machinery
# ---------------------------------------------------------------------------
def _resolve_objective(objective: str | Callable[..., float]) -> tuple[Any, str]:
    """Map the objective to a sklearn scorer (or pass a callable through)."""
    if callable(objective) and not isinstance(objective, str):
        return objective, "custom"
    try:
        return get_scorer(str(objective)), str(objective)
    except ValueError as exc:
        raise ModelError(
            f"unknown objective {objective!r} — pass a sklearn scorer name "
            "(e.g. 'r2', 'neg_mean_squared_error', 'accuracy', 'roc_auc') or a "
            "callable(estimator, X, y) -> float"
        ) from exc


def _make_splitter(task: str, y: np.ndarray, cv: int) -> Any:
    """The CV splitter for the inferred task (stratified for classes).

    Falls back to a plain shuffled ``KFold`` when stratification is
    impossible — the smallest class holds fewer members than the fold
    count (singleton-class targets that the loader heuristics still
    call classification); ``StratifiedKFold`` would raise on those and
    every trial would fail with an opaque error.
    """
    if task == "auto":
        numeric = np.issubdtype(np.asarray(y).dtype, np.number)
        n_unique = np.unique(y).size
        task = "regression" if numeric and n_unique > 10 else "classification"
    if task == "classification":
        _, counts = np.unique(np.asarray(y), return_counts=True)
        if int(counts.min()) >= int(cv):
            return StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
        _LOG.debug(
            "stratification impossible (smallest class has %d member(s) < cv=%d)"
            " — falling back to plain KFold",
            int(counts.min()),
            int(cv),
        )
    return KFold(n_splits=cv, shuffle=True, random_state=0)


def make_cv_splitter(task: str, y: Any, cv: int) -> Any:
    """The tuners' fold protocol, as a public helper.

    ``KFold(n_splits=cv, shuffle=True, random_state=0)`` for regression,
    ``StratifiedKFold`` for classification — falling back to the plain
    shuffled ``KFold`` when stratification is impossible (the smallest
    class holds fewer members than the fold count).  The tuners score
    every trial under exactly these folds, so scoring a *baseline*
    configuration with the same splitter is apples-to-apples (the
    playground's Tune button and the tuning notebook both do this).

    Parameters
    ----------
    task:
        ``'auto'`` (numeric targets with > 10 distinct values infer
        regression; anything else classification), ``'regression'`` or
        ``'classification'``.
    y:
        The target vector (anything ``numpy.asarray`` accepts).
    cv:
        Fold count (>= 2).

    Raises
    ------
    ModelError
        For an invalid ``cv`` or ``task`` argument.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.tuner import make_cv_splitter
    >>> make_cv_splitter("regression", np.arange(30.0), 3)
    KFold(n_splits=3, random_state=0, shuffle=True)
    >>> make_cv_splitter("classification", ["a"] * 10 + ["b"] * 10, 2)
    StratifiedKFold(n_splits=2, random_state=0, shuffle=True)
    >>> make_cv_splitter("classification", ["a"] * 2 + ["b"] * 30, 4)
    KFold(n_splits=4, random_state=0, shuffle=True)
    >>> make_cv_splitter("regression", np.arange(10.0), 1)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.ModelError: cv must be an integer >= 2, got 1
    """
    if (
        isinstance(cv, bool)
        or not isinstance(cv, (int, np.integer))
        or int(cv) < 2
    ):
        raise ModelError(f"cv must be an integer >= 2, got {cv!r}")
    if task not in {"auto", "regression", "classification"}:
        raise ModelError(
            f"task must be 'auto', 'regression' or 'classification', got {task!r}"
        )
    return _make_splitter(task, np.asarray(y), int(cv))


def _collect_space(hypermodel: HyperModel) -> list[_HyperParameterSpec]:
    """Run the hypermodel once in space-collection mode (defaults drawn).

    Raises
    ------
    ModelError
        When the hypermodel raises on its default configuration (the
        space can never be collected) or returns an object without
        ``fit``.
    """
    hp = HyperParameters()
    try:
        estimator = hypermodel(hp)
    except Exception as exc:
        raise ModelError(
            "the hypermodel failed during space collection (the default "
            f"configuration must build): {type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(estimator, "fit"):
        raise ModelError(
            "the hypermodel must return a scikit-learn-compatible estimator "
            f"with fit(); got {type(estimator).__name__}"
        )
    return hp.space


def _evaluate_trial(
    hypermodel: HyperModel,
    values: dict[str, Any],
    X: Any,
    y: Any,
    scorer: Any,
    splitter: Any,
) -> tuple[float, float, str | None]:
    """Build + cross-validate one config; returns ``(score, std, error)``.

    A fit that raises inside :func:`cross_val_score` is scored ``nan``
    (``error_score=np.nan``) rather than crashing the search; a build
    failure surfaces as the error string instead.
    """
    hp = HyperParameters(values=values)
    try:
        estimator = hypermodel(hp)
        scores = cross_val_score(
            estimator, X, y, scoring=scorer, cv=splitter, error_score=np.nan
        )
    except Exception as exc:  # noqa: BLE001 — a failed trial is a result, not a crash
        return float("nan"), float("nan"), f"{type(exc).__name__}: {exc}"
    return float(np.nanmean(scores)), float(np.nanstd(scores)), None


# ---------------------------------------------------------------------------
# RandomSearch
# ---------------------------------------------------------------------------
class RandomSearch:
    """Random search over the hypermodel's declared space.

    Parameters
    ----------
    hypermodel:
        Callable ``(hp) -> estimator`` (see :func:`build_ann_regressor`
        for the shape).
    objective:
        sklearn scorer name (``'r2'``, ``'accuracy'``, ...) or a
        callable ``(estimator, X, y) -> float``; higher is better.
    max_trials:
        Number of trials (>= 1).
    cv:
        Cross-validation fold count (>= 2).
    task:
        ``'auto'`` (numeric targets with > 10 distinct values infer
        regression; anything else classification/stratified folds),
        ``'regression'`` or ``'classification'``.
    seed:
        Integer seed for the value sampler and the fold shuffle.

    Raises
    ------
    ModelError
        For invalid constructor values, an unknown objective, a
        hypermodel that does not return a fittable estimator, or when
        every trial fails.
    """

    def __init__(
        self,
        hypermodel: HyperModel,
        objective: str | Callable[..., float] = "r2",
        max_trials: int = 10,
        cv: int = 3,
        task: str = "auto",
        seed: int = 42,
    ) -> None:
        self.hypermodel = hypermodel
        self.objective = objective
        self.max_trials = max_trials
        self.cv = cv
        self.task = task
        self.seed = seed

    def _validate(self) -> None:
        if (
            isinstance(self.max_trials, bool)
            or not isinstance(self.max_trials, (int, np.integer))
            or int(self.max_trials) < 1
        ):
            raise ModelError(
                f"max_trials must be an integer >= 1, got {self.max_trials!r}"
            )
        if isinstance(self.cv, bool) or not isinstance(self.cv, (int, np.integer)) or int(self.cv) < 2:
            raise ModelError(f"cv must be an integer >= 2, got {self.cv!r}")
        if self.task not in {"auto", "regression", "classification"}:
            raise ModelError(
                f"task must be 'auto', 'regression' or 'classification', "
                f"got {self.task!r}"
            )

    def search(self, X: Any, y: Any) -> TunerResults:
        """Run the search; returns :class:`TunerResults`.

        The winning estimator is rebuilt from ``best_params`` and fitted
        on the full ``(X, y)`` — available as ``results.best_estimator``.
        """
        self._validate()
        scorer, objective_label = _resolve_objective(self.objective)
        space = _collect_space(self.hypermodel)
        splitter = make_cv_splitter(self.task, y, int(self.cv))
        rng = np.random.default_rng(int(self.seed))

        trials: list[TrialResult] = []
        best: tuple[float, dict[str, Any]] | None = None  # (score, values)
        for trial_id in range(1, int(self.max_trials) + 1):
            values = {spec.name: spec.sample(rng) for spec in space}
            score, std, error = _evaluate_trial(
                self.hypermodel, values, X, y, scorer, splitter
            )
            _LOG.debug(
                "trial %d/%d: score %.4f %s", trial_id, self.max_trials, score, values
            )
            trials.append(
                TrialResult(
                    trial_id=trial_id,
                    hyperparameters=values,
                    score=score,
                    score_std=std,
                    error=error,
                )
            )
            if error is None and np.isfinite(score) and (best is None or score > best[0]):
                best = (score, values)

        results = TunerResults(trials, objective_label)
        if best is not None:
            # Refit the winning configuration on the full data.
            winner = HyperParameters(values=best[1])
            results._best_estimator = self.hypermodel(winner).fit(X, y)
        return results


# ---------------------------------------------------------------------------
# Hyperband
# ---------------------------------------------------------------------------
class Hyperband:
    """Hyperband successive halving over a resource hyperparameter.

    Classic schedule (Li et al., 2013) with ``eta`` (default 3): bracket
    ``s`` starts ``n(s)`` fresh configs at resource ``max_resource /
    eta**s``; every rung keeps only the top ``ceil(n / eta)`` configs
    and multiplies their resource by ``eta``.  Every rung evaluation is
    recorded as a trial, so ``results.trials`` mixes budgets by design.

    Parameters
    ----------
    hypermodel:
        Callable ``(hp) -> estimator``.  Must declare the resource slot
        by name — e.g. ``hp.Int("epochs", 30, 150)`` — the tuner
        overrides its value per rung (clamped to
        ``[min_resource, max_resource]``).
    resource:
        Name of the resource hyperparameter (default ``'epochs'``).
    max_resource:
        Largest resource budget (integer >= ``min_resource``).
    min_resource:
        Smallest resource budget (>= 1).
    eta:
        Halving/promotion factor (>= 2).
    objective, cv, task, seed:
        Same contract as :class:`RandomSearch`.

    Raises
    ------
    ModelError
        For invalid constructor values, when the hypermodel does not
        declare the ``resource`` slot, or when every trial fails.
    """

    def __init__(
        self,
        hypermodel: HyperModel,
        objective: str | Callable[..., float] = "r2",
        resource: str = "epochs",
        max_resource: int = 81,
        min_resource: int = 3,
        eta: int = 3,
        cv: int = 3,
        task: str = "auto",
        seed: int = 42,
    ) -> None:
        self.hypermodel = hypermodel
        self.objective = objective
        self.resource = resource
        self.max_resource = max_resource
        self.min_resource = min_resource
        self.eta = eta
        self.cv = cv
        self.task = task
        self.seed = seed

    def _validate(self) -> None:
        for name, value in (("max_resource", self.max_resource),
                            ("min_resource", self.min_resource)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ModelError(f"{name} must be an integer >= 1, got {value!r}")
        if int(self.min_resource) > int(self.max_resource):
            raise ModelError(
                f"min_resource {self.min_resource} > max_resource {self.max_resource}"
            )
        if isinstance(self.eta, bool) or not isinstance(self.eta, (int, np.integer)) or int(self.eta) < 2:
            raise ModelError(f"eta must be an integer >= 2, got {self.eta!r}")
        if isinstance(self.cv, bool) or not isinstance(self.cv, (int, np.integer)) or int(self.cv) < 2:
            raise ModelError(f"cv must be an integer >= 2, got {self.cv!r}")
        if self.task not in {"auto", "regression", "classification"}:
            raise ModelError(
                f"task must be 'auto', 'regression' or 'classification', "
                f"got {self.task!r}"
            )

    def search(self, X: Any, y: Any) -> TunerResults:
        """Run the brackets; returns :class:`TunerResults`."""
        self._validate()
        scorer, objective_label = _resolve_objective(self.objective)
        space = _collect_space(self.hypermodel)
        names = {spec.name for spec in space}
        if self.resource not in names:
            raise ModelError(
                f"the hypermodel does not declare the resource hyperparameter "
                f"{self.resource!r} — add hp.Int({self.resource!r}, ...) to the "
                "builder (or pass resource=<declared name>)"
            )
        splitter = make_cv_splitter(self.task, y, int(self.cv))
        rng = np.random.default_rng(int(self.seed))
        eta, r_max, r_min = int(self.eta), int(self.max_resource), int(self.min_resource)

        def clamp(value: int) -> int:
            return int(min(max(value, r_min), r_max))

        trials: list[TrialResult] = []
        best: tuple[float, dict[str, Any]] | None = None
        trial_counter = 0

        # Classic Hyperband brackets (Li et al., 2013): bracket ``s``
        # starts ceil((s_max+1) * eta**s / (s+1)) configs at resource
        # max_resource / eta**s and runs exactly ``s + 1`` rungs,
        # keeping the top ceil(n / eta) after each.
        s_max = math.floor(math.log(r_max / r_min, eta)) if r_max > r_min else 0
        for bracket in reversed(range(s_max + 1)):
            n_configs = math.ceil((s_max + 1) * eta**bracket / (bracket + 1))
            r0 = clamp(r_max // eta**bracket)
            configs: list[dict[str, Any]] = [
                {spec.name: spec.sample(rng) for spec in space} for _ in range(n_configs)
            ]
            for rung in range(bracket + 1):
                rung_resource = clamp(r0 * eta**rung)
                scored: list[tuple[float, dict[str, Any]]] = []
                for values in configs:
                    values[self.resource] = rung_resource
                    trial_counter += 1
                    score, std, error = _evaluate_trial(
                        self.hypermodel, values, X, y, scorer, splitter
                    )
                    trials.append(
                        TrialResult(
                            trial_id=trial_counter,
                            hyperparameters=dict(values),
                            score=score,
                            score_std=std,
                            resource=float(rung_resource),
                            error=error,
                        )
                    )
                    if error is None and np.isfinite(score):
                        scored.append((score, values))
                        if best is None or score > best[0]:
                            best = (score, values)
                if not scored:
                    break  # every config failed at this rung
                scored.sort(key=lambda pair: pair[0], reverse=True)
                keep = math.ceil(len(configs) / eta)
                configs = [values for _, values in scored[:keep]]

        results = TunerResults(trials, objective_label, label="Hyperband")
        if best is not None:
            winner = HyperParameters(values=best[1])
            results._best_estimator = self.hypermodel(winner).fit(X, y)
        return results


# ---------------------------------------------------------------------------
# Ready-made hypermodels
# ---------------------------------------------------------------------------
def build_ann_regressor(hp: HyperParameters) -> Any:
    """sklearn MLP regressor hypermodel (the legacy ANN upgrade).

    Searchable: ``n_layers`` (1-3), ``units`` (16-128 step 16),
    ``alpha`` (L2), ``learning_rate_init`` and the ``max_iter``
    resource (200-1000, step 100).
    """
    from sklearn.neural_network import MLPRegressor

    n_layers = hp.Int("n_layers", 1, 3, default=1)
    units = hp.Int("units", 16, 128, step=16, default=32)
    alpha = hp.Choice("alpha", [1e-4, 1e-3, 1e-2], default=1e-3)
    learning_rate_init = hp.Choice(
        "learning_rate_init", [1e-3, 3e-3], default=1e-3
    )
    max_iter = hp.Int("max_iter", 200, 1000, step=100, default=400)
    return MLPRegressor(
        hidden_layer_sizes=tuple(units for _ in range(n_layers)),
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=42,
    )


def build_ann_classifier(hp: HyperParameters) -> Any:
    """sklearn MLP classifier hypermodel (the legacy ANN upgrade).

    Same searchable space as :func:`build_ann_regressor`.
    """
    from sklearn.neural_network import MLPClassifier

    n_layers = hp.Int("n_layers", 1, 3, default=1)
    units = hp.Int("units", 16, 128, step=16, default=32)
    alpha = hp.Choice("alpha", [1e-4, 1e-3, 1e-2], default=1e-3)
    learning_rate_init = hp.Choice(
        "learning_rate_init", [1e-3, 3e-3], default=1e-3
    )
    max_iter = hp.Int("max_iter", 200, 1000, step=100, default=400)
    return MLPClassifier(
        hidden_layer_sizes=tuple(units for _ in range(n_layers)),
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=42,
    )


def build_deep_regressor(hp: HyperParameters) -> Any:
    """Deep MLP regressor hypermodel (PyTorch, ``batchnorm`` searchable).

    Searchable: ``n_layers`` (1-3), ``units`` (16-128 step 16),
    ``dropout`` (0-0.4 step 0.1), ``learning_rate``, ``activation``
    (relu/tanh/gelu) and ``batchnorm`` (Boolean) — plus the ``epochs``
    resource (30-150, step 10).  Requires the optional torch dependency
    (``pip install 'pharmaml[deep]'``).
    """
    from pharmaml.ml.neural import DeepMLPRegressor

    n_layers = hp.Int("n_layers", 1, 3, default=2)
    units = hp.Int("units", 16, 128, step=16, default=64)
    dropout = hp.Float("dropout", 0.0, 0.4, step=0.1, default=0.1)
    learning_rate = hp.Choice("learning_rate", [1e-3, 3e-4], default=1e-3)
    activation = hp.Choice("activation", ["relu", "tanh", "gelu"], default="relu")
    batchnorm = hp.Boolean("batchnorm", default=False)
    epochs = hp.Int("epochs", 30, 150, step=10, default=80)
    return DeepMLPRegressor(
        hidden_layers=tuple(units for _ in range(n_layers)),
        dropout=dropout,
        learning_rate=learning_rate,
        activation=activation,
        batchnorm=batchnorm,
        epochs=epochs,
        patience=10,
        random_state=42,
    )


def build_deep_classifier(hp: HyperParameters) -> Any:
    """Deep MLP classifier hypermodel (PyTorch, ``batchnorm`` searchable).

    Same searchable space as :func:`build_deep_regressor`.
    """
    from pharmaml.ml.neural import DeepMLPClassifier

    n_layers = hp.Int("n_layers", 1, 3, default=2)
    units = hp.Int("units", 16, 128, step=16, default=64)
    dropout = hp.Float("dropout", 0.0, 0.4, step=0.1, default=0.1)
    learning_rate = hp.Choice("learning_rate", [1e-3, 3e-4], default=1e-3)
    activation = hp.Choice("activation", ["relu", "tanh", "gelu"], default="relu")
    batchnorm = hp.Boolean("batchnorm", default=False)
    epochs = hp.Int("epochs", 30, 150, step=10, default=80)
    return DeepMLPClassifier(
        hidden_layers=tuple(units for _ in range(n_layers)),
        dropout=dropout,
        learning_rate=learning_rate,
        activation=activation,
        batchnorm=batchnorm,
        epochs=epochs,
        patience=10,
        random_state=42,
    )


def build_hsvr(hp: HyperParameters) -> Any:
    """:class:`~pharmaml.ml.hierarchical.HierarchicalSVR` hypermodel.

    Searchable: ``n_clusters`` (2-5), ``C``, ``gamma`` and the local
    blending ``alpha`` (0-1, step 0.1).  Declares no resource slot, so
    :class:`Hyperband` over it raises by contract — use
    :class:`RandomSearch`.
    """
    from pharmaml.ml.hierarchical import HierarchicalSVR

    n_clusters = hp.Int("n_clusters", 2, 5, default=3)
    C = hp.Choice("C", [0.1, 1.0, 10.0], default=1.0)
    gamma = hp.Choice("gamma", ["scale", "auto"], default="scale")
    alpha = hp.Float("alpha", 0.0, 1.0, step=0.1, default=0.8)
    return HierarchicalSVR(
        n_clusters=n_clusters, C=C, gamma=gamma, alpha=alpha, random_state=42
    )
