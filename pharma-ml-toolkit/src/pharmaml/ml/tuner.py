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

:func:`TPESearch` absorbs Optuna's flagship Tree-structured Parzen
Estimator (no optuna dependency — the programming model, exactly like
the Keras-Tuner round): per hyperparameter, one Parzen (KDE) mixture is
fitted to the best ``gamma(n) = min(ceil(0.1 n), 25)`` completed trials
and one to the rest, ``n_ei_candidates`` candidates are drawn from the
"below" mixture and the argmax of ``log pdf_below - log pdf_above``
(the expected-improvement surrogate) wins; the first
``n_startup_trials`` informative trials are plain uniform draws.
:class:`PercentilePruner` / :func:`MedianPruner` implement Optuna's
report/should-prune fold loop — a trial whose running CV mean falls
below a percentile of the completed trials at the same fold is cut
short and recorded with ``state='pruned'`` — and
:func:`param_importances` ranks the hyperparameters of a finished
search with Optuna's random-forest MDI evaluator.  Trial records carry
Optuna's states (complete/pruned/fail), the per-fold intermediates and
the trial durations.

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
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from pharmaml.exceptions import ModelError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "HyperParameters",
    "Hyperband",
    "HyperparameterImportance",
    "MedianPruner",
    "PercentilePruner",
    "RandomSearch",
    "TPESearch",
    "TrialResult",
    "TunerResults",
    "build_ann_classifier",
    "build_ann_regressor",
    "build_deep_classifier",
    "build_deep_regressor",
    "build_hsvr",
    "make_cv_splitter",
    "param_importances",
    "resource_bounds",
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
    log: bool = False  # sample in the log domain (Optuna's suggest_float(log=True))
    default: Any = None

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw one value from this slot's distribution."""
        if self.kind == "float":
            if self.log:
                # log-domain draw: exp(uniform(log(low), log(high))) —
                # the honest prior for scale parameters (learning rates);
                # the bounds are validated positive at registration
                if self.low <= 0.0 or self.high <= 0.0:
                    raise ModelError(
                        f"hyperparameter {self.name!r}: log sampling requires "
                        f"positive bounds, got [{self.low}, {self.high}]"
                    )
                return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
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
        log: bool = False,
    ) -> float:
        """Register a float slot and return this trial's value.

        ``log=True`` samples in the log domain —
        ``exp(uniform(log(min), log(max)))`` — the honest prior for
        scale parameters such as learning rates (Optuna's own
        recommendation for ``suggest_float``).  It refuses ``step`` (a
        grid and a log prior are contradictory — Optuna's own rule) and
        non-positive bounds (the log domain needs ``min_value > 0``).

        Raises
        ------
        ModelError
            When ``min_value > max_value``, ``step`` is not positive,
            ``log=True`` together with a ``step``, or ``log=True`` with
            ``min_value <= 0``.

        Examples
        --------
        >>> import numpy as np
        >>> from pharmaml.ml.tuner import HyperParameters
        >>> hp = HyperParameters()
        >>> hp.Float("dropout", 0.0, 0.4, step=0.1)   # collection mode: default
        0.0
        >>> hp.Float("lr", 1e-5, 1e-1, log=True)     # the log-domain default
        1e-05
        >>> spec = hp.space[-1]
        >>> rng = np.random.default_rng(0)
        >>> draws = [spec.sample(rng) for _ in range(200)]
        >>> all(1e-5 <= v <= 1e-1 for v in draws)
        True
        >>> min(draws) > 0.0                            # never 0, never negative
        True
        >>> HyperParameters().Float("lr", 1e-5, 1e-1, step=0.1, log=True)
        Traceback (most recent call last):
        ...
        pharmaml.exceptions.ModelError: Float('lr'): log=True requires step=None, got step=0.1
        >>> HyperParameters().Float("lr", 0.0, 1.0, log=True)
        Traceback (most recent call last):
        ...
        pharmaml.exceptions.ModelError: Float('lr'): log=True requires min_value > 0, got 0.0
        """
        min_value, max_value = float(min_value), float(max_value)
        if min_value > max_value:
            raise ModelError(
                f"Float({name!r}): min_value {min_value} > max_value {max_value}"
            )
        if step is not None and step <= 0:
            raise ModelError(f"Float({name!r}): step must be > 0, got {step}")
        if log:
            if step is not None:
                raise ModelError(
                    f"Float({name!r}): log=True requires step=None, got step={step}"
                )
            if min_value <= 0.0:
                raise ModelError(
                    f"Float({name!r}): log=True requires min_value > 0, "
                    f"got {min_value}"
                )
        return self._register(
            _HyperParameterSpec(
                name=name,
                kind="float",
                low=min_value,
                high=max_value,
                step=step,
                log=bool(log),
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
        Mean cross-validated objective; ``nan`` for a failed or pruned
        trial.
    score_std:
        Standard deviation across the CV folds.
    resource:
        The resource budget this trial ran at (Hyperband only; ``None``
        for RandomSearch).
    error:
        The exception message when the trial failed, else ``None``.
    state:
        Optuna's trial state — ``'complete'``, ``'pruned'`` or
        ``'fail'``; ``None`` (the legacy default) is derived on the
        fly by :attr:`resolved_state`.
    fold_scores:
        The per-fold objective values (the pruner loop's partial list
        for a pruned trial); ``None`` when the trial scored through the
        ``cross_val_score`` fast path.
    duration:
        Wall-clock seconds the trial took (sampling + evaluation);
        ``None`` when not recorded.

    Examples
    --------
    >>> from pharmaml.ml.tuner import TrialResult
    >>> TrialResult(1, {"units": 32}, 0.9, 0.02).resolved_state
    'complete'
    >>> TrialResult(2, {}, float("nan"), 0.0, error="ValueError: x").resolved_state
    'fail'
    >>> pruned = TrialResult(3, {}, float("nan"), float("nan"), state="pruned")
    >>> (pruned.resolved_state, pruned.pruned, pruned.failed)
    ('pruned', True, False)
    """

    trial_id: int
    hyperparameters: dict[str, Any]
    score: float
    score_std: float
    resource: float | None = None
    error: str | None = None
    state: str | None = None
    fold_scores: tuple[float, ...] | None = None
    duration: float | None = None

    @property
    def resolved_state(self) -> str:
        """The trial's state — the explicit one, else the derived.

        Derivation for legacy records (``state=None``): a trial with an
        error or a non-finite score is ``'fail'``, everything else
        ``'complete'``.
        """
        if self.state is not None:
            return self.state
        if self.error is not None or not math.isfinite(self.score):
            return "fail"
        return "complete"

    @property
    def failed(self) -> bool:
        """True when the trial resolved to the ``'fail'`` state.

        A PRUNED trial is not failed — it was cut short on purpose and
        keeps its partial :attr:`fold_scores`.
        """
        return self.resolved_state == "fail"

    @property
    def pruned(self) -> bool:
        """True when the pruner cut this trial short."""
        return self.resolved_state == "pruned"


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
        scored = [
            t
            for t in trials
            if t.resolved_state == "complete" and math.isfinite(t.score)
        ]
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
            (
                t
                for t in self.trials
                if t.resolved_state == "complete" and math.isfinite(t.score)
            ),
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
        pruned = sum(1 for t in self.trials if t.pruned)
        if pruned:
            lines.append(f"  ({pruned} trial(s) pruned by the pruner)")
        return "\n".join(lines)

    def trials_frame(self) -> pd.DataFrame:
        """The trials as a DataFrame (one row per trial, params as columns).

        Columns: the original ``trial_id / score / score_std / resource
        / failed`` block, then one column per hyperparameter, then the
        Optuna-style ``state`` / ``duration`` / ``n_folds`` tail.
        """
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
            row["state"] = trial.resolved_state
            row["duration"] = trial.duration
            row["n_folds"] = len(trial.fold_scores) if trial.fold_scores is not None else 0
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"<TunerResults: {len(self.trials)} trials, objective={self.objective!r}, "
            f"best_score={self.best_score:.4f}>"
        )


# ---------------------------------------------------------------------------
# Pruners (Optuna's report / should-prune fold loop)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PercentilePruner:
    """Prune trials whose running CV mean falls below a percentile.

    The Optuna ``PercentilePruner`` absorbed into the tuners' fold
    loop: after every fold the trial *reports* its running mean
    (Optuna's ``trial.report(value, step)``), and the pruner compares
    it against the completed trials' intermediate values AT THE SAME
    STEP — prune iff the current trial's best running mean is below
    the ``p``-th percentile of them.  The comparison runs on the
    higher-is-better sklearn scorer convention, the inversion of
    Optuna's default minimize direction (Optuna prunes when the
    running loss rises above the percentile; here it is the running
    score falling below it).

    Refusal guards (Optuna's own): fewer than ``n_startup_trials``
    completed trials -> never prune; fewer than ``n_min_trials``
    completed curves long enough to reach the current step -> never
    prune; an all-NaN same-step slice -> never prune (a percentile of
    noise is noise).

    Parameters
    ----------
    percentile:
        The percentile of the completed trials' same-step values the
        current trial's running best must beat, in ``(0, 100]``.
    n_startup_trials:
        Completed trials required before pruning is allowed (>= 1).
    n_min_trials:
        Same-step values required before pruning is allowed (>= 1).

    Raises
    ------
    ModelError
        For a percentile outside ``(0, 100]`` or a count below 1.

    Examples
    --------
    >>> from pharmaml.ml.tuner import MedianPruner, PercentilePruner
    >>> pruner = PercentilePruner(percentile=50.0, n_startup_trials=3)
    >>> pruner.should_prune([[0.9], [0.8]], [0.1])          # below startup
    False
    >>> pruner = PercentilePruner(percentile=50.0, n_startup_trials=2)
    >>> pruner.should_prune([[0.9], [0.8], [0.7]], [0.5])   # median 0.8
    True
    >>> pruner.should_prune([[0.9], [0.8], [0.7]], [0.85])
    False
    >>> MedianPruner(n_startup_trials=2).should_prune([[0.9], [0.8], [0.7]], [0.5])
    True
    >>> PercentilePruner(percentile=0.0)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.ModelError: percentile must be in (0, 100], got 0.0
    """

    percentile: float = 50.0
    n_startup_trials: int = 5
    n_min_trials: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < float(self.percentile) <= 100.0:
            raise ModelError(
                f"percentile must be in (0, 100], got {self.percentile!r}"
            )
        for name in ("n_startup_trials", "n_min_trials"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ModelError(f"{name} must be an integer >= 1, got {value!r}")

    def should_prune(
        self,
        completed_intermediates: Sequence[Sequence[float]],
        current_intermediates: Sequence[float],
    ) -> bool:
        """Whether the current trial should be cut short after a fold.

        Parameters
        ----------
        completed_intermediates:
            The completed trials' intermediate curves (running fold
            means) — one sequence per trial.
        current_intermediates:
            The current trial's running fold means so far; ``step`` is
            its length.
        """
        if not current_intermediates:
            return False
        if len(completed_intermediates) < int(self.n_startup_trials):
            return False
        step = len(current_intermediates)
        values = [
            float(curve[step - 1])
            for curve in completed_intermediates
            if len(curve) >= step
        ]
        if len(values) < int(self.n_min_trials):
            return False
        _LOG.debug(
            "pruner: step %d — %d same-step value(s), percentile %.1f",
            step,
            len(values),
            float(self.percentile),
        )
        threshold = float(np.nanpercentile(values, float(self.percentile)))
        if not math.isfinite(threshold):
            return False  # an all-NaN slice is not a signal
        return float(np.max(current_intermediates)) < threshold


def MedianPruner(n_startup_trials: int = 5, n_min_trials: int = 1) -> PercentilePruner:
    """The median pruner: ``PercentilePruner(50)``.

    Optuna's own definition — ``MedianPruner`` IS the percentile
    pruner at the 50th percentile; this factory mirrors it as the
    module-level shortcut, so ``MedianPruner()`` and
    ``PercentilePruner(percentile=50.0)`` make identical decisions.

    Parameters
    ----------
    n_startup_trials, n_min_trials:
        Same guards as :class:`PercentilePruner`.

    Examples
    --------
    >>> from pharmaml.ml.tuner import MedianPruner
    >>> MedianPruner().percentile
    50.0
    >>> MedianPruner(n_startup_trials=1).should_prune([[0.9], [0.7]], [0.5])
    True
    """
    return PercentilePruner(
        percentile=50.0,
        n_startup_trials=n_startup_trials,
        n_min_trials=n_min_trials,
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


def resource_bounds(hypermodel: HyperModel, resource: str) -> tuple[int, int]:
    """The declared ``[min, max]`` grid of a hypermodel's resource slot.

    The Hyperband ladder for a *shipped* hypermodel: the resource slot's
    own declared bounds ARE the ladder — ``max_resource`` is the grid's
    largest value (the full budget a plain ``Train`` runs at) and
    ``min_resource`` the smallest, so every rung resource stays inside
    the declared space.  The playground's Tune card uses this to size
    ``Hyperband`` without hardcoding the families' grids.

    Parameters
    ----------
    hypermodel:
        Callable ``(hp) -> estimator`` (e.g.
        :func:`build_ann_regressor`).
    resource:
        Name of the resource slot (e.g. ``'max_iter'``, ``'epochs'``).

    Returns
    -------
    tuple[int, int]
        ``(min_resource, max_resource)`` — the declared grid's first
        and last values.

    Raises
    ------
    ModelError
        When the hypermodel does not declare the slot, or declares it
        as anything other than an ``hp.Int`` grid (resources must be
        countable budgets).

    Examples
    --------
    >>> resource_bounds(build_ann_regressor, "max_iter")
    (200, 1000)
    >>> resource_bounds(build_hsvr, "epochs")
    Traceback (most recent call last):
        ...
    pharmaml.exceptions.ModelError: the hypermodel does not declare the resource hyperparameter 'epochs'
    """
    for spec in _collect_space(hypermodel):
        if spec.name == resource:
            if spec.kind != "int" or not spec.values:
                raise ModelError(
                    f"the resource hyperparameter {resource!r} must be an "
                    f"hp.Int slot with a discrete grid, got kind {spec.kind!r}"
                )
            return int(spec.values[0]), int(spec.values[-1])
    raise ModelError(
        f"the hypermodel does not declare the resource hyperparameter "
        f"{resource!r}"
    )


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


def _fold_slice(data: Any, indices: Any) -> Any:
    """Row-slice ``X`` / ``y`` for one fold (DataFrame/Series-aware)."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return np.asarray(data)[np.asarray(indices)]


def _intermediate_curve(fold_scores: Sequence[float]) -> list[float]:
    """Running means of the fold scores — the pruner's reported values.

    Optuna's ``trial.report(value, step)`` intermediate curve, in the
    toolkit's fold currency: after fold ``k`` the reported value is
    the mean of the first ``k`` fold scores.

        >>> from pharmaml.ml.tuner import _intermediate_curve
        >>> _intermediate_curve([0.5, 1.0, 1.5])
        [0.5, 0.75, 1.0]
    """
    return [float(np.nanmean(fold_scores[: i + 1])) for i in range(len(fold_scores))]


def _evaluate_trial_folds(
    hypermodel: HyperModel,
    values: dict[str, Any],
    X: Any,
    y: Any,
    scorer: Any,
    splitter: Any,
    pruner: PercentilePruner,
    completed_intermediates: list[list[float]],
) -> tuple[float, float, str | None, str, tuple[float, ...]]:
    """Build + fold-loop one config with pruning reports.

    The manual twin of :func:`_evaluate_trial` used when a pruner is
    wired in: iterate the splitter's folds, build a fresh estimator per
    fold, score with ``scorer(estimator, X_test, y_test)`` (the sklearn
    scorer call convention), and after every fold report the running
    mean to the pruner.  Returns
    ``(score, std, error, state, fold_scores)`` — a pruned trial comes
    back as ``(nan, nan, None, 'pruned', partial_folds)``; a fold that
    raises fails the whole trial with the error string.
    """
    fold_scores: list[float] = []
    try:
        for train_index, test_index in splitter.split(X, y):
            X_train = _fold_slice(X, train_index)
            X_test = _fold_slice(X, test_index)
            y_train = _fold_slice(y, train_index)
            y_test = _fold_slice(y, test_index)
            try:
                estimator = hypermodel(HyperParameters(values=values))
                estimator.fit(X_train, y_train)
                fold_scores.append(float(scorer(estimator, X_test, y_test)))
            except Exception as exc:  # noqa: BLE001 — a failed fold fails the trial
                return (
                    float("nan"),
                    float("nan"),
                    f"{type(exc).__name__}: {exc}",
                    "fail",
                    tuple(fold_scores),
                )
            intermediates = _intermediate_curve(fold_scores)
            _LOG.debug(
                "fold %d: score %.4f intermediate %.4f",
                len(fold_scores),
                fold_scores[-1],
                intermediates[-1],
            )
            if pruner.should_prune(completed_intermediates, intermediates):
                _LOG.debug(
                    "trial pruned after fold %d (running best %.4f)",
                    len(fold_scores),
                    intermediates[-1],
                )
                return float("nan"), float("nan"), None, "pruned", tuple(fold_scores)
    except Exception as exc:  # noqa: BLE001 — a broken splitter is a failed trial
        return (
            float("nan"),
            float("nan"),
            f"{type(exc).__name__}: {exc}",
            "fail",
            tuple(fold_scores),
        )
    if not fold_scores:
        return (
            float("nan"),
            float("nan"),
            "the splitter produced no folds",
            "fail",
            (),
        )
    return (
        float(np.nanmean(fold_scores)),
        float(np.nanstd(fold_scores)),
        None,
        "complete",
        tuple(fold_scores),
    )


def _validate_search_args(max_trials: Any, cv: Any, task: str) -> None:
    """The shared constructor guards (max_trials / cv / task)."""
    if (
        isinstance(max_trials, bool)
        or not isinstance(max_trials, (int, np.integer))
        or int(max_trials) < 1
    ):
        raise ModelError(f"max_trials must be an integer >= 1, got {max_trials!r}")
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


def _run_trials(
    *,
    hypermodel: HyperModel,
    space: list[_HyperParameterSpec],
    oracle: Callable[[list[TrialResult]], dict[str, Any]],
    X: Any,
    y: Any,
    scorer: Any,
    splitter: Any,
    max_trials: int,
    pruner: PercentilePruner | None,
    on_trial: Callable[[dict[str, Any]], None] | None,
    objective_label: str,
    label: str,
) -> TunerResults:
    """The trial loop shared by :class:`RandomSearch` and :class:`TPESearch`.

    ``oracle(trials_so_far)`` draws one trial's values (uniform for
    RandomSearch, the TPE below/above machinery for TPESearch).  With
    a pruner the trial is scored through the manual fold loop
    (:func:`_evaluate_trial_folds`); without one, the
    :func:`cross_val_score` fast path — unchanged behaviour for
    existing callers.  Every trial records its wall-clock duration,
    and the ``on_trial`` payload carries ``state`` and ``fold_scores``
    alongside the original keys.
    """
    trials: list[TrialResult] = []
    best: tuple[float, dict[str, Any]] | None = None  # (score, values)
    for trial_id in range(1, int(max_trials) + 1):
        started = perf_counter()
        values = oracle(trials)
        if pruner is None:
            score, std, error = _evaluate_trial(
                hypermodel, values, X, y, scorer, splitter
            )
            state: str | None = None
            fold_scores: tuple[float, ...] | None = None
        else:
            completed_curves = [
                _intermediate_curve(t.fold_scores)
                for t in trials
                if t.resolved_state == "complete"
                and t.fold_scores is not None
                and len(t.fold_scores) > 0
            ]
            score, std, error, state, fold_scores = _evaluate_trial_folds(
                hypermodel, values, X, y, scorer, splitter, pruner, completed_curves
            )
        trial = TrialResult(
            trial_id=trial_id,
            hyperparameters=values,
            score=score,
            score_std=std,
            error=error,
            state=state,
            fold_scores=fold_scores,
            duration=perf_counter() - started,
        )
        _LOG.debug(
            "trial %d/%d: score %.4f %s", trial_id, max_trials, score, values
        )
        trials.append(trial)
        if error is None and np.isfinite(score) and (best is None or score > best[0]):
            best = (score, values)
        if on_trial is not None:
            on_trial({
                "trial": trial_id,
                "total": int(max_trials),
                "score": score,
                "best_score": best[0] if best is not None else None,
                "error": error,
                "hyperparameters": values,
                "state": trial.resolved_state,
                "fold_scores": trial.fold_scores,
            })

    results = TunerResults(trials, objective_label, label=label)
    if best is not None:
        # Refit the winning configuration on the full data.
        winner = HyperParameters(values=best[1])
        results._best_estimator = hypermodel(winner).fit(X, y)
    return results


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
    on_trial:
        Optional progress callback invoked after every trial with the
        payload ``{'trial': i, 'total': max_trials, 'score',
        'best_score', 'error', 'hyperparameters', 'state',
        'fold_scores'}`` — one call per sampled trial, in trial order
        (``best_score`` is ``None`` until a trial succeeds; ``state``
        is the trial's resolved state and ``fold_scores`` its per-fold
        objective values, ``None`` unless a pruner is wired in).
        Designed for GUI progress bars and ETA displays; it must never
        raise.
    pruner:
        Optional :class:`PercentilePruner` — with one, trials are
        scored through the manual fold loop and cut short
        (``state='pruned'``) when their running fold mean falls below
        the pruner's percentile of the completed trials at the same
        fold.  ``None`` (the default) keeps the
        :func:`sklearn.model_selection.cross_val_score` fast path —
        byte-identical behaviour for existing callers.

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
        on_trial: Callable[[dict[str, Any]], None] | None = None,
        pruner: PercentilePruner | None = None,
    ) -> None:
        self.hypermodel = hypermodel
        self.objective = objective
        self.max_trials = max_trials
        self.cv = cv
        self.task = task
        self.seed = seed
        self.on_trial = on_trial
        self.pruner = pruner

    def _validate(self) -> None:
        _validate_search_args(self.max_trials, self.cv, self.task)
        if self.pruner is not None and not isinstance(self.pruner, PercentilePruner):
            raise ModelError(
                "pruner must be a PercentilePruner, got "
                f"{type(self.pruner).__name__}"
            )

    def search(self, X: Any, y: Any) -> TunerResults:
        """Run the search; returns :class:`TunerResults`.

        The winning estimator is rebuilt from ``best_params`` and fitted
        on the full ``(X, y)`` — available as ``results.best_estimator``.
        With a ``pruner``, trials whose running fold mean falls below
        the pruner's threshold are cut short and recorded with
        ``state='pruned'`` (and their partial fold scores).
        """
        self._validate()
        scorer, objective_label = _resolve_objective(self.objective)
        space = _collect_space(self.hypermodel)
        splitter = make_cv_splitter(self.task, y, int(self.cv))
        rng = np.random.default_rng(int(self.seed))

        def oracle(_trials: list[TrialResult]) -> dict[str, Any]:
            return {spec.name: spec.sample(rng) for spec in space}

        return _run_trials(
            hypermodel=self.hypermodel,
            space=space,
            oracle=oracle,
            X=X,
            y=y,
            scorer=scorer,
            splitter=splitter,
            max_trials=int(self.max_trials),
            pruner=self.pruner,
            on_trial=self.on_trial,
            objective_label=objective_label,
            label="RandomSearch",
        )


# ---------------------------------------------------------------------------
# TPESearch — Optuna's Tree-structured Parzen Estimator
# ---------------------------------------------------------------------------
def _gamma(n: int, cap: int = 25) -> int:
    """TPE's below-set size: ``gamma(n) = min(ceil(0.1 * n), cap)``.

    Optuna's ``default_gamma`` — the number of best trials the below
    mixture is fitted to::

        >>> from pharmaml.ml.tuner import _gamma
        >>> [_gamma(n) for n in (0, 1, 9, 10, 11, 60, 250)]
        [0, 1, 1, 1, 2, 6, 25]

    The cap clamps the below set for large ``n`` (Optuna pins it at
    25; ``gamma_cap`` lets a search shrink it further).
    """
    return min(math.ceil(0.1 * n), int(cap))


def _below_above_split(
    completed: Sequence[TrialResult], gamma_n: int
) -> tuple[list[TrialResult], list[TrialResult]]:
    """Split the completed trials into the below (best) and above sets.

    Higher-is-better, ties resolved towards the earlier trial (Python's
    stable sort over the evaluation order) — Optuna's
    ``_split_complete_trials`` rule; both sets come back in trial
    order (Optuna sorts its split the same way)::

        >>> from pharmaml.ml.tuner import TrialResult, _below_above_split
        >>> trials = [TrialResult(i, {}, float(i), 0.0) for i in (1, 2, 3)]
        >>> below, above = _below_above_split(trials, 1)
        >>> [t.trial_id for t in below], [t.trial_id for t in above]
        ([3], [1, 2])
        >>> tied = [TrialResult(i, {}, 1.0, 0.0) for i in (1, 2)]
        >>> below, _ = _below_above_split(tied, 1)
        >>> below[0].trial_id          # the tie keeps the earlier trial
        1
    """
    ranked = sorted(completed, key=lambda t: t.score, reverse=True)
    below = sorted(ranked[:gamma_n], key=lambda t: t.trial_id)
    above = sorted(ranked[gamma_n:], key=lambda t: t.trial_id)
    return below, above


def _parzen_kernels(
    observations: Sequence[float], low: float, high: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one Parzen (KDE) mixture over ``[low, high]``.

    Optuna's ``_ParzenEstimator`` recipe, univariate form: one normal
    kernel per observation plus ONE prior kernel at the domain centre
    (``sigma = high - low``, weight 1 — ``consider_prior`` with
    ``prior_weight=1``).  The observation bandwidths are the sorted
    neighbour gaps with the array padded at the ends by ``low`` /
    ``high`` (the endpoint rule), clipped by the magic clip to
    ``[(high-low)/min(100, 1+n_kernels), high-low]`` where
    ``n_kernels`` counts observations plus the prior.  Weights are
    uniform over the observations plus the prior, normalized.  A
    zero-observation mixture (the startup edge) is the prior kernel
    alone.

    Returns ``(mus, sigmas, weights)``.
    """
    span = high - low
    mus = np.sort(np.asarray(observations, dtype=float))
    k = int(mus.size)
    if k == 0:
        return (
            np.array([(low + high) / 2.0]),
            np.array([span]),
            np.array([1.0]),
        )
    padded = np.concatenate(([low], mus, [high]))
    sigmas = np.maximum(padded[1:-1] - padded[:-2], padded[2:] - padded[1:-1])
    n_kernels = k + 1  # + 1 for the prior kernel (Optuna's own count)
    sigmas = np.clip(sigmas, span / min(100, 1 + n_kernels), span)
    mus = np.append(mus, (low + high) / 2.0)
    sigmas = np.append(sigmas, span)
    weights = np.full(k + 1, 1.0)
    return mus, sigmas, weights / weights.sum()


def _parzen_logpdf(
    x: float, mus: np.ndarray, sigmas: np.ndarray, weights: np.ndarray
) -> float:
    """``log sum_k w_k * N(x; mu_k, sigma_k)`` — one mixture's log-density."""
    logs = (
        np.log(weights)
        - np.log(np.sqrt(2.0 * np.pi) * sigmas)
        - 0.5 * ((x - mus) / sigmas) ** 2
    )
    peak = float(np.max(logs))
    if peak == -math.inf:
        return -math.inf
    return peak + float(np.log(np.sum(np.exp(logs - peak))))


def _categorical_weights(
    observations: Sequence[Any], values: Sequence[Any]
) -> np.ndarray:
    """Smoothed categorical weights: counts plus a uniform prior.

    Each observed value adds weight 1 to its slot, the prior weight
    1.0 is spread uniformly over the choices, and the vector is
    normalized — a zero-observation mixture is the uniform
    distribution (Optuna's smoothed categorical counts).
    """
    counts = np.zeros(len(values), dtype=float)
    for observed in observations:
        for index, choice in enumerate(values):
            if choice == observed:
                counts[index] += 1.0
                break
    weights = counts + 1.0 / len(values)
    return weights / weights.sum()


def _tpe_sample_categorical(
    values: tuple[Any, ...],
    below_obs: Sequence[Any],
    above_obs: Sequence[Any],
    rng: np.random.Generator,
    n_candidates: int,
) -> Any:
    """Draw one categorical value with the TPE below/above rule.

    Candidates are drawn from the below smoothed counts; each is scored
    by ``log p_below - log p_above`` (a degenerate above set with no
    observations contributes 0 — its uniform log-density is constant
    and would not change the argmax); the argmax wins (ties keep the
    first candidate).
    """
    below_weights = _categorical_weights(below_obs, values)
    above_weights = _categorical_weights(above_obs, values)
    best_index, best_score = 0, -math.inf
    for _ in range(n_candidates):
        index = int(rng.choice(len(values), p=below_weights))
        score = math.log(below_weights[index])
        if above_obs:
            score -= math.log(above_weights[index])
        if score > best_score:
            best_index, best_score = index, score
    return values[best_index]


def _tpe_sample_numeric(
    spec: _HyperParameterSpec,
    below_obs: Sequence[float],
    above_obs: Sequence[float],
    rng: np.random.Generator,
    n_candidates: int,
) -> Any:
    """Draw one int/float value with the Parzen below/above machinery.

    The working domain follows Optuna: plain floats stay in ``[low,
    high]`` (the log domain when ``spec.log`` — observations and bounds
    transformed with ``log``, draws back with ``exp``); stepped floats
    widen the domain by ``+/- step/2`` and round drawn candidates back
    onto the step grid; ints are continuous over ``[low - 0.5,
    high + 0.5]`` and rounded.  Each candidate is drawn from the below
    mixture (component by weights, then a normal clipped to the domain
    bounds — the honest clipped-normal truncation, not Optuna's true
    truncated normal) and scored by ``log pdf_below - log pdf_above``
    evaluated at the candidate's own value; the argmax wins (ties keep
    the first candidate).  A degenerate above set with no observations
    scores 0 for every candidate, i.e. the acquisition reduces to
    ``log pdf_below``.
    """
    log_mode = spec.kind == "float" and bool(spec.log)
    step: float | None = None
    low, high = float(spec.low), float(spec.high)
    if spec.kind == "int":
        # Int slots declare a discrete grid, not low/high bounds — the
        # grid's endpoints ARE the declared [min, max].
        if not spec.values:
            return 0
        low, high = float(spec.values[0]), float(spec.values[-1])
        if high <= low:
            return int(low)
        work_low, work_high = low - 0.5, high + 0.5
        below_work = [float(v) for v in below_obs]
        above_work = [float(v) for v in above_obs]
    elif log_mode:
        if high <= low or low <= 0.0:
            return low
        work_low, work_high = math.log(low), math.log(high)
        below_work = [math.log(float(v)) for v in below_obs]
        above_work = [math.log(float(v)) for v in above_obs]
    else:
        if high <= low:
            return low
        if spec.kind == "float" and spec.step is not None:
            step = float(spec.step)
            work_low, work_high = low - step / 2.0, high + step / 2.0
        else:
            work_low, work_high = low, high
        below_work = [float(v) for v in below_obs]
        above_work = [float(v) for v in above_obs]

    def candidate(x: float) -> tuple[int | float, float]:
        """A clipped draw -> ``(param value, working-domain coordinate)``."""
        if spec.kind == "int":
            int_value = int(min(max(round(x), low), high))
            return int_value, float(int_value)
        if log_mode:
            log_value = float(min(max(math.exp(x), low), high))
            return log_value, math.log(log_value)
        plain = float(min(max(x, low), high))
        if step is not None:
            n_steps = round((plain - low) / step)
            plain = float(min(max(low + n_steps * step, low), high))
        return plain, plain

    below_mus, below_sigmas, below_weights = _parzen_kernels(
        below_work, work_low, work_high
    )
    above_mus, above_sigmas, above_weights = _parzen_kernels(
        above_work, work_low, work_high
    )
    best_value: int | float | None = None
    best_score = -math.inf
    for _ in range(n_candidates):
        index = int(rng.choice(len(below_mus), p=below_weights))
        draw = float(
            min(max(rng.normal(below_mus[index], below_sigmas[index]), work_low), work_high)
        )
        value, coordinate = candidate(draw)
        score = _parzen_logpdf(coordinate, below_mus, below_sigmas, below_weights)
        if above_work:
            score -= _parzen_logpdf(
                coordinate, above_mus, above_sigmas, above_weights
            )
        if score > best_score:
            best_value, best_score = value, score
    if best_value is None:  # pragma: no cover - n_candidates validated >= 1
        return low
    return best_value


def _tpe_sample_param(
    spec: _HyperParameterSpec,
    below_obs: Sequence[Any],
    above_obs: Sequence[Any],
    rng: np.random.Generator,
    n_candidates: int,
) -> Any:
    """Route one slot through the TPE machinery (fixed -> its constant)."""
    if spec.kind == "fixed":
        return spec.values[0]
    if spec.kind in ("choice", "boolean"):
        return _tpe_sample_categorical(
            spec.values, below_obs, above_obs, rng, n_candidates
        )
    return _tpe_sample_numeric(spec, below_obs, above_obs, rng, n_candidates)


def _tpe_values(
    space: Sequence[_HyperParameterSpec],
    trials: Sequence[TrialResult],
    rng: np.random.Generator,
    n_startup_trials: int,
    n_ei_candidates: int,
    gamma_cap: int,
) -> dict[str, Any]:
    """Draw one trial's values with the TPE below/above rule.

    ``n = len(completed) + len(pruned)`` — failed trials never inform
    the sampler (Optuna's own count).  Below ``n_startup_trials``
    every parameter is a uniform ``spec.sample(rng)`` draw — the honest
    bootstrap, and byte-identical to :class:`RandomSearch`'s sequence
    for the same seed.  From there on: ``below`` is the top
    ``gamma(n) = min(ceil(0.1 n), gamma_cap)`` completed trials by
    score, ``above`` every other completed trial (pruned and failed
    trials are excluded from both mixtures — the honest scope; Optuna
    fills its below set with pruned trials once the complete ones run
    out), and each parameter is drawn through the Parzen machinery.
    """
    completed = [t for t in trials if t.resolved_state == "complete"]
    pruned = [t for t in trials if t.resolved_state == "pruned"]
    n = len(completed) + len(pruned)
    if n < n_startup_trials:
        _LOG.debug(
            "tpe trial %d: startup (n=%d < n_startup_trials=%d) — uniform draw",
            len(trials) + 1,
            n,
            n_startup_trials,
        )
        return {spec.name: spec.sample(rng) for spec in space}
    gamma_n = _gamma(n, gamma_cap)
    below, above = _below_above_split(completed, gamma_n)
    _LOG.debug(
        "tpe trial %d: n=%d gamma=%d — below=%d above=%d",
        len(trials) + 1,
        n,
        gamma_n,
        len(below),
        len(above),
    )
    values: dict[str, Any] = {}
    for spec in space:
        below_obs = [t.hyperparameters[spec.name] for t in below]
        above_obs = [t.hyperparameters[spec.name] for t in above]
        values[spec.name] = _tpe_sample_param(
            spec, below_obs, above_obs, rng, n_ei_candidates
        )
    return values


class TPESearch:
    """Tree-structured Parzen Estimator search — Optuna's flagship sampler.

    The ``TPESampler`` programming model absorbed into the house
    ``hypermodel(hp) -> estimator`` contract (no optuna dependency,
    exactly the re-implementation move the Keras-Tuner round made):
    every trial splits the informative trials into a small ``below``
    (best) set of size ``gamma(n) = min(ceil(0.1 n), gamma_cap)`` and
    the ``above`` rest, fits one Parzen (KDE) mixture per
    hyperparameter to each set — the observation-bandwidth / prior
    kernel / magic-clip recipe documented on
    :func:`_parzen_kernels` — draws ``n_ei_candidates`` candidates
    from the below mixture and keeps the argmax of
    ``log pdf_below - log pdf_above`` (the expected-improvement
    surrogate).  The first ``n_startup_trials`` informative trials are
    plain uniform draws.  ``n`` counts COMPLETE + PRUNED trials —
    failed trials never inform the sampler (Optuna's own rule).

    Honest deviations from Optuna (one line each, all recorded in
    ``scratch/shap_optuna_review.md`` §2):

    * sampling is independent per parameter — no multivariate KDE,
      ``group`` decomposition or ``constant_liar`` (the toolkit's
      categorical-rich spaces make the univariate Parzen the honest
      scope);
    * pruned trials are EXCLUDED from both mixtures (Optuna fills the
      below set with pruned trials after the complete ones are used
      up);
    * candidates are clipped normals, not true truncated normals (the
      clip is the documented simplification of Optuna's truncation).

    Same ``search(X, y) -> TunerResults`` contract as
    :class:`RandomSearch`: the winning estimator is rebuilt from
    ``best_params`` and refitted on the full data, progress callbacks
    fire per trial, and two searches with the same seed and data
    produce identical trial sequences.

    Parameters
    ----------
    hypermodel, objective, max_trials, cv, task, seed, on_trial, pruner:
        Same contract as :class:`RandomSearch`.
    n_startup_trials:
        Informative-trial count before the Parzen machinery takes over
        (integer >= 0; the first trials are uniform draws).
    n_ei_candidates:
        Candidates drawn from the below mixture per parameter per
        trial (integer >= 1; Optuna's default 24).
    gamma_cap:
        Upper bound on the below-set size ``gamma(n)`` (integer >= 1;
        Optuna's default cap is 25).

    Raises
    ------
    ModelError
        For invalid constructor values, an unknown objective, a
        hypermodel that does not return a fittable estimator, or when
        every trial fails.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.tuner import RandomSearch, TPESearch, build_ann_regressor
    >>> rng = np.random.default_rng(7)
    >>> X = rng.uniform(0, 1, (60, 3))
    >>> y = 3.0 * X[:, 0] + 1.0
    >>> tuner = TPESearch(build_ann_regressor, objective="r2",
    ...                   max_trials=5, cv=2, seed=11, n_startup_trials=2)
    >>> results = tuner.search(X, y)
    >>> results.best_score > 0.9         # a linear surface is learnable
    True
    >>> results.summary().splitlines()[0]
    'TPESearch: 5 trial(s), objective=r2'
    >>> twin = TPESearch(build_ann_regressor, objective="r2",
    ...                 max_trials=5, cv=2, seed=11,
    ...                 n_startup_trials=2).search(X, y)
    >>> twin.best_params == results.best_params   # deterministic
    True
    >>> rand = RandomSearch(build_ann_regressor, max_trials=2, cv=2,
    ...                     seed=5).search(X, y)
    >>> tpe = TPESearch(build_ann_regressor, max_trials=2, cv=2, seed=5,
    ...                 n_startup_trials=10).search(X, y)
    >>> [t.hyperparameters for t in tpe.trials] == [t.hyperparameters for t in rand.trials]
    True
    >>> TPESearch(build_ann_regressor, max_trials=0).search(X, y)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.ModelError: max_trials must be an integer >= 1, got 0
    >>> TPESearch(build_ann_regressor, n_ei_candidates=0).search(X, y)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.ModelError: n_ei_candidates must be an integer >= 1, got 0
    """

    def __init__(
        self,
        hypermodel: HyperModel,
        objective: str | Callable[..., float] = "r2",
        max_trials: int = 10,
        cv: int = 3,
        task: str = "auto",
        seed: int = 42,
        on_trial: Callable[[dict[str, Any]], None] | None = None,
        pruner: PercentilePruner | None = None,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma_cap: int = 25,
    ) -> None:
        self.hypermodel = hypermodel
        self.objective = objective
        self.max_trials = max_trials
        self.cv = cv
        self.task = task
        self.seed = seed
        self.on_trial = on_trial
        self.pruner = pruner
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma_cap = gamma_cap

    def _validate(self) -> None:
        _validate_search_args(self.max_trials, self.cv, self.task)
        if self.pruner is not None and not isinstance(self.pruner, PercentilePruner):
            raise ModelError(
                "pruner must be a PercentilePruner, got "
                f"{type(self.pruner).__name__}"
            )
        if (
            isinstance(self.n_startup_trials, bool)
            or not isinstance(self.n_startup_trials, (int, np.integer))
            or int(self.n_startup_trials) < 0
        ):
            raise ModelError(
                f"n_startup_trials must be an integer >= 0, "
                f"got {self.n_startup_trials!r}"
            )
        if (
            isinstance(self.n_ei_candidates, bool)
            or not isinstance(self.n_ei_candidates, (int, np.integer))
            or int(self.n_ei_candidates) < 1
        ):
            raise ModelError(
                f"n_ei_candidates must be an integer >= 1, "
                f"got {self.n_ei_candidates!r}"
            )
        if (
            isinstance(self.gamma_cap, bool)
            or not isinstance(self.gamma_cap, (int, np.integer))
            or int(self.gamma_cap) < 1
        ):
            raise ModelError(f"gamma_cap must be an integer >= 1, got {self.gamma_cap!r}")

    def search(self, X: Any, y: Any) -> TunerResults:
        """Run the search; returns :class:`TunerResults`.

        The winning estimator is rebuilt from ``best_params`` and fitted
        on the full ``(X, y)`` — available as ``results.best_estimator``.
        With a ``pruner``, trials whose running fold mean falls below
        the pruner's threshold are cut short and recorded with
        ``state='pruned'`` (and their partial fold scores).
        """
        self._validate()
        scorer, objective_label = _resolve_objective(self.objective)
        space = _collect_space(self.hypermodel)
        splitter = make_cv_splitter(self.task, y, int(self.cv))
        rng = np.random.default_rng(int(self.seed))

        def oracle(trials: list[TrialResult]) -> dict[str, Any]:
            return _tpe_values(
                space,
                trials,
                rng,
                int(self.n_startup_trials),
                int(self.n_ei_candidates),
                int(self.gamma_cap),
            )

        return _run_trials(
            hypermodel=self.hypermodel,
            space=space,
            oracle=oracle,
            X=X,
            y=y,
            scorer=scorer,
            splitter=splitter,
            max_trials=int(self.max_trials),
            pruner=self.pruner,
            on_trial=self.on_trial,
            objective_label=objective_label,
            label="TPESearch",
        )


# ---------------------------------------------------------------------------
# Hyperband
# ---------------------------------------------------------------------------
def _hyperband_total_trials(eta: int, r_max: int, r_min: int) -> int:
    """Exact evaluation count of the full Hyperband bracket schedule.

    Mirrors :meth:`Hyperband.search`'s own arithmetic (bracket widths
    and the ``ceil(n / eta)`` rung shrink) without touching a
    hypermodel, so progress consumers can show ``trial / total`` and
    an ETA before the first trial finishes::

        >>> from pharmaml.ml.tuner import _hyperband_total_trials
        >>> _hyperband_total_trials(3, 81, 3)   # the classic textbook grid
        70

    The count assumes no bracket aborts early (every rung producing
    at least one scored config); a doomed bracket stops short and the
    real trial count lands below this figure.
    """
    s_max = math.floor(math.log(r_max / r_min, eta)) if r_max > r_min else 0
    total = 0
    for bracket in range(s_max + 1):
        configs = math.ceil((s_max + 1) * eta**bracket / (bracket + 1))
        for _rung in range(bracket + 1):
            total += configs
            configs = math.ceil(configs / eta)
    return total


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
    on_trial:
        Optional progress callback invoked after every rung evaluation
        with the payload ``{'trial': i, 'total': n, 'score',
        'best_score', 'error', 'resource'}`` — ``total`` is the exact
        evaluation count of the full bracket schedule (mirrored by
        :func:`_hyperband_total_trials`), so ``i / total`` is a honest
        progress fraction for ETA displays.  A rung where every config
        fails ends its bracket early, leaving ``total`` an overestimate
        for that (already doomed) search.  The callback must never
        raise.

    Raises
    ------
    ModelError
        For invalid constructor values, when the hypermodel does not
        declare the ``resource`` slot, or when every trial fails.

    See Also
    --------
    resource_bounds:
        The declared grid of a shipped hypermodel's resource slot —
        the ``(min_resource, max_resource)`` pair that keeps every
        rung inside the family's declared space.
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
        on_trial: Callable[[dict[str, Any]], None] | None = None,
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
        self.on_trial = on_trial

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
        total_trials = _hyperband_total_trials(eta, r_max, r_min)

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
                    if self.on_trial is not None:
                        self.on_trial({
                            "trial": trial_counter,
                            "total": total_trials,
                            "score": score,
                            "best_score": best[0] if best is not None else None,
                            "error": error,
                            "resource": float(rung_resource),
                        })
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
# Hyperparameter importances (Optuna's MDI evaluator)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HyperparameterImportance:
    """Per-hyperparameter importances of a finished search.

    The typed record :func:`param_importances` returns — the random
    forest's mean-decrease-in-impurity attribution, summed back per
    original hyperparameter (one-hot groups collapse onto their param)
    and normalized to sum 1.

    Attributes
    ----------
    names:
        The hyperparameter names, in the trials' declaration order.
    importances:
        Normalised importance per name.  A param constant across the
        analysed trials is reported with ``0.0`` — excluded from the
        forest, honestly unmeasured rather than silently dropped.
    n_trials:
        The number of completed trials the forest was fitted on.
    method:
        The evaluator — always ``'mdi'`` (mean decrease in impurity).
    objective:
        The search objective label the importances explain.

    Examples
    --------
    >>> from pharmaml.ml.tuner import HyperparameterImportance
    >>> imp = HyperparameterImportance(
    ...     names=("lr", "depth"), importances=(0.8, 0.2), n_trials=6,
    ...     objective="r2")
    >>> imp.ranking()
    [('lr', 0.8), ('depth', 0.2)]
    >>> imp.ranking(top_n=1)
    [('lr', 0.8)]
    >>> imp.as_frame()["parameter"].tolist()
    ['lr', 'depth']
    >>> print(imp)
    hyperparameter importances (mdi, objective=r2, n=6): lr=0.800, depth=0.200
    """

    names: tuple[str, ...]
    importances: tuple[float, ...]
    n_trials: int
    method: str = "mdi"
    objective: str = ""

    def as_frame(self) -> pd.DataFrame:
        """The importances as a DataFrame (parameter, importance)."""
        return pd.DataFrame(
            {
                "parameter": list(self.names),
                "importance": np.asarray(self.importances, dtype=float),
            }
        )

    def ranking(self, top_n: int | None = None) -> list[tuple[str, float]]:
        """``[(parameter, importance), ...]`` descending by importance.

        Ties resolve alphabetically by name (deterministic ordering).
        """
        order = sorted(
            range(len(self.names)),
            key=lambda i: (-self.importances[i], self.names[i]),
        )
        keep = len(order) if top_n is None else min(int(top_n), len(order))
        return [
            (self.names[i], float(self.importances[i])) for i in order[:keep]
        ]

    def __str__(self) -> str:
        top = self.ranking(top_n=5)
        rendered = ", ".join(f"{name}={value:.3f}" for name, value in top)
        return (
            f"hyperparameter importances ({self.method}, "
            f"objective={self.objective}, n={self.n_trials}): {rendered}"
        )


def param_importances(
    results: TunerResults, *, n_trees: int = 64, seed: int = 42
) -> HyperparameterImportance:
    """Random-forest (MDI) hyperparameter importances of a finished search.

    Optuna's ``get_param_importances`` with the
    ``MeanDecreaseImpurityImportanceEvaluator`` — the evaluator Optuna's
    own docstring spells out in one paragraph: fit a
    :class:`sklearn.ensemble.RandomForestRegressor` on the (encoded
    hyperparameters -> objective score) table of the COMPLETED trials
    and read ``feature_importances_``.  Numeric params (int/float)
    become one column each; choice/boolean/fixed params become one-hot
    columns (``value == choice -> 1.0``), and the one-hot groups are
    summed back onto their original param with ``np.add.at`` (Optuna's
    own recipe).  Importances are normalized to sum 1.0.

    Two honest notes, both Optuna's own behaviour: at least two
    completed trials are required (a one-point forest measures nothing),
    and a param constant across the trials is excluded from the forest
    but still REPORTED, with importance ``0.0``.  The degenerate
    all-zero case (nothing was measurable at all) falls back to the
    uniform distribution over the names.

    Parameters
    ----------
    results:
        A finished search's :class:`TunerResults`.
    n_trees:
        Trees in the importance forest (default 64, Optuna's own).
    seed:
        The forest's seed — the evaluation is deterministic for a fixed
        seed.

    Returns
    -------
    HyperparameterImportance

    Raises
    ------
    ModelError
        When fewer than two trials resolved ``'complete'``.

    Examples
    --------
    >>> from pharmaml.ml.tuner import TrialResult, TunerResults, param_importances
    >>> trials = [
    ...     TrialResult(1, {"lr": 0.10, "depth": 2}, 0.10, 0.0),
    ...     TrialResult(2, {"lr": 0.30, "depth": 1}, 0.30, 0.0),
    ...     TrialResult(3, {"lr": 0.50, "depth": 2}, 0.50, 0.0),
    ...     TrialResult(4, {"lr": 0.70, "depth": 1}, 0.70, 0.0),
    ...     TrialResult(5, {"lr": 0.90, "depth": 2}, 0.90, 0.0),
    ... ]
    >>> importance = param_importances(TunerResults(trials, "r2"))
    >>> importance.ranking()[0][0]           # lr drives the score exactly
    'lr'
    >>> importance.importances[importance.names.index("lr")] > 0.5
    True
    >>> importance.n_trials, importance.method
    (5, 'mdi')
    >>> param_importances(TunerResults(trials[:1], "r2"))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.ModelError: param_importances needs at least 2 completed trials, got 1 (Optuna's own rule)
    """
    completed = [
        t
        for t in results.trials
        if t.resolved_state == "complete" and math.isfinite(t.score)
    ]
    if len(completed) < 2:
        raise ModelError(
            f"param_importances needs at least 2 completed trials, "
            f"got {len(completed)} (Optuna's own rule)"
        )
    names: list[str] = []
    for trial in completed:
        for name in trial.hyperparameters:
            if name not in names:
                names.append(name)

    encoded: list[np.ndarray] = []
    groups: list[int] = []  # encoded column -> index into names
    for index, name in enumerate(names):
        values = [t.hyperparameters.get(name) for t in completed]
        numeric = all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, bool)
            for v in values
        )
        if numeric:
            column = np.asarray([float(v) for v in values])  # type: ignore[arg-type]
            if float(column.max()) == float(column.min()):
                continue  # constant — reported at 0.0, excluded from the forest
            encoded.append(column)
            groups.append(index)
        else:
            choices = list(dict.fromkeys(v for v in values if v is not None))
            if len(choices) <= 1:
                continue  # constant — reported at 0.0, excluded from the forest
            for choice in choices:
                encoded.append(
                    np.asarray([1.0 if v == choice else 0.0 for v in values])
                )
                groups.append(index)

    importances = np.zeros(len(names))
    if encoded:
        forest = RandomForestRegressor(
            n_estimators=int(n_trees), random_state=int(seed)
        )
        forest.fit(
            np.column_stack(encoded),
            np.asarray([float(t.score) for t in completed]),
        )
        # Sum the one-hot groups back onto their original param —
        # Optuna's own np.add.at recipe.
        np.add.at(importances, np.asarray(groups), forest.feature_importances_)
    total = float(importances.sum())
    if total > 0.0:
        importances = importances / total
    elif names:
        importances = np.full(len(names), 1.0 / len(names))
    _LOG.debug(
        "param_importances: %d completed trial(s), %d encoded column(s)",
        len(completed),
        len(encoded),
    )
    return HyperparameterImportance(
        names=tuple(names),
        importances=tuple(float(v) for v in importances),
        n_trials=len(completed),
        objective=results.objective,
    )


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
