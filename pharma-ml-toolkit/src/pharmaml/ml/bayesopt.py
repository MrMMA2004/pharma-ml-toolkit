"""keras-tuner's Bayesian oracle: the GP surrogate and its acquisition.

``keras-team/keras-tuner`` was reviewed file by file for this module:
the engine (``base_tuner.py``, ``oracle.py``, ``trial.py``,
``hyperparameters.py`` -- the trial/oracle protocol), the tuners
directory (``randomsearch.py``, ``bayesian.py``, ``hyperband.py``,
``gridsearch.py``, ``sklearn_tuner.py``), the distributions
(``backend``/``protos``) and the docs tree.  The toolkit's tuner
already carries three of the four oracles -- RandomSearch, grid, the
TPE (Optuna's flagship) and Hyperband -- and what the review found
missing is exactly the fourth: ``BayesianOptimizationOracle``, the
Gaussian-process surrogate that picks each next trial.

What that oracle is, stripped of the Keras graph: a GP regression
over the (hyperparameters → score) pairs seen so far, and an
acquisition function that balances exploitation (low predicted mean)
against exploration (high posterior uncertainty).  keras-tuner's own
choices land here verbatim -- the RBF kernel, the diagonal jitter
``alpha=1e-4``, the exploration factor ``beta=2.6``, the
``num_initial_points = 3 × dimensions`` rule for the random warm-up,
and the acquisition the docstring names UCB (for the tuner's
minimisation semantics it is the lower confidence bound
``mean − β·σ`` that actually drives the pick -- stated, not hidden).

* :class:`GaussianProcessSurrogate` -- the RBF-kernel GP: fit on
  (X, y), posterior mean and std at candidates, the length-scale
  median heuristic.
* :func:`lower_confidence_bound` / :func:`expected_improvement` --
  the two acquisition functions (UCB's minimisation form, and EI --
  the classic the oracle's docstring contrasts against).
* :func:`bayesian_search` -- the search itself: seeded Latin-hypercube
  warm-up, then GP-guided picks over a dense candidate grid, the best
  trial tracked oracle-style, and a diagnostics report.

The honest scope: the Keras ``HyperModel``/``Tuner`` machinery, the
distributed execution tiers and the HyperParameters DSL stay with the
Keras engine; what lands is the oracle's mathematics -- the part a
NumPy toolkit can run exactly.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.bayesopt import (
...     GaussianProcessSurrogate, bayesian_search, expected_improvement,
...     lower_confidence_bound,
... )
>>> def objective(x: np.ndarray) -> float:      # a 1-D valley to minimise
...     return float((x[0] - 0.3) ** 2 + 0.1)
>>> bounds = [("learning_rate", 0.001, 1.0)]
>>> report = bayesian_search(objective, bounds, n_trials=12, seed=42)
>>> report.n_trials
12
>>> 0.05 < report.best_x["learning_rate"] < 0.7
True
>>> report.best_score < 0.2                     # the GP found the valley
True
>>> rng = np.random.default_rng(0)
>>> X = rng.uniform(size=(8, 2))
>>> y = np.array([np.sin(row[0]) + row[1] for row in X])
>>> gp = GaussianProcessSurrogate.fit(X, y, seed=42)
>>> mean, std = gp.posterior(np.array([[0.5, 0.5]]))
>>> 0.4 < float(mean[0]) < 1.4      # interpolating the sin()+x surface
True
>>> lcb = lower_confidence_bound(gp, np.array([[0.5, 0.5]]), beta=2.6)
>>> float(lcb[0]) < float(mean[0])
True
>>> ei = expected_improvement(gp, np.array([[0.5, 0.5]]), best=float(y.min()))
>>> float(ei[0]) >= 0.0
True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed threaded through the warm-up and the candidate grid (the pin).
RANDOM_STATE = 42

#: keras-tuner's default exploration factor (bayesian.py: beta=2.6).
DEFAULT_BETA = 2.6

#: keras-tuner's default diagonal jitter (bayesian.py: alpha=1e-4).
DEFAULT_ALPHA = 1e-4

#: keras-tuner's warm-up rule: 3 random points per dimension.
INITIAL_POINTS_PER_DIM = 3


# ---------------------------------------------------------------------------
# The GP surrogate
# ---------------------------------------------------------------------------


def _check_matrix(
    values: np.ndarray | list[list[float]], name: str, n_columns: int | None = None
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise DataValidationError(f"{name} must be 2-D, got {array.ndim}-D")
    if array.shape[0] < 1:
        raise DataValidationError(f"{name} must have at least one row")
    if n_columns is not None and array.shape[1] != n_columns:
        raise DataValidationError(f"{name} has {array.shape[1]} columns, expected {n_columns}")
    if not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} must be finite (no NaN/inf)")
    return array


class GaussianProcessSurrogate:
    """An RBF-kernel Gaussian process over the trials seen so far.

    The kernel is ``exp(-||x-x'||² / (2 ℓ²))`` with ``ℓ`` the median
    heuristic (the median pairwise distance, Duan & Kercheval's
    robust default), the diagonal jitter ``alpha`` matching keras-
    tuner's GP noise, and the posterior solved through the Cholesky
    factor of ``K + alpha·I`` -- the standard numerically stable path.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, length_scale: float, alpha: float) -> None:
        self.x = x
        self.y = y
        self.length_scale = float(length_scale)
        self.alpha = float(alpha)
        self._cholesky = np.linalg.cholesky(
            self._kernel(self.x, self.x) + alpha * np.eye(x.shape[0])
        )
        self._weights = np.linalg.solve(self._cholesky.T, np.linalg.solve(self._cholesky, y))

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        squared = np.sum(a**2, axis=1)[:, None] + np.sum(b**2, axis=1)[None, :] - 2.0 * (a @ b.T)
        np.maximum(squared, 0.0, out=squared)
        return np.exp(-squared / (2.0 * self.length_scale**2))

    @classmethod
    def fit(
        cls,
        x: np.ndarray | list[list[float]],
        y: np.ndarray | list[float],
        *,
        alpha: float = DEFAULT_ALPHA,
        length_scale: float | None = None,
        seed: int = RANDOM_STATE,  # kept for interface symmetry with the toolkit
    ) -> GaussianProcessSurrogate:
        """Fit on the observed trials; ``length_scale=None`` uses the median heuristic.

        Raises
        ------
        DataValidationError
        On empty or non-finite inputs, or fewer rows than one.
        """
        matrix = _check_matrix(x, "x")
        targets = np.asarray(y, dtype=float).reshape(-1)
        if targets.size != matrix.shape[0]:
            raise DataValidationError(f"y has {targets.size} entries, x has {matrix.shape[0]} rows")
        if not np.all(np.isfinite(targets)):
            raise DataValidationError("y must be finite (no NaN/inf)")
        if alpha <= 0:
            raise DataValidationError(f"alpha must be > 0, got {alpha!r}")
        if length_scale is None:
            if matrix.shape[0] > 1:
                differences = matrix[:, None, :] - matrix[None, :, :]
                distances = np.linalg.norm(differences, axis=-1)
                scale = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
            else:
                scale = 1.0
            length_scale = max(scale, 1e-6)
        if length_scale <= 0:
            raise DataValidationError(f"length_scale must be > 0, got {length_scale!r}")
        return cls(matrix, targets, length_scale, alpha)

    def posterior(
        self, candidates: np.ndarray | list[list[float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior (mean, std) at the candidate points.

        Raises
        ------
        DataValidationError
        On dimension mismatch or non-finite candidates.
        """
        points = _check_matrix(candidates, "candidates", self.x.shape[1])
        cross = self._kernel(points, self.x)
        mean = cross @ self._weights
        solved = np.linalg.solve(self._cholesky, cross.T)
        variance = np.maximum(1.0 - np.sum(solved**2, axis=0), 0.0)
        return mean, np.sqrt(variance + self.alpha)


# ---------------------------------------------------------------------------
# The acquisition functions
# ---------------------------------------------------------------------------


def lower_confidence_bound(
    surrogate: GaussianProcessSurrogate,
    candidates: np.ndarray | list[list[float]],
    beta: float = DEFAULT_BETA,
) -> np.ndarray:
    """``mean − β·std`` -- UCB's minimisation form (keras-tuner's pick).

    The oracle's docstring names UCB; with the tuner's minimisation
    semantics the driving quantity is the *lower* confidence bound:
    exploit low means, explore high spreads.  ``beta=2.6`` is the
    library's default exploration factor.
    """
    if beta < 0:
        raise DataValidationError(f"beta must be >= 0, got {beta!r}")
    mean, std = surrogate.posterior(candidates)
    return mean - beta * std


def expected_improvement(
    surrogate: GaussianProcessSurrogate,
    candidates: np.ndarray | list[list[float]],
    best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """The classic expected improvement over the best observed score.

    ``EI(x) = E[max(best − f(x), 0)]`` under the GP posterior (Jones
    et al. 1998's closed form with the exploration slack ``xi``) --
    the acquisition keras-tuner's docstring contrasts its UCB against,
    implemented here so the choice is runnable, not just cited.
    """
    if xi < 0:
        raise DataValidationError(f"xi must be >= 0, got {xi!r}")
    mean, std = surrogate.posterior(candidates)
    improvement = best - xi - mean
    safe_std = np.maximum(std, 1e-12)
    z = improvement / safe_std
    # Phi and phi without scipy: the toolkit's own normal helpers below
    return _ei_closed_form(improvement, z, safe_std)


def _standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    """Φ(z) via the erf relationship (math.erf, vectorised through numpy)."""
    import math

    return np.array([0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in z])


def _standard_normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def _ei_closed_form(improvement: np.ndarray, z: np.ndarray, std: np.ndarray) -> np.ndarray:
    return improvement * _standard_normal_cdf(z) + std * _standard_normal_pdf(z)


# ---------------------------------------------------------------------------
# The search (the oracle's loop)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesTrial:
    """One trial: the parameters tried and the score observed."""

    params: dict[str, float]
    score: float
    source: str  # "warm-up" | "gp"


@dataclass(frozen=True)
class BayesianSearchReport:
    """The oracle's ledger: trials, the best, the acquisition diagnostics.

    ``best_x`` / ``best_score`` follow the oracle's best-so-far
    tracking; ``n_warmup`` records how many trials were the seeded
    Latin hypercube before the GP took over.
    """

    trials: tuple[BayesTrial, ...]
    best_x: dict[str, float]
    best_score: float
    n_trials: int
    n_warmup: int
    beta: float
    bounds: tuple[tuple[str, float, float], ...]

    @property
    def gp_trials(self) -> int:
        return self.n_trials - self.n_warmup


def _latin_hypercube(
    n_points: int, n_dims: int, bounds: tuple[tuple[str, float, float], ...], seed: int
) -> list[dict[str, float]]:
    """A seeded Latin hypercube over the box (one point per stratum per dim)."""
    rng = np.random.default_rng(seed)
    points: list[dict[str, float]] = []
    strata = np.arange(n_points)
    columns = []
    for _dim in range(n_dims):
        shuffled = rng.permutation(strata)
        within = (shuffled + rng.random(size=n_points)) / n_points
        columns.append(within)
    for index in range(n_points):
        point: dict[str, float] = {}
        for dim, (name, low, high) in enumerate(bounds):
            unit = float(np.clip(columns[dim][index], 0.0, 1.0))
            point[name] = low + unit * (high - low)
        points.append(point)
    return points


def _check_bounds(
    bounds: list[tuple[str, float, float]] | tuple[tuple[str, float, float], ...],
) -> tuple[tuple[str, float, float], ...]:
    if not bounds:
        raise DataValidationError("bounds must not be empty")
    checked = []
    names: set[str] = set()
    for entry in bounds:
        if len(entry) != 3:
            raise DataValidationError(f"each bound is (name, low, high), got {entry!r}")
        name, low, high = entry
        if not isinstance(name, str) or not name:
            raise DataValidationError(f"parameter names must be non-empty strings, got {name!r}")
        if name in names:
            raise DataValidationError(f"duplicate parameter name {name!r}")
        names.add(name)
        if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
            raise DataValidationError(
                f"bounds for {name!r} must satisfy high > low, got {(low, high)!r}"
            )
        checked.append((name, float(low), float(high)))
    return tuple(checked)


def bayesian_search(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[str, float, float]] | tuple[tuple[str, float, float], ...],
    n_trials: int = 15,
    num_initial_points: int | None = None,
    beta: float = DEFAULT_BETA,
    alpha: float = DEFAULT_ALPHA,
    candidates_per_step: int = 200,
    seed: int = RANDOM_STATE,
) -> BayesianSearchReport:
    """The BayesianOptimizationOracle loop, minimising ``objective``.

    The protocol, keras-tuner's own: ``num_initial_points`` (default
    ``3 × dimensions``) seeded Latin-hypercube trials first; then for
    each remaining trial, refit the GP on everything seen and pick
    the candidate minimising the lower confidence bound
    (``mean − β·σ``) from a seeded dense grid inside the box.  The
    objective receives one row (the parameter vector) and returns one
    float; the best-so-far is tracked oracle-style.

    Raises
    ------
    DataValidationError
        On non-callable objectives, bad bounds, or non-positive
        trial/candidate counts.
    """
    if not callable(objective):
        raise DataValidationError("objective must be callable: row -> score")
    checked_bounds = _check_bounds(bounds)
    if n_trials < 2:
        raise DataValidationError(f"n_trials must be >= 2, got {n_trials}")
    if candidates_per_step < 2:
        raise DataValidationError(f"candidates_per_step must be >= 2, got {candidates_per_step}")
    if num_initial_points is None:
        num_initial_points = INITIAL_POINTS_PER_DIM * len(checked_bounds)
    if num_initial_points < 1 or num_initial_points >= n_trials:
        raise DataValidationError(
            f"num_initial_points must be in [1, n_trials-1], got {num_initial_points}"
        )
    rng = np.random.default_rng(seed)
    names = [name for name, _low, _high in checked_bounds]
    lows = np.array([low for _n, low, _h in checked_bounds])
    highs = np.array([high for _n, _h, high in checked_bounds])

    trials: list[BayesTrial] = []
    observed_x: list[list[float]] = []
    observed_y: list[float] = []

    for point in _latin_hypercube(num_initial_points, len(names), checked_bounds, seed):
        row = np.array([[point[name] for name in names]])
        score = float(objective(row[0]))
        if not np.isfinite(score):
            raise DataValidationError(f"objective returned a non-finite score {score!r} at {point}")
        trials.append(BayesTrial(params=point, score=score, source="warm-up"))
        observed_x.append([point[name] for name in names])
        observed_y.append(score)

    for step in range(n_trials - num_initial_points):
        surrogate = GaussianProcessSurrogate.fit(
            np.array(observed_x), np.array(observed_y), alpha=alpha
        )
        candidates = lows + rng.random((candidates_per_step, len(names))) * (highs - lows)
        acquisition = lower_confidence_bound(surrogate, candidates, beta=beta)
        choice = candidates[int(np.argmin(acquisition))]
        point = {name: float(value) for name, value in zip(names, choice)}
        score = float(objective(choice))
        if not np.isfinite(score):
            raise DataValidationError(f"objective returned a non-finite score {score!r} at {point}")
        trials.append(BayesTrial(params=point, score=score, source="gp"))
        observed_x.append([point[name] for name in names])
        observed_y.append(score)
        _LOG.debug("bayes step %d: picked %s score %.6g", step + 1, point, score)

    best_index = int(np.argmin(observed_y))
    best_x = {name: float(observed_x[best_index][dim]) for dim, name in enumerate(names)}
    report = BayesianSearchReport(
        trials=tuple(trials),
        best_x=best_x,
        best_score=float(observed_y[best_index]),
        n_trials=len(trials),
        n_warmup=int(num_initial_points),
        beta=float(beta),
        bounds=checked_bounds,
    )
    _LOG.debug(
        "bayesian search: %d trials (%d warm-up), best %.6g",
        report.n_trials,
        report.n_warmup,
        report.best_score,
    )
    return report
