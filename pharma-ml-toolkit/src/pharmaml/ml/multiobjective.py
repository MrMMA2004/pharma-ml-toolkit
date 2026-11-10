"""Multi-objective hyperparameter optimization -- the Pareto study.

The best-of-ml-python HPO/AutoML section's leaders converge on one
capability this toolkit's search stack lacked: **the trade-off itself as
a first-class result**.  Optuna ships multi-objective studies (vector
``values``, the ParetoFront pruner, MOTPE), Ax/BoTorch drive expected
hypervolume improvement through a GP over vector objectives, and
nevergrad runs a portfolio of gradient-free optimizers per objective.
The generative side of this house has taught Pareto fronts for
*molecules* since the first ``--pareto`` recipes (score vs novelty vs
QED); this module brings the same honesty to the *hyperparameter*
search: a pharma ML pipeline rarely wants "the most accurate model" --
it wants the accuracy-vs-fit-seconds-vs-latency menu and the corner it
can afford.

What lands (pure NumPy, the optuna convention: every objective is
MINIMIZED):

* :func:`dominates` / :func:`pareto_indices` -- the Pareto order and the
  non-dominated set.  Pairwise ``O(n^2 * k)`` at study scale (tens to
  hundreds of trials) -- the honest complexity statement; the generative
  front extractor makes the same choice at molecule scale.
* :func:`hypervolume_2d` -- the exact 2-D dominated-area indicator (the
  staircase sweep), THE convergence diagnostic for a front's quality.
  The reference point must dominate every point (the WFG contract).
* :func:`crowding_distance` -- NSGA-II's spread metric (boundary members
  ``+inf``): the shortlist pruner when the front outgrows the report.
* :func:`chebyshev_scalarize` -- the MOEA/D Tchebycheff decomposition
  that turns the EXISTING single-objective samplers (random + TPE) into
  front-chasers: each trial carries a weight vector, and its Chebyshev
  distance to the running ideal point is the scalar the sampler sees.
* :class:`MultiObjectiveStudy` / :class:`ParetoReport` -- the study
  bookkeeping: :class:`MOTrial` records (params, values, weights,
  scalar); the front, the crowding and the hypervolume curve are
  recomputed on demand.
* :func:`tune_model_multi` -- the sklearn adapter over
  ``cross_validate``'s own ledger: ``("score", "fit_seconds")`` by
  default, ``"score_seconds"`` available -- the same numbers
  :mod:`pharmaml.ml.cv_results` records per fold.

The honest scope, stated once: EHVI needs a GP posterior over a *vector*
objective (this house's GP in :mod:`pharmaml.ml.bayesopt` is scalar),
MOTPE's per-side multivariate Parzen estimators and NSGA-II's genetic
operators stay upstream (the generative GA carries the latter for
molecules), and the exact hypervolume beyond two objectives (WFG's
sweep) is not vendored -- the 2-D sweep is the indicator this module
reports.  What transfers exactly is the order theory, the indicator,
the decomposition, and the study arithmetic.

Examples
--------
>>> from pharmaml.ml.multiobjective import dominates, pareto_indices
>>> dominates((0.5, 2.0), (0.6, 2.0))      # better on one, equal on the other
True
>>> dominates((0.5, 2.0), (0.5, 2.0))      # mutual non-domination
False
>>> pareto_indices([(1.0, 2.0), (3.0, 1.0), (2.0, 2.0), (1.0, 5.0)])
[0, 1]

The dominated area under a two-objective front (the staircase sweep):

>>> from pharmaml.ml.multiobjective import hypervolume_2d
>>> hypervolume_2d([(1.0, 1.0), (1.5, 0.5)], ref=(2.0, 2.0))
1.25
>>> hypervolume_2d([(1.0, 1.0), (1.5, 1.8)], ref=(2.0, 2.0))  # dominated adds 0
1.0

NSGA-II crowding: the trade-off corners score ``inf``, crowded interiors low:

>>> from pharmaml.ml.multiobjective import crowding_distance
>>> pts = [(0.2, 0.9), (0.9, 0.4), (0.25, 0.85), (0.5, 0.6)]
>>> [round(d, 4) for d in crowding_distance(pts)]
[inf, inf, 1.0286, 1.8286]

The Tchebycheff decomposition (``max_j w_j |v_j - ideal_j|``, minimized):

>>> from pharmaml.ml.multiobjective import chebyshev_scalarize
>>> chebyshev_scalarize((0.5, 2.0), weights=(1.0, 0.5), ideal=(0.4, 1.0))
0.5
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.hpo import RandomSampler, SearchSpace, TPEOptimizer, TrialState

__all__ = [
    "MOTrial",
    "MultiObjectiveStudy",
    "ParetoReport",
    "ParetoShortlist",
    "chebyshev_scalarize",
    "crowding_distance",
    "dominates",
    "hypervolume_2d",
    "pareto_indices",
    "tune_model_multi",
]


# -- validation helpers ------------------------------------------------------ #


def _as_points(points: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    """Validate a (n_points, n_objectives) finite float matrix."""
    matrix = np.asarray(points, dtype=float)
    if matrix.ndim != 2:
        raise DataValidationError(f"{name} must be 2-D, got {matrix.ndim}-D")
    if matrix.shape[0] < 1:
        raise DataValidationError(f"{name} must have at least one row")
    if matrix.shape[1] < 1:
        raise DataValidationError(f"{name} must have at least one objective column")
    if not np.all(np.isfinite(matrix)):
        raise DataValidationError(f"{name} must be finite (no NaN/inf) -- dominance is undefined")
    return matrix


def _as_vector(values: Sequence[float], name: str, n: int) -> tuple[float, ...]:
    """Validate one objective vector of the expected length."""
    if len(values) != n:
        raise DataValidationError(f"{name} must have {n} entries, got {len(values)}")
    out = tuple(float(v) for v in values)
    for v in out:
        if not math.isfinite(v):
            raise DataValidationError(f"{name} must be finite (no NaN/inf), got {v!r}")
    return out


# -- the Pareto order -------------------------------------------------------- #


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Pareto dominance under the optuna convention (all objectives minimized).

    ``a`` dominates ``b`` iff ``a`` is at least as good on every
    objective and strictly better on at least one.

    >>> dominates((0.5, 2.0), (0.6, 2.0))
    True
    >>> dominates((0.5, 2.1), (0.6, 2.0))
    False
    >>> dominates((0.5, 2.0), (0.5, 2.0))
    False
    >>> dominates((0.5,), (0.5, 2.0))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: dominance needs equal-length objective vectors, got 1 and 2
    """
    if len(a) != len(b):
        raise DataValidationError(
            f"dominance needs equal-length objective vectors, got {len(a)} and {len(b)}"
        )
    fa = _as_vector(a, "a", len(a))
    fb = _as_vector(b, "b", len(b))
    return all(x <= y for x, y in zip(fa, fb)) and any(x < y for x, y in zip(fa, fb))


def pareto_indices(points: Sequence[Sequence[float]] | np.ndarray) -> list[int]:
    """The non-dominated indices of a point cloud (ascending input order).

    Pairwise dominance filtering -- ``O(n^2 * k)`` at n points and k
    objectives, the honest cost at study scale (tens to hundreds of
    trials; the generative front extractor makes the same choice at
    molecule scale).  Objective-space duplicates survive together:
    mutual non-domination is the definition.

    >>> pareto_indices([(1.0, 2.0), (3.0, 1.0), (2.0, 2.0), (1.0, 5.0)])
    [0, 1]
    >>> pareto_indices([(1.0, 1.0), (1.0, 1.0), (2.0, 0.5)])
    [0, 1, 2]
    >>> pareto_indices([(5.0, 5.0)])
    [0]
    """
    matrix = _as_points(points, "points")
    kept: list[int] = []
    for i in range(matrix.shape[0]):
        row = matrix[i]
        # any j whose point dominates row knocks row out; j == i never does
        dominated_by = np.all(matrix <= row, axis=1) & np.any(matrix < row, axis=1)
        if not bool(dominated_by.any()):
            kept.append(i)
    return kept


# -- the hypervolume indicator (exact 2-D) ----------------------------------- #


def hypervolume_2d(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    ref: tuple[float, float],
) -> float:
    """The exact 2-D dominated hypervolume under minimisation.

    The dominated region is the union of the axis-aligned rectangles
    ``[p, ref]`` over every point ``p``; the value is its area, computed
    by the staircase sweep: sort by the first objective ascending, walk
    the strict-decreasing prefix of the second, and each stair
    contributes ``(next_x - x) * (ref_y - y)``.  Dominated points never
    leave the staircase, so they never change the value.

    The reference point must dominate every point (each coordinate
    ``>=`` the corresponding point coordinate) -- the WFG contract; a
    non-dominating reference raises instead of silently clipping.

    >>> hypervolume_2d([(1.0, 1.0), (1.5, 0.5)], ref=(2.0, 2.0))
    1.25
    >>> hypervolume_2d([(1.0, 1.0)], ref=(2.0, 2.0))
    1.0
    >>> hypervolume_2d([(1.0, 1.0), (1.5, 1.8)], ref=(2.0, 2.0))  # dominated adds 0
    1.0
    >>> hypervolume_2d([], ref=(2.0, 2.0))
    0.0
    >>> hypervolume_2d([(1.0, 3.0)], ref=(2.0, 2.0))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: reference point (2.0, 2.0) must dominate every point -- (1.0, 3.0) escapes on objective 2
    """
    if len(points) == 0:
        return 0.0
    matrix = _as_points(points, "points")
    if matrix.shape[1] != 2:
        raise DataValidationError(
            f"the exact hypervolume sweep is 2-D only, got {matrix.shape[1]} objectives"
        )
    ref_x, ref_y = float(ref[0]), float(ref[1])
    if not (math.isfinite(ref_x) and math.isfinite(ref_y)):
        raise DataValidationError(f"reference point must be finite, got {ref!r}")
    for i in range(matrix.shape[0]):
        if matrix[i, 0] > ref_x or matrix[i, 1] > ref_y:
            raise DataValidationError(
                f"reference point {ref!r} must dominate every point -- "
                f"{(float(matrix[i, 0]), float(matrix[i, 1]))!r} escapes on objective "
                f"{1 if matrix[i, 0] > ref_x else 2}"
            )
    order = np.lexsort((matrix[:, 1], matrix[:, 0]))  # x asc, then y asc
    staircase: list[int] = []
    best_y = math.inf
    for idx in order:
        y = float(matrix[idx, 1])
        if y < best_y:
            staircase.append(int(idx))
            best_y = y
    area = 0.0
    for step, idx in enumerate(staircase):
        x, y = float(matrix[idx, 0]), float(matrix[idx, 1])
        next_x = (
            float(matrix[staircase[step + 1], 0])
            if step + 1 < len(staircase)
            else ref_x
        )
        area += (next_x - x) * (ref_y - y)
    return float(area)


# -- NSGA-II crowding -------------------------------------------------------- #


def crowding_distance(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """NSGA-II's crowding distance (boundaries ``+inf``), per point.

    For every objective the points are sorted; the two extremes of that
    objective gain ``+inf`` and each interior point gains the normalised
    gap between its neighbours ``(next - prev) / span``.  A degenerate
    objective (zero span) contributes nothing -- identical values carry
    no spread information.

    >>> [round(d, 4) for d in crowding_distance([(0.2, 0.9), (0.9, 0.4), (0.25, 0.85), (0.5, 0.6)])]
    [inf, inf, 1.0286, 1.8286]
    >>> [round(d, 4) for d in crowding_distance([(1.0, 1.0), (2.0, 2.0)])]
    [inf, inf]
    """
    matrix = _as_points(points, "points")
    n = matrix.shape[0]
    if n <= 2:
        return np.full(n, math.inf)
    distance = np.zeros(n)
    for dim in range(matrix.shape[1]):
        order = np.argsort(matrix[:, dim], kind="stable")
        column = matrix[order, dim]
        distance[order[0]] = math.inf
        distance[order[-1]] = math.inf
        span = float(column[-1] - column[0])
        if span > 0.0:
            distance[order[1:-1]] += (column[2:] - column[:-2]) / span
    return distance


# -- the MOEA/D decomposition ------------------------------------------------ #


def chebyshev_scalarize(
    values: Sequence[float],
    *,
    weights: Sequence[float],
    ideal: Sequence[float],
) -> float:
    """The Tchebycheff scalarization ``max_j w_j |v_j - ideal_j|``.

    MOEA/D's decomposition: one scalar per weight vector, minimized.
    A zero weight is the corner subproblem that optimizes its objective
    alone (the front's endpoint); at least one weight must be positive,
    and negative weights are meaningless.

    >>> chebyshev_scalarize((0.5, 2.0), weights=(1.0, 0.5), ideal=(0.4, 1.0))
    0.5
    >>> chebyshev_scalarize((0.5, 2.0), weights=(0.0, 1.0), ideal=(0.4, 1.0))
    1.0
    >>> chebyshev_scalarize((0.5,), weights=(1.0, 1.0), ideal=(0.4,))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: values, weights and ideal must share one length, got 1, 2, 1
    """
    if not (len(values) == len(weights) == len(ideal)):
        raise DataValidationError(
            f"values, weights and ideal must share one length, got "
            f"{len(values)}, {len(weights)}, {len(ideal)}"
        )
    fv = _as_vector(values, "values", len(values))
    fw = tuple(float(w) for w in weights)
    for w in fw:
        if w < 0.0 or not math.isfinite(w):
            raise DataValidationError(f"weights must be finite and >= 0, got {fw!r}")
    if not any(w > 0.0 for w in fw):
        raise DataValidationError(f"at least one weight must be positive, got {fw!r}")
    fi = _as_vector(ideal, "ideal", len(ideal))
    return float(max(w * abs(v - z) for v, w, z in zip(fv, fw, fi)))


def _simplex_lattice(n_objectives: int, n_vectors: int) -> list[tuple[float, ...]]:
    """Das & Dennis's simplex-lattice weight set (deterministic spread).

    All compositions of ``levels - 1`` into ``n_objectives`` non-negative
    parts, normalised to sum 1, generated in lexicographic order and cut
    to ``n_vectors`` -- the corner-heavy prefix (endpoints first,
    interiors later).  ``levels`` is the smallest giving at least
    ``n_vectors`` vectors, capped so the lattice cannot explode.
    """
    limit = 12

    def count(levels: int) -> int:
        # compositions of levels-1 into k parts: C(levels-1+k-1, k-1)
        return math.comb(levels - 1 + n_objectives - 1, n_objectives - 1)

    levels = 2
    while levels < limit and count(levels) < n_vectors:
        levels += 1
    compositions: list[tuple[int, ...]] = []

    def build(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            compositions.append(tuple(prefix + [remaining]))
            return
        for value in range(remaining + 1):
            build(prefix + [value], remaining - value, slots - 1)

    build([], levels - 1, n_objectives)
    out: list[tuple[float, ...]] = []
    for comp in compositions[:n_vectors]:
        out.append(tuple(part / (levels - 1) for part in comp))
    return out


def _random_simplex(rng: random.Random, n_objectives: int) -> tuple[float, ...]:
    """One seeded uniform simplex draw (the Dirichlet(1,...,1) shortcut).

    Exponential draws normalised to sum 1 -- the uniform distribution
    on the simplex, and Optuna's own approach for random multi-objective
    weights.
    """
    draws = [rng.expovariate(1.0) for _ in range(n_objectives)]
    total = sum(draws)
    return tuple(d / total for d in draws)


# -- the study ---------------------------------------------------------------- #


@dataclass
class MOTrial:
    """One multi-objective evaluation: params, the objective vector, the weights.

    ``values`` follows the optuna convention -- every objective
    minimized.  ``scalar`` is this trial's Tchebycheff distance to the
    running ideal point under ``weights`` -- the single number the
    underlying sampler was told.
    """

    number: int
    params: dict[str, Any]
    values: tuple[float, ...]
    weights: tuple[float, ...]
    scalar: float
    state: str = TrialState.COMPLETE


class MultiObjectiveStudy:
    """The optuna multi-objective study, decomposed onto one sampler.

    Each trial takes the next weight vector from the study's spread (the
    deterministic simplex-lattice cycling when ``weight_mode="spread"``,
    a seeded uniform simplex draw when ``"random"``), evaluates the
    vector objective, and tells the underlying single-objective engine
    (random or TPE, unchanged) the trial's Tchebycheff distance to the
    running ideal point.  The decomposition is MOEA/D's; what it buys is
    that BOTH existing samplers chase a front instead of a point.

    The honest drift note: the ideal point advances as the front does,
    so the scalar landscape is not strictly stationary across trials
    (MOEA/D fixes weights per subproblem and leans on neighbourhood
    exchange for the same effect).  The hypervolume curve in
    :meth:`report` is the convergence check that the drift still
    descends.

    Parameters
    ----------
    objective:
        Maps a parameter dict to a sequence of ``n_objectives`` finite
        floats (all minimized).
    space:
        The :class:`~pharmaml.ml.hpo.SearchSpace` to search.
    sampler:
        ``"random"`` or ``"tpe"`` -- the engine under the decomposition.
    n_objectives:
        The objective-vector length; validated against every call.
    weight_mode:
        ``"spread"`` (Das-Dennis lattice, cycled) or ``"random"``
        (seeded uniform simplex draws).
    n_weight_vectors:
        The spread's size (>= 2); ignored for ``"random"``.
    """

    def __init__(
        self,
        objective: Callable[[Mapping[str, Any]], Sequence[float]],
        space: SearchSpace,
        *,
        n_objectives: int = 2,
        sampler: str = "random",
        seed: int = 0,
        weight_mode: str = "spread",
        n_weight_vectors: int = 8,
        n_startup: int = 10,
    ) -> None:
        if sampler not in {"random", "tpe"}:
            raise DataValidationError(f"unknown sampler {sampler!r} -- 'random' or 'tpe'")
        if weight_mode not in {"spread", "random"}:
            raise DataValidationError(
                f"unknown weight_mode {weight_mode!r} -- 'spread' or 'random'"
            )
        if n_objectives < 2:
            raise DataValidationError(
                f"a Pareto study needs at least two objectives, got {n_objectives}"
            )
        if weight_mode == "spread" and n_weight_vectors < 2:
            raise DataValidationError(
                f"n_weight_vectors must be >= 2, got {n_weight_vectors}"
            )
        if not callable(objective):
            raise DataValidationError("objective must be callable: params -> value sequence")
        self.objective = objective
        self.space = space
        self.n_objectives = int(n_objectives)
        self.seed = int(seed)
        self._engine: RandomSampler
        if sampler == "tpe":
            self._engine = TPEOptimizer(space, n_startup=n_startup, seed=seed)
        else:
            self._engine = RandomSampler(space, seed=seed)
        self._weights: list[tuple[float, ...]] = (
            _simplex_lattice(n_objectives, n_weight_vectors)
            if weight_mode == "spread"
            else []
        )
        self._weight_rng = random.Random(seed)
        self._weight_cursor = 0
        self._ideal: tuple[float, ...] | None = None
        self.trials: list[MOTrial] = []

    # -- the weight supply ------------------------------------------------- #

    def _next_weights(self) -> tuple[float, ...]:
        if self._weights:
            weights = self._weights[self._weight_cursor % len(self._weights)]
            self._weight_cursor += 1
            return weights
        return _random_simplex(self._weight_rng, self.n_objectives)

    # -- the loop ----------------------------------------------------------- #

    def run(self, n_trials: int) -> list[MOTrial]:
        """Evaluate ``n_trials`` and return them (also appended to history).

        >>> from pharmaml.ml.hpo import SearchSpace
        >>> study = MultiObjectiveStudy(
        ...     lambda p: (p["x"], 1.0 - p["x"]), SearchSpace.uniform("x", 0.0, 1.0),
        ...     n_objectives=2, sampler="random", seed=3, weight_mode="spread",
        ...     n_weight_vectors=4)
        >>> trials = study.run(6)
        >>> len(trials) == 6 and all(len(t.values) == 2 for t in trials)
        True
        >>> study.trials[0].weights           # the lattice cycles corner-first
        (0.0, 1.0)
        >>> study.trials[1].weights
        (0.3333333333333333, 0.6666666666666666)
        """
        if n_trials < 1:
            raise DataValidationError(f"n_trials must be >= 1, got {n_trials}")
        for _ in range(n_trials):
            params = self._engine.ask()
            values = _as_vector(self.objective(params), "objective()", self.n_objectives)
            self._ideal = (
                values
                if self._ideal is None
                else tuple(min(a, b) for a, b in zip(self._ideal, values))
            )
            weights = self._next_weights()
            scalar = chebyshev_scalarize(values, weights=weights, ideal=self._ideal)
            self._engine.tell(params, scalar)
            self.trials.append(
                MOTrial(
                    number=len(self.trials),
                    params=dict(params),
                    values=values,
                    weights=weights,
                    scalar=scalar,
                )
            )
        return self.trials[-n_trials:]

    def report(self, objective_names: Sequence[str] | None = None) -> ParetoReport:
        """The Pareto report over the completed trials.

        >>> from pharmaml.ml.hpo import SearchSpace
        >>> study = MultiObjectiveStudy(
        ...     lambda p: (p["x"], 1.0 - p["x"]), SearchSpace.uniform("x", 0.0, 1.0),
        ...     n_objectives=2, sampler="random", seed=3)
        >>> trials = study.run(8)
        >>> report = study.report(["loss_a", "loss_b"])
        >>> len(report.trials) >= 1 and report.hypervolume is not None
        True
        """
        return ParetoReport.from_study(self, objective_names)


@dataclass
class ParetoReport:
    """The front, its spread, its indicator: what a trade-off decision reads.

    ``trials`` are the non-dominated :class:`MOTrial` members in study
    order; ``crowding`` is NSGA-II's spread metric aligned with them;
    ``hypervolume`` is the exact 2-D indicator when the study is
    bi-objective (``None`` otherwise), under the report's reference
    point ``ref`` -- the observed nadir padded by 10% of the
    ideal-to-nadir range per axis (the documented margin; the exact
    nadir would zero the boundary members' contribution).
    ``hypervolume_curve`` is the per-trial indicator under that same
    FIXED reference -- monotonically non-decreasing by construction
    (a growing point set grows the union).
    """

    objective_names: tuple[str, ...]
    trials: list[MOTrial] = field(default_factory=list)
    crowding: tuple[float, ...] = ()
    ref: tuple[float, float] | None = None
    hypervolume: float | None = None
    hypervolume_curve: tuple[float, ...] = ()
    n_trials: int = 0

    @classmethod
    def from_study(
        cls, study: MultiObjectiveStudy, objective_names: Sequence[str] | None = None
    ) -> ParetoReport:
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        names = tuple(objective_names) if objective_names is not None else tuple(
            f"objective_{j + 1}" for j in range(study.n_objectives)
        )
        if len(names) != study.n_objectives:
            raise DataValidationError(
                f"objective_names has {len(names)} entries, the study has "
                f"{study.n_objectives} objectives"
            )
        if not completed:
            return cls(objective_names=names, n_trials=len(study.trials))
        matrix = np.array([t.values for t in completed], dtype=float)
        front_rows = pareto_indices(matrix)
        front = [completed[i] for i in front_rows]
        crowding = tuple(float(d) for d in crowding_distance(matrix[front_rows]))
        report = cls(
            objective_names=names,
            trials=front,
            crowding=crowding,
            n_trials=len(study.trials),
        )
        if study.n_objectives == 2:
            ideal = matrix.min(axis=0)
            nadir = matrix.max(axis=0)
            margin = np.maximum(0.1 * (nadir - ideal), 1e-12)
            ref_point = (float(nadir[0] + margin[0]), float(nadir[1] + margin[1]))
            report.ref = ref_point
            report.hypervolume = hypervolume_2d(matrix, ref=ref_point)
            report.hypervolume_curve = tuple(
                hypervolume_2d(matrix[: i + 1], ref=ref_point) for i in range(len(matrix))
            )
        return report

    # -- the reading helpers ------------------------------------------------ #

    def best_for(self, objective_name: str) -> MOTrial:
        """The front member minimizing one named objective (its endpoint).

        Raises
        ------
        DataValidationError
            On an unknown objective name or an empty front.
        """
        if not self.trials:
            raise DataValidationError("the front is empty -- no best to name")
        try:
            j = self.objective_names.index(objective_name)
        except ValueError:
            raise DataValidationError(
                f"unknown objective {objective_name!r}; the report carries "
                f"{list(self.objective_names)}"
            ) from None
        return min(self.trials, key=lambda t: t.values[j])

    def prune(self, keep: int) -> ParetoShortlist:
        """Select up to ``keep`` front members using full-front crowding.

        Larger NSGA-II crowding distances rank first; ties use the original
        front order. The returned trials retain that original order and their
        crowding values are measured on the *full* front. If there are more
        boundary members than slots, only the earliest ones are retained.
        This is a one-shot selection, not iterative crowding recomputation.
        Empty fronts return an empty shortlist. The shortlist is distinct from
        the report: it does not claim the full study's hypervolume as its own.

        ``keep`` must be a positive integer. Trial objects are shared with the
        report; this operation does not modify the report or study.
        """
        if isinstance(keep, bool) or not isinstance(keep, Integral) or keep < 1:
            raise DataValidationError(f"keep must be a positive integer, got {keep!r}")
        if len(self.crowding) != len(self.trials):
            raise DataValidationError("crowding must align with the report's trials")
        if any(math.isnan(value) for value in self.crowding):
            raise DataValidationError("crowding distances cannot be NaN")
        indices = sorted(range(len(self.trials)), key=lambda i: (-self.crowding[i], i))
        chosen = sorted(indices[:keep])
        return ParetoShortlist(
            objective_names=self.objective_names,
            trials=tuple(self.trials[i] for i in chosen),
            crowding=tuple(self.crowding[i] for i in chosen),
            n_front=len(self.trials),
            n_trials=self.n_trials,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """The tidy frame: number, objectives, scalar, crowding.

        >>> from pharmaml.ml.hpo import SearchSpace
        >>> study = MultiObjectiveStudy(
        ...     lambda p: (p["x"], 1.0 - p["x"]), SearchSpace.uniform("x", 0.0, 1.0),
        ...     n_objectives=2, sampler="random", seed=3)
        >>> study.run(4)
        [...MOTrial...]
        >>> frame = study.report(["loss_a", "loss_b"]).to_dataframe()
        >>> list(frame.columns)
        ['number', 'loss_a', 'loss_b', 'scalar', 'crowding']
        """
        rows = []
        for trial, crowd in zip(self.trials, self.crowding):
            row: dict[str, Any] = {"number": trial.number}
            for name, value in zip(self.objective_names, trial.values):
                row[name] = value
            row["scalar"] = trial.scalar
            row["crowding"] = crowd
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class ParetoShortlist:
    """Selected trials and their crowding measured on the original full front.

    ``n_front`` is the size before pruning, and ``n_trials`` counts all
    study trials. Objectives are minimized. The trial records themselves are
    shared with the source report. Hypervolume belongs to the full report and
    is deliberately not carried by this selection.
    """

    objective_names: tuple[str, ...]
    trials: tuple[MOTrial, ...]
    crowding: tuple[float, ...]
    n_front: int
    n_trials: int

    def to_dataframe(self) -> pd.DataFrame:
        """Export selected trial objectives, scalar and full-front crowding."""
        rows = []
        for trial, crowd in zip(self.trials, self.crowding):
            row: dict[str, Any] = {"number": trial.number}
            for name, value in zip(self.objective_names, trial.values):
                row[name] = value
            row["scalar"] = trial.scalar
            row["crowding"] = crowd
            rows.append(row)
        return pd.DataFrame(rows, columns=["number", *self.objective_names, "scalar", "crowding"])


# -- the sklearn adapter ------------------------------------------------------ #


#: The tunable objective ledger: cross_validate key each name reads, and
#: the sign flip that enforces the minimize convention.
_SUPPORTED_OBJECTIVES: dict[str, tuple[str, float]] = {
    "score": ("test_score", -1.0),  # bigger is better -> minimize the negation
    "fit_seconds": ("fit_time", 1.0),
    "score_seconds": ("score_time", 1.0),
}


def tune_model_multi(
    name: str,
    space: SearchSpace,
    X: Any,
    y: Any,
    *,
    task: str = "classification",
    cv: int = 3,
    n_trials: int = 20,
    objectives: Sequence[str] = ("score", "fit_seconds"),
    sampler: str = "tpe",
    seed: int = 0,
) -> ParetoReport:
    """Cross-validated multi-objective HPO over the sklearn registry models.

    Each candidate parameterization is scored once by
    ``cross_validate``; the requested objectives are read straight off
    its ledger (``score`` = the negated mean CV score, ``fit_seconds`` /
    ``score_seconds`` = the mean fold times -- the same numbers
    :mod:`pharmaml.ml.cv_results` records).  The study then does what a
    single number cannot: it returns the accuracy-vs-cost front.

    >>> import numpy as np
    >>> from pharmaml.ml.hpo import SearchSpace
    >>> from pharmaml.ml.multiobjective import tune_model_multi
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 4))
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> space = SearchSpace.uniform("alpha", 0.01, 1.0, log=True)
    >>> report = tune_model_multi(
    ...     "ridge_classifier", space, X, y, task="classification", cv=2,
    ...     n_trials=6, sampler="random", seed=1)
    >>> report.objective_names
    ('score', 'fit_seconds')
    >>> len(report.trials) >= 1 and all(len(t.values) == 2 for t in report.trials)
    True
    """
    from sklearn.model_selection import cross_validate

    from pharmaml.exceptions import ModelError
    from pharmaml.ml.models import build_model

    if len(objectives) < 2:
        raise DataValidationError(
            f"a Pareto study needs at least two objectives, got {list(objectives)!r}"
        )
    if len(set(objectives)) != len(objectives):
        raise DataValidationError(f"objectives must be distinct, got {list(objectives)!r}")
    for label in objectives:
        if label not in _SUPPORTED_OBJECTIVES:
            raise DataValidationError(
                f"unsupported objective {label!r}; available: "
                f"{sorted(_SUPPORTED_OBJECTIVES)}"
            )

    try:
        build_model(name, task=task, params={"random_state": seed})
        seed_kwargs: dict[str, Any] = {"random_state": seed}
    except ModelError:
        seed_kwargs = {}

    keys = [_SUPPORTED_OBJECTIVES[label][0] for label in objectives]
    signs = [_SUPPORTED_OBJECTIVES[label][1] for label in objectives]

    def objective(params: Mapping[str, Any]) -> tuple[float, ...]:
        model = build_model(name, task=task, params={**dict(params), **seed_kwargs})
        ledger = cross_validate(model, X, y, cv=cv)
        return tuple(
            float(sign) * float(np.mean(ledger[key])) for key, sign in zip(keys, signs)
        )

    study = MultiObjectiveStudy(
        objective,
        space,
        n_objectives=len(objectives),
        sampler=sampler,
        seed=seed,
    )
    study.run(n_trials)
    return study.report(tuple(objectives))
