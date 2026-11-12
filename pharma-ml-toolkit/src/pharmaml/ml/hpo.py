"""Offline hyperparameter optimization -- the TPE / SHA / Hyperband lineages.

Rounds 188-189 checked optuna and ray-tune file by file and lifted what
survives without a distributed worker fleet or a storage backend:

* **TPE -- Tree-structured Parzen Estimators** (Bergstra et al. 2011,
  optuna's default sampler).  Trials are split into a *good* group (the
  best ``gamma`` of them) and the rest; each numeric dimension gets two
  Parzen estimators -- Gaussian kernels centred on the observed points,
  bandwidth scaled to the range -- and the next candidate is drawn from
  the good side, kept when the likelihood ratio ``l(x) / g(x)`` beats a
  uniform draw.  The split rule is optuna's own:
  ``gamma = min(ceil(0.1 * n), 25)`` -- capped so tiny studies stay
  explorative.  Categorical dimensions use the empirical counts with
  the house smoothing; log-scale floats sample in log space.
* **Successive halving** (Jamieson & Talwalkar) -- *n* configs at base
  budget *r*, keep the top ``1 / eta``, repeat until one survivor.
* **Hyperband** (Li et al. 2017) -- the bracket set that makes SHA's
  budget allocation robust to not knowing *n* in advance: bracket *s*
  runs ``ceil((R / r) * eta**s / (s + 1))`` configs at ``r * eta**-s``
  base budget, halving up to the full budget *R*.
* **The study bookkeeping** -- trial records with parameters, value,
  and state (RUNNING/COMPLETE/PRUNED), the running best, and the
  report the GUI card renders.

The honest scope, stated once: optuna's pruner communication between
objective and sampler (the ``Trial.should_prune`` callback family) is
reduced to the successive-halving verdict function; ray-tune's search
algorithms beyond TPE (HyperOpt's anneal, nevergrad's portfolio) and
its trial actors/schedulers are what this toolkit does not vendor.
What transfers is the math: the parzen split, the halving arithmetic,
and the bracket schedule.

The engines are pure NumPy + stdlib and work on any objective that
maps a parameter dict to a float (smaller = better, the optuna
convention).  :func:`tune_model` adapts the sklearn registry models to
that objective with cross-validated scoring.

Examples
--------
>>> from pharmaml.ml.hpo import SearchSpace, TPEOptimizer
>>> space = SearchSpace.uniform("lr", 0.001, 0.5, log=True)
>>> opt = TPEOptimizer(space, seed=7)
>>> trials = [opt.ask_and_run(lambda p: abs(p["lr"] - 0.05) + 0.01) for _ in range(24)]
>>> best = min(trials, key=lambda t: t.value)
>>> 0.0 < best.params["lr"] < 0.5
True
>>> len(opt.trials) == 24 and opt.best().value == best.value
True

>>> from pharmaml.ml.hpo import hyperband_brackets
>>> brackets = hyperband_brackets(max_budget=81, eta=3)
>>> len(brackets)  # R=81, eta=3 -> s = 0..4 (floor(log_eta(81)) + 1)
5
>>> (brackets[4].base_budget, brackets[4].n_configs)  # the fast bracket
(1.0, 81)

Successive halving keeps the top ``1/eta`` at every rung:

>>> from pharmaml.ml.hpo import successive_halving
>>> survivors = successive_halving(values=[0.9, 0.5, 0.7, 0.1, 0.6], eta=2)
>>> survivors  # the 4th value (0.1) is the best of the pack
[3]
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "CategoricalParam",
    "FloatParam",
    "HPOStudy",
    "HyperbandBracket",
    "IntParam",
    "ParamSpec",
    "RandomSampler",
    "SearchSpace",
    "TPEOptimizer",
    "Trial",
    "TrialState",
    "gamma_split",
    "hyperband_brackets",
    "parzen_density",
    "successive_halving",
    "tune_model",
]

# -- the trial record ----------------------------------------------------- #


class TrialState:
    """The optuna trial states, reduced to the four the study uses."""

    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"


@dataclass
class Trial:
    """One hyperparameter evaluation: the params, the value, the state.

    ``value`` follows the optuna convention -- smaller is better; the
    study adapts sklearn's bigger-is-better scores by negating them.
    """

    number: int
    params: dict[str, Any]
    value: float | None = None
    state: str = TrialState.RUNNING

    def is_finished(self) -> bool:
        return self.state != TrialState.RUNNING


# -- the search space ----------------------------------------------------- #


@dataclass
class ParamSpec:
    """Base class: one named dimension of the search space."""

    name: str

    def sample(self, rng: random.Random) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def parzen(self, points: Sequence[Any]) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class FloatParam(ParamSpec):
    """A float on ``[low, high]``; ``log=True`` samples in log space."""

    low: float = 0.0
    high: float = 1.0
    log: bool = False

    def sample(self, rng: random.Random) -> float:
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return math.exp(rng.uniform(lo, hi))
        return rng.uniform(self.low, self.high)

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass
class IntParam(ParamSpec):
    """An integer on ``[low, high]`` inclusive."""

    low: int = 0
    high: int = 10

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)


@dataclass
class CategoricalParam(ParamSpec):
    """A discrete choice; the parzen side is the empirical distribution."""

    choices: tuple[Any, ...] = ()

    def sample(self, rng: random.Random) -> Any:
        return self.choices[rng.randrange(len(self.choices))]


class SearchSpace:
    """An ordered collection of parameter specs with sampling helpers."""

    def __init__(self, *specs: ParamSpec) -> None:
        self.specs: dict[str, ParamSpec] = {s.name: s for s in specs}

    @classmethod
    def uniform(
        cls, name: str, low: float, high: float, log: bool = False
    ) -> SearchSpace:
        return cls(FloatParam(name, low, high, log=log))

    @classmethod
    def randint(cls, name: str, low: int, high: int) -> SearchSpace:
        return cls(IntParam(name, low, high))

    def add(self, spec: ParamSpec) -> SearchSpace:
        self.specs[spec.name] = spec
        return self

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return {name: spec.sample(rng) for name, spec in self.specs.items()}

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    def __len__(self) -> int:
        return len(self.specs)


# -- the TPE mathematics --------------------------------------------------- #


def gamma_split(n_trials: int, beta_cap: int = 25) -> int:
    """Optuna's split size: ``min(ceil(0.1 * n), beta_cap)``.

    >>> gamma_split(10)
    1
    >>> gamma_split(0)
    0
    >>> gamma_split(1000)
    25
    """
    if n_trials <= 0:
        return 0
    return min(math.ceil(0.1 * n_trials), beta_cap)


def parzen_density(
    points: np.ndarray, low: float, high: float
) -> tuple[np.ndarray, np.ndarray]:
    """A Gaussian-kernel Parzen estimator over ``[low, high]``.

    Returns the kernel centres and their bandwidths.  The bandwidth is
    optuna's rule scaled to the house constants: each point's sigma is
    the distance to the nearest range edge (never smaller than a tiny
    fraction of the range), divided by ``sqrt(count + 1)`` so denser
    regions sharpen.  The two edge sentinels (``low`` and ``high``
    themselves as kernel centres) are the estimator's prior mass.

    >>> import numpy as np
    >>> centres, sigmas = parzen_density(np.array([0.5]), 0.0, 1.0)
    >>> centres.shape, sigmas.shape
    ((3,), (3,))
    >>> centres[2] == 0.5 and sigmas[2] > 0  # [low, high, *points]
    True
    """
    count = len(points)
    span = max(high - low, 1e-12)
    all_points = np.concatenate(
        [np.array([low, high], dtype=float), np.asarray(points, dtype=float)]
    )
    if count == 0:
        sigmas = np.full(len(all_points), span / 50.0)
        return all_points, sigmas
    denom = math.sqrt(count + 1.0)
    edge_dist = np.minimum(all_points - low, high - all_points)
    sigmas = np.maximum(edge_dist, span * 1e-3) / denom
    return all_points, sigmas


def _identity(v: float) -> float:
    """The identity transform (the linear-scale twin of log/exp)."""
    return v


def _log_density_at(
    x: float, centres: np.ndarray, sigmas: np.ndarray
) -> float:
    """Average log N(x; centre, sigma) over the kernels (uniform weights)."""
    if len(centres) == 0:
        return 0.0
    var = np.maximum(sigmas, 1e-12) ** 2
    log_norm = -0.5 * np.log(2.0 * math.pi * var)
    quad = -0.5 * (x - centres) ** 2 / var
    log_liks = log_norm + quad
    # log of the mean via the stable max-shift
    m = float(np.max(log_liks))
    if not np.isfinite(m):  # pragma: no cover - degenerate bandwidths
        return 0.0
    return m + math.log(float(np.mean(np.exp(log_liks - m))))


class RandomSampler:
    """The baseline: uniform draws from the space, bookkept like TPE."""

    def __init__(self, space: SearchSpace, seed: int = 0) -> None:
        self.space = space
        self.rng = random.Random(seed)
        self._trials: list[Trial] = []

    @property
    def trials(self) -> list[Trial]:
        return self._trials

    def ask(self) -> dict[str, Any]:
        return self.space.sample(self.rng)

    def tell(self, params: dict[str, Any], value: float) -> Trial:
        trial = Trial(
            number=len(self._trials),
            params=dict(params),
            value=float(value),
            state=TrialState.COMPLETE,
        )
        self._trials.append(trial)
        return trial

    def ask_and_run(self, objective: Callable[[Mapping[str, Any]], float]) -> Trial:
        params = self.ask()
        return self.tell(params, float(objective(params)))

    def best(self) -> Trial:
        if not self._trials:
            raise ValueError("no trials yet")
        return min(self._trials, key=lambda t: t.value or math.inf)


class TPEOptimizer(RandomSampler):
    """The Tree-structured Parzen Estimator sampler.

    Until ``n_startup`` trials exist this is plain random sampling
    (optuna's own warm-up); afterwards each new point is drawn from the
    good-side Parzen estimator of every numeric dimension and kept when
    the likelihood ratio beats a uniform threshold.

    Parameters
    ----------
    space:
        The :class:`SearchSpace` to search.
    n_startup:
        Trials answered by random sampling before TPE kicks in
        (optuna's default is 10).
    n_candidates:
        Candidates drawn per ask; the best ratio wins (optuna's
        ``n_ei_candidates``, default 24, reduced by the house rule).
    seed:
        Deterministic RNG seed.
    """

    def __init__(
        self,
        space: SearchSpace,
        n_startup: int = 10,
        n_candidates: int = 12,
        seed: int = 0,
    ) -> None:
        super().__init__(space, seed=seed)
        self.n_startup = max(1, int(n_startup))
        self.n_candidates = max(1, int(n_candidates))

    # -- the good/bad split --------------------------------------------- #

    def _split(self) -> list[Trial]:
        """The completed trials sorted by value; TPE's l-side is the top gamma."""
        done = [t for t in self._trials if t.value is not None]
        done.sort(key=lambda t: float(t.value or math.inf))
        return done

    def ask(self) -> dict[str, Any]:
        done = self._split()
        if len(done) < self.n_startup:
            return self.space.sample(self.rng)
        gamma = gamma_split(len(done))
        good, bad = done[:gamma], done[gamma:]
        if not bad:  # every trial is on the good side -- random fallback
            return self.space.sample(self.rng)
        return self._tpe_sample(good, bad)

    def _tpe_sample(
        self, good: Sequence[Trial], bad: Sequence[Trial]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, spec in self.space.specs.items():
            if isinstance(spec, FloatParam):
                params[name] = self._tpe_float(spec, good, bad)
            else:  # int / categorical: the empirical good-side draw
                params[name] = self._empirical_choice(spec, good, bad)
        return params

    def _tpe_float(
        self, spec: FloatParam, good: Sequence[Trial], bad: Sequence[Trial]
    ) -> float:
        """Draw ``n_candidates`` from the good-side Parzen, keep the best l/g.

        This is optuna's actual sampling rule: candidates come from the
        l-estimator (the good side), and the one maximising the log
        likelihood ratio ``log l(x) - log g(x)`` wins.
        """
        if spec.log:
            lo, hi = math.log(spec.low), math.log(spec.high)
            wrap: Callable[[float], float] = math.exp
            unwrap: Callable[[float], float] = math.log
        else:
            lo, hi = spec.low, spec.high
            wrap, unwrap = _identity, _identity

        g_pts = np.array([unwrap(float(t.params[spec.name])) for t in good])
        b_pts = np.array([unwrap(float(t.params[spec.name])) for t in bad])
        g_c, g_s = parzen_density(g_pts, lo, hi)
        b_c, b_s = parzen_density(b_pts, lo, hi)

        best_x, best_log_ratio = None, -math.inf
        for _ in range(self.n_candidates):
            # draw from the good side: pick a kernel, then a normal around it
            k = self.rng.randrange(len(g_c))
            x = self.rng.gauss(float(g_c[k]), max(float(g_s[k]), 1e-9))
            x = min(max(x, lo), hi)
            log_ratio = _log_density_at(x, g_c, g_s) - _log_density_at(x, b_c, b_s)
            if log_ratio > best_log_ratio:
                best_x, best_log_ratio = x, log_ratio
        assert best_x is not None
        return float(wrap(best_x))

    def _empirical_choice(
        self, spec: ParamSpec, good: Sequence[Trial], bad: Sequence[Trial]
    ) -> Any:
        if isinstance(spec, CategoricalParam):
            pool = [t.params[spec.name] for t in good] or list(spec.choices)
            return self.rng.choice(list(pool))
        if isinstance(spec, IntParam):
            pool = [t.params[spec.name] for t in good]
            if pool and self.rng.random() < 0.7:
                return int(self.rng.choice(list(pool)))
        return spec.sample(self.rng)


# -- successive halving + hyperband ---------------------------------------- #


@dataclass
class HyperbandBracket:
    """One Hyperband bracket: the (n_configs, base_budget) pair per rung."""

    bracket: int
    max_budget: float
    eta: float
    base_budget: float
    n_configs: int

    def rungs(self) -> list[tuple[int, float]]:
        """(survivors, budget) per halving round of this bracket."""
        out: list[tuple[int, float]] = []
        n = self.n_configs
        budget = self.base_budget
        while n >= 1:
            out.append((int(n), float(budget)))
            n = math.ceil(n / self.eta)
            budget *= self.eta
            if budget > self.max_budget + 1e-9:
                break
        return out


def hyperband_brackets(
    max_budget: float, eta: float = 3.0
) -> list[HyperbandBracket]:
    """Li et al.'s bracket set for ``max_budget`` and reduction factor ``eta``.

    Bracket ``s`` (0..floor(log_eta(R))) starts
    ``ceil((s_max + 1) * eta**s / (s + 1))`` configurations at base
    budget ``R / eta**s`` (the paper's ``n = ceil((B/R) * eta^s / (s+1))``
    with the total budget ``B = (s_max + 1) * R``); together the brackets
    explore the full fast/cheap-vs-slow/thorough trade-off.

    >>> brackets = hyperband_brackets(27.0, 3.0)
    >>> [(b.bracket, b.n_configs, b.base_budget) for b in brackets]
    [(0, 4, 27.0), (1, 6, 9.0), (2, 12, 3.0), (3, 27, 1.0)]
    """
    if eta <= 1.0:
        raise ValueError(f"eta must be > 1, got {eta}")
    if max_budget <= 0:
        raise ValueError(f"max_budget must be positive, got {max_budget}")
    log_eta = math.log(eta)
    s_max = math.floor(math.log(max_budget) / log_eta)
    total_factor = s_max + 1  # B / R, the paper's budget normalization
    brackets: list[HyperbandBracket] = []
    for s in range(s_max + 1):
        base = max_budget / eta**s
        n = math.ceil(total_factor * (eta**s) / (s + 1))
        brackets.append(
            HyperbandBracket(
                bracket=s, max_budget=max_budget, eta=eta,
                base_budget=base, n_configs=n,
            )
        )
    return brackets


def successive_halving(
    values: Sequence[float], eta: float = 3.0
) -> list[int]:
    """The SHA survivor walk over already-measured values.

    Treats ``values`` as one rung's results, keeps the indices of the
    best ``ceil(len / eta)`` (ties broken by index), and recurses on
    the survivors until one is left.  Returns the survivor indices in
    ranking order -- the promotion list SHA would have produced.

    >>> successive_halving([0.4, 0.9, 0.1, 0.5], eta=2)
    [2]
    """
    if eta <= 1.0:
        raise ValueError(f"eta must be > 1, got {eta}")
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    survivors = order
    while len(survivors) > 1:
        keep = max(1, math.ceil(len(survivors) / eta))
        survivors = survivors[:keep]
    return survivors


# -- the study -------------------------------------------------------------- #


@dataclass
class HPOStudy:
    """The optuna study, offline: a sampler plus the running best-trial log."""

    objective: Callable[[Mapping[str, Any]], float]
    space: SearchSpace
    sampler: str = "tpe"
    seed: int = 0
    n_startup: int = 10
    history: list[Trial] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sampler not in {"tpe", "random"}:
            raise ValueError(f"unknown sampler {self.sampler!r}")
        engine: RandomSampler
        if self.sampler == "tpe":
            engine = TPEOptimizer(
                self.space, n_startup=self.n_startup, seed=self.seed
            )
        else:
            engine = RandomSampler(self.space, seed=self.seed)
        self._engine = engine

    def run(self, n_trials: int) -> list[Trial]:
        """Evaluate ``n_trials`` and return them (also appended to history)."""
        for _ in range(n_trials):
            trial = self._engine.ask_and_run(self.objective)
            self.history.append(trial)
        return self.history[-n_trials:]

    # -- (part5-continuation round 8, §13) resumable-study accessors -------- #
    #
    # The study's "complete state" (§13's requirement before a Pause/
    # Resume may exist): the finished trials AND the sampler's RNG
    # state.  The naive resume -- replaying the trial history into a
    # fresh engine -- is NOT bit-exact: every ask() draws from the
    # engine's ``random.Random``, whose state advances per trial, so a
    # resumed TPE would propose different candidates than the
    # uninterrupted run.  Capturing the RNG state alongside the
    # history makes the resume exact (proven by the round-8 tests).

    def capture_state(self) -> dict[str, Any]:
        """The complete resumable state: trial history + RNG state.

        The RNG state (``random.Random.getstate()`` -- a 625-int
        tuple) rides as a plain list; ``restore_state`` rebuilds it
        with ``setstate``.  Together with the trials this is
        everything a bit-exact resume needs.
        """
        rng_state = self._engine.rng.getstate()
        return {
            "trials": [
                {
                    "number": t.number,
                    "params": dict(t.params),
                    "value": t.value,
                    "state": t.state,
                }
                for t in self._engine.trials
            ],
            "rng_state": [rng_state[0], list(rng_state[1]), rng_state[2]],
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Rebuild the study from a :meth:`capture_state` payload.

        Raises
        ------
        ValueError
            If the payload is malformed (missing keys, non-numeric
            values) -- the caller treats this as "start fresh".
        """
        trials: list[Trial] = []
        for entry in state["trials"]:
            value = entry.get("value")
            trials.append(
                Trial(
                    number=int(entry["number"]),
                    params=dict(entry["params"]),
                    value=None if value is None else float(value),
                    state=str(entry["state"]),
                )
            )
        self._engine._trials = trials
        self.history = list(trials)
        version, internal, gauss = state["rng_state"]
        # gauss_next is None until the first gaussian draw consumed it;
        # round-trip it as-is (setstate accepts None there)
        gauss_value: float | None = None if gauss is None else float(gauss)
        self._engine.rng.setstate(
            (int(version), tuple(int(x) for x in internal), gauss_value)
        )

    def best(self) -> Trial:
        return self._engine.best()

    @property
    def trials(self) -> list[Trial]:
        return self._engine.trials


# -- the sklearn adapter ----------------------------------------------------- #


@dataclass
class TuneReport:
    """What the GUI card renders: best params, best score, the trial log."""

    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    history: list[Trial]
    task: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_params": dict(self.best_params),
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "task": self.task,
            "trials": [
                {"number": t.number, "params": t.params, "value": t.value}
                for t in self.history
            ],
        }


def tune_model(
    name: str,
    space: SearchSpace,
    X: Any,
    y: Any,
    *,
    task: str = "classification",
    cv: int = 3,
    n_trials: int = 20,
    sampler: str = "tpe",
    seed: int = 0,
    progress: Callable[..., Any] | None = None,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    checkpoint_every: int = 1,
    checkpoint_resume: bool = True,
) -> TuneReport:
    """Cross-validated HPO over the sklearn registry models.

    Builds ``name`` per candidate parameterization (the space's float /
    int / categorical values are passed straight through as estimator
    kwargs on top of ``random_state=seed``), scores it with the
    toolkit's cross-validated pipeline convention, and negates the
    score so *smaller is better* inside the study (the optuna rule).

    (part5-continuation round 8, §13) With ``checkpoint_dir`` the
    study becomes a resumable §13 operation: every
    ``checkpoint_every`` completed trials, an envelope-carrying
    checkpoint captures the COMPLETE state -- the finished trials AND
    the sampler's RNG state (without the latter a resumed TPE would
    propose different candidates; exact restore or nothing).  A later
    call with the same model/space/data resumes from the newest valid
    checkpoint: completed trials are never re-evaluated and the
    resumed report is bit-identical to an uninterrupted run.  A
    cooperative stop (``progress`` returning ``False``) preserves the
    partial report on :class:`~pharmaml.exceptions.JobCancelled`.

    >>> import numpy as np
    >>> from pharmaml.ml.hpo import SearchSpace, tune_model
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 4))
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> space = SearchSpace.uniform("alpha", 0.01, 1.0, log=True)
    >>> report = tune_model("ridge_classifier", space, X, y, task="classification",
    ...                     cv=2, n_trials=4, sampler="random", seed=1)
    >>> report.n_trials == 4 and report.best_score > 0
    True
    """
    from sklearn.model_selection import cross_val_score  # local: keeps import lazy

    from pharmaml.exceptions import JobCancelled, ModelError
    from pharmaml.ml.models import build_model

    # probe once: estimators like KNN have no random_state knob
    try:
        build_model(name, task=task, params={"random_state": seed})
        seed_kwargs: dict[str, Any] = {"random_state": seed}
    except ModelError:
        seed_kwargs = {}

    def objective(params: Mapping[str, Any]) -> float:
        model = build_model(
            name, task=task, params={**dict(params), **seed_kwargs}
        )
        score = float(np.mean(cross_val_score(model, X, y, cv=cv)))
        return -score

    study = HPOStudy(
        objective=objective, space=space, sampler=sampler, seed=seed
    )

    # -- (round 8, §13) the resumable wrapper around the trial loop ------
    if checkpoint_dir is not None:
        if (
            not isinstance(checkpoint_every, int)
            or isinstance(checkpoint_every, bool)
            or checkpoint_every < 1
        ):
            raise ValueError(
                f"checkpoint_every must be a positive int, got {checkpoint_every!r}"
            )
        import hashlib
        import json

        from pharmaml.utils.checkpoint_schema import OperationSpec
        from pharmaml.utils.checkpoints import CheckpointManager, PersistenceError

        try:
            manager = CheckpointManager(checkpoint_dir, name="hpo-study")
            spec = OperationSpec(
                operation_type="hpo",
                operation_id=f"tune-{name}-{seed}",
            )
        except PersistenceError as exc:
            _LOG.warning(
                "checkpoint home unusable (%s); tuning without checkpoints", exc
            )
            manager = None  # type: ignore[assignment]
            spec = None  # type: ignore[assignment]

        # config fingerprint: everything that shapes the search EXCEPT the
        # data -- model name, space bounds, sampler, seed, cv, n_trials
        def _space_blob() -> list[dict[str, Any]]:
            blob: list[dict[str, Any]] = []
            for pname, spec_ in space.specs.items():
                entry: dict[str, Any] = {"name": pname, "kind": type(spec_).__name__}
                if isinstance(spec_, FloatParam):
                    entry.update(low=spec_.low, high=spec_.high, log=spec_.log)
                elif isinstance(spec_, IntParam):
                    entry.update(low=spec_.low, high=spec_.high)
                elif isinstance(spec_, CategoricalParam):
                    entry.update(choices=[repr(c) for c in spec_.choices])
                blob.append(entry)
            return blob

        config_fp = hashlib.sha256(
            json.dumps(
                {
                    "model": name,
                    "task": task,
                    "cv": int(cv),
                    "n_trials": int(n_trials),
                    "sampler": sampler,
                    "seed": int(seed),
                    "space": _space_blob(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        # dataset fingerprint: shape + a content digest of X and y (the
        # adapter consumes numpy arrays / DataFrames alike)
        def _array_digest(a: Any) -> str:
            arr = np.asarray(a)
            digest = hashlib.sha256()
            digest.update(f"{arr.dtype}:{arr.shape}".encode())
            digest.update(np.ascontiguousarray(arr).tobytes())
            return digest.hexdigest()

        try:
            dataset_fp = hashlib.sha256(
                (_array_digest(X) + ":" + _array_digest(y)).encode()
            ).hexdigest()
        except (TypeError, ValueError):
            dataset_fp = "opaque-dataset"

        done = 0
        if manager is not None and checkpoint_resume:
            try:
                latest = manager.latest()
            except OSError:  # pragma: no cover - scan failure
                latest = None
                _LOG.warning("checkpoint scan failed; starting fresh")
            if latest is not None and latest.resumable:
                st = latest.state
                if (
                    st.get("config_fp") != config_fp
                    or st.get("dataset_fp") != dataset_fp
                ):
                    _LOG.warning(
                        "hpo checkpoint at step %d is for a different "
                        "config/dataset; starting fresh (exact restore or "
                        "nothing)",
                        latest.step,
                    )
                else:
                    try:
                        study.restore_state(st)
                        done = len(study.trials)
                        _LOG.info(
                            "hpo study resumed from checkpoint step %d (%d "
                            "trials already evaluated)",
                            latest.step,
                            done,
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        _LOG.warning(
                            "checkpoint payload unreadable (%s); starting fresh", exc
                        )

        boundaries_since_save = 0
        for _ in range(done, n_trials):
            trial = study._engine.ask_and_run(objective)
            study.history.append(trial)
            done += 1
            boundaries_since_save += 1
            if (
                manager is not None
                and spec is not None
                and (done >= n_trials or boundaries_since_save >= checkpoint_every)
            ):
                boundaries_since_save = 0
                try:
                    manager.save(
                        step=done,
                        total=n_trials,
                        state={
                            "config_fp": config_fp,
                            "dataset_fp": dataset_fp,
                            **study.capture_state(),
                        },
                        note=f"hpo study {done}/{n_trials} trials",
                        operation=spec,
                    )
                except (PersistenceError, OSError, TypeError, ValueError) as exc:
                    _LOG.warning(
                        "checkpoint write failed at %d/%d (%s); continuing "
                        "without it",
                        done,
                        n_trials,
                        exc,
                    )
            if progress is not None and progress(done, n_trials, "hpo study") is False:
                raise JobCancelled(
                    f"hpo study cancelled after {done}/{n_trials} trials -- "
                    "stopped at a checkpoint by the caller",
                    partial_results=TuneReport(
                        best_params=dict(study.best().params) if study.trials else {},
                        best_score=-float(study.best().value or math.inf) if study.trials else -math.inf,
                        n_trials=len(study.trials),
                        history=list(study.trials),
                        task=task,
                    ),
                )
    else:
        # the checkpoint-less path: the historical loop, plus the
        # round-205 progress contract
        for i in range(n_trials):
            trial = study._engine.ask_and_run(objective)
            study.history.append(trial)
            if (
                progress is not None
                and progress(i + 1, n_trials, "hpo study") is False
            ):
                from pharmaml.exceptions import JobCancelled

                raise JobCancelled(
                    f"hpo study cancelled after {i + 1}/{n_trials} trials -- "
                    "stopped at a checkpoint by the caller"
                )

    best = study.best()
    return TuneReport(
        best_params=dict(best.params),
        best_score=-float(best.value or math.inf),
        n_trials=len(study.trials),
        history=list(study.trials),
        task=task,
    )


# -- the pruner verdict (SHA's should_prune, standalone) --------------------- #


def halving_verdict(
    value: float, better_half: Sequence[float], patience: int = 0
) -> Literal["CONTINUE", "PRUNE"]:
    """The successive-halving ``should_prune`` answer for one running trial.

    A trial is pruned when its partial value is worse than every entry
    of the surviving cohort (``better_half`` -- the promoted group of
    the last rung).  ``patience`` lets a trial survive that many
    consecutive prunable evaluations before the axe, the house's
    safety valve against noisy intermediate budgets.

    >>> halving_verdict(0.9, [0.1, 0.3])
    'PRUNE'
    >>> halving_verdict(0.2, [0.1, 0.3])
    'CONTINUE'
    """
    if not better_half:
        return "CONTINUE"
    worse_than_all = all(value > b for b in better_half)
    if worse_than_all and patience <= 0:
        return "PRUNE"
    return "CONTINUE"
