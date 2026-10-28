"""REINVENT4's diversity filters, ported to the toolkit's stack.

The instruction (round 100x-164): *"Check the contents and files in
this GitHub link completely and one by one and add the missing items to
playground.py: github.com/MolecularAI/REINVENT4."*  The repository was
cloned and walked file by file; this module is the port of
``reinvent/runmodes/RL/memories/`` -- the mechanism that keeps an RL
(or GA, or MCTS) run from filling its leaderboard with one lucky
scaffold's children:

* :class:`BucketCounter` -- the exact ``bucket_counter.py`` semantics
  (a ``Counter`` whose buckets are *unlimited* in size; ``bucket_full``
  means ``count > max_size``, strictly greater);
* the **five filters** -- :class:`IdenticalMurckoScaffoldFilter`
  (``identical_murcko_scaffold.py``), :class:`IdenticalTopologicalScaffoldFilter`
  (the ``MakeScaffoldGeneric`` variant), :class:`ScaffoldSimilarityFilter`
  (``scaffold_similarity.py``: atom-pair fingerprints, **Dice**
  similarity, the closest memorised scaffold replaces the new one when
  similarity >= ``minsimilarity`` -- collapsing near-duplicate
  scaffolds into shared buckets), :class:`PenalizeSameSmilesFilter`
  (``penalize_same_smiles.py``: every *repeat* SMILES scores
  ``score * penalty_multiplier``; no scaffolds, no buckets) and
  :class:`NoFilter` (the omitted-``[diversity_filter]``-section
  behaviour);
* the **five bucket penalty functions** from
  ``runmodes/RL/intrinsic_penalty/penalties/`` -- ``step`` (the classic
  REINVENT4 zero-when-full), ``linear``, ``sigmoid``, ``tanh`` and
  ``erf``, each an exact formula port, selected via
  ``penalty_function=``;
* :class:`DiversityFilteredObjective` -- the toolkit integration: a
  *stateful wrapper around any*
  :class:`~pharmaml.generative.ScoringObjective` that applies
  ``filter.update_score`` to every scored batch.  Because every
  optimiser in :mod:`pharmaml.generative` (GA / RL / MCTS) scores its
  candidates through the objective, wrapping it reproduces REINVENT4's
  scaffold memory across the whole run -- with the stats
  (``scaffold_stats``: buckets, full buckets, memorised SMILES,
  penalised scores) surfaced for the result reports.

The REINVENT4 scoring flow (``diversity_filter.py::score_scaffolds``,
reproduced exactly):

1. compute the Murcko scaffold of the (valid) SMILES -- generic
   (all-carbon) for the topological variant;
2. ``ScaffoldSimilarity`` only: replace the scaffold with the closest
   memorised scaffold when atom-pair Dice >= ``minsimilarity``;
3. a SMILES already in ``smiles_memory`` scores ``0.0`` (the global
   duplicate penalty);
4. a score >= ``minscore`` (default 0.4) adds the SMILES to the memory
   and increments its scaffold bucket -- and the bucket count ``n``
   feeds the penalty function: ``step`` (the default) is 1.0 until
   the bucket overflows and 0.0 after -- REINVENT4's classic
   behaviour bit-exactly -- while the four soft penalties (the
   ``intrinsic_penalty`` system's own functions) decay smoothly with
   ``n``.

Examples
--------
>>> from pharmaml.generative.diversity import (
...     IdenticalMurckoScaffoldFilter, make_diversity_filter)
>>> f = IdenticalMurckoScaffoldFilter(bucket_size=2, minscore=0.5)
>>> scores = [0.9, 0.9, 0.9, 0.9, 0.9]         # five equally good hits
>>> smilies = ["CCO"] * 5                       # ... the same molecule
>>> updated = f.update_score(scores, smilies)
>>> updated.tolist()                            # repeats zero out
[0.9, 0.0, 0.0, 0.0, 0.0]

A full *scaffold* bucket -- the mechanism that actually forces
diversity (same scaffold, different molecules):

>>> f2 = IdenticalMurckoScaffoldFilter(bucket_size=1, minscore=0.5)
>>> benzene_kids = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"]
>>> updated = f2.update_score([0.9, 0.9, 0.9], benzene_kids)
>>> [round(v, 6) for v in updated]
[0.9, 0.0, 0.0]

The factory -- REINVENT4's ``[diversity_filter] type = ...`` TOML key:

>>> f3 = make_diversity_filter("ScaffoldSimilarity", bucket_size=25)
>>> f3.minsimilarity
0.4
>>> make_diversity_filter("no-such-filter")
Traceback (most recent call last):
    ...
pharmaml.exceptions.DataValidationError: no diversity filter named 'no-such-filter' -- the registry holds: identical_murcko_scaffold, identical_topological_scaffold, no_filter, penalize_same_smiles, scaffold_similarity

The objective wrapper -- a constant objective through an
identical-Murcko filter:

>>> from pharmaml.generative.diversity import DiversityFilteredObjective
>>> from pharmaml.generative import ScoringObjective
>>> class Lucky(ScoringObjective):
...     name = "lucky"
...     def score(self, smiles):
...         return 0.9
>>> wrapped = DiversityFilteredObjective(
...     Lucky(), "identical_murcko_scaffold", bucket_size=1, minscore=0.5)
>>> wrapped.score("CCO"), wrapped.score("CCO")
(0.9, 0.0)
>>> wrapped.scaffold_stats()["n_memorized_smiles"]
1
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

from .scoring import ScoringObjective

_LOG = get_logger(__name__)

__all__ = [
    "DIVERSITY_FILTERS",
    "SCAFFOLD_PENALTY_FUNCTIONS",
    "BucketCounter",
    "DiversityFilter",
    "DiversityFilteredObjective",
    "IdenticalMurckoScaffoldFilter",
    "IdenticalTopologicalScaffoldFilter",
    "NoFilter",
    "PenalizeSameSmilesFilter",
    "ScaffoldSimilarityFilter",
    "make_diversity_filter",
    "scaffold_penalty_erf",
    "scaffold_penalty_linear",
    "scaffold_penalty_sigmoid",
    "scaffold_penalty_step",
    "scaffold_penalty_tanh",
]


# ---------------------------------------------------------------------------
# The bucket counter (runmodes/RL/memories/bucket_counter.py)
# ---------------------------------------------------------------------------


class BucketCounter(Counter):  # type: ignore[type-arg]
    """REINVENT4's bucket counter: unlimited buckets, fullness checks.

    A ``Counter`` subclass whose ``add`` increments a bucket and whose
    ``bucket_full`` reports ``count > max_size`` (**strictly** greater
    -- REINVENT4's own off-by-one, pinned here by their test suite and
    ours).  The buckets themselves are never limited in size; only the
    fullness *flag* exists.
    """

    def __init__(self, max_size: int = 10, *args: Any, **kwargs: Any) -> None:
        if not isinstance(max_size, int) or isinstance(max_size, bool) or max_size < 1:
            raise DataValidationError(
                f"bucket_size must be a positive int, got {max_size!r}"
            )
        super().__init__(*args, **kwargs)
        self.max_size = max_size

    def add(self, item: Hashable) -> None:
        """Add one item to its (unlimited) bucket."""
        self[item] += 1

    def bucket_full(self, item: Hashable) -> bool:
        """Whether the item's bucket exceeds ``max_size``."""
        return self[item] > self.max_size

    def bucket_count(self, item: Hashable) -> int:
        """Number of items in the item's bucket."""
        return self[item]

    def count_full(self) -> int:
        """How many buckets are above the maximum size."""
        return sum(1 for count in self.values() if count > self.max_size)

    def __reduce__(self) -> Any:
        # REINVENT4 pickles the counter into agent checkpoints; the
        # toolkit keeps the same reconstruction protocol.
        return self.__class__, (self.max_size, dict(self))


# ---------------------------------------------------------------------------
# The bucket penalty functions (intrinsic_penalty/penalties/*.py)
# ---------------------------------------------------------------------------


def scaffold_penalty_step(bucket_count: int, bucket_size: int) -> float:
    """``StepPenalty``: 0.0 when the bucket is full, else 1.0.

    The classic REINVENT4 behaviour (the default everywhere the
    ``intrinsic_penalty`` system is not configured).
    """
    return 0.0 if bucket_count > bucket_size else 1.0


def scaffold_penalty_linear(bucket_count: int, bucket_size: int) -> float:
    """``LinearPenalty``: ``max(0, 1 - n / B)``."""
    return max(0.0, 1.0 - bucket_count / bucket_size)


def scaffold_penalty_sigmoid(bucket_count: int, bucket_size: int) -> float:
    """``SigmoidPenalty``: ``1 - 1 / (1 + exp(-((2n/B - 1) / 0.15)))``."""
    exponent = bucket_count / bucket_size
    exponent *= 2.0
    exponent -= 1.0
    exponent /= 0.15
    exponent *= -1.0
    return 1.0 - 1.0 / (1.0 + math.exp(exponent))


def scaffold_penalty_tanh(bucket_count: int, bucket_size: int) -> float:
    """``TanhPenalty``: ``1 - tanh(3 (n - 1) / B)``."""
    return 1.0 - math.tanh(3.0 * (bucket_count - 1) / bucket_size)


def scaffold_penalty_erf(bucket_count: int, bucket_size: int) -> float:
    """``ErfPenalty``: ``1 + erf(sqrt(pi)/B) - erf(sqrt(pi) n / B)``."""
    return 1.0 + math.erf(math.sqrt(math.pi) / bucket_size) - math.erf(
        math.sqrt(math.pi) * bucket_count / bucket_size
    )


#: The penalty-function registry (``intrinsic_penalty`` ``penalty_function``
#: key): ``n`` is the bucket's instance count, ``B`` the bucket size.
SCAFFOLD_PENALTY_FUNCTIONS: dict[str, Callable[[int, int], float]] = {
    "step": scaffold_penalty_step,
    "linear": scaffold_penalty_linear,
    "sigmoid": scaffold_penalty_sigmoid,
    "tanh": scaffold_penalty_tanh,
    "erf": scaffold_penalty_erf,
}


# ---------------------------------------------------------------------------
# The diversity filters (runmodes/RL/memories/*.py)
# ---------------------------------------------------------------------------


def _murcko_scaffold(smiles: str, topological: bool) -> str:
    """Murcko scaffold SMILES (generic when ``topological``)."""
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if topological:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=scaffold)
    except Exception:  # noqa: BLE001 - hostile SMILES must not crash a run
        return ""


class DiversityFilter(ABC):
    """Base class: the scaffold memory every concrete filter shares.

    Parameters
    ----------
    bucket_size:
        Scaffold bucket size (REINVENT4's TOML default 25).
    minscore:
        The score threshold above which a SMILES is *memorised*
        (REINVENT4's default 0.4) -- below it the molecule never enters
        the memory and is never penalised either.
    minsimilarity:
        The ``ScaffoldSimilarity`` atom-pair Dice threshold (default
        0.4); ignored by the other filters.
    penalty_multiplier:
        The ``PenalizeSameSmiles`` repeat multiplier (default 0.5).
    penalty_function:
        One of :data:`SCAFFOLD_PENALTY_FUNCTIONS` -- ``"step"`` (the
        classic zero-when-full REINVENT4 behaviour) or one of the four
        soft penalties from the ``intrinsic_penalty`` system.
    """

    def __init__(
        self,
        bucket_size: int = 25,
        minscore: float = 0.4,
        minsimilarity: float = 0.4,
        penalty_multiplier: float = 0.5,
        penalty_function: str = "step",
    ) -> None:
        if penalty_function not in SCAFFOLD_PENALTY_FUNCTIONS:
            known = ", ".join(sorted(SCAFFOLD_PENALTY_FUNCTIONS))
            raise DataValidationError(
                f"penalty_function must be one of: {known} (got "
                f"{penalty_function!r})"
            )
        for label, value in (
            ("minscore", minscore),
            ("minsimilarity", minsimilarity),
            ("penalty_multiplier", penalty_multiplier),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise DataValidationError(
                    f"{label} must be a finite number, got {value!r}"
                )
        if not 0 <= minscore <= 1:
            raise DataValidationError(f"minscore must be in [0, 1], got {minscore!r}")
        if not 0 <= minsimilarity <= 1:
            raise DataValidationError(
                f"minsimilarity must be in [0, 1], got {minsimilarity!r}"
            )
        if not 0 < penalty_multiplier <= 1:
            raise DataValidationError(
                f"penalty_multiplier must be in (0, 1], got {penalty_multiplier!r}"
            )
        self.bucket_size = int(bucket_size)
        self.minscore = float(minscore)
        self.minsimilarity = float(minsimilarity)
        self.penalty_multiplier = float(penalty_multiplier)
        self.penalty_function = str(penalty_function)
        self.scaffold_memory = BucketCounter(self.bucket_size)
        self.smiles_memory: set[str] = set()
        self.n_zeroed = 0
        self.n_penalized = 0

    # -- the REINVENT4 scoring flow ----------------------------------------
    def update_score(
        self, scores: Sequence[float], smilies: Sequence[str]
    ) -> np.ndarray:
        """Return the penalised copy of ``scores`` for ``smilies``.

        The order matches REINVENT4's ``update_score(scores, smilies,
        mask)`` with an all-True mask (the toolkit's objectives score
        only what the optimiser feeds them); invalid SMILES keep their
        incoming score (they scored 0.0 upstream already).
        """
        updated = np.asarray(scores, dtype=np.float64).copy()
        if len(updated) != len(smilies):
            raise DataValidationError(
                f"{len(updated)} scores but {len(smilies)} SMILES -- the "
                "two lists must match"
            )
        self._update_score(updated, list(smilies))
        return updated

    @abstractmethod
    def _update_score(self, scores: np.ndarray, smilies: list[str]) -> None:
        """In-place update by the concrete filter."""

    def score_scaffolds(
        self, scores: np.ndarray, smilies: list[str], topological: bool
    ) -> list[str | None]:
        """The ``score_scaffolds`` flow, reproduced exactly.

        Returns the per-SMILES scaffold (``None`` where the SMILES is
        unparseable -- REINVENT4's ``[None] * len(smilies)`` initial
        state, only overwritten for valid entries).
        """
        scaffolds: list[str | None] = [None] * len(smilies)
        penalty = SCAFFOLD_PENALTY_FUNCTIONS[self.penalty_function]
        for i, smiles in enumerate(smilies):
            if not isinstance(smiles, str) or not smiles.strip():
                continue  # invalid: score untouched, scaffold stays None
            scaffold: str | None = _murcko_scaffold(smiles, topological)
            if isinstance(self, ScaffoldSimilarityFilter):
                scaffold = self._find_similar_scaffold(scaffold)
            if smiles in self.smiles_memory:
                scores[i] = 0.0
                self.n_zeroed += 1
            scaffolds[i] = scaffold
            if scores[i] >= self.minscore:
                self.smiles_memory.add(smiles)
                self.scaffold_memory.add(scaffold)
                # the bucket count feeds the penalty function at every
                # memorisation: "step" is 1.0 until the bucket overflows
                # and 0.0 after (REINVENT4's classic behaviour,
                # bit-exactly); the soft penalties (the intrinsic-
                # penalty system's functions) decay smoothly with n
                multiplier = penalty(
                    self.scaffold_memory.bucket_count(scaffold),
                    self.scaffold_memory.max_size,
                )
                if multiplier < 1.0:
                    scores[i] = float(scores[i]) * multiplier
                    self.n_penalized += 1
                    if multiplier == 0.0:
                        self.n_zeroed += 1
        return scaffolds

    # -- the reporting surface ---------------------------------------------
    def scaffold_stats(self) -> dict[str, Any]:
        """The memory's summary for run reports."""
        return {
            "filter": type(self).__name__,
            "n_scaffolds": len(self.scaffold_memory),
            "n_full_buckets": self.scaffold_memory.count_full(),
            "n_memorized_smiles": len(self.smiles_memory),
            "n_zeroed": self.n_zeroed,
            "n_penalized": self.n_penalized,
        }

    def reset(self) -> None:
        """Purge the memory (REINVENT4's ``purge_memories`` between stages)."""
        self.scaffold_memory = BucketCounter(self.bucket_size)
        self.smiles_memory = set()
        self.n_zeroed = 0
        self.n_penalized = 0


class IdenticalMurckoScaffoldFilter(DiversityFilter):
    """Penalise repeated Murcko scaffolds (``identical_murcko_scaffold``)."""

    def _update_score(self, scores: np.ndarray, smilies: list[str]) -> None:
        self.score_scaffolds(scores, smilies, topological=False)


class IdenticalTopologicalScaffoldFilter(DiversityFilter):
    """The generic-scaffold variant (``identical_topological_scaffold``).

    ``MakeScaffoldGeneric`` flattens every atom to carbon and drops
    bond orders, so an oxazole and a benzene share one bucket.
    """

    def _update_score(self, scores: np.ndarray, smilies: list[str]) -> None:
        self.score_scaffolds(scores, smilies, topological=True)


class ScaffoldSimilarityFilter(DiversityFilter):
    """Bucket by *similar* scaffolds (``scaffold_similarity``).

    Atom-pair fingerprints of every memorised scaffold; a new scaffold
    whose **Dice** similarity to the closest memorised one is at least
    ``minsimilarity`` joins *that* bucket instead of opening a new one
    -- near-duplicate scaffolds collapse together.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scaffold_fingerprints: dict[str, Any] = {}

    def _update_score(self, scores: np.ndarray, smilies: list[str]) -> None:
        self.score_scaffolds(scores, smilies, topological=False)

    def _find_similar_scaffold(self, scaffold: str | None) -> str | None:
        """Replace ``scaffold`` with its closest memorised neighbour."""
        if not scaffold:
            return scaffold
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator

        # REINVENT4 pairs Pairs.GetAtomPairFingerprint (the count-based
        # atom-pair fingerprint, now deprecated) with BulkDiceSimilarity;
        # the toolkit's precedent is the modern generator API (the
        # SimilarityObjective's GetMorganGenerator switch) -- same
        # atom-pair fingerprint, bit-vector form, same Dice threshold
        # semantics, no per-call deprecation warning.
        generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
        mol = Chem.MolFromSmiles(scaffold)
        if mol is None:  # pragma: no cover - _murcko_scaffold round-trips
            return scaffold
        fp = generator.GetFingerprint(mol)
        memorised = list(self.scaffold_fingerprints.values())
        if memorised:
            similarities = DataStructs.BulkDiceSimilarity(fp, memorised)
            closest = int(np.argmax(similarities))
            if similarities[closest] >= self.minsimilarity:
                scaffold = list(self.scaffold_fingerprints.keys())[closest]
                fp = self.scaffold_fingerprints[scaffold]
        self.scaffold_fingerprints[scaffold] = fp
        return scaffold


class PenalizeSameSmilesFilter(DiversityFilter):
    """Multiply every *repeat* SMILES (``penalize_same_smiles``).

    No scaffolds, no buckets -- every valid SMILES enters the memory
    and a re-occurrence scores ``score * penalty_multiplier``.
    """

    def _update_score(self, scores: np.ndarray, smilies: list[str]) -> None:
        for i, smiles in enumerate(smilies):
            if not isinstance(smiles, str) or not smiles.strip():
                continue
            if smiles in self.smiles_memory:
                scores[i] *= self.penalty_multiplier
                self.n_penalized += 1
            self.smiles_memory.add(smiles)


class NoFilter(DiversityFilter):
    """The omitted-``[diversity_filter]`` behaviour: score untouched."""

    def _update_score(self, scores: np.ndarray, smilies: list[str]) -> None:
        # REINVENT4's NoFilter still records the SMILES seen (the
        # CSV reporter's duplicate accounting); the toolkit keeps the
        # memory fill without ever touching a score.
        for i, smiles in enumerate(smilies):
            if not isinstance(smiles, str) or not smiles.strip():
                continue
            if scores[i] >= self.minscore:
                self.smiles_memory.add(smiles)


#: The filter registry (REINVENT4's ``[diversity_filter] type = ...`` key,
#: lowercased with underscores, exactly their TOML spellings).
DIVERSITY_FILTERS: dict[str, type[DiversityFilter]] = {
    "identical_murcko_scaffold": IdenticalMurckoScaffoldFilter,
    "identical_topological_scaffold": IdenticalTopologicalScaffoldFilter,
    "scaffold_similarity": ScaffoldSimilarityFilter,
    "penalize_same_smiles": PenalizeSameSmilesFilter,
    "no_filter": NoFilter,
}


def make_diversity_filter(
    kind: str,
    bucket_size: int = 25,
    minscore: float = 0.4,
    minsimilarity: float = 0.4,
    penalty_multiplier: float = 0.5,
    penalty_function: str = "step",
) -> DiversityFilter:
    """Build a diversity filter by its REINVENT4 TOML ``type`` key.

    ``kind`` normalises exactly like the library's other registries:
    lowercased, dashes swapped for underscores -- ``"IdenticalMurckoScaffold"``,
    ``"identical_murcko_scaffold"`` and ``"identical-murcko-scaffold"``
    all resolve.

    Example
    -------
    >>> f = make_diversity_filter("ScaffoldSimilarity", bucket_size=10)
    >>> type(f).__name__
    'ScaffoldSimilarityFilter'
    """

    if not isinstance(kind, str) or not kind.strip():
        raise DataValidationError("filter kind must be a non-empty string")
    # separator-free lookup so the REINVENT4 TOML spelling
    # ("scaffold_similarity"), the class name ("ScaffoldSimilarity")
    # and dashed forms all resolve to the one registry entry
    normalized = kind.lower().replace("_", "").replace("-", "")
    lookup = {
        key.lower().replace("_", "").replace("-", ""): key
        for key in DIVERSITY_FILTERS
    }
    if normalized not in lookup:
        known = ", ".join(sorted(DIVERSITY_FILTERS))
        raise DataValidationError(
            f"no diversity filter named {kind!r} -- the registry holds: {known}"
        )
    return DIVERSITY_FILTERS[lookup[normalized]](
        bucket_size=bucket_size,
        minscore=minscore,
        minsimilarity=minsimilarity,
        penalty_multiplier=penalty_multiplier,
        penalty_function=penalty_function,
    )


class DiversityFilteredObjective(ScoringObjective):
    """Any objective through a REINVENT4 diversity filter.

    A *stateful* wrapper: every :meth:`score` call runs the wrapped
    objective and then the filter's ``update_score`` on that single
    SMILES -- so the scaffold memory accumulates across an entire
    optimiser run (GA / RL / MCTS all score their candidates through
    the objective, which is exactly where REINVENT4 hooks its filters).
    The optimiser itself needs no change.

    Parameters
    ----------
    objective:
        Any :class:`~pharmaml.generative.ScoringObjective`.
    filter_kind:
        A :data:`DIVERSITY_FILTERS` key (or ``"no_filter"`` for the
        pass-through that still records the memory).
    filter_params:
        Keyword parameters forwarded to :func:`make_diversity_filter`
        (``bucket_size``, ``minscore``, ``minsimilarity``,
        ``penalty_multiplier``, ``penalty_function``).

    Example
    -------
    >>> from pharmaml.generative import QedObjective
    >>> wrapped = DiversityFilteredObjective(
    ...     QedObjective(), "penalize_same_smiles", penalty_multiplier=0.5)
    >>> round(wrapped.score("CCO"), 6)       # first occurrence: full QED
    0.406808
    >>> round(wrapped.score("CCO"), 6)       # the repeat halves
    0.203404
    """

    def __init__(
        self,
        objective: ScoringObjective,
        filter_kind: str = "no_filter",
        **filter_params: Any,
    ) -> None:
        super().__init__()
        if not hasattr(objective, "score") or not callable(objective.score):
            raise DataValidationError(
                "objective must expose a callable score() method "
                f"(got {objective!r})"
            )
        self.objective = objective
        self.filter = make_diversity_filter(filter_kind, **filter_params)
        self.name = str(getattr(objective, "name", "objective"))

    @property
    def n_scored(self) -> int:
        """The wrapped objective's scoring counter (it counts, not us)."""
        return int(getattr(self.objective, "n_scored", 0))

    @property
    def n_invalid(self) -> int:
        """The wrapped objective's invalid counter (it parses, not us)."""
        return int(getattr(self.objective, "n_invalid", 0))

    def reset_stats(self) -> None:
        """Reset the wrapped objective's counters (NOT the filter memory --
        clearing scaffold memory mid-run would forget what the run has
        already seen; use :meth:`reset_filter` deliberately)."""
        reset = getattr(self.objective, "reset_stats", None)
        if callable(reset):
            reset()

    def reset_filter(self) -> None:
        """Purge the diversity filter's memory (the stage boundary)."""
        self.filter.reset()

    def score(self, smiles: str) -> float:
        base = float(self.objective.score(smiles))
        updated = self.filter.update_score([base], [str(smiles)])
        return float(min(max(updated[0], 0.0), 1.0))

    def score_many(self, smiles_list: Sequence[str]) -> list[float]:
        """Score a batch, then apply the filter to the whole batch.

        One ``update_score`` call for the batch -- REINVENT4's own
        granularity (its RL loop filters per sampled batch, not per
        single molecule; duplicates *within* one batch are penalised
        together).
        """
        base = [float(self.objective.score(smiles)) for smiles in smiles_list]
        updated = self.filter.update_score(base, [str(s) for s in smiles_list])
        return [float(min(max(v, 0.0), 1.0)) for v in updated]

    def scaffold_stats(self) -> dict[str, Any]:
        """The filter's memory summary (for the run report)."""
        return self.filter.scaffold_stats()
