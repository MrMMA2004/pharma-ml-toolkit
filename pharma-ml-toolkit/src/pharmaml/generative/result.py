"""Result container for every generation run (GA and RL alike)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = ["GenerationResult"]

#: Leaderboard columns ``to_dataframe()`` always provides; a breakdown key
#: colliding with one of them would silently CLOBBER the column (the
#: literal ``"score"`` key replacing the total-score column) -- a typed
#: construction-time error instead.
_RESERVED_COLUMNS = ("rank", "smiles", "score")


@dataclass
class GenerationResult:
    """Ranked molecules from one generative optimisation run.

    Attributes
    ----------
    smiles:
        Deduplicated **canonical** SMILES, best first.
    scores:
        Total objective score of each molecule (aligned with ``smiles``).
    breakdown:
        Per-objective scores (aligned with ``smiles``); empty dicts when
        the objective was not a :class:`~pharmaml.generative.MultiObjective`.
    best:
        Best-scoring SMILES (``""`` when nothing valid was generated).
    best_score:
        Score of ``best`` (``0.0`` when nothing valid was generated).
    history:
        One record per generation / RL iteration with the run's progress
        statistics (best / mean score, valid fraction, ...).
    metadata:
        Method, seed, iterations, elapsed seconds, valid / unique counts.

    Raises
    ------
    DataValidationError
        If ``smiles`` / ``scores`` / ``breakdown`` lengths disagree --
        hand-built results must stay row-aligned like the optimisers'
        own output -- or a breakdown key collides with a reserved
        leaderboard column (``"rank"`` / ``"smiles"`` / ``"score"``).

    Example
    -------
    >>> from pharmaml.generative import GenerationResult
    >>> result = GenerationResult(
    ...     smiles=["CCO", "CCN"], scores=[0.5, 0.4],
    ...     breakdown=[{"qed": 0.5, "novelty": 0.9}, {"qed": 0.4, "novelty": 0.6}],
    ... )
    >>> result.best, round(result.best_score, 1)
    ('CCO', 0.5)
    >>> result.to_dataframe()   # ranked leaderboard (rank 1 = best)
       rank smiles  score  qed  novelty
    0     1    CCO    0.5  0.5      0.9
    1     2    CCN    0.4  0.4      0.6
    """

    smiles: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    breakdown: list[dict[str, float]] = field(default_factory=list)
    best: str = ""
    best_score: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.smiles)
        if len(self.scores) != n:
            raise DataValidationError(
                f"GenerationResult is misaligned: {n} smiles but "
                f"{len(self.scores)} score(s) -- the two lists must have "
                "the same length"
            )
        if self.breakdown and len(self.breakdown) != n:
            raise DataValidationError(
                f"GenerationResult is misaligned: {n} smiles but "
                f"{len(self.breakdown)} breakdown row(s) -- the breakdown "
                "must have one dict per molecule"
            )
        if not self.breakdown:
            # one independent dict per row (a shared ``[{}] * n`` would alias)
            self.breakdown = [{} for _ in range(n)]
        for entry in self.breakdown:
            for key in entry:
                if key in _RESERVED_COLUMNS:
                    raise DataValidationError(
                        f"breakdown key {key!r} collides with the reserved "
                        f"to_dataframe() column {key!r} "
                        f"({', '.join(map(repr, _RESERVED_COLUMNS))}) -- the "
                        "breakdown entry would silently clobber the "
                        "leaderboard column; rename the objective (its "
                        "``name`` attribute) to a non-reserved key"
                    )
        if not self.smiles:
            self.best, self.best_score = "", 0.0
        elif not self.best:
            self.best, self.best_score = self.smiles[0], self.scores[0]

    # -- convenience -----------------------------------------------------
    def __len__(self) -> int:
        """Number of ranked molecules."""
        return len(self.smiles)

    def is_empty(self) -> bool:
        """True when no valid molecule was generated."""
        return not self.smiles

    def to_dataframe(self) -> pd.DataFrame:
        """Leaderboard table: rank, SMILES, total score (+ per-objective).

        Columns are ``rank`` (1 = best), ``smiles``, ``score`` and one
        column per sub-objective when a breakdown is available.
        """
        frame = pd.DataFrame(
            {
                "rank": range(1, len(self.smiles) + 1),
                "smiles": self.smiles,
                "score": self.scores,
            }
        )
        keys: list[str] = []
        for entry in self.breakdown:
            for key in entry:
                if key not in keys:
                    keys.append(key)
        for key in keys:
            frame[key] = [entry.get(key, float("nan")) for entry in self.breakdown]
        return frame

    def summary(self) -> str:
        """One-line human summary of the run (CLI / log friendly)."""
        method = self.metadata.get("method", "generation")
        iterations = self.metadata.get("iterations", len(self.history))
        elapsed = self.metadata.get("elapsed_s", float("nan"))
        return (
            f"{method}: {len(self.smiles)} unique molecule(s) "
            f"from {iterations} iteration(s) in {elapsed:.1f}s -- "
            f"best {self.best!r} (score {self.best_score:.3f})"
        )
