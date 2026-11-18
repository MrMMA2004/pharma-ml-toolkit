"""Result container for every generation run (GA and RL alike)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = ["GenerationResult", "SAFilteredCandidates"]

#: Leaderboard columns ``to_dataframe()`` always provides; a breakdown key
#: colliding with one of them would silently CLOBBER the column (the
#: literal ``"score"`` key replacing the total-score column) -- a typed
#: construction-time error instead.
_RESERVED_COLUMNS = ("rank", "smiles", "score")

#: The raw-score column ``sa_frame()`` appends (part3 section 24.3).  A
#: breakdown whose MultiObjective included
#: :class:`~pharmaml.generative.SAScoreObjective` legitimately carries a
#: ``"sa_score"`` *reward* key -- sa_frame renames that column to
#: ``"sa_score_reward"`` instead of letting the two silently clobber.
_SA_RAW_COLUMN = "sa_score"
_SA_REWARD_COLUMN = "sa_score_reward"


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

    The part3 section 24.3/24.4 synthesizability surface rides the same
    container -- the *raw* Ertl SA score (never the optimization reward):

    >>> picked = result.filter_by_sa(max_sa=2.0)   # CCO 1.98 in, CCN 2.174 out
    >>> (picked.n_kept, picked.n_rejected, picked.n_total)
    (1, 1, 2)
    >>> picked.kept.smiles
    ['CCO']
    >>> round(result.sa_frame(sort="easiest").loc[0, "sa_score"], 2)
    1.98
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

    # -- part3 section 24.3/24.4: synthesizability surface -----------------
    # The raw Ertl SA score ([1, 10], lower = easier to synthesise), kept
    # strictly distinct from the [0, 1] optimization reward the
    # SAScoreObjective breakdown records -- the spec's "clearly distinguish
    # between SA score filtering, ranking, and optimization objective".

    def sa_frame(self, sort: str = "rank") -> pd.DataFrame:
        """The section 24.3 results table: leaderboard + raw SA score.

        Columns are ``to_dataframe()``'s (``rank`` / ``smiles`` /
        ``score`` + one per breakdown objective) plus ``sa_score`` (the
        raw Ertl value in ``[1, 10]``; ``NaN`` for invalid SMILES) and
        ``sa_valid``.  ``sort`` is one of:

        * ``"rank"`` (default) -- the run's own objective order;
        * ``"easiest"`` -- raw SA ascending (invalid rows last);
        * ``"hardest"`` -- raw SA descending (invalid rows last).

        The ``rank`` column always shows the *objective* rank, so a
        sorted table still reveals how the two orderings disagree.

        A breakdown that legitimately carries the SAScoreObjective's
        ``"sa_score"`` *reward* key gets that column renamed to
        ``sa_score_reward`` first -- the raw score owns the name, and
        neither column silently clobbers the other.

        Raises
        ------
        pharmaml.exceptions.DataValidationError
            On an unknown ``sort`` key.
        pharmaml.exceptions.DependencyError
            When RDKit (or its Contrib SA_Score database) is unavailable.

        Example
        -------
        >>> from pharmaml.generative import GenerationResult
        >>> result = GenerationResult(
        ...     smiles=["CCN", "CCO", "not-a-smiles"], scores=[0.4, 0.5, 0.0])
        >>> frame = result.sa_frame(sort="easiest")
        >>> list(frame["smiles"])
        ['CCO', 'CCN', 'not-a-smiles']
        >>> [round(v, 2) if v == v else None for v in frame["sa_score"]]
        [1.98, 2.17, None]
        >>> list(frame["sa_valid"])
        [True, True, False]
        >>> result.sa_frame(sort="nope")
        Traceback (most recent call last):
        ...
        pharmaml.exceptions.DataValidationError: sort must be one of 'rank', 'easiest', 'hardest' -- got 'nope'
        """
        from pharmaml.chem.sa_scoring import sa_scores

        if sort not in ("rank", "easiest", "hardest"):
            raise DataValidationError(
                "sort must be one of 'rank', 'easiest', 'hardest' -- got "
                f"{sort!r}"
            )
        frame = self.to_dataframe()
        if _SA_RAW_COLUMN in frame.columns:  # the reward breakdown's key
            frame = frame.rename(columns={_SA_RAW_COLUMN: _SA_REWARD_COLUMN})
        batch = sa_scores(self.smiles)
        frame[_SA_RAW_COLUMN] = [
            r.score if r.valid else float("nan") for r in batch.results
        ]
        frame["sa_valid"] = [r.valid for r in batch.results]
        if sort != "rank":
            frame = frame.sort_values(
                _SA_RAW_COLUMN, ascending=(sort == "easiest"), na_position="last"
            ).reset_index(drop=True)
        return frame

    def filter_by_sa(
        self,
        min_sa: float | None = None,
        max_sa: float | None = None,
    ) -> SAFilteredCandidates:
        """Section 24.4 candidate filtering on the raw SA score.

        ``max_sa`` is the "maximum acceptable SA score" constraint,
        ``min_sa`` the "minimum required score" one; both bounds are
        inclusive.  With neither set there is **no SA filtering** -- the
        run's own result is returned unchanged (spec: candidates are
        never discarded unless the user explicitly enables filtering).

        Both sides are preserved: :attr:`SAFilteredCandidates.kept` is a
        new :class:`GenerationResult` holding the accepted rows (in the
        run's own rank order, breakdown intact, the filter recorded in
        its ``metadata["sa_filter"]``) and
        :attr:`SAFilteredCandidates.rejected` a frame of ``rank`` /
        ``smiles`` / ``score`` / ``sa_score`` / ``reason`` rows -- the
        "preserve rejected candidate information where appropriate for
        later analysis" rule.  ``n_kept + n_rejected == n_total`` always.

        Invalid SMILES are rejected (with that reason) only when a
        constraint is active -- they carry no score to compare.

        Raises
        ------
        pharmaml.exceptions.DataValidationError
            When ``min_sa`` exceeds ``max_sa`` or a bound is not a finite
            number.
        pharmaml.exceptions.DependencyError
            When RDKit (or its Contrib SA_Score database) is unavailable.

        Example
        -------
        >>> from pharmaml.generative import GenerationResult
        >>> result = GenerationResult(
        ...     smiles=["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CCO"], scores=[0.9, 0.5])
        >>> picked = result.filter_by_sa(max_sa=2.0)
        >>> picked.kept.smiles
        ['CCO']
        >>> picked.rejected.loc[0, "reason"]
        'sa_score 2.298 above max_sa 2.0'
        >>> result.filter_by_sa().kept is result    # no constraints: no-op
        True
        """
        from pharmaml.chem.sa_scoring import sa_scores

        for bound, label in ((min_sa, "min_sa"), (max_sa, "max_sa")):
            if bound is not None and (
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(bound)
            ):
                raise DataValidationError(
                    f"{label}={bound!r} must be a finite number or None"
                )
        if (
            min_sa is not None
            and max_sa is not None
            and min_sa > max_sa
        ):
            raise DataValidationError(
                f"min_sa ({min_sa}) must not exceed max_sa ({max_sa})"
            )
        if min_sa is None and max_sa is None:
            return SAFilteredCandidates(
                kept=self, rejected=_empty_rejected_frame()
            )
        batch = sa_scores(self.smiles)
        kept_rows: list[int] = []
        rejected_rows: list[dict[str, Any]] = []
        for i, scored in enumerate(batch.results):
            if not scored.valid or scored.score is None:
                rejected_rows.append(
                    {
                        "rank": i + 1,
                        "smiles": scored.smiles,
                        "score": self.scores[i],
                        "sa_score": float("nan"),
                        "reason": "invalid SMILES -- no SA score to compare",
                    }
                )
                continue
            if min_sa is not None and scored.score < min_sa:
                rejected_rows.append(
                    {
                        "rank": i + 1,
                        "smiles": scored.smiles,
                        "score": self.scores[i],
                        "sa_score": scored.score,
                        "reason": f"sa_score {scored.score} below min_sa {min_sa}",
                    }
                )
                continue
            if max_sa is not None and scored.score > max_sa:
                rejected_rows.append(
                    {
                        "rank": i + 1,
                        "smiles": scored.smiles,
                        "score": self.scores[i],
                        "sa_score": scored.score,
                        "reason": f"sa_score {scored.score} above max_sa {max_sa}",
                    }
                )
                continue
            kept_rows.append(i)
        kept = GenerationResult(
            smiles=[self.smiles[i] for i in kept_rows],
            scores=[self.scores[i] for i in kept_rows],
            breakdown=[self.breakdown[i] for i in kept_rows],
            history=list(self.history),
            metadata={
                **self.metadata,
                "sa_filter": {
                    "min_sa": min_sa,
                    "max_sa": max_sa,
                    "kept": len(kept_rows),
                    "rejected": len(rejected_rows),
                },
            },
        )
        rejected = pd.DataFrame(
            rejected_rows,
            columns=["rank", "smiles", "score", "sa_score", "reason"],
        )
        return SAFilteredCandidates(kept=kept, rejected=rejected)

    def sa_summary(self) -> dict[str, Any]:
        """Direction-aware SA distribution statistics (section 24.3).

        Statistics are over the valid subset only (the "actual computed
        values" rule); ``n_druglike_band`` counts scores inside the
        conventional 1-4 drug-like band.  The payload names the method,
        the direction and the limitations so no interface can present the
        score without its context -- a favourable SA score is *not*
        evidence a molecule can actually be synthesised.

        Example
        -------
        >>> from pharmaml.generative import GenerationResult
        >>> result = GenerationResult(smiles=["CCO", "CCN"], scores=[0.5, 0.4])
        >>> summary = result.sa_summary()
        >>> (summary["n_scored"], summary["n_invalid"], summary["n_druglike_band"])
        (2, 0, 2)
        >>> summary["direction"]
        'lower = estimated easier to synthesise'
        >>> "method" in summary and "limitations" in summary
        True
        """
        from pharmaml.chem.sa_scoring import (
            SA_SCORE_LIMITATIONS,
            SA_SCORE_METHOD,
            sa_scores,
        )

        batch = sa_scores(self.smiles)
        summary: dict[str, Any] = {
            "n_total": len(batch.results),
            "n_scored": batch.n_scored,
            "n_invalid": batch.n_invalid,
            "method": SA_SCORE_METHOD,
            "direction": "lower = estimated easier to synthesise",
            "limitations": SA_SCORE_LIMITATIONS,
            "elapsed_seconds": batch.elapsed_seconds,
        }
        scores = batch.valid_scores
        if scores:
            import numpy as np

            arr = np.asarray(scores, dtype=float)
            summary.update(
                {
                    "min": round(float(arr.min()), 3),
                    "max": round(float(arr.max()), 3),
                    "mean": round(float(arr.mean()), 3),
                    "median": round(float(np.median(arr)), 3),
                    "std": round(
                        float(arr.std(ddof=1)) if len(scores) > 1 else 0.0, 3
                    ),
                    "n_druglike_band": int(
                        ((arr >= 1.0) & (arr <= 4.0)).sum()
                    ),
                }
            )
        return summary


def _empty_rejected_frame() -> pd.DataFrame:
    """The zero-row ``rejected`` table (so no-op filters stay typed)."""
    return pd.DataFrame(
        [], columns=["rank", "smiles", "score", "sa_score", "reason"]
    )


@dataclass(frozen=True)
class SAFilteredCandidates:
    """Section 24.4 outcome: kept AND rejected are both preserved.

    ``kept`` is a fresh :class:`GenerationResult` (the run's own rank
    order among survivors, breakdown aligned, filter provenance in
    ``metadata["sa_filter"]``); ``rejected`` is the ``rank`` / ``smiles``
    / ``score`` / ``sa_score`` / ``reason`` table of everything the
    constraint turned away -- nothing is silently discarded, per the
    spec's "preserve rejected candidate information" rule.
    """

    kept: GenerationResult
    rejected: pd.DataFrame

    @property
    def n_kept(self) -> int:
        """Candidates inside the constraint (any the run generated)."""
        return len(self.kept)

    @property
    def n_rejected(self) -> int:
        """Candidates turned away (preserved with reasons, not dropped)."""
        return len(self.rejected)

    @property
    def n_total(self) -> int:
        """kept + rejected -- every generated candidate is accounted for."""
        return self.n_kept + self.n_rejected
