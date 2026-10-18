"""Multi-objective Pareto front extraction for generation runs.

A :class:`~pharmaml.generative.MultiObjective` run optimises the WEIGHTED
GEOMETRIC MEAN of its sub-objectives, so the leaderboard ranks molecules by
one blended number -- and hides the trade-offs.  This module re-reads the
run's per-objective ``breakdown`` (every
:class:`~pharmaml.generative.GenerationResult` already carries it) and
extracts the **Pareto front**: the molecules that are *not dominated* by any
other molecule, where ``a`` dominates ``b`` iff ``a`` scores at least ``b``
on every selected objective and strictly more on at least one.  A molecule
sitting deep in the leaderboard (rank 150 of 159) can be Pareto-optimal --
unbeatable on one objective without sacrificing another -- and those are
exactly the candidates a medchem decision wants to see next to the
weighted-sum winner.

The extraction is a pure function of the result (deterministic, no RNG, no
state); the front preserves the leaderboard's order (best total score
first).  :func:`pharmaml.viz.generative_plots.plot_pareto_front` renders one
run as the classic scatter with the front highlighted.

Example
-------
>>> from pharmaml.generative import GenerationResult, pareto_front
>>> result = GenerationResult(
...     smiles=["A", "B", "C", "D"], scores=[0.9, 0.8, 0.7, 0.1],
...     breakdown=[{"qed": 0.9, "novelty": 0.2},
...                {"qed": 0.4, "novelty": 0.9},
...                {"qed": 0.3, "novelty": 0.3},
...                {"qed": 0.85, "novelty": 0.25}])
>>> front = pareto_front(result)
>>> front.smiles                      # C is dominated (by B alone); A/B/D trade off
('A', 'B', 'D')
>>> front.dominates                   # how many molecules each member dominates
(0, 1, 0)
>>> text = str(front)                 # one-line audit
>>> text.startswith("Pareto front over 2 objectives [qed, novelty]: 3 of 4")
True
>>> "strongest member 'B' dominates 1 others" in text
True
>>> front.to_dataframe()
   rank smiles   qed  novelty  dominates
0     1      A  0.90     0.20          0
1     2      B  0.40     0.90          1
2     3      D  0.85     0.25          0
>>> pareto_front(result, objectives=["qed"])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: a Pareto front needs at least two objectives, got 1: ['qed']
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.generative.result import GenerationResult

__all__ = ["ParetoFront", "pareto_front"]


def _finite(value: object) -> bool:
    """True when ``value`` is a real number usable in dominance tests."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


@dataclass(frozen=True, slots=True)
class ParetoFront:
    """The non-dominated set of one generation run (audit-ready).

    Attributes
    ----------
    objectives:
        The objective names the front was computed over, in column order.
    smiles:
        Front members, leaderboard order preserved (best total score
        first) -- the front itself is unordered, this ordering is a
        convenience for display.
    values:
        Per-objective scores aligned with ``smiles`` (one dict per front
        member, exactly the selected ``objectives`` as keys).
    dominates:
        For each front member: how many of the OTHER included molecules it
        dominates (0 = a pure trade-off corner).
    n_total:
        Molecules examined (after the non-finite exclusion below).
    n_excluded:
        Molecules dropped because at least one selected objective value
        was NaN / +-inf -- dominance is undefined for them.

    Notes
    -----
    Rank 1 of the leaderboard is always on the front (anything dominating
    it would have a strictly larger weighted-geometric-mean score and
    would have outranked it), so ``dominates[0]`` is the front's largest
    domination count by construction only in the two-objective case --
    deeper members can still dominate many molecules on partial scores.
    """

    objectives: tuple[str, ...]
    smiles: tuple[str, ...]
    values: tuple[dict[str, float], ...]
    dominates: tuple[int, ...]
    n_total: int
    n_excluded: int

    def __len__(self) -> int:
        """Number of front members."""
        return len(self.smiles)

    def is_empty(self) -> bool:
        """True when no molecule survived the dominance filter."""
        return not self.smiles

    def to_dataframe(self) -> pd.DataFrame:
        """Front table: rank, SMILES, one column per objective, dominates."""
        frame = pd.DataFrame(
            {"rank": range(1, len(self.smiles) + 1), "smiles": list(self.smiles)}
        )
        for objective in self.objectives:
            frame[objective] = [entry[objective] for entry in self.values]
        frame["dominates"] = list(self.dominates)
        return frame

    def __str__(self) -> str:
        head = ", ".join(self.objectives)
        if self.is_empty():
            return (
                f"Pareto front over {len(self.objectives)} objectives "
                f"[{head}]: 0 of {self.n_total} molecules non-dominated "
                f"({self.n_excluded} excluded for non-finite values)"
            )
        strongest = max(range(len(self.smiles)), key=lambda i: self.dominates[i])
        return (
            f"Pareto front over {len(self.objectives)} objectives [{head}]: "
            f"{len(self.smiles)} of {self.n_total} molecules non-dominated "
            f"({self.n_excluded} excluded for non-finite values); "
            f"strongest member {self.smiles[strongest]!r} dominates "
            f"{self.dominates[strongest]} others"
        )


def pareto_front(
    result: GenerationResult,
    *,
    objectives: Sequence[str] | None = None,
) -> ParetoFront:
    """Extract the non-dominated molecules (Pareto front) of a run.

    Parameters
    ----------
    result:
        A :class:`~pharmaml.generative.GenerationResult` whose ``breakdown``
        carries per-objective scores (any
        :class:`~pharmaml.generative.MultiObjective` run fills it; a
        single-objective run does not).
    objectives:
        The sub-objectives to compute dominance over (default: every
        breakdown key, in first-appearance order).  Pass a subset to
        project -- e.g. ``["qed", "novelty"]`` of a three-objective run.

    Returns
    -------
    ParetoFront
        The non-dominated set, leaderboard order preserved.

    Raises
    ------
    DataValidationError
        If the breakdown carries no objective keys at all (a
        single-objective run), fewer than two objectives are selected,
        an unknown objective name is requested, or ``objectives``
        contains duplicates.

    Notes
    -----
    Dominance: ``a`` dominates ``b`` iff ``a >= b`` on every selected
    objective and ``a > b`` on at least one.  Molecules with a NaN / +-inf
    value in any selected objective are excluded from candidacy (counted
    in ``ParetoFront.n_excluded``) -- their dominance is undefined, and a
    silent NaN comparison would corrupt the front.  The pairwise scan is
    O(n^2 * k) over n molecules and k objectives: a 1,000-molecule
    leaderboard over three objectives is ~3M comparisons (about a
    second) -- leaderboard-scale inputs are the intended size.
    """
    smiles = list(getattr(result, "smiles", []) or [])
    breakdown = list(getattr(result, "breakdown", []) or [])
    if len(smiles) != len(breakdown):  # pragma: no cover - result invariant
        raise DataValidationError(
            f"result is misaligned: {len(smiles)} smiles but "
            f"{len(breakdown)} breakdown rows"
        )
    keys: list[str] = []
    for entry in breakdown:
        for key in entry:
            if key not in keys:
                keys.append(key)
    if not objectives:
        if not keys:
            raise DataValidationError(
                "result carries no objective breakdown -- a Pareto front "
                "needs a MultiObjective run (its per-objective breakdown), "
                "not a single ScoringObjective run"
            )
        selected = keys
    else:
        selected = list(objectives)
        unknown = [name for name in selected if name not in keys]
        if unknown:
            raise DataValidationError(
                f"objective(s) {unknown!r} not in this run's breakdown; "
                f"available: {keys!r}"
            )
        duplicates = sorted(
            {name for name in selected if selected.count(name) > 1}
        )
        if duplicates:
            raise DataValidationError(
                f"objectives contains duplicate name(s) {duplicates!r} -- "
                "each objective may be selected once"
            )
    if len(selected) < 2:
        raise DataValidationError(
            f"a Pareto front needs at least two objectives, got "
            f"{len(selected)}: {selected!r}"
        )

    # -- candidacy: rows with finite values on every selected objective ----
    rows: list[tuple[str, list[float]]] = []
    excluded = 0
    for name, entry in zip(smiles, breakdown):
        values = [entry.get(name) for name in selected]
        if all(_finite(value) for value in values):
            rows.append((name, [float(value) for value in values]))
        else:
            excluded += 1

    # -- pairwise dominance: a >= b everywhere and a > b somewhere -------- #
    width = len(selected)
    counts = [0] * len(rows)
    on_front = [True] * len(rows)
    for i, (_smiles_i, values_i) in enumerate(rows):
        for j in range(len(rows)):
            if i == j:
                continue
            values_j = rows[j][1]
            ge = True
            gt = False
            for k in range(width):
                if values_i[k] < values_j[k]:
                    ge = False
                    break
                if values_i[k] > values_j[k]:
                    gt = True
            if ge and gt:  # i dominates j
                counts[i] += 1
                on_front[j] = False

    front_indices = [i for i, keep in enumerate(on_front) if keep]
    return ParetoFront(
        objectives=tuple(selected),
        smiles=tuple(rows[i][0] for i in front_indices),
        values=tuple(
            {name: rows[i][1][k] for k, name in enumerate(selected)}
            for i in front_indices
        ),
        dominates=tuple(counts[i] for i in front_indices),
        n_total=len(rows),
        n_excluded=excluded,
    )
