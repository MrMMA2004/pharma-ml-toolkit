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

Real runs can leave the front itself unwieldy -- a three-objective GA over
a 300-molecule population for 30 generations routinely leaves 40+ members
(a measured 46 of 8,108 in the round that added pruning).  The NSGA-II
**crowding distance** (:meth:`ParetoFront.crowding_distance`) ranks the
members by how isolated each sits in objective space -- the trade-off
corners score +inf, crowded interiors score low -- and
:meth:`ParetoFront.prune` keeps the most isolated ``k`` for a small,
well-spread medchem shortlist.

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
>>> front.to_dataframe()           # crowding: inf = trade-off corner
   rank smiles   qed  novelty  dominates  crowding
0     1      A  0.90     0.20          0       inf
1     2      B  0.40     0.90          1       inf
2     3      D  0.85     0.25          0       2.0
>>> front.crowding_distance()       # NSGA-II: A/B corner the trade-off
(inf, inf, 2.0)
>>> pruned = front.prune(2)         # the 2 most isolated of 3
>>> pruned.smiles
('A', 'B')
>>> pruned.n_front                  # size of the un-pruned front
3
>>> front.prune(5) is front         # keep >= size: nothing to drop
True
>>> pareto_front(result, objectives=["qed"])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: a Pareto front needs at least two objectives, got 1: ['qed']
>>> front.prune(0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: keep must be at least 1, got 0 -- a pruned front with no member is the empty front, not a selection
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
    n_front:
        Size of the front this one was pruned from (``None`` for an
        un-pruned front).  Pairwise dominance does not depend on which
        molecules are present, so a pruned front keeps every member's
        original ``dominates`` count -- the audit survives the cut.

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
    n_front: int | None = None

    def __len__(self) -> int:
        """Number of front members."""
        return len(self.smiles)

    def is_empty(self) -> bool:
        """True when no molecule survived the dominance filter."""
        return not self.smiles

    def crowding_distance(self) -> tuple[float, ...]:
        """NSGA-II crowding distance of every front member.

        For each objective the members are sorted by value; the two
        boundary positions (best and worst) mark their members' distance
        as +inf -- the trade-off corners NSGA-II always preserves -- and
        every interior member accumulates the normalised gap between its
        neighbours, ``(next - prev) / (max - min)``.  An objective with no
        spread (``max == min``) contributes 0 to everyone: it carries no
        isolation information.  Sorts tie-break by leaderboard position,
        so the result is deterministic (no RNG, a pure function of the
        front).

        Returns
        -------
        tuple[float, ...]
            One value per member, aligned with ``smiles``; ``+inf`` marks
            a boundary member (best or worst on at least one objective),
            larger finite values mean more isolated (a sparser region of
            objective space), 0 means squeezed between immediate
            neighbours on every objective.  An empty front returns
            ``()``; a lone member or a fully degenerate front (no spread
            on any objective) returns 0.0s -- crowding measures *spread*,
            and there is none to measure.

        Example
        -------
        >>> from pharmaml.generative import GenerationResult, pareto_front
        >>> front = pareto_front(GenerationResult(
        ...     smiles=["A", "B", "D"], scores=[0.9, 0.8, 0.7],
        ...     breakdown=[{"qed": 0.9, "novelty": 0.2},
        ...                {"qed": 0.4, "novelty": 0.9},
        ...                {"qed": 0.85, "novelty": 0.25}]))
        >>> front.crowding_distance()          # A/B corner, D interior
        (inf, inf, 2.0)
        """
        count = len(self.smiles)
        if count == 0:
            return ()
        distances = [0.0] * count
        for name in self.objectives:
            values = [entry[name] for entry in self.values]
            low, high = min(values), max(values)
            if high <= low:  # no spread -> no isolation information
                continue
            order = sorted(range(count), key=lambda i: (values[i], i))
            distances[order[0]] = math.inf
            distances[order[-1]] = math.inf
            for position in range(1, count - 1):
                index = order[position]
                gap = values[order[position + 1]] - values[order[position - 1]]
                distances[index] += gap / (high - low)
        return tuple(distances)

    def prune(self, keep: int) -> ParetoFront:
        """The ``keep`` most isolated members (crowding-distance selection).

        Ranks the members by :meth:`crowding_distance` (largest first;
        +inf boundary corners first among them; ties broken by
        leaderboard rank), keeps the top ``keep``, and re-emits them in
        LEADERBOARD order -- the front's standing convention.  A real
        three-objective run can leave 40+ front members; this is the
        principled way to cut them to a shortlist that still spans the
        trade-off surface (rank-ordered truncation would drop the corners
        the front exists to show).

        The returned front carries ``n_front`` = the un-pruned size, keeps
        every member's original ``dominates`` count (pairwise dominance
        cannot change with membership), and preserves ``n_total`` /
        ``n_excluded`` from the original scan.

        Parameters
        ----------
        keep:
            How many members to retain.  ``keep >= len(front)`` returns
            the front itself (it is frozen and immutable, safe to share);
            ``keep < 1`` raises.

        Returns
        -------
        ParetoFront
            The pruned front (or this one, when nothing would be dropped).

        Raises
        ------
        DataValidationError
            If ``keep`` is less than 1.

        Example
        -------
        >>> from pharmaml.generative import GenerationResult, pareto_front
        >>> front = pareto_front(GenerationResult(
        ...     smiles=["A", "B", "C", "D"], scores=[0.9, 0.8, 0.7, 0.1],
        ...     breakdown=[{"qed": 0.9, "novelty": 0.2},
        ...                {"qed": 0.4, "novelty": 0.9},
        ...                {"qed": 0.3, "novelty": 0.3},
        ...                {"qed": 0.85, "novelty": 0.25}]))
        >>> pruned = front.prune(2)
        >>> pruned.smiles              # the two corners, leaderboard order
        ('A', 'B')
        >>> pruned.n_front, len(pruned)
        (3, 2)
        >>> "pruned to the 2 most isolated of 3" in str(pruned)
        True
        """
        if keep < 1:
            raise DataValidationError(
                f"keep must be at least 1, got {keep} -- a pruned front "
                "with no member is the empty front, not a selection"
            )
        if keep >= len(self.smiles):
            return self
        distances = self.crowding_distance()
        # largest distance first (+inf corners lead); leaderboard rank as
        # the deterministic tie-break -- the textbook NSGA-II sort
        ranked = sorted(range(len(self.smiles)), key=lambda i: (-distances[i], i))
        selected = sorted(ranked[:keep])  # re-emit in leaderboard order
        return ParetoFront(
            objectives=self.objectives,
            smiles=tuple(self.smiles[i] for i in selected),
            values=tuple(self.values[i] for i in selected),
            dominates=tuple(self.dominates[i] for i in selected),
            n_total=self.n_total,
            n_excluded=self.n_excluded,
            n_front=len(self.smiles),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Front table: rank, SMILES, objectives, dominates, crowding."""
        frame = pd.DataFrame(
            {"rank": range(1, len(self.smiles) + 1), "smiles": list(self.smiles)}
        )
        for objective in self.objectives:
            frame[objective] = [entry[objective] for entry in self.values]
        frame["dominates"] = list(self.dominates)
        frame["crowding"] = list(self.crowding_distance())
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
        # a pruned front reports the ORIGINAL non-dominated count (the
        # prune suffix carries the cut) -- claiming "3 of 36 non-dominated"
        # for a 7-member front cut to 3 would be factually wrong
        non_dominated = (
            self.n_front if self.n_front is not None else len(self.smiles)
        )
        base = (
            f"Pareto front over {len(self.objectives)} objectives [{head}]: "
            f"{non_dominated} of {self.n_total} molecules non-dominated "
            f"({self.n_excluded} excluded for non-finite values)"
        )
        if self.n_front is not None:
            base += (
                f"; pruned to the {len(self.smiles)} most isolated of "
                f"{self.n_front} by crowding distance"
            )
        return (
            base
            + f"; strongest member {self.smiles[strongest]!r} dominates "
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
