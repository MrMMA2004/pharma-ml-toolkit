"""SMILES genetic algorithm: de novo and seeded molecule generation.

A pragmatic population-based optimiser in the GuacaMol / LEADD tradition,
CPU-friendly and deterministic given ``seed``:

* **genome** = canonical SMILES string (character level);
* **mutation** = random character substitution / insertion / deletion,
  followed by RDKit sanitisation and re-canonicalisation (invalid mutants
  are discarded) -- OR, in ``mutation="scaffold"`` mode, a
  scaffold-preserving decoration resampling (see :func:`_scaffold_mutate`)
  -- OR, in ``mutation="topology"`` mode, a ring-size move drawn from
  :func:`~pharmaml.generative.ring_topology_mutate` (ring contraction /
  expansion, the single-parent medchem counterpart of the scaffold-hop
  crossover's ring-topology geometry);
* **crossover** = single-point splice of two parents, sanitised the same
  way -- OR, with ``crossover="scaffold_hop"``, a BREED-style
  homologous-cut recombination (see
  :func:`~pharmaml.generative.scaffold_hop_crossover`): both parents are
  cut at a pair of matching single acyclic bonds and the fragments
  recombined cross-wise, or at matching single ring bonds (rings merged,
  or a two-ended linker -- a single atom or a short acyclic path of up
  to four atoms -- transplanted into the cut: ring homologation by a
  whole linker alphabet -- ring topology recombines too), or a whole
  ring excised from one parent between its two single anchor bonds and
  spliced across the other's opened ring bond (fused / bridged
  bicyclics -- ring-*system* topology recombines too), or a whole ring
  spliced INTO the other's cut acyclic bond (the ring becomes the
  spacer -- ring-as-linker), or each parent's bounded acyclic middle
  exchanged for the other's homologous middle (two-point segment
  swap -- the linker / bioisostere exchange: an ester O for an amide
  N), so
  ring systems survive by construction and
  the offspring can hop scaffolds.  The character splice is not used in
  ``mutation="scaffold"`` mode (it routinely destroys ring systems,
  defeating scaffold preservation); the structural scaffold-hop
  crossover IS available there;
* **selection** = tournament of ``tournament_k`` molecules;
* **elitism** = the top ``elitism`` molecules survive a generation
  verbatim, so the best score is monotone across generations;
* **operator selection** = fixed-rate by default
  (``operator_selection="uniform"``: the ``crossover_rate`` Bernoulli,
  the flat candidate-pool draw inside the scaffold-hop crossover, the
  mutation mode's structural if-chain -- bit-identical to the
  historical optimizer) **or adaptive** (``operator_selection=
  "adaptive"``): three levels of operator choice bias themselves by
  per-operator credit instead of staying fixed -- level 1 (crossover vs
  clone), the scaffold-hop geometry (the seven buckets, renormalised
  over the geometries present in each pair's candidate pool; the splice
  crossover degenerates to the single "splice" operator) and the
  mutation family (the mode's primary mutation vs the character
  mutation).  A drawn operator earns ``(valid offspring ? score : 0.0)
  - mid-parent`` on a sliding ``aos_window``-attempt window (a drawn
  operator whose own product was discarded earns the null reward
  ``0.0 - mid-parent``); selection is a softmax over the windowed
  credit at ``aos_temperature`` with an exploration floor of
  ``1/(k*5)`` -- exactly one ``rng.random()`` per selection, so an
  adaptive run is just as deterministic as a uniform one.  Adaptive
  runs gain ``"operator_selection"``, the per-level ``"operator_stats"``
  tables and ``"operator_trajectory"`` -- one record per generation
  (index-aligned with ``history``) holding the full-precision operator
  mix in force at that generation's start, so record 0 is the fair prior
  and the final record is the closing mix (``trajectory[-1][level][op]``
  equals ``operator_stats[level][op]["final_probability"]`` up to that
  stat's 4-decimal rounding, i.e. within 5e-5); uniform runs' metadata
  is unchanged.

Two initialisation modes:

* **de novo** (``seed_smiles=None``): the population is sampled from
  ``corpus`` (default: :data:`~pharmaml.generative.DEFAULT_CORPUS`);
* **seeded** (``seed_smiles=[...]``): the user's molecules anchor
  generation 0 and corpus molecules pad the population up to
  ``population_size`` (pass ``corpus=seed_smiles`` too for a strictly
  seed-isolated run), so the search explores the seeds' neighbourhood
  alongside the corpus's diversity -- the "with some sample molecules"
  mode.

Example
-------
>>> from pharmaml.generative import GeneticOptimizer, QedObjective
>>> result = GeneticOptimizer(QedObjective(), population_size=32,
...                           generations=5, seed=42).optimize()
>>> result.best_score >= 0.5
True
>>> # adaptive operator selection: the GA learns which operators pay off
>>> adaptive = GeneticOptimizer(QedObjective(), population_size=16,
...                             generations=4, seed=42,
...                             operator_selection="adaptive").optimize()
>>> adaptive.metadata["operator_selection"]
'adaptive'
>>> stats = adaptive.metadata["operator_stats"]
>>> sorted(stats)
['geometry', 'level1', 'mutation']
>>> sorted(stats["level1"])
['clone', 'crossover']
>>> stats["level1"]["clone"]["draws"] > 0 and stats["mutation"]["character"]["draws"] > 0
True
>>> traj = adaptive.metadata["operator_trajectory"]
>>> len(traj) == adaptive.metadata["iterations"] == len(adaptive.history)
True
>>> round(traj[0]["level1"]["crossover"], 1)   # record 0: the fair prior
0.5
>>> traj[0]["mutation"] == {"character": 1.0}   # default mutation: 1-operator family
True
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

from ._utils import canonical_smiles, require_rdkit, smiles_alphabet
from .corpus import DEFAULT_CORPUS
from .result import GenerationResult
from .scaffold_hop import (
    GEOMETRY_ORDER,
    _checked_min_heavy_atoms,
    _scaffold_hop_core,
    scaffold_hop_crossover,
)
from .topology import ring_topology_mutate

__all__ = ["GeneticOptimizer", "run_genetic_optimization"]

_LOG = get_logger(__name__)

#: Probability that a resampled decoration slot is dropped entirely (rather
#: than replaced with a corpus decoration) by the scaffold mutation.
_DROP_PROBABILITY = 0.25

#: Exploration-floor margin for the adaptive operator selection: every
#: operator keeps at least ``1 / (k * margin)`` of its level's draw
#: probability (``k`` = operators at the level), so a consistently failing
#: operator starves towards the floor but can never be fully abandoned --
#: the starvation pathology guard (credits stay finite, selection never
#: degenerates into a NaN or a division by zero).
_AOS_FLOOR_MARGIN = 5.0


def _validated_score(objective: Any, smiles: str) -> float:
    """Score ``smiles``, coercing stray values into ``[0, 1]``.

    A non-numeric return value raises the same typed error as
    :class:`~pharmaml.generative.MultiObjective` -- the objective contract
    (``score(smiles) -> float``) is enforced consistently everywhere.
    """
    try:
        value = float(objective.score(smiles))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"objective {getattr(objective, 'name', objective)!r} returned a "
            f"non-numeric score for {smiles!r} -- objectives must map SMILES "
            "to floats"
        ) from exc
    if not math.isfinite(value):
        return 0.0
    return min(max(value, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Scaffold-preserving mutation machinery
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def _scaffold_split(smiles: str) -> tuple[str, tuple[str, ...]] | None:
    """Split ``smiles`` into its (labelled) Murcko core + decorations.

    Returns ``(core_smiles, decoration_smiles)`` where the core carries
    ``[*:k]`` dummy atoms at every attachment point (labelled 1..k in a
    deterministic order) and each decoration fragment carries the matching
    label -- so ``Chem.molzip(core, decorations)`` rebuilds exactly the
    parent molecule.  ``None`` when there is nothing to split (acyclic
    molecule, bare ring system without decorations, unparseable input, or
    any RDKit failure -- robustness first).
    """
    Chem = require_rdkit()
# rdkit-stubs does not re-export DisableLog/EnableLog on rdkit.RDLogger,
    # but rdBase (their defining module) is fully typed.
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.rdBase import DisableLog, EnableLog

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    DisableLog("rdApp.error")
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:  # noqa: BLE001 - RDKit can raise on hostile input
        return None
    finally:
        EnableLog("rdApp.error")
    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return None  # acyclic parent: no framework to preserve
    match = mol.GetSubstructMatch(scaffold)
    if not match:
        return None
    in_scaffold = set(match)
    # attachment pairs (scaffold atom position, outside neighbour), ranked
    # deterministically: label k = k-th attachment in sorted pair order.
    pairs = sorted(
        (j, neighbour.GetIdx())
        for j, atom_idx in enumerate(match)
        for neighbour in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbour.GetIdx() not in in_scaffold
    )
    if not pairs:
        return None  # bare scaffold: no decorations to resample
    ranks_by_j: dict[int, list[int]] = {}
    for rank, (j, _neighbour) in enumerate(pairs, start=1):
        ranks_by_j.setdefault(j, []).append(rank)

    core = Chem.ReplaceSidechains(mol, scaffold)
    if core is None:
        return None
    core_match = core.GetSubstructMatch(scaffold)
    if not core_match:
        return None
    dummies_by_j: dict[int, list[int]] = {}
    for atom in core.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbours = atom.GetNeighbors()
        if len(neighbours) != 1:
            return None
        j = core_match.index(neighbours[0].GetIdx())
        dummies_by_j.setdefault(j, []).append(atom.GetIdx())
    rw_core = Chem.RWMol(core)
    for j, dummy_ids in dummies_by_j.items():
        ranks = ranks_by_j.get(j, [])
        if len(dummy_ids) != len(ranks):
            return None
        for dummy_id, rank in zip(sorted(dummy_ids), ranks):
            atom = rw_core.GetAtomWithIdx(dummy_id)
            atom.SetAtomMapNum(rank)
            atom.SetIsotope(0)
    core_smiles = Chem.MolToSmiles(rw_core.GetMol())

    # ReplaceCore(labelByIndex=True) labels each decoration dummy with the
    # scaffold-atom position it was attached to -- relabel to the ranks.
    side = Chem.ReplaceCore(mol, scaffold, labelByIndex=True)
    if side is None:
        return None
    rank_queues = {j: list(ranks) for j, ranks in ranks_by_j.items()}
    decoration_smiles: list[str] = []
    for fragment in Chem.GetMolFrags(side, asMols=True):
        rw_fragment = Chem.RWMol(fragment)
        saw_dummy = False
        for atom in rw_fragment.GetAtoms():
            if atom.GetAtomicNum() != 0:
                continue
            queue = rank_queues.get(atom.GetIsotope())
            if not queue:
                return None
            atom.SetAtomMapNum(queue.pop(0))
            atom.SetIsotope(0)
            saw_dummy = True
        if not saw_dummy:
            return None  # untracked fragment: refuse rather than lose atoms
        decoration_smiles.append(Chem.MolToSmiles(rw_fragment.GetMol()))
    return core_smiles, tuple(decoration_smiles)


def _strip_dummy_label(fragment_smiles: str) -> str:
    """Remove the attachment label from a single-dummy decoration SMILES."""
    Chem = require_rdkit()
    fragment = Chem.MolFromSmiles(fragment_smiles)
    if fragment is None:  # pragma: no cover - split output always parses
        return fragment_smiles
    rw = Chem.RWMol(fragment)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
    return Chem.MolToSmiles(rw.GetMol())


@lru_cache(maxsize=8)
def _decoration_pool(corpus_key: tuple[str, ...]) -> tuple[str, ...]:
    """Corpus-derived decoration vocabulary (memoised on the corpus tuple)."""
    pool: list[str] = []
    for smiles in corpus_key:
        split = _scaffold_split(smiles)
        if split is None:
            continue
        for fragment_smiles in split[1]:
            if fragment_smiles.count("[*") != 1:
                continue  # only single-attachment decorations are reusable
            pool.append(fragment_smiles)
    return tuple(dict.fromkeys(_strip_dummy_label(s) for s in pool))


def _build_decoration_pool(corpus: Sequence[str]) -> list[str]:
    """Decoration vocabulary derived from ``corpus`` (deterministic order).

    Every corpus molecule is decomposed once into Murcko scaffold +
    decorations (see :func:`_scaffold_split`); the single-attachment
    decorations, label-stripped and de-duplicated, form the vocabulary the
    scaffold mutation samples from.  Murcko decorations are always acyclic
    (any ring would be part of the framework), which is what makes scaffold
    preservation structural rather than lucky.
    """
    return list(_decoration_pool(tuple(corpus)))


def _scaffold_mutate(
    smiles: str,
    pool: Sequence[str],
    rng: random.Random,
    max_smiles_length: int,
) -> str | None:
    """Decoration-resampling mutation that preserves the Murcko scaffold.

    Keeps the parent's Murcko framework verbatim and replaces a random
    non-empty subset of its decorations with entries sampled from the
    corpus-derived ``pool`` (or drops them).  Returns the canonical mutant
    SMILES, or ``None`` when the parent has no scaffold / no decorations,
    the pool is unusable for it, the mutant would exceed
    ``max_smiles_length``, or RDKit raises anywhere in the pipeline -- the
    caller then falls back to the standard character mutation
    (robustness first: never crash, never emit a dummy-bearing or empty
    molecule).
    """
    Chem = require_rdkit()
    from rdkit.rdBase import DisableLog, EnableLog

    # Occasional chemically invalid re-attachments are EXPECTED (they fall
    # back to the character mutation); RDKit logs the sanitisation failures
    # to stderr before the exception surfaces -- keep the loop quiet (the
    # canonical_smiles guard-family precedent).
    DisableLog("rdApp.error")
    try:
        split = _scaffold_split(smiles)
        if split is None:
            return None
        core_smiles, fragment_smiles = split
        # only single-attachment decorations can be replaced/dropped;
        # bridging multi-dummy fragments are always kept as-is
        replaceable = [
            index for index, fragment in enumerate(fragment_smiles)
            if fragment.count("[*") == 1
        ]
        if not replaceable:
            return None
        n_change = rng.randint(1, len(replaceable))
        chosen = set(rng.sample(replaceable, n_change))

        rw_core = Chem.RWMol(Chem.MolFromSmiles(core_smiles))
        kept: list[str] = []
        for index, fragment in enumerate(fragment_smiles):
            if index not in chosen:
                kept.append(fragment)
                continue
            parsed = Chem.MolFromSmiles(fragment)
            if parsed is None:  # pragma: no cover - split output always parses
                return None
            label = next(
                atom.GetAtomMapNum()
                for atom in parsed.GetAtoms()
                if atom.GetAtomicNum() == 0
            )
            if rng.random() < _DROP_PROBABILITY:
                # Drop the decoration by turning its core dummy into an
                # explicit hydrogen (plain RemoveAtom would leave e.g. an
                # aromatic N without its H -> "Can't kekulize" on sanitise;
                # an unmatched dummy would leak [*:k] into the offspring).
                # RemoveHs below collapses the graph-H back to implicit.
                for dummy_id in (
                    atom.GetIdx()
                    for atom in rw_core.GetAtoms()
                    if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == label
                ):
                    hydrogen = rw_core.GetAtomWithIdx(dummy_id)
                    hydrogen.SetAtomicNum(1)
                    hydrogen.SetAtomMapNum(0)
                    hydrogen.SetIsotope(0)
                    hydrogen.SetFormalCharge(0)
                    hydrogen.SetNumExplicitHs(0)
                    hydrogen.SetNoImplicit(True)
            else:
                replacement = Chem.MolFromSmiles(pool[rng.randrange(len(pool))])
                if replacement is None:  # pragma: no cover - pool always parses
                    return None
                rw_replacement = Chem.RWMol(replacement)
                for atom in rw_replacement.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        atom.SetAtomMapNum(label)
                kept.append(Chem.MolToSmiles(rw_replacement.GetMol()))

        core = rw_core.GetMol()
        Chem.SanitizeMol(core)
        if kept:
            side = Chem.MolFromSmiles(".".join(kept))
            if side is None:  # pragma: no cover - kept fragments always parse
                return None
            product = Chem.molzip(core, side)
        else:
            product = core  # every decoration dropped: the bare framework
        Chem.SanitizeMol(product)
        product = Chem.RemoveHs(product)  # graph-H drops -> implicit / [nH]
        mutant = Chem.MolToSmiles(product)
        if not mutant or len(mutant) > max_smiles_length:
            return None
        return mutant
    except Exception:  # noqa: BLE001 - robustness-first fallback contract
        return None
    finally:
        EnableLog("rdApp.error")


class _OperatorBook:
    """Sliding-window credit book + softmax-with-floor selection (one level).

    The adaptive operator selection's bookkeeping for ONE biasable level
    of the GA loop: level 1 (crossover vs clone), the recombination
    geometry (the seven scaffold-hop buckets, or the single ``"splice"``
    bucket in splice mode) or the mutation family (the mode's primary
    mutation vs the character mutation).  Each drawn operator earns one
    reward per attempt,

    ``reward = (valid offspring ? score(offspring) : 0.0) - mid_parent``,

    pushed into a sliding window of the last ``window`` attempts.  The
    null-reward convention: a drawn operator whose own product was
    discarded (its call returned ``None`` / fell back) takes ``0.0 -
    mid_parent`` -- failed operators are charged, not forgiven.  Credit
    is the window mean (an empty window reads as the neutral ``0.0``,
    never NaN); selection is

    ``p_i = floor + (1 - k * floor) * softmax(credit_i / temperature)``

    with ``floor = 1 / (k * _AOS_FLOOR_MARGIN)`` -- exactly one
    ``rng.random()`` per selection, resolved by cumulative-sum inversion
    over the book's fixed operator order.
    """

    def __init__(
        self, operators: Sequence[str], window: int, temperature: float
    ) -> None:
        self.operators: list[str] = list(operators)
        self.window = window
        self.temperature = temperature
        self.rewards: dict[str, deque[float]] = {
            op: deque(maxlen=window) for op in self.operators
        }
        self.draws: dict[str, int] = {op: 0 for op in self.operators}
        self.successes: dict[str, int] = {op: 0 for op in self.operators}
        self.best: dict[str, tuple[float, str]] = {}

    def credit(self, op: str) -> float:
        """Mean of ``op``'s reward window (0.0 when the window is empty)."""
        window = self.rewards[op]
        if not window:
            return 0.0  # empty window -> neutral prior, never NaN
        return sum(window) / len(window)

    def probabilities(
        self, present: Sequence[str] | None = None
    ) -> dict[str, float]:
        """Selection probabilities over ``present`` (default: all operators).

        The floor keeps every operator a minimum ``1 / (k * margin)``
        share of the draws, so a starving operator is demoted, never
        abandoned; the softmax rest carries the credit signal.
        """
        ops = [op for op in self.operators if present is None or op in present]
        if not ops:
            return {}
        if len(ops) == 1:
            return {ops[0]: 1.0}
        k = len(ops)
        floor = 1.0 / (k * _AOS_FLOOR_MARGIN)
        credits = [self.credit(op) for op in ops]
        top = max(credits)
        exps = [math.exp((value - top) / self.temperature) for value in credits]
        total = sum(exps)
        return {
            op: floor + (1.0 - k * floor) * (weight / total)
            for op, weight in zip(ops, exps)
        }

    def draw(self, rng: random.Random, present: Sequence[str] | None = None) -> str:
        """Draw one operator: a single ``rng.random()`` via cumsum inversion.

        Raises ``ValueError`` when ``present`` names no operator of this
        book (an empty draw is a caller bug, not a float-tail case).
        """
        probabilities = self.probabilities(present)
        operators = list(probabilities)
        if not operators:
            raise ValueError(
                f"no operator of this book is present in {present!r}"
            )
        pick = rng.random()
        cumulative = 0.0
        for op in operators:
            cumulative += probabilities[op]
            if pick < cumulative:
                return op
        return operators[-1]  # float-tail guard (deterministic)

    def push(
        self,
        op: str,
        reward: float,
        success: bool,
        child: str | None,
        child_score: float | None,
    ) -> None:
        """Record one attempt: reward into the window, tallies, best offspring."""
        self.rewards[op].append(reward)
        self.draws[op] += 1
        if success:
            self.successes[op] += 1
            if (
                child is not None
                and child_score is not None
                and (op not in self.best or child_score > self.best[op][0])
            ):
                self.best[op] = (child_score, child)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """The metadata artifact: operator -> tallies / credit / probability.

        ``credit``, ``final_probability`` and ``best_offspring_score``
        are rounded to 4 decimal places (a display artifact -- a
        level's rounded probabilities can sum to 1 with an error of
        up to ~5e-4; see :meth:`GeneticOptimizer.optimize`'s Returns
        notes).  The live, unrounded probabilities are
        :meth:`probabilities` -- the GA itself draws from those.
        """
        probabilities = self.probabilities()
        table: dict[str, dict[str, Any]] = {}
        for op in self.operators:
            best_score, best_child = self.best.get(op, (0.0, None))
            table[op] = {
                "draws": self.draws[op],
                "successes": self.successes[op],
                "credit": round(self.credit(op), 4),
                "final_probability": round(probabilities.get(op, 0.0), 4),
                "best_offspring": best_child,
                "best_offspring_score": round(best_score, 4),
            }
        return table


class GeneticOptimizer:
    """Tournament-selection genetic algorithm over canonical SMILES.

    Parameters
    ----------
    objective:
        Anything with ``score(smiles) -> float`` (any
        :class:`~pharmaml.generative.ScoringObjective`).
    population_size:
        Molecules per generation (default 64).
    generations:
        Number of generations (default 20).
    mutation_rate:
        Expected fraction of characters mutated per offspring (default
        0.08; at least one character whenever mutation fires).
    crossover_rate:
        Probability an offspring is a splice of two parents instead of a
        clone of the first (default 0.7).  Ignored in ``scaffold`` mutation
        mode (see ``mutation``) and in adaptive operator selection (the
        crossover-vs-clone choice becomes a credit-driven draw -- see
        ``operator_selection``).
    elitism:
        Top molecules copied verbatim into the next generation (default
        4; guarantees a monotone best score).
    tournament_k:
        Molecules compared per selection tournament (default 3).
    max_smiles_length:
        Hard length cap on every genome -- offspring that exceed it are
        rejected (default 120).
    min_heavy_atoms:
        Opt-in minimum heavy-atom sanity gate threaded into the
        ``"scaffold_hop"`` crossover (default 1 = gate off, every
        historical trajectory byte-identical).  Motivating comic: at
        certain draw states the BREED cut breeds ``'OO'`` (hydrogen
        peroxide) or ``'CC'`` (ethane) children -- contract-conformant
        but with no medchem meaning; ``min_heavy_atoms=3`` rejects
        exactly that class.  Scope: the gate polices the *crossover*
        offspring; the character-mutation fallback can still mint short
        strings by deletion (selection pressure handles those).  Plain
        and numpy integers are accepted; bools and values below 1 raise
        :class:`~pharmaml.exceptions.DataValidationError`.
    mutation:
        ``"uniform"`` (default) -- the classic character-level
        substitution / insertion / deletion.  ``"scaffold"`` --
        **scaffold-preserving decoration resampling**: the offspring keeps
        the parent's Murcko framework verbatim while a random subset of
        its decorations is replaced with entries sampled from a
        decoration vocabulary derived from the corpus (or dropped).  This
        is a pragmatic lead-optimisation operator in the spirit of
        LEADD's scaffold decoration: it explores the decoration space of
        one fixed framework.  It is **not** a full scaffold-decorator
        generative model like REINVENT 2.0 (no attachment-point
        generative grammar, no learned decoration distribution -- just
        corpus-derived resampling with ``Chem.molzip`` re-assembly).
        Parents without a Murcko scaffold (acyclic molecules) or without
        decorations (bare ring systems) fall back to the uniform
        mutation -- never an error.  ``"topology"`` --
        **ring-size mutation**: each offspring undergoes one ring
        contraction (an atom removed from a saturated ring, its ring
        neighbours rejoined) or expansion (a methylene inserted into a
        saturated single ring bond), drawn from
        :func:`~pharmaml.generative.ring_topology_moves` (only fully
        saturated single-bond rings are touched; fused/spiro atoms and
        shared bonds stay put; the ring-size window is 5..8 by
        default).  Acyclic or aromatic-only parents fall back to the
        uniform mutation -- never an error.  Pair with
        ``crossover="scaffold_hop"`` for the full structural move set:
        recombination plus ring-size mutation.
    crossover:
        ``"splice"`` (default) -- the classic single-point character
        splice.  ``"scaffold_hop"`` -- **BREED-style structural
        recombination** via
        :func:`~pharmaml.generative.scaffold_hop_crossover`: both parents
        are cut at a pair of homologous bonds (matching
        atomic number / degree / aromaticity / formal charge at the
        join) and recombined with ``Chem.molzip`` -- either a pair of
        single acyclic bonds (fragments swapped cross-wise) or single
        ring bonds (rings merged into one combined ring, or a two-ended
        linker -- a single atom or a contiguous acyclic path of up to
        four atoms -- transplanted into a cut ring bond: ring
        homologation by one atom or a whole linker chain -- or a whole
        ring excised from one parent between its two single anchor
        bonds and spliced across the other's opened ring bond: a fused
        / bridged bicyclic child holding one ring from each parent, the
        full Nicolaou move set -- or a whole ring spliced INTO the
        other's cut acyclic bond, so both bridge halves re-join
        *through* it: the ring becomes the spacer, the ring-as-linker
        move -- or a two-point segment swap: each parent's bounded
        acyclic middle (a single atom or a contiguous path of up to
        four atoms) excised from the flank parent and replaced by the
        other's homologous middle, the only size-conserving geometry
        and the medchem linker / bioisostere exchange -- an ester O for
        an amide N, an ether O for a thioether S).  Every
        fragment arrives intact, so ring systems
        survive by construction, and whenever the parents' frameworks
        span different fragments (biaryls, diaryl ethers, sulfonyl
        bridges) the offspring's framework differs from BOTH parents --
        the scaffold-hopping operator.  Parents without a viable cut
        fall back to ``parent_a`` -- never an error.  The character
        splice is bypassed in ``mutation="scaffold"`` mode (it destroys
        ring systems); the structural scaffold-hop crossover is NOT --
        ``mutation="scaffold", crossover="scaffold_hop"`` is the full
        structural mode (framework recombination + decoration
        resampling).
    seed:
        Seed for the dedicated ``random.Random`` instance (default 42).
        Two runs with the same seed produce identical results -- in
        uniform AND adaptive operator-selection mode.
    operator_selection:
        ``"uniform"`` (default) -- the historical fixed-rate operator
        draws: the ``crossover_rate`` Bernoulli, the flat candidate-pool
        draw inside the scaffold-hop crossover, the mutation mode's
        structural if-chain.  The run is bit-identical to the optimizer
        without this parameter.  ``"adaptive"`` -- **adaptive operator
        selection**: three levels of operator choice bias themselves by
        per-operator credit -- level 1 (crossover vs clone per
        offspring), the recombination geometry (per scaffold-hop
        crossover: the seven buckets of
        :data:`~pharmaml.generative.scaffold_hop.GEOMETRY_ORDER`,
        renormalised over the geometries present in the pair's candidate
        pool; the splice crossover degenerates to the single
        ``"splice"`` operator) and the mutation family (per offspring:
        the mode's primary mutation -- topology moves or scaffold
        decoration resampling -- vs the character mutation; uniform
        mutation mode has no family choice, so no draw is spent).  A
        drawn operator earns ``(valid offspring ? score : 0.0) -
        mid-parent`` on a sliding ``aos_window``-attempt window (its
        product discarded -- ``None`` return / fallback -- the null
        reward ``0.0 - mid_parent``); selection is a softmax over the
        windowed credit at ``aos_temperature`` with an exploration
        floor of ``1 / (k * 5)``, one ``rng.random()`` per selection.
        Adaptive runs add ``"operator_selection": "adaptive"`` and
        ``"operator_stats"`` (per level: ``{operator: {draws,
        successes, credit, final_probability, best_offspring,
        best_offspring_score}}``) to the result metadata; uniform runs'
        metadata is unchanged.  See :meth:`optimize`'s Returns notes for
        the 4-decimal-place rounding of the stat values, the
        scaffold+splice level-1 bypass and the geometry-draw shortfall
        on no-candidate pairs.
    aos_window:
        Adaptive mode: the sliding credit window per operator (default
        20 -- the last 20 attempts of each operator decide its credit).
    aos_temperature:
        Adaptive mode: the softmax temperature over credits (default
        0.1 -- credit differences of about 0.1 already swing the draw
        probabilities decisively; higher flattens the selection, lower
        hardens it).

    Raises
    ------
    DataValidationError
        On out-of-range hyperparameters.

    Example
    -------
    >>> from pharmaml.generative import GeneticOptimizer, QedObjective
    >>> # lead-optimisation mode: keep aspirin's framework, vary decorations
    >>> result = GeneticOptimizer(
    ...     QedObjective(), population_size=16, generations=3, seed=42,
    ...     mutation="scaffold",
    ... ).optimize(seed_smiles=["CC(=O)Oc1ccccc1C(=O)O"],
    ...              corpus=["CC(=O)Oc1ccccc1C(=O)O"])
    >>> len(result) > 0
    True
    >>> # scaffold-hop mode: BREED-style homologous-cut recombination
    >>> hopped = GeneticOptimizer(
    ...     QedObjective(), population_size=16, generations=3, seed=42,
    ...     crossover="scaffold_hop",
    ... ).optimize()
    >>> hopped.metadata["crossover"]
    'scaffold_hop'
    >>> # adaptive mode: operator draws follow per-operator credit
    >>> tuned = GeneticOptimizer(
    ...     QedObjective(), population_size=16, generations=3, seed=42,
    ...     crossover="scaffold_hop", operator_selection="adaptive",
    ... ).optimize()
    >>> tuned.metadata["operator_selection"]
    'adaptive'
    >>> "geometry" in tuned.metadata["operator_stats"]
    True
    """

    def __init__(
        self,
        objective: Any,
        *,
        population_size: int = 64,
        generations: int = 20,
        mutation_rate: float = 0.08,
        crossover_rate: float = 0.7,
        elitism: int = 4,
        tournament_k: int = 3,
        max_smiles_length: int = 120,
        mutation: str = "uniform",
        crossover: str = "splice",
        operator_selection: str = "uniform",
        aos_window: int = 20,
        aos_temperature: float = 0.1,
        seed: int = 42,
        min_heavy_atoms: int = 1,
    ) -> None:
        if not callable(getattr(objective, "score", None)):
            raise DataValidationError(
                "objective must expose a callable score(smiles) method"
            )
        if population_size < 4:
            raise DataValidationError(
                f"population_size must be >= 4, got {population_size}"
            )
        if generations < 1:
            raise DataValidationError(f"generations must be >= 1, got {generations}")
        if not 0.0 < mutation_rate <= 1.0:
            raise DataValidationError(
                f"mutation_rate must be in (0, 1], got {mutation_rate}"
            )
        if not 0.0 <= crossover_rate <= 1.0:
            raise DataValidationError(
                f"crossover_rate must be in [0, 1], got {crossover_rate}"
            )
        if not 0 <= elitism < population_size:
            raise DataValidationError(
                f"elitism must be in [0, population_size), got {elitism} "
                f"with population_size={population_size}"
            )
        if tournament_k < 2:
            raise DataValidationError(f"tournament_k must be >= 2, got {tournament_k}")
        if max_smiles_length < 20:
            raise DataValidationError(
                f"max_smiles_length must be >= 20, got {max_smiles_length}"
            )
        if mutation not in ("uniform", "scaffold", "topology"):
            raise DataValidationError(
                f"mutation must be 'uniform', 'scaffold' or 'topology', "
                f"got {mutation!r}"
            )
        if crossover not in ("splice", "scaffold_hop"):
            raise DataValidationError(
                f"crossover must be 'splice' or 'scaffold_hop', got {crossover!r}"
            )
        if operator_selection not in ("uniform", "adaptive"):
            raise DataValidationError(
                f"operator_selection must be 'uniform' or 'adaptive', "
                f"got {operator_selection!r}"
            )
        if aos_window < 1:
            raise DataValidationError(f"aos_window must be >= 1, got {aos_window}")
        if not math.isfinite(aos_temperature) or aos_temperature <= 0.0:
            raise DataValidationError(
                f"aos_temperature must be a finite positive number, "
                f"got {aos_temperature}"
            )
        try:
            self.min_heavy_atoms = _checked_min_heavy_atoms(min_heavy_atoms)
        except ValueError as exc:
            raise DataValidationError(str(exc)) from exc
        self.objective = objective
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_k = tournament_k
        self.max_smiles_length = max_smiles_length
        self.mutation = mutation
        self.crossover = crossover
        self.operator_selection = operator_selection
        self.aos_window = aos_window
        self.aos_temperature = aos_temperature
        self.seed = seed

    # -- internals -------------------------------------------------------
    def _mutation_alphabet(self, corpus: list[str]) -> list[str]:
        """Characters the mutator draws from (the corpus alphabet)."""
        alphabet = smiles_alphabet(corpus)
        return alphabet or list("CCcO()=")

    def _mutate(
        self,
        smiles: str,
        alphabet: list[str],
        rng: random.Random,
        decoration_pool: Sequence[str] = (),
    ) -> str:
        """Mutate ``smiles``: ring-size move, decoration resampling or character edits.

        In ``topology`` mode a ring contraction / expansion is drawn
        first (``None`` returns -- acyclic parent, aromatic-only rings,
        over-length mutant -- fall through).  In ``scaffold`` mode (with
        a non-empty decoration pool) the scaffold-preserving operator
        fires next; ``None`` returns (acyclic parent, RDKit failure,
        over-length mutant) fall through to the classic character-level
        mutation so evolution never stalls.
        """
        if self.mutation == "topology":
            mutant = ring_topology_mutate(
                smiles, rng, max_smiles_length=self.max_smiles_length
            )
            if mutant is not None:
                return mutant
        if decoration_pool:
            mutant = _scaffold_mutate(smiles, decoration_pool, rng, self.max_smiles_length)
            if mutant is not None:
                return mutant
        return self._character_mutate(smiles, alphabet, rng)

    def _character_mutate(
        self, smiles: str, alphabet: list[str], rng: random.Random
    ) -> str:
        """Character-level substitution / insertion / deletion mutation.

        The classic uniform mutation and the universal fallback: at
        least one character is substituted, inserted or deleted, drawn
        from the corpus alphabet.  Extracted verbatim from ``_mutate``
        (round 100x-38) so the adaptive mutation-family stage can draw
        it directly with identical rng behaviour.
        """
        chars = list(smiles)
        n_mutations = max(1, round(len(chars) * self.mutation_rate))
        for _ in range(n_mutations):
            if not chars:
                break
            position = rng.randrange(len(chars))
            draw = rng.random()
            if draw < 0.45:  # substitute
                chars[position] = alphabet[rng.randrange(len(alphabet))]
            elif draw < 0.8:  # insert
                chars.insert(position, alphabet[rng.randrange(len(alphabet))])
            else:  # delete
                del chars[position]
        return "".join(chars)

    def _crossover(self, parent_a: str, parent_b: str, rng: random.Random) -> str:
        """Recombine two parents: splice or scaffold-hop BREED cut.

        Both operators fall back to ``parent_a`` when they cannot
        produce a viable child (short genomes for the splice; no
        homologous cut or both recombinations unsanitisable / over-length
        for the scaffold hop).
        """
        if self.crossover == "scaffold_hop":
            child = scaffold_hop_crossover(
                parent_a,
                parent_b,
                rng,
                max_smiles_length=self.max_smiles_length,
                min_heavy_atoms=self.min_heavy_atoms,
            )
            return child if child is not None else parent_a
        if len(parent_a) < 2 or len(parent_b) < 2:
            return parent_a
        cut_a = rng.randrange(1, len(parent_a))
        cut_b = rng.randrange(1, len(parent_b))
        child = parent_a[:cut_a] + parent_b[cut_b:]
        if len(child) > self.max_smiles_length:
            return parent_a
        return child

    def _recombine(
        self,
        parent_a: str,
        parent_b: str,
        rng: random.Random,
        geometry_weights: Mapping[str, float],
    ) -> tuple[str, bool, str | None]:
        """Adaptive-mode recombination with an operator trace.

        Returns ``(child, recombined, geometry)``: ``child`` is the
        product (``parent_a`` whenever the crossover operator fell
        back), ``recombined`` tells whether the drawn crossover
        operator's own product survived into the child (``False`` on
        every fallback -- the null-reward case), and ``geometry`` is the
        drawn geometry for the level-2 credit (``None`` only when the
        scaffold-hop scan found no candidate at all).  The level-2
        weights flow into :func:`_scaffold_hop_core`, which renormalises
        them over the geometries present in the pair's pool.  The splice
        branch mirrors :meth:`_crossover` line for line -- deliberately
        not shared with it, so the uniform-mode path stays
        byte-identical.
        """
        if self.crossover == "scaffold_hop":
            child, geometry = _scaffold_hop_core(
                parent_a,
                parent_b,
                rng,
                max_smiles_length=self.max_smiles_length,
                geometry_weights=geometry_weights,
                min_heavy_atoms=self.min_heavy_atoms,
            )
            if child is not None:
                return child, True, geometry
            return parent_a, False, geometry
        # splice crossover: the geometry level degenerates to the single
        # "splice" operator -- no draw, no rng consumption
        if len(parent_a) < 2 or len(parent_b) < 2:
            return parent_a, False, "splice"
        cut_a = rng.randrange(1, len(parent_a))
        cut_b = rng.randrange(1, len(parent_b))
        child = parent_a[:cut_a] + parent_b[cut_b:]
        if len(child) > self.max_smiles_length:
            return parent_a, False, "splice"
        return child, True, "splice"

    def _tournament(
        self, population: list[str], fitness: dict[str, float], rng: random.Random
    ) -> str:
        """Pick the fittest of ``tournament_k`` random molecules."""
        best = population[rng.randrange(len(population))]
        for _ in range(self.tournament_k - 1):
            candidate = population[rng.randrange(len(population))]
            if fitness[candidate] > fitness[best]:
                best = candidate
        return best

    def _initial_population(
        self, seed_smiles: Sequence[str] | None, canonical_pool: list[str], rng: random.Random
    ) -> tuple[list[str], list[str]]:
        """Build the generation-0 population (seeds first, corpus padding).

        Returns ``(population, canonical_seeds_used)``.
        """
        initial: list[str] = []
        kept: list[str] = []
        if seed_smiles is not None:
            for smiles in list(seed_smiles):  # caller's sequence stays untouched
                canonical = (
                    canonical_smiles(smiles) if isinstance(smiles, str) else None
                )
                if canonical is None:
                    _LOG.warning("dropping invalid seed SMILES %r", smiles)
                elif len(canonical) > self.max_smiles_length:
                    _LOG.warning(
                        "dropping seed SMILES %r -- longer than max_smiles_length=%d",
                        smiles,
                        self.max_smiles_length,
                    )
                else:
                    kept.append(canonical)
            if not kept:
                raise DataValidationError(
                    "no valid seed molecule(s) provided -- invalid entries were "
                    "dropped with a warning, but at least one must survive"
                )
            _LOG.info(
                "seeded mode: %d user molecule(s) anchor the population", len(kept)
            )
            initial.extend(kept)
        while len(initial) < self.population_size:
            initial.append(canonical_pool[rng.randrange(len(canonical_pool))])
        # de-duplicate while preserving order (seeds come first)
        seen: set[str] = set()
        population: list[str] = []
        for smiles in initial:
            if smiles not in seen:
                seen.add(smiles)
                population.append(smiles)
        # top up with *unique* corpus molecules where the pool allows it...
        attempts = 0
        while (
            len(population) < self.population_size
            and attempts < 10 * self.population_size
        ):
            attempts += 1
            candidate = canonical_pool[rng.randrange(len(canonical_pool))]
            if candidate not in seen:
                seen.add(candidate)
                population.append(candidate)
        if len(population) < self.population_size:
            # ... and with duplicated members when the corpus holds fewer
            # unique molecules than population_size (a tiny user corpus must
            # not hang the run: evolution still works on repeated genomes).
            _LOG.warning(
                "corpus holds only %d unique molecule(s) -- duplicating members "
                "to reach population_size=%d",
                len(population),
                self.population_size,
            )
            base = list(population)
            index = 0
            while len(population) < self.population_size:
                population.append(base[index % len(base)])
                index += 1
        return population, kept

    # -- public API ------------------------------------------------------
    def optimize(
        self,
        seed_smiles: Sequence[str] | None = None,
        corpus: Sequence[str] | None = None,
    ) -> GenerationResult:
        """Run the evolution and return the ranked :class:`GenerationResult`.

        Parameters
        ----------
        seed_smiles:
            Seeded mode: molecules the population is initialised from.
            Invalid entries are warned about and dropped (an error is
            raised only when *no* seed survives).  The caller's list is
            never mutated.  When MORE valid seeds arrive than
            ``population_size``, generation 0 scores them all and the
            population contracts to ``population_size`` from the next
            generation (user molecules are never silently discarded).
        corpus:
            De novo mode: sampling pool for the initial population
            (default: :data:`~pharmaml.generative.DEFAULT_CORPUS`, an
            immutable tuple).  A pool with fewer unique valid molecules
            than ``population_size`` is tolerated: members are
            duplicated (with a warning) rather than hanging the run.

        Returns
        -------
        GenerationResult
            The ranked run.  On adaptive runs the ``metadata`` carries
            ``"operator_selection": "adaptive"``, the per-level
            ``"operator_stats"`` tables (see ``operator_selection``) and
            ``"operator_trajectory"`` -- one full-precision record per
            generation, index-aligned with ``history`` (record 0 is the
            fair prior, the final record the closing mix).
            Notes for ``operator_stats`` consumers:

            * ``credit``, ``final_probability`` and
              ``best_offspring_score`` are rounded to 4 decimal places
              for display, so a level's rounded probabilities can sum
              to 1 with an error of up to ~5e-4 (the seven-operator
              geometry level; the unrounded selection probabilities sum
              to 1 within 1e-12) -- renormalise loosely or not at all,
              never at a tight tolerance;
            * with ``mutation="scaffold"`` and ``crossover="splice"``
              the character splice is bypassed, so the level-1 book is
              never drawn yet still reported with draws 0/0 and the
              neutral 0.5/0.5 probabilities (the geometry book
              likewise reports zero draws);
            * geometry-level draws can fall short of the level-1
              crossover draws whenever a pair offers no viable cut
              candidate at all (the pair falls back to ``parent_a``
              without a geometry draw -- not an operator failure).

        Raises
        ------
        DataValidationError
            If ``seed_smiles`` / ``corpus`` is a bare string (it would
            otherwise be iterated as single-atom CHARACTERS), every
            provided seed molecule is invalid, or the corpus is empty
            after validity / length filtering.  Invalid corpus entries
            are dropped with a warning, never silently.
        """
        started = time.perf_counter()
        rng = random.Random(self.seed)

        # A bare string would silently iterate as CHARACTERS ("CCO" ->
        # ['C', 'C', 'O'] single-atom molecules) -- the exact trap
        # NoveltyObjective guards against for its corpus, so the same
        # typed DataValidationError fires here.
        if seed_smiles is not None and isinstance(seed_smiles, str):
            raise DataValidationError(
                "seed_smiles must be a sequence of SMILES strings, not a "
                f"single string (got {seed_smiles!r} of length "
                f"{len(seed_smiles)})"
            )
        if corpus is not None and isinstance(corpus, str):
            raise DataValidationError(
                "corpus must be a sequence of SMILES strings, not a single "
                f"string (got {corpus!r} of length {len(corpus)})"
            )

        # An explicitly passed *empty* corpus must raise (not silently fall
        # back to the default corpus), so truthiness tricks are off limits.
        pool = list(DEFAULT_CORPUS if corpus is None else corpus)
        n_non_string = sum(1 for s in pool if not isinstance(s, str))
        if n_non_string:
            # silent data loss is the failure mode this package refuses:
            # invalid SEEDS warn, so must non-string corpus entries
            # (the NoveltyObjective warning, sibling-style).
            _LOG.warning(
                "dropping %d non-string corpus entr(ies) -- corpus entries "
                "must be SMILES strings",
                n_non_string,
            )
        pool = [s for s in pool if isinstance(s, str)]
        if not pool:
            raise DataValidationError(
                "corpus is empty (or holds no string entries) -- pass a "
                "non-empty list of SMILES strings"
                + (f" ({n_non_string} non-string entr(ies) dropped)" if n_non_string else "")
            )
        canonical_pool = []
        for smiles in pool:
            canonical = canonical_smiles(smiles)
            if canonical is not None and len(canonical) <= self.max_smiles_length:
                canonical_pool.append(canonical)
        n_dropped = len(pool) - len(canonical_pool)
        if n_dropped:
            # same sibling-style warning the optimizer already gives for
            # invalid seeds: dropped entries must not vanish silently
            _LOG.warning(
                "dropping %d invalid/unparseable corpus entr(ies) (or over "
                "the max_smiles_length cap) -- the initial population is "
                "drawn from the remaining %d valid molecule(s)",
                n_dropped,
                len(canonical_pool),
            )
        # de-duplicate while preserving order: duplicate spellings of the
        # same molecule (e.g. "CCO" and "OCC") must not skew the initial
        # sampling, and a pool with fewer UNIQUE molecules than
        # population_size must not hang the unique-fill loop below.
        canonical_pool = list(dict.fromkeys(canonical_pool))
        if not canonical_pool:
            raise DataValidationError(
                "corpus holds no valid SMILES under the length cap -- nothing "
                "to initialise the population from"
            )
        alphabet = self._mutation_alphabet(canonical_pool)
        population, kept_seeds = self._initial_population(seed_smiles, canonical_pool, rng)
        decoration_pool: list[str] = []
        if self.mutation == "scaffold":
            decoration_pool = _build_decoration_pool(canonical_pool)
            if not decoration_pool:
                _LOG.warning(
                    "corpus yields no scaffold decorations -- scaffold "
                    "mutations fall back to character mutation"
                )

        # Adaptive operator selection: one credit book per biasable level
        # (level 1 crossover-vs-clone; the recombination geometry -- the
        # seven scaffold-hop buckets, or the single "splice" bucket in
        # splice mode; the mutation family -- the mode's primary operator
        # vs the character mutation, character alone when the mode has no
        # second operator).  Constructing the books is side-effect-free,
        # so the uniform path is untouched.
        adaptive = self.operator_selection == "adaptive"
        level1_book = _OperatorBook(
            ("crossover", "clone"), self.aos_window, self.aos_temperature
        )
        geometry_operators: tuple[str, ...] = (
            ("splice",) if self.crossover == "splice" else GEOMETRY_ORDER
        )
        geometry_book = _OperatorBook(
            geometry_operators, self.aos_window, self.aos_temperature
        )
        if self.mutation == "topology":
            family_operators: tuple[str, ...] = ("topology", "character")
        elif self.mutation == "scaffold" and decoration_pool:
            family_operators = ("scaffold", "character")
        else:
            family_operators = ("character",)
        mutation_book = _OperatorBook(
            family_operators, self.aos_window, self.aos_temperature
        )

        produced: dict[str, float] = {}
        history: list[dict[str, Any]] = []
        operator_trajectory: list[dict[str, Any]] = []
        accepted_total = 0
        attempts_total = 0
        last_fraction = 1.0

        for generation in range(self.generations):
            fitness = {
                smiles: _validated_score(self.objective, smiles)
                for smiles in population
            }
            for smiles, score in fitness.items():
                produced[smiles] = max(produced.get(smiles, 0.0), score)
            ranked = sorted(population, key=lambda s: (-fitness[s], s))
            history.append(
                {
                    "generation": generation,
                    "best_score": fitness[ranked[0]],
                    "mean_score": sum(fitness.values()) / len(fitness),
                    "best_smiles": ranked[0],
                    "n_unique": len(set(population)),
                    "valid_offspring_fraction": last_fraction,
                }
            )
            if adaptive:
                # the operator mix in force at the start of this generation
                # (full precision; the final record doubles as the closing
                # mix -- the last generation produces no offspring).
                # probabilities() is pure and consumes no rng, so the
                # molecule streams are bit-identical either way.
                operator_trajectory.append(
                    {
                        "generation": generation,
                        "level1": level1_book.probabilities(),
                        "geometry": geometry_book.probabilities(),
                        "mutation": mutation_book.probabilities(),
                    }
                )
            if generation == self.generations - 1:
                break

            next_population = ranked[: self.elitism]
            attempts = 0
            while len(next_population) < self.population_size:
                attempts += 1
                if attempts > 50 * self.population_size:  # pragma: no cover
                    _LOG.warning(
                        "mutation/crossover yield collapsed in generation %d "
                        "-- padding the population with corpus molecules",
                        generation,
                    )
                    while len(next_population) < self.population_size:
                        next_population.append(
                            canonical_pool[rng.randrange(len(canonical_pool))]
                        )
                    break
                parent_a = self._tournament(population, fitness, rng)
                drawn_l1: str | None = None
                drawn_geometry: str | None = None
                drawn_family: str | None = None
                recombined = False
                family_survived = False
                mid_parent = 0.0
                if self.mutation == "scaffold" and self.crossover == "splice":
                    # The character-level splice destroys ring systems and
                    # would defeat scaffold preservation, so it is bypassed
                    # in scaffold mode (crossover_rate does not apply).
                    # The structural scaffold-hop crossover is NOT bypassed
                    # -- it is the mode's natural partner.
                    child = parent_a
                else:
                    parent_b = self._tournament(population, fitness, rng)
                    mid_parent = (fitness[parent_a] + fitness[parent_b]) / 2.0
                    if adaptive:
                        # level 1: the crossover-vs-clone Bernoulli becomes a
                        # credit-driven draw (uniform mode: the exact
                        # historical rng.random() comparison, unchanged)
                        drawn_l1 = level1_book.draw(rng)
                        use_cross = drawn_l1 == "crossover"
                    else:
                        use_cross = rng.random() < self.crossover_rate
                    if use_cross:
                        if adaptive:
                            (
                                child,
                                recombined,
                                drawn_geometry,
                            ) = self._recombine(
                                parent_a,
                                parent_b,
                                rng,
                                geometry_book.probabilities(),
                            )
                        else:
                            child = self._crossover(parent_a, parent_b, rng)
                    else:
                        child = parent_a
                if not adaptive:
                    child = self._mutate(child, alphabet, rng, decoration_pool)
                elif len(family_operators) == 2:
                    # level 3: the mutation family's structural if-chain
                    # becomes a weighted draw with identical fallback
                    # semantics (a failed primary mutation falls back to
                    # the character mutation -- the surviving operator
                    # earns the reward, the failed one the null reward)
                    drawn_family = mutation_book.draw(rng)
                    if drawn_family == family_operators[0]:
                        if self.mutation == "topology":
                            mutant = ring_topology_mutate(
                                child, rng, max_smiles_length=self.max_smiles_length
                            )
                        else:
                            mutant = _scaffold_mutate(
                                child, decoration_pool, rng, self.max_smiles_length
                            )
                        if mutant is not None:
                            child = mutant
                            family_survived = True
                        else:
                            child = self._character_mutate(child, alphabet, rng)
                    else:
                        child = self._character_mutate(child, alphabet, rng)
                        family_survived = True
                else:
                    # single-operator family (uniform mutation, or a scaffold
                    # corpus without decorations): no draw, no rng spent --
                    # the character mutation is the only operator
                    child = self._character_mutate(child, alphabet, rng)
                    drawn_family = family_operators[0]
                    family_survived = True
                # validity gates -- the uniform path's two `continue`s,
                # restructured so the adaptive credit assignment also sees
                # the failed attempts (identical call order and appends)
                valid = False
                child_canonical: str | None = None
                score_child: float | None = None
                if child and len(child) <= self.max_smiles_length:
                    child_canonical = canonical_smiles(child)
                    if (
                        child_canonical is not None
                        and len(child_canonical) <= self.max_smiles_length
                    ):
                        valid = True
                        if adaptive:
                            score_child = _validated_score(
                                self.objective, child_canonical
                            )
                        if child_canonical not in next_population:
                            next_population.append(child_canonical)
                if adaptive:
                    # credit assignment: the end-to-end reward vs the null
                    # reward for a drawn operator whose product was
                    # discarded (mid_parent = 0.0 in the scaffold+splice
                    # bypass, where no second parent exists)
                    end_to_end = (score_child if score_child is not None else 0.0) - (
                        mid_parent
                    )
                    null_reward = -mid_parent
                    if drawn_l1 is not None:
                        level1_book.push(
                            drawn_l1,
                            end_to_end
                            if (drawn_l1 == "clone" or recombined)
                            else null_reward,
                            valid,
                            child_canonical if valid else None,
                            score_child,
                        )
                    if drawn_geometry is not None:
                        geometry_book.push(
                            drawn_geometry,
                            end_to_end if recombined else null_reward,
                            valid and recombined,
                            child_canonical if valid else None,
                            score_child,
                        )
                    if drawn_family is not None:
                        mutation_book.push(
                            drawn_family,
                            end_to_end if family_survived else null_reward,
                            valid and family_survived,
                            child_canonical if valid else None,
                            score_child,
                        )
            accepted = self.population_size - self.elitism
            attempts_total += attempts
            accepted_total += accepted
            if attempts:
                last_fraction = accepted / attempts
            history[-1]["valid_offspring_fraction"] = last_fraction
            population = next_population

        ranked_all = sorted(produced.items(), key=lambda kv: (-kv[1], kv[0]))
        smiles_list = [smiles for smiles, _ in ranked_all]
        score_list = [score for _, score in ranked_all]
        breakdown: list[dict[str, float]] = []
        score_with_breakdown = getattr(self.objective, "score_with_breakdown", None)
        for smiles in smiles_list:
            if callable(score_with_breakdown):
                breakdown.append(score_with_breakdown(smiles)[1])
            else:
                breakdown.append({})
        elapsed = time.perf_counter() - started
        metadata: dict[str, Any] = {
            "method": "genetic_algorithm",
            "seed": self.seed,
            "iterations": self.generations,
            "elapsed_s": elapsed,
            "valid_generated": accepted_total + len(population),
            "unique_generated": len(smiles_list),
            "population_size": self.population_size,
            "mutation": self.mutation,
            "crossover": self.crossover,
            "seed_molecules": kept_seeds,
        }
        if adaptive:
            # adaptive runs ONLY -- uniform-run metadata stays byte-
            # identical (no new keys, no reordered values)
            metadata["operator_selection"] = "adaptive"
            metadata["operator_stats"] = {
                "level1": level1_book.snapshot(),
                "geometry": geometry_book.snapshot(),
                "mutation": mutation_book.snapshot(),
            }
            # a SEPARATE top-level key (never nested inside operator_stats:
            # the per-operator stat tables keep their exact pinned shape)
            metadata["operator_trajectory"] = operator_trajectory
        return GenerationResult(
            smiles=smiles_list,
            scores=score_list,
            breakdown=breakdown,
            best=smiles_list[0] if smiles_list else "",
            best_score=score_list[0] if score_list else 0.0,
            history=history,
            metadata=metadata,
        )


def run_genetic_optimization(
    objective: Any,
    seed_smiles: Sequence[str] | None = None,
    corpus: Sequence[str] | None = None,
    **kwargs: Any,
) -> GenerationResult:
    """One-call genetic optimisation (thin :class:`GeneticOptimizer` wrapper).

    Example
    -------
    >>> from pharmaml.generative import QedObjective, run_genetic_optimization
    >>> result = run_genetic_optimization(QedObjective(), generations=5, seed=7)
    >>> len(result) > 0
    True
    """
    optimizer = GeneticOptimizer(objective, **kwargs)
    return optimizer.optimize(seed_smiles=seed_smiles, corpus=corpus)
