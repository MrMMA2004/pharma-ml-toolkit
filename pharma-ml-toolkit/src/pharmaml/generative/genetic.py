"""SMILES genetic algorithm: de novo and seeded molecule generation.

A pragmatic population-based optimiser in the GuacaMol / LEADD tradition,
CPU-friendly and deterministic given ``seed``:

* **genome** = canonical SMILES string (character level);
* **mutation** = random character substitution / insertion / deletion,
  followed by RDKit sanitisation and re-canonicalisation (invalid mutants
  are discarded) -- OR, in ``mutation="scaffold"`` mode, a
  scaffold-preserving decoration resampling (see :func:`_scaffold_mutate`);
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
  spacer -- ring-as-linker), so
  ring systems survive by construction and
  the offspring can hop scaffolds.  The character splice is not used in
  ``mutation="scaffold"`` mode (it routinely destroys ring systems,
  defeating scaffold preservation); the structural scaffold-hop
  crossover IS available there;
* **selection** = tournament of ``tournament_k`` molecules;
* **elitism** = the top ``elitism`` molecules survive a generation
  verbatim, so the best score is monotone across generations.

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
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

from ._utils import canonical_smiles, require_rdkit, smiles_alphabet
from .corpus import DEFAULT_CORPUS
from .result import GenerationResult
from .scaffold_hop import scaffold_hop_crossover

__all__ = ["GeneticOptimizer", "run_genetic_optimization"]

_LOG = get_logger(__name__)

#: Probability that a resampled decoration slot is dropped entirely (rather
#: than replaced with a corpus decoration) by the scaffold mutation.
_DROP_PROBABILITY = 0.25


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
        mode (see ``mutation``).
    elitism:
        Top molecules copied verbatim into the next generation (default
        4; guarantees a monotone best score).
    tournament_k:
        Molecules compared per selection tournament (default 3).
    max_smiles_length:
        Hard length cap on every genome -- offspring that exceed it are
        rejected (default 120).
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
        mutation -- never an error.
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
        move).  Every
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
        Two runs with the same seed produce identical results.

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
        seed: int = 42,
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
        if mutation not in ("uniform", "scaffold"):
            raise DataValidationError(
                f"mutation must be 'uniform' or 'scaffold', got {mutation!r}"
            )
        if crossover not in ("splice", "scaffold_hop"):
            raise DataValidationError(
                f"crossover must be 'splice' or 'scaffold_hop', got {crossover!r}"
            )
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
        """Mutate ``smiles``: decoration resampling or character edits.

        In ``scaffold`` mode (with a non-empty decoration pool) the
        scaffold-preserving operator fires first; ``None`` returns (acyclic
        parent, RDKit failure, over-length mutant) fall through to the
        classic character-level mutation so evolution never stalls.
        """
        if decoration_pool:
            mutant = _scaffold_mutate(smiles, decoration_pool, rng, self.max_smiles_length)
            if mutant is not None:
                return mutant
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
                parent_a, parent_b, rng, max_smiles_length=self.max_smiles_length
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

        produced: dict[str, float] = {}
        history: list[dict[str, Any]] = []
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
                if self.mutation == "scaffold" and self.crossover == "splice":
                    # The character-level splice destroys ring systems and
                    # would defeat scaffold preservation, so it is bypassed
                    # in scaffold mode (crossover_rate does not apply).
                    # The structural scaffold-hop crossover is NOT bypassed
                    # -- it is the mode's natural partner.
                    child = parent_a
                else:
                    parent_b = self._tournament(population, fitness, rng)
                    if rng.random() < self.crossover_rate:
                        child = self._crossover(parent_a, parent_b, rng)
                    else:
                        child = parent_a
                child = self._mutate(child, alphabet, rng, decoration_pool)
                if not child or len(child) > self.max_smiles_length:
                    continue
                canonical = canonical_smiles(child)
                if canonical is None or len(canonical) > self.max_smiles_length:
                    continue
                if canonical not in next_population:
                    next_population.append(canonical)
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
        return GenerationResult(
            smiles=smiles_list,
            scores=score_list,
            breakdown=breakdown,
            best=smiles_list[0] if smiles_list else "",
            best_score=score_list[0] if score_list else 0.0,
            history=history,
            metadata={
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
            },
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
