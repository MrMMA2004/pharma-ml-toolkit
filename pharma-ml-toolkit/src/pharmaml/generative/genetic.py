"""SMILES genetic algorithm: de novo and seeded molecule generation.

A pragmatic population-based optimiser in the GuacaMol / LEADD tradition,
CPU-friendly and deterministic given ``seed``:

* **genome** = canonical SMILES string (character level);
* **mutation** = random character substitution / insertion / deletion,
  followed by RDKit sanitisation and re-canonicalisation (invalid mutants
  are discarded);
* **crossover** = single-point splice of two parents, sanitised the same
  way;
* **selection** = tournament of ``tournament_k`` molecules;
* **elitism** = the top ``elitism`` molecules survive a generation
  verbatim, so the best score is monotone across generations.

Two initialisation modes:

* **de novo** (``seed_smiles=None``): the population is sampled from
  ``corpus`` (default: :data:`~pharmaml.generative.DEFAULT_CORPUS`);
* **seeded** (``seed_smiles=[...]``): the population starts from the
  user's molecules (duplicated to fill the population), so the search
  explores their neighbourhood -- the "with some sample molecules" mode.

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
from typing import Any

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

from ._utils import canonical_smiles, smiles_alphabet
from .corpus import DEFAULT_CORPUS
from .result import GenerationResult

__all__ = ["GeneticOptimizer", "run_genetic_optimization"]

_LOG = get_logger(__name__)


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
        clone of the first (default 0.7).
    elitism:
        Top molecules copied verbatim into the next generation (default
        4; guarantees a monotone best score).
    tournament_k:
        Molecules compared per selection tournament (default 3).
    max_smiles_length:
        Hard length cap on every genome -- offspring that exceed it are
        rejected (default 120).
    seed:
        Seed for the dedicated ``random.Random`` instance (default 42).
        Two runs with the same seed produce identical results.

    Raises
    ------
    DataValidationError
        On out-of-range hyperparameters.
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
        self.objective = objective
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_k = tournament_k
        self.max_smiles_length = max_smiles_length
        self.seed = seed

    # -- internals -------------------------------------------------------
    def _mutation_alphabet(self, corpus: list[str]) -> list[str]:
        """Characters the mutator draws from (the corpus alphabet)."""
        alphabet = smiles_alphabet(corpus)
        return alphabet or list("CCcO()=")

    def _mutate(self, smiles: str, alphabet: list[str], rng: random.Random) -> str:
        """Substitute / insert / delete ~``mutation_rate`` of the characters."""
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
        """Single-point splice; over-long children fall back to parent A."""
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
            If every provided seed molecule is invalid, or the corpus is
            empty after validity / length filtering.
        """
        started = time.perf_counter()
        rng = random.Random(self.seed)

        # An explicitly passed *empty* corpus must raise (not silently fall
        # back to the default corpus), so truthiness tricks are off limits.
        pool = list(DEFAULT_CORPUS if corpus is None else corpus)
        n_non_string = sum(1 for s in pool if not isinstance(s, str))
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
                parent_b = self._tournament(population, fitness, rng)
                if rng.random() < self.crossover_rate:
                    child = self._crossover(parent_a, parent_b, rng)
                else:
                    child = parent_a
                child = self._mutate(child, alphabet, rng)
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
