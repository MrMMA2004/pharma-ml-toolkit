"""Molecule generation from 0 to 100: objectives, genetic algorithm, RL.

The generative subpackage answers the question *"can the toolkit invent
molecules, not just analyse them?"* with two complementary optimisers,
one shared objective library and one bundled corpus -- the full story
runs in seconds on a laptop CPU:

1. **Imports** -- the bundled, curated corpus
   (:data:`~pharmaml.generative.DEFAULT_CORPUS`, 244 canonical drug-like
   SMILES) and the objective library from :mod:`pharmaml.generative.scoring`;
2. **Define the objective** -- what does a *good* molecule mean here?
   QED drug-likeness, a LogP window, similarity to a seed molecule, a
   required pharmacophore -- or a :class:`~pharmaml.generative.MultiObjective`
   blending several (one zero vetoes the whole molecule);
3. **Optimise** --
   :class:`~pharmaml.generative.GeneticOptimizer` (a pragmatic SMILES
   genetic algorithm: character mutation + splice crossover + tournament
   selection + elitism; CPU-strong, de novo *or* seeded from your own
   sample molecules) and/or :class:`~pharmaml.generative.ReinventOptimizer`
   (REINVENT-style reinforcement learning: a character-level GRU prior
   trained on the corpus, fine-tuned by REINFORCE with a running
   baseline, entropy bonus, prior anchor and experience replay -- the
   "with some sample molecules" mode works by seeding the prior's corpus);
4. **Inspect** -- every run returns a
   :class:`~pharmaml.generative.GenerationResult` (ranked deduplicated
   canonical SMILES, per-objective breakdown, per-iteration history,
   metadata) with ``to_dataframe()`` for the leaderboard.

Example
-------
>>> from pharmaml.generative import (
...     DEFAULT_CORPUS, QedObjective, LogPBandObjective, MultiObjective,
...     GeneticOptimizer, ReinventOptimizer)
>>>
>>> objective = MultiObjective(
...     [QedObjective(), LogPBandObjective(1.0, 3.0)], weights=[0.7, 0.3])
>>>
>>> ga = GeneticOptimizer(objective, population_size=48, generations=10,
...                       seed=42).optimize()          # de novo mode
>>> seeded = GeneticOptimizer(objective, population_size=48, generations=10,
...                           seed=42).optimize(seed_smiles=["CC(=O)Oc1ccccc1C(=O)O"])
>>> rl = ReinventOptimizer(objective, iterations=15, seed=42).optimize()
>>> rl.best, round(rl.best_score, 3)
('CCOc1ccccc1O', 0.754)

Dependencies: RDKit (``pip install 'pharmaml[chem]'``) is required for
scoring and the genetic algorithm; PyTorch (``pip install
'pharmaml[deep]'``) additionally for the RNN / RL half.  Both are
imported lazily -- importing this subpackage needs neither.
"""

from __future__ import annotations

from .corpus import DEFAULT_CORPUS
from .genetic import GeneticOptimizer, run_genetic_optimization
from .result import GenerationResult
from .rl import ReinventOptimizer, run_rl_optimization
from .rnn import SmilesRnn
from .scoring import (
    LogPBandObjective,
    MolecularWeightBandObjective,
    MultiObjective,
    QedObjective,
    ScoringObjective,
    SimilarityObjective,
    SubstructureObjective,
    TpsaBandObjective,
)

__all__ = [
    "DEFAULT_CORPUS",
    "GenerationResult",
    "GeneticOptimizer",
    "LogPBandObjective",
    "MolecularWeightBandObjective",
    "MultiObjective",
    "QedObjective",
    "ReinventOptimizer",
    "ScoringObjective",
    "SimilarityObjective",
    "SmilesRnn",
    "SubstructureObjective",
    "TpsaBandObjective",
    "run_genetic_optimization",
    "run_rl_optimization",
]
