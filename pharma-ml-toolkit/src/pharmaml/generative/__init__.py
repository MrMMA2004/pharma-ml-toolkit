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
   genetic algorithm: character, scaffold-preserving or ring-topology
   mutation; splice **or** BREED-style
   scaffold-hop crossover -- homologous bridge cuts swap fragments,
   ring-bond cuts merge rings or transplant two-ended linkers (a single
   atom or a short acyclic path of up to four atoms), whole rings
   excised between their two anchor bonds splice across the cut into
   fused bicyclics, whole rings spliced INTO cut acyclic bonds become
   the linker, and two-point segment swaps exchange each parent's
   bounded acyclic middle for the other's homologous middle (the
   linker / bioisostere exchange: an ester O for an amide N) -- so ring
   *topology*, ring-*system* topology and *linker chemistry*
   recombine too; tournament selection, elitism; **adaptive operator
   selection** -- ``operator_selection="adaptive"`` makes the GA learn
   in-run which operators pay off (credit-assigned softmax-with-floor
   draws over crossover-vs-clone, the crossover geometries and the
   mutation families, with per-operator statistics and the per-generation
   operator-mix trajectory ``metadata["operator_trajectory"]`` (rendered
   by :func:`pharmaml.viz.plot_operator_trajectory`) in the result
   metadata; CLI ``--adaptive-operators``); CPU-strong,
   de novo *or* seeded from your own sample molecules -- the
   scaffold-hop crossover also works standalone via
   :func:`~pharmaml.generative.scaffold_hop_crossover`, and the
   single-parent ring-size moves (contraction / expansion) via
   :func:`~pharmaml.generative.ring_topology_moves` /
   :func:`~pharmaml.generative.ring_topology_mutate`) and/or
   :class:`~pharmaml.generative.ReinventOptimizer`
   (REINVENT-style reinforcement learning: a character-level GRU prior
   trained on the corpus, fine-tuned by REINFORCE with a running
   baseline, entropy bonus, prior anchor and experience replay -- the
   "with some sample molecules" mode works by seeding the prior's corpus);
4. **Inspect** -- every run returns a
   :class:`~pharmaml.generative.GenerationResult` (ranked deduplicated
   canonical SMILES, per-objective breakdown, per-iteration history,
   metadata) with ``to_dataframe()`` for the leaderboard; a
   :class:`~pharmaml.generative.MultiObjective` run additionally answers
   *what are the trade-offs?* -- :func:`~pharmaml.generative.pareto_front`
   extracts the non-dominated set from the breakdown (the molecules where
   improving one objective necessarily sacrifices another; front members
   can sit arbitrarily deep in the leaderboard), rendered by
   :func:`pharmaml.viz.plot_pareto_front`; an RL run's ``history``
   additionally records the **prior-drift introspection** (the squared
   agent-vs-prior log-probability gap the anchor penalises, the
   exploration entropy, and both mean sequence log-probabilities --
   iteration 0's drift is exactly ``0.0`` because the agent starts as a
   bit-exact copy of the frozen prior), rendered by
   :func:`pharmaml.viz.plot_rl_introspection`.

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
('CCCNc1ccccc1', 0.728)

Dependencies: RDKit (``pip install 'pharmaml[chem]'``) is required for
scoring and the genetic algorithm; PyTorch (``pip install
'pharmaml[deep]'``) additionally for the RNN / RL half.  Both are
imported lazily -- importing this subpackage needs neither.
"""

from __future__ import annotations

from .bioisostere import bioisostere_moves, bioisostere_mutate
from .corpus import DEFAULT_CORPUS
from .genetic import GeneticOptimizer, run_genetic_optimization
from .pareto import ParetoFront, pareto_front
from .result import GenerationResult
from .rl import ReinventOptimizer, run_rl_optimization
from .rnn import SmilesRnn
from .scaffold_hop import scaffold_hop_crossover
from .scoring import (
    LogPBandObjective,
    MolecularWeightBandObjective,
    MultiObjective,
    NoveltyObjective,
    QedObjective,
    ScoringObjective,
    SimilarityObjective,
    SubstructureObjective,
    TpsaBandObjective,
)
from .topology import ring_topology_moves, ring_topology_mutate

__all__ = [
    "DEFAULT_CORPUS",
    "GenerationResult",
    "GeneticOptimizer",
    "LogPBandObjective",
    "MolecularWeightBandObjective",
    "MultiObjective",
    "NoveltyObjective",
    "ParetoFront",
    "QedObjective",
    "ReinventOptimizer",
    "ScoringObjective",
    "SimilarityObjective",
    "SmilesRnn",
    "SubstructureObjective",
    "TpsaBandObjective",
    "bioisostere_moves",
    "bioisostere_mutate",
    "pareto_front",
    "ring_topology_moves",
    "ring_topology_mutate",
    "run_genetic_optimization",
    "run_rl_optimization",
    "scaffold_hop_crossover",
]
