"""100x-21-b probe 9: cross-process determinism re-check after the corpus fix.

Runs one GA and one RL optimisation in THIS process and prints a hash of
the ranked output; the worklog's convention is to run the script twice in
separate interpreters and compare.
"""
from __future__ import annotations

import hashlib
import sys

from pharmaml.generative import (
    GeneticOptimizer,
    MultiObjective,
    QedObjective,
    LogPBandObjective,
    ReinventOptimizer,
)

objective = MultiObjective(
    [QedObjective(), LogPBandObjective(1.0, 3.0)], weights=[0.7, 0.3]
)

ga = GeneticOptimizer(objective, population_size=48, generations=10,
                      seed=42).optimize()
rl = ReinventOptimizer(objective, iterations=15, seed=42).optimize()

ga_blob = repr((ga.smiles[:50], [round(s, 6) for s in ga.scores[:50]])).encode()
rl_blob = repr((rl.smiles[:50], [round(s, 6) for s in rl.scores[:50]])).encode()
print("GA best:", ga.best, round(ga.best_score, 4), "n:", len(ga))
print("RL best:", rl.best, round(rl.best_score, 4), "n:", len(rl))
print("GA sha256:", hashlib.sha256(ga_blob).hexdigest()[:16])
print("RL sha256:", hashlib.sha256(rl_blob).hexdigest()[:16])
sys.stdout.flush()
