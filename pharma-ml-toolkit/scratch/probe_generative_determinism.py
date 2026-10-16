"""Determinism evidence for the generative module (task 100x-21-a).

Two SEPARATE interpreter processes each run the GA and the RL optimiser with
the same seed and write their results to JSON; this script then compares the
two dumps byte-for-byte.  Run from the repository root::

    python scratch/probe_generative_determinism.py
"""

from __future__ import annotations

import json
import subprocess
import sys

INNER = """
import json, sys
from pharmaml.generative import (
    DEFAULT_CORPUS, GeneticOptimizer, LogPBandObjective, MultiObjective,
    QedObjective, ReinventOptimizer, SmilesRnn,
)

objective = MultiObjective(
    [QedObjective(), LogPBandObjective(1.0, 3.0)], weights=[0.7, 0.3]
)
ga = GeneticOptimizer(objective, population_size=48, generations=20, seed=42).optimize()
prior = SmilesRnn(hidden_dim=96, seed=42)
prior.train_prior(DEFAULT_CORPUS, epochs=8)
rl = ReinventOptimizer(
    objective, prior=prior, iterations=15, batch_size=32, seed=42
).optimize()
means = [entry["mean_score"] for entry in rl.history]
json.dump(
    {
        "ga_smiles": ga.smiles,
        "ga_scores": ga.scores,
        "rl_smiles": rl.smiles,
        "rl_scores": rl.scores,
        "rl_means": means,
    },
    open(sys.argv[1], "w"),
)
"""


def main() -> int:
    for run in (1, 2):
        subprocess.run(
            [sys.executable, "-c", INNER, f"/tmp/det_{run}.json"], check=True
        )
        print(f"process {run} finished")
    first = json.load(open("/tmp/det_1.json"))
    second = json.load(open("/tmp/det_2.json"))
    print(
        "GA determinism across processes:",
        first["ga_smiles"] == second["ga_smiles"]
        and first["ga_scores"] == second["ga_scores"],
    )
    print(
        "RL determinism across processes:",
        first["rl_smiles"] == second["rl_smiles"]
        and first["rl_scores"] == second["rl_scores"],
    )
    print(
        "GA best:", first["ga_smiles"][0], round(first["ga_scores"][0], 4),
        "| n =", len(first["ga_smiles"]),
    )
    print(
        "RL best:", first["rl_smiles"][0], round(first["rl_scores"][0], 4),
        "| n =", len(first["rl_smiles"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
