"""100x-21-b probe 2: duck-typed MultiObjective + GA population oversizing.

Suggested probe #2 (MultiObjective protocol) + own hunt: does the GA
population silently exceed population_size when len(valid seeds) >
population_size?
"""
from __future__ import annotations

import math

from pharmaml.generative import (
    GeneticOptimizer,
    MultiObjective,
    QedObjective,
    ReinventOptimizer,
    ScoringObjective,
)

# ---- 1. duck-typed objective: only name + score, no subclass -------------
class DuckObjective:
    name = "duck"

    def __init__(self, value):
        self.value = value

    def score(self, smiles):
        return self.value


print("== duck-typed MultiObjective ==")
try:
    obj = MultiObjective([DuckObjective(0.5), DuckObjective(0.8)])
    total, breakdown = obj.score_with_breakdown("CCO")
    print(f"total={total!r} breakdown={breakdown!r}")
    print("n_invalid:", obj.n_invalid, "n_scored:", obj.n_scored)
    print("repr:", repr(obj))
    # reset_stats on MultiObjective?
    obj.reset_stats()
    print("after reset n_scored:", obj.n_scored)
except Exception as exc:
    print(f"FAIL: {type(exc).__name__}: {exc}")

# duck-typed with name as non-string and value-returning non-float
class WeirdDuck:
    name = 42

    def score(self, smiles):
        return 1  # int

try:
    obj = MultiObjective([DuckObjective(0.5), WeirdDuck()])
    print("weird duck total:", obj.score("CCO"))
except Exception as exc:
    print(f"weird duck FAIL: {type(exc).__name__}: {exc}")

# ---- 2. NaN / inf leaking through a misbehaving child objective ----------
class NanDuck:
    name = "nan"

    def score(self, smiles):
        return float("nan")


class InfDuck:
    name = "inf"

    def score(self, smiles):
        return float("inf")


print("\n== NaN/inf child scores ==")
try:
    obj = MultiObjective([DuckObjective(0.5), NanDuck()])
    total, breakdown = obj.score_with_breakdown("CCO")
    print(f"NaN child -> total={total!r} breakdown={breakdown}")
except Exception as exc:
    print(f"NaN child FAIL: {type(exc).__name__}: {exc}")
try:
    obj = MultiObjective([DuckObjective(0.5), InfDuck()])
    print("inf child -> total:", obj.score("CCO"))
except Exception as exc:
    print(f"inf child FAIL: {type(exc).__name__}: {exc}")

# ---- 3. GA population when seeds > population_size -----------------------
print("\n== GA: seeds exceed population_size ==")
seeds = ["CCO", "CCN", "CCS", "CCF", "CCCl", "CCBr", "CCI", "CC=O",
         "CCC", "CCCO", "CCCCO", "CCCCCO"]
result = GeneticOptimizer(
    QedObjective(), population_size=8, generations=2, seed=1
).optimize(seed_smiles=seeds, corpus=["CCO", "CCN"])
print("history n_unique per gen:", [h["n_unique"] for h in result.history])
print("population_size was 8; first generation n_unique =", result.history[0]["n_unique"])
print("metadata seed_molecules:", len(result.metadata["seed_molecules"]))

# ---- 4. GA objective that RETURNS NaN directly (not via MultiObjective) --
class NanObjective:
    name = "nan_obj"

    def score(self, smiles):
        return float("nan")


result = GeneticOptimizer(NanObjective(), population_size=8, generations=2,
                          seed=1).optimize(corpus=["CCO", "CCN", "CCOCC"])
print("\nGA with NaN objective: best_score =", result.best_score,
      "all scores:", sorted(set(result.scores))[:5])

# ---- 5. GA objective that RAISES ------------------------------------------
class RaisingObjective:
    name = "boom"

    def score(self, smiles):
        raise RuntimeError("boom")


try:
    GeneticOptimizer(RaisingObjective(), population_size=8, generations=2,
                     seed=1).optimize(corpus=["CCO", "CCN", "CCOCC"])
    print("\nGA with raising objective: NO ERROR (silently swallowed?)")
except Exception as exc:
    print(f"\nGA with raising objective: {type(exc).__name__}: {exc}")

# ---- 6. RL with a NaN objective (clamping check) --------------------------
print("\n== RL with NaN objective (3 iterations, tiny) ==")
try:
    rl = ReinventOptimizer(NanObjective(), iterations=2, batch_size=4,
                           seed=3, prior_epochs=1).optimize()
    print("RL NaN objective: completed,", len(rl), "molecules; scores:",
          sorted(set(rl.scores))[:3])
except Exception as exc:
    print(f"RL NaN objective FAIL: {type(exc).__name__}: {exc}")
print("probe 2 complete")
