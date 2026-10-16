"""100x-21-b probe 4: GenerationResult length mismatches + seed-value edges.

Suggested probe #6 (mismatched smiles/scores) + own hunts: huge/negative
seeds through GA and RL (torch.manual_seed range), corpus non-string
entries, DEFAULT_CORPUS mutability.
"""
from __future__ import annotations

import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.generative import (
    DEFAULT_CORPUS,
    GenerationResult,
    GeneticOptimizer,
    QedObjective,
    ReinventOptimizer,
)

print("== GenerationResult length mismatches ==")
cases = {
    "smiles>scores": (["CCO", "CCN", "CCC"], [0.5]),
    "scores>smiles": (["CCO"], [0.5, 0.9, 0.1]),
    "scores empty": (["CCO"], []),
    "breakdown short": (["CCO", "CCN"], [0.5, 0.6], [{"qed": 0.5}]),
    "breakdown long": (["CCO"], [0.5], [{"qed": 0.5}, {"qed": 0.6}]),
}
for label, args in cases.items():
    try:
        result = GenerationResult(*args[:2], breakdown=args[2] if len(args) > 2 else None)
        frame = result.to_dataframe()
        print(f"OK   {label}: len={len(result)} frame={frame.shape} "
              f"best={result.best!r}")
    except Exception as exc:
        print(f"RAISE {label}: {type(exc).__name__}: {exc}")

# empty smiles with scores
try:
    result = GenerationResult(smiles=[], scores=[0.5])
    print(f"OK   smiles-empty-scores: best={result.best!r} "
          f"frame={result.to_dataframe().shape}")
except Exception as exc:
    print(f"RAISE smiles-empty-scores: {type(exc).__name__}: {exc}")

print("\n== seed value edges ==")
for seed in (0, -5, 2**31, 2**63 - 1, 2**64, 10**25):
    try:
        result = GeneticOptimizer(QedObjective(), population_size=8,
                                  generations=2, seed=seed).optimize(
            corpus=["CCO", "CCN", "CCOCC"])
        print(f"OK   GA seed={seed}: best={result.best_score:.3f}")
    except Exception as exc:
        print(f"RAISE GA seed={seed}: {type(exc).__name__}: {exc}")

for seed in (0, -5, 2**31, 2**63 - 1, 2**64):
    try:
        result = ReinventOptimizer(QedObjective(), iterations=1, batch_size=4,
                                   seed=seed, prior_epochs=1).optimize()
        print(f"OK   RL seed={seed}: {len(result)} molecules")
    except Exception as exc:
        print(f"RAISE RL seed={seed}: {type(exc).__name__}: {exc}")

print("\n== SmilesRnn seed range ==")
from pharmaml.generative import SmilesRnn

for seed in (0, -5, 2**63 - 1, 2**64, 10**25):
    try:
        prior = SmilesRnn(seed=seed)
        prior.train_prior(["CCO", "CCN", "CCOCC"], epochs=1)
        print(f"OK   RNN seed={seed}: trained")
    except Exception as exc:
        print(f"RAISE RNN seed={seed}: {type(exc).__name__}: {exc}")

print("\n== corpus with non-string entries ==")
for corpus in ([None, 123], ["CCO", None], [b"CCO"]):
    try:
        result = GeneticOptimizer(QedObjective(), population_size=8,
                                  generations=1, seed=1).optimize(corpus=corpus)
        print(f"OK   corpus={corpus!r}: {len(result)} molecules")
    except Exception as exc:
        print(f"RAISE corpus={corpus!r}: {type(exc).__name__}: {exc}")

print("\n== DEFAULT_CORPUS mutability ==")
print("type:", type(DEFAULT_CORPUS).__name__)
before = list(DEFAULT_CORPUS)
try:
    DEFAULT_CORPUS.append("JUNKNOTSMILES")  # type: ignore[union-attr]
    print("MUTATED: append succeeded; len now", len(DEFAULT_CORPUS))
    # a later de-novo GA run now sees the junk entry
    result = GeneticOptimizer(QedObjective(), population_size=8,
                              generations=1, seed=1).optimize()
    junk_used = any("JUNK" in s for s in result.smiles)
    print("junk entry present in a fresh optimize() run:", junk_used)
finally:
    DEFAULT_CORPUS.pop()
    assert list(DEFAULT_CORPUS) == before
print("restored; len", len(DEFAULT_CORPUS))
print("probe 4 complete")
