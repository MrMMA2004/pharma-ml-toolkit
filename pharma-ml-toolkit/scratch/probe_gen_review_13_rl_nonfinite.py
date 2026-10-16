"""100x-21-b own-hunt probe D: non-finite RL hyperparameters end-to-end.

Mechanism check for the D3 family: which non-finite hyperparameters
silently NaN the agent (then crash later runs / produce garbage scores),
vs which are caught today.
"""
from __future__ import annotations

import math

from pharmaml.exceptions import DataValidationError
from pharmaml.generative import (
    DEFAULT_CORPUS,
    GeneticOptimizer,
    MultiObjective,
    QedObjective,
    ReinventOptimizer,
    SmilesRnn,
    SubstructureObjective,
)

PRIOR = SmilesRnn(hidden_dim=64, seed=42)
PRIOR.train_prior(DEFAULT_CORPUS[:80], epochs=2)


def run(label: str, **kwargs) -> None:
    try:
        result = ReinventOptimizer(
            QedObjective(), prior=PRIOR, iterations=2, batch_size=8, **kwargs
        ).optimize()
        losses = [
            round(h["loss"], 4) if isinstance(h["loss"], float) and math.isfinite(h["loss"])
            else h["loss"]
            for h in result.history
        ]
        finite_scores = all(math.isfinite(s) for s in result.scores)
        print(f"OK   {label}: iters={len(result.history)} losses={losses} "
              f"best={result.best_score:.3f} scores_finite={finite_scores} n={len(result)}")
    except DataValidationError as exc:
        print(f"TYPED {label}: {str(exc)[:110]}")
    except Exception as exc:  # noqa: BLE001
        print(f"RAW!! {label}: {type(exc).__name__}: {str(exc)[:110]}")


run("lr=nan", lr=float("nan"))
run("lr=inf", lr=float("inf"))
run("anchor_weight=nan", anchor_weight=float("nan"))
run("anchor_weight=inf", anchor_weight=float("inf"))
run("entropy_bonus=nan", entropy_bonus=float("nan"))
run("entropy_bonus=inf", entropy_bonus=float("inf"))
run("tol=nan", tol=float("nan"))
run("grad_clip=nan", grad_clip=float("nan"))

# ---- GA: tournament_k > population + iterator corpus ---------------------
class CountingIter:
    def __init__(self, items):
        self.items = items
        self.consumed = 0

    def __iter__(self):
        self.consumed += 1
        return iter(self.items)


corpus_iter = CountingIter(list(DEFAULT_CORPUS[:24]))
result = GeneticOptimizer(
    QedObjective(), population_size=8, generations=2, seed=1, tournament_k=50
).optimize(corpus=corpus_iter)
print(f"GA tournament_k=50 > population=8: OK {len(result)} molecules; "
      f"corpus iter consumed {corpus_iter.consumed}x")

# ---- MultiObjective: out-of-range child + huge weights (exotic pin) -----
class BigObjective:
    def __init__(self, value, name):
        self.value = value
        self.name = name

    def score(self, smiles):
        return self.value


try:
    total, _ = MultiObjective(
        [BigObjective(2.0, "big"), SubstructureObjective("c1ccccc1")], weights=[1.0, 1.0]
    ).score_with_breakdown("CCO")
    print(f"MultiObjective child=2.0 (out of contract): total={total} (clipped at 1.0)")
except Exception as exc:  # noqa: BLE001
    print(f"MultiObjective child=2.0: RAW {type(exc).__name__}: {exc}")

try:
    total, _ = MultiObjective(
        [BigObjective(0.9, "a"), BigObjective(0.8, "b")], weights=[1e300, 1e300]
    ).score_with_breakdown("CCO")
    print(f"MultiObjective huge weights: total={total}")
except Exception as exc:  # noqa: BLE001
    print(f"MultiObjective huge weights: RAW {type(exc).__name__}: {str(exc)[:90]}")
print("probe D complete")
