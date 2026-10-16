"""100x-21-b own-hunt probe A: non-finite hyperparameters + GA length caps.

Candidate defects from the line-by-line read:
1. SmilesRnn.sample temperature=NaN passes ``temperature <= 0`` guard
   (NaN compares False) -> what escapes? raw torch RuntimeError?
2. SmilesRnn.train_prior lr<=0 / NaN / inf unvalidated (RL validates
   lr<=0 only via ``if lr <= 0`` which NaN also slips).
3. ReinventOptimizer negative master seed (in torch range) end-to-end.
4. GA result must never contain a molecule longer than max_smiles_length.
5. SmilesRnn grad_clip <= 0 in train_prior.
"""
from __future__ import annotations

import math

from pharmaml.exceptions import DataValidationError
from pharmaml.generative import (
    DEFAULT_CORPUS,
    GeneticOptimizer,
    QedObjective,
    ReinventOptimizer,
    SmilesRnn,
)


def show(label: str, fn) -> None:
    try:
        value = fn()
        print(f"OK   {label}: {value}")
    except DataValidationError as exc:
        print(f"RAISE[{type(exc).__name__}] {label}: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"RAW!! {label}: {type(exc).__name__}: {str(exc)[:140]}")


# ---- 1. non-finite temperatures on a trained prior ----------------------
prior = SmilesRnn(hidden_dim=64, seed=42)
prior.train_prior(DEFAULT_CORPUS[:80], epochs=2)

show("sample temperature=nan", lambda: prior.sample(4, temperature=float("nan"), seed=1))
show("sample temperature=inf", lambda: prior.sample(4, temperature=float("inf"), seed=1))
show("sample temperature='hot' (str)", lambda: prior.sample(4, temperature="hot", seed=1))

# ---- 2. train_prior lr / grad_clip edges --------------------------------
show("train_prior lr=0", lambda: prior.train_prior(["CCO", "CCN"], epochs=1, lr=0.0))
show("train_prior lr=-1", lambda: prior.train_prior(["CCO", "CCN"], epochs=1, lr=-1.0))
show("train_prior lr=nan", lambda: prior.train_prior(["CCO", "CCN"], epochs=1, lr=float("nan")))
show("train_prior lr=inf", lambda: prior.train_prior(["CCO", "CCN"], epochs=1, lr=float("inf")))
show("train_prior grad_clip=0", lambda: prior.train_prior(["CCO"], epochs=1, grad_clip=0.0))
show("train_prior grad_clip=-1", lambda: prior.train_prior(["CCO"], epochs=1, grad_clip=-1.0))

# ---- 3. RL hyperparameter NaN slips + negative seed end-to-end ----------
show("RL lr=nan (constructed)", lambda: ReinventOptimizer(QedObjective(), lr=float("nan")))
show(
    "RL lr=inf runs 1 iter",
    lambda: len(
        ReinventOptimizer(
            QedObjective(), corpus=["CCO", "CCN", "CCS"], iterations=1,
            batch_size=8, prior_epochs=1, lr=float("inf"), seed=1,
        ).optimize().history
    ),
)
show(
    "RL temperature=nan (constructed)",
    lambda: ReinventOptimizer(QedObjective(), temperature=float("nan")),
)
show(
    "RL anchor_weight=nan (constructed)",
    lambda: ReinventOptimizer(QedObjective(), anchor_weight=float("nan")),
)
show(
    "RL entropy_bonus=inf (constructed)",
    lambda: ReinventOptimizer(QedObjective(), entropy_bonus=float("inf")),
)
show(
    "RL seed=-5 end-to-end",
    lambda: ReinventOptimizer(
        QedObjective(), prior=prior, iterations=2, batch_size=8, seed=-5
    ).optimize().best_score,
)

# ---- 4. GA length cap invariant ------------------------------------------
result = GeneticOptimizer(
    QedObjective(), population_size=24, generations=6, seed=11, max_smiles_length=40
).optimize()
too_long = [s for s in result.smiles if len(s) > 40]
print(f"GA max_smiles_length=40: {len(result)} molecules, over-cap: {too_long}")

# ---- 5. GA mutation_rate as bool / crossover_rate bool -------------------
show("GA mutation_rate=True", lambda: GeneticOptimizer(QedObjective(), mutation_rate=True))
show("GA seed='abc' (str)", lambda: GeneticOptimizer(QedObjective(), seed="abc"))
print("probe A complete")
