"""100x-21-b probe 3: RL degenerate configs + RNN UNK/temperature edges.

Suggested probes #3 (anchor_weight=0, entropy_bonus=0, replay_size=0,
patience=1, iterations=0, batch_size=1) and #4 (train_prior retrain with
NEW characters, extreme temperatures, sample(0)/sample(negative)).
"""
from __future__ import annotations

import math

from pharmaml.exceptions import DataValidationError
from pharmaml.generative import (
    DEFAULT_CORPUS,
    QedObjective,
    ReinventOptimizer,
    SmilesRnn,
    run_rl_optimization,
)

print("== RL degenerate hyperparameters ==")


def rl(label, **kwargs):
    try:
        result = ReinventOptimizer(QedObjective(), **kwargs).optimize()
        hist = result.history
        print(f"OK   {label}: {len(hist)} iterations, {len(result)} molecules, "
              f"best={result.best_score:.3f}, early={result.metadata['early_stopped']}, "
              f"loss0={hist[0]['loss']:.4f}")
    except Exception as exc:
        print(f"RAISE {label}: {type(exc).__name__}: {exc}")


rl("anchor_weight=0", iterations=3, batch_size=4, anchor_weight=0.0,
   prior_epochs=1, seed=1, patience=99)
rl("entropy_bonus=0", iterations=3, batch_size=4, entropy_bonus=0.0,
   prior_epochs=1, seed=1, patience=99)
rl("replay_size=0", iterations=3, batch_size=4, replay_size=0,
   prior_epochs=1, seed=1, patience=99)
rl("patience=1", iterations=5, batch_size=4, patience=1,
   prior_epochs=1, seed=1)
rl("iterations=1", iterations=1, batch_size=4, prior_epochs=1, seed=1)
rl("iterations=0", iterations=0)
rl("batch_size=1", iterations=2, batch_size=1)
rl("batch_size=3", iterations=2, batch_size=3, prior_epochs=1, seed=1)
rl("tol=1e9 (always improved)", iterations=3, batch_size=4, tol=1e9,
   prior_epochs=1, seed=1)
rl("all-zero scores objective", iterations=3, batch_size=4, prior_epochs=1,
   seed=1, patience=99)  # Qed already > 0; use a zero objective below


class ZeroObjective:
    name = "zero"

    def score(self, smiles):
        return 0.0


rl("zero objective (std=0 advantage)", iterations=3, batch_size=4,
   prior_epochs=1, seed=1, patience=99)

# objective intentionally NOT passed here -- reuse rl() pattern


class ZeroObj2:
    name = "zero2"

    def score(self, smiles):
        return 0.0


try:
    result = ReinventOptimizer(ZeroObj2(), iterations=3, batch_size=4,
                               prior_epochs=1, seed=1, patience=99).optimize()
    print("zero-objective direct: losses:",
          [round(h["loss"], 4) for h in result.history])
except Exception as exc:
    print(f"zero-objective direct FAIL: {type(exc).__name__}: {exc}")

print("\n== RNN: retrain with NEW characters (UNK path) ==")
prior = SmilesRnn(seed=5)
losses1 = prior.train_prior(DEFAULT_CORPUS[:40], epochs=1)
vocab1 = list(prior.vocab)
print("vocab size after first train:", len(vocab1))
# 'B' and '%' never appear in the corpus: they must map to UNK, not crash
losses2 = prior.train_prior(["CBCCB", "C%CC%"], epochs=1)
print("retrain losses:", [round(l, 4) for l in losses2])
enc = prior._encode("CB%")
unk = prior.vocab.index("<unk>")
print("encoded 'CB%' ->", enc, "UNK index:", unk)
sampled = prior.sample(8, temperature=1.0, seed=9)
print("sample after retrain:", sampled[:4])

print("\n== RNN: extreme temperatures ==")
prior2 = SmilesRnn(seed=5)
prior2.train_prior(DEFAULT_CORPUS[:60], epochs=2)
for temp in (1e-9, 1e-6, 0.01, 10.0, 100.0):
    try:
        out = prior2.sample(6, temperature=temp, seed=11)
        uniq = len(set(out))
        print(f"OK   temperature={temp}: n={len(out)} unique={uniq} "
              f"sample={out[:2]!r}")
    except Exception as exc:
        print(f"RAISE temperature={temp}: {type(exc).__name__}: {exc}")

print("\n== RNN: sample(0) / sample(negative) ==")
for n in (0, -1, -100):
    try:
        out = prior2.sample(n, seed=3)
        print(f"OK   sample({n}) -> {out!r}")
    except Exception as exc:
        print(f"RAISE sample({n}): {type(exc).__name__}: {exc}")

print("\n== RNN: max_length edges ==")
for ml in (1, 0, -5, 2001):
    try:
        out = prior2.sample(2, max_length=ml, seed=3)
        print(f"OK   max_length={ml}: maxlen={max(len(s) for s in out)}")
    except Exception as exc:
        print(f"RAISE max_length={ml}: {type(exc).__name__}: {exc}")

print("\n== RNN: bool temperature (isinstance int) ==")
try:
    out = prior2.sample(2, temperature=True, seed=3)
    print(f"OK   temperature=True: {out!r}")
except Exception as exc:
    print(f"RAISE temperature=True: {type(exc).__name__}: {exc}")

print("probe 3 complete")
