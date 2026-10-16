"""100x-21-b own-hunt probe B: isolate the RL negative-seed failure.

`ReinventOptimizer(prior=..., seed=-5)` died with a raw
"RuntimeError: probability tensor contains either inf, nan or element < 0".
Find the exact mechanism: which seed values, which iteration, which tensor.
"""
from __future__ import annotations

import torch

from pharmaml.generative import DEFAULT_CORPUS, QedObjective, ReinventOptimizer, SmilesRnn


def run(seed: int, iterations: int = 2, use_prior: bool = True) -> None:
    prior = SmilesRnn(hidden_dim=64, seed=42)
    prior.train_prior(DEFAULT_CORPUS[:80], epochs=2)
    try:
        if use_prior:
            result = ReinventOptimizer(
                QedObjective(), prior=prior, iterations=iterations,
                batch_size=8, seed=seed,
            ).optimize()
        else:
            result = ReinventOptimizer(
                QedObjective(), corpus=["CCO", "CCN", "CCS"], iterations=iterations,
                batch_size=8, prior_epochs=1, seed=seed,
            ).optimize()
        losses = [round(h["loss"], 4) if h["loss"] == h["loss"] else "nan"
                  for h in result.history]
        print(f"OK   seed={seed} use_prior={use_prior}: "
              f"{len(result.history)} iters, losses={losses}, best={result.best_score:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"RAW!! seed={seed} use_prior={use_prior}: "
              f"{type(exc).__name__}: {str(exc)[:110]}")


for seed in (-5, -1, -42, -1000000, 0, 5, 42):
    run(seed)

# direct generator sanity for the exact per-iteration seeds of seed=-5
print("\n-- torch.Generator.manual_seed on the exact iteration seeds of seed=-5 --")
for iteration in range(2):
    sample_seed = (-5 * 1_000_003 + iteration) % (2**31)
    gen = torch.Generator()
    try:
        gen.manual_seed(sample_seed)
        draw = torch.multinomial(
            torch.softmax(torch.randn(3, 5, generator=gen), dim=-1), 1, generator=gen
        )
        print(f"iteration {iteration}: sample_seed={sample_seed} multinomial OK {draw.flatten().tolist()}")
    except Exception as exc:  # noqa: BLE001
        print(f"iteration {iteration}: sample_seed={sample_seed} RAW {type(exc).__name__}: {exc}")

# negative master seed on a fresh SmilesRnn (the corpus-prior path)
print("\n-- SmilesRnn(seed=-5): manual_seed / Generator paths --")
rnn = SmilesRnn(seed=-5)
try:
    losses = rnn.train_prior(["CCO", "CCN", "CCS", "CCF"], epochs=1)
    print(f"train_prior OK: {losses}")
    print("sample:", rnn.sample(4, seed=1))
except Exception as exc:  # noqa: BLE001
    print(f"RAW!! {type(exc).__name__}: {str(exc)[:110]}")
print("probe B complete")
