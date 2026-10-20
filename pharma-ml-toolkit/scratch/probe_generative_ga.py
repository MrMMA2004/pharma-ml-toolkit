#!/usr/bin/env python
"""Probe: GA mutation/crossover viability on the corpus (validity rates)."""

from __future__ import annotations

import random

from rdkit import Chem

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "probe_corpus", "scratch/probe_corpus_list.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CORPUS = _mod.DEFAULT_CORPUS

ALPHABET = sorted({c for s in CORPUS for c in s})
rng = random.Random(0)


def mutate(smi: str, rate: float) -> str:
    chars = list(smi)
    n_mut = max(1, int(len(chars) * rate)) if rng.random() < 0.9 else 0
    for _ in range(n_mut):
        if not chars:
            break
        pos = rng.randrange(len(chars))
        op = rng.random()
        if op < 0.45:  # substitute
            chars[pos] = rng.choice(ALPHABET)
        elif op < 0.8:  # insert
            chars.insert(pos, rng.choice(ALPHABET))
        else:  # delete
            del chars[pos]
    return "".join(chars)


def crossover(a: str, b: str) -> str:
    if len(a) < 2 or len(b) < 2:
        return a
    i = rng.randrange(1, len(a))
    j = rng.randrange(1, len(b))
    child = a[:i] + b[j:]
    return child if len(child) <= 100 else a


def valid(smi: str) -> bool:
    return Chem.MolFromSmiles(smi) is not None


def main() -> int:
    n = 300
    mut_ok = sum(valid(mutate(rng.choice(CORPUS), 0.08)) for _ in range(n))
    print(f"single mutation valid rate: {mut_ok}/{n} = {mut_ok / n:.2f}")
    mut_ok = sum(valid(mutate(rng.choice(CORPUS), 0.05)) for _ in range(n))
    print(f"rate 0.05: {mut_ok / n:.2f}")
    x_ok = sum(
        valid(crossover(rng.choice(CORPUS), rng.choice(CORPUS))) for _ in range(n)
    )
    print(f"crossover valid rate: {x_ok / n:.2f}")

    # full mini-GA sanity: QED objective, 16 generations, pop 48
    from rdkit.Chem.QED import qed

    def score(smi):
        mol = Chem.MolFromSmiles(smi)
        return qed(mol) if mol else 0.0

    rng2 = random.Random(42)
    pop = [rng2.choice(CORPUS) for _ in range(48)]
    best_hist = []
    for gen in range(16):
        scored = sorted(((score(s), s) for s in pop), key=lambda t: -t[0])
        best_hist.append(scored[0][0])
        survivors = [s for _, s in scored[:4]]  # elitism
        while len(survivors) < 48:
            # tournament k=3
            cand = [rng2.choice(pop) for _ in range(3)]
            parent_a = max(cand, key=score)
            cand = [rng2.choice(pop) for _ in range(3)]
            parent_b = max(cand, key=score)
            child = crossover(parent_a, parent_b) if rng2.random() < 0.7 else parent_a
            child = mutate(child, 0.08)
            mol = Chem.MolFromSmiles(child)
            if mol is not None and len(child) <= 100:
                child = Chem.MolToSmiles(mol)
                survivors.append(child)
        pop = survivors
    print("best-per-gen:", [f"{b:.3f}" for b in best_hist])
    print("final best:", max(pop, key=score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
