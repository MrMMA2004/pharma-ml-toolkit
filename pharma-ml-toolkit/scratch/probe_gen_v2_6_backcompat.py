"""Probe 6: back-compat pins A/B/C from /tmp/task22a/backcompat.txt
(recorded on the PRE-22-a code) + GA integration checks for scaffold mode:
cross-process determinism, warm/cold cache determinism, monotone-best
elitism, metadata, RL smoke.
"""
from __future__ import annotations

import sys

from pharmaml.generative import (
    GeneticOptimizer,
    LogPBandObjective,
    MultiObjective,
    QedObjective,
    ReinventOptimizer,
)

# --- Pin A: de novo qed pop32 gen5 seed42 ------------------------------------
a = GeneticOptimizer(QedObjective(), population_size=32, generations=5, seed=42).optimize()
pin_a = (a.best, round(a.best_score, 6), len(a))
exp_a = ("COc1ccc2[nH]c(CCC(C)=O)cc2c1", 0.855097, 137)
print(f"pin A: {'OK' if pin_a == exp_a else 'MISMATCH'} {pin_a} vs {exp_a}")

# --- Pin B: seeded qed pop16 gen3 seed42 -------------------------------------
b = GeneticOptimizer(
    QedObjective(), population_size=16, generations=3, seed=42
).optimize(seed_smiles=["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1"])
pin_b = (b.best, round(b.best_score, 6), len(b))
exp_b = ("COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC", 0.853356, 39)
print(f"pin B: {'OK' if pin_b == exp_b else 'MISMATCH'} {pin_b} vs {exp_b}")

# --- Pin C: multi .7qed/.3logp pop24 gen4 seed42 ------------------------------
c = GeneticOptimizer(
    MultiObjective([QedObjective(), LogPBandObjective(1.0, 3.0)], weights=[0.7, 0.3]),
    population_size=24, generations=4, seed=42,
).optimize()
pin_c = (
    c.best,
    round(c.best_score, 6),
    len(c),
    [round(e["best_score"], 6) for e in c.history],
)
exp_c = (
    "CC(=O)Nc1ccbcc2[nH]cnc2ccc1",
    0.839569,
    80,
    [0.833081, 0.833081, 0.839569, 0.839569],
)
print(f"pin C: {'OK' if pin_c == exp_c else 'MISMATCH'} {pin_c} vs {exp_c}")

# --- scaffold-mode in-process determinism (warm vs cold cache) ----------------
r1 = GeneticOptimizer(
    QedObjective(), population_size=16, generations=4, seed=42, mutation="scaffold"
).optimize()
r2 = GeneticOptimizer(
    QedObjective(), population_size=16, generations=4, seed=42, mutation="scaffold"
).optimize()
print(f"scaffold warm/cold determinism: {r1.smiles == r2.smiles and r1.scores == r2.scores}")
print(f"metadata mutation: {r1.metadata['mutation']!r}")

# --- monotone best across generations in scaffold mode ------------------------
bests = [e["best_score"] for e in r1.history]
mono = all(bests[i] <= bests[i + 1] for i in range(len(bests) - 1))
print(f"scaffold monotone best: {mono} ({[round(x, 4) for x in bests]})")

# --- elitism=0 edge in scaffold mode (documented range allows it) -------------
r0 = GeneticOptimizer(
    QedObjective(), population_size=8, generations=3, seed=42, elitism=0,
    mutation="scaffold",
).optimize(corpus=["CC(=O)Oc1ccccc1C(=O)O"])
print(f"elitism=0 scaffold run: n={len(r0)} best_score={r0.best_score:.3f}")

# --- tiny-corpus scaffold runs (anti-collapse padding) ------------------------
for corpus in (["CCO"], ["CCO", "CCN"], ["CC(=O)Oc1ccccc1C(=O)O"]):
    rs = GeneticOptimizer(
        QedObjective(), population_size=8, generations=3, seed=1, mutation="scaffold"
    ).optimize(corpus=corpus)
    print(f"tiny corpus {len(corpus)}: n={len(rs)} best={rs.best!r} ({rs.best_score:.3f})")

# --- RL smoke (untouched by v2) ------------------------------------------------
if "--with-rl" in sys.argv:
    rl = ReinventOptimizer(QedObjective(), iterations=2, seed=42)
    res = rl.optimize(corpus=["CCO", "CCN", "CCOCC", "c1ccccc1"])
    print(f"RL smoke: n={len(res)} best_score={res.best_score:.3f} metadata={res.metadata['method']}")
