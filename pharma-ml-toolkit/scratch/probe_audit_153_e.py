"""Probe E: GA bioisostere mode -- operator stats bookkeeping + determinism.

Round-36 audit (100x-153). Deterministic (seed everything), no network.
"""
from __future__ import annotations

import random

from pharmaml.generative import GeneticOptimizer, bioisostere_moves, bioisostere_mutate
from pharmaml.generative.scoring import QedObjective

# ---------------------------------------------------------------- E1: adaptive stats
opt = GeneticOptimizer(
    QedObjective(),
    population_size=12,
    generations=4,
    seed=11,
    mutation="bioisostere",
    operator_selection="adaptive",
    elitism=2,
)
result = opt.optimize(seed_smiles=["CCO", "CCN", "CCS", "CC(=O)O"])
stats = result.metadata["operator_stats"]
mut = stats["mutation"]
print("E1 mutation operators:", sorted(mut))
total_draws = sum(entry["draws"] for entry in mut.values())
# every non-elite offspring of every generation but the last spends one draw
expected = (12 - 2) * (4 - 1)
print("E1 draws total:", total_draws, "expected:", expected, "match:", total_draws == expected)
print("E1 bioisostere draws/successes:", mut["bioisostere"]["draws"], mut["bioisostere"]["successes"])
print("E1 character draws/successes:", mut["character"]["draws"], mut["character"]["successes"])
print("E1 level1 draws total:", sum(e["draws"] for e in stats["level1"].values()))
print("E1 geometry draws total:", sum(e["draws"] for e in stats["geometry"].values()))
# successes can only count up to draws
ok = all(e["successes"] <= e["draws"] for e in mut.values())
print("E1 successes <= draws:", ok)
# best-offspring bookkeeping for a successful operator
bio = mut["bioisostere"]
print("E1 bioisostere best_offspring is None or str:",
      bio["best_offspring"] is None or isinstance(bio["best_offspring"], str))

# ---------------------------------------------------------------- E2: trajectory record 0 = fair prior
traj = result.metadata["operator_trajectory"]
print("E2 trajectory length == generations:", len(traj) == 4)
print("E2 record 0 mutation mix:", traj[0]["mutation"])
k = len(traj[0]["mutation"])
floor = 1.0 / (k * 5)
print("E2 record 0 sums to 1:", abs(sum(traj[0]["mutation"].values()) - 1.0) < 1e-9)
print("E2 record 0 is fair prior (0.5/0.5):", traj[0]["mutation"] == {"bioisostere": 0.5, "character": 0.5})

# ---------------------------------------------------------------- E3: adaptive determinism
r1 = GeneticOptimizer(
    QedObjective(), population_size=10, generations=3, seed=5,
    mutation="bioisostere", operator_selection="adaptive",
).optimize(seed_smiles=["CCO", "CCN", "CCS"])
r2 = GeneticOptimizer(
    QedObjective(), population_size=10, generations=3, seed=5,
    mutation="bioisostere", operator_selection="adaptive",
).optimize(seed_smiles=["CCO", "CCN", "CCS"])
print("E3 adaptive deterministic:", r1.smiles == r2.smiles)
s1 = r1.metadata["operator_stats"]["mutation"]
s2 = r2.metadata["operator_stats"]["mutation"]
print("E3 stats deterministic:", s1 == s2)

# ---------------------------------------------------------------- E4: uniform-mode determinism
u1 = GeneticOptimizer(
    QedObjective(), population_size=10, generations=3, seed=5, mutation="bioisostere",
).optimize(seed_smiles=["CCO", "CCN", "CCS"])
u2 = GeneticOptimizer(
    QedObjective(), population_size=10, generations=3, seed=5, mutation="bioisostere",
).optimize(seed_smiles=["CCO", "CCN", "CCS"])
print("E4 uniform deterministic:", u1.smiles == u2.smiles)

# ---------------------------------------------------------------- E5: fallback when no viable move
# every benzene mutant is 7 chars; cap below that -> None -> character fallback
print("E5 mutate with tiny cap:", bioisostere_mutate("c1ccccc1", random.Random(0), max_smiles_length=6))
print("E5 moves of benzene:", bioisostere_moves("c1ccccc1"))
print("E5 unparseable:", bioisostere_mutate("not-a-smiles", random.Random(0)))
try:
    bioisostere_mutate("CCO", random.Random(0), max_smiles_length=0)
    print("E5 FAIL zero cap accepted")
except Exception as exc:  # noqa: BLE001
    print("E5 zero cap typed:", type(exc).__name__, str(exc)[:60])

# ---------------------------------------------------------------- E6: uniform-mode run where
# every bioisostere draw fails -> character fallback still mints offspring
# (acyclic methane-ish seeds have no moves? probe what happens with "C")
print("E6 moves of methane:", bioisostere_moves("C"))
print("E6 moves of ethane:", bioisostere_moves("CC"))

# ---------------------------------------------------------------- E7: the adaptive null-reward path
# build a run where the primary fails often: seeds with no heteroatom
# exchanges possible... benzene's only move is pyridine (viable), so use
# short caps instead: max_smiles_length small in the optimizer is risky
# (corpus filter), so just verify the push bookkeeping invariant directly.
book_ok = True
from pharmaml.generative.genetic import _OperatorBook  # noqa: E402

book = _OperatorBook(("bioisostere", "character"), window=8, temperature=0.3)
import random as _r

rng = _r.Random(3)
for _ in range(20):
    op = book.draw(rng)
    book.push(op, rng.random() - 0.5, rng.random() < 0.5, None, None)
snap = book.snapshot()
print("E7 book draws sum:", sum(e["draws"] for e in snap.values()) == 20)
print("E7 snapshot keys:", sorted(snap["bioisostere"]))

print("PROBE_E_DONE")
