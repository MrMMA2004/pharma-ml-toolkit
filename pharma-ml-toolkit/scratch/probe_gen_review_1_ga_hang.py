"""100x-21-b probe 1: GA hang paths, timeout-bounded (suggested probe #1).

Runs each GA fill scenario inside THIS process with a wall-clock watchdog
(thread + time limit) so an infinite loop is reported, not fatal.
"""
from __future__ import annotations

import sys
import threading
import time

from pharmaml.generative import GeneticOptimizer, QedObjective

LIMIT_S = 20.0


def _watchdog(label: str) -> threading.Timer:
    def _fire() -> None:
        sys.stderr.write(f"WATCHDOG: {label} exceeded {LIMIT_S}s -- HANG\n")
        sys.stderr.flush()
        # give the parent the evidence then abort the process
        import os

        os._exit(124)
    timer = threading.Timer(LIMIT_S, _fire)
    timer.daemon = True
    timer.start()
    return timer


def run(label: str, optimize_kwargs: dict | None = None, **kwargs):
    timer = _watchdog(label)
    t0 = time.perf_counter()
    try:
        result = GeneticOptimizer(QedObjective(), **kwargs).optimize(
            **(optimize_kwargs or {})
        )
        dt = time.perf_counter() - t0
        print(f"OK   {label}: {dt:.2f}s, {len(result)} molecules, "
              f"best={result.best_score:.3f}")
    except Exception as exc:
        dt = time.perf_counter() - t0
        print(f"RAISE {label}: {dt:.2f}s, {type(exc).__name__}: {exc}")
    finally:
        timer.cancel()


# (a) tiny corpus, default population 64 (the fixed HIGH hang)
run("tiny-corpus default pop", optimize_kwargs={"corpus": ["CCO", "CCN", "CCOCC"]},
    generations=3, seed=1)

# (b) tiny corpus, population far above unique pool (duplicate padding path)
run("tiny-corpus pop=200", optimize_kwargs={"corpus": ["CCO", "CCN"]},
    population_size=200, generations=2, seed=1)

# (c) duplicate spellings of one molecule, huge population
run("dup-spellings pop=256",
    optimize_kwargs={"corpus": ["OCC", "CCO", "C(C)O", "O(C)C"] * 50},
    population_size=256, generations=2, seed=1)

# (d) single-molecule corpus
run("single-molecule corpus", optimize_kwargs={"corpus": ["CCO"]},
    population_size=64, generations=2, seed=1)

# (e) seeded mode with more seeds than population_size
run("seeds>population",
    optimize_kwargs={"seed_smiles": ["CCO", "CCN", "CCS", "CCF", "CCCl",
                                      "CCBr", "CCI", "CC=O", "CCC", "CCCO",
                                      "CCCCO", "CCCCCO"],
                     "corpus": ["CCO", "CCN"]},
    population_size=8, generations=3, seed=1)

# (f) seeded mode with ONE seed and a 1-molecule corpus, pop 64
run("1-seed+1-corpus pop=64",
    optimize_kwargs={"seed_smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
                     "corpus": ["CCO"]},
    population_size=64, generations=3, seed=1)

# (g) pathological objective: everything scores 0 -> selection degenerates,
# unique-fill of next_population must still terminate
class ZeroObjective:
    name = "zero"
    def score(self, smiles):
        return 0.0

timer = _watchdog("all-zero objective")
t0 = time.perf_counter()
try:
    result = GeneticOptimizer(ZeroObjective(), population_size=48,
                              generations=5, seed=1).optimize()
    print(f"OK   all-zero objective: {time.perf_counter()-t0:.2f}s, "
          f"{len(result)} molecules, best={result.best_score}")
except Exception as exc:
    print(f"RAISE all-zero objective: {type(exc).__name__}: {exc}")
finally:
    timer.cancel()

# (h) hostile: max_smiles_length at the floor (20) with the long default corpus
run("maxlen=20 default corpus", generations=3, seed=1, max_smiles_length=20)

print("probe 1 complete")
