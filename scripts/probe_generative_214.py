"""Probe: print the real numbers the Generative Workbench tests will assert.

Run: from the toolkit venv, ``python scripts/probe_generative_214.py`` (repo
root; the TK constant below is parents-based, so any cwd works).

The page's shared corpus is the bundled 12-molecule seed table
(examples/data/seed_molecules.csv) or the built-in 244-molecule drug-like
corpus.  The probe exercises every engine the page wires --
GeneticOptimizer ( QedObjective + the MultiObjective triple over the seed
table, seeded mode), ReinventOptimizer (RL, the CLI idiom: seeds anchored
ahead of the bundle), MctsOptimizer (ChemTSv2-style over prefixes),
pareto_front (the MultiObjective run's trade-off set), ring_topology_moves
(piperidine's 3 mutants), scaffold_hop_crossover (aspirin x the ibuprofen
core, seed 42) and the GuacaMol oracle book on aspirin.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

TK = Path(__file__).resolve().parents[1] / "pharma-ml-toolkit"
import csv
import sys

sys.path.insert(0, str(TK / "src"))

import numpy as np


def seed_molecules() -> list[str]:
    with (TK / "examples" / "data" / "seed_molecules.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row["smiles"].strip() for row in rows if row.get("smiles", "").strip()]


def build_objective(kind: str):
    from pharmaml.generative import (
        LogPBandObjective,
        MolecularWeightBandObjective,
        MultiObjective,
        QedObjective,
        TpsaBandObjective,
    )

    if kind == "qed":
        return QedObjective()
    if kind == "logp":
        return LogPBandObjective(1.0, 3.0)
    if kind == "multi":
        return MultiObjective(
            [
                QedObjective(),
                MolecularWeightBandObjective(150, 450),
                TpsaBandObjective(30, 90),
            ]
        )
    raise ValueError(kind)


def probe_ga() -> None:
    from pharmaml.generative import GeneticOptimizer

    print("\n=== GeneticOptimizer (QED, built-in corpus, pop 24 x 16 generations) ===")
    started = time.perf_counter()
    optimizer = GeneticOptimizer(
        build_objective("qed"),
        population_size=24,
        generations=16,
        mutation="topology",
        crossover="splice",
        operator_selection="uniform",
        seed=42,
    )
    result = optimizer.optimize()
    seconds = time.perf_counter() - started
    print(f"best={result.best!r} best_score={result.best_score:.6f}")
    print(f"unique={len(result.smiles)} elapsed_s={result.metadata.get('elapsed_s'):.1f} probe_s={seconds:.1f}")
    print(f"metadata keys: {sorted(result.metadata)}")
    frame = result.to_dataframe()
    print(f"dataframe shape: {frame.shape} columns: {list(frame.columns)}")
    print("top5:")
    print(frame.head(5).to_string(index=False))
    print(f"history records: {len(result.history)}")
    if result.history:
        print(f"history[0] keys: {sorted(result.history[0])}")
        print(f"history[0]: {result.history[0]}")
        print(f"history[-1]: {result.history[-1]}")
    best_scores = [rec.get("best_score") for rec in result.history]
    mean_scores = [rec.get("mean_score") for rec in result.history]
    print(f"history best[:3]={[round(v, 6) for v in best_scores[:3]]}")
    print(f"history best[-1]={best_scores[-1]:.6f} mean[-1]={mean_scores[-1]:.6f}")
    valid = [rec.get("valid_offspring_fraction") for rec in result.history]
    print(f"history valid[:3]={[round(v, 4) for v in valid[:3]]} valid[-1]={valid[-1]:.4f}")


def probe_ga_multi() -> None:
    from pharmaml.generative import GeneticOptimizer, pareto_front

    print("\n=== GeneticOptimizer (MultiObjective QED x MW x TPSA, seeded 12) ===")
    optimizer = GeneticOptimizer(
        build_objective("multi"),
        population_size=24,
        generations=16,
        mutation="topology",
        crossover="splice",
        operator_selection="uniform",
        seed=42,
    )
    seeds = seed_molecules()
    result = optimizer.optimize(seed_smiles=seeds)
    print(f"best={result.best!r} best_score={result.best_score:.6f} unique={len(result.smiles)}")
    frame = result.to_dataframe()
    print(f"dataframe columns: {list(frame.columns)}")
    print("top5:")
    print(frame.head(5).to_string(index=False))
    front = pareto_front(result)
    print(f"\nfront str: {front!s}")
    print(f"front objectives: {front.objectives}")
    print(f"front size: {len(front.smiles)}")
    front_frame = front.to_dataframe()
    print(f"front frame columns: {list(front_frame.columns)} shape: {front_frame.shape}")
    print("front rows:")
    print(front_frame.to_string(index=False))
    deepest = max(result.smiles.index(s) for s in front.smiles) + 1
    print(f"deepest front member rank: {deepest} of {len(result)}")
    # the trade-off scatter's axes: the first two objectives
    x = [row[front.objectives[0]] for row in result.breakdown]
    print(f"first-objective range over run: {min(x):.4f}..{max(x):.4f}")
    return result


def probe_rl() -> None:
    from pharmaml.generative import DEFAULT_CORPUS, ReinventOptimizer

    print("\n=== ReinventOptimizer (RL, QED, seeds anchored, 12 x 16, prior 8) ===")
    seeds = seed_molecules()
    corpus = list(dict.fromkeys([*seeds, *DEFAULT_CORPUS]))
    started = time.perf_counter()
    optimizer = ReinventOptimizer(
        build_objective("qed"),
        corpus=corpus,
        iterations=12,
        batch_size=16,
        prior_epochs=8,
        seed=42,
    )
    result = optimizer.optimize()
    seconds = time.perf_counter() - started
    print(f"best={result.best!r} best_score={result.best_score:.6f}")
    print(f"unique={len(result.smiles)} probe_s={seconds:.1f}")
    print(f"metadata: { {k: v for k, v in result.metadata.items() if k != 'elapsed_s'} }")
    print(f"history records: {len(result.history)}")
    if result.history:
        print(f"history[0] keys: {sorted(result.history[0])}")
        valid = [rec.get("valid_fraction", rec.get("valid_offspring_fraction")) for rec in result.history]
        keys = sorted(result.history[0])
        print(f"RL history keys: {keys}")
        print(f"RL history[0]: { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.history[0].items()} }")


def probe_mcts() -> None:
    from pharmaml.generative import MctsOptimizer

    print("\n=== MctsOptimizer (QED, 20 iterations, seed 42) ===")
    started = time.perf_counter()
    optimizer = MctsOptimizer(
        build_objective("qed"),
        iterations=20,
        seed=42,
    )
    result = optimizer.optimize()
    seconds = time.perf_counter() - started
    print(f"best={result.best!r} best_score={result.best_score:.6f}")
    print(f"unique={len(result.smiles)} probe_s={seconds:.1f}")
    print(f"metadata: { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.metadata.items() if k != 'elapsed_s'} }")


def probe_topology() -> None:
    from pharmaml.generative import ring_topology_moves

    print("\n=== ring_topology_moves('C1CCNCC1') ===")
    mutants = ring_topology_moves("C1CCNCC1")
    print(f"count={len(mutants)} mutants={mutants}")
    aromatic = ring_topology_moves("c1ccccc1")
    print(f"benzene (aromatic, never touched): count={len(aromatic)}")
    acyclic = ring_topology_moves("CCO")
    print(f"ethanol (acyclic): count={len(acyclic)}")


def probe_breed() -> None:
    from pharmaml.generative import scaffold_hop_crossover

    print("\n=== scaffold_hop_crossover (aspirin x ibuprofen core, seed 42, 4 children) ===")
    parent_a = "CC(=O)Oc1ccccc1C(=O)O"
    parent_b = "CC(C)Cc1ccc(O)cc1"
    rng = random.Random(42)
    litter = [scaffold_hop_crossover(parent_a, parent_b, rng) for _ in range(4)]
    print(f"litter: {litter}")
    viable = [child for child in litter if child]
    print(f"viable={len(viable)}/4")


def probe_oracles() -> None:
    from pharmaml.chem.oracles import (
        penalized_logp,
        property_filter,
        qed_score,
        rediscovery_score,
        sa_score,
    )

    print("\n=== the GuacaMol oracle book on aspirin (rediscovery vs salicylic acid) ===")
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    reference = "O=C(O)c1ccccc1O"
    qed = qed_score(smiles)
    sa = sa_score(smiles)
    logp = penalized_logp(smiles)
    redis = rediscovery_score(smiles, reference)
    props = property_filter(smiles)
    print(f"qed={qed:.6f} sa={sa:.6f} logp={logp:.6f} rediscovery={redis:.6f}")
    print(f"property filter ok={props['ok']}")
    print(f"property values: {props['values']}")
    print(f"violations: {props['violations']}")
    # the invalid-SMILES path
    print(f"qed('not-a-smiles')={qed_score('not-a-smiles')}")


def main() -> int:
    print(f"seed molecules: {len(seed_molecules())} -> {seed_molecules()[:4]} ...")
    probe_ga()
    multi_result = probe_ga_multi()
    probe_rl()
    probe_mcts()
    probe_topology()
    probe_breed()
    probe_oracles()
    print("\nprobe complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
