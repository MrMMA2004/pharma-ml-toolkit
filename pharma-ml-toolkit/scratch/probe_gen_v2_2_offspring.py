"""Probe 2: scan scaffold-mode GA offspring for leftover dummies / artifacts.

Hunts: leftover ``[*:k]`` dummies, disconnected fragments, non-canonical
output, Murcko-scaffold preservation rates across seeds and modes.
"""
from __future__ import annotations

import random

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from pharmaml.generative import DEFAULT_CORPUS, GeneticOptimizer, QedObjective
from pharmaml.generative._utils import canonical_smiles
from pharmaml.generative.genetic import _scaffold_mutate, _build_decoration_pool

RDLogger.DisableLog("rdApp.error")


def murcko(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


bad = []
n_mols = 0
# --- Part A: full GA runs, many seeds, de novo + seeded ---------------------
for seed in (0, 1, 7, 42, 123, 2024):
    for corpus in (None, ["CC(=O)Oc1ccccc1C(=O)O"]):
        result = GeneticOptimizer(
            QedObjective(), population_size=16, generations=5, seed=seed,
            mutation="scaffold",
        ).optimize(
            seed_smiles=["CC(=O)Oc1ccccc1C(=O)O"] if corpus else None,
            corpus=corpus,
        )
        for smi in result.smiles:
            n_mols += 1
            if "*" in smi:
                bad.append(("dummy", seed, smi))
            if "." in smi:
                bad.append(("fragment", seed, smi))
            if canonical_smiles(smi) != smi:
                bad.append(("noncanonical", seed, smi))
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                bad.append(("unparseable", seed, smi))
            else:
                # any property-computable weirdness
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    bad.append(("unsanitizable", seed, smi))

print(f"Part A: scanned {n_mols} unique offspring across 12 runs; issues: {len(bad)}")
for kind, seed, smi in bad[:20]:
    print(f"  {kind} seed={seed}: {smi}")

# --- Part B: raw _scaffold_mutate stress, many parents x seeds --------------
pool = _build_decoration_pool(list(DEFAULT_CORPUS))
print(f"\nPart B: pool size {len(pool)}; sample: {pool[:8]}")
parents = [
    "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "CN1CCCC1c1cccnc1", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "c1ccc([C@H](C)N)cc1", "C[N+](C)(C)c1ccccc1", "c1ccc(S(=O)(=O)N)cc1",
    "c1ccc(B(O)O)cc1", "Cn1cccc1", "OC(=O)[C@@H]1CCCN1",
    "Cc1c[nH]c(=O)[nH]c1=O", "Nc1ncnc2c1ncn2",  # adenine: bare scaffold
    "c1ccc2c(c1)ncnc2N",  # fused heteroaromatic with N decorations
]
mut_bad = []
n_mut = 0
none_count = 0
for parent in parents:
    parent_scaffold = murcko(parent)
    for seed in range(8):
        rng = random.Random(seed)
        for _ in range(60):
            mutant = _scaffold_mutate(parent, pool, rng, 120)
            if mutant is None:
                none_count += 1
                continue
            n_mut += 1
            if "*" in mutant:
                mut_bad.append(("dummy", parent, mutant))
            if "." in mutant:
                mut_bad.append(("fragment", parent, mutant))
            if murcko(mutant) != parent_scaffold:
                mut_bad.append(("scaffold-change", parent, mutant))
print(f"Part B: {n_mut} mutants, {none_count} None-returns; issues: {len(mut_bad)}")
for kind, parent, mutant in mut_bad[:20]:
    print(f"  {kind}: parent={parent} mutant={mutant}")

# --- Part C: weird corpora: charged / isotope / tiny pools ------------------
weird_corpora = {
    "charged": ["C[N+](C)(C)c1ccccc1", "c1ccc(C(=O)[O-])cc1"],
    "isotopes": ["[2H]c1ccccc1", "c1ccc([13CH3])cc1"],
    "tiny": ["CC(=O)Oc1ccccc1C(=O)O"],
    "tiny_acyclic": ["CCO", "CCN"],
    "chiral": ["N[C@@H](Cc1ccccc1)C(=O)O"],
    "bridged": ["COc1ccc2c(c1)OCOc1ccccc21"],
}
for label, corpus in weird_corpora.items():
    try:
        result = GeneticOptimizer(
            QedObjective(), population_size=12, generations=4, seed=3,
            mutation="scaffold",
        ).optimize(seed_smiles=[corpus[0]], corpus=corpus)
        issues = [s for s in result.smiles if "*" in s or "." in s]
        print(f"Part C [{label}]: {len(result.smiles)} unique, issues={len(issues)} "
              f"best={result.best[:50]!r} score={result.best_score:.3f}")
        for s in issues[:5]:
            print(f"    ISSUE: {s}")
    except Exception as exc:
        print(f"Part C [{label}]: RAISED {type(exc).__name__}: {exc}")
