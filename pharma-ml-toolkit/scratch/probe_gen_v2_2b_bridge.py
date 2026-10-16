"""Probe 2b: investigate the '-2' mutant from the bridged corpus run.

Part C of probe 2 produced best='COc1ccc2c(c1)OCOc1ccccc1-2' from the
bridged corpus ['COc1ccc2c(c1)OCOc1ccccc21'] — the ring-2 closure bond
apparently became an explicit SINGLE bond.  Is the core really preserved
verbatim?  What is this molecule?
"""
from __future__ import annotations

import random

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from pharmaml.generative.genetic import _scaffold_mutate, _scaffold_split, _build_decoration_pool

RDLogger.DisableLog("rdApp.error")

PARENT = "COc1ccc2c(c1)OCOc1ccccc21"
MUTANT = "COc1ccc2c(c1)OCOc1ccccc1-2"


def info(label: str, smi: str) -> None:
    mol = Chem.MolFromSmiles(smi)
    print(f"{label}: {smi}")
    if mol is None:
        print("    UNPARSEABLE")
        return
    print(f"    atoms={mol.GetNumAtoms()} formula={Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    print(f"    MW={Descriptors.MolWt(mol):.3f} rings={mol.GetRingInfo().NumRings()}")
    print(f"    murcko={Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))}")
    aromatic_bonds = [b.GetBondType() for b in mol.GetBonds()]
    print(f"    bond types={sorted(str(b) for b in aromatic_bonds)}")


info("parent", PARENT)
info("mutant", MUTANT)

split = _scaffold_split(PARENT)
print(f"\nparent split: {split}")
pool = _build_decoration_pool([PARENT])
print(f"pool: {pool}")

# enumerate all mutants reachable
rng = random.Random(3)
seen = {}
for _ in range(400):
    m = _scaffold_mutate(PARENT, pool, rng, 120)
    if m is None:
        continue
    seen[m] = seen.get(m, 0) + 1
print(f"\ndistinct mutants from 400 draws: {len(seen)}")
for m, count in sorted(seen.items(), key=lambda kv: -kv[1]):
    same = "SAME" if m == PARENT else "DIFF"
    print(f"  [{same}] x{count:3d}  {m}")

# check: does the mutant's Murcko equal the parent's Murcko?
pm = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(PARENT)))
for m in seen:
    mm = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(m)))
    if mm != pm:
        print(f"  SCAFFOLD CHANGED: {m} -> {mm}")
