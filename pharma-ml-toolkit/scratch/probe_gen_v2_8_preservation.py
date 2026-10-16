"""Probe 8: exhaustive Murcko-preservation check across ALL splittable
corpus parents (the "framework kept verbatim" contract), plus the
all-decorations-dropped mutant vs MurckoScaffold.GetScaffoldForMol.
"""
from __future__ import annotations

import random

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from pharmaml.generative.corpus import DEFAULT_CORPUS
from pharmaml.generative.genetic import _build_decoration_pool, _scaffold_mutate, _scaffold_split

RDLogger.DisableLog("rdApp.error")


def murcko(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


pool = _build_decoration_pool(list(DEFAULT_CORPUS))
parents = [s for s in DEFAULT_CORPUS if _scaffold_split(s) is not None]
print(f"splittable corpus parents: {len(parents)}")

violations: list[tuple[str, str]] = []
checked = 0
for parent in parents:
    parent_murcko = murcko(parent)
    rng = random.Random(12345)
    for _ in range(40):
        mutant = _scaffold_mutate(parent, pool, rng, 120)
        if mutant is None:
            continue
        checked += 1
        if murcko(mutant) != parent_murcko:
            violations.append((parent, mutant))
print(f"mutants checked: {checked}; Murcko-preservation violations: {len(violations)}")
for parent, mutant in violations[:10]:
    print(f"  parent={parent}\n  mutant={mutant}")

# --- all-dropped mutant vs MurckoScaffold ------------------------------------
# force the drop branch on every decoration by using an empty-ish pool trick:
# instead, re-implement the drop for every decoration and compare frameworks.
diffs = []
for parent in parents:
    split = _scaffold_split(parent)
    core_smi, frags = split
    Chem = Chem
    rw = Chem.RWMol(Chem.MolFromSmiles(core_smi))
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(True)
    core = rw.GetMol()
    try:
        Chem.SanitizeMol(core)
        bare = Chem.MolToSmiles(Chem.RemoveHs(core))
    except Exception:
        continue
    if bare != murcko(parent):
        diffs.append((parent, bare, murcko(parent)))
print(f"\nall-dropped vs MurckoScaffold differences: {len(diffs)}")
for parent, bare, mur in diffs[:10]:
    print(f"  parent={parent}\n    all-drop={bare}\n    murcko  ={mur}")
