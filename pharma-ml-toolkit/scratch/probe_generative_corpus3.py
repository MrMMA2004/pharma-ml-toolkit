#!/usr/bin/env python
"""Probe v3: final corpus assembly — fixes + additions, full validation."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed

EXTRA_CHECKS: list[tuple[str, str, float]] = [
    # (label, smiles, expected molecular weight)
    ("salbutamol", "CC(C)(N)Cc1ccc(O)c(CO)c1O", 239.3),
    ("chloramphenicol", "O=C(NC(CO)C(O)c1ccc([N+](=O)[O-])cc1)C(Cl)Cl", 323.1),
    ("ciprofloxacin", "C1CC1N2C=C(C(=O)C(=N2)C(=O)O)C3=C1C=C(C=C3F)N4CCNCC4", 331.3),
    ("quinoxaline", "c1ccc2nccnc2c1", 130.1),
    ("cinnoline", "c1ccc2nncc2c1", 130.1),
    ("benzoxazole", "c1ccc2ocnc2c1", 119.1),
    ("melatonin", "COc1ccc2[nH]c(CCNC(C)=O)cc2c1", 232.3),
    ("clonidine", "Clc1cccc(Cl)c1NC1=NCCN1", 229.1),
    ("phenylephrine", "CNCC(O)c1cccc(O)c1", 167.2),
    ("synephrine", "CNC(C)Oc1ccc(O)cc1", 167.2),
]


def main() -> int:
    for label, smi, want in EXTRA_CHECKS:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"FAIL  {label:16s} {smi}")
            continue
        got = Descriptors.MolWt(mol)
        ok = abs(got - want) < 0.6
        print(
            f"{'OK ' if ok else 'MW?'}  {label:16s} mw={got:7.1f} (want {want}) "
            f"qed={qed(mol):.3f}  {Chem.MolToSmiles(mol)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
