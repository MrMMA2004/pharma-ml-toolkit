"""100x-21-b probe 8b: finalize the true structures (2nd iteration).

Naproxen/trimethoprim/methimazole candidates corrected; verify formula +
MW + canonical spelling for every replacement.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

CANDIDATES = {
    "cimetidine": ("CN=C(NC#N)NCCSCc1c(C)nc[nH]1", 252.346, "C10H16N6S"),
    "naproxen": ("COc1ccc2cc(C(C)C(=O)O)ccc2c1", 230.262, "C14H14O3"),
    "salbutamol": ("CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", 239.311, "C13H21NO3"),
    "lidocaine": ("CCN(CC)CC(=O)Nc1c(C)cccc1C", 234.337, "C14H22N2O"),
    "procainamide": ("CCN(CC)CCNC(=O)c1ccc(N)cc1", 235.325, "C13H21N3O"),
    "trimethoprim": ("c1(Cc2cnc(N)nc2N)cc(OC)c(OC)c(OC)c1", 290.321, "C14H18N4O3"),
    "clofibrate": ("CCOC(=O)C(C)(C)Oc1ccc(Cl)cc1", 242.699, "C12H15ClO3"),
    "pregabalin": ("CC(C)CCC(CN)C(=O)O", 159.226, "C8H17NO2"),
    "gabapentin": ("NCC1(CC(=O)O)CCCCC1", 171.237, "C9H17NO2"),
    "amantadine": ("NC1C2CC3CC1CC(C2)C3", 151.248, "C10H17N"),
    "mefenamic acid": ("O=C(O)c1ccccc1Nc1cccc(C)c1C", 241.289, "C15H15NO2"),
    "chlorpropamide": ("CCCNC(=O)NS(=O)(=O)c1ccc(Cl)cc1", 276.745, "C10H13ClN2O3S"),
    "methimazole": ("CN1C=CNC1=S", 114.169, "C4H6N2S"),
    "methimazole-alt": ("Cn1cc[nH]c1=S", 114.169, "C4H6N2S"),
    "menthol": ("CC1CCC(C(C)C)CC1O", 156.265, "C10H20O"),
    "thymol": ("Cc1ccc(O)c(C(C)C)c1", 150.221, "C10H14O"),
}

for name, (candidate, published, formula) in CANDIDATES.items():
    mol = Chem.MolFromSmiles(candidate)
    if mol is None:
        print(f"UNPARSEABLE {name}: {candidate}")
        continue
    canonical = Chem.MolToSmiles(mol)
    mw = Descriptors.MolWt(mol)
    actual_formula = rdMolDescriptors.CalcMolFormula(mol)
    delta = mw - published
    flag = "OK " if abs(delta) <= 0.02 and actual_formula == formula else "BAD"
    print(f"{flag} {name:18s} canonical={canonical:45s} MW={mw:8.3f} "
          f"({delta:+.3f}) {actual_formula} (want {formula})")
print("probe 8b complete")
