"""Task 100x-21-a probe: pin down the TRUE trimethoprim ring pattern.

The 3,4,5-trimethoxybenzyl motif (anchor CH2 at C1, OMe at C3/C4/C5 -- two
meta + one para) is unambiguous: build it directly and let RDKit canonicalise.
Then compare the two candidate encodings and the corpus draft entry.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

PYRIMIDINE = "Cc2cnc(N)c(N)n2"  # 2,4-diaminopyrimidine-5-ylmethyl fragment

candidates = {
    # anchor carbon first, OMe on ring positions 3,4,5 (consecutive triple)
    "P1 anchor-first (3,4,5)": f"c1({PYRIMIDINE})cc(OC)c(OC)c(OC)c1",
    # draft candidate: OMe first -- where do they really sit?
    "P2 OMe-first (draft)": f"COc1cc({PYRIMIDINE})c(OC)c(OC)c1",
}

for label, smi in candidates.items():
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, smi
    can = Chem.MolToSmiles(mol)
    # SMARTS: benzene with three OMe on three ADJACENT carbons
    consecutive = Chem.MolFromSmarts("c1c(OC)c(OC)c(OC)cc1")
    has_consecutive = mol.HasSubstructMatch(consecutive)
    # SMARTS: anchor carbon para to an OMe and meta to the other two:
    # CH2 carbon whose ring position 4 (para) carries OMe
    para = Chem.MolFromSmarts("[CH2]c1cc(OC)c(OC)c(OC)c1")
    has_para_anchor = mol.HasSubstructMatch(para)
    print(
        f"{label:26s} formula={rdMolDescriptors.CalcMolFormula(mol):11s} "
        f"MW={Descriptors.MolWt(mol):7.2f} consecutive-OMe={has_consecutive} "
        f"CH2-para-OMe={has_para_anchor}"
    )
    print(f"{'':26s} canonical={can!r}")

print()
print("same molecule:", end=" ")
a = Chem.MolToSmiles(Chem.MolFromSmiles(candidates["P1 anchor-first (3,4,5)"]))
b = Chem.MolToSmiles(Chem.MolFromSmiles(candidates["P2 OMe-first (draft)"]))
print(a == b)
