#!/usr/bin/env python
"""Probe v2: fix the 14 invalid candidates + add ~25 more; re-validate."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed

# Only the fixed/replacement/new candidates (validated ones carry over from v1).
FIXES_AND_NEW: list[tuple[str, str]] = [
    ("cimetidine A", "CN=C(NCCSCc1nc(C)n[nH]1)NC#N"),
    ("cimetidine B", "N#CNC(=NCCSCc1nc(C)n[nH]1)NC"),
    ("trimethoprim A", "COc1cc(C)c(Cc2nc(N)c(N)nc2)cc1OC"),
    ("trimethoprim B", "COc1cc(C)c(Cc2cnc(N)nc2N)cc1OC"),
    ("sulfamethoxazole A", "Cc1cc(on1)... DUMMY"),
    ("sulfamethoxazole B", "Cc1cc(c(no1)C)NS(=O)(=O)c2ccc(N)cc2"),
    ("sulfamethoxazole C", "CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N"),
    ("metronidazole A", "Cc1ncc([N+](=O)[O-])n1CCO"),
    ("metronidazole B", "CC1=CC([N+](=O)[O-])=NN1CCO"),
    ("acyclovir A", "Nc1ncnc2n(COC(CO)O)cnc12"),
    ("acyclovir B", "Nc1ncnc2c1ncn2COC(CO)O"),
    ("ciprofloxacin A", "O=C(O)c1cn(C2CC2)c2cc(F)c(N)c(=O)c12"),
    ("ciprofloxacin B", "O=C(O)c1cn(C2CC2)c2cc(F)c(N3CCNCC3)c(=O)c12"),
    ("progesterone A", "CC(=O)C1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C"),
    ("progesterone B", "CC(=O)CC1(C)CCC2C3CCC4(C)C3CCC4(C)C12"),
    ("phenylalanine", "NC(CCc1ccccc1)C(=O)O"),
    ("testosterone", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"),
    ("estradiol", "CC12CCC3C(CCc4cc(O)ccc34)C1CCC2O"),
    ("estradiol B", "CC12CCC3c4ccc(O)c4CCC3C1CCC2O"),
    ("oxazole", "O1C=NC=C1"),
    ("isoxazole", "O1N=CC=C1"),
    ("isothiazole", "S1N=CC=C1"),
    ("benzofuran", "o1ccc2ccccc12"),
    ("benzothiophene", "s1ccc2ccccc12"),
    ("indazole", "n1nccc2ccccc12"),
    ("quinazoline", "n1cnc2ccccc2c1"),
    ("quinoxaline", "n1cncc2ccccc12"),
    ("cinnoline", "n1cccc2nccnc12"),
    ("phenanthrene", "c1ccc2c(c1)ccc1ccccc12"),
    ("imidazopyridine", "c1cc2ccnc2[nH]1"),
    ("benzocaine", "CCOC(=O)c1ccc(N)cc1"),
    ("procaine", "CCN(CC)CCOC(=O)c1ccc(N)cc1"),
    ("tolbutamide", "CCCCNC(=O)NS(=O)(=O)c1ccc(C)cc1"),
    ("chlorpropamide", "CCCCNC(=O)NS(=O)(=O)c1ccc(Cl)cc1"),
    ("melatonin", "COc1ccc2[nH]c(CCNC(C)=O)c2c1"),
    ("indole-3-acetic acid", "O=C(O)Cc1c2ccccc2[nH]c1"),
    ("tryptophol", "OCCc1c[nH]c2ccccc12"),
    ("indole-3-carboxaldehyde", "O=Cc1c[nH]c2ccccc12"),
    ("saccharin", "O=C1NS(=O)(=O)c2ccccc12"),
    ("succinimide", "O=C1CCC(=O)N1"),
    ("primidone", "CCC1(c2ccccc2)C(=O)NC(=O)NC1"),
    ("chloramphenicol B", "O=C(NC(O)C(O)c1ccc([N+](=O)[O-])cc1)C(Cl)Cl"),
    ("iodobenzene", "Ic1ccccc1"),
    ("betaine", "C[N+](C)(C)CC(=O)[O-]"),
    ("pantothenic acid", "CC(C)(CO)C(O)C(=O)NCCC(=O)O"),
    ("urocanic acid", "O=C(O)C=Cc1c[nH]cn1"),
    ("salicylamide", "O=C(N)c1ccccc1O"),
    ("salicylic acid", "OC(=O)c1ccccc1O"),
    ("choline", "OCC[N+](C)(C)C"),
    ("phenethylamine", "NCCc1ccccc1"),
    ("histamine", "NCCc1c[nH]cn1"),
]


def main() -> int:
    for label, smi in FIXES_AND_NEW:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"FAIL  {label:26s} {smi}")
            continue
        can = Chem.MolToSmiles(mol)
        print(
            f"OK    {label:26s} {can}  "
            f"qed={qed(mol):.3f} mw={Descriptors.MolWt(mol):.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
