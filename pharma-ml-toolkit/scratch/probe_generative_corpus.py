#!/usr/bin/env python
"""Probe: curate + validate the DEFAULT_CORPUS candidate list for pharmaml.generative.

Every candidate is parsed, canonicalized and deduped with RDKit; failures are
reported so they can be fixed/dropped before the list is embedded in corpus.py.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed

CANDIDATES: list[str] = [
    # --- instantly recognizable drugs -----------------------------------
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
    "CC(=O)Nc1ccc(O)cc1",               # paracetamol
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",       # ibuprofen
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",       # caffeine
    "CN(C)C(=N)N=C(N)N",                # metformin
    "CN1CCCC1c1cccnc1",                 # nicotine
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",  # warfarin
    "CN=C(NCCSc1nc(C)n[nH]1)N#N",       # cimetidine
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C",       # theophylline
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",   # diclofenac
    "COc1ccc2cc(CC(C)C(=O)O)c(C)cc2c1",  # naproxen
    "CC(C(=O)O)c1cccc(c1)C(=O)c2ccccc2",  # ketoprofen
    "COc1ccc2c(c1)n(C(=O)c1ccc(Cl)cc1)c(C)c2CC(=O)O",  # indomethacin
    "CC(C)NCC(O)COc1cccc2ccccc12",      # propranolol
    "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",   # atenolol
    "COCCc1ccc(OCC(O)CNC(C)C)cc1",      # metoprolol
    "CC(C)(N)Cc1ccc(O)c(CO)c1O",        # salbutamol
    "CN(C)CCOc1ccccc1c1ccccc1",         # diphenhydramine
    "CN(C)CCc1ccccc1Oc1ccccc1C",        # chlorphenamine
    "CCN(CC)C(=O)Nc1c(C)cccc1C",        # lidocaine
    "CCN(CC)CCCC(=O)Nc1ccc(N)cc1",      # procainamide
    "CN1C(=O)CN=C(C)c2ccccc21",         # diazepam
    "CCC1(c2ccccc2)C(=O)NC(=O)NC1=O",   # phenobarbital
    "CC(=O)Nc1ccc(OCC)cc1",             # phenacetin
    "Nc1ccc(S(N)(=O)=O)cc1",            # sulfanilamide
    "Nc1ccc(S(=O)(=O)Nc2ncccn2)cc1",    # sulfadiazine
    "COc1cc(C)c(Cc2cnc(N)nc2N)cc(OC)c1OC",  # trimethoprim
    "COc1ccc(S(=O)(=O)Nc2cc(on2)C)c(N)cc1",  # sulfamethoxazole
    "CC(=O)Nc1nnc(s1)S(N)(=O)=O",       # acetazolamide
    "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",  # phenytoin
    "NC(=O)Nc1ccccc1c1ccccc1",          # carbamazepine
    "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",  # fluoxetine
    "CCOC(C)(C)Oc1ccc(Cl)cc1",          # clofibrate
    "CC(C)CC(N)C(=O)O",                 # pregabalin
    "NCC1(CC(=O)O)CCCC1",               # gabapentin
    "NC12CCCCC1CCCC2",                  # amantadine
    "O=C(O)c1cccc(c1)Nc1cccc(c1)C(F)(F)F",  # flufenamic acid
    "Cc1cccc(Nc2ccccc2C(=O)O)c1",       # mefenamic acid
    "CCN(CC)c1ncccc1COc1ccccc1CC",      # ? probe (drop if odd)
    "COc1ccc(CCN)cc1O",                 # ? probe (drop if odd)
    "CC(C)N1cnc2c1c(=O)[nH]c(=O)n2C",   # ? probe
    "O=S(=O)(N)c1ccc(N)cc1",            # sulfanilic amide dup probe
    "NC(=O)c1cccnc1",                   # nicotinamide
    "NNC(=O)c1cccnc1",                  # isoniazid
    "NC(=O)c1cnccn1",                   # pyrazinamide
    "CC(C)(O)Cn1cc(c(N(=O)=O)[nH]1)CCO",  # metronidazole
    "OCC(O)C1OC(=O)C(O)=C1O",           # ascorbic acid
    "Cc1ncc(C)c(CO)c1O",                # pyridoxine
    "Cc1ncc(CO)c(C)c1O",                # pyridoxine alt form
    "O=c1[nH]cnc2[nH]ncc12",            # allopurinol
    "Cc1n[nH]c(S)n1",                   # methimazole
    "CSc1nnc(s1)SCC",                   # ? probe
    "Nc1nc(N)nc2c1nccn2COC(CO)O",       # acyclovir probe
    "O=C(O)c1cn(c2ccc(F)c(N)c(=O)c2)C2CC2",  # ciprofloxacin probe (no piperazine)
    "CC(O)C(Oc1ccc([N+](=O)[O-])cc1)C(N)C(=O)O",  # chloramphenicol probe
    "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",  # amoxicillin probe
    "N#Cc1ccccc1",                      # benzonitrile
    "CC(O)C#N",                         # lactonitrile probe
    # --- steroids -------------------------------------------------------
    "CC12CCC3C(C1)CCC4(C)C3CCC4(O)C(=O)CC",  # progesterone probe
    "CC12CCC3C(CCC(=O)C3)C1CCC2O",      # testosterone probe
    "CC12CCC3C(CCc4cc(O)ccc34)C1CCC2O",  # estradiol probe
    # --- amino acids ----------------------------------------------------
    "NCC(=O)O", "CC(N)C(=O)O", "CC(C)C(N)C(=O)O", "CC(C)CC(N)C(=O)O",
    "CCC(C)C(N)C(=O)O", "OC(=O)C1CCCN1", "Nc(CCc1ccccc1)C(=O)O",
    "Oc1ccc(CC(N)C(=O)O)cc1", "O=C(O)C(N)Cc1c[nH]c2ccccc12",
    "OCC(N)C(=O)O", "CC(O)C(N)C(=O)O", "SCC(N)C(=O)O", "CSCC(N)C(=O)O",
    "OC(=O)CC(N)C(=O)O", "OC(=O)CCC(N)C(=O)O", "NC(=O)CC(N)C(=O)O",
    "NC(=O)CCC(N)C(=O)O", "NCCCC(N)C(=O)O", "NC(N)=NCCC(N)C(=O)O",
    "O=C(O)C(N)Cc1c[nH]cn1",
    # --- nucleobases ----------------------------------------------------
    "Nc1ncnc2[nH]cnc12", "Nc1ncc(=O)[nH]c1", "Cc1c[nH]c(=O)[nH]c1=O",
    "C1=CNC(=O)NC1=O", "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",
    "N=C1NC(=O)C2=NC=NC2=C1N",
    # --- heterocycles ---------------------------------------------------
    "c1ccncc1", "c1ccncn1", "c1cnccn1", "c1cc[nH]c1", "c1ccoc1", "c1ccsc1",
    "c1cnc[nH]1", "c1ccoc1n", "c1cscn1", "c1ccno1", "c1cc[nH]n1",
    "c1ccc2[nH]ccc2c1", "c1ccc2ccco2", "c1ccc2ccsc2", "c1ccc2[nH]ncc2c1",
    "c1ccc2[nH]cnc2c1", "c1ccc2ncccc2c1", "c1ccc2cnccc2c1", "c1cnc2cccn2c1",
    "c1cnc2ccncn2c1", "c1cnc2nnccn2c1", "c1ccc2nnncc2c1", "c1ccc2ccccc2c1",
    "c1ccc2cccc2ccc1", "c1ccc2c(c1)cccc2", "c1ncnc2[nH]cnc12",
    "C1COCCN1", "C1CNCCN1", "C1CCNCC1", "C1CCNC1", "C1CCOC1", "C1COCCO1",
    "O1CCSCC1", "C1CCSNC1", "CC1CCNCC1", "OC1CCCCC1", "Nc1ncnc2c1cccc2",
    "Cc1ccncc1", "Cc1cnccc1", "Oc1ccccn1", "OC(=O)c1cccnc1", "OC(=O)c1ccccn1",
    "Oc1cccnc1", "c1ccc2ocnc2c1", "c1ccc2scnc2c1", "Cc1cnc(s1)N",
    "O=S1(=O)CCCC1", "O=C1SCCS1", "O=C1NCCS1",
    # --- functional-group tour ------------------------------------------
    "CCO", "CCCO", "CCCCO", "CC(C)O", "OCCO", "OCC(O)CO", "CCOCC",
    "CCOC(C)=O", "CCOC(=O)C", "CCCOC(=O)CC", "O=C(OCC)c1ccccc1",
    "COC(=O)c1ccccc1O", "O=C(O)CCc1ccccc1", "OC(=O)c1ccccc1",
    "O=C(O)c1ccccc1C(=O)O", "OC(=O)C(=O)O", "OC(=O)CC", "OC(=O)CCC",
    "CC(O)C(=O)O", "CC(=O)C(=O)O", "OC(=O)CCC(=O)O", "OC(=O)C=CC(=O)O",
    "OCC(O)C(=O)O", "OCC(O)C(O)C(=O)O", "OC(=O)CC(O)(CC(=O)O)C(=O)O",
    "OC(=O)C=Cc1ccccc1", "CC(N)=O", "NC(N)=O", "O=C(N)c1ccccc1",
    "O=C(N)c1ccccc1O", "CC(=O)Nc1ccccc1", "Nc1ccccc1", "Cc1ccccc1N",
    "NCc1ccccc1", "NCCc1ccccc1", "CCN(CC)CC", "NCCc1c[nH]cn1",
    "NCCc1ccc(O)c(O)c1", "NCCc1c[nH]c2ccc(O)cc12", "NCCc1ccc(O)cc1",
    "NCCc1c[nH]c2ccccc12", "NCCCC(=O)O", "CC(C)=O", "O=C1CCCCC1",
    "CC(=O)c1ccccc1", "O=C(c1ccccc1)c1ccccc1", "COc1cc(C=O)ccc1O",
    "O=Cc1ccccc1", "O=C1CCCCC1", "Oc1ccccc1", "Oc1ccccc1O",
    "Oc1cccc(O)c1", "Oc1ccc(O)cc1", "Oc1ccc(O)cc1O", "Cc1ccc(O)cc1",
    "COc1cc(CC=C)ccc1O", "Cc1cc(C)c(O)c(C)c1", "CC(C)CCCC(C)C1CCC(C(C)C)CC1O",
    "OCc1ccccc1", "OCCc1ccccc1", "COc1ccccc1", "Clc1ccccc1", "Brc1ccccc1",
    "Fc1ccccc1", "O=[N+]([O-])c1ccccc1", "CS(C)=O", "CSc1ccccc1",
    "O=C(O)C(F)(F)F", "OC(=O)C(Cl)Cl", "O=C(O)c1ccc(Cl)cc1",
    "O=C(O)c1ccc(N)cc1", "O=C(O)c1ccc(O)cc1",
    # --- esters/amides of drug-ish acids --------------------------------
    "CCOC(=O)c1ccc(N)cc1", "CCOC(=O)C", "CCOC(=O)CC",
    "COc1ccc(CC(=O)OCC)cc1",
    # --- sugars ----------------------------------------------------------
    "OCC1OC(O)C(O)C(O)C1O", "OCC1OC(O)CC1O", "OC1C(O)C(O)OC(CO)C1O",
    "OCC1OC(O)C(O)C(O)C1CO",
    # --- CNS-ish / biogenic amines --------------------------------------
    "CNCC(O)c1ccc(O)c(O)c1", "CNCCc1ccc(O)c(O)c1",
    "OCC(O)CN", "CC(O)CN", "OCCN", "CCNCO",
    "CNC(C)CC(O)c1ccc(O)cc1",
    "O=C(O)CC(c1ccccc1)N",
    "CC(N)Cc1ccccc1",
    # --- vitamins / cofactors -------------------------------------------
    "Cc1ccc(O)c(C)c1", "O=C1CCCCC1", "CC1=CC(=O)c2ccccc2C1=O",
    "O=C(O)c1cccnc1", "O=P(O)(O)OCC(O)COP(O)(O)=O",
    # --- beta-lactam-ish / misc probes ----------------------------------
    "O=C(O)c1ccc(OCCN)cc1",
    "CC(=O)Nc1ccc(F)cc1",
    "O=C(NCCc1ccccc1)c1ccccc1",
    "O=C(NCc1ccccc1)c1ccccc1O",
    "CN(C)C(=O)c1ccccc1",
    "CN(C)CCO",
    "CN(C)CO",
    "O=C(O)c1ccc(OCCN1CCCCC1)cc1",
    "O=C(O)CCCN1CCCCC1",
    "O=C(N)c1ccc(Cl)cc1",
    "O=C(N)c1ccc(Br)cc1",
    "O=C(Nc1ccccc1)c1ccccc1",
]


def main() -> int:
    invalid: list[str] = []
    canonical: list[str] = []
    for smi in CANDIDATES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(smi)
            continue
        canonical.append(Chem.MolToSmiles(mol))

    seen: set[str] = set()
    deduped: list[str] = []
    dupes: list[str] = []
    for can in canonical:
        if can in seen:
            dupes.append(can)
        else:
            seen.add(can)
            deduped.append(can)

    print(f"candidates: {len(CANDIDATES)}")
    print(f"invalid   : {len(invalid)}")
    for smi in invalid:
        print(f"  INVALID: {smi}")
    print(f"canonical : {len(canonical)}")
    print(f"dupes     : {len(dupes)}")
    for smi in dupes:
        print(f"  DUPE: {smi}")
    print(f"deduped   : {len(deduped)}")

    stats = {
        "qed": [],
        "logp": [],
        "mw": [],
        "tpsa": [],
        "rotb": [],
        "ha": [],
    }
    for can in deduped:
        mol = Chem.MolFromSmiles(can)
        stats["qed"].append(qed(mol))
        stats["logp"].append(Descriptors.MolLogP(mol))
        stats["mw"].append(Descriptors.MolWt(mol))
        stats["tpsa"].append(rdMolDescriptors.CalcTPSA(mol))
        stats["rotb"].append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        stats["ha"].append(mol.GetNumHeavyAtoms())
    n = len(deduped)
    for key, values in stats.items():
        print(
            f"{key:>5}: mean={sum(values)/n:.2f} min={min(values):.2f} "
            f"max={max(values):.2f}"
        )
    lens = [len(s) for s in deduped]
    print(f"smiles length: min={min(lens)} max={max(lens)} mean={sum(lens)/n:.1f}")

    print("\n--- canonical list for corpus.py ---")
    print("DEFAULT_CORPUS: list[str] = [")
    for can in deduped:
        print(f'    "{can}",')
    print("]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
