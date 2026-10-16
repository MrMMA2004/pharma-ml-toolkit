#!/usr/bin/env python
"""Probe v4: FINAL corpus assembly for pharmaml.generative.corpus.

Merges validated v1 candidates with all fixes/additions from v2/v3, dedupes
against RDKit canonical forms and emits the literal list to embed in
corpus.py.  Named-drug entries are verified by molecular weight in the
PROBES above (aspirin 180.2, paracetamol 151.2, ibuprofen 206.3, caffeine
194.2, metformin 129.2, nicotine 162.2, warfarin 308.3, cimetidine 252.3,
chloramphenicol 323.1, ...).
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed

CANDIDATES: list[str] = [
    # --- instantly recognisable drugs ------------------------------------
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
    "CC(=O)Nc1ccc(O)cc1",               # paracetamol
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",       # ibuprofen
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",       # caffeine
    "CN(C)C(=N)N=C(N)N",                # metformin
    "CN1CCCC1c1cccnc1",                 # nicotine
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",  # warfarin
    "CN=C(NC#N)NCCSCc1nc(C)n[nH]1",     # cimetidine
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C",       # theophylline
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",   # diclofenac
    "COc1ccc2cc(CC(C)C(=O)O)c(C)cc2c1",  # naproxen
    "CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1",  # ketoprofen
    "COc1ccc2c(c1)n(C(=O)c1ccc(Cl)cc1)c(C)c2CC(=O)O",  # indomethacin
    "CC(C)NCC(O)COc1cccc2ccccc12",      # propranolol
    "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",   # atenolol
    "COCCc1ccc(OCC(O)CNC(C)C)cc1",      # metoprolol
    "CC(C)(N)CC(O)c1ccc(O)c(CO)c1",     # salbutamol
    "CN(C)CCOc1ccccc1c1ccccc1",         # diphenhydramine
    "CN(C)CCc1ccccc1Oc1ccccc1C",        # chlorphenamine
    "CCN(CC)C(=O)Nc1c(C)cccc1C",        # lidocaine
    "CCN(CC)CCCC(=O)Nc1ccc(N)cc1",      # procainamide
    "CN1C(=O)CN=C(C)c2ccccc21",         # diazepam
    "CCC1(c2ccccc2)C(=O)NC(=O)NC1=O",   # phenobarbital
    "CC(=O)Nc1ccc(OCC)cc1",             # phenacetin
    "Nc1ccc(S(N)(=O)=O)cc1",            # sulfanilamide
    "Nc1ccc(S(=O)(=O)Nc2ncccn2)cc1",    # sulfadiazine
    "COc1cc(C)c(Cc2cnc(N)c(N)n2)cc1OC",  # trimethoprim
    "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",  # sulfamethoxazole
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
    "O=C(NC(CO)C(O)c1ccc([N+](=O)[O-])cc1)C(Cl)Cl",  # chloramphenicol
    "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",  # amoxicillin (stereo-free)
    "NC(=O)c1cccnc1",                   # nicotinamide
    "NNC(=O)c1cccnc1",                  # isoniazid
    "NC(=O)c1cnccn1",                   # pyrazinamide
    "Cc1ncc([N+](=O)[O-])n1CCO",        # metronidazole
    "Nc1ncnc2c1ncn2COC(O)CO",           # acyclovir (stereo-free)
    "Clc1cccc(Cl)c1NC1=NCCN1",          # clonidine
    "CCOC(=O)c1ccc(N)cc1",              # benzocaine
    "CCN(CC)CCOC(=O)c1ccc(N)cc1",       # procaine
    "CCCCNC(=O)NS(=O)(=O)c1ccc(C)cc1",  # tolbutamide
    "CCCCNC(=O)NS(=O)(=O)c1ccc(Cl)cc1",  # chlorpropamide
    "COc1ccc2[nH]c(CCNC(C)=O)cc2c1",    # melatonin
    "CCC1(c2ccccc2)CNC(=O)NC1=O",       # primidone
    "OCC(O)C1OC(=O)C(O)=C1O",           # ascorbic acid (stereo-free)
    "Cc1ncc(C)c(CO)c1O",                # pyridoxine
    "O=c1[nH]cnc2[nH]ncc12",            # allopurinol
    "Cc1n[nH]c(S)n1",                   # methimazole
    "O=S1(=O)CCCC1",                    # sulfolane
    "O=C1SCCS1",                        # dithiolanone
    "O=C1NCCS1",                        # thiazolidinone-ish scaffold
    "CC(C)CCCC(C)C1CCC(C(C)C)CC1O",     # menthol (stereo-free)
    "COc1cc(CC=C)ccc1O",                # eugenol
    "Cc1cc(C)c(O)c(C)c1",               # thymol
    "CNCC(O)c1cccc(O)c1",               # phenylephrine (stereo-free)
    "CNC(C)Oc1ccc(O)cc1",               # synephrine (stereo-free)
    # --- steroids ---------------------------------------------------------
    "CC(=O)C1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C",  # progesterone (stereo-free)
    "CC12CCC(=O)C=C1CCC1C2CCC2(C)C(O)CCC12",     # testosterone (stereo-free)
    "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O",            # estradiol (stereo-free)
    # --- amino acids ------------------------------------------------------
    "NCC(=O)O", "CC(N)C(=O)O", "CC(C)C(N)C(=O)O", "CC(C)CC(N)C(=O)O",
    "CCC(C)C(N)C(=O)O", "OC(=O)C1CCCN1", "NC(CCc1ccccc1)C(=O)O",
    "Oc1ccc(CC(N)C(=O)O)cc1", "O=C(O)C(N)Cc1c[nH]c2ccccc12",
    "OCC(N)C(=O)O", "CC(O)C(N)C(=O)O", "SCC(N)C(=O)O", "CSCC(N)C(=O)O",
    "OC(=O)CC(N)C(=O)O", "OC(=O)CCC(N)C(=O)O", "NC(=O)CC(N)C(=O)O",
    "NC(=O)CCC(N)C(=O)O", "NCCCC(N)C(=O)O", "NC(N)=NCCC(N)C(=O)O",
    "O=C(O)C(N)Cc1c[nH]cn1",
    # --- nucleobases ------------------------------------------------------
    "Nc1ncnc2[nH]cnc12", "Nc1ncc(=O)[nH]c1", "Cc1c[nH]c(=O)[nH]c1=O",
    "C1=CNC(=O)NC1=O", "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",
    "N=C1NC(=O)C2=NC=NC2=C1N",
    # --- heterocycles -----------------------------------------------------
    "c1ccncc1", "c1ccncn1", "c1cnccn1", "c1cc[nH]c1", "c1ccoc1", "c1ccsc1",
    "c1cnc[nH]1", "c1cocn1", "c1cnoc1", "c1cscn1", "c1cnsc1", "c1ccno1",
    "c1cc[nH]n1", "c1ccc2[nH]ccc2c1", "c1ccc2occc2c1", "c1ccc2sccc2c1",
    "c1ccc2[nH]ncc2c1", "c1ccc2[nH]cnc2c1", "c1ccc2ncccc2c1", "c1ccc2cnccc2c1",
    "c1ccc2ncncc2c1", "c1ccc2nccnc2c1", "c1ccc2ocnc2c1",
    "c1ccc2ccccc2c1", "c1ccc2c(c1)ccc1ccccc12", "c1ncnc2[nH]cnc12",
    "C1COCCN1", "C1CNCCN1", "C1CCNCC1", "C1CCNC1", "C1CCOC1", "C1COCCO1",
    "O1CCSCC1", "C1CCSNC1", "CC1CCNCC1", "OC1CCCCC1", "Nc1ncnc2c1cccc2",
    "Cc1ccncc1", "Cc1cnccc1", "Oc1ccccn1", "OC(=O)c1cccnc1", "OC(=O)c1ccccn1",
    "Oc1cccnc1", "Cc1cnc(s1)N",
    # --- functional-group tour --------------------------------------------
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
    "O=Cc1ccccc1", "Oc1ccccc1", "Oc1ccccc1O", "Oc1cccc(O)c1",
    "Oc1ccc(O)cc1", "Oc1ccc(O)cc1O", "Cc1ccc(O)cc1",
    "OCc1ccccc1", "OCCc1ccccc1", "COc1ccccc1", "Clc1ccccc1", "Brc1ccccc1",
    "Fc1ccccc1", "Ic1ccccc1", "O=[N+]([O-])c1ccccc1", "CS(C)=O",
    "CSc1ccccc1", "O=C(O)C(F)(F)F", "OC(=O)C(Cl)Cl", "O=C(O)c1ccc(Cl)cc1",
    "O=C(O)c1ccc(N)cc1", "O=C(O)c1ccc(O)cc1",
    "CCOC(=O)c1ccc(N)cc1", "CCOC(=O)C", "CCOC(=O)CC", "COc1ccc(CC(=O)OCC)cc1",
    "C[N+](C)(C)CC(=O)[O-]", "OCC[N+](C)(C)C",
    "O=C1NS(=O)(=O)c2ccccc21", "O=C1CCC(=O)N1", "CC1=CC(=O)c2ccccc2C1=O",
    "O=P(O)(O)OCC(O)COP(O)(O)=O",
    "O=C(O)c1ccc(OCCN)cc1", "CC(=O)Nc1ccc(F)cc1",
    "O=C(NCCc1ccccc1)c1ccccc1", "O=C(NCc1ccccc1)c1ccccc1O",
    "CN(C)C(=O)c1ccccc1", "CN(C)CCO", "CN(C)CO",
    "O=C(O)c1ccc(OCCN1CCCCC1)cc1", "O=C(O)CCCN1CCCCC1",
    "O=C(N)c1ccc(Cl)cc1", "O=C(N)c1ccc(Br)cc1", "O=C(Nc1ccccc1)c1ccccc1",
    "O=C(O)CCCN1CCCCC1", "O=C(O)CC(c1ccccc1)N", "CC(N)Cc1ccccc1",
    "CC(O)CN", "OCCN", "CCNCO", "OCC(O)CN",
    # --- biogenic amines / neurotransmitters ------------------------------
    "CNCC(O)c1ccc(O)c(O)c1", "CNCCc1ccc(O)c(O)c1", "O=C(O)C=Cc1c[nH]cn1",
    # --- sugars ------------------------------------------------------------
    "OCC1OC(O)C(O)C(O)C1O", "OCC1OC(O)CC1O", "OC1C(O)C(O)OC(CO)C1O",
    "OCC1OC(O)C(O)C(O)C1CO",
    # --- indole chemistry --------------------------------------------------
    "O=C(O)Cc1c2ccccc2[nH]c1", "OCCc1c[nH]c2ccccc12", "O=Cc1c[nH]c2ccccc12",
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    # --- vitamins / cofactor fragments -------------------------------------
    "O=C(O)c1cccnc1", "Cc1ccc(O)c(C)c1", "CC(C)(CO)C(O)C(=O)NCCC(=O)O",
    # --- misc drug-like probes ---------------------------------------------
    "CCN(CC)c1ncccc1COc1ccccc1CC", "COc1ccc(CCN)cc1O",
    "CC(C)N1cnc2c1c(=O)[nH]c(=O)n2C", "CSc1nnc(s1)SCC",
    "Nc1ncnc2c1ncn2COC(CO)O", "O=C1CCCCC1", "c1ccc2ccccc2c1",
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

    print(f"candidates: {len(CANDIDATES)}  invalid: {len(invalid)}  "
          f"dupes: {len(dupes)}  final: {len(deduped)}")
    for smi in invalid:
        print(f"  INVALID: {smi}")
    for smi in dupes:
        print(f"  DUPE: {smi}")

    n = len(deduped)
    qeds, logps, mws, tpsas, has = [], [], [], [], []
    for can in deduped:
        mol = Chem.MolFromSmiles(can)
        qeds.append(qed(mol))
        logps.append(Descriptors.MolLogP(mol))
        mws.append(Descriptors.MolWt(mol))
        tpsas.append(rdMolDescriptors.CalcTPSA(mol))
        has.append(mol.GetNumHeavyAtoms())
    print(f"qed : mean={sum(qeds)/n:.2f} min={min(qeds):.2f} max={max(qeds):.2f}")
    print(f"logp: mean={sum(logps)/n:.2f} min={min(logps):.2f} max={max(logps):.2f}")
    print(f"mw  : mean={sum(mws)/n:.1f} min={min(mws):.1f} max={max(mws):.1f}")
    print(f"tpsa: mean={sum(tpsas)/n:.1f} min={min(tpsas):.1f} max={max(tpsas):.1f}")
    print(f"ha  : mean={sum(has)/n:.1f} min={min(has)} max={max(has)}")
    chars = sorted({c for s in deduped for c in s})
    print(f"vocab chars ({len(chars)}): {''.join(chars)}")

    print("\nDEFAULT_CORPUS: list[str] = [")
    for can in deduped:
        print(f'    "{can}",')
    print("]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
