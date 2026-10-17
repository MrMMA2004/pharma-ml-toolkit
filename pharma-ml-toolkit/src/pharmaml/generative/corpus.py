"""Bundled seed corpus for de novo molecule generation.

``DEFAULT_CORPUS`` is the training set of the generative subpackage: a
 curated, deduplicated tuple of canonical, RDKit-parseable, *stereo-free*
SMILES covering the chemical space a medicinal chemist would call
"small drug-like molecule".  It is embedded as a plain Python constant
(an immutable ``tuple`` -- a caller cannot accidentally mutate the
default corpus for the whole process) on purpose -- the package ships no
data files, so the corpus is always importable, needs no packaging
configuration and cannot drift from the code that tests it.

Design rationale
----------------
* **~60 instantly recognisable drugs** (aspirin, paracetamol, ibuprofen,
  caffeine, metformin, nicotine, warfarin, cimetidine, diclofenac,
  naproxen, propranolol, atenolol, diazepam, chloramphenicol, amoxicillin,
  steroids, ...) anchor the corpus in real pharmacology; every named entry
  was validated against its published molecular weight.
* **Amino acids and nucleobases** cover the polar / H-bonding corner and
  give the character-level RNN plenty of charged and heteroatom-rich
  vocabulary.
* **A heterocycle tour** (furan → thiophene → imidazole → oxazoles →
  thiazoles → indole → benzofuran / benzothiophene / benzoxazole →
  indazole → quinazoline / quinoxaline → purine → fused rings) samples
  the ring systems that dominate med-chem scaffolds.
* **A functional-group tour** (alcohols, ethers, esters, carboxylic
  acids, amides, amines, ketones, phenols, nitriles, halides, nitro,
  sulfonyl) spans the linkage alphabet the genetic algorithm mutates.
* **Varying sizes** (3–48 characters, 46–365 Da, LogP −3.2 … +4.7) so
  generated molecules inherit a realistic size distribution.

Every entry is the exact output of ``Chem.MolToSmiles`` for its molecule
(canonical form); ``tests/test_generative.py`` re-verifies validity,
canonicity and deduplication on every run.

Examples
--------
>>> from pharmaml.generative import DEFAULT_CORPUS
>>> len(DEFAULT_CORPUS), len(set(DEFAULT_CORPUS))   # no duplicates
(244, 244)
>>> DEFAULT_CORPUS[0]                        # aspirin anchors the corpus
'CC(=O)Oc1ccccc1C(=O)O'
>>> "CCO" in DEFAULT_CORPUS                  # ethanol is part of the vocabulary
True
>>> type(DEFAULT_CORPUS).__name__            # immutable: safe package default
'tuple'

>>> from rdkit import Chem
>>> from rdkit.Chem import Descriptors
>>> mols = [Chem.MolFromSmiles(s) for s in DEFAULT_CORPUS]
>>> all(mol is not None for mol in mols)     # every entry parses
True
>>> all(Chem.MolToSmiles(mol) == s for s, mol in zip(DEFAULT_CORPUS, mols))
True
>>> any("@" in s for s in DEFAULT_CORPUS)    # stereo-free by design
False
>>> min(len(s) for s in DEFAULT_CORPUS), max(len(s) for s in DEFAULT_CORPUS)
(3, 48)
>>> weights = [Descriptors.MolWt(mol) for mol in mols]
>>> round(min(weights), 2), round(max(weights), 2)   # the 46-365 Da envelope
(46.07, 365.41)

>>> import pandas as pd
>>> from pharmaml.utils.io import read_table, write_table
>>> frame = pd.DataFrame({"smiles": list(DEFAULT_CORPUS[:3])})
>>> out = write_table(frame, "/tmp/pharmaml_doctest_corpus.csv")
>>> read_table(out)["smiles"].tolist()       # aspirin, paracetamol, ibuprofen
['CC(=O)Oc1ccccc1C(=O)O', 'CC(=O)Nc1ccc(O)cc1', 'CC(C)Cc1ccc(C(C)C(=O)O)cc1']
>>> out.unlink()
"""

from __future__ import annotations

__all__ = ["DEFAULT_CORPUS"]

#: Canonical stereo-free SMILES of the bundled generation corpus
#: (244 molecules; immutable tuple -- see module docstring).
DEFAULT_CORPUS: tuple[str, ...] = (
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # caffeine
    "CN(C)C(=N)N=C(N)N",  # metformin
    "CN1CCCC1c1cccnc1",  # nicotine
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",  # warfarin
    "CN=C(NC#N)NCCSCc1[nH]cnc1C",  # cimetidine
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C",  # theophylline
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",  # diclofenac
    "COc1ccc2cc(C(C)C(=O)O)ccc2c1",  # naproxen (stereo-free)
    "CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1",  # ketoprofen
    "COc1ccc2c(CC(=O)O)c(C)n(C(=O)c3ccc(Cl)cc3)c2c1",
    "CC(C)NCC(O)COc1cccc2ccccc12",  # propranolol
    "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",  # atenolol
    "COCCc1ccc(OCC(O)CNC(C)C)cc1",  # metoprolol
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",  # salbutamol (stereo-free)
    "CN(C)CCOc1ccccc1-c1ccccc1",
    "Cc1ccccc1Oc1ccccc1CCN(C)C",
    "CCN(CC)CC(=O)Nc1c(C)cccc1C",  # lidocaine
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",  # procainamide
    "CC1=NCC(=O)N(C)c2ccccc21",
    "CCC1(c2ccccc2)C(=O)NC(=O)NC1=O",  # phenobarbital
    "CCOc1ccc(NC(C)=O)cc1",
    "Nc1ccc(S(N)(=O)=O)cc1",  # sulfanilamide
    "Nc1ccc(S(=O)(=O)Nc2ncccn2)cc1",  # sulfadiazine
    "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC",  # trimethoprim
    "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",  # sulfamethoxazole
    "CC(=O)Nc1nnc(S(N)(=O)=O)s1",
    "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",  # phenytoin
    "NC(=O)Nc1ccccc1-c1ccccc1",
    "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",  # fluoxetine
    "CCOC(=O)C(C)(C)Oc1ccc(Cl)cc1",  # clofibrate
    "CC(C)CCC(CN)C(=O)O",  # pregabalin (stereo-free)
    "NCC1(CC(=O)O)CCCCC1",  # gabapentin
    "NC1C2CC3CC(C2)CC1C3",  # amantadine
    "O=C(O)c1cccc(Nc2cccc(C(F)(F)F)c2)c1",
    "Cc1cccc(Nc2ccccc2C(=O)O)c1C",  # mefenamic acid
    "O=C(NC(CO)C(O)c1ccc([N+](=O)[O-])cc1)C(Cl)Cl",  # chloramphenicol
    "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",  # amoxicillin (stereo-free)
    "NC(=O)c1cccnc1",  # nicotinamide
    "NNC(=O)c1cccnc1",  # isoniazid
    "NC(=O)c1cnccn1",  # pyrazinamide
    "Cc1ncc([N+](=O)[O-])n1CCO",  # metronidazole
    "Nc1ncnc2c1ncn2COC(O)CO",  # acyclovir (stereo-free)
    "Clc1cccc(Cl)c1NC1=NCCN1",  # clonidine
    "CCOC(=O)c1ccc(N)cc1",  # benzocaine
    "CCN(CC)CCOC(=O)c1ccc(N)cc1",  # procaine
    "CCCCNC(=O)NS(=O)(=O)c1ccc(C)cc1",  # tolbutamide
    "CCCNC(=O)NS(=O)(=O)c1ccc(Cl)cc1",  # chlorpropamide
    "COc1ccc2[nH]c(CCNC(C)=O)cc2c1",  # melatonin
    "CCC1(c2ccccc2)CNC(=O)NC1=O",  # primidone
    "O=C1OC(C(O)CO)C(O)=C1O",
    "Cc1cnc(C)c(O)c1CO",
    "O=c1[nH]cnc2[nH]ncc12",  # allopurinol
    "Cn1cc[nH]c1=S",  # methimazole
    "O=S1(=O)CCCC1",
    "O=C1SCCS1",
    "O=C1NCCS1",
    "CC(C)C1CCC(C)C(O)C1",  # menthol (stereo-free)
    "C=CCc1ccc(O)c(OC)c1",
    "Cc1ccc(O)c(C(C)C)c1",  # thymol
    "CNCC(O)c1cccc(O)c1",  # phenylephrine (stereo-free)
    "CNC(C)Oc1ccc(O)cc1",  # synephrine (stereo-free)
    "CC(=O)C1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C",  # progesterone (stereo-free)
    "CC12CCC(=O)C=C1CCC1C2CCC2(C)C(O)CCC12",  # testosterone (stereo-free)
    "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O",  # estradiol (stereo-free)
    "NCC(=O)O",
    "CC(N)C(=O)O",
    "CC(C)C(N)C(=O)O",
    "CCC(C)C(N)C(=O)O",
    "O=C(O)C1CCCN1",
    "NC(Cc1ccccc1)C(=O)O",  # phenylalanine (100x-21-b: was homophenylalanine)
    "NC(Cc1ccc(O)cc1)C(=O)O",
    "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "NC(CO)C(=O)O",
    "CC(O)C(N)C(=O)O",
    "NC(CS)C(=O)O",
    "CSCCC(N)C(=O)O",  # methionine (100x-21-b: was S-methylcysteine)
    "NC(CC(=O)O)C(=O)O",
    "NC(CCC(=O)O)C(=O)O",
    "NC(=O)CC(N)C(=O)O",
    "NC(=O)CCC(N)C(=O)O",
    "NCCCCC(N)C(=O)O",  # lysine (100x-21-b: was ornithine)
    "N=C(N)NCCCC(N)C(=O)O",  # arginine (100x-21-b: was one-CH2-short)
    "NC(Cc1c[nH]cn1)C(=O)O",
    "Nc1ncnc2[nH]cnc12",
    "Nc1c[nH]c(=O)cn1",
    "Cc1c[nH]c(=O)[nH]c1=O",
    "O=c1cc[nH]c(=O)[nH]1",
    "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",
    "Nc1nc2[nH]cnc2c(=O)[nH]1",  # guanine (100x-21-b: was a non-nucleobase C6H5N5O)
    "c1ccncc1",
    "c1cncnc1",
    "c1cnccn1",
    "c1cc[nH]c1",
    "c1ccoc1",
    "c1ccsc1",
    "c1c[nH]cn1",
    "c1cocn1",
    "c1cnoc1",
    "c1cscn1",
    "c1cnsc1",
    "c1cn[nH]c1",
    "c1ccc2[nH]ccc2c1",
    "c1ccc2occc2c1",
    "c1ccc2sccc2c1",
    "c1ccc2[nH]ncc2c1",
    "c1ccc2[nH]cnc2c1",
    "c1ccc2ncccc2c1",
    "c1ccc2cnccc2c1",
    "c1ccc2ncncc2c1",
    "c1ccc2nccnc2c1",
    "c1ccc2ocnc2c1",
    "c1ccc2ccccc2c1",
    "c1ccc2c(c1)ccc1ccccc12",
    "c1ncc2nc[nH]c2n1",
    "C1COCCN1",
    "C1CNCCN1",
    "C1CCNCC1",
    "C1CCNC1",
    "C1CCOC1",
    "C1COCCO1",
    "C1CSCCO1",
    "C1CCSNC1",
    "CC1CCNCC1",
    "OC1CCCCC1",
    "Nc1ncnc2ccccc12",
    "Cc1ccncc1",
    "Cc1cccnc1",
    "Oc1ccccn1",
    "O=C(O)c1cccnc1",
    "O=C(O)c1ccccn1",
    "Oc1cccnc1",
    "Cc1cnc(N)s1",
    "CCO",
    "CCCO",
    "CCCCO",
    "CC(C)O",
    "OCCO",
    "OCC(O)CO",
    "CCOCC",
    "CCOC(C)=O",
    "CCCOC(=O)CC",
    "CCOC(=O)c1ccccc1",
    "COC(=O)c1ccccc1O",
    "O=C(O)CCc1ccccc1",
    "O=C(O)c1ccccc1",
    "O=C(O)c1ccccc1C(=O)O",
    "O=C(O)C(=O)O",
    "CCC(=O)O",
    "CCCC(=O)O",
    "CC(O)C(=O)O",
    "CC(=O)C(=O)O",
    "O=C(O)CCC(=O)O",
    "O=C(O)C=CC(=O)O",
    "O=C(O)C(O)CO",
    "O=C(O)C(O)C(O)CO",
    "O=C(O)CC(O)(CC(=O)O)C(=O)O",
    "O=C(O)C=Cc1ccccc1",
    "CC(N)=O",
    "NC(N)=O",
    "NC(=O)c1ccccc1",
    "NC(=O)c1ccccc1O",
    "CC(=O)Nc1ccccc1",
    "Nc1ccccc1",
    "Cc1ccccc1N",
    "NCc1ccccc1",
    "NCCc1ccccc1",
    "CCN(CC)CC",
    "NCCc1c[nH]cn1",
    "NCCc1ccc(O)c(O)c1",
    "NCCc1c[nH]c2ccc(O)cc12",
    "NCCc1ccc(O)cc1",
    "NCCc1c[nH]c2ccccc12",
    "NCCCC(=O)O",
    "CC(C)=O",
    "O=C1CCCCC1",
    "CC(=O)c1ccccc1",
    "O=C(c1ccccc1)c1ccccc1",
    "COc1cc(C=O)ccc1O",
    "O=Cc1ccccc1",
    "Oc1ccccc1",
    "Oc1ccccc1O",
    "Oc1cccc(O)c1",
    "Oc1ccc(O)cc1",
    "Oc1ccc(O)c(O)c1",
    "Cc1ccc(O)cc1",
    "OCc1ccccc1",
    "OCCc1ccccc1",
    "COc1ccccc1",
    "Clc1ccccc1",
    "Brc1ccccc1",
    "Fc1ccccc1",
    "Ic1ccccc1",
    "O=[N+]([O-])c1ccccc1",
    "CS(C)=O",
    "CSc1ccccc1",
    "O=C(O)C(F)(F)F",
    "O=C(O)C(Cl)Cl",
    "O=C(O)c1ccc(Cl)cc1",
    "Nc1ccc(C(=O)O)cc1",
    "O=C(O)c1ccc(O)cc1",
    "CCOC(=O)CC",
    "CCOC(=O)Cc1ccc(OC)cc1",
    "C[N+](C)(C)CC(=O)[O-]",
    "C[N+](C)(C)CCO",
    "O=C1NS(=O)(=O)c2ccccc21",  # saccharin
    "O=C1CCC(=O)N1",  # succinimide
    "CC1=CC(=O)c2ccccc2C1=O",  # menadione
    "O=P(O)(O)OCC(O)COP(=O)(O)O",
    "NCCOc1ccc(C(=O)O)cc1",
    "CC(=O)Nc1ccc(F)cc1",
    "O=C(NCCc1ccccc1)c1ccccc1",
    "O=C(NCc1ccccc1)c1ccccc1O",
    "CN(C)C(=O)c1ccccc1",
    "CN(C)CCO",
    "CN(C)CO",
    "O=C(O)C1=CC=C2C=CCCCCCN1CCO2",
    "O=C(O)CCCN1CCCCC1",
    "NC(=O)c1ccc(Cl)cc1",
    "NC(=O)c1ccc(Br)cc1",
    "O=C(Nc1ccccc1)c1ccccc1",
    "NC(CC(=O)O)c1ccccc1",
    "CC(N)Cc1ccccc1",
    "CC(O)CN",
    "NCCO",
    "CCNCO",
    "NCC(O)CO",
    "CNCC(O)c1ccc(O)c(O)c1",
    "CNCCc1ccc(O)c(O)c1",
    "O=C(O)C=Cc1c[nH]cn1",
    "OCC1OC(O)C(O)C(O)C1O",
    "OCC1OC(O)CC1O",
    "OCC1OC(O)C(O)C(O)C1CO",
    "O=C(O)Cc1c[nH]c2ccccc12",
    "OCCc1c[nH]c2ccccc12",
    "O=Cc1c[nH]c2ccccc12",
    "Cc1ccc(O)c(C)c1",
    "CC(C)(CO)C(O)C(=O)NCCC(=O)O",  # pantothenic acid
    "CCc1ccccc1OCc1cccnc1N(CC)CC",
    "COc1ccc(CCN)cc1O",
    "CC(C)n1cnc2c1c(=O)[nH]c(=O)n2C",
    "CCSc1nnc(SC)s1",
)
