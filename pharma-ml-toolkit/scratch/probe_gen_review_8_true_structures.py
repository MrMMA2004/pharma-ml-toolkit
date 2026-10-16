"""100x-21-b probe 8: construct the TRUE structures for the 15 mislabeled
corpus drugs, canonicalize them and MW-check against published values.

Each candidate below is the stereo-free neutral form; the fix will splice
the RDKit-canonical spelling into DEFAULT_CORPUS in place.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors

# name -> (candidate true SMILES, published MW)
CANDIDATES = {
    # guanidine N-CH2CH2-S-CH2-(4-methyl-1H-imidazol-5-yl): IMIDAZOLE ring
    "cimetidine": ("CN=C(NC#N)NCCSCc1c(C)nc[nH]1", 252.346),
    # (S)-naproxen, stereo-free: 6-methoxy-2-naphthyl-CH(CH3)-COOH
    "naproxen": ("COc1ccc2cc(C(C)C(=O)O)cc2c1", 230.262),
    # salbutamol: Ar-CH(OH)-CH2-NH-C(CH3)3
    "salbutamol": ("CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", 239.311),
    # lidocaine: Ar-NH-CO-CH2-N(Et)2
    "lidocaine": ("CCN(CC)CC(=O)Nc1c(C)cccc1C", 234.337),
    # procainamide: Ar-CO-NH-CH2CH2-N(Et)2
    "procainamide": ("CCN(CC)CCNC(=O)c1ccc(N)cc1", 235.325),
    # trimethoprim: 3,4,5-trimethoxybenzyl + 2,4-diaminopyrimidine
    "trimethoprim": ("COc1cc(OC)c(Cc2cnc(N)c(N)n2)c1OC", 290.321),
    # clofibrate: ethyl 2-(4-chlorophenoxy)-2-methylpropionate (ester!)
    "clofibrate": ("CCOC(=O)C(C)(C)Oc1ccc(Cl)cc1", 242.699),
    # pregabalin stereo-free: 3-(aminomethyl)-5-methylhexanoic acid
    "pregabalin": ("CC(C)CCC(CN)C(=O)O", 159.226),
    # gabapentin: 1-(aminomethyl)cyclohexane-1-acetic acid (6-ring!)
    "gabapentin": ("NCC1(CC(=O)O)CCCCC1", 171.237),
    # amantadine: 1-aminoadamantane
    "amantadine": ("NC1C2CC3CC1CC(C2)C3", 151.248),
    # mefenamic acid: N-(2,3-dimethylphenyl)anthranilic acid
    "mefenamic acid": ("O=C(O)c1ccccc1Nc1cccc(C)c1C", 241.289),
    # chlorpropamide: p-chlorophenylsulfonyl-PROPYL-urea
    "chlorpropamide": ("CCCNC(=O)NS(=O)(=O)c1ccc(Cl)cc1", 276.745),
    # methimazole: 1-methyl-1,3-dihydro-2H-imidazole-2-thione
    "methimazole": ("CN1C=NC(=S)N1", 114.169),
    # menthol stereo-free: 5-methyl-2-isopropylcyclohexan-1-ol
    "menthol": ("CC1CCC(C(C)C)CC1O", 156.265),
    # thymol: 2-isopropyl-5-methylphenol
    "thymol": ("Cc1ccc(O)c(C(C)C)c1", 150.221),
}

for name, (candidate, published) in CANDIDATES.items():
    mol = Chem.MolFromSmiles(candidate)
    if mol is None:
        print(f"UNPARSEABLE {name}: {candidate}")
        continue
    canonical = Chem.MolToSmiles(mol)
    mw = Descriptors.MolWt(mol)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    delta = mw - published
    flag = "OK " if abs(delta) <= 0.02 else "BAD"
    print(f"{flag} {name:16s} canonical={canonical:45s} MW={mw:8.3f} "
          f"(delta {delta:+.3f}) {formula}")

# uniqueness against the current corpus
from pharmaml.generative import DEFAULT_CORPUS

current = set(DEFAULT_CORPUS)
print()
for name, (candidate, _) in CANDIDATES.items():
    mol = Chem.MolFromSmiles(candidate)
    canonical = Chem.MolToSmiles(mol)
    if canonical in current:
        print(f"COLLISION: fixed {name} already in corpus as {canonical}")
print("probe 8 complete")
