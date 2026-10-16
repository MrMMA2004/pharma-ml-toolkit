"""100x-21-b probe 7: named-drug MW claims vs computed/published values.

Suggested probe #7: spot-check the corpus comments (~60 named drugs)
against RDKit-computed molecular weights and a table of published
values; also verify the docstring's claimed ranges (46-365 Da,
LogP -3.2..+4.9, 3-48 characters).
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

from pharmaml.generative import DEFAULT_CORPUS

# (index-in-comment, canonical smiles from corpus, name, published MW)
PUBLISHED = {
    "aspirin": 180.159, "paracetamol": 151.163, "ibuprofen": 206.287,
    "caffeine": 194.192, "metformin": 129.164, "nicotine": 162.232,
    "warfarin": 308.334, "cimetidine": 252.346, "theophylline": 180.167,
    "diclofenac": 296.149, "naproxen": 230.262, "ketoprofen": 254.283,
    "propranolol": 259.348, "atenolol": 266.337, "metoprolol": 267.365,
    "salbutamol": 239.311, "lidocaine": 234.337, "procainamide": 235.325,
    "phenobarbital": 232.237, "sulfanilamide": 172.205,
    "sulfadiazine": 250.284, "trimethoprim": 290.321,
    "sulfamethoxazole": 253.279, "phenytoin": 252.268,
    "fluoxetine": 309.330, "clofibrate": 242.699, "pregabalin": 159.226,
    "gabapentin": 171.237, "amantadine": 151.248, "mefenamic acid": 241.289,
    "chloramphenicol": 323.132, "amoxicillin": 365.405,
    "nicotinamide": 122.128, "isoniazid": 137.139, "pyrazinamide": 123.113,
    "metronidazole": 171.155, "acyclovir": 225.205, "clonidine": 230.093,
    "benzocaine": 165.192, "procaine": 236.311, "tolbutamide": 270.353,
    "chlorpropamide": 276.745, "melatonin": 232.278, "primidone": 218.252,
    "allopurinol": 136.112, "methimazole": 114.169, "menthol": 156.265,
    "thymol": 150.221, "phenylephrine": 167.206, "synephrine": 167.206,
    "progesterone": 314.469, "testosterone": 288.424, "estradiol": 272.388,
    "pantothenic acid": 219.235, "saccharin": 183.185,
    "succinimide": 99.089, "menadione": 172.180,
}

NAMED_COMMENTS = {}
for line_no, entry in enumerate(DEFAULT_CORPUS):
    pass  # comments are in source; map by index below instead

# map corpus index -> (name) via source comments, extracted programmatically
import inspect
import re

import pharmaml.generative.corpus as corpus_mod

source = inspect.getsource(corpus_mod)
named: dict[int, str] = {}
pattern = re.compile(r'^\s*"([^"]+)",\s*#\s*([A-Za-z][A-Za-z0-9 ()/-]*?)\s*(?:\(stereo-free\))?\s*$')
for line in source.splitlines():
    match = pattern.match(line)
    if match:
        named[match.group(1)] = match.group(2).strip()

print(f"entries with a name comment: {len(named)}")
bad = 0
for smiles, name in named.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"UNPARSEABLE {name}: {smiles}")
        bad += 1
        continue
    mw = Descriptors.MolWt(mol)
    published = PUBLISHED.get(name.lower())
    if published is None:
        print(f"no published reference for {name!r} (computed {mw:.2f})")
        continue
    delta = abs(mw - published)
    flag = "OK " if delta <= 0.35 else "BAD"
    if delta > 0.35:
        bad += 1
    print(f"{flag} {name:18s} computed={mw:8.3f} published={published:8.3f} "
          f"delta={delta:+.3f}")

print(f"\nmismatch count (|delta| > 0.35 Da): {bad}")

# ---- docstring range claims ---------------------------------------------
mws, logps, lengths = [], [], []
for smiles in DEFAULT_CORPUS:
    mol = Chem.MolFromSmiles(smiles)
    mws.append(Descriptors.MolWt(mol))
    logps.append(Descriptors.MolLogP(mol))
    lengths.append(len(smiles))
print(f"MW range: {min(mws):.1f} .. {max(mws):.1f} (docstring claims 46-365)")
print(f"LogP range: {min(logps):.2f} .. {max(logps):.2f} (docstring claims -3.2..+4.9)")
print(f"length range: {min(lengths)} .. {max(lengths)} (docstring claims 3-48)")
print("probe 7 complete")
