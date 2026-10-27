"""Probe A+B: interpret.py decode edges + signed_feature_correlations edges.

Round-36 audit (100x-153). Deterministic, self-contained, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy

from pharmaml.chem.interpret import decode_morgan_bit, signed_feature_correlations
from pharmaml.exceptions import DataValidationError

print(f"scipy {scipy.__version__}, numpy {np.__version__}, pandas {pd.__version__}")

# ---------------------------------------------------------------- A1: radius-0 drawing
# DrawMorganBit on a bit whose ONLY hits are radius 0 (aspirin bit 650).
info = decode_morgan_bit("CC(=O)Oc1ccccc1C(=O)O", 650)  # with_drawing default True
print("A1 radius-0 drawing ok:", info.drawing is not None and info.drawing.startswith("<?xml"))
print("A1 radius-0 hits:", [(h.atom_index, h.radius, h.environment_smiles) for h in info.hits])

# ---------------------------------------------------------------- A2: mixed-radius bit
# Find a bit that fires at TWO DIFFERENT radii on one molecule, then check
# each hit's environment size matches ITS OWN radius (the D2.py bug fix).
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
add = rdFingerprintGenerator.AdditionalOutput()
add.AllocateBitInfoMap()
gen.GetFingerprint(mol, additionalOutput=add)
bmap = add.GetBitInfoMap()
mixed = [(b, hits) for b, hits in bmap.items() if len({r for _, r in hits}) > 1]
print("A2 bits with mixed radii on aspirin:", len(mixed))
if mixed:
    bit, hits = mixed[0]
    print("A2 example bit", bit, "hits", list(hits))
    rec = decode_morgan_bit("CC(=O)Oc1ccccc1C(=O)O", int(bit))
    for hit in rec.hits:
        print(
            "   hit atom", hit.atom_index, "radius", hit.radius,
            "env smiles", hit.environment_smiles, "n_atoms", len(hit.atom_indices),
        )
    # every hit's atom count must be consistent with its own radius:
    # a radius-r environment on a connected heavy-atom molecule has
    # fewer atoms than a radius-(r+1) one around the SAME center.
    same_center = {}
    for h in rec.hits:
        same_center.setdefault(h.atom_index, []).append(h)
    for center, hs in same_center.items():
        if len(hs) > 1:
            sizes = [(h.radius, len(h.atom_indices)) for h in sorted(hs, key=lambda x: x.radius)]
            print("   same-center env sizes (radius, n):", sizes)

# ---------------------------------------------------------------- A3: hits order
print("A3 hit order ascending atom index:",
      [h.atom_index for h in info.hits] == sorted(h.atom_index for h in info.hits))

# ---------------------------------------------------------------- A4: smiles guard
for bad in (123, None, "   ", ""):
    try:
        decode_morgan_bit(bad, 650)  # type: ignore[arg-type]
        print("A4 FAIL: accepted", repr(bad))
    except DataValidationError as exc:
        print("A4 typed for", repr(bad), "->", str(exc)[:60])

# ---------------------------------------------------------------- A5: bool radius/bit
try:
    decode_morgan_bit("CCO", 80, radius=True)
    print("A5 FAIL bool radius accepted")
except DataValidationError as exc:
    print("A5 bool radius typed:", str(exc)[:50])
try:
    decode_morgan_bit("CCO", True)
    print("A5 FAIL bool bit accepted")
except DataValidationError as exc:
    print("A5 bool bit typed:", str(exc)[:50])

# ---------------------------------------------------------------- A6: negative bit
try:
    decode_morgan_bit("CCO", -1)
    print("A6 FAIL negative bit accepted")
except DataValidationError as exc:
    print("A6 negative bit typed:", str(exc)[:70])

# ---------------------------------------------------------------- B1: spearman statistic attr
y = [1.0, 2.0, 3.0, 4.0]
frame = pd.DataFrame({"cube": [v ** 3 for v in y], "lin": y})
s = signed_feature_correlations(frame, y, method="spearman")
print("B1 spearman ok:", float(s.iloc[0]), s.index.tolist())

# ---------------------------------------------------------------- B2: bool column
fb = pd.DataFrame({"flag": [True, False, True, False], "lin": y})
try:
    sb = signed_feature_correlations(fb, y)
    print("B2 bool column correlated:", float(sb["flag"]))
except Exception as exc:  # noqa: BLE001
    print("B2 bool column FAILED:", type(exc).__name__, exc)

# ---------------------------------------------------------------- B3: ties in |r|
ft = pd.DataFrame({"neg": [-v for v in y], "pos": y})
st = signed_feature_correlations(ft, y)
print("B3 tie order (pos first):", st.index.tolist(), st.tolist())

# ---------------------------------------------------------------- B4: complex column leak
fc = pd.DataFrame({"cplx": np.array([1 + 2j, 3 + 1j, 2 + 2j, 1 + 1j]), "lin": y})
try:
    signed_feature_correlations(fc, y)
    print("B4 complex column accepted silently (unexpected)")
except DataValidationError as exc:
    print("B4 complex column typed:", str(exc)[:70])
except TypeError as exc:
    print("B4 complex column leaks RAW TypeError:", exc)

# ---------------------------------------------------------------- B5: nullable Int64 with NA
fna = pd.DataFrame({"nullable": pd.array([1, 2, None, 4], dtype="Int64"), "lin": y})
try:
    sna = signed_feature_correlations(fna, y)
    print("B5 nullable Int64 ok:", float(sna["nullable"]))
except Exception as exc:  # noqa: BLE001
    print("B5 nullable Int64 FAILED:", type(exc).__name__, exc)

# ---------------------------------------------------------------- B6: all-inf column
finf = pd.DataFrame({"inf": [np.inf, -np.inf, np.inf, 1.0], "lin": y})
try:
    sinf = signed_feature_correlations(finf, y)
    print("B6 all-inf column dropped:", "inf" not in sinf.index, sinf.index.tolist())
except DataValidationError as exc:
    print("B6 typed:", str(exc)[:80])

# ---------------------------------------------------------------- B7: target as Series + generator
f7 = pd.DataFrame({"a": y})
s7 = signed_feature_correlations(f7, pd.Series(y))
print("B7 Series target ok:", float(s7["a"]))
try:
    signed_feature_correlations(f7, (v for v in y))
    print("B7 generator target accepted")
except Exception as exc:  # noqa: BLE001
    print("B7 generator target FAILED:", type(exc).__name__, exc)

# ---------------------------------------------------------------- B8: empty frame, 0 rows
try:
    signed_feature_correlations(pd.DataFrame(columns=["a"]), [])
    print("B8 empty accepted")
except DataValidationError as exc:
    print("B8 empty typed:", str(exc)[:60])

print("PROBE_A_B_DONE")
