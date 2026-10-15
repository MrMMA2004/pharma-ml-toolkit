"""Probe suspected defects in chem/descriptors.py + chem/similarity.py (pre-fix evidence)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pharmaml.chem.descriptors import compute_descriptors, compute_fingerprints, compute_maccs
from pharmaml.chem.similarity import similarity_scan, top_matches, tanimoto_from_bitarrays, tanimoto_matrix

print("=== P1: compute_descriptors with a GENERATOR (docstring says 'Iterable') ===")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = compute_descriptors(iter(["CCO", "CCN"]), progress=False)
    print("P1 ok:", frame.shape, frame["SMILES"].tolist())
except Exception as exc:
    print(f"P1 CRASH: {type(exc).__name__}: {exc}")

print("\n=== P2: empty-string SMILES -> 'valid' empty molecule? ===")
mat, valid = compute_fingerprints(["", "CCO"])
print(f"P2 valid_idx={valid} matrix={mat.shape} row0 all-zero={bool((mat[0] == 0).all())}")
frame = compute_descriptors(["", "CCO"], progress=False)
print(f"P2 descriptor row for '': MolWt={frame.iloc[0]['MolWt']!r} HeavyAtomCount={frame.iloc[0]['HeavyAtomCount']!r}")

print("\n=== P3: similarity_scan with pd.Series carrying a shuffled integer index ===")
refs = pd.Series(["CCO", "CCN", "CCF"], index=[1, 0, 2])  # positional 0 = 'CCO' (label 1)
res = similarity_scan(refs, ["CCO"], threshold=0.99, fingerprint="morgan")
print(f"P3 reference strings reported: {res['reference'].tolist()}  (positional truth: ['CCO'])")
print(f"P3 reference_index reported:   {res['reference_index'].tolist()}")
print(f"P3 similarity values:          {res['similarity'].tolist()}")

print("\n=== P3b: Series with index not containing 0 at all ===")
refs2 = pd.Series(["CCO", "CCN"], index=[10, 20])
try:
    res2 = similarity_scan(refs2, ["CCO"], threshold=0.99, fingerprint="morgan")
    print(f"P3b result: {res2['reference'].tolist()}")
except Exception as exc:
    print(f"P3b CRASH: {type(exc).__name__}: {exc}")

print("\n=== P4: top_matches with an INVALID query SMILES ===")
res3 = top_matches("C1CC(` bad", ["CCO", "CCN", "CCF"])
print(f"P4 result type={type(res3).__name__} shape={res3.shape} -- silent empty, no error")

print("\n=== P5: compute_maccs silently drops invalid SMILES (no warning)? ===")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import logging
    class _ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []
        def emit(self, record):
            self.records.append(record.getMessage())
    import pharmaml.chem.descriptors as desc_mod
    handler = _ListHandler()
    desc_mod._LOG.addHandler(handler)
    m, v = compute_maccs(["CCO", "DEADBEEF$$$"])
    desc_mod._LOG.removeHandler(handler)
print(f"P5 valid_idx={v} rows={m.shape} log-warnings={handler.records}")

print("\n=== P6: NaN in a float 'bit array' silently becomes a SET bit ===")
a = np.array([1.0, np.nan, 0.0])
b = np.array([1.0, 0.0, 0.0])
print(f"P6 astype(bool) of {a} -> {a.astype(bool)}")
print(f"P6 tanimoto_from_bitarrays(a,b) = {tanimoto_from_bitarrays(a, b)}  (NaN counted as shared bit!)")

print("\n=== P7: bare STRING input explodes into per-character 'molecules' ===")
mat7, valid7 = compute_fingerprints("CCO")
print(f"P7 compute_fingerprints('CCO') -> matrix {mat7.shape}, valid_idx {valid7} (3 single-atom fps!)")

print("\n=== P8: tanimoto_matrix uint32 matmul sanity / all-zero rows ===")
rng = np.random.default_rng(0)
r8 = (rng.random((3, 32)) > 0.7).astype(np.uint8)
t8 = np.zeros((2, 32), dtype=np.uint8)
m8 = tanimoto_matrix(r8, t8)
print(f"P8 zero-target rows -> {m8[:, 0].tolist()} (pairwise convention says 1.0)")
print(f"P8 dtype={m8.dtype}")

print("\n=== P9: similarity_scan tie ordering (quicksort) in top_matches ===")
lib = ["CCO", "CNO", "CFO", "CClO"]  # distinct
res9 = top_matches("CCO", lib, k=4, fingerprint="morgan", n_bits=64)
print(f"P9 order: {res9['target'].tolist()}")

print("\n=== P10: duplicate SMILES + mixed invalid in similarity_scan (list input sanity) ===")
res10 = similarity_scan(["CCO", "bad$$"], ["CCO", "CCO"], threshold=0.5, fingerprint="morgan")
print(f"P10 rows={len(res10)} refs shown={res10['reference'].unique().tolist()}")

print("\nDONE")
