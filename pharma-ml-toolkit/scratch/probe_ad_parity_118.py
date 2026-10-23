"""Python side of the ml-ad.ts parity pair (100x-118).

Builds the deterministic LCG dataset (bit-exact in TS via the same
integer recurrence), runs the SHIPPED ``knn_applicability_domain`` and
writes the numeric readout the bun twin compares against.

Run from the toolkit root:
    python scratch/probe_ad_parity_118.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pharmaml.ml.applicability import knn_applicability_domain

OUT = Path("/tmp/ad_parity_py.json")

# deterministic minstd (Lehmer) generator -> [0, 1) floats, exact in
# BOTH languages: the product stays < 2**46 so JS doubles never lose a
# bit (the classic glibc LCG overflows Number precision — 1.1e9 * 2**31
# > 2**53 — and silently diverges)
STATE = 42
MOD = 2**31 - 1


def lcg() -> float:
    global STATE
    STATE = (48271 * STATE) % MOD
    return STATE / MOD


rows = [[lcg(), lcg(), lcg()] for _ in range(60)]
frame = pd.DataFrame(rows, columns=["a", "b", "c"])
query = pd.DataFrame(
    [rows[0], rows[7], rows[15], [5.0, -4.0, 6.0]],
    columns=["a", "b", "c"],
)

result = knn_applicability_domain(frame, query, k=5, percentile=95.0)
payload = {
    "trainMean": [float(v) for v in result.train_knn_mean],
    "testMean": [float(v) for v in result.test_knn_mean],
    "threshold": float(result.distance_threshold),
    "inside": [bool(v) for v in result.inside_ad],
    "featureNames": result.feature_names,
}
OUT.write_text(json.dumps(payload), encoding="utf-8")
print(f"python side written: {OUT} "
      f"(threshold={payload['threshold']:.6g}, "
      f"inside={['T' if v else 'F' for v in payload['inside']]})")
