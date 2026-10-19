"""Probe 5: partial-overlap alignment where the EXPRESSION side is longer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pharmaml.biology.omics import correlate_with_response

rng = np.random.default_rng(7)

print("=== N: expression longer than the shared labels ===")
expr = pd.DataFrame(
    rng.normal(size=(4, 2)), index=["a", "b", "c", "d"], columns=["G1", "G2"]
)
resp = pd.Series(rng.normal(size=2), index=["a", "b"])
try:
    out = correlate_with_response(expr, resp)
    print("result:\n", out.to_string())
except Exception as exc:
    print(f"CRASH: {type(exc).__name__}: {exc}")

print()
print("=== O: NaN labels in the expression index ===")
expr2 = pd.DataFrame(
    rng.normal(size=(3, 2)), index=["a", "b", np.nan], columns=["G1", "G2"]
)
resp2 = pd.Series(rng.normal(size=2), index=["a", "b"])
try:
    out = correlate_with_response(expr2, resp2)
    print("result:\n", out.to_string())
except Exception as exc:
    print(f"CRASH: {type(exc).__name__}: {exc}")

print()
print("=== P: does Index.intersection match NaN to NaN? ===")
left = pd.Index(["a", np.nan])
right = pd.Index([np.nan, "a"])
print("intersection:", left.intersection(right).tolist())
print("duplicated on [a, nan, nan]:", pd.Index(["a", np.nan, np.nan]).duplicated().tolist())
