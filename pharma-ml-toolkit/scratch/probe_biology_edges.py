"""Probe 2: the ALSO-REVIEW items (verify-before-fix)."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from pharmaml.biology.response import (
    call_sensitive_resistant,
    ln_ic50_to_pic50,
    pic50_to_ln_ic50,
)
from pharmaml.biology.omics import (
    correlate_with_response,
    normalize_expression,
)

print("=== A: call_sensitive_resistant quantile edge cases ===")
# all-tied values
tied = pd.Series([1.0, 1.0, 1.0, 1.0])
print("all-tied labels:", call_sensitive_resistant(tied).tolist())
# two values
two = pd.Series([1.0, 2.0])
print("two values:", call_sensitive_resistant(two).tolist())
# one value
one = pd.Series([5.0])
print("one value:", call_sensitive_resistant(one).tolist())
# empty
empty = pd.Series([], dtype=float)
print("empty:", call_sensitive_resistant(empty).tolist())
# ties at the threshold of a spread vector
spread = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
print("ties-at-cut:", call_sensitive_resistant(spread).tolist())

print()
print("=== B: ln_ic50_to_pic50 round-trip on NaN / extreme arrays ===")
arr = np.array([0.0, -8.0, np.nan, 5.0])
fwd = ln_ic50_to_pic50(arr)
back = pic50_to_ln_ic50(fwd)
print("fwd:", fwd)
print("back:", back)
print("nan passthrough:", math.isnan(fwd[2]), math.isnan(back[2]))
# negative / zero pic50 inputs
print("pic50=-1 ->", pic50_to_ln_ic50(-1.0), "(= ln(10^7 uM) fine)")
print("pic50=0  ->", pic50_to_ln_ic50(0.0))

print()
print("=== C: normalize_expression edge cases ===")
# values between 0 and 1
frac = pd.DataFrame({"G": [0.0, 0.25, 0.5, 1.0]})
print("log2 of fractions:", normalize_expression(frac, z_score=False)["G"].tolist())
# all-equal frame
eq = pd.DataFrame({"G": [3.0, 3.0, 3.0]})
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("all-equal z:", normalize_expression(eq)["G"].tolist())
# integer dtype
ints = pd.DataFrame({"G": pd.Series([1, 2, 3], dtype="int64")})
print("int dtype ->", normalize_expression(ints, z_score=False)["G"].tolist(),
      normalize_expression(ints)["G"].tolist())
# all-NaN gene
nan_gene = pd.DataFrame({"G": [np.nan, np.nan], "V": [1.0, 3.0]})
print("all-NaN gene z:", normalize_expression(nan_gene)["G"].tolist())

print()
print("=== D: correlate_with_response with duplicate index labels ===")
rng = np.random.default_rng(3)
# duplicate labels in the EXPRESSION index
expr = pd.DataFrame(
    rng.normal(size=(3, 2)), index=["a", "a", "b"], columns=["G1", "G2"]
)
resp = pd.Series(rng.normal(size=2), index=["a", "b"])
try:
    out = correlate_with_response(expr, resp)
    print("expression-dup result:\n", out.to_string())
except Exception as exc:
    print(f"expression-dup CRASH: {type(exc).__name__}: {exc}")

# duplicate labels in the RESPONSE index
expr2 = pd.DataFrame(
    rng.normal(size=(2, 2)), index=["a", "b"], columns=["G1", "G2"]
)
resp2 = pd.Series(rng.normal(size=3), index=["a", "a", "b"])
try:
    out = correlate_with_response(expr2, resp2)
    print("response-dup result:\n", out.to_string())
except Exception as exc:
    print(f"response-dup CRASH: {type(exc).__name__}: {exc}")

# NaN in the response index
resp3 = pd.Series([1.0, 2.0, 3.0], index=["a", "b", np.nan])
try:
    out = correlate_with_response(expr2, resp3)
    print("nan-index result:\n", out.to_string())
except Exception as exc:
    print(f"nan-index CRASH: {type(exc).__name__}: {exc}")

print()
print("=== E: pivot behaviour with NaN keys (response_matrix path) ===")
frame = pd.DataFrame(
    {
        "cell_line": ["A", "A", None],
        "compound": ["x", "x", "y"],
        "v": [1.0, 2.0, 9.0],
    }
)
try:
    pivoted = frame.pivot(index="cell_line", columns="compound", values="v")
    print("pivot with NaN key:\n", pivoted.to_string())
except Exception as exc:
    print(f"pivot CRASH: {type(exc).__name__}: {exc}")
