"""Probe 3: follow-up edges from probe results."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pharmaml.biology.biomarkers import sensitivity_associations, top_biomarkers
from pharmaml.biology.omics import correlate_with_response

print("=== F: bool-ish object columns in auto-detection ===")
mixed = pd.Series([True, False, None])  # nullable boolean -> object dtype
print("dtype:", mixed.dtype)
print("to_numeric:", pd.to_numeric(mixed, errors="coerce").tolist())
print("to_numeric on 'True' string:", pd.to_numeric(pd.Series(["True"]), errors="coerce").tolist())

print()
print("=== E2: pivot with NaN keys and UNIQUE pairs ===")
frame = pd.DataFrame(
    {
        "cell_line": ["A", "B", None],
        "compound": ["x", "x", "y"],
        "v": [1.0, 2.0, 9.0],
    }
)
try:
    pivoted = frame.pivot(index="cell_line", columns="compound", values="v")
    print("pivot result:\n", pivoted.to_string())
except Exception as exc:
    print(f"pivot CRASH: {type(exc).__name__}: {exc}")

print()
print("=== G: alignment warning gating (unmatched response row, no dup labels) ===")
rng = np.random.default_rng(5)
expr = pd.DataFrame(rng.normal(size=(2, 2)), index=["a", "b"], columns=["G1", "G2"])
resp = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "zzz"])  # zzz unmatched
out = correlate_with_response(expr, resp)
print("no crash; did a warning fire about 1 unmatched response value?")
print(out.to_string())

print()
print("=== H: BH-FDR with all-NaN p column / single gene ===")
expr_single = pd.DataFrame({"G": [1.0, 2.0, 3.0, 4.0]}, index=list("wxyz"))
resp_single = pd.Series([4.0, 3.0, 2.0, 1.0], index=list("wxyz"))
assoc = sensitivity_associations(expr_single, resp_single)
print("single gene:\n", assoc.to_string())
# all-NaN p: constant gene only
expr_const = pd.DataFrame({"G": [7.0] * 4}, index=list("wxyz"))
assoc2 = sensitivity_associations(expr_const, resp_single)
print("constant-only:\n", assoc2.to_string())
print("significant dtype:", assoc2["significant"].dtype, assoc2["significant"].tolist())
print("top_biomarkers on all-NaN:", len(top_biomarkers(assoc2, n=5)))

print()
print("=== I: NaN cell line in CLI top-biomarkers replicate collapse ===")
subset = pd.DataFrame(
    {
        "cell_line": ["A", "A", None, "B"],
        "pIC50": [1.0, 3.0, 9.0, 5.0],
    }
)
collapsed = subset.groupby("cell_line", as_index=False).agg({"pIC50": "median"})
print("CLI-style collapse result:\n", collapsed.to_string())
print("NaN row survived?", None in set(collapsed["cell_line"]))

print()
print("=== J: duplicated() on NaN keys (n_pairs consistency) ===")
dup = pd.DataFrame({"a": [np.nan, np.nan, "x"], "b": ["d", "d", "d"]})
print("drop_duplicates:\n", dup.drop_duplicates().to_string())
gb = dup.groupby(["a", "b"], dropna=False).size()
print("groupby dropna=False:\n", gb.to_string())
