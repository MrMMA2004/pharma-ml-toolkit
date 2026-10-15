"""Probe 4: summarize_compounds all-NaN group; response_matrix NaN columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pharmaml.biology.response import (
    load_response_table,
    response_matrix,
    summarize_compounds,
)

print("=== K: summarize_compounds with an all-NaN-metric compound ===")
frame = pd.DataFrame(
    {
        "cell_line": ["A", "B", "C", "A", "B", "C"],
        "compound": ["d", "d", "d", "dead", "dead", "dead"],
        "v": [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
    }
)
screen = load_response_table(frame, metrics=["v"])
try:
    print(summarize_compounds(screen, "v").to_string())
except Exception as exc:
    print(f"CRASH: {type(exc).__name__}: {exc}")

print()
print("=== K2: summarize_compounds with a NaN compound key (dropna=False goal) ===")
frame2 = pd.DataFrame(
    {
        "cell_line": ["A", "B", "C", "A", "B"],
        "compound": ["d", "d", "d", None, None],
        "v": [1.0, 2.0, 3.0, 7.0, 9.0],
    }
)
screen2 = load_response_table(frame2, metrics=["v"])
try:
    print(summarize_compounds(screen2, "v").to_string())
except Exception as exc:
    print(f"CRASH: {type(exc).__name__}: {exc}")

print()
print("=== L: response_matrix keeps NaN column labels? ===")
frame3 = pd.DataFrame(
    {
        "cell_line": ["A", "B", "C"],
        "compound": ["x", "x", None],
        "v": [1.0, 2.0, 9.0],
    }
)
screen3 = load_response_table(frame3, metrics=["v"])
m = response_matrix(screen3, "v")
print(m.to_string())
print("index:", list(m.index), "columns:", list(m.columns))

print()
print("=== M: collapse + matrix round trip with NaN keys ===")
frame4 = pd.DataFrame(
    {
        "cell_line": ["A", "A", None, None, "B"],
        "compound": ["x", "x", "y", "y", "x"],
        "v": [1.0, 3.0, 8.0, 4.0, 2.0],
    }
)
screen4 = load_response_table(frame4, metrics=["v"])
from pharmaml.biology.response import collapse_replicates

collapsed = collapse_replicates(screen4)
print("collapsed:\n", collapsed.frame.to_string())
try:
    m4 = response_matrix(collapsed, "v")
    print("matrix:\n", m4.to_string())
except Exception as exc:
    print(f"CRASH: {type(exc).__name__}: {exc}")
