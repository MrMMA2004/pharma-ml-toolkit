"""Probe 1-5: the confirmed defect families in pharmaml.biology.response."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pharmaml.biology.response import (
    audit_response_table,
    collapse_replicates,
    load_response_table,
    response_matrix,
    summarize_compounds,
)

print("=== DEFECT 1: bool column auto-detected as a metric ===")
frame = pd.DataFrame(
    {
        "cell_line": ["A549", "HCT116", "MCF7"],
        "compound": ["cisplatin"] * 3,
        "msi_status": [True, False, True],  # annotation flag, NOT a metric
        "LN_IC50": [1.0, 2.0, 3.0],
    }
)
screen = load_response_table(frame)
print("metrics:", screen.metrics)
print("msi_status in metrics?", "msi_status" in screen.metrics)

print()
print("=== DEFECT 2: collapse_replicates drops NaN-key rows ===")
frame2 = pd.DataFrame(
    {
        "cell_line": ["A", "A", None],
        "compound": ["d", "d", "d"],
        "v": [1.0, 3.0, 99.0],
    }
)
screen2 = load_response_table(frame2, metrics=["v"])
print("rows in :", len(screen2.frame))
collapsed = collapse_replicates(screen2)
print("rows out:", len(collapsed.frame))
print(collapsed.frame.to_string())
print("NaN row survived?", None in set(collapsed.frame["cell_line"]))

print()
print("=== DEFECT 3: audit blind spot for missing identifiers ===")
frame3 = pd.DataFrame(
    {
        "cell_line": ["A", "B", None, "A"],
        "compound": ["d", None, "d", "d"],
        "v": [1.0, 2.0, 3.0, 4.0],
    }
)
screen3 = load_response_table(frame3, metrics=["v"])
print(screen3.summary())
print(audit_response_table(screen3).to_string(index=False))

print()
print("=== DEFECT 4: literal 'nan' string in cell_lines/compounds ===")
print("cell_lines:", screen3.cell_lines)
print("compounds:", screen3.compounds)

print()
print("=== DEFECT 5 sweep: _duplicate_pairs with NaN keys ===")
frame5 = pd.DataFrame(
    {
        "cell_line": ["A", "A", None, None],
        "compound": ["d", "d", "d", "d"],
        "v": [1.0, 2.0, 3.0, 4.0],
    }
)
screen5 = load_response_table(frame5, metrics=["v"])
print("rows:", len(screen5.frame))
print("n_pairs (drop_duplicates default):", screen5.n_pairs)
print("duplicate pairs:", screen5._duplicate_pairs())

print()
print("=== DEFECT 5 sweep: response_matrix after collapse with NaN keys ===")
m = response_matrix(collapse_replicates(screen5), "v")
print(m.to_string())

print()
print("=== DEFECT 5 sweep: summarize_compounds with NaN compound keys ===")
frame6 = pd.DataFrame(
    {
        "cell_line": ["A", "B", "C", "A", "B", "C"],
        "compound": ["d", "d", "d", None, None, None],
        "v": [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
    }
)
screen6 = load_response_table(frame6, metrics=["v"])
print(summarize_compounds(screen6, "v").to_string())
