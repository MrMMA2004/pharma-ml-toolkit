"""Smoke: build a small synthetic screen CSV and run the biology CLI verbs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
csv_path = HERE / "smoke_screen.csv"

rng = np.random.default_rng(5)
cells = ["A549", "HCT116", "MCF7", "HL60"]
compounds = ["cisplatin", "erlotinib"]
rows = []
for cell in cells:
    for compound in compounds:
        rows.append(
            {
                "cell_line": cell,
                "compound": compound,
                "tissue": "lung" if cell == "A549" else "colon",
                "msi_status": cell in ("A549", "HCT116"),  # bool flag column
                "LN_IC50": round(float(rng.uniform(-2.0, 4.0)), 4),
            }
        )
rows[0]["LN_IC50"] = float("nan")            # one missing metric value
rows.append({**rows[1], "LN_IC50": 1.5})     # duplicate (HCT116, cisplatin)
rows.append({"cell_line": None, "compound": "erlotinib", "tissue": "colon",
             "msi_status": False, "LN_IC50": 2.0})   # row without a cell line
frame = pd.DataFrame(rows)
frame.to_csv(csv_path, index=False)
print(f"wrote {csv_path} ({len(frame)} rows)")

for command in (
    ["biology", "response-audit", "--data", str(csv_path), "--tissue-column", "tissue"],
):
    print(f"\n$ pharmaml {' '.join(command)}")
    proc = subprocess.run(
        [sys.executable, "-m", "pharmaml.cli", *command],
        capture_output=True, text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr)
    print("exit code:", proc.returncode)
