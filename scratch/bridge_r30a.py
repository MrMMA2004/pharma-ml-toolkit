"""R30-a end-to-end: the report bridge renders the ligands group.

Feeds report_bridge.py the exact JSON the route would send for a
session whose only content is a ligand-set reconciliation, and
checks the response: HTML with the metrics card, the bar figure and
the bucket tables; census figures >= 1.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

payload = {
    "theme": "light",
    "subtitle": "R30-a ligand-set demo",
    "sections": [
        {
            "kind": "metrics",
            "title": "Ligand-set reconciliation — A vs B",
            "items": [
                ["common (both tables)", "4"],
                ["only in A", "60"],
                ["only in B", "1"],
                ["unique names compared", "65"],
                ["normalisation", "extensions stripped, case folded"],
            ],
        },
        {
            "kind": "figure",
            "title": "Ligand-set reconciliation — bucket counts",
            "xLabel": "number of unique ligand names",
            "yLabel": "bucket",
            "traces": [
                {
                    "name": "ligands",
                    "style": "bar",
                    "points": [[2, 4], [1, 60], [0, 1]],
                }
            ],
        },
        {
            "kind": "table",
            "title": "Common ligands — 4",
            "columns": ["#", "ligand name"],
            "rows": [[1, "aspirin"], [2, "caffeine"], [3, "ibuprofen"], [4, "metformin"]],
        },
        {
            "kind": "table",
            "title": "Only in A — 60",
            "columns": ["#", "ligand name"],
            "rows": [[1, "ligand_A_only_01"], [2, "ligand_A_only_02"],
                     ["", "+ 58 more (see Ligand Sets in the Studio or the Explorer report settings)"]],
        },
    ],
}

proc = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "report_bridge.py")],
    input=json.dumps(payload), capture_output=True, text=True, timeout=120,
)
if proc.returncode != 0:
    print("BRIDGE FAILED:", proc.returncode)
    print(proc.stdout[-800:])
    print(proc.stderr[-800:])
    sys.exit(1)

result = json.loads(proc.stdout)
html = result["html"]
checks = {
    "figures >= 1": result["figures"] >= 1,
    "tables == 2": result["tables"] == 2,
    "metrics card in html": "Ligand-set reconciliation" in html,
    "bucket figure embedded": html.count("data:image/png;base64,") >= 1,
    "confession row kept": "+ 58 more" in html,
    "theme honoured": result["theme"] == "light",
}
failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
print("bytes:", result["bytes"])
if failed:
    sys.exit(f"failed: {failed}")
print("\nbridge_r30a: ALL CHECKS PASSED")
