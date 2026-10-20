"""103 driver: re-arm the Huber-vs-OLS outlier demo after the pIC50 clip.

Round 100x-103 clipped the qsar_regression endpoint to the assay-plausible
window [3, 11] (pre-range -4.3..14.7 was chemically impossible).  The ML
notebook's §9 robustness demo corrupted 15 training pIC50 values by
+4 log units -- on the old wide target that produced gross outliers, but
on the compressed censored target OLS barely noticed (clean-test RMSE
1.057 vs Huber's 1.098: the assert "Huber should beat plain least squares
under outliers" failed, honest casualty of the data repair).

The cure keeps the pedagogy and the additive-log-units narrative: corrupt
by **+8 log units -- the full width of the assay window** -- so every
corrupted label lands beyond the 11.0 ceiling (11..19, i.e. IC50 below
10 femtomolar: mis-entered plate values).  Verified margins on the shipped
split: OLS 1.2439 vs Huber 1.0982 (was 1.0573 vs 1.0982 -- a 0.146 RMSE
margin instead of a 0.041 deficit).

Applies to complete_ml_workflow.ipynb AND its no-output twin (code cells
only).  Execution is deferred to scratch/reexecute_all_notebooks_64.py,
as ever.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "examples" / "complete_ml_workflow.ipynb",
    ROOT / "examples" / "complete_ml_workflow_no_output.ipynb",
]

OLD_COMMENT = """# Gross activity outliers wreck least-squares fits because squared error
# grows quadratically with the residual; Huber's linear-epsilon loss
# saturates instead. Corrupt 15 training pIC50 values by +4 log units,
# then judge both models on the *clean* held-out set:
ry_corrupt = ry_tr.copy()
ry_corrupt.iloc[:15] += 4.0"""

NEW_COMMENT = """# Gross activity outliers wreck least-squares fits because squared error
# grows quadratically with the residual; Huber's linear-epsilon loss
# saturates instead. Corrupt 15 training pIC50 values by +8 log units --
# the full width of the assay-plausible window [3, 11], so every corrupted
# label lands beyond the ceiling (a mis-entered plate value) -- then judge
# both models on the *clean* held-out set:
ry_corrupt = ry_tr.copy()
ry_corrupt.iloc[:15] += 8.0"""


def main() -> None:
    patched = 0
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        hits = 0
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if "ry_corrupt" not in src:
                continue
            if OLD_COMMENT in src:
                cell["source"] = src.replace(OLD_COMMENT, NEW_COMMENT).splitlines(keepends=True)
                hits += 1
            elif NEW_COMMENT in src:
                print(f"  {path.name}: already patched")
                hits += 1
            else:
                raise SystemExit(f"UNEXPECTED SOURCE in {path.name}:\n{src[:400]}")
        if hits == 0:
            raise SystemExit(f"no ry_corrupt cell found in {path.name}")
        path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"  {path.name}: {hits} cell(s) patched")
        patched += hits
    print(f"OK: {patched} cell(s) across {len(NOTEBOOKS)} notebooks")


if __name__ == "__main__":
    main()
