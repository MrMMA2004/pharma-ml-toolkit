"""100x-124 — variable-genes (MAD) parity guard, Python side.

Emits /tmp/mad_ref_124.json with the SAME ranked rows the TS twin
emits over two hand-crafted matrices (no randomness — the ranking is
exact): a 9-line odd panel and an 8-line even panel, each with MAD
ties (broken by gene name ascending, the library contract), a constant
gene (MAD 0), negatives and decimals.  The Python side walks the
library's own ``most_variable_genes`` plus the desktop's honest MAD
recomputation.  The suite's TestVariableGenesParity compares the
twins row-by-row.

Run from the toolkit root: python scratch/probe_mad_parity_124.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pharmaml.biology import most_variable_genes  # noqa: E402

MATRIX_ODD = {
    # 9 cell lines -> odd medians (the middle element)
    "GENE_A": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # MAD 2
    "GENE_B": [1, 2, 3, 4, 5, 6, 7, 8, 100],  # MAD 2 (a tie, by hand)
    "GENE_C": [5, 5, 5, 5, 5, 5, 5, 5, 5],  # constant -> MAD 0
    "GENE_D": [10, 20, 30, 40, 50, 60, 70, 80, 90],  # MAD 20
    "EGFR": [9.1, 8.2, 7.3, 6.4, 5.5, 4.6, 3.7, 2.8, 1.9],  # MAD 1.8
    "KRT8": [2, 4, 6, 8, 10, 12, 14, 16, 18],  # MAD 4
    "GENE_E": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],  # MAD 2
    "GENE_F": [-3, -2, -1, 0, 1, 2, 3, 4, 5],  # MAD 2 (negatives)
    "GENE_G": [7, 7, 7, 8, 8, 8, 9, 9, 9],  # MAD 1
    "GENE_H": [1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10.0, 11.25],  # MAD 2.5
    "GENE_I": [100, 200, 300, 400, 500, 600, 700, 800, 900],  # MAD 200
    "GENE_J": [3, 1, 4, 1, 5, 9, 2, 6, 5],  # MAD 2 (unsorted values)
}

MATRIX_EVEN = {
    # 8 cell lines -> even medians (the average of the two middles)
    "GENE_A": [1, 2, 3, 4, 5, 6, 7, 8],  # MAD 2.5
    "GENE_B": [1, 2, 3, 4, 5, 6, 7, 9],  # MAD 2.5 (a tie)
    "EGFR": [8, 7, 6, 5, 4, 3, 2, 1],  # MAD 2.5 (a tie)
    "GENE_C": [2, 2, 2, 2, 2, 2, 2, 2],  # constant -> MAD 0
    "GENE_D": [0, 10, 20, 30, 40, 50, 60, 70],  # MAD 20
    "GENE_E": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],  # MAD 2.5
    "GENE_F": [-1, -2, -3, -4, -5, -6, -7, -8],  # MAD 2.5
    "GENE_G": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # MAD 1
}

TOP_N = 12


def ranked_rows(matrix: dict) -> list[dict]:
    frame = pd.DataFrame(matrix, index=[f"CELL_{i}" for i in range(len(next(iter(matrix.values()))))])
    genes = most_variable_genes(frame, n=TOP_N)
    # the desktop's honest recomputation: MAD per kept gene
    mads = (frame[genes] - frame[genes].median()).abs().median()
    return [
        {"rank": index + 1, "gene": gene, "mad": float(mads[gene])}
        for index, gene in enumerate(genes)
    ]


def main() -> None:
    payload = {
        "odd": ranked_rows(MATRIX_ODD),
        "even": ranked_rows(MATRIX_EVEN),
        "topN": TOP_N,
    }
    out = Path("/tmp/mad_ref_124.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(payload['odd'])} odd rows, {len(payload['even'])} even rows)")


if __name__ == "__main__":
    main()
