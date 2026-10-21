"""100x-114 — walkthrough §9 gains the Tanimoto heatmap figure.

The standing open item (d): tanimoto_matrix is toured in the ML
notebook, but walkthrough §9 showed the similarity scan as TABLES only.
This patch inserts one code cell after the similarity_scan/top_matches
cell: the two reference drugs + aspirin's three nearest library hits as
one compute_fingerprints -> tanimoto_matrix -> plot_similarity_heatmap
picture.  Idempotent: skips when a §9 cell already calls
plot_similarity_heatmap.
"""

from __future__ import annotations

import nbformat

PATH = "examples/pharmaml_walkthrough.ipynb"

CELL_SOURCE = '''\
from pharmaml.chem import compute_fingerprints, tanimoto_matrix
from pharmaml.viz import plot_similarity_heatmap

# the same scan as a picture: the two reference drugs plus aspirin's
# three nearest library hits, one Tanimoto heat-map — the GUI twin of
# Tanimoto-edited.py's matrix view.  The tables above answer "which
# pairs pass the threshold"; the heat-map shows the BLOCKS: near-
# duplicates glow along the diagonal, distinct scaffolds stay dark.
panel = list(dict.fromkeys(references + nearest["target"].tolist()))
bits, _valid = compute_fingerprints(panel, kind="rdkit", n_bits=2048)
heat = tanimoto_matrix(bits, bits)
labels = [s if len(s) <= 22 else s[:21] + "…" for s in panel]

fig_heat = plot_similarity_heatmap(
    heat,
    labels=labels,
    title="References vs nearest library hits — Tanimoto",
)
'''


def main() -> None:
    nb = nbformat.read(PATH, as_version=4)

    marker_cell = None
    for i, cell in enumerate(nb.cells):
        src = "".join(cell.source)
        if "top_matches(" in src and "similarity_scan" in src:
            marker_cell = i
        if "plot_similarity_heatmap" in src and cell.cell_type == "code":
            if marker_cell is not None and i > marker_cell:
                print(f"already patched (heatmap cell at {i}) — nothing to do")
                return

    if marker_cell is None:
        raise SystemExit("similarity_scan/top_matches cell not found — aborting")

    new_cell = nbformat.v4.new_code_cell(CELL_SOURCE)
    nb.cells.insert(marker_cell + 1, new_cell)
    nbformat.write(nb, PATH)
    print(f"inserted the heatmap cell after cell {marker_cell} ({PATH})")


if __name__ == "__main__":
    main()
