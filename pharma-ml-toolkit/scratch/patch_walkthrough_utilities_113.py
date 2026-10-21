#!/usr/bin/env python3
"""Append the "Data utilities" section to the walkthrough notebook.

Round 100x-113 — the notebook axis of the per-file coverage contract:
every rar workflow family already has a playground.py surface (the
100x-111 per-file audit) and the library suite pins the calls; the
notebooks documented their own axis separately in COVERAGE.md without
per-file evidence.  This patch closes the biggest notebook gaps with a
new walkthrough section touring the every-day chores on the shipped
demo data: converters, table chores, approved drugs, similarity scans,
docking config batches and Vina result folder walks.

Idempotent: re-running replaces the section instead of duplicating it.
Run from the toolkit root:  python scratch/patch_walkthrough_utilities_113.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "examples" / "pharmaml_walkthrough.ipynb"

MARKER = "## 9 · Data utilities"


def _md(source: str) -> Any:
    return nbformat.v4.new_markdown_cell(source)


def _code(source: str) -> Any:
    return nbformat.v4.new_code_cell(source)


CELLS: list[Any] = [
    _md("""\
## 9 · Data utilities — the every-day lab chores

The original toolkit's most-run scripts were plain chores: converting
CSV↔SDF↔`.smi`, merging per-ligand tables, splitting a huge file,
listing stems, filtering approved drugs, scanning Tanimoto similarity
and walking docking campaign folders.  The library productionised each
one — this section tours them on the shipped demo data (the same
surface the desktop playground's Utilities tab drives with a mouse)."""),
    _code("""\
from pharmaml.chem.converters import smi_to_csv, sdf_to_table, table_to_sdf

Path("artifacts").mkdir(exist_ok=True)

library = pd.read_csv(DATA_DIR / "ligand_library.csv")

# CSV table -> 3D SDF (ETKDG embed + UFF; every extra column an SD tag).
# Rows whose SMILES fail to parse are counted, never silently dropped.
report = table_to_sdf(library, "artifacts/library.sdf",
                      smiles_col="SMILES", name_col="name")
print(f"wrote {report.molecules} molecules "
      f"({report.embedded_3d} with 3D coordinates, "
      f"{report.failures} unparseable kept honest) -> {report.output_sdf}")

# ...and back again: SDF -> table (titles, canonical SMILES, typed tags)
round_trip = sdf_to_table("artifacts/library.sdf")
print(round_trip[["name", "SMILES"]].head(2).to_string(index=False))

# .smi two-column list -> CSV (the 'Convert SMI to CSV.py' chore)
Path("artifacts/list.smi").write_text("CCO ethanol\\nCC(C)CO isopropanol\\n")
smi_csv = smi_to_csv("artifacts/list.smi", "artifacts/list.csv")
print(pd.read_csv(smi_csv).to_string(index=False))"""),
    _code("""\
from pharmaml.chem.converters import split_table_into_chunks
from pharmaml.chem.tables import annotate_with_smiles, compare_tables, merge_tables
from pharmaml.ml.preprocessing import clean_numeric_frame

# row-wise merge of several tables, named columns first (CSV_merger.py)
merged = merge_tables(
    [pd.read_csv(DATA_DIR / "ligand_library.csv"),
     pd.read_csv(DATA_DIR / "seed_molecules.csv")],
    first_columns=["name", "smiles"],
)
print(f"merged {len(merged)} rows; first columns: {list(merged.columns)[:2]}")

# compare one ligand-name column across two tables (notcommon.py +
# common_between_two_excel.py -> one set reconciliation)
comparison = compare_tables(
    pd.read_csv(DATA_DIR / "ligand_library.csv"),
    pd.read_csv(DATA_DIR / "seed_molecules.csv"),
    column="name",
)
print("common:", len(comparison.common),
      "| only in A:", len(comparison.only_in_a),
      "| only in B:", len(comparison.only_in_b))

# annotate bare ligand names with SMILES from a lookup (smilesgetter.py)
annotated = annotate_with_smiles(
    pd.DataFrame({"ligand": ["aspirin", "naproxen", "unknown"]}),
    pd.read_csv(DATA_DIR / "ligand_library.csv"),
    name_columns=["ligand"],
    reference_name_column="name",
)
print(annotated.to_string(index=False))

# stream a large CSV into fixed-size parts (Split CSV file.py)
parts = split_table_into_chunks(
    DATA_DIR / "qsar_regression.csv", "artifacts/parts", chunk_rows=40,
)
print(f"split into {len(parts)} parts: "
      f"{[p.name for p in parts[:2]]} ...")

# drop non-numeric columns but keep the SMILES (Non-numerical remover.py)
cleaned = clean_numeric_frame(
    pd.read_csv(DATA_DIR / "walkthrough_classification.csv"),
    keep_cols=["SMILES"],
)
print("kept:", list(cleaned.columns)[:4], "...")"""),
    _code("""\
from pharmaml.chem.tables import merge_table_folder
from pharmaml.docking.campaign import list_file_names
from pharmaml.pharma.approvals import filter_approved_drugs

# list the stems of every csv in a folder (Nameextracter.py)
names = list_file_names(DATA_DIR, extension=".csv")
print(f"{len(names)} csv files in data/ | first 3: {names[:3]}")

# filter approved drugs from an approval table (approved.py)
approvals = pd.DataFrame({
    "drug": ["aspirin", "caffeine", "ibuprofen"],
    "regulatory approval": ["approved", "withdrawn", "approved"],
})
print("approved:", filter_approved_drugs(approvals))

# merge every part back into one table (filemerger.py) — split's twin
merged_back = merge_table_folder("artifacts/parts")
print(f"merged back {len(merged_back)} rows from the {len(parts)} parts")"""),
    _code("""\
from pharmaml.chem.similarity import similarity_scan, top_matches

references = ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
library_smiles = pd.read_csv(DATA_DIR / "ligand_library.csv")["SMILES"].dropna().tolist()

# threshold scan: reference x library, long-format pairs (Tanimoto calculator.py)
pairs = similarity_scan(references, library_smiles, threshold=0.4)
print(pairs.sort_values("similarity", ascending=False).head(4).to_string(index=False))

# top-k nearest neighbours of one query (Tanimoto-edited.py)
nearest = top_matches("CC(=O)Oc1ccccc1C(=O)O", library_smiles, k=3)
print(nearest.to_string(index=False))"""),
    _md("""\
### Docking campaign chores

The campaign side of the same coin: read a GPF back as typed
parameters, batch-write the `r.gpf`/`r.dpf` pair into every ligand
folder of a campaign tree (custom receptor atom types included — the
Ni-carrying `gpferreceptorni.py` case), and walk a results folder for
per-log best affinities."""),
    _code("""\
from pharmaml.docking.campaign import write_config_batch
from pharmaml.docking.preparation import read_gpf

# read one GPF back as typed parameters (gpfreader.py)
Path("artifacts/r.gpf").write_text(
    "ligand_types C HD OA\\nreceptor r.pdbqt\\nreceptor_types A C HD N OA\\n"
    "npts 40 40 40\\nspacing 0.375\\ngridcenter 0 0 0\\n"
    "smooth 0.5\\ndielectric -0.1465\\n"
)
params = read_gpf("artifacts/r.gpf")
print("ligand types:", params["ligand_types"],
      "| npts:", params["npts"], "| spacing:", params["spacing"])

# batch-write r.gpf + r.dpf per campaign folder (gpfer.py / dpfer.py /
# gpferreceptorni.py — the Ni receptor types are the point here)
LIGAND_PDBQT = (
    "REMARK  Name = demo\\nROOT\\n"
    "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     C\\n"
    "ATOM      2  O   LIG A   1       2.000   0.000   0.000  1.00  0.00     OA\\n"
    "ENDROOT\\nTORSDOF 1\\n"
)
campaign = Path("artifacts/campaign")
for name in ("01_demo", "02_empty"):
    (campaign / name).mkdir(parents=True, exist_ok=True)
(campaign / "01_demo" / "l.pdbqt").write_text(LIGAND_PDBQT)
batch = write_config_batch(campaign, receptor_types="A C HD N Ni OA SA")
print(batch[["folder", "centre", "status"]].to_string(index=False))"""),
    _code("""\
from pharmaml.docking.parsing import scan_vina_directory
from pharmaml.docking.workflow import run_vina_batch
from pharmaml.exceptions import DockingError

# a folder walk over Vina logs (vina_reader.py): best affinity per log,
# strongest binder first
vina_root = Path("artifacts/vina_results")
vina_root.mkdir(exist_ok=True)
for name, best in (("lig01", "-7.1"), ("lig02", "-8.4")):
    (vina_root / f"{name}.log").write_text(
        f"Estimated Free Energy of Binding   = {best} (kcal/mol)\\n"
        f"REMARK VINA RESULT:    {best:<10} 0.000      0.000\\n"
    )
scan = scan_vina_directory(vina_root)
print(scan.to_string(index=False))

# batch-docking every ligand with one config (vinar_multiple.py) —
# gates the engine BEFORE queue time; no vina binary here, so the
# actionable error fires instead of a cryptic crash:
Path("artifacts/vina.conf").write_text(
    "receptor = r.pdbqt\\ncenter_x = 0\\ncenter_y = 0\\ncenter_z = 0\\n"
    "size_x = 20\\nsize_y = 20\\nsize_z = 20\\n")
Path("artifacts/r.pdbqt").write_text("ATOM receptor\\n")
try:
    run_vina_batch("artifacts/vina.conf", "artifacts/campaign",
                   "artifacts/vina_results")
except DockingError as exc:
    print(f"[run_vina_batch] {exc}")"""),
]


def main() -> int:
    nb = nbformat.read(NB_PATH, as_version=4)

    # idempotency: drop a previous copy of the section, keep everything else
    keep = []
    inside = False
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.startswith(MARKER):
            inside = True
            continue
        if inside and cell.cell_type == "markdown" and cell.source.startswith("## "):
            inside = False
        if not inside:
            keep.append(cell)
    if inside:  # section ran to the end of the notebook
        pass
    nb.cells = keep

    nb.cells.extend(CELLS)  # already proper v4 cell nodes

    nbformat.write(nb, NB_PATH)
    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"walkthrough now has {len(nb.cells)} cells "
          f"({code_cells} code); section 9 appended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
