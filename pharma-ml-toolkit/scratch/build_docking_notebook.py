"""Rebuild ``examples/complete_docking_workflow.ipynb``.

Adds the two new campaign-entry sections requested for "The Complete
Docking Workflow (Zero to One Hundred)":

* §1 — import the ligand library from CSV (``load_ligand_csv`` +
  ``prepare_ligand_library``; the ``smilestopdbqt.py`` /
  ``ligand_preparation_autodock4.py`` upgrade),
* §2 — import & prepare the protein (``prepare_protein`` +
  ``distribute_receptor``; the missing receptor step + ``adder.py``),
* renumbers the previous §1–§9 to §3–§11, refreshes the title TOC and
  the recap, and re-executes the whole notebook with nbclient so all
  outputs are committed (Meeko + RDKit are real in this environment).

Run from the toolkit root:

    /home/z/.venv/bin/python scratch/build_docking_notebook.py
"""

from __future__ import annotations

import base64
import gzip
import os
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "complete_docking_workflow.ipynb"
SOURCE_PDB = ROOT / "examples" / "data" / "lysozyme_1aki.pdb"

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell


# ---------------------------------------------------------------------------
# New section cells
# ---------------------------------------------------------------------------


def ligand_section_cells() -> list:
    header = md(
        """\
## 1 · Import the ligand library — CSV in

The legacy workflow started from a `ligands.csv` with a hard-coded
`SMILES` column — three scripts in the rar archive did this job with
three slightly different behaviours (`smilestopdbqt.py`, `smiqt.py`,
`ligand_preparation_autodock4.py`). `load_ligand_csv` unifies them:

* the SMILES column is **auto-detected** (`SMILES`, `smiles`,
  `canonical_smiles`, ...); the ligand-name column too, falling back
  to row numbers,
* every SMILES is **syntax-checked with RDKit** before any 3D
  embedding happens, so a typo costs seconds, not a crashed batch,
* invalid rows are reported explicitly and exported to
  `failed_smiles.csv` — the one good idea in
  `ligand_preparation_autodock4.py`, kept.

`prepare_ligand_library` then turns the valid rows into the campaign
tree every downstream tool expects: **one folder per ligand, each
containing a prepared `l.pdbqt`** — exactly the layout the original
scripts created, and what `scan_docking_directory` audits in §6."""
    )

    write_csv = code(
        """\
# A demo ligand library: 8 real drug-like molecules plus two
# deliberately broken rows, so the validation accounting has something
# to report. Swap this cell for your own CSV -- everything below runs
# unchanged.
LIGANDS_CSV = WORK / "ligands.csv"

ligand_table = pd.DataFrame(
    {
        "name": [
            "aspirin", "ibuprofen", "paracetamol", "caffeine",
            "naproxen", "benzocaine", "coumarin", "vanillin",
            "broken_row", "empty_row",
        ],
        "SMILES": [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CC(=O)Nc1ccc(O)cc1",
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
            "CCOC(=O)c1ccc(N)cc1",
            "O=C1Oc2ccccc2C=C1",
            "COc1cc(C=O)ccc1O",
            "not_a_smiles",   # <- will be reported as invalid
            None,             # <- empty cell, also invalid
        ],
    }
)
ligand_table.to_csv(LIGANDS_CSV, index=False)
print(f"ligand library -> {LIGANDS_CSV}  ({len(ligand_table)} rows)")
ligand_table"""
    )

    load = code(
        """\
from pharmaml.docking import load_ligand_csv

library = load_ligand_csv(LIGANDS_CSV)
print(library)
library.frame"""
    )

    failures = code(
        """\
# The invalid rows, with the reason -- and the failed_smiles.csv export
# the legacy script wrote after every batch:
if len(library.invalid_frame):
    print(library.invalid_frame[["name", "smiles", "error"]].to_string(index=False))
    failed_path = library.write_failures(WORK / "failed_smiles.csv")
    print(f"\\ninvalid rows exported -> {failed_path.name}")
else:
    print("no invalid rows in this library")"""
    )

    batch = code(
        """\
from pharmaml.docking import prepare_ligand_library

LIGAND_FOLDERS = WORK / "csv_ligand_prep"   # <- your campaign directory

prep = prepare_ligand_library(library, LIGAND_FOLDERS)
print(prep)
prep.frame[["folder", "name", "ok", "atoms", "torsdof"]]"""
    )

    tree = code(
        """\
# The campaign tree (one folder per ligand, l.pdbqt inside) --
# scan_docking_directory-ready, and the layout the §6 demo campaign
# imitates:
for entry in sorted(LIGAND_FOLDERS.iterdir()):
    print(f"  {entry.relative_to(WORK)}" + ("/" if entry.is_dir() else ""))"""
    )

    return [header, write_csv, load, failures, batch, tree]


def protein_section_cells(blob: str) -> list:
    header = md(
        """\
## 2 · Import & prepare the protein — PDB in, receptor PDBQT out

The rar scripts assumed the hard part was already done: `gpfer.py`
hard-coded `receptor_types = "A C HD N NA OA SA"` and a ready-made
`r.pdbqt` in every ligand folder, prepared *outside* the codebase with
the retired AutoDockTools. `prepare_protein` brings that step into the
package:

* import a protein from a PDB file (or an already-prepared PDBQT
  receptor),
* clean it — waters, non-protein HETATM records (ions, buffers,
  glycans), alternate-location conformers, chain selection,
* write a rigid receptor PDBQT with **AutoDock4 atom types, Gasteiger
  partial charges and polar hydrogens** via Meeko's polymer pipeline,
* audit every atom that went in, came out, or was dropped.

Demo input: hen egg-white lysozyme (PDB entry
[1AKI](https://www.rcsb.org/1AKI), public domain) — *the* classic
docking-tutorial receptor — with its 78 crystallographic waters, plus a
sprinkling of the usual suspects a wet PDB contains (a chloride ion, a
sulfate, an alternate conformer) so every cleaning rule visibly fires.

`distribute_receptor` finally copies the prepared receptor into the §1
campaign folders — the job of the archive's `adder.py`, minus the
engine executables it also (needlessly) copied everywhere."""
    )

    write_pdb = code(
        f"""\
import base64
import gzip

# Lysozyme 1AKI atom records, embedded gz+base64 so this notebook
# re-runs anywhere offline (also shipped as
# examples/data/lysozyme_1aki.pdb). In practice you write this file
# from your own source: an RCSB download, a crystallography
# collaborator, an AlphaFold export, ...
_LYSOZYME_BLOB = (
    {blob}
)

PROTEIN_PDB = WORK / "lysozyme_wet.pdb"
LYSOZYME_PDB_TEXT = gzip.decompress(base64.b64decode(_LYSOZYME_BLOB)).decode()
PROTEIN_PDB.write_text(LYSOZYME_PDB_TEXT, encoding="utf-8")
n_atom = LYSOZYME_PDB_TEXT.count("ATOM")
n_het = LYSOZYME_PDB_TEXT.count("HETATM")
print(f"{{PROTEIN_PDB.name}}: {{n_atom}} ATOM + {{n_het}} HETATM records")"""
    )

    census = code(
        """\
from collections import Counter

atom_records = [l for l in PROTEIN_PDB.read_text().splitlines()
                if l.startswith(("ATOM", "HETATM"))]
hetatm_residues = Counter(
    l[17:20].strip() for l in atom_records if l.startswith("HETATM")
)
print(f"{sum(1 for l in atom_records if l.startswith('ATOM')):5d} ATOM   records  (the polymer)")
for res, count in sorted(hetatm_residues.items()):
    print(f"{count:5d} HETATM records  ({res})")"""
    )

    sprinkle = code(
        """\
def _hetatm(serial, name, resname, chain, resseq, x, y, z, element):
    return (
        f"HETATM{serial:5d} {name:>4s} {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}"
    )

# 1AKI is unusually clean, so we sprinkle in what a typical wet PDB
# carries: a chloride counter-ion, a sulfate from the crystallization
# buffer, and an alternate conformer (altloc B) of an existing atom.
lines = PROTEIN_PDB.read_text().splitlines()
wet = list(lines)
wet.append(_hetatm(2001, "CL", "CL", "A", 301, 20.0, 20.0, 20.0, "Cl"))
for i, (nm, x, y, z) in enumerate(
    [("S", 21, 21, 21), ("O1", 22, 21, 21), ("O2", 20, 21, 21),
     ("O3", 21, 22, 21), ("O4", 21, 20, 21)]
):
    element = "S" if nm == "S" else "O"
    wet.append(_hetatm(2002 + i, nm, "SO4", "A", 302, float(x), float(y), float(z), element))
clone = list(lines[10])          # duplicate one atom as alternate conformer B
clone[16] = "B"
wet.insert(11, "".join(clone))
PROTEIN_PDB.write_text("\\n".join(wet) + "\\nTER\\nEND\\n", encoding="utf-8")
print(f"doctored input: {len(wet)} records (was {len(lines)})")"""
    )

    prepare = code(
        """\
from pharmaml.docking import prepare_protein

RECEPTOR = WORK / "r.pdbqt"     # the filename gpfer.py's GPF lines expect
protein = prepare_protein(PROTEIN_PDB, RECEPTOR)
print(protein)"""
    )

    peek = code(
        """\
from pharmaml.docking import extract_pdbqt_atom_types

receptor_lines = RECEPTOR.read_text().splitlines()
print("\\n".join(receptor_lines[:4]))
print(f"... ({len(receptor_lines)} lines total)\\n")

# The seven types gpfer.py hard-coded ("A C HD N NA OA SA") -- now
# derived from the actual receptor instead of assumed:
print("receptor atom types:", extract_pdbqt_atom_types(RECEPTOR))
pd.Series(protein.atom_types).sort_values(ascending=False).to_frame("count")"""
    )

    distribute = code(
        """\
from pharmaml.docking import distribute_receptor

targets = distribute_receptor(RECEPTOR, LIGAND_FOLDERS)
print(f"r.pdbqt distributed into {len(targets)} ligand folders of §1's campaign"
      "  (adder.py's job, without the engine binaries)")"""
    )

    passthrough = code(
        """\
# Already have a prepared r.pdbqt from a collaborator or an older
# campaign? The same call re-filters it (waters, HETATM, altlocs)
# without any Meeko step -- charges and types pass through untouched:
passthrough = prepare_protein(RECEPTOR, WORK / "r_clean.pdbqt")
print(f"mode={passthrough.mode}, {passthrough.output_atoms} atoms, "
      f"charges kept: [{passthrough.charge_min:+.3f}, {passthrough.charge_max:+.3f}]")"""
    )

    return [
        header, write_pdb, census, sprinkle, prepare, peek, distribute, passthrough
    ]


NEW_TITLE = """\
# PharmaML — The Complete Docking Workflow (Zero to One Hundred)

One notebook, every stage of a molecular-docking campaign, in the order you
would actually run it:

| § | Stage | Toolkit |
|---|---|---|
| 1 | **Import the ligand library** — CSV in, validated library out | `pharmaml.docking` |
| 2 | **Import & prepare the protein** — PDB in, receptor PDBQT out | `pharmaml.docking` |
| 3 | Ligand preparation — SMILES → 3D PDBQT | `pharmaml.docking` |
| 4 | Grid box & GPF — enclose the binding site | `pharmaml.docking` |
| 5 | Run the engines — AutoDock4 / Vina (guarded) | `pharmaml.docking` |
| 6 | Parse AutoDock4 results — clusters, audit, campaign scan | `pharmaml.docking` |
| 7 | Parse Vina output — pose tables | `pharmaml.docking` |
| 8 | Rescoring — ligand efficiency + consensus | `pharmaml.docking` |
| 9 | Visualization — energy distribution, top ligands | `pharmaml.viz` |
| 10 | Publish — one-click HTML report + csv/md exports | `pharmaml.viz` |

**Setup** — the parsing/analysis path needs only the core package; ligand
**and** protein preparation additionally need RDKit + Meeko:

```bash
pip install -e ".[reports]"        # core + plotting
pip install rdkit meeko gemmi       # §1–§3 preparation
```

The demo campaign below is *synthetic but structurally identical* to real
AutoDock4 output, so every cell runs without a docking engine installed.
Point `DOCKING_DIR` at **your** campaign folder and re-run — the parsers,
rescoring and plots work unchanged on real `.dlg` trees.
"""

NEW_RECAP = """\
## 11 · Recap

| Stage | One-liner |
|---|---|
| Import ligand CSV | `load_ligand_csv(ligands.csv)` → validated `LigandLibrary` |
| Batch ligand prep | `prepare_ligand_library(library, out_dir)` → folder-per-ligand tree |
| Import & prep protein | `prepare_protein(protein.pdb, r.pdbqt)` → cleaned, typed, charged receptor |
| Distribute receptor | `distribute_receptor(r.pdbqt, campaign_root)` |
| Single ligand | `smiles_to_pdbqt(smiles, out_dir, name=...)` |
| Grid box | `compute_grid_box(reference_ligand.pdbqt, padding=4)` |
| GPF file | `write_gpf(ligand.pdbqt, box=box)` |
| Run AutoDock4 | `run_autodock4(ligand_dir)` / `batch_run_directories(base)` |
| Run Vina | `run_vina(receptor, ligand, out, center, size)` |
| One ligand | `parse_dlg_summary(r.dlg)` |
| Audit campaign | `check_docking_completion(docking_run)` |
| Scan campaign | `scan_docking_directory(docking_run)` |
| Vina poses | `parse_vina_output(out.pdbqt)` |
| Rescore | `ligand_efficiency_series(frame)` + `consensus_rescore(frame, cols)` |
| Visualize | `plot_energy_distribution(summary)` / `plot_top_ligands(summary)` |
| Publish | `save_visual_bundle(figures, tables, metrics, "reports")` |

CLI equivalents:

```bash
pharmaml docking prep-ligands --ligands ligands.csv --out ligand_folders/
pharmaml docking prep-protein --protein protein.pdb --out r.pdbqt
pharmaml docking scan      --directory docking_run/ --out summary.csv
pharmaml docking check     --directory docking_run/
```
"""


def lysozyme_blob() -> str:
    """gz+base64 of the 1AKI atom records, wrapped as implicit string
    concatenation chunks (readable + valid Python inside the cell)."""
    text = SOURCE_PDB.read_text(encoding="utf-8", errors="ignore")
    atoms = [l for l in text.splitlines() if l.startswith(("ATOM", "HETATM"))]
    payload = "\n".join(atoms) + "\n"
    blob = base64.b64encode(gzip.compress(payload.encode(), 9)).decode()
    width = 96
    wrapped = "\n    ".join(f'"{blob[i : i + width]}"' for i in range(0, len(blob), width))
    return wrapped


def renumber_sections(nb) -> None:
    """Shift the previous §1–§9 headers (and their cross-references) to §3–§11."""
    for number in range(9, 0, -1):  # reverse: 9→11 first, avoids double-shift
        old, new = f"## {number} · ", f"## {number + 2} · "
        for cell in nb.cells:
            if cell.cell_type == "markdown" and cell.source.startswith(old):
                cell.source = new + cell.source[len(old):]
    for cell in nb.cells:
        cell.source = cell.source.replace(
            "# The atom types Meeko wrote — these drive the affinity maps in §2.",
            "# The atom types Meeko wrote — these drive the affinity maps in §4.",
        )
        cell.source = cell.source.replace(
            "(In your project: skip this cell and\n# set DOCKING_DIR to your real campaign folder.)",
            "(In your project: skip this cell and\n# set DOCKING_DIR to your real campaign folder — e.g. the tree §1–§2\n# just built.)",
        )
        cell.source = cell.source.replace(
            "ligand atom type (extracted from the PDBQT itself).",
            "ligand atom type (extracted from the PDBQT itself).\n"
            "The `receptor` line points at `r.pdbqt` — the file §2 prepared and\n"
            "`distribute_receptor` copied into each ligand folder.",
        )


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    n_before = len(nb.cells)

    # 1. new title/TOC
    nb.cells[0].source = NEW_TITLE

    # 2. renumber the PREVIOUS sections (§1–§9 -> §3–§11) BEFORE
    #    inserting the new cells, so the new §1/§2 headers stay put
    renumber_sections(nb)

    # 3. insert the two new sections right after the setup cell
    blob = lysozyme_blob()
    new_cells = ligand_section_cells() + protein_section_cells(blob)
    nb.cells[2:2] = new_cells

    # 4. new recap (last cell)
    nb.cells[-1].source = NEW_RECAP

    # 5. execute the whole notebook with committed outputs
    #    (worker caps keep the sandbox's tight memory envelope safe)
    os.environ.update(
        {
            "LOKY_MAX_CPU_COUNT": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    client = NotebookClient(nb, timeout=900, kernel_name="python3")
    client.execute()

    nbformat.write(nb, NOTEBOOK)

    # 6. sanity report
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    errors = [
        out
        for c in code_cells
        for out in c.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    figures = sum(
        1
        for c in code_cells
        for out in c.get("outputs", [])
        if "image/png" in out.get("data", {})
    )
    print(f"cells: {len(nb.cells)} (was {n_before}) | code: {len(code_cells)}")
    print(f"error outputs: {len(errors)} | inline figures: {figures}")
    print(f"file size: {NOTEBOOK.stat().st_size / 1024:.0f} KB")
    if errors:
        for err in errors[:3]:
            print("ERROR:", err.get("ename"), str(err.get("evalue"))[:200])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
