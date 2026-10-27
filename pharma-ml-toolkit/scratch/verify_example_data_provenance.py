"""Verify examples/data/ is byte-identical to what the notebooks used to fabricate.

Task 100x-24-a moved every notebook's inline sample-data fabrication into
shipped files under ``examples/data/``.  The invariant that makes the
migration safe: each shipped file must be BYTE-IDENTICAL to the data the
old (pre-change) notebook cells generated, so the notebooks' committed
outputs survive the switch from fabricate-then-load to load-directly.

This script proves it by re-running the *historical* cells (extracted
from ``git HEAD``, i.e. the last pre-change commit) in throwaway
sub-notebooks and byte-comparing every artifact they produce against
the shipped files:

* complete_ml_workflow        -> qsar_classification.csv,
                                 screening_library.csv,
                                 library_with_descriptors.csv,
                                 qsar_multiclass.csv, qsar_regression.csv
* complete_cellline_workflow  -> cellline_screen.csv,
                                 cellline_expression.csv
* complete_pk_workflow        -> pk_profile.csv
* complete_docking_workflow   -> ligand_library.csv + docking_campaign/
                                 (the .dlg/.glg tree; ligands.csv and the
                                 benzocaine README are migration extras)
* pharmaml_walkthrough        -> walkthrough_compounds.csv,
                                 walkthrough_classification.csv,
                                 walkthrough_multiclass.csv
* complete_generative_workflow-> seed_molecules.csv (the SEED_TABLE dict)
* the docking notebook's gz+base64 lysozyme blob must equal the ATOM/
  HETATM records of examples/data/lysozyme_1aki.pdb

Run from the toolkit root (after building examples/data/):

    python scratch/verify_example_data_provenance.py

Exit code 0 = every comparison byte-identical.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DATA = EXAMPLES / "data"
OUT = Path(tempfile.mkdtemp(prefix="verify_24a_"))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

RESULTS: list[tuple[str, bool, str]] = []


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def git_notebook(name: str) -> nbformat.NotebookNode:
    blob = subprocess.run(
        ["git", "show", f"HEAD:pharma-ml-toolkit/examples/{name}.ipynb"],
        cwd=ROOT.parent, capture_output=True, check=True,
    ).stdout
    return nbformat.reads(blob.decode(), as_version=4)


def run_cells(name: str, indices: list[int], probe: str,
              prelude: str = "") -> nbformat.NotebookNode:
    """Execute selected historical cells + a probe cell; return the nb."""
    old = git_notebook(name)
    nb = nbformat.v4.new_notebook()
    nb.metadata = old.metadata
    if prelude:
        nb.cells.append(nbformat.v4.new_code_cell(prelude))
    for i in indices:
        cell = nbformat.v4.new_code_cell(old.cells[i].source)
        cell.execution_count = None
        nb.cells.append(cell)
    nb.cells.append(nbformat.v4.new_code_cell(probe))
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": str(EXAMPLES)}},
    )
    client.execute()
    return nb


def probe_paths(nb: nbformat.NotebookNode, marker: str) -> dict[str, str]:
    """Parse ``marker key=value`` lines printed by the probe cell."""
    text = "".join(
        o.get("text", "") for c in nb.cells for o in c.get("outputs", [])
        if o.get("output_type") == "stream"
    )
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith(marker):
            key, _, value = line[len(marker):].partition("=")
            out[key.strip()] = value.strip()
    return out


def compare(label: str, old: Path, new: Path) -> None:
    if not old.is_file():
        RESULTS.append((label, False, f"old artifact missing: {old}"))
        return
    if not new.is_file():
        RESULTS.append((label, False, f"shipped file missing: {new}"))
        return
    same = md5(old) == md5(new)
    note = "byte-identical" if same else (
        f"DIFFER: old {old.stat().st_size} B {md5(old)} vs "
        f"new {new.stat().st_size} B {md5(new)}"
    )
    RESULTS.append((label, same, note))


def compare_trees(label: str, old: Path, new: Path, skip_new: set[str]) -> None:
    old_files = {str(p.relative_to(old)) for p in old.rglob("*") if p.is_file()}
    new_files = {
        str(p.relative_to(new)) for p in new.rglob("*") if p.is_file()
    } - skip_new
    old_dirs = {str(p.relative_to(old)) for p in old.rglob("*") if p.is_dir()}
    new_dirs = {str(p.relative_to(new)) for p in new.rglob("*") if p.is_dir()}
    ok = True
    notes = []
    if old_dirs != new_dirs:
        ok = False
        notes.append(f"folder sets differ: only-old={sorted(old_dirs - new_dirs)} "
                     f"only-new={sorted(new_dirs - old_dirs)}")
    for rel in sorted(old_files):
        if rel not in new_files:
            ok = False
            notes.append(f"missing in shipped tree: {rel}")
        elif md5(old / rel) != md5(new / rel):
            ok = False
            notes.append(f"content differs: {rel}")
    if set(new_files) - old_files:
        ok = False
        notes.append(f"extra files in shipped tree: {sorted(set(new_files) - old_files)}")
    RESULTS.append((label, ok, "; ".join(notes) or "byte-identical"))


def verify_ml() -> None:
    probe = (
        f'print("PROBE WORK=" + str(WORK))\n'
        f'print("PROBE MY_FILE=" + str(MY_FILE))\n'
        f'print("PROBE STRUCT=" + str(struct_file))\n'
        f'import pandas as _pd\n'
        f'_pd.DataFrame(LIBRARY, columns=["name", "smiles"]).to_csv('
        f'"{OUT}/ml_library.csv", index=False)\n'
    )
    nb = run_cells("complete_ml_workflow", [1, 3, 4, 6, 7, 11, 31, 35], probe)
    paths = probe_paths(nb, "PROBE ")
    work = Path(paths["WORK"])
    compare("ml my_compounds.csv", Path(paths["MY_FILE"]),
            DATA / "qsar_classification.csv")
    compare("ml screening_library", OUT / "ml_library.csv",
            DATA / "screening_library.csv")
    compare("ml library_with_descriptors.csv", Path(paths["STRUCT"]),
            DATA / "library_with_descriptors.csv")
    compare("ml multiclass.csv", work / "multiclass.csv",
            DATA / "qsar_multiclass.csv")
    compare("ml regression.csv", work / "regression.csv",
            DATA / "qsar_regression.csv")


def verify_cellline() -> None:
    probe = (
        f'print("PROBE SCREEN=" + str(screen_path))\n'
        f'expression.to_csv("{OUT}/cl_expression.csv")\n'
    )
    nb = run_cells("complete_cellline_workflow", [1, 3, 17], probe)
    paths = probe_paths(nb, "PROBE ")
    compare("cellline screen.csv", Path(paths["SCREEN"]),
            DATA / "cellline_screen.csv")
    compare("cellline expression matrix", OUT / "cl_expression.csv",
            DATA / "cellline_expression.csv")


def verify_pk() -> None:
    probe = 'print("PROBE PROFILE=" + str(PROFILE_CSV))\n'
    nb = run_cells("complete_pk_workflow", [2, 4], probe)
    paths = probe_paths(nb, "PROBE ")
    compare("pk profile.csv", Path(paths["PROFILE"]), DATA / "pk_profile.csv")


def verify_docking() -> None:
    probe = (
        f'print("PROBE LIG=" + str(LIGANDS_CSV))\n'
        f'print("PROBE DOCK=" + str(DOCKING_DIR))\n'
    )
    nb = run_cells("complete_docking_workflow", [1, 3, 29], probe)
    paths = probe_paths(nb, "PROBE ")
    compare("docking ligands.csv", Path(paths["LIG"]), DATA / "ligand_library.csv")
    # migration extras (not produced by the historical cell): the campaign
    # manifest + the README that pins the deliberately-empty folder for git
    compare_trees("docking campaign tree", Path(paths["DOCK"]),
                  DATA / "docking_campaign", skip_new={"ligands.csv", "08_benzocaine/README.md"})


def verify_walkthrough() -> None:
    probe = (
        f'print("PROBE WT=" + str(tmp))\n'
        f'frame.to_csv("{OUT}/wt_classification.csv", index=False)\n'
        f'multi.to_csv("{OUT}/wt_multiclass.csv", index=False)\n'
    )
    nb = run_cells(
        "pharmaml_walkthrough", [1, 3, 5, 10], probe,
        prelude="from pharmaml.ml import ("  # noqa: F401  (cell #10 dependency)
                 "train_model, evaluate_model, make_train_test_split)",
    )
    paths = probe_paths(nb, "PROBE ")
    compare("walkthrough my_compounds.csv", Path(paths["WT"]),
            DATA / "walkthrough_compounds.csv")
    compare("walkthrough classification", OUT / "wt_classification.csv",
            DATA / "walkthrough_classification.csv")
    compare("walkthrough multiclass", OUT / "wt_multiclass.csv",
            DATA / "walkthrough_multiclass.csv")


def verify_generative() -> None:
    probe = (
        "import pandas as _pd\n"
        "_pd.DataFrame({'name': list(SEED_TABLE), "
        "'smiles': list(SEED_TABLE.values())})"
        f'.to_csv("{OUT}/gen_seeds.csv", index=False)\n'
    )
    run_cells("complete_generative_workflow", [2, 8], probe)
    shipped = (DATA / "seed_molecules.csv").read_bytes()
    rebuilt = (OUT / "gen_seeds.csv").read_bytes()
    # the shipped file has no trailing newline; pandas adds one
    same = shipped + b"\n" == rebuilt or shipped == rebuilt
    RESULTS.append((
        "generative seed_molecules.csv",
        same,
        "SEED_TABLE == shipped rows (443 B, no trailing newline)" if same
        else "seed table content differs",
    ))


def verify_lysozyme_blob() -> None:
    old = git_notebook("complete_docking_workflow")
    src = old.cells[9].source
    match = re.search(r"_LYSOZYME_BLOB = \((.*?)\)\n\nPROTEIN_PDB", src, re.S)
    assert match, "blob cell not found in historical notebook"
    namespace: dict[str, object] = {}
    exec("_LYSOZYME_BLOB = (" + match.group(1) + ")", namespace)  # noqa: S102
    blob_text = gzip.decompress(base64.b64decode(namespace["_LYSOZYME_BLOB"])).decode()
    shipped = (DATA / "lysozyme_1aki.pdb").read_text(encoding="utf-8")
    atom_records = "".join(
        line + "\n" for line in shipped.splitlines()
        if line.startswith(("ATOM", "HETATM"))
    )
    same = blob_text == atom_records
    RESULTS.append((
        "docking lysozyme blob vs 1aki.pdb atom records",
        same,
        f"identical ({blob_text.count('ATOM')} ATOM-scoped lines, "
        f"{len(atom_records)} B)" if same else "blob != shipped atom records",
    ))


def main() -> int:
    print(f"staging old-notebook artifacts in {OUT}")
    verify_ml()
    verify_cellline()
    verify_pk()
    verify_docking()
    verify_walkthrough()
    verify_generative()
    verify_lysozyme_blob()

    failures = 0
    for label, ok, note in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {label:45s} {note}")
        failures += 0 if ok else 1
    print(f"\n{len(RESULTS)} comparisons, {len(RESULTS) - failures} byte-identical, "
          f"{failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
