"""Build + execute examples/complete_docking_workflow.ipynb with nbclient.

The notebook is the docking "zero-to-one-hundred" artifact: every stage of
a structure-based virtual-screening campaign in the order you would
actually run it — ligand preparation, grid setup, running the engines,
parsing results, rescoring, ranking, visualization and the HTML report.

The demo campaign tree is *synthetic but structurally identical* to real
AutoDock4 output (same cluster-table format the parsers expect), so every
cell runs end-to-end without a docking engine installed.

Run:  PYTHONPATH=src python scratch/build_docking_notebook.py
"""
import nbformat
from nbclient import NotebookClient

# ---------------------------------------------------------------- markdown
MD_HEADER = """\
# PharmaML — The Complete Docking Workflow (Zero to One Hundred)

One notebook, every stage of a molecular-docking campaign, in the order you
would actually run it:

| § | Stage | Toolkit |
|---|---|---|
| 1 | **Ligand preparation** — SMILES → 3D PDBQT | `pharmaml.docking` |
| 2 | Grid box & GPF — enclose the binding site | `pharmaml.docking` |
| 3 | Run the engines — AutoDock4 / Vina (guarded) | `pharmaml.docking` |
| 4 | Parse AutoDock4 results — clusters, audit, campaign scan | `pharmaml.docking` |
| 5 | Parse Vina output — pose tables | `pharmaml.docking` |
| 6 | Rescoring — ligand efficiency + consensus | `pharmaml.docking` |
| 7 | Visualization — energy distribution, top ligands | `pharmaml.viz` |
| 8 | Publish — one-click HTML report + csv/md exports | `pharmaml.viz` |

**Setup** — the parsing/analysis path needs only the core package; ligand
preparation additionally needs RDKit + Meeko:

```bash
pip install -e ".[reports]"        # core + plotting
pip install rdkit-pypi meeko       # §1 ligand preparation
```

The demo campaign below is *synthetic but structurally identical* to real
AutoDock4 output, so every cell runs without a docking engine installed.
Point `DOCKING_DIR` at **your** campaign folder and re-run — the parsers,
rescoring and plots work unchanged on real `.dlg` trees.
"""

MD_1 = """\
## 1 · Ligand preparation — SMILES → 3D PDBQT

The legacy `smilestopdbqt.py` embedded paths and mutated global state;
`smiles_to_pdbqt` embeds each molecule with ETKDG, MMFF-optimises it, writes
the PDBQT through Meeko, and repairs any non-AD4 atom types — all with
explicit output directories.
"""

MD_2 = """\
## 2 · Grid box & GPF — enclose the binding site

`compute_grid_box` derives the search box from a co-crystallised (reference)
ligand plus padding — the calculation `gpfer.py` did by hand. `write_gpf`
then emits one AutoDock4 grid parameter file with an affinity-map line per
ligand atom type (extracted from the PDBQT itself).
"""

MD_3 = """\
## 3 · Run the engines — AutoDock4 / Vina

`run_autodock4`, `run_vina` and `batch_run_directories` wrap the engines with
argument-list subprocess calls (no `shell=True`, no `os.chdir`), availability
checks *before* hours of queue time, and per-run timings.  Neither engine is
installed in this demo environment — the cell below shows the *actionable*
error you get instead of a cryptic crash:

> In a production environment (AutoDock Tools / Vina on `PATH`) the same
> calls return a `DockingRunResult` with timings and output paths.
"""

MD_4 = """\
## 4 · Parse AutoDock4 results — clusters, audit, campaign scan

The parsers accept **text or paths** and return typed dataclasses /
DataFrames, so they are unit-testable without a docking engine — and so this
notebook can build a demo campaign whose `.dlg` files are byte-compatible
with the real format:

* `parse_dlg_cluster_table` — one ligand's cluster table (the
  `readerwithdetails.py` regex, now typed);
* `parse_dlg_summary` — best (lowest-energy) *and* most-populated cluster;
* `check_docking_completion` — the `checker.py` audit: every folder needs
  its `r.dlg` **and** the `Successful Completion` sentinel;
* `scan_docking_directory` — the whole campaign in one table, with explicit
  status columns for failed folders instead of a crash.

The demo tree includes one folder with **no log at all** and one
**interrupted** run, so the audit columns are exercised.  Finally,
`batch_run_directories` re-runs the whole campaign *idempotently*: folders
whose `.dlg` already exists are skipped, and failures are collected as
results (not raised) — the difference between a queued night job that
finishes and one that dies at folder 3.
"""

MD_5 = """\
## 5 · Parse Vina output — pose tables

`parse_vina_output` reads either the stdout mode table or the
`REMARK VINA RESULT` lines of an output PDBQT, and returns typed poses.
"""

MD_6 = """\
## 6 · Rescoring — ligand efficiency + consensus

Raw binding energy favours big molecules; `ligand_efficiency` normalises by
heavy-atom count (Hopkins 2004). `consensus_rescore` then min-max normalises
several evidence columns — energy (lower better), cluster population (higher
better), ligand efficiency — into one 0–1 score.  Heavy-atom counts are
computed from the actual SMILES with RDKit.
"""

MD_7 = """\
## 7 · Visualization — energy distribution & top ligands

Two figures the legacy archive never drew: the campaign-wide energy
histogram (with median and best ligand marked) and the ranked top-ligand
bars (strongest binding on top).
"""

MD_8 = """\
## 8 · Publish — the one-click visual bundle

`save_visual_bundle` renders the figures and tables into a directory **and**
a self-contained `report.html` (figures embedded as base64 — no external
assets).  `export_table` writes any table as csv/tsv/markdown/json/html.
"""

MD_9 = """\
## 9 · Recap

| Stage | One-liner |
|---|---|
| Prepare ligands | `smiles_to_pdbqt(smiles, out_dir, name=...)` |
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
pharmaml docking scan  --directory docking_run/ --out summary.csv
pharmaml docking check --directory docking_run/
```
"""

# ---------------------------------------------------------------- code
CODE_SETUP = """\
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pharmaml import __version__

pd.set_option("display.max_columns", 14)
pd.set_option("display.width", 120)

# Silence the fork() DeprecationWarning the process pool emits on this
# Python build (harmless here, noisy in committed outputs).
warnings.filterwarnings("ignore", message=".*use of fork.*")

# Scratch directory for this run's artifacts (use any path you like).
WORK = Path(tempfile.mkdtemp(prefix="docking_workflow_"))

# RDKit's C++ parser prints errors for invalid SMILES; silence the noise
# (the pharmaml layer reports invalid rows explicitly).
try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

print(f"pharmaml {__version__}")
print(f"artifacts -> {WORK}")"""

# §1 ------------------------------------------------------------------
CODE_1A = """\
from pharmaml.docking import smiles_to_pdbqt, extract_pdbqt_atom_types

LIGAND_DIR = WORK / "ligands"

# Three demo ligands (in your project: loop over your screening library).
for name, smi in [
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]:
    path = smiles_to_pdbqt(smi, LIGAND_DIR, name=name)
    print(f"prepared {name:12s} -> {path.name}  ({path.stat().st_size} bytes)")"""

CODE_1B = """\
# The atom types Meeko wrote — these drive the affinity maps in §2.
for pdbqt in sorted(LIGAND_DIR.glob("*.pdbqt")):
    print(f"{pdbqt.stem:12s}: {extract_pdbqt_atom_types(pdbqt)}")"""

CODE_1C = """\
# What a prepared ligand looks like (torsion tree + partial charges).
print((LIGAND_DIR / "aspirin.pdbqt").read_text()[:400])"""

# §2 ------------------------------------------------------------------
CODE_2A = """\
from pharmaml.docking import GridBox, compute_grid_box

# In a real project this is the CO-CRYSTALLISED ligand carved from the
# receptor PDB; here the aspirin we just prepared plays that role.
reference = LIGAND_DIR / "aspirin.pdbqt"

box = compute_grid_box(reference, padding=4.0)
print(f"centre : ({box.center_x:.2f}, {box.center_y:.2f}, {box.center_z:.2f}) A")
print(f"size   : {tuple(round(s, 1) for s in box.size)} A  "
      f"(npts {box.npts} @ {box.spacing} A spacing)")
print(f"vina   : {box.as_vina_kwargs()}")"""

CODE_2B = """\
from pharmaml.docking import write_gpf

gpf = write_gpf(reference, receptor_file="r.pdbqt", box=box,
                output_path=WORK / "r.gpf")
print(gpf.read_text())"""

# §3 ------------------------------------------------------------------
CODE_3 = """\
from pharmaml.docking import run_autodock4, run_vina
from pharmaml.exceptions import DockingError

# AutoDock4 needs autogrid4/autodock4 on PATH; Vina needs `vina`.
for label, call in [
    ("run_vina", lambda: run_vina(
        receptor="r.pdbqt", ligand=reference, out=WORK / "aspirin_out.pdbqt",
        center=(box.center_x, box.center_y, box.center_z),
        size=box.size, seed=42,
    )),
    ("run_autodock4", lambda: run_autodock4(WORK)),   # r.gpf exists -> binary check fires
]:
    try:
        call()
    except DockingError as exc:
        print(f"[{label}] {exc}")"""

# §4 ------------------------------------------------------------------
CODE_4A = """\
# --- DEMO CAMPAIGN ------------------------------------------------------
# A synthetic AutoDock4 campaign: one folder per ligand, each with an
# r.dlg whose cluster table uses the exact real format.  One folder is
# missing its log entirely and one was interrupted before clustering, so
# the audit has something to find.  (In your project: skip this cell and
# set DOCKING_DIR to your real campaign folder.)
def make_dlg(seed: int, n_clusters: int = 5, total_runs: int = 50) -> str:
    rng = np.random.default_rng(seed)
    lowest = np.sort(rng.uniform(-11.8, -4.5, size=n_clusters))[::-1]
    sizes = rng.multinomial(total_runs, [1 / n_clusters] * n_clusters) + 1
    mean = lowest + rng.uniform(0.05, 0.6, size=n_clusters)   # mean sits above the lowest
    lines = [
        "AutoDock4 log file (demo)",
        "",
        "________",
        "CLUSTERING HISTOGRAM",
        "__________________",
        "______________________________________________________________________",
        "Cluster Rank | Lowest Energy | Run | Mean Energy | # in Cluster",
        "_____________|_______________|_____|_____________|________________",
    ]
    run = int(rng.integers(1, total_runs))
    for rank in range(n_clusters):
        lines.append(
            f"     {rank + 1}       |     {lowest[rank]:7.2f}    |  {run:3d} |"
            f"    {mean[rank]:7.2f}    |   {sizes[rank]:3d} |"
        )
    lines += [
        "____________|_______________|_____|_____________|________________",
        "",
        "Successful Completion",
    ]
    return "\\n".join(lines) + "\\n"


CAMPAIGN_LIGANDS = [
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O"),
    ("metformin", "CN(C)C(=N)N=C(N)N"),
    ("benzocaine", "CCOC(=O)c1ccc(N)cc1"),
    ("propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
    ("nicotine", "CN1CCCC1c1cccnc1"),
    ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
    ("lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("coumarin", "O=C1Oc2ccccc2C=C1"),
    ("vanillin", "COc1cc(C=O)ccc1O"),
]

DOCKING_DIR = WORK / "docking_run"     # <- replace with your campaign folder
for i, (name, _) in enumerate(CAMPAIGN_LIGANDS, start=1):
    folder = DOCKING_DIR / f"{i:02d}_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    if name == "benzocaine":
        continue                          # no r.dlg written -> MISSING_LOG
    if name == "nicotine":
        (folder / "r.dlg").write_text(
            "AutoDock4 log file (demo)\\n\\n... run interrupted before clustering\\n",
            encoding="utf-8",
        )
        (folder / "r.glg").write_text("autogrid4 log (demo)\\n", encoding="utf-8")
        continue                          # no cluster table -> PARSE_ERROR
    (folder / "r.dlg").write_text(make_dlg(seed=100 + i), encoding="utf-8")
    (folder / "r.glg").write_text("autogrid4 log (demo)\\n", encoding="utf-8")

print(f"demo campaign: {DOCKING_DIR}")
print(f"{len(list(DOCKING_DIR.iterdir()))} ligand folders "
      f"(12 OK, 1 missing log, 1 interrupted)")"""

CODE_4B = """\
from pharmaml.docking import parse_dlg_cluster_table

clusters = parse_dlg_cluster_table(DOCKING_DIR / "01_aspirin" / "r.dlg")
pd.DataFrame([asdict(c) for c in clusters])"""

CODE_4C = """\
from pharmaml.docking import parse_dlg_summary

summary_one = parse_dlg_summary(DOCKING_DIR / "01_aspirin" / "r.dlg")
for key, value in summary_one.items():
    print(f"{key:22s}: {value}")
print("\\n'top_' = lowest energy overall; 'best_' = most populated cluster")"""

CODE_4D = """\
from pharmaml.docking import check_docking_completion

audit = check_docking_completion(DOCKING_DIR)
print(audit["status"].value_counts().to_dict(), "\\n")
audit"""

CODE_4E = """\
from pharmaml.docking import scan_docking_directory

summary = scan_docking_directory(DOCKING_DIR)
summary.round(2)"""

CODE_4F = """\
import warnings

from pharmaml.docking import batch_run_directories

# IPython resets warning filters per cell, so scope the fork() noise here.
warnings.filterwarnings("ignore", message=".*use of fork.*")

# Idempotent campaign re-run: folders whose outputs (.glg/.dlg) already
# exist are skipped, per-folder failures are collected (benzocaine has no
# files at all), and the interrupted run is *not* re-docked because its
# .dlg exists -- which is exactly why the completion audit above exists
# as a separate check.
for r in batch_run_directories(DOCKING_DIR, workers=2):
    status = "ok" if r.ok else f"FAILED: {r.error}"
    print(f"[batch] {r.directory.name:16s} -> {status}")"""

# §5 ------------------------------------------------------------------
CODE_5 = """\
from pharmaml.docking import parse_vina_output

# REMARK lines as written into a real Vina output PDBQT.
vina_pdbqt = "\\n".join([
    "REMARK VINA RESULT:   -10.3      0.000      0.000",
    "REMARK VINA RESULT:    -9.6      1.842      2.301",
    "REMARK VINA RESULT:    -9.4      2.105      2.777",
    "REMARK VINA RESULT:    -8.9      2.733      3.418",
    "REMARK VINA RESULT:    -8.1      4.006      5.114",
])

poses = parse_vina_output(vina_pdbqt)
pd.DataFrame([asdict(p) for p in poses])"""

# §6 ------------------------------------------------------------------
CODE_6A = """\
from pharmaml.docking import ligand_efficiency, ligand_efficiency_series

# Rank only the ligands that actually completed.
ok = summary[summary["status"] == "OK"].copy()
ok["ligand"] = ok["folder"].str.split("_").str[1]

# Heavy-atom counts from the actual SMILES (RDKit).
from rdkit import Chem

smiles_map = dict(CAMPAIGN_LIGANDS)
ok["num_heavy_atoms"] = [
    Chem.MolFromSmiles(smiles_map[name]).GetNumHeavyAtoms() for name in ok["ligand"]
]

ok["ligand_efficiency"] = ligand_efficiency_series(ok)
ok[["folder", "best_lowest_energy", "num_heavy_atoms", "ligand_efficiency"]].round(3)"""

CODE_6B = """\
from pharmaml.docking import consensus_rescore

# Energy (lower better) + cluster population (higher better) + LE (higher
# better) -> one 0-1 consensus; direction defaults handle all three.
ok["consensus"] = consensus_rescore(
    ok, columns=["best_lowest_energy", "best_num_in_cluster", "ligand_efficiency"]
)

ranked = ok.sort_values("consensus", ascending=False).reset_index(drop=True)
ranked[["folder", "best_lowest_energy", "best_num_in_cluster",
        "ligand_efficiency", "consensus"]].round(3).head(8)"""

# §7 ------------------------------------------------------------------
CODE_7A = """\
import warnings

from pharmaml.viz import plot_energy_distribution

# Version-compatibility noise from matplotlib/pyparsing in this env.
warnings.filterwarnings("ignore", category=DeprecationWarning)

fig_energy = plot_energy_distribution(
    ok, title="Campaign binding energies (best cluster per ligand)"
)"""

CODE_7B = """\
from pharmaml.viz import plot_top_ligands

fig_top = plot_top_ligands(ok, top_n=10, title="Top 10 ligands by binding energy")"""

# §8 ------------------------------------------------------------------
CODE_8A = """\
from pharmaml.viz import save_visual_bundle

campaign_metrics = {
    "ligands docked": int(len(ok)),
    "failed folders": int((summary["status"] != "OK").sum()),
    "best energy (kcal/mol)": float(ok["best_lowest_energy"].min()),
    "best ligand": str(ok.loc[ok["best_lowest_energy"].idxmin(), "folder"]),
    "median energy (kcal/mol)": float(ok["best_lowest_energy"].median()),
    "mean ligand efficiency": round(float(ok["ligand_efficiency"].mean()), 3),
}

BUNDLE = WORK / "docking_reports"
artifacts = save_visual_bundle(
    figures={"energy_distribution": fig_energy, "top_ligands": fig_top},
    tables={
        "docking_summary": ok[["folder", "n_clusters", "best_lowest_energy",
                               "best_mean_energy", "best_num_in_cluster",
                               "num_heavy_atoms", "ligand_efficiency",
                               "consensus"]],
        "completion_audit": audit,
        "vina_poses": pd.DataFrame([asdict(p) for p in poses]),
    },
    metrics=campaign_metrics,
    out_dir=BUNDLE,
    title="Docking campaign report",
    subtitle=f"{len(ok)} ligands · AutoDock4 demo campaign",
    formats=("png", "svg"),
    close=False,   # keep figures open so they also render above
)
pd.DataFrame({"artifact": [p.name for p in artifacts],
              "kind": [p.suffix.lstrip(".") for p in artifacts]})"""

CODE_8B = """\
from pharmaml.viz import export_table

exports = [
    export_table(ok, BUNDLE / "docking_summary.md"),
    export_table(ranked, BUNDLE / "consensus_ranking.csv"),
    export_table(audit, BUNDLE / "completion_audit.json"),
]
report_html = BUNDLE / "report.html"
print(f"{report_html}  ({report_html.stat().st_size / 1024:.0f} KB, zero external assets)")
[p.name for p in exports]"""

# ---------------------------------------------------------------- build
cells = [nbformat.v4.new_markdown_cell(MD_HEADER),
         nbformat.v4.new_code_cell(CODE_SETUP)]
for md, code_blocks in (
    (MD_1, [CODE_1A, CODE_1B, CODE_1C]),
    (MD_2, [CODE_2A, CODE_2B]),
    (MD_3, [CODE_3]),
    (MD_4, [CODE_4A, CODE_4B, CODE_4C, CODE_4D, CODE_4E, CODE_4F]),
    (MD_5, [CODE_5]),
    (MD_6, [CODE_6A, CODE_6B]),
    (MD_7, [CODE_7A, CODE_7B]),
    (MD_8, [CODE_8A, CODE_8B]),
    (MD_9, []),
):
    cells.append(nbformat.v4.new_markdown_cell(md))
    cells.extend(nbformat.v4.new_code_cell(code) for code in code_blocks)

nb = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbformat.validate(nb)

import os

client = NotebookClient(
    nb,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": "/home/z/my-project/pharma-ml-toolkit/examples"}},
    env={**os.environ, "PYTHONPATH": "/home/z/my-project/pharma-ml-toolkit/src",
         # cap BLAS/loky workers: the build sandbox is memory-constrained
         "LOKY_MAX_CPU_COUNT": "1", "OMP_NUM_THREADS": "1",
         "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
)
client.execute()

out = "/home/z/my-project/pharma-ml-toolkit/examples/complete_docking_workflow.ipynb"
nbformat.write(nb, out)

n_code = sum(1 for c in nb.cells if c.cell_type == "code")
n_img = sum(
    1 for c in nb.cells if c.cell_type == "code"
    for o in c.get("outputs", [])
    if o.get("data", {}).get("image/png")
)
print(f"written: {out} | cells: {len(nb.cells)} ({n_code} code) | inline figures: {n_img}")
