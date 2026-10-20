"""26-R7 driver: notebooks read the shipped examples/data/ inventory.

The user's standing requirement (round 24): "Jupyter files must work
entirely based on the files I give them and not create their own sample
data."  A sandbox reset rolled the six executed notebooks back to their
inline-fabrication versions; this script re-does the conversion in one
reproducible pass (lineage: scratch/reexecute_cellline_notebook.py +
the scratch/patch_*_notebook.py family):

1. EDIT — every inline data-fabrication cell is replaced by a load of the
   corresponding shipped file (built byte-identically by
   scratch/build_example_data.py, whose provenance map names the exact
   cell each file came from).  Surrounding prose is touched minimally so
   the zero-to-one-hundred narrative stays coherent.
2. GUARD — the nine banned fabrication tokens must leave the notebooks
   entirely (the same tokens tests/test_notebook_variants.py pins; each
   one verifiably hit a real fabrication cell pre-conversion — see the
   evidence block at the bottom of this docstring).
3. EXECUTE — all six notebooks re-run end-to-end with nbclient under the
   worker-cap env (LOKY/OMP/MKL/OPENBLAS = 1), cwd = examples/ so the
   relative ``DATA_DIR = Path("data")`` resolves to examples/data/.
4. TWINS — the five ``*_no_output.ipynb`` variants are regenerated via
   scratch/strip_notebook_outputs.py so they stay code-synced.

Conversion map (notebook -> cell -> shipped file; cells counted over ALL
cells, markdown included):

  complete_ml_workflow.ipynb
      cell 3  -> data/qsar_classification.csv       (420 x 11)
      cell 6  -> data/screening_library.csv         (79 rows: 77 valid + 2 planted invalid)
      cell 11 -> data/library_with_descriptors.csv  (77 x 10)
      cell 31 -> data/qsar_multiclass.csv           (360 x 21)
      cell 35 -> data/qsar_regression.csv           (260 x 21, pIC50)
  complete_cellline_workflow.ipynb
      cell 3  -> data/cellline_screen.csv           (185 rows: 23 lines / 11 tissues)
      cell 17 -> data/cellline_expression.csv       (23 x 300)
  complete_pk_workflow.ipynb
      cell 4  -> data/pk_profile.csv                (49 samples; AUC lin 17.9607 / log 17.2536)
  complete_docking_workflow.ipynb
      cell 3  -> data/ligand_library.csv            (10 rows: 8 valid + 2 planted invalid)
      cell 9  -> data/lysozyme_1aki.pdb             (RCSB 1AKI; atom-record subset, 1001 + 78)
      cell 29 -> data/docking_campaign/             (14 folders: 12 OK + 2 audit cases;
                   copied into WORK first -- the §7 housekeeping verbs mutate
                   the campaign root, so the shipped tree stays pristine)
  complete_generative_workflow.ipynb
      cell 8  -> data/seed_molecules.csv            (12 rows, 443 B, no trailing newline)
  pharmaml_walkthrough.ipynb
      cell 3  -> data/walkthrough_compounds.csv     (180 x 8)
      cell 5  -> data/walkthrough_classification.csv (300 x 9)
      cell 10 -> data/walkthrough_multiclass.csv    (300 x 9, 3 classes)

Pre-conversion token evidence (verified against git HEAD on 2026-10-16,
the round-23 committed inline-data state -- 22 cell-hits):

  'from sklearn.datasets import make_classification'  walkthrough 5, 10
  'np.random.default_rng('                            ml 3, 11 · cellline 1 · pk 4 · docking 29 · walkthrough 3
  'rng.normal('                                       ml 3, 11 · cellline 3, 17 · pk 4 · walkthrough 3
  'rng.choice('                                       ml 3 · walkthrough 3
  'LIBRARY = ['                                       ml 6
  '"SMILES": ['                                       docking 3
  'CAMPAIGN_LIGANDS = ['                              docking 29
  'SEED_TABLE = {'                                    generative 8
  'from make_synthetic_qsar_data import'              ml 31, 35

Expected output deltas after re-execution (package code moved on since
the outputs were committed — re-execution refreshes them honestly):
  * PK §3 AUC log 17.2347 -> ~17.2536 (the round-20a log-trapezoid fix);
  * docking §4 grid box size (22.5, 22.5, 22.5) -> (22.1, 22.1, 22.1)
    and vina size 22.5 -> 22.125 (the fencepost correction
    (npts - 1) * spacing);
  * docking §6 batch re-run: 10_nicotine now FAILED (missing r.dpf) —
    the sentinel-trust rule no longer trusts interrupted logs;
  * generative §7 RL numbers (running-baseline algorithm: early stopped
    True -> False, 107 unique molecules, best score 0.601, leaderboard
    1975 -> 1968 rows) and any GA leaderboard values the round-26
    generative fixes moved.

Run from the toolkit root:

    python scratch/reexecute_notebooks_26.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Single-thread determinism for the kernel subprocesses (and for anything
# imported below).  Must be set before numpy/sklearn/touch any BLAS.
for _var in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

NOTEBOOKS = (
    "complete_ml_workflow.ipynb",
    "complete_cellline_workflow.ipynb",
    "complete_pk_workflow.ipynb",
    "complete_docking_workflow.ipynb",
    "complete_generative_workflow.ipynb",
    "pharmaml_walkthrough.ipynb",
)

# The no-inline-fabrication contract (mirrors tests/test_notebook_variants.py).
BANNED_TOKENS = (
    "from sklearn.datasets import make_classification",
    "np.random.default_rng(",
    "rng.normal(",
    "rng.choice(",
    "LIBRARY = [",
    '"SMILES": [',
    "CAMPAIGN_LIGANDS = [",
    "SEED_TABLE = {",
    "from make_synthetic_qsar_data import",
)

DATA_COMMENT = (
    "# The shipped demo inventory (examples/data) -- this notebook loads\n"
    "# its demo tables from there instead of fabricating them inline.\n"
    'DATA_DIR = Path("data")\n'
)


# ---------------------------------------------------------------------------
# The edits: (notebook, cell index, anchor, replacement source, done-marker)
# ---------------------------------------------------------------------------
ML_SETUP = f'''import sys
import tempfile
from pathlib import Path

import pandas as pd

from pharmaml import __version__

pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 110)

# Scratch directory for this run's artifacts (use any path you like).
WORK = Path(tempfile.mkdtemp(prefix="ml_workflow_"))

{DATA_COMMENT}
print(f"pharmaml {{__version__}}")
print(f"artifacts -> {{WORK}}")
'''

ML_CELL3 = '''# --- DEMO DATA ---------------------------------------------------------
# A realistic QSAR table shipped with the toolkit: 8 descriptors, a
# 3-level activity target, plus SMILES/ID metadata columns that should
# NOT become features.  (In your project: set MY_FILE to your own path.)
MY_FILE = DATA_DIR / "qsar_classification.csv"   # <- replace with your own file

frame_demo = pd.read_csv(MY_FILE)
print(f"demo table: {MY_FILE}  ({len(frame_demo)} rows x {frame_demo.shape[1]} columns)")
'''

ML_CELL6 = '''from pharmaml.chem import compute_descriptors, compute_fingerprints, compute_maccs

# Silence RDKit's C++ parser spam -- pharmaml logs invalid rows explicitly.
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

# A demo screening library of real drug-like structures (77 valid
# molecules), plus two *deliberately* broken entries so the invalid-SMILES
# accounting is visible.  (In your project: use your own SMILES column --
# frame["SMILES"] works directly.)
library_table = pd.read_csv(DATA_DIR / "screening_library.csv")
LIBRARY = list(library_table.itertuples(index=False, name=None))  # (name, smiles) pairs

smiles_series = pd.Series([smi for _, smi in LIBRARY], name="SMILES")

descriptor_frame = compute_descriptors(smiles_series, progress=False)

print(f"{descriptor_frame.shape[0]} valid molecules x "
      f"{descriptor_frame.shape[1] - 1} descriptors (RDKit 2D set)")
print(f"dropped {len(LIBRARY) - len(descriptor_frame)} invalid SMILES ")
'''

ML_CELL11 = '''# Structures in, model out: the computed descriptors become X.  The
# shipped table pairs them with an activity column (LogP + H-bond donors
# + noise, generated once from the library descriptors) -- the role your
# measured IC50/pIC50 column plays in a real project.
struct_file = DATA_DIR / "library_with_descriptors.csv"   # <- replace with your own file

ds_struct = load_dataset(struct_file)   # target auto-detected -> 'activity'
print(ds_struct.summary())
'''

ML_CELL31 = '''# --- Recipe B: K classes — you pick K ------------------------------------
# The 4-class demo table ships as examples/data/qsar_multiclass.csv,
# built by examples/make_synthetic_qsar_data.py with seed 11 -- the same
# generator you can run for any other class count K.
N_CLASSES = 4                                   # <-- select: 2, 3, 4, 5, ...
ACTIVITY_LABELS = ["inactive", "weak", "moderate", "strong", "potent"]

multi_file = DATA_DIR / "qsar_multiclass.csv"   # <- replace with your own file
ds_multi = load_dataset(multi_file)
print(ds_multi.summary())
'''

ML_CELL35 = '''# The regression demo table ships as examples/data/qsar_regression.csv
# (make_synthetic_qsar_data.py, seed 21): 20 descriptors + a continuous
# pIC50 endpoint.
reg_file = DATA_DIR / "qsar_regression.csv"     # <- replace with your own file
ds_reg = load_dataset(reg_file)                 # numeric target -> regression

print(ds_reg.summary())
print("task:", ds_reg.task, "| target:", ds_reg.target)
'''

CELLLINE_SETUP = f'''import sys
import tempfile
from pathlib import Path

import pandas as pd

from pharmaml.biology import (
    load_response_table, audit_response_table, collapse_replicates,
    ln_ic50_to_pic50, response_matrix, summarize_compounds,
    call_sensitive_resistant, normalize_expression, impute_expression,
    most_variable_genes, sensitivity_associations, top_biomarkers,
    direction_of_association,
)

WORK = Path(tempfile.mkdtemp(prefix="cellline_"))
# The shipped demo inventory: the screen and the expression matrix load
# from examples/data/ instead of being fabricated inline.
DATA_DIR = Path("data")
pd.set_option("display.width", 120)

print("python:", sys.version.split()[0], "| working dir:", WORK)
'''

CELLLINE_CELL3 = '''# --- DEMO DATA (shipped) --------------------------------------------------
# The GDSC-style screen shipped as examples/data/cellline_screen.csv:
# 23 cell lines x 8 compounds with the planted EGFR signal, plus one
# duplicate pair (assay repeat) and one missing value for §2 to audit.
# (In your project: point screen_path at your own GDSC/CCLE export.)
screen_path = DATA_DIR / "cellline_screen.csv"   # <- replace with your own export

screen_frame = pd.read_csv(screen_path)
print(f"{len(screen_frame)} rows loaded <- {screen_path}")
'''

CELLLINE_CELL17 = '''# --- DEMO EXPRESSION MATRIX (shipped) --------------------------------------
# examples/data/cellline_expression.csv: rows = cell lines, columns = genes
# (298 random genes + EGFR carrying the lung-line signal + KRT8 carrying a
# breast signature; same seed-42 chain that built the screen, so the
# planted biomarker agrees with it).
expression = pd.read_csv(DATA_DIR / "cellline_expression.csv", index_col="cell_line")

expr_norm = normalize_expression(expression)
top_genes = most_variable_genes(expr_norm, n=40)
print(f"expression: {expr_norm.shape[0]} cell lines x {expr_norm.shape[1]} genes (normalized)")
print("top variable genes (first 6):", top_genes[:6])
print("EGFR rank:", top_genes.index("EGFR") + 1 if "EGFR" in top_genes else "not in top 40")
'''

PK_SETUP = f'''import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pharmaml import __version__

pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 110)

# Scratch directory for this run's artifacts (use any path you like).
WORK = Path(tempfile.mkdtemp(prefix="pk_workflow_"))

# The shipped demo inventory: the demo profile loads from examples/data/
# instead of being simulated inline.
DATA_DIR = Path("data")

print(f"pharmaml {{__version__}}")
print(f"artifacts -> {{WORK}}")
'''

PK_CELL4 = '''# --- DEMO DATA ---------------------------------------------------------
# A one-compartment oral concentration-time profile: ka = 1.2/h,
# ke = 0.35/h, a 250 mg dose into Vd = 35 L with F = 0.9, plus realistic
# 5% assay noise on every sample.  Generated once with seed 7 by
# scratch/build_example_data.py and shipped as examples/data/pk_profile.csv.
# (In your project: set PROFILE_CSV to your own file.)
PROFILE_CSV = DATA_DIR / "pk_profile.csv"   # <- replace with your own file

profile = pd.read_csv(PROFILE_CSV)
print(f"demo profile: {PROFILE_CSV}  ({len(profile)} samples)")
'''

DOCKING_SETUP = f'''import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from pharmaml import __version__

pd.set_option("display.max_columns", 14)
pd.set_option("display.width", 120)

# Silence the fork() DeprecationWarning the process pool emits on this
# Python build (harmless here, noisy in committed outputs).
warnings.filterwarnings("ignore", message=".*use of fork.*")

# Scratch directory for this run's artifacts (use any path you like).
WORK = Path(tempfile.mkdtemp(prefix="docking_workflow_"))

# The shipped demo inventory: the ligand library, the receptor PDB and
# the demo campaign tree all live under examples/data/.
DATA_DIR = Path("data")

# RDKit's C++ parser prints errors for invalid SMILES; silence the noise
# (the pharmaml layer reports invalid rows explicitly).
try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

print(f"pharmaml {{__version__}}")
print(f"artifacts -> {{WORK}}")
'''

DOCKING_CELL3 = '''# A demo ligand library: 8 real drug-like molecules plus two
# deliberately broken rows, so the validation accounting has something
# to report. Swap this cell for your own CSV -- everything below runs
# unchanged.
LIGANDS_CSV = DATA_DIR / "ligand_library.csv"   # <- replace with your own file

ligand_table = pd.read_csv(LIGANDS_CSV)
print(f"ligand library <- {LIGANDS_CSV}  ({len(ligand_table)} rows)")
ligand_table
'''

DOCKING_CELL9 = '''# Lysozyme 1AKI, shipped as examples/data/lysozyme_1aki.pdb (RCSB entry
# 1AKI, public domain).  The demo keeps just the 1001 ATOM + 78 HETATM
# records; in practice you start from your own source: an RCSB download,
# a crystallography collaborator, an AlphaFold export, ...
PROTEIN_PDB = WORK / "lysozyme_wet.pdb"   # doctored in the next cell, never the shipped file

LYSOZYME_PDB_TEXT = "".join(
    line + "\\n"
    for line in (DATA_DIR / "lysozyme_1aki.pdb").read_text(encoding="utf-8").splitlines()
    if line.startswith(("ATOM", "HETATM"))
)
PROTEIN_PDB.write_text(LYSOZYME_PDB_TEXT, encoding="utf-8")
n_atom = LYSOZYME_PDB_TEXT.count("ATOM")
n_het = LYSOZYME_PDB_TEXT.count("HETATM")
print(f"{PROTEIN_PDB.name}: {n_atom} ATOM + {n_het} HETATM records")
'''

DOCKING_CELL29 = '''# --- DEMO CAMPAIGN ------------------------------------------------------
# A synthetic AutoDock4 campaign shipped under examples/data/
# docking_campaign/ (regenerated deterministically by
# scratch/build_example_data.py): one folder per ligand, each with an
# r.dlg whose cluster table uses the exact real format.  One folder is
# missing its log entirely and one was interrupted before clustering, so
# the audit has something to find.
#
# The notebook runs on a scratch COPY of the shipped tree: the §7
# housekeeping verbs collect files and preview cleanups *inside* the
# campaign root, and the shipped inventory itself must stay pristine
# (same pattern as the protein above).  (In your project: set
# DOCKING_DIR to your real campaign folder -- e.g. the tree §1-§2 just
# built -- and drop the copy step.)
import shutil

DOCKING_DIR = WORK / "docking_campaign"   # scratch copy, never the shipped tree
shutil.copytree(DATA_DIR / "docking_campaign", DOCKING_DIR)

# The campaign manifest (one (name, SMILES) row per folder) -- §9 maps
# ligand names back to structures through it.
CAMPAIGN_LIGANDS = list(
    pd.read_csv(DOCKING_DIR / "ligands.csv").itertuples(index=False, name=None)
)

n_folders = len([p for p in DOCKING_DIR.iterdir() if p.is_dir()])
print(f"demo campaign: {DATA_DIR / 'docking_campaign'} -> {DOCKING_DIR} (scratch copy)")
print(f"{n_folders} ligand folders (12 OK, 1 missing log, 1 interrupted)")
'''

DOCKING_CELL34 = '''import warnings

from pharmaml.docking import batch_run_directories

# IPython resets warning filters per cell, so scope the fork() noise here.
warnings.filterwarnings("ignore", message=".*use of fork.*")

# Idempotent campaign re-run: folders whose .dlg carries AutoDock's
# completion sentinel are skipped, per-folder failures are collected as
# results (benzocaine has no files at all; nicotine's interrupted log is
# re-attempted and fails loudly on its missing r.dpf) -- which is exactly
# why the completion audit above exists as a separate check.
for r in batch_run_directories(DOCKING_DIR, workers=2):
    status = "ok" if r.ok else f"FAILED: {r.error}"
    print(f"[batch] {r.directory.name:16s} -> {status}")
'''

GEN_SETUP = f'''import tempfile
from pathlib import Path

import pandas as pd
import rdkit
import torch

from pharmaml import __version__

pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 110)

# Scratch directory for this run's artifacts (use any path you like).
WORK = Path(tempfile.mkdtemp(prefix="generative_workflow_"))

# The shipped demo inventory: the §1b seed table loads from examples/data/
# instead of being embedded inline.
DATA_DIR = Path("data")

# One seed for every run in this notebook: reproducible leaderboards.
SEED = 42

print(f"pharmaml {{__version__}} · rdkit {{rdkit.__version__}} · torch {{torch.__version__}}")
print(f"artifacts -> {{WORK}}")
'''

GEN_CELL8 = '''# --- DEMO SEEDS --------------------------------------------------------
# The 12 named drugs shipped as examples/data/seed_molecules.csv
# (regenerated deterministically by scratch/build_example_data.py).
# (In your project: point SEEDS_CSV at your own molecules instead.)
SEEDS_CSV = DATA_DIR / "seed_molecules.csv"   # <- replace with your own file

seeds = pd.read_csv(SEEDS_CSV)
seed_smiles = seeds["smiles"].tolist()
print(f"{len(seeds)} seed molecules  <-  {SEEDS_CSV}")
seeds
'''

WALK_SETUP = f'''from pathlib import Path

import numpy as np
import pandas as pd

from pharmaml import __version__

# The shipped demo inventory: every demo table below is loaded from
# examples/data/ instead of fabricated inline.
DATA_DIR = Path("data")

print(f"pharmaml {{__version__}}")
'''

WALK_CELL3 = '''from pharmaml.data import infer_task, load_dataset

# The shipped demo table: descriptors + string target + SMILES metadata
# column -- exactly the shape a user file arrives in.
demo_file = DATA_DIR / "walkthrough_compounds.csv"   # <- replace with your own path

ds = load_dataset(demo_file)               # target auto-detected -> 'activity'
print(ds.summary())
print("\\nclasses:", ds.classes)
print("dropped non-numeric columns:", ds.dropped_columns)
print("task inference on a pIC50-like target:",
      infer_task(pd.Series(np.linspace(4, 9, 100))))
'''

WALK_CELL5 = '''frame = pd.read_csv(DATA_DIR / "walkthrough_classification.csv")
frame.head()
'''

WALK_CELL10 = '''multi = pd.read_csv(DATA_DIR / "walkthrough_multiclass.csv")

X_tr, X_te, y_tr, y_te = make_train_test_split(
    multi.drop(columns=["Target"]), multi["Target"], test_size=0.25
)
mc = train_model("logistic_regression", "classification", X_tr, y_tr, tune=False)
mc_report = evaluate_model(mc.model, X_te, y_te, "classification")

print(f"3-class accuracy={mc_report.accuracy:.3f}  f1_macro={mc_report.f1_macro:.3f}  "
      f"roc_auc(ovr)={mc_report.roc_auc:.3f}")
print("classes present in train and test folds:", sorted(y_tr.unique()), sorted(y_te.unique()))
print("k x k confusion matrix:\\n", mc_report.confusion)
'''

# Code-cell replacements: (notebook, index, anchor, new source, done-marker)
CODE_EDITS = (
    ("complete_ml_workflow.ipynb", 1, 'prefix="ml_workflow_"', ML_SETUP,
     'DATA_DIR = Path("data")'),
    ("complete_ml_workflow.ipynb", 3, "rng = np.random.default_rng(11)", ML_CELL3,
     'DATA_DIR / "qsar_classification.csv"'),
    ("complete_ml_workflow.ipynb", 6, "LIBRARY = [", ML_CELL6,
     'DATA_DIR / "screening_library.csv"'),
    ("complete_ml_workflow.ipynb", 11, "rng = np.random.default_rng(7)", ML_CELL11,
     'DATA_DIR / "library_with_descriptors.csv"'),
    ("complete_ml_workflow.ipynb", 31,
     "from make_synthetic_qsar_data import make_classification_table", ML_CELL31,
     'DATA_DIR / "qsar_multiclass.csv"'),
    ("complete_ml_workflow.ipynb", 35,
     "from make_synthetic_qsar_data import make_regression_table", ML_CELL35,
     'DATA_DIR / "qsar_regression.csv"'),
    ("complete_cellline_workflow.ipynb", 1, "rng = np.random.default_rng(42)",
     CELLLINE_SETUP, 'DATA_DIR = Path("data")'),
    ("complete_cellline_workflow.ipynb", 3, "CELL_LINES = {", CELLLINE_CELL3,
     'DATA_DIR / "cellline_screen.csv"'),
    ("complete_cellline_workflow.ipynb", 17, "DEMO EXPRESSION MATRIX", CELLLINE_CELL17,
     'DATA_DIR / "cellline_expression.csv"'),
    ("complete_pk_workflow.ipynb", 2, 'prefix="pk_workflow_"', PK_SETUP,
     'DATA_DIR = Path("data")'),
    ("complete_pk_workflow.ipynb", 4, "def make_pk_profile(", PK_CELL4,
     'DATA_DIR / "pk_profile.csv"'),
    ("complete_docking_workflow.ipynb", 1, 'prefix="docking_workflow_"', DOCKING_SETUP,
     'DATA_DIR = Path("data")'),
    ("complete_docking_workflow.ipynb", 3, '"SMILES": [', DOCKING_CELL3,
     'DATA_DIR / "ligand_library.csv"'),
    ("complete_docking_workflow.ipynb", 9, "_LYSOZYME_BLOB", DOCKING_CELL9,
     'DATA_DIR / "lysozyme_1aki.pdb"'),
    ("complete_docking_workflow.ipynb", 29,
     ("CAMPAIGN_LIGANDS = [", 'DOCKING_DIR = DATA_DIR / "docking_campaign"'),
     DOCKING_CELL29, 'shutil.copytree(DATA_DIR / "docking_campaign"'),
    ("complete_docking_workflow.ipynb", 34, "batch_run_directories(DOCKING_DIR, workers=2)",
     DOCKING_CELL34, "completion sentinel are skipped"),
    ("complete_generative_workflow.ipynb", 2, 'prefix="generative_workflow_"', GEN_SETUP,
     'DATA_DIR = Path("data")'),
    ("complete_generative_workflow.ipynb", 8, "SEED_TABLE = {", GEN_CELL8,
     'DATA_DIR / "seed_molecules.csv"'),
    ("pharmaml_walkthrough.ipynb", 1, "pharmaml {__version__}", WALK_SETUP,
     'DATA_DIR = Path("data")'),
    ("pharmaml_walkthrough.ipynb", 3, "rng = np.random.default_rng(5)", WALK_CELL3,
     'DATA_DIR / "walkthrough_compounds.csv"'),
    ("pharmaml_walkthrough.ipynb", 5,
     "from sklearn.datasets import make_classification", WALK_CELL5,
     'DATA_DIR / "walkthrough_classification.csv"'),
    ("pharmaml_walkthrough.ipynb", 10,
     "from sklearn.datasets import make_classification as _mk", WALK_CELL10,
     'DATA_DIR / "walkthrough_multiclass.csv"'),
)

# Markdown prose patches: (notebook, index, old substring, new substring).
PROSE_EDITS = (
    ("complete_ml_workflow.ipynb", 0,
     "Every number below is generated from fixed seeds, so the notebook is fully\n"
     "reproducible; replace the demo file in §1 with your own and re-run all cells.",
     "Every demo table below ships in `examples/data/` (regenerated deterministically\n"
     "by `scratch/build_example_data.py`), so the notebook runs offline and stays\n"
     "fully reproducible; replace the demo file in §1 with your own and re-run all\n"
     "cells."),
    ("complete_cellline_workflow.ipynb", 2,
     "The synthetic panel: **24 cell lines** across 10 tissues, **8\n"
     "compounds**, with one deliberate duplicate pair and one missing value\n"
     "(§2 needs something honest to audit).",
     "The synthetic panel: **23 cell lines** across 11 tissues, **8\n"
     "compounds**, with one deliberate duplicate pair and one missing value\n"
     "(§2 needs something honest to audit). It ships as\n"
     "`examples/data/cellline_screen.csv` (185 rows, seed 42; regenerate\n"
     "deterministically via `scratch/build_example_data.py`)."),
    ("complete_pk_workflow.ipynb", 3,
     "The demo profile below is a one-compartment oral simulation with realistic\n"
     "5 % assay noise — the generator is embedded in the cell so the notebook\n"
     "re-runs anywhere offline. **Swap it for your own file and every cell below\n"
     "runs unchanged.**",
     "The demo profile below ships as `examples/data/pk_profile.csv` — a\n"
     "one-compartment oral simulation with realistic 5 % assay noise, generated\n"
     "once with seed 7 (`scratch/build_example_data.py` keeps the generator).\n"
     "**Swap it for your own file and every cell below runs unchanged.**"),
    ("complete_pk_workflow.ipynb", 0,
     "Every number below comes from a fixed-seed simulation, so the notebook is\n"
     "fully reproducible; replace the demo profile in §1 with your own CSV and\n"
     "re-run all cells.",
     "Every number below comes from the shipped fixed-seed demo profile\n"
     "(`examples/data/pk_profile.csv`), so the notebook runs offline and stays\n"
     "fully reproducible; replace it in §1 with your own CSV and re-run all cells."),
    ("complete_docking_workflow.ipynb", 0,
     "The demo campaign below is *synthetic but structurally identical* to real\n"
     "AutoDock4 output, so every cell runs without a docking engine installed.",
     "The demo campaign below is *synthetic but structurally identical* to real\n"
     "AutoDock4 output — 14 folders shipped under `examples/data/\n"
     "docking_campaign/` (12 complete runs + two planted audit cases) — so every\n"
     "cell runs without a docking engine installed."),
    ("complete_docking_workflow.ipynb", 28,
     "Finally,\n`batch_run_directories` re-runs the whole campaign *idempotently*: folders\n"
     "whose `.dlg` already exists are skipped, and failures are collected as\n"
     "results (not raised)",
     "Finally,\n`batch_run_directories` re-runs the whole campaign *idempotently*: a folder\n"
     "whose `.dlg` already carries AutoDock's `Successful Completion` sentinel is\n"
     "skipped (an interrupted log is re-run, never trusted just because the file\n"
     "exists), and failures are collected as results (not raised)"),
    ("complete_generative_workflow.ipynb", 7,
     "The demo table below is the same\n12 named drugs shipped as `examples/data/seed_molecules.csv` — swap the\n"
     "dictionary for your own CSV and every cell below runs unchanged.",
     "The demo table below is the\n"
     "shipped `examples/data/seed_molecules.csv` — 12 named drugs (regenerated\n"
     "deterministically via `scratch/build_example_data.py`) — point\n"
     "`SEEDS_CSV` at your own CSV and every cell below runs unchanged."),
    # Post-re-execution honesty pass: the running-baseline RL (round 26)
    # no longer triggers the plateau early stop on this seed, so the §7
    # prose and the §11 recap bullet must not claim it did.
    ("complete_generative_workflow.ipynb", 29,
     "plateau triggers the built-in early stop — RL runs until it stops\nimproving, not until the iteration budget burns out.",
     "plateau triggers the built-in early stop — RL runs until it stops\nimproving, not until the iteration budget burns out (this run kept\nimproving to the final iteration, so the stop never fired)."),
    ("complete_generative_workflow.ipynb", 39,
     "fine-tuning bent it toward the objective and stopped itself on the\n  plateau (§7);",
     "fine-tuning bent it toward the objective; the plateau early-stop stayed\n  armed but never fired — the mean score improved to the last iteration\n  (§7);"),
    ("pharmaml_walkthrough.ipynb", 0,
     "This notebook exercises **every domain** of the `pharmaml` package on synthetic data:",
     "This notebook exercises **every domain** of the `pharmaml` package on the\n"
     "shipped demo files in `examples/data/`:"),
    ("pharmaml_walkthrough.ipynb", 0,
     "No external datasets are required; every number below is generated with fixed seeds,\n"
     "so the notebook is fully reproducible.",
     "No external datasets are required — the demo tables ship in `examples/data/`\n"
     "(regenerated deterministically via `scratch/build_example_data.py`), so the\n"
     "notebook runs offline and is fully reproducible."),
    ("pharmaml_walkthrough.ipynb", 4,
     "Synthetic binary-activity table → stratified split",
     "The shipped binary-activity table → stratified split"),
)


def _fail(message: str) -> None:
    raise SystemExit(f"DRIVER FAILED: {message}")


def apply_edits() -> int:
    """Replace the fabrication cells + patch the prose.  Idempotent."""
    changed = 0
    notebooks: dict[str, object] = {}
    for name in NOTEBOOKS:
        notebooks[name] = nbformat.read(EXAMPLES / name, as_version=4)

    for name, index, anchor, new_source, done in CODE_EDITS:
        nb = notebooks[name]
        cell = nb.cells[index]
        source = cell.source
        if done in source:
            continue  # already converted (idempotent re-run)
        if cell.cell_type != "code":
            _fail(f"{name} cell {index} is {cell.cell_type}, expected code")
        anchors = anchor if isinstance(anchor, tuple) else (anchor,)
        if not any(a in source for a in anchors):
            _fail(f"{name} cell {index} lost its anchor {anchors!r} -- "
                  "cell numbering drifted; update the driver's edit table")
        compile(new_source, f"{name}#{index}", "exec")  # never write broken code
        cell.source = new_source
        cell.outputs = []
        cell.execution_count = None
        changed += 1

    for name, index, old, new in PROSE_EDITS:
        nb = notebooks[name]
        cell = nb.cells[index]
        if new in cell.source:
            continue  # already patched
        if cell.cell_type != "markdown":
            _fail(f"{name} cell {index} is {cell.cell_type}, expected markdown")
        if old not in cell.source:
            _fail(f"{name} cell {index} lost its prose anchor -- "
                  f"update the driver's PROSE_EDITS entry starting {old[:48]!r}")
        cell.source = cell.source.replace(old, new, 1)
        changed += 1

    for name, nb in notebooks.items():
        nbformat.write(nb, EXAMPLES / name)
    return changed


def guard() -> None:
    """Zero banned fabrication tokens across all six executed notebooks."""
    hits = []
    for name in NOTEBOOKS:
        nb = nbformat.read(EXAMPLES / name, as_version=4)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            for token in BANNED_TOKENS:
                if token in cell.source:
                    hits.append((name, i, token))
    if hits:
        for name, i, token in hits:
            print(f"  BANNED {name} cell {i}: {token!r}")
        _fail(f"{len(hits)} banned fabrication token hit(s) remain")
    print(f"guard: 0 hits for all {len(NOTEBOOKS) * len(BANNED_TOKENS)} "
          "notebook x token combinations")


def execute_notebooks(names: tuple[str, ...] = NOTEBOOKS) -> None:
    """Re-run notebooks top-to-bottom with nbclient (cwd=examples/)."""
    from nbclient import NotebookClient

    for name in names:
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        started = time.perf_counter()
        client = NotebookClient(
            nb,
            timeout=2400,
            kernel_name="python3",
            resources={"metadata": {"path": str(EXAMPLES)}},
        )
        client.execute()
        nbformat.write(nb, path)

        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        executed = sum(1 for c in code_cells if c.get("execution_count"))
        errors = [o for c in code_cells for o in c.get("outputs", [])
                  if o.output_type == "error"]
        figures = sum(1 for c in code_cells for o in c.get("outputs", [])
                      if o.get("data", {}).get("image/png"))
        if errors or executed != len(code_cells):
            _fail(f"{name}: {len(errors)} error output(s), "
                  f"{executed}/{len(code_cells)} cells executed")
        print(f"executed {name}: {len(code_cells)} code cells, {figures} figures, "
              f"{time.perf_counter() - started:.0f} s")


def regenerate_twins() -> None:
    """Refresh the five *_no_output.ipynb twins (code-synced by design)."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scratch" / "strip_notebook_outputs.py")],
        check=True, capture_output=True, text=True,
    )
    print(result.stdout.strip())


def spot_checks() -> None:
    """Print the headline outputs the round asked to verify."""
    checks = (
        # (notebook, cell matcher, blurb)
        ("complete_pk_workflow.ipynb", "auc_log = auc_trapezoid",
         "PK §3 AUC linear/log"),
        ("complete_pk_workflow.ipynb", "summary = nca_summary",
         "PK §2 NCA summary"),
        ("complete_docking_workflow.ipynb", "box = compute_grid_box",
         "docking §4 grid box"),
        ("complete_docking_workflow.ipynb", "audit = check_docking_completion",
         "docking §6 completion audit"),
        ("complete_docking_workflow.ipynb",
         "batch_run_directories(DOCKING_DIR, workers=2)",
         "docking §6 batch re-run"),
        ("complete_generative_workflow.ipynb", "early stopped",
         "generative §7 RL summary"),
        ("complete_generative_workflow.ipynb", "seed_frameworks",
         "generative §6 scaffold A/B"),
        ("complete_ml_workflow.ipynb", "N_CLASSES = ",
         "ML §8 multiclass summary"),
        ("complete_cellline_workflow.ipynb", "EGFR vs erlotinib",
         "cellline §7 biomarker verdict"),
    )
    for name, needle, blurb in checks:
        nb = nbformat.read(EXAMPLES / name, as_version=4)
        cell = next((c for c in nb.cells if c.cell_type == "code" and needle in c.source),
                    None)
        if cell is None:
            _fail(f"{name}: spot-check cell {needle!r} vanished")
        text = "".join(o.get("text", "") for o in cell.get("outputs", [])
                       if o.get("output_type") == "stream")
        print(f"\n--- {blurb} ({name}) ---")
        print("\n".join(line for line in text.splitlines() if line.strip())[:1200])


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    stage = args[0] if args else "all"
    if stage in {"--execute", "execute"}:
        execute_notebooks((args[1],))
        return 0
    if stage in {"--twins", "twins"}:
        regenerate_twins()
        return 0
    if stage in {"--checks", "checks"}:
        spot_checks()
        return 0
    if stage in {"--edits", "edits"}:
        # prose/cell patches only (e.g. after a markdown honesty pass) --
        # no re-execution, committed outputs are preserved untouched.
        print(f"edits applied : {apply_edits()}")
        guard()
        return 0

    missing = [p for p in (
        "qsar_classification.csv", "screening_library.csv",
        "library_with_descriptors.csv", "qsar_multiclass.csv",
        "qsar_regression.csv", "cellline_screen.csv", "cellline_expression.csv",
        "pk_profile.csv", "ligand_library.csv", "lysozyme_1aki.pdb",
        "docking_campaign", "seed_molecules.csv", "walkthrough_compounds.csv",
        "walkthrough_classification.csv", "walkthrough_multiclass.csv",
    ) if not (EXAMPLES / "data" / p).exists()]
    if missing:
        _fail(f"examples/data is missing {missing} -- run scratch/build_example_data.py")

    print(f"edits applied : {apply_edits()}")
    guard()
    execute_notebooks()
    regenerate_twins()
    spot_checks()
    print("\n26-R7 driver OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
