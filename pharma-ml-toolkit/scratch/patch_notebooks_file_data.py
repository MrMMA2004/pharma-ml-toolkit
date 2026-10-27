"""Make every notebook load its input data from ``examples/data/``.

Task 100x-24-a: "Jupyter files must work entirely based on the files I
give them and not create their own sample data."  Every inline-data
cell in the six executed notebooks is replaced by a load cell pointing
at the shipped files under ``examples/data/`` (byte-identical to what
the old cells fabricated -- see ``scratch/build_example_data.py`` and
``scratch/verify_example_data_provenance.py``).  Cells that merely
DEMONSTRATE a deterministic toolkit API (``pk.simulate``-style curves,
the §5 §6 PD grids, ...) stay: they are the product, not sample data.

Like ``patch_ml_notebook.py`` before it, this patch finds stable anchor
cells instead of rebuilding from scratch.  Every replacement asserts
its anchor appears exactly once, so a half-applied patch cannot pass
silently; re-running the script is a no-op (anchors are gone).

Per notebook:

* complete_ml_workflow      -- §1 demo table, §2 screening library,
                               §2 structures->model table, §8 multiclass,
                               §9 regression: now load qsar_classification.csv,
                               screening_library.csv,
                               library_with_descriptors.csv,
                               qsar_multiclass.csv, qsar_regression.csv.
* complete_cellline_workflow-- §1 screen + §6 expression matrix: now load
                               cellline_screen.csv / cellline_expression.csv.
* complete_pk_workflow      -- §1 observed profile: now loads pk_profile.csv.
* complete_docking_workflow -- §1 ligand library (ligand_library.csv),
                               §2 receptor (lysozyme_1aki.pdb instead of the
                               gz+base64 blob), §6 campaign (copied from
                               examples/data/docking_campaign/ so the §7
                               housekeeping mutates only the working copy).
* complete_generative_workflow -- §1b seeds: now reads seed_molecules.csv
                               (the SEED_TABLE dict is gone).
* pharmaml_walkthrough      -- §1/§2/§3 tables: now load
                               walkthrough_compounds.csv /
                               walkthrough_classification.csv /
                               walkthrough_multiclass.csv.

After patching, the script re-executes each notebook with nbclient
(under the worker-cap env vars) and regenerates the no-output twins
via scratch/strip_notebook_outputs.py.

Run from the toolkit root:

    python scratch/patch_notebooks_file_data.py            # patch only
    python scratch/patch_notebooks_file_data.py --exec     # patch + execute
    python scratch/patch_notebooks_file_data.py --exec --only complete_pk_workflow
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

DATA_DIR_SNIPPET = 'DATA_DIR = Path("data") if Path("data").is_dir() else Path("examples/data")'


# ---------------------------------------------------------------------------
# new cell sources
# ---------------------------------------------------------------------------

ML_DEMO_TABLE = f"""\
# --- THE DEMO TABLE (shipped with the toolkit) ---------------------------
# examples/data/qsar_classification.csv: a realistic QSAR table -- 8
# descriptors, a 3-level activity target, plus SMILES/ID metadata columns
# that should NOT become features -- regenerated deterministically by
# scratch/build_example_data.py.
# (In your project: set MY_FILE to your own path.)
{DATA_DIR_SNIPPET}

MY_FILE = DATA_DIR / "qsar_classification.csv"   # <- replace with your own file

demo_frame = pd.read_csv(MY_FILE)
print(f"demo table: {{MY_FILE}}  ({{demo_frame.shape[0]}} rows x {{demo_frame.shape[1]}} columns)")
"""

ML_LIBRARY_LOAD = """\
# The demo screening library shipped with the toolkit: 77 valid
# drug-like structures plus two *deliberately* broken entries so the
# invalid-SMILES accounting is visible (examples/data/
# screening_library.csv, regenerated deterministically by
# scratch/build_example_data.py).  (In your project: use your own
# SMILES column -- frame["SMILES"] works directly.)
LIBRARY = list(
    pd.read_csv(DATA_DIR / "screening_library.csv").itertuples(index=False, name=None)
)
"""

ML_STRUCT_ACTIVITY = """\
# Structures in, model out: the computed descriptors become X.  The
# shipped table examples/data/library_with_descriptors.csv carries the
# screening library's eight chemist-readable descriptors plus an
# activity call (planted as LogP + H-bond donors + assay noise when the
# file was generated -- in a real project this is where your measured
# IC50/pIC50 column lands).  Regenerated deterministically by
# scratch/build_example_data.py.
struct_file = DATA_DIR / "library_with_descriptors.csv"
ds_struct = load_dataset(struct_file)   # target auto-detected -> 'activity'
print(ds_struct.summary())
"""

ML_MULTICLASS = f"""\
# --- Recipe B: K classes -- the shipped 4-class table --------------------
# examples/data/qsar_multiclass.csv: 360 compounds, 20 descriptors and a
# 4-level target (inactive/weak/moderate/strong), generated once with
# the shipped generator (examples/make_synthetic_qsar_data.py
# --classes 4 --seed 11) by scratch/build_example_data.py.  For a
# different K, build your own table with that script and point
# MULTI_FILE at it.
MULTI_FILE = DATA_DIR / "qsar_multiclass.csv"   # <- replace with your own file
ds_multi = load_dataset(MULTI_FILE)
N_CLASSES = len(ds_multi.classes)               # the shipped table: 4
print(ds_multi.summary())
"""

ML_REGRESSION = """\
# The shipped regression table: 260 compounds, 20 descriptors, a
# continuous pIC50 target (examples/data/qsar_regression.csv, generated
# once with examples/make_synthetic_qsar_data.py --n 260 --seed 21 by
# scratch/build_example_data.py).  Point REG_FILE at your own table for
# real endpoints.
REG_FILE = DATA_DIR / "qsar_regression.csv"     # <- replace with your own file
ds_reg = load_dataset(REG_FILE)                 # numeric target -> regression

print(ds_reg.summary())
print("task:", ds_reg.task, "| target:", ds_reg.target)
"""

ML_SETUP_NEW = """\
import sys
import tempfile
from pathlib import Path

import pandas as pd

from pharmaml import __version__

pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 110)

# Scratch directory for this run's artifacts (use any path you like).
WORK = Path(tempfile.mkdtemp(prefix="ml_workflow_"))

print(f"pharmaml {__version__}")
print(f"artifacts -> {WORK}")
"""

CL_SETUP_NEW = """\
import sys
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
pd.set_option("display.width", 120)

print("python:", sys.version.split()[0], "| working dir:", WORK)
"""

CL_SCREEN_LOAD = f"""\
# --- THE DEMO SCREEN (shipped with the toolkit) --------------------------
# examples/data/cellline_screen.csv: a GDSC-style long-format screen --
# 24 cell lines across 10 tissues, 8 compounds, one deliberate duplicate
# pair and one missing value for section 2 to audit -- regenerated
# deterministically by scratch/build_example_data.py.
# (In your project: skip this cell and point the loader at your own
# GDSC/CCLE export.)
{DATA_DIR_SNIPPET}

screen_path = DATA_DIR / "cellline_screen.csv"   # <- replace with your own export

screen_frame = pd.read_csv(screen_path)
print(f"demo screen: {{screen_path}}  ({{len(screen_frame)}} rows)")
"""

CL_EXPRESSION_LOAD = """\
# --- THE DEMO EXPRESSION MATRIX (shipped with the toolkit) ---------------
# examples/data/cellline_expression.csv: 24 cell lines x 300 genes, with
# EGFR over-expressed in the lung lines and KRT8 in the breast lines --
# the planted biomarkers section 7 recovers.  Regenerated
# deterministically by scratch/build_example_data.py; in your project,
# point the loader at your own CCLE/expression matrix (rows = cell
# lines, columns = genes).
expression = pd.read_csv(DATA_DIR / "cellline_expression.csv", index_col=0)

expr_norm = normalize_expression(expression)
top_genes = most_variable_genes(expr_norm, n=40)
print(f"expression: {expr_norm.shape[0]} cell lines x {expr_norm.shape[1]} genes (normalized)")
print("top variable genes (first 6):", top_genes[:6])
print("EGFR rank:", top_genes.index("EGFR") + 1 if "EGFR" in top_genes else "not in top 40")
"""

PK_PROFILE_LOAD = f"""\
# --- THE DEMO PROFILE (shipped with the toolkit) -------------------------
# examples/data/pk_profile.csv: a synthetic oral concentration-time
# profile -- 1-compartment kinetics (ka = 1.2/h, ke = 0.35/h), a 250 mg
# dose into Vd = 35 L with F = 0.9, plus realistic 5% assay noise on
# every sample -- regenerated deterministically by
# scratch/build_example_data.py.
# (In your project: set PROFILE_CSV to your own file.)
{DATA_DIR_SNIPPET}

PROFILE_CSV = DATA_DIR / "pk_profile.csv"   # <- replace with your own file
print(f"demo profile: {{PROFILE_CSV}}")
"""

DK_LIGANDS_LOAD = f"""\
# A demo ligand library shipped with the toolkit: 8 real drug-like
# molecules plus two deliberately broken rows, so the validation
# accounting has something to report (examples/data/ligand_library.csv,
# regenerated deterministically by scratch/build_example_data.py).
# Swap the path for your own CSV -- everything below runs unchanged.
{DATA_DIR_SNIPPET}

LIGANDS_CSV = DATA_DIR / "ligand_library.csv"   # <- replace with your own CSV

ligand_table = pd.read_csv(LIGANDS_CSV)
print(f"ligand library: {{LIGANDS_CSV}}  ({{len(ligand_table)}} rows)")
ligand_table
"""

DK_RECEPTOR_LOAD = """\
# The receptor input ships with the toolkit: examples/data/
# lysozyme_1aki.pdb -- hen egg-white lysozyme, the real RCSB 1AKI entry
# (public domain).  In practice you write this file from your own source:
# an RCSB download, a crystallography collaborator, an AlphaFold export,
# ... -- only the ATOM/HETATM records become the receptor, so we extract
# them the way any PDB-cleaning pipeline would.
LYSOZYME_PDB_TEXT = "".join(
    line + "\\n"
    for line in (DATA_DIR / "lysozyme_1aki.pdb").read_text(encoding="utf-8").splitlines()
    if line.startswith(("ATOM", "HETATM"))
)

PROTEIN_PDB = WORK / "lysozyme_wet.pdb"
PROTEIN_PDB.write_text(LYSOZYME_PDB_TEXT, encoding="utf-8")
n_atom = LYSOZYME_PDB_TEXT.count("ATOM")
n_het = LYSOZYME_PDB_TEXT.count("HETATM")
print(f"{PROTEIN_PDB.name}: {n_atom} ATOM + {n_het} HETATM records")
"""

DK_CAMPAIGN_LOAD = f"""\
# --- THE DEMO CAMPAIGN (shipped with the toolkit) ------------------------
# examples/data/docking_campaign/: a synthetic AutoDock4 campaign -- one
# folder per ligand, each with an r.dlg whose cluster table uses the
# exact real format; one folder is missing its log entirely and one was
# interrupted before clustering, so the audit has something to find.
# The tree is regenerated deterministically by
# scratch/build_example_data.py and copied into this run's scratch
# directory so the section 7 housekeeping can mutate its working copy
# freely.  (In your project: skip the copy and point DOCKING_DIR at
# your real campaign folder.)
import shutil

# The manifest maps folder names to SMILES (the ligand-efficiency
# rescoring in section 9 needs it); it travels with the campaign tree.
CAMPAIGN_LIGANDS = list(
    pd.read_csv(DATA_DIR / "docking_campaign" / "ligands.csv").itertuples(index=False, name=None)
)

DOCKING_DIR = WORK / "docking_run"     # <- replace with your campaign folder
shutil.copytree(
    DATA_DIR / "docking_campaign", DOCKING_DIR,
    ignore=shutil.ignore_patterns("ligands.csv"),
)

print(f"demo campaign: {{DOCKING_DIR}}")
print(f"{{len(list(DOCKING_DIR.iterdir()))}} ligand folders "
      f"(12 OK, 1 missing log, 1 interrupted)")
"""

GEN_SEEDS_LOAD = f"""\
# --- DEMO SEEDS (shipped with the toolkit) -------------------------------
# The 12 named drugs from examples/data/seed_molecules.csv -- the same
# file the CLI's --seeds flag accepts.  (In your project: point the
# loader at your own molecules CSV instead.)
{DATA_DIR_SNIPPET}

seeds = pd.read_csv(DATA_DIR / "seed_molecules.csv")
seed_smiles = seeds["smiles"].tolist()
print(f"{{len(seeds)}} seed molecules")
seeds
"""

WT_COMPOUNDS_LOAD = f"""\
from pathlib import Path

from pharmaml.data import infer_task, load_dataset

# A user-style file shipped with the toolkit: descriptors + string target
# + SMILES metadata column (examples/data/walkthrough_compounds.csv,
# regenerated deterministically by scratch/build_example_data.py).
{DATA_DIR_SNIPPET}

tmp = DATA_DIR / "walkthrough_compounds.csv"   # <- replace with your own path

ds = load_dataset(tmp)                  # target auto-detected -> 'activity'
print(ds.summary())
print("\\nclasses:", ds.classes)
print("dropped non-numeric columns:", ds.dropped_columns)
print("task inference on a pIC50-like target:",
      infer_task(pd.Series(np.linspace(4, 9, 100))))
"""

WT_CLASSIFICATION_LOAD = """\
# A descriptor table + binary target shipped with the toolkit
# (examples/data/walkthrough_classification.csv, regenerated
# deterministically by scratch/build_example_data.py).
frame = pd.read_csv(DATA_DIR / "walkthrough_classification.csv")
frame.head()
"""

WT_MULTICLASS_LOAD = """\
# The 3-class variant of the same table layout, shipped as
# examples/data/walkthrough_multiclass.csv (deterministic provenance via
# scratch/build_example_data.py).
multi = pd.read_csv(DATA_DIR / "walkthrough_multiclass.csv")

X_tr, X_te, y_tr, y_te = make_train_test_split(
    multi.drop(columns=["Target"]), multi["Target"], test_size=0.25
)
mc = train_model("logistic_regression", "classification", X_tr, y_tr, tune=False)
mc_report = evaluate_model(mc.model, X_te, y_te, "classification")

print(f"3-class accuracy={mc_report.accuracy:.3f}  f1_macro={mc_report.f1_macro:.3f}  "
      f"roc_auc(ovr)={mc_report.roc_auc:.3f}")
print("classes present in train and test folds:", sorted(y_tr.unique()), sorted(y_te.unique()))
print("k x k confusion matrix:\\n", mc_report.confusion)
"""


# ---------------------------------------------------------------------------
# patch helpers
# ---------------------------------------------------------------------------

def replace_cell(nb, index: int, anchor: str, new_source: str, label: str) -> bool:
    """Replace a whole cell source, asserting the old content matches."""
    cell = nb.cells[index]
    if anchor in cell.source:
        if cell.source == new_source:
            return False  # already patched
        assert cell.cell_type == "code", f"{label}: cell {index} is not code"
        cell.source = new_source
        return True
    if new_source in cell.source or "build_example_data" in cell.source:
        return False  # already patched (partial anchors)
    raise AssertionError(f"{label}: anchor not found in cell {index}: {anchor[:60]!r}")


def replace_text(nb, index: int, old: str, new: str, label: str) -> bool:
    """Replace an exact substring inside a cell (must occur exactly once)."""
    cell = nb.cells[index]
    if new in cell.source:
        return False  # already patched (append-style edits are idempotent)
    count = cell.source.count(old)
    if count == 0:
        return False  # already patched (or absent)
    assert count == 1, f"{label}: anchor occurs {count}x in cell {index}"
    cell.source = cell.source.replace(old, new)
    return True


def patch_ml(nb) -> list[str]:
    log = []
    log.append(f"cell0 title: {replace_text(nb, 0, '''Every number below is generated from fixed seeds, so the notebook is fully
reproducible; replace the demo file in §1 with your own and re-run all cells.''', '''Every demo table below ships with the toolkit under `examples/data/` —
regenerated deterministically by `scratch/build_example_data.py` — so the
notebook is fully reproducible; replace the demo file in §1 with your own
and re-run all cells.''', 'ml')}")
    log.append(f"cell2 §1 md: {replace_text(nb, 2, '''The demo file carries a `SMILES` column that `load_dataset` drops as
non-numeric — §2 turns those structures into features.''', '''The demo file carries a `SMILES` column that `load_dataset` drops as
non-numeric — §2 turns those structures into features. The demo table
ships as `examples/data/qsar_classification.csv` (8 descriptors, a
3-level activity target, SMILES and compound-ID metadata columns),
regenerated deterministically by `scratch/build_example_data.py`.''', 'ml')}")
    log.append(f"cell1 setup: {replace_cell(nb, 1, '''import numpy as np
import pandas as pd

from pharmaml import __version__''', ML_SETUP_NEW, 'ml')}")
    log.append(f"cell3 demo table: {replace_cell(nb, 3, '# --- DEMO DATA ---', ML_DEMO_TABLE, 'ml')}")

    # cell 6: replace the embedded LIBRARY list with a CSV load (surgery)
    cell = nb.cells[6]
    start = cell.source.find("# A demo screening library")
    end = cell.source.find("]\n\nsmiles_series")
    if start != -1 and end != -1:
        cell.source = (
            cell.source[:start] + ML_LIBRARY_LOAD + "\n" + cell.source[end + 3:]
        )
        log.append("cell6 library: True")
    elif "screening_library.csv" in cell.source:
        log.append("cell6 library: False (already patched)")
    else:
        raise AssertionError("ml cell 6: library anchors not found")

    log.append(f"cell11 structures: {replace_cell(nb, 11, '# Structures in, model out', ML_STRUCT_ACTIVITY, 'ml')}")
    log.append(f"cell30 §8 md: {replace_text(nb, 30, '''A target with 3+ distinct values is a first-class citizen: stratified
splits, k×k confusion matrix, one-vs-rest ROC per class, macro-averaged
metrics.''', '''A target with 3+ distinct values is a first-class citizen: stratified
splits, k×k confusion matrix, one-vs-rest ROC per class, macro-averaged
metrics. The demo table ships as `examples/data/qsar_multiclass.csv`
(360 compounds, 4 levels); build your own K-class table with
`examples/make_synthetic_qsar_data.py --classes K` and point the loader
at it.''', 'ml')}")
    log.append(f"cell31 multiclass: {replace_cell(nb, 31, '# --- Recipe B: K classes', ML_MULTICLASS, 'ml')}")
    log.append(f"cell35 regression: {replace_cell(nb, 35, 'from make_synthetic_qsar_data import make_regression_table', ML_REGRESSION, 'ml')}")
    log.append(f"cell52 recap: {replace_text(nb, 52, '| Binary or K classes | `make_classification_table(n_classes=N_CLASSES)` |', '''| Binary or K classes | bring a K-class CSV (`examples/make_synthetic_qsar_data.py` builds one) |''', 'ml')}")
    return log


def patch_cellline(nb) -> list[str]:
    log = []
    log.append(f"cell0 title: {replace_text(nb, 0, '''The demo data is synthetic but structurally honest: EGFR expression is
planted to drive erlotinib/gefitinib sensitivity, so the biomarker call
in §7 genuinely recovers it. Point the loaders at your own GDSC/CCLE
exports (same columns) and every cell works unchanged.''', '''The demo data is synthetic but structurally honest — and it ships with
the toolkit (`examples/data/cellline_screen.csv` and
`examples/data/cellline_expression.csv`, regenerated deterministically
by `scratch/build_example_data.py`): EGFR expression is planted to
drive erlotinib/gefitinib sensitivity, so the biomarker call in §7
genuinely recovers it. Point the loaders at your own GDSC/CCLE exports
(same columns) and every cell works unchanged.''', 'cellline')}")
    log.append(f"cell1 setup: {replace_cell(nb, 1, 'rng = np.random.default_rng(42)', CL_SETUP_NEW, 'cellline')}")
    log.append(f"cell2 §1 md: {replace_text(nb, 2, '''The synthetic panel: **24 cell lines** across 10 tissues, **8
compounds**, with one deliberate duplicate pair and one missing value
(§2 needs something honest to audit).''', '''The synthetic panel: **24 cell lines** across 10 tissues, **8
compounds**, with one deliberate duplicate pair and one missing value
(§2 needs something honest to audit). The demo screen ships as
`examples/data/cellline_screen.csv` (regenerated deterministically by
`scratch/build_example_data.py`).''', 'cellline')}")
    log.append(f"cell3 screen: {replace_cell(nb, 3, '# --- DEMO DATA (synthetic, seeded) ---', CL_SCREEN_LOAD, 'cellline')}")
    log.append(f"cell16 §6 md: {replace_text(nb, 16, '''`most_variable_genes` picks the MAD-highest genes (the ones that carry
panel structure — 300 random genes are planted, plus EGFR with its
lung-line signal, and another gene carrying a breast signature so
selection isn't a one-gene show).''', '''`most_variable_genes` picks the MAD-highest genes (the ones that carry
panel structure — the shipped matrix `examples/data/cellline_expression.csv`
carries 300 random genes, plus EGFR with its lung-line signal, and
another gene with a breast signature so selection isn't a one-gene show).''', 'cellline')}")
    log.append(f"cell17 expression: {replace_cell(nb, 17, '# --- DEMO EXPRESSION MATRIX (synthetic, seeded) ---', CL_EXPRESSION_LOAD, 'cellline')}")
    return log


def patch_pk(nb) -> list[str]:
    log = []
    log.append(f"cell0 title: {replace_text(nb, 0, '''Every number below comes from a fixed-seed simulation, so the notebook is
fully reproducible; replace the demo profile in §1 with your own CSV and
re-run all cells.''', '''The demo profile ships with the toolkit (`examples/data/pk_profile.csv`,
a fixed-seed simulation regenerated deterministically by
`scratch/build_example_data.py`); replace it with your own CSV and
re-run all cells.''', 'pk')}")
    log.append(f"cell3 §1 md: {replace_text(nb, 3, '''The demo profile below is a one-compartment oral simulation with realistic
5 % assay noise — the generator is embedded in the cell so the notebook
re-runs anywhere offline. **Swap it for your own file and every cell below
runs unchanged.**''', '''The demo profile ships as `examples/data/pk_profile.csv` — a
one-compartment oral simulation with realistic 5 % assay noise,
regenerated deterministically by `scratch/build_example_data.py`.
**Swap it for your own file and every cell below runs unchanged.**''', 'pk')}")
    log.append(f"cell4 profile: {replace_cell(nb, 4, '# --- DEMO DATA ---', PK_PROFILE_LOAD, 'pk')}")
    return log


def patch_docking(nb) -> list[str]:
    log = []
    log.append(f"cell0 title: {replace_text(nb, 0, '''The demo campaign below is *synthetic but structurally identical* to real
AutoDock4 output, so every cell runs without a docking engine installed.
Point `DOCKING_DIR` at **your** campaign folder and re-run — the parsers,
rescoring and plots work unchanged on real `.dlg` trees.''', '''Every demo input ships under `examples/data/` — the ligand library
(`ligand_library.csv`), the receptor (`lysozyme_1aki.pdb`, the real RCSB
1AKI entry) and the demo campaign (`docking_campaign/`, synthetic but
structurally identical to real AutoDock4 output) — regenerated
deterministically by `scratch/build_example_data.py`, so every cell runs
without a docking engine installed. Point `DOCKING_DIR` at **your**
campaign folder and re-run — the parsers, rescoring and plots work
unchanged on real `.dlg` trees.''', 'docking')}")
    log.append(f"cell1 setup: {replace_text(nb, 1, '''import numpy as np
import pandas as pd''', 'import pandas as pd', 'docking')}")
    log.append(f"cell3 ligands: {replace_cell(nb, 3, '# A demo ligand library: 8 real drug-like molecules plus two', DK_LIGANDS_LOAD, 'docking')}")
    log.append(f"cell9 receptor: {replace_cell(nb, 9, '_LYSOZYME_BLOB = (', DK_RECEPTOR_LOAD, 'docking')}")
    log.append(f"cell28 §6 md: {replace_text(nb, 28, '''and so this
notebook can build a demo campaign whose `.dlg` files are byte-compatible
with the real format:''', '''and so the
toolkit ships a demo campaign (`examples/data/docking_campaign/`) whose
`.dlg` files are byte-compatible with the real format:''', 'docking')}")
    log.append(f"cell29 campaign: {replace_cell(nb, 29, '# --- DEMO CAMPAIGN ---', DK_CAMPAIGN_LOAD, 'docking')}")
    return log


def patch_generative(nb) -> list[str]:
    log = []
    log.append(f"cell7 §1b md: {replace_text(nb, 7, '''The demo table below is the same
12 named drugs shipped as `examples/data/seed_molecules.csv` — swap the
dictionary for your own CSV and every cell below runs unchanged.''', '''The demo table below loads the
12 named drugs shipped as `examples/data/seed_molecules.csv` — swap the
path for your own CSV and every cell below runs unchanged.''', 'generative')}")
    log.append(f"cell8 seeds: {replace_cell(nb, 8, '# --- DEMO SEEDS ---', GEN_SEEDS_LOAD, 'generative')}")
    return log


def patch_walkthrough(nb) -> list[str]:
    log = []
    log.append(f"cell0 title: {replace_text(nb, 0, '''This notebook exercises **every domain** of the `pharmaml` package on synthetic data:''', '''This notebook exercises **every domain** of the `pharmaml` package on the
demo files shipped under `examples/data/`:''', 'walkthrough')}")
    log.append(f"cell0 title2: {replace_text(nb, 0, '''No external datasets are required; every number below is generated with fixed seeds,
so the notebook is fully reproducible.''', '''No external datasets are required; every demo table ships with the toolkit
(regenerated deterministically by `scratch/build_example_data.py`), so the
notebook is fully reproducible — swap in your own files to make it yours.''', 'walkthrough')}")
    log.append(f"cell3 compounds: {replace_cell(nb, 3, '# Simulate a user file: descriptors + string target + SMILES metadata column.', WT_COMPOUNDS_LOAD, 'walkthrough')}")
    log.append(f"cell4 §2 md: {replace_text(nb, 4, '''Synthetic binary-activity table → stratified split → `random_forest` with 5-fold
cross-validated grid search → held-out evaluation → artifact round-trip.''', '''Shipped binary-activity table (`examples/data/walkthrough_classification.csv`) →
stratified split → `random_forest` with 5-fold cross-validated grid search →
held-out evaluation → artifact round-trip.''', 'walkthrough')}")
    log.append(f"cell5 binary: {replace_cell(nb, 5, 'from sklearn.datasets import make_classification', WT_CLASSIFICATION_LOAD, 'walkthrough')}")
    log.append(f"cell9 §3 md: {replace_text(nb, 9, '''and ROC-AUC switches to one-vs-rest.''', '''and ROC-AUC switches to one-vs-rest. The demo table ships as
`examples/data/walkthrough_multiclass.csv`.''', 'walkthrough')}")
    log.append(f"cell10 multiclass: {replace_cell(nb, 10, 'from sklearn.datasets import make_classification as _mk', WT_MULTICLASS_LOAD, 'walkthrough')}")
    return log


PATCHERS = {
    "complete_ml_workflow": patch_ml,
    "complete_cellline_workflow": patch_cellline,
    "complete_pk_workflow": patch_pk,
    "complete_docking_workflow": patch_docking,
    "complete_generative_workflow": patch_generative,
    "pharmaml_walkthrough": patch_walkthrough,
}


def execute_notebook(name: str) -> None:
    path = EXAMPLES / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(
        nb,
        timeout=2400,
        kernel_name="python3",
        resources={"metadata": {"path": str(EXAMPLES)}},
    )
    client.execute()
    nbformat.write(nb, path)
    print(f"executed: {name}.ipynb")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exec", action="store_true", help="re-execute after patching")
    parser.add_argument("--only", choices=list(PATCHERS), help="restrict to one notebook")
    parser.add_argument("--strip", action="store_true", help="regenerate no-output twins")
    args = parser.parse_args()

    for name, patcher in PATCHERS.items():
        if args.only and name != args.only:
            continue
        path = EXAMPLES / f"{name}.ipynb"
        nb = nbformat.read(path, as_version=4)
        log = patcher(nb)
        changed = any(entry.endswith("True") for entry in log)
        if changed:
            nbformat.write(nb, path)
        print(f"{name}: {'PATCHED' if changed else 'unchanged (already patched)'}")
        for entry in log:
            print(f"    {entry}")

    if args.exec:
        names = [args.only] if args.only else list(PATCHERS)
        for name in names:
            execute_notebook(name)

    if args.strip:
        subprocess.run(
            [sys.executable, str(ROOT / "scratch" / "strip_notebook_outputs.py")],
            check=True,
        )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.exit(main())
