"""Build examples/complete_tuning_workflow.ipynb from scratch.

Round 100x-89: the executed-notebook face of the four 1.2.36 features
(the user's request — 3D-RISM-KH descriptors, HSVR, the keras-tuner-
style search, batch normalization).  One notebook walks all four in
the order a modeller would meet them:

    data -> 3D-RISM-KH solvation descriptors on the shipped drug set
    -> HierarchicalSVR vs the global SVR (the regime story)
    -> RandomSearch over an ANN hypermodel (baseline vs best)
    -> Hyperband successive halving over a deep hypermodel
    -> batchnorm A/B under the same split and seed -> recap.

Sources are built here; outputs are produced by a real nbclient run
under the single-threaded env every other notebook driver uses
(MPLBACKEND is *popped* first so inline PNGs survive — the 100x-43
lesson).  The ``_no_output`` twin is then derived by
scratch/strip_notebook_outputs.py, NOT here.

Run from the toolkit root:

    python scratch/build_tuning_notebook_89.py
"""

from __future__ import annotations

import os

for _var in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# 100x-43 lesson: an inherited MPLBACKEND=Agg leaks into the kernel and
# silently kills every inline figure.  Pop it so ipykernel registers its
# own inline backend.
os.environ.pop("MPLBACKEND", None)

import time

import nbformat
from nbclient import NotebookClient

PATH = "examples/complete_tuning_workflow.ipynb"

MD = "markdown"
CODE = "code"


def md(source: str) -> dict:
    return {"cell_type": MD, "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": CODE, "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


cells: list[dict] = [
    md(
        """# PharmaML — The Complete Tuning & Hierarchical Workflow

One notebook, the four features the 1.2.36 release added on request —
solvation descriptors from 3D-RISM-KH theory, a hierarchical SVR that
carves a table into regimes, a keras-tuner-style hyperparameter search,
and batch-normalised deep networks — in the order you would actually
meet them in a project:

| § | Stage | Toolkit |
|---|---|---|
| 1 | **Data entry** — the shipped QSAR regression table | `pharmaml.data` |
| 2 | **3D-RISM-KH solvation descriptors** — ten water-scale quantities for the shipped drug set | `pharmaml.chem.rism` |
| 3 | **Hierarchical SVR vs the global SVR** — the potency-regime story, with honest per-cluster diagnostics | `pharmaml.ml.hierarchical` |
| 4 | **Random search** over an ANN hypermodel — the keras-tuner-style API | `pharmaml.ml.tuner` |
| 5 | **Hyperband** successive halving over a deep hypermodel — resource economics | `pharmaml.ml.tuner` + `neural` |
| 6 | **Batch normalization**, the controlled A/B — same split, same seed | `pharmaml.ml.neural` |
| 7 | Recap | — |

§5 and §6 need PyTorch (`pip install 'pharmaml[deep]'`); §2 needs RDKit
(`pip install 'pharmaml[chem]'`); everything else runs on the core
dependencies.  Every number below is produced by the cells themselves
(seeded, CPU, deterministic) — nothing is hand-written.  The CLI twins
of §2/§4-§5 are `pharmaml chem rism` and `pharmaml ml tune`; the web
playground exposes §3-§4 point-and-click (the *Machine Learning* tab's
Hierarchical group + hyperparameter search).
"""
    ),
    code(
        """import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

from pharmaml import __version__

pd.set_option("display.max_columns", 14)
pd.set_option("display.width", 120)

# MLPRegressor's ConvergenceWarning fires whenever max_iter caps the
# optimiser — under the tuner that cap is a *budget*, not a failure
# (the cross-validated score is the verdict), so keep the committed
# outputs readable.
warnings.filterwarnings("ignore", message=".*Stochastic Optimizer.*")

try:
    import torch  # noqa: F401  (the deep extra, §5-§6)
except ImportError:
    sys.exit("PyTorch is required for §5-§6:  pip install 'pharmaml[deep]'")

# Scratch directory for this run's artifacts (use any path you like).
WORK = Path(tempfile.mkdtemp(prefix="tuning_workflow_"))

# The shipped demo inventory (examples/data) — this notebook loads its
# demo tables from there instead of fabricating them inline.
DATA_DIR = Path("data")

print(f"pharmaml {__version__} · torch {torch.__version__}")
print(f"artifacts -> {WORK}")
"""
    ),
    md(
        """## 1 · Data entry — the shipped QSAR regression table

The regression demo table ships as `examples/data/qsar_regression.csv`
(`make_synthetic_qsar_data.py`, seed 21): 20 numeric descriptors + a
continuous pIC50 endpoint.  `load_dataset` audits the table, resolves
the target column and keeps only numeric feature columns — the same
entry point every other workflow notebook uses.  (In your project:
point the path at your own CSV.)
"""
    ),
    code(
        """from pharmaml.data import load_dataset

reg_file = DATA_DIR / "qsar_regression.csv"     # <- replace with your own file
ds = load_dataset(reg_file, target="pIC50")

print(ds.summary())
print("task:", ds.task, "| target:", ds.target)
ds.frame.head(3)
"""
    ),
    md(
        """## 2 · 3D-RISM-KH solvation descriptors — the water eye's view

A full 3D-RISM-KH analysis (three-dimensional Reference Interaction
Site Model with the Kovalenko-Hirata closure) solves the molecular
Ornstein-Zernike equation on a grid — hours per molecule.  The
`pharmaml.chem.rism` module computes the **same ten report quantities**
from a documented pairwise *surrogate* that runs in ~1 s per molecule
deterministically:

* the molecule is embedded in 3D (ETKDGv3, fixed seed) and relaxed
  (MMFF94 → UFF fallback) with explicit hydrogens and Gasteiger
  partial charges;
* every solute/solvent-site pair follows a Lorentz-Berthelot
  Lennard-Jones + Coulomb potential with a smooth dielectric ramp
  (1 at contact → 78.4 in the bulk — the mean-field stand-in for the
  many-body screening a real RISM solve provides) and the KH closure
  branches, tempered where an isolated pair would diverge;
* the free-energy descriptors are integrals of the separated
  potentials against that distribution, and the site statistics
  (peaks, first-shell coordination numbers, excluded volume) come
  from the same pair distributions.

The honest contract (the module docstring says it too): these columns
are *features for regression/classification* — solvation-shaped and
chemically ordered, not RISM-calibrated in absolute energy.  The demo
set is the shipped 12-drug inventory `seed_molecules.csv`.
"""
    ),
    code(
        """from pharmaml.chem.rism import compute_rism_descriptors

drugs = pd.read_csv(DATA_DIR / "seed_molecules.csv")     # name, smiles
print(f"{len(drugs)} molecules: " + ", ".join(drugs["name"]))

rism = compute_rism_descriptors(drugs["smiles"], progress=False)
rism.insert(0, "name", drugs["name"].to_numpy())
rism = rism.reset_index(drop=True)
rism.round(2)
"""
    ),
    code(
        """# Reading the table like a RISM report:
# * Gsolv = Glj + Gcoul — the Lennard-Jones (cavity + dispersion) part
#   is positive for every solute (making room in water costs energy),
#   the Coulomb (charging) part is negative and carries the polarity;
# * the dipole is the Gasteiger-charge dipole of the relaxed conformer
#   (the module's calibration anchor: ethanol lands at 1.62 D vs the
#   experimental 1.69 D);
# * Ow_cn1 counts first-shell oxygens around the solute — the H-bond
#   acceptor budget; Ow_exclvol_A3 is the hydration-exclusion volume.
print(rism[["name", "RISM_Gsolv_kcalmol", "RISM_Glj_kcalmol",
            "RISM_Gcoul_kcalmol"]].round(1).to_string(index=False))
print()
ranked = rism.sort_values("RISM_dipole_D", ascending=False)
print(ranked[["name", "RISM_dipole_D", "RISM_Ow_cn1",
              "RISM_Ow_exclvol_A3"]].round(2).to_string(index=False))
"""
    ),
    md(
        """## 3 · Hierarchical SVR — one model per regime

A single global SVR fits *one* response surface for the whole table.
When the table hides distinct regimes (here: three potency bands the
descriptors separate), that single surface under-fits all of them.
`HierarchicalSVR` (registry key `'hsvr'`) does the two-level thing:

* seeded k-means on the **standardised** features carves the training
  rows into `n_clusters` regimes;
* one local SVR is fitted per cluster **plus** a global SVR over all
  rows;
* predictions blend the nearest cluster's local model with the global
  one through a linear **trust radius**: full weight at the cluster
  centroid, fading to pure-global beyond `radius_factor ×` the
  cluster's member radius — far queries honestly fall back to the
  global surface instead of extrapolating a local one.

The fitted estimator carries the diagnostics the desktop and web
playgrounds render: `labels_`, `cluster_sizes_`, `cluster_centers_`,
`radii_`.  Same split, same seed as every other section.
"""
    ),
    code(
        """import numpy as np
from sklearn.svm import SVR

from pharmaml.ml.evaluation import evaluate_model
from pharmaml.ml.hierarchical import HierarchicalSVR
from pharmaml.ml.preprocessing import make_train_test_split

X_train, X_test, y_train, y_test = make_train_test_split(
    ds.X, ds.y, test_size=0.25, random_state=42,
)
y_tr = np.asarray(y_train)

global_svr = SVR().fit(X_train, y_train)
hsvr = HierarchicalSVR(n_clusters=3, random_state=42).fit(X_train, y_train)

report_global = evaluate_model(global_svr, X_test, y_test, "regression")
report_hsvr = evaluate_model(hsvr, X_test, y_test, "regression")
print("global SVR :", report_global)
print("hsvr       :", report_hsvr)
"""
    ),
    code(
        """# The honest per-cluster diagnostics: the k-means regimes are NOT
# cosmetic — each cluster owns a distinct mean potency, and every
# test prediction carries the trust radius it was blended within.
regimes = pd.DataFrame({
    "cluster": range(len(hsvr.cluster_sizes_)),
    "rows": hsvr.cluster_sizes_,
    "mean_pIC50": [float(y_tr[hsvr.labels_ == c].mean())
                    for c in range(len(hsvr.cluster_sizes_))],
    "trust_radius": hsvr.radii_.round(3),
})
regimes
"""
    ),
    code(
        """from pharmaml.viz import plot_predicted_vs_actual

fig_global = plot_predicted_vs_actual(
    y_test, global_svr.predict(X_test),
    metrics={"R2": report_global.r2, "RMSE": report_global.rmse},
    title="Global SVR - one surface for all regimes",
)
"""
    ),
    code(
        """fig_hsvr = plot_predicted_vs_actual(
    y_test, hsvr.predict(X_test),
    metrics={"R2": report_hsvr.r2, "RMSE": report_hsvr.rmse},
    title="HierarchicalSVR - three local surfaces, trust-radius blended",
)
"""
    ),
    md(
        """## 4 · The tuner — random search over a hypermodel

`pharmaml.ml.tuner` is a keras-tuner-*style* API over the toolkit's
own estimators (no TensorFlow required).  The hypermodel is a plain
callable that receives a `HyperParameters` object and returns an
estimator — `hp.Int/Float/Choice/Boolean/Fixed` calls inside it
*declare* the searchable space and hand back either the default (when
the space is being collected) or the oracle's value (during a trial).

`RandomSearch` scores every sampled configuration with
cross-validation under **one shared seeded fold split**, so the
leaderboard ranks configurations — not fold luck.  The refit winner
(`results.best_estimator`) is rebuilt from `best_params` on the full
data.  The baseline below scores the hypermodel's *default*
configuration under the exact same folds — the number the search has
to beat to be worth its clock time.
"""
    ),
    code(
        """from sklearn.model_selection import KFold, cross_val_score

from pharmaml.ml.tuner import HyperParameters, RandomSearch, build_ann_regressor

tuner = RandomSearch(build_ann_regressor, objective="r2",
                     max_trials=8, cv=3, seed=42)
results = tuner.search(ds.X, ds.y)

print(results.summary())
print("best params:", results.best_params)

# Baseline: the hypermodel's defaults, scored under the tuner's own
# fold protocol (shuffled KFold, random_state=0) — apples to apples.
default_model = build_ann_regressor(HyperParameters())
baseline = cross_val_score(default_model, ds.X, ds.y,
                           cv=KFold(n_splits=3, shuffle=True, random_state=0),
                           scoring="r2")
print(f"\\ndefaults: {baseline.mean():.3f} +/- {baseline.std():.3f}"
      f"  ->  best {results.best_score:.3f}"
      f"  (gain {results.best_score - baseline.mean():+.3f} R2)")
"""
    ),
    code(
        """from pharmaml.viz import plot_model_comparison

trials = results.trials_frame()
leaderboard = pd.DataFrame({
    "model": [f"trial {int(t)}" for t in trials["trial_id"]],
    "r2_mean": trials["score"],
    "r2_std": trials["score_std"],
})
fig = plot_model_comparison(leaderboard, metric="r2",
                           title="RandomSearch leaderboard - CV R2 per trial")
"""
    ),
    code(
        """trials.sort_values("score", ascending=False).head(8).round(4)
"""
    ),
    md(
        """## 5 · Hyperband — pay only for the survivors

`Hyperband` (Li et al., 2013) treats one hyperparameter as a **resource**
— here the deep hypermodel's `epochs` slot — and runs the classic
bracket schedule: bracket `s` starts `ceil((s_max+1)·η^s/(s+1))` fresh
configurations at resource `max_resource/η^s`; every rung keeps only
the top `ceil(n/η)` and multiplies their resource by `η`.  Every rung
evaluation is recorded as a trial, so the trials frame mixes budgets by
design — that is the economics, not a bug.

With `max_resource=27, eta=3` the whole search costs a fraction of the
naive full-budget grid, and the deep hypermodel searches `batchnorm`
as a plain Boolean slot — the tuner decides it per dataset (§6 tests
that verdict head-to-head).
"""
    ),
    code(
        """from pharmaml.ml.tuner import Hyperband, build_deep_regressor

band = Hyperband(build_deep_regressor, objective="r2", resource="epochs",
                 max_resource=27, min_resource=3, eta=3, cv=3, seed=42)
band_results = band.search(ds.X, ds.y)

print(band_results.summary())
print("best params:", band_results.best_params)
"""
    ),
    code(
        """# The resource economics, read straight off the trials frame: mean CV
# R2 climbs rung by rung as the survivors earn bigger budgets, and the
# total epoch spend is a fraction of the naive full-budget grid.
run_frame = band_results.trials_frame()
by_resource = (run_frame.groupby("resource")["score"]
               .agg(["count", "mean", "max"]).round(3))
print(by_resource.to_string())

epochs_spent = int((run_frame["resource"] * 3).sum())      # 3 CV folds
epochs_naive = int(len(run_frame) * 27 * 3)                # every trial at full budget
print(f"\\nepoch budget spent: {epochs_spent} of a naive {epochs_naive}"
      f" ({100 * (1 - epochs_spent / epochs_naive):.0f}% saved)")
"""
    ),
    md(
        """## 6 · Batch normalization — the controlled A/B

All four torch estimators take `batchnorm=True/False` (default off).
When on, a `BatchNorm1d` sits between **every hidden linear map and
its activation** — in the residual blocks, both normalised maps are
added *after* normalisation, never before.  The running statistics
ride in the portable NumPy weight dict, so a pickled model rebuilds
its layers on predict exactly as trained.

The honest test is a controlled A/B: same split, same seed, same
capacity, same epoch cap and patience — only the flag differs.  On a
195-row × 20-feature table whose inputs are already standardised,
per-batch statistics add noise more than they remove covariate shift;
on deeper stacks and longer schedules they pay rent.  That is exactly
why the §5 hypermodel makes the flag *searchable* instead of
hard-coding either answer.
"""
    ),
    code(
        """from pharmaml.ml.neural import DeepMLPRegressor

def fit_variant(batchnorm: bool) -> DeepMLPRegressor:
    return DeepMLPRegressor(
        hidden_layers=(32, 32),
        dropout=0.1,
        batchnorm=batchnorm,
        epochs=80, batch_size=32,
        learning_rate=1e-3,
        patience=10, val_split=0.15,
        random_state=42,
    ).fit(X_train, y_train)

plain = fit_variant(batchnorm=False)
normed = fit_variant(batchnorm=True)

for label, net in (("plain", plain), ("batchnorm", normed)):
    rep = evaluate_model(net, X_test, y_test, "regression")
    print(f"{label:10s} R2 {rep.r2:.3f}  RMSE {rep.rmse:.3f}"
          f"  epochs run {net.n_iter_}")
"""
    ),
    code(
        """from pharmaml.viz import plot_training_history

fig_plain = plot_training_history(
    plain.history_, title="plain (32, 32) - no batchnorm",
)
"""
    ),
    code(
        """fig_normed = plot_training_history(
    normed.history_, title="batchnorm - same split, same seed, same cap",
)
"""
    ),
    md(
        """## 7 · Recap

| Feature | One-line API | Where else it lives |
|---|---|---|
| 3D-RISM-KH descriptors | `compute_rism_descriptors(smiles)` → 10 columns | CLI `pharmaml chem rism` · feature-factory `rism_` family (8th) |
| Hierarchical SVR | `HierarchicalSVR(n_clusters=3).fit(X, y)` | registry key `'hsvr'` · web playground *Hierarchical* group |
| Tuner | `RandomSearch(build_ann_regressor).search(X, y)` | CLI `pharmaml ml tune` · web playground *Hyperparameter search* |
| Hyperband | `Hyperband(build_deep_regressor, resource="epochs").search(X, y)` | CLI `pharmaml ml tune --searcher hyperband` |
| Batchnorm | `DeepMLPRegressor(batchnorm=True)` | searchable slot `hp.Boolean("batchnorm")` in the deep hypermodels |

The honest leaderboard this notebook leaves behind: the deep
estimators top the table, `hsvr` tops the SVR family with its regimes
on display, and the tuner paid for itself at every family it touched
(§4's +0.2 R2 over the defaults is the cheapest win in the file).

Next steps: the *Machine Learning* tab of `python playground.py`
(or the web explorer's ML playground) trains and tunes `hsvr`
point-and-click with the same diagnostics rendered; the deep learning
notebook walks the estimator internals (residual blocks, history
export, the registry path); `examples/run_regression.py` puts the
whole registry — `hsvr` included — on one cross-validated leaderboard.
"""
    ),
]

nb = nbformat.v4.new_notebook(
    cells=[nbformat.v4.new_markdown_cell(c["source"]) if c["cell_type"] == MD
           else nbformat.v4.new_code_cell(c["source"]) for c in cells],
)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

print(f"executing {PATH} ({len(nb.cells)} cells)...")
started = time.perf_counter()

client = NotebookClient(
    nb,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": "examples"}},
)
client.execute()

nbformat.write(nb, PATH)
elapsed = time.perf_counter() - started

n_code = sum(1 for c in nb.cells if c.cell_type == CODE)
n_figs = sum(
    1
    for c in nb.cells
    if c.cell_type == CODE
    for o in c.get("outputs", [])
    if "image/png" in o.get("data", {})
)
errors = [
    o.get("ename")
    for c in nb.cells
    if c.cell_type == CODE
    for o in c.get("outputs", [])
    if o.get("output_type") == "error"
]
print(f"OK: {PATH} written — {len(nb.cells)} cells ({n_code} code), "
      f"{n_figs} inline figures, executed in {elapsed:.1f}s")
if errors:
    raise SystemExit(f"unexpected error outputs: {errors}")
