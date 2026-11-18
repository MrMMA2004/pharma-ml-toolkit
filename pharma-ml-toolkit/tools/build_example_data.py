"""Rebuild the examples/data/ inventory after a sandbox reset.

Provenance discipline: every CSV's generation logic is EXTRACTED FROM the
corresponding notebook's inline data cell (same seeds, same code path), so
each file is byte-identical to what that cell wrote into its scratch WORK
directory.  The notebooks still carry the inline cells (round-23 state);
a later round converts them to read these files instead.

Source cells (notebook -> cell -> output file):

  examples/complete_ml_workflow.ipynb
      cell 3  -> qsar_classification.csv      420 x 11 (active 118 / weak 155 / inactive 147)
      cell 6  -> screening_library.csv        79-row SMILES library (77 valid + 2 planted invalid)
      cell 11 -> library_with_descriptors.csv 77 x 10 (SMILES + 8 key descriptors + activity)
      cell 31 -> qsar_multiclass.csv          360 x 21 (inactive 125 / weak 102 / moderate 79 / strong 54)
      cell 35 -> qsar_regression.csv          260 x 21, pIC50 target
  examples/complete_cellline_workflow.ipynb
      cell 3  -> cellline_screen.csv          185 rows (23 cell lines / 11 tissues)
      cell 17 -> cellline_expression.csv      23 x 300 expression matrix (same rng chain as cell 3)
  examples/complete_pk_workflow.ipynb
      cell 4  -> pk_profile.csv               49 rows, t = 0..24 h; AUC linear 17.9607 / log 17.2536
  examples/complete_docking_workflow.ipynb
      cell 3  -> ligand_library.csv           10-row ligand CSV (8 valid + 2 invalid rows)
      cell 29 -> docking_campaign/            14 folders: 12 complete runs + 2 planted audit cases
  examples/complete_generative_workflow.ipynb
      cell 8  -> seed_molecules.csv           12 rows, 443 B, NO trailing newline (pinned byte size)
  examples/pharmaml_walkthrough.ipynb
      cell 3  -> walkthrough_compounds.csv        180 x 8 (active 39 / inactive 88 / weak 53)
      cell 5  -> walkthrough_classification.csv   300 x 9
      cell 10 -> walkthrough_multiclass.csv       300 x 9

Also ensures examples/data/lysozyme_1aki.pdb (1AKI) and tests/data/1crn.pdb
(1CRN) exist, downloading from RCSB only when missing.

Properties: deterministic, idempotent (files are only rewritten when their
bytes actually change -- a second run touches nothing), cwd-independent
(all paths resolve from ``__file__``), single-threaded (worker-cap env vars
are set before numpy/sklearn/RDKit import).

``--check`` mode (100x-115): the non-mutating round-trip audit.  Nothing
is written and no network fetch happens; every file the builder owns is
compared byte-for-byte against what the generators produce, missing /
differing files are reported as DRIFT, and orphans (files on disk the
builder never writes) are flagged too.  Exit code 1 on any drift.  This
is the provenance-closure leg the pytest suite runs as a guard
(tests/test_data_provenance.py).

Drift attribution (100x-116): every drift line names the BUILDER
FUNCTION that owns the drifted file — ``... differs from the builder
output (owner: build_qsar_classification)`` — so a drift report is an
actionable pointer ("rerun that function, or re-extract its notebook
cell"), not just a file name.  ``STATS["owners"]`` carries the full
path -> function map for programmatic consumers.

Usage::

    python tools/build_example_data.py           # rebuild (idempotent)
    python tools/build_example_data.py --check    # audit only, no writes
"""

from __future__ import annotations

import os

# Single-thread determinism for any sklearn / RDKit / BLAS work below.
# Must be set before those libraries are imported, hence the import order.
for _var in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import functools
import hashlib
import importlib.util
import io
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../pharma-ml-toolkit
DATA_DIR = ROOT / "examples" / "data"
TESTS_DATA_DIR = ROOT / "tests" / "data"
_SRC = ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

# Byte-exact pins from the committed notebook outputs (guards drift loudly).
PIN_QSAR_CLS_CENSUS = {"active": 118, "weak": 155, "inactive": 147}
PIN_QSAR_MULTI_CENSUS = {"inactive": 125, "weak": 102, "moderate": 79, "strong": 54}
PIN_WALKTHROUGH_CENSUS = {"active": 39, "inactive": 88, "weak": 53}
PIN_AUC_LINEAR = 17.9607
PIN_AUC_LOG = 17.2536
PIN_SEED_BYTES = 443
PIN_DLG_BYTES = 682  # fixed-width cluster rows -> every complete r.dlg is exactly this
PIN_GLG_BYTES = 21
PIN_INTERRUPTED_DLG_BYTES = 65
PIN_CAMPAIGN_AUDIT = {"OK": 12, "MISSING_LOG": 1, "INCOMPLETE": 1}
PIN_ASPIRIN_SUMMARY = {  # parse_dlg_summary(01_aspirin/r.dlg), notebook cell 31 output
    "n_clusters": 5, "top_rank": 1, "top_lowest_energy": -4.91,
    "top_mean_energy": -4.74, "top_num_in_cluster": 15,
}
PIN_LYSOZYME_BYTES = 116_397
PIN_CRN_BYTES = 49_491

STATS = {
    "written": 0,
    "unchanged": 0,
    "fetched": [],
    "drift": [],   # --check mode only: files that differ / are missing
    "paths": [],   # every file the builder claims ownership of
    "owners": {},  # 100x-116: path -> builder function that owns it
}

# Flipped by main("--check"): writes become drift records instead.
CHECK_ONLY = {"active": False}

# 100x-116 attribution: the @_owned decorator sets this while a build
# function runs, so _write_if_changed/_fetch_pdb can name the culprit in
# every drift line.  "?" only appears if a write happens outside every
# decorated function — itself a bug worth seeing in a drift report.
_OWNER = {"name": "?"}


def _owned(fn):
    """Set _OWNER while ``fn`` runs (drift lines name the culprit)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        previous = _OWNER["name"]
        _OWNER["name"] = fn.__name__
        try:
            return fn(*args, **kwargs)
        finally:
            _OWNER["name"] = previous

    return wrapper


def _reset_stats() -> None:
    """Make main() safely re-runnable in one process (the pytest guard
    calls it twice: once clean, once with planted drift)."""
    STATS.update(written=0, unchanged=0, fetched=[], drift=[], paths=[], owners={})
    CHECK_ONLY["active"] = False
    _OWNER["name"] = "?"


def _fail(message: str) -> None:
    raise SystemExit(f"BUILD FAILED: {message}")


def _write_if_changed(path: Path, data: bytes) -> None:
    """Write bytes only when they differ from what is on disk (idempotency).

    In ``--check`` mode nothing is written: a differing or missing file
    is recorded in ``STATS["drift"]`` instead — naming the owning builder
    function (100x-116) — so the audit never mutates the tree it is
    auditing and every drift line is an actionable pointer.
    """
    STATS["paths"].append(path)
    STATS["owners"][str(path)] = _OWNER["name"]
    if path.is_file() and path.read_bytes() == data:
        STATS["unchanged"] += 1
        return
    if CHECK_ONLY["active"]:
        state = "differs from" if path.is_file() else "is missing from"
        STATS["drift"].append(
            f"{path} {state} the builder output (owner: {_OWNER['name']})"
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    STATS["written"] += 1


def _write_frame(path: Path, frame: pd.DataFrame, **to_csv: object) -> None:
    """Serialise a DataFrame exactly as ``to_csv`` would and write idempotently."""
    buffer = io.StringIO()
    frame.to_csv(buffer, **to_csv)  # type: ignore[arg-type]
    _write_if_changed(path, buffer.getvalue().encode("utf-8"))


def _check(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _load_qsar_generator():
    """Import examples/make_synthetic_qsar_data.py (the ML notebook's generator)."""
    spec = importlib.util.spec_from_file_location(
        "make_synthetic_qsar_data", ROOT / "examples" / "make_synthetic_qsar_data.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# complete_ml_workflow.ipynb -- cell 3 (demo QSAR table)
# ---------------------------------------------------------------------------
@_owned
def build_qsar_classification() -> None:
    rng = np.random.default_rng(11)
    n = 420
    signal = rng.normal(size=(n, 3))
    noise = rng.normal(scale=0.4, size=(n, 8))
    descriptors = noise.copy()
    descriptors[:, :3] += 0.6 * signal      # first 3 descriptors are informative
    frame_demo = pd.DataFrame(descriptors, columns=[f"descr_{i:02d}" for i in range(8)])
    score = frame_demo[["descr_00", "descr_01", "descr_02"]].sum(axis=1)
    frame_demo["activity"] = np.where(
        score > np.quantile(score, 0.72), "active",
        np.where(score > np.quantile(score, 0.35), "weak", "inactive"),
    )
    frame_demo["SMILES"] = rng.choice(["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"], size=n)
    frame_demo["compound_id"] = [f"CMPD-{i:04d}" for i in range(n)]
    _write_frame(DATA_DIR / "qsar_classification.csv", frame_demo, index=False)


# ---------------------------------------------------------------------------
# complete_ml_workflow.ipynb -- cell 6 (the 79-entry demo screening library)
# ---------------------------------------------------------------------------
LIBRARY = [
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
    ("metoprolol", "COCCc1ccc(OCC(C)NCC(C)C)cc1"),
    ("lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("procaine", "CCN(CC)CCOC(=O)c1ccc(N)cc1"),
    ("fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"),
    ("salicylic acid", "OC(=O)c1ccccc1O"),
    ("benzoic acid", "OC(=O)c1ccccc1"),
    ("benzaldehyde", "O=Cc1ccccc1"),
    ("benzamide", "NC(=O)c1ccccc1"),
    ("salicylamide", "NC(=O)c1ccccc1O"),
    ("methyl salicylate", "COC(=O)c1ccccc1O"),
    ("ethyl benzoate", "CCOC(=O)c1ccccc1"),
    ("phenyl acetate", "CC(=O)Oc1ccccc1"),
    ("acetophenone", "CC(=O)c1ccccc1"),
    ("benzophenone", "O=C(c1ccccc1)c1ccccc1"),
    ("benzyl alcohol", "OCc1ccccc1"),
    ("phenetole", "CCOc1ccccc1"),
    ("phenol", "Oc1ccccc1"),
    ("aniline", "Nc1ccccc1"),
    ("anisole", "COc1ccccc1"),
    ("toluene", "Cc1ccccc1"),
    ("ethylbenzene", "CCc1ccccc1"),
    ("propylbenzene", "CCCc1ccccc1"),
    ("o-xylene", "Cc1ccccc1C"),
    ("mesitylene", "Cc1c(C)c(C)cc(C)c1"),
    ("styrene", "C=Cc1ccccc1"),
    ("allylbenzene", "C=CCc1ccccc1"),
    ("chlorobenzene", "Clc1ccccc1"),
    ("bromobenzene", "Brc1ccccc1"),
    ("fluorobenzene", "Fc1ccccc1"),
    ("nitrobenzene", "O=[N+]([O-])c1ccccc1"),
    ("benzonitrile", "N#Cc1ccccc1"),
    ("biphenyl", "c1ccc(-c2ccccc2)cc1"),
    ("diphenyl ether", "O(c1ccccc1)c1ccccc1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
    ("indole", "c1ccc2[nH]ccc2c1"),
    ("quinoline", "c1ccc2ncccc2c1"),
    ("imidazole", "c1ncc[nH]1"),
    ("pyrrole", "c1cc[nH]c1"),
    ("furan", "c1ccoc1"),
    ("pyridine", "c1ccncc1"),
    ("pyrimidine", "c1cncnc1"),
    ("pyrazine", "c1cnccn1"),
    ("thiazole", "c1cscn1"),
    ("vanillin", "COc1cc(C=O)ccc1O"),
    ("eugenol", "COc1cc(CC=C)ccc1O"),
    ("guaiacol", "COc1ccccc1O"),
    ("catechol", "Oc1ccccc1O"),
    ("resorcinol", "Oc1cccc(O)c1"),
    ("hydroquinone", "Oc1ccc(O)cc1"),
    ("phthalic acid", "OC(=O)c1ccccc1C(=O)O"),
    ("cinnamic acid", "OC(=O)C=Cc1ccccc1"),
    ("cinnamaldehyde", "O=CC=Cc1ccccc1"),
    ("coumarin", "O=C1Oc2ccccc2C=C1"),
    ("sulfanilamide", "Nc1ccc(S(N)(=O)=O)cc1"),
    ("phenacetin", "CCOc1ccc(NC(C)=O)cc1"),
    ("glycine", "NCC(=O)O"),
    ("alanine", "CC(N)C(=O)O"),
    ("valine", "CC(C)CC(N)C(=O)O"),
    ("leucine", "CC(C)CCC(N)C(=O)O"),
    ("phenylalanine", "NC(Cc1ccccc1)C(=O)O"),
    ("tyrosine", "NC(Cc1ccc(O)cc1)C(=O)O"),
    ("serine", "NC(CO)C(=O)O"),
    ("histamine", "CNCCc1c[nH]cn1"),
    ("dopamine", "NCCc1ccc(O)c(O)c1"),
    ("acetylcholine", "CC[N+](C)(C)CCOC(C)=O"),
    ("broken-smiles", "not_a_smiles"),
    ("unclosed-ring", "C1CC"),
]


@_owned
def build_screening_library() -> None:
    library = pd.DataFrame(LIBRARY, columns=["name", "SMILES"])
    _write_frame(DATA_DIR / "screening_library.csv", library, index=False)


# ---------------------------------------------------------------------------
# complete_ml_workflow.ipynb -- cells 6 + 7 + 11 (descriptors + activity)
# ---------------------------------------------------------------------------
@_owned
def build_library_with_descriptors() -> None:
    from pharmaml.chem import compute_descriptors

    try:  # silence RDKit's C++ parser spam (same as the notebook cell)
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass

    smiles_series = pd.Series([smi for _, smi in LIBRARY], name="SMILES")
    descriptor_frame = compute_descriptors(smiles_series, progress=False)

    key_descriptors = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
                       "NumRotatableBonds", "LabuteASA", "RingCount"]
    key_frame = descriptor_frame[["SMILES"] + key_descriptors].round(2)
    key_frame.index = [LIBRARY[i][0] for i in descriptor_frame.index]
    key_frame.index.name = "compound"

    rng = np.random.default_rng(7)
    activity_score = (1.5 * key_frame["MolLogP"] + 1.0 * key_frame["NumHDonors"]
                      + rng.normal(scale=0.25, size=len(key_frame)))
    struct_table = key_frame.reset_index(drop=True)
    struct_table["activity"] = np.where(
        activity_score > np.median(activity_score), "active", "inactive"
    )
    _write_frame(DATA_DIR / "library_with_descriptors.csv", struct_table, index=False)


# ---------------------------------------------------------------------------
# complete_ml_workflow.ipynb -- cells 31 + 35 (multiclass + regression tables)
# ---------------------------------------------------------------------------
@_owned
def build_qsar_multiclass_and_regression() -> None:
    generator = _load_qsar_generator()

    # cell 31: Recipe B, K = 4 classes.
    multi = generator.make_classification_table(
        n_samples=360,
        n_classes=4,
        class_labels=["inactive", "weak", "moderate", "strong", "potent"][:4],
        seed=11,
    )
    _write_frame(DATA_DIR / "qsar_multiclass.csv", multi, index=False)

    # cell 35: continuous pIC50 endpoint.
    reg_table = generator.make_regression_table(n_samples=260, seed=21)
    _write_frame(DATA_DIR / "qsar_regression.csv", reg_table, index=False)


# ---------------------------------------------------------------------------
# complete_cellline_workflow.ipynb -- cells 3 + 17 (shared rng(42) chain)
# ---------------------------------------------------------------------------
CELL_LINES = {
    # name: tissue
    "MCF-7": "breast", "T-47D": "breast", "BT-549": "breast",
    "A549": "lung", "H1299": "lung", "H1975": "lung",
    "HepG2": "liver", "Huh7": "liver",
    "K-562": "haematopoietic", "Jurkat": "haematopoietic",
    "HCT-116": "colon", "HT-29": "colon", "DLD-1": "colon",
    "U-87 MG": "CNS", "U-251": "CNS",
    "PC-3": "prostate", "DU-145": "prostate",
    "MIA PaCa-2": "pancreas", "PANC-1": "pancreas",
    "SK-OV-3": "ovary", "OVCAR-3": "ovary",
    "HeLa": "cervix", "Saos-2": "bone",
}
DRUGS = {
    # name: (base LN_IC50, tissue sensitivity bonus, egfr coefficient)
    "erlotinib":  (3.10, {"lung": -0.45, "haematopoietic": +0.30}, -0.90),
    "gefitinib":  (3.30, {"lung": -0.30}, -0.55),
    "everolimus": (2.60, {"haematopoietic": -0.35, "breast": -0.15}, 0.0),
    "dasatinib":  (2.80, {"haematopoietic": -0.50}, 0.0),
    "sorafenib":  (3.90, {"liver": -0.25}, 0.0),
    "crizotinib": (3.60, {"lung": -0.20}, -0.25),
    "paclitaxel": (2.30, {"breast": -0.20, "ovary": -0.20}, 0.0),
    "cisplatin":  (2.90, {"CNS": +0.20}, 0.0),
}


@_owned
def build_cellline_data() -> None:
    # --- cell 3: the screen (one rng continues into cell 17's matrix) ---
    rng = np.random.default_rng(42)
    egfr_level = {
        name: rng.normal(9.0, 0.8) if tissue == "lung" else rng.normal(5.5, 0.9)
        for name, tissue in CELL_LINES.items()
    }
    egfr_z = (pd.Series(egfr_level) - pd.Series(egfr_level).mean()) / pd.Series(egfr_level).std()

    rows = []
    for cell, tissue in CELL_LINES.items():
        for drug, (base, tissue_map, egfr_coef) in DRUGS.items():
            ln_ic50 = base + tissue_map.get(tissue, 0.0) + egfr_coef * egfr_z[cell]
            ln_ic50 += rng.normal(0.0, 0.18)
            rows.append((cell, tissue, drug, ln_ic50))
    screen_frame = pd.DataFrame(rows, columns=["cell_line", "tissue", "compound", "LN_IC50"])

    # One duplicate pair (assay repeat) and one missing value, planted for §2.
    screen_frame = pd.concat(
        [screen_frame, screen_frame.iloc[[5]]], ignore_index=True
    )  # duplicate of row 5
    screen_frame.loc[100, "LN_IC50"] = np.nan
    _write_frame(DATA_DIR / "cellline_screen.csv", screen_frame, index=False)

    # --- cell 17: the expression matrix (same rng object continues) ---
    genes = [f"GENE_{i:03d}" for i in range(1, 299)] + ["EGFR", "KRT8"]
    base = rng.normal(6.0, 1.0, size=(len(CELL_LINES), len(genes) - 2))
    expression = pd.DataFrame(base, index=list(CELL_LINES), columns=genes[:-2])
    expression["EGFR"] = [egfr_level[c] for c in CELL_LINES]        # planted biomarker
    expression["KRT8"] = [                                        # planted breast marker
        rng.normal(9.0, 0.7) if t == "breast" else rng.normal(5.0, 0.8)
        for t in CELL_LINES.values()
    ]
    expression.index.name = "cell_line"
    _write_frame(DATA_DIR / "cellline_expression.csv", expression, index=True)


# ---------------------------------------------------------------------------
# complete_pk_workflow.ipynb -- cell 4 (oral concentration-time profile)
# ---------------------------------------------------------------------------
def make_pk_profile(ka: float = 1.2, ke: float = 0.35, f: float = 0.9, dose: float = 250.0,
                    vd: float = 35.0, duration_h: float = 24.0, step_h: float = 0.5,
                    error_pct: float = 0.05, seed: int = 7) -> pd.DataFrame:
    """One-compartment oral profile with proportional assay noise (cell verbatim)."""
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, duration_h + step_h, step_h)
    coefficient = f * dose * ka / (vd * (ka - ke))
    conc = coefficient * (np.exp(-ke * times) - np.exp(-ka * times))
    conc *= 1.0 + rng.normal(scale=error_pct, size=times.size)
    conc = np.clip(conc, 0.0, None)
    return pd.DataFrame({"time_h": times, "conc_ng_ml": conc})


@_owned
def build_pk_profile() -> None:
    # The notebook writes this via pharmaml.utils.io.write_table, which is
    # exactly frame.to_csv(path, index=False) -- same bytes, idempotent here.
    profile = make_pk_profile()
    _write_frame(DATA_DIR / "pk_profile.csv", profile, index=False)


# ---------------------------------------------------------------------------
# pharmaml_walkthrough.ipynb -- cells 3 + 5 + 10
# ---------------------------------------------------------------------------
# The walkthrough's SMILES metadata column vocabulary: the toolkit's own
# 12-molecule seed universe (seed_molecules.csv), pinned here so the builder
# stays order-independent.  The 100x-103 database audit found the previous
# constant "CCO(C)C" was an invalid SMILES (trivalent oxygen) repeated in all
# 180 rows -- zero diversity, zero validity.  Resampled from this vocabulary
# the column finally demos what a real user file's metadata column looks like.
WALKTHROUGH_SMILES_VOCAB = [
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",     # ibuprofen
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",     # caffeine
    "CN(C)C(=N)N=C(N)N",              # metformin
    "CN1CCCC1c1cccnc1",               # nicotine
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",  # warfarin
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C",     # theophylline
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",  # diclofenac
    "COc1ccc2cc(C(C)C(=O)O)ccc2c1",   # naproxen
    "CC(C)NCC(O)COc1cccc2ccccc12",    # propranolol
    "CCN(CC)CC(=O)Nc1c(C)cccc1C",     # lidocaine
]


@_owned
def build_walkthrough_tables() -> None:
    # cell 3: descriptors + string target + SMILES metadata column.
    rng = np.random.default_rng(5)
    user = pd.DataFrame(rng.normal(size=(180, 6)), columns=[f"descr_{i}" for i in range(6)])
    user["activity"] = rng.choice(["inactive", "weak", "active"], size=180, p=[0.45, 0.35, 0.20])
    # Drawn AFTER the activity draw: the descriptor + activity bytes stay
    # identical to the committed pins; only the metadata column changes.
    user["SMILES"] = rng.choice(WALKTHROUGH_SMILES_VOCAB, size=180)
    _write_frame(DATA_DIR / "walkthrough_compounds.csv", user, index=False)

    # cell 5: binary classification table.
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300, n_features=8, n_informative=5, flip_y=0.04, random_state=42
    )
    frame = pd.DataFrame(X, columns=[f"descr_{i:02d}" for i in range(X.shape[1])])
    frame["Target"] = y
    _write_frame(DATA_DIR / "walkthrough_classification.csv", frame, index=False)

    # cell 10: 3-class variant.
    X3, y3 = make_classification(
        n_samples=300, n_features=8, n_informative=6, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=8,
    )
    multi = pd.DataFrame(X3, columns=[f"descr_{i:02d}" for i in range(X3.shape[1])])
    multi["Target"] = y3
    _write_frame(DATA_DIR / "walkthrough_multiclass.csv", multi, index=False)


# ---------------------------------------------------------------------------
# complete_docking_workflow.ipynb -- cell 3 (demo ligand library CSV)
# ---------------------------------------------------------------------------
@_owned
def build_ligand_library() -> None:
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
    _write_frame(DATA_DIR / "ligand_library.csv", ligand_table, index=False)


# ---------------------------------------------------------------------------
# complete_generative_workflow.ipynb -- cell 8 (embedded seed table)
# ---------------------------------------------------------------------------
SEED_TABLE = {
    "aspirin":     "CC(=O)Oc1ccccc1C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen":   "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "caffeine":    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "metformin":   "CN(C)C(=N)N=C(N)N",
    "nicotine":    "CN1CCCC1c1cccnc1",
    "warfarin":    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "theophylline": "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "diclofenac":  "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "naproxen":    "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "propranolol": "CC(C)NCC(O)COc1cccc2ccccc12",
    "lidocaine":   "CCN(CC)CC(=O)Nc1c(C)cccc1C",
}


@_owned
def build_seed_molecules() -> None:
    """12 named drugs, pinned at 443 B with NO trailing newline.

    pandas' ``to_csv`` always appends a trailing newline (444 B) -- this
    byte size is pinned by the round-22 reconstruction and the generative
    notebook's "byte-identical content" claim, so the file is a manual
    join with the final newline stripped.
    """
    text = "name,smiles\n" + "".join(f"{name},{smiles}\n" for name, smiles in SEED_TABLE.items())
    data = text.rstrip("\n").encode("utf-8")
    _check(len(data) == PIN_SEED_BYTES,
           f"seed_molecules.csv is {len(data)} B, pinned size is {PIN_SEED_BYTES} B")
    _write_if_changed(DATA_DIR / "seed_molecules.csv", data)


# ---------------------------------------------------------------------------
# round 204/205 -- the Recalculate-from-metadata card's bundled example.
# The file shipped in round 204 but no builder path claimed it, so
# --check mode flagged it as an orphan (the provenance guard caught the
# gap the same day -- this step closes it).
# ---------------------------------------------------------------------------
_SELECTION_EXAMPLE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
    "CC(=O)Nc1ccc(O)cc1",      # paracetamol
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "COc1cc2c(cc1OC)C(=O)C(C)(O)CC2",  # a chromanone scaffold
    "c1ccc2c(c1)ncnc2N",       # quinazoline
    "CC(=O)Nc1nncc(c1C(C)=O)N",  # a heteroaryl amide
    "OC(=O)c1ccccc1N",         # anthranilic acid
]

_SELECTION_SOURCE_NOTE = (
    "Bundled example: variance-ranked top-24 of the ecfp + descriptors "
    "families over eight diverse drug-like molecules (aspirin, caffeine, "
    "paracetamol, ibuprofen, a chromanone scaffold, quinazoline, a "
    "heteroaryl amide, anthranilic acid) -- the Recalculate from metadata "
    "card loads this file so the roundtrip is one click."
)


@_owned
def build_selection_metadata() -> None:
    """The recalculation card's bundled example, via the factory's own saver.

    ``examples/data/selection_metadata.json`` is regenerated through the
    SAME pipeline the playground card drives: :func:`compute_feature_frame`
    over the eight fixed drug-like molecules, :func:`select_top_features`
    (variance, k=24 -- fully deterministic over a fixed corpus), then
    :func:`selection_metadata` for the JSON-ready dict.  ``source_note``
    and ``example_smiles`` are the builder's own provenance keys, appended
    after the saver's keys exactly as the round-204 file had them.
    """
    from pharmaml.chem.features import (
        compute_feature_frame,
        rank_features,
        select_top_features,
        selection_metadata,
    )

    frame, report = compute_feature_frame(
        _SELECTION_EXAMPLE_SMILES, families=("descriptors", "ecfp")
    )
    scores = rank_features(frame).reindex(frame.columns)  # variance; column order
    selection = select_top_features(frame, scores, k=24)
    metadata = selection_metadata(selection, report, k=24)
    metadata["source_note"] = _SELECTION_SOURCE_NOTE
    metadata["example_smiles"] = list(_SELECTION_EXAMPLE_SMILES)
    # round-204's file ends WITHOUT a trailing newline (json.dumps output
    # verbatim) -- kept byte-identical so --check stays clean.
    payload = json.dumps(metadata, indent=2).encode("utf-8")
    _write_if_changed(DATA_DIR / "selection_metadata.json", payload)


# ---------------------------------------------------------------------------
# complete_docking_workflow.ipynb -- cell 29 (synthetic AutoDock4 campaign)
# ---------------------------------------------------------------------------
def make_dlg(seed: int, n_clusters: int = 5, total_runs: int = 50) -> str:
    """Synthetic AutoDock4 DLG ending in the 'Successful Completion' sentinel."""
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
    return "\n".join(lines) + "\n"


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

BENZOCAINE_README = """\
# 08_benzocaine -- MISSING_LOG audit case

This run never produced its r.dlg / r.glg: the log was lost before the
audit, so the folder deliberately holds no docking output.
check_docking_completion reports MISSING_LOG here.  The campaign's other
planted audit case is 10_nicotine (INCOMPLETE).
"""

CAMPAIGN_README = """\
# Demo docking campaign (synthetic)

14 AutoDock4-style ligand folders (01_aspirin .. 14_vanillin); each
complete run holds a synthetic r.dlg cluster table + r.glg.

Audit cases:
  08_benzocaine  MISSING_LOG  (no r.dlg/r.glg; README inside)
  10_nicotine    INCOMPLETE   (interrupted before clustering)

Audit: 12 OK / 1 MISSING_LOG / 1 INCOMPLETE
"""


@_owned
def build_docking_campaign() -> None:
    campaign = DATA_DIR / "docking_campaign"
    for i, (name, _) in enumerate(CAMPAIGN_LIGANDS, start=1):
        folder = campaign / f"{i:02d}_{name}"
        if name == "benzocaine":
            # no r.dlg / r.glg written -> MISSING_LOG audit case
            _write_if_changed(folder / "README.md", BENZOCAINE_README.encode("utf-8"))
            continue
        if name == "nicotine":
            # no cluster table -> parses as INCOMPLETE
            _write_if_changed(
                folder / "r.dlg",
                b"AutoDock4 log file (demo)\n\n... run interrupted before clustering\n",
            )
            _write_if_changed(folder / "r.glg", b"autogrid4 log (demo)\n")
            continue
        dlg = make_dlg(seed=100 + i)
        _check(len(dlg.encode()) == PIN_DLG_BYTES,
               f"{folder.name}/r.dlg is {len(dlg.encode())} B, pinned ~{PIN_DLG_BYTES} B")
        _write_if_changed(folder / "r.dlg", dlg.encode("utf-8"))
        _write_if_changed(folder / "r.glg", b"autogrid4 log (demo)\n")

    # Campaign-root manifest + README explaining the two audit cases.
    manifest = pd.DataFrame(CAMPAIGN_LIGANDS, columns=["name", "SMILES"])
    _write_frame(campaign / "ligands.csv", manifest, index=False)
    _write_if_changed(campaign / "README.md", CAMPAIGN_README.encode("utf-8"))


# ---------------------------------------------------------------------------
# PDB fixtures (fetched from RCSB only when missing)
# ---------------------------------------------------------------------------
def _fetch_pdb(url: str, dest: Path, pinned_bytes: int) -> None:
    STATS["paths"].append(dest)
    STATS["owners"][str(dest)] = _OWNER["name"]
    if dest.is_file():
        if CHECK_ONLY["active"] and dest.stat().st_size != pinned_bytes:
            STATS["drift"].append(
                f"{dest} is {dest.stat().st_size} B, pinned size is {pinned_bytes} B "
                f"(owner: {_OWNER['name']})"
            )
        return
    if CHECK_ONLY["active"]:
        STATS["drift"].append(
            f"{dest} is missing (fetch skipped in --check mode) "
            f"(owner: {_OWNER['name']})"
        )
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    _check(len(data) == pinned_bytes,
           f"{dest.name} downloaded as {len(data)} B, pinned size is {pinned_bytes} B")
    dest.write_bytes(data)
    STATS["fetched"].append(str(dest))


@_owned
def ensure_pdbs() -> None:
    _fetch_pdb("https://files.rcsb.org/download/1AKI.pdb",
               DATA_DIR / "lysozyme_1aki.pdb", PIN_LYSOZYME_BYTES)
    _fetch_pdb("https://files.rcsb.org/download/1CRN.pdb",
               TESTS_DATA_DIR / "1crn.pdb", PIN_CRN_BYTES)


# ---------------------------------------------------------------------------
# Verification: re-read every file and pin the notebook-committed numbers
# ---------------------------------------------------------------------------
def verify() -> None:
    def read(name: str, **kwargs: object) -> pd.DataFrame:
        return pd.read_csv(DATA_DIR / name, **kwargs)  # type: ignore[arg-type]

    # QSAR tables (ML notebook pins).
    cls = read("qsar_classification.csv")
    _check(cls.shape == (420, 11), f"qsar_classification.csv shape {cls.shape} != (420, 11)")
    census = cls["activity"].value_counts().to_dict()
    _check(census == PIN_QSAR_CLS_CENSUS,
           f"qsar_classification.csv census {census} != {PIN_QSAR_CLS_CENSUS}")

    multi = read("qsar_multiclass.csv")
    _check(multi.shape == (360, 21), f"qsar_multiclass.csv shape {multi.shape} != (360, 21)")
    mcensus = multi["Target"].value_counts().to_dict()
    _check(mcensus == PIN_QSAR_MULTI_CENSUS,
           f"qsar_multiclass.csv census {mcensus} != {PIN_QSAR_MULTI_CENSUS}")

    reg = read("qsar_regression.csv")
    _check(reg.shape == (260, 21) and "pIC50" in reg.columns,
           f"qsar_regression.csv shape {reg.shape}, columns {list(reg.columns)[:3]}...")
    # 100x-103 database-audit pin: the endpoint stays inside the
    # assay-plausible window (pre-103 it ranged -4.3..14.7).
    _check(reg["pIC50"].between(3.0, 11.0).all(),
           f"qsar_regression.csv pIC50 outside [3, 11]: "
           f"min={reg['pIC50'].min():.3f} max={reg['pIC50'].max():.3f}")

    lib = read("screening_library.csv")
    _check(len(lib) == 79, f"screening_library.csv has {len(lib)} rows, pinned 79")

    struct = read("library_with_descriptors.csv")
    _check(struct.shape == (77, 10), f"library_with_descriptors.csv shape {struct.shape} != (77, 10)")
    aspirin = struct.iloc[0]
    for column, want in (("MolWt", 180.16), ("MolLogP", 1.31), ("TPSA", 63.60),
                         ("NumHDonors", 1), ("NumHAcceptors", 3)):
        got = float(aspirin[column])
        _check(abs(got - want) < 0.01,
               f"library_with_descriptors.csv aspirin {column} = {got}, pinned ~{want}")

    # Cell-line tables (cellline notebook pins).
    screen = read("cellline_screen.csv")
    _check(len(screen) == 185, f"cellline_screen.csv has {len(screen)} rows, pinned 185")
    _check(screen["cell_line"].nunique() == 23 and screen["tissue"].nunique() == 11,
           f"cellline_screen.csv {screen['cell_line'].nunique()} lines / "
           f"{screen['tissue'].nunique()} tissues, pinned 23 / 11")
    _check(screen["LN_IC50"].isna().sum() == 1, "cellline_screen.csv missing-value plant drifted")

    expr = read("cellline_expression.csv", index_col="cell_line")
    _check(expr.shape == (23, 300), f"cellline_expression.csv shape {expr.shape} != (23, 300)")

    # PK profile: the committed-output pins, computed through the package
    # (log value reflects the round-20a log-trapezoid fix; the notebook's
    # own committed output predates it).
    from pharmaml.pharma.pk import auc_trapezoid

    pk = read("pk_profile.csv")
    times = pk["time_h"].to_numpy()
    conc = pk["conc_ng_ml"].to_numpy()
    auc_linear = auc_trapezoid(times, conc)
    positive = conc > 0
    auc_log = auc_trapezoid(times[positive], conc[positive], method="log")
    _check(len(pk) == 49, f"pk_profile.csv has {len(pk)} rows, pinned 49")
    _check(float(times[0]) == 0.0 and float(conc[0]) == 0.0,
           "pk_profile.csv must start with a t=0 zero concentration sample")
    _check(abs(auc_linear - PIN_AUC_LINEAR) < 5e-5,
           f"pk_profile.csv AUC linear {auc_linear:.4f} != {PIN_AUC_LINEAR}")
    _check(abs(auc_log - PIN_AUC_LOG) < 5e-5,
           f"pk_profile.csv AUC log {auc_log:.4f} != {PIN_AUC_LOG}")

    # Walkthrough tables (walkthrough notebook pins).
    compounds = read("walkthrough_compounds.csv")
    _check(compounds.shape == (180, 8), f"walkthrough_compounds.csv shape {compounds.shape} != (180, 8)")
    wcensus = compounds["activity"].value_counts().to_dict()
    _check(wcensus == PIN_WALKTHROUGH_CENSUS,
           f"walkthrough_compounds.csv census {wcensus} != {PIN_WALKTHROUGH_CENSUS}")
    # 100x-103 database-audit pins: the SMILES metadata column must be the
    # 12-molecule seed vocabulary, every entry RDKit-valid (the pre-103 file
    # carried one invalid trivalent-oxygen string in all 180 rows).
    from rdkit import Chem

    wsmiles = compounds["SMILES"].astype(str).tolist()
    _check(set(wsmiles) == set(WALKTHROUGH_SMILES_VOCAB),
           "walkthrough_compounds.csv SMILES vocabulary drifted from the 12-molecule seed set")
    _check(len(wsmiles) == 180 and all(Chem.MolFromSmiles(s) is not None for s in wsmiles),
           "walkthrough_compounds.csv contains an invalid SMILES")
    _check(read("walkthrough_classification.csv").shape == (300, 9),
           "walkthrough_classification.csv shape drifted from (300, 9)")
    wm = read("walkthrough_multiclass.csv")
    _check(wm.shape == (300, 9) and wm["Target"].nunique() == 3,
           "walkthrough_multiclass.csv shape/classes drifted from (300, 9) / 3 classes")

    _check(len(read("ligand_library.csv")) == 10, "ligand_library.csv must hold 10 rows")

    seed_path = DATA_DIR / "seed_molecules.csv"
    _check(seed_path.stat().st_size == PIN_SEED_BYTES,
           f"seed_molecules.csv is {seed_path.stat().st_size} B, pinned {PIN_SEED_BYTES} B (no trailing newline)")
    _check(not seed_path.read_bytes().endswith(b"\n"), "seed_molecules.csv must NOT end with a newline")
    _check(len(read("seed_molecules.csv")) == 12, "seed_molecules.csv must hold 12 rows")

    # Docking campaign: audit with the package's own tooling.
    from pharmaml.docking import check_docking_completion, parse_dlg_summary
    from pharmaml.docking.parsing import parse_dlg_cluster_table
    from pharmaml.exceptions import DockingParseError

    campaign = DATA_DIR / "docking_campaign"
    audit = check_docking_completion(campaign)
    counts = audit["status"].value_counts().to_dict()
    _check(counts == PIN_CAMPAIGN_AUDIT,
           f"campaign audit {counts} != {PIN_CAMPAIGN_AUDIT}")
    _check(audit.set_index("folder").loc["10_nicotine", "status"] == "INCOMPLETE",
           "10_nicotine must parse as INCOMPLETE")
    _check(audit.set_index("folder").loc["08_benzocaine", "status"] == "MISSING_LOG",
           "08_benzocaine must report MISSING_LOG")
    aspirin_summary = parse_dlg_summary(campaign / "01_aspirin" / "r.dlg")
    subset = {key: aspirin_summary[key] for key in PIN_ASPIRIN_SUMMARY}
    _check(subset == PIN_ASPIRIN_SUMMARY,
           f"01_aspirin summary {subset} != notebook pin {PIN_ASPIRIN_SUMMARY}")
    for i, (name, _) in enumerate(CAMPAIGN_LIGANDS, start=1):
        dlg = campaign / f"{i:02d}_{name}" / "r.dlg"
        if name == "benzocaine":
            _check(not dlg.is_file(), "08_benzocaine must have no r.dlg")
            continue
        if name == "nicotine":
            _check(dlg.stat().st_size == PIN_INTERRUPTED_DLG_BYTES,
                   f"10_nicotine/r.dlg is {dlg.stat().st_size} B, pinned {PIN_INTERRUPTED_DLG_BYTES}")
            unparseable = False
            try:
                parse_dlg_cluster_table(dlg)
            except DockingParseError:
                unparseable = True
            _check(unparseable, "10_nicotine/r.dlg must NOT parse (interrupted before clustering)")
            continue
        _check(dlg.stat().st_size == PIN_DLG_BYTES,
               f"{i:02d}_{name}/r.dlg is {dlg.stat().st_size} B, pinned {PIN_DLG_BYTES}")
        glg = campaign / f"{i:02d}_{name}" / "r.glg"
        _check(glg.stat().st_size == PIN_GLG_BYTES,
               f"{i:02d}_{name}/r.glg is {glg.stat().st_size} B, pinned {PIN_GLG_BYTES}")

    # batch_run_directories smoke: idempotent re-run -- existing outputs are
    # skipped.  Benzocaine always fails loudly with "missing r.gpf" (it has
    # no files at all).  Nicotine's outcome depends on the package's
    # dlg-trust semantics: round-23 skipped any existing r.dlg (notebook's
    # committed "10_nicotine -> ok"); the sentinel-trust rule added later
    # re-runs interrupted logs and fails on the missing r.dpf instead.
    # Both outcomes confirm the fixture, so only the stable parts are pinned.
    from pharmaml.docking import batch_run_directories

    results = batch_run_directories(campaign, workers=2)
    by_folder = {r.directory.name: r for r in results}
    _check(len(by_folder) == 14, f"batch_run_directories saw {len(by_folder)} folders, expected 14")
    for i, (name, _) in enumerate(CAMPAIGN_LIGANDS, start=1):
        result = by_folder[f"{i:02d}_{name}"]
        if name == "benzocaine":
            _check(not result.ok and "r.gpf" in (result.error or ""),
                   f"benzocaine batch failure should be the missing r.gpf, got {result.error!r}")
        elif name == "nicotine":
            note = "ok (existing r.dlg trusted)" if result.ok else f"not ok ({result.error})"
            print(f"[batch] 10_nicotine -> {note}")
        else:
            _check(result.ok, f"{i:02d}_{name} should re-run as ok, got {result.error!r}")


def print_inventory() -> None:
    files = sorted(p for p in DATA_DIR.rglob("*") if p.is_file())
    total = 0
    print("\n=== examples/data inventory ===")
    for path in files:
        size = path.stat().st_size
        total += size
        digest = hashlib.md5(path.read_bytes()).hexdigest()[:8]
        shape = ""
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            shape = f"  {frame.shape[0]}x{frame.shape[1]}"
        print(f"{path.relative_to(DATA_DIR)!s:<44}{shape:<10}{size:>9} B  md5:{digest}")
    print(f"--- {len(files)} files, {total:,} B total")


def _drift_report() -> int:
    """--check mode gate: 0 when disk == builder output, else 1.

    Beyond byte-identity of the owned files, this also flags ORPHANS --
    files sitting under ``examples/data`` that no builder path claims --
    so the inventory can never silently grow a file the notebooks do
    not know about.
    """
    disk = {p for p in DATA_DIR.rglob("*") if p.is_file()}
    owned = {Path(p) for p in STATS["paths"]}
    for orphan in sorted(disk - owned):
        STATS["drift"].append(f"{orphan} is on disk but the builder never writes it")
    drift = STATS["drift"]
    if drift:
        print(f"DRIFT: {len(drift)} problem(s) between examples/data and the builder output:")
        for line in drift:
            print(f"  - {line}")
        return 1
    print(
        f"check OK: {STATS['unchanged']} file(s) byte-identical to the builder output, "
        f"{len(owned)} owned, no orphans under {DATA_DIR}"
    )
    return 0



# -- the contact-map/PLIF demo pose (r264) --------------------------------
# Ten synthetic atoms placed clash-free (>= 2.4 A) at 2.9-4.2 A from
# real 1AKI binding-cleft atoms by the r264 sphere-search placement
# (seed 20261111, one-shot; embedded here as immutable content the way
# the campaign README is -- the geometry IS the deliverable).  The
# receptor side is examples/data/lysozyme_1aki.pdb; the REMARK lines
# inside carry the provenance.
CONTACTS_DEMO_LIGAND_PDBQT = (
    b"REMARK  contacts_demo_ligand -- SYNTHETIC ligand-pose geometry for the\n"
    b"REMARK  contact-map/PLIF demo: ten atoms placed at 2.9-4.2 A from real\n"
    b"REMARK  1AKI atoms (ASP52/101 carboxylates, ASN59/103 donors, TRP62/63/108\n"
    b"REMARK  walls, VAL109/ALA107 carbons), clash-free (no receptor atom closer\n"
    b"REMARK  than 2.4 A). Not a bonded molecule -- a documented geometry demo,\n"
    b"REMARK  the twin of the synthetic r.dlg campaign. Receptor side:\n"
    b"REMARK  examples/data/lysozyme_1aki.pdb (real hen egg-white lysozyme, 1.5 A).\n"
    b"ATOM      1   N1 LIG L   1      22.400  37.419  12.054  1.00  0.00    0.000 NA\n"
    b"ATOM      2   N2 LIG L   1      16.981  27.436  -3.992  1.00  0.00    0.000 NA\n"
    b"ATOM      3   O2 LIG L   1      15.833  26.612  13.856  1.00  0.00    0.000 OA\n"
    b"ATOM      4   N3 LIG L   1      13.150  28.904   7.502  1.00  0.00    0.000 ND\n"
    b"ATOM      5   C4 LIG L   1      12.879  35.184   2.240  1.00  0.00    0.000  C\n"
    b"ATOM      6   C5 LIG L   1      19.582  31.320   1.114  1.00  0.00    0.000  C\n"
    b"ATOM      7   C6 LIG L   1      26.593  23.046   2.990  1.00  0.00    0.000  C\n"
    b"ATOM      8   N4 LIG L   1      18.082  32.433  -0.803  1.00  0.00    0.000 ND\n"
    b"ATOM      9   C7 LIG L   1      18.579  31.761   4.768  1.00  0.00    0.000  C\n"
    b"ATOM     10   C8 LIG L   1      14.850  24.554  -4.476  1.00  0.00    0.000  C\n"
    b"TER\n"
)

PIN_CONTACTS_DEMO_BYTES = 1303


@_owned
def build_contacts_demo_ligand() -> None:
    data = CONTACTS_DEMO_LIGAND_PDBQT
    _check(len(data) == PIN_CONTACTS_DEMO_BYTES,
           f"contacts_demo_ligand.pdbqt is {len(data)} B, pinned {PIN_CONTACTS_DEMO_BYTES} B")
    _write_if_changed(DATA_DIR / "contacts_demo_ligand.pdbqt", data)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    unknown = [a for a in argv if a != "--check"]
    if unknown:
        raise SystemExit(f"unknown argument(s) {unknown} — only --check is supported")
    check_only = "--check" in argv
    _reset_stats()
    CHECK_ONLY["active"] = check_only
    ensure_pdbs()
    build_contacts_demo_ligand()
    build_qsar_classification()
    build_screening_library()
    build_library_with_descriptors()
    build_qsar_multiclass_and_regression()
    build_cellline_data()
    build_pk_profile()
    build_walkthrough_tables()
    build_ligand_library()
    build_seed_molecules()
    build_selection_metadata()
    build_docking_campaign()
    if check_only:
        if _drift_report() != 0:
            raise SystemExit(1)
        verify()
        print("check OK: verify pins passed (byte-identity + orphans + content)")
        return
    verify()
    print_inventory()
    fetched = STATS["fetched"]
    print(f"\nbuilder OK: {STATS['written']} file(s) written, "
          f"{STATS['unchanged']} unchanged (byte-identical)"
          + (f", fetched {fetched}" if fetched else ""))


if __name__ == "__main__":
    main()
