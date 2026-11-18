"""Probe: print the real numbers the Docking Workbench tests will assert.

Run from the toolkit root: python scratch/probe_docking_211.py
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Lipinski

from pharmaml.docking import (
    DOCKSTRING_OBJECTIVES,
    DOCKSTRING_TARGETS,
    check_docking_completion,
    consensus_rescore,
    get_target,
    ligand_efficiency_series,
    load_ligand_csv,
    parse_dlg_cluster_table,
    parse_dockstring_search_box as parse_search_box,
    scan_docking_directory,
    target_frame,
)

TK = Path(__file__).resolve().parents[1]
CAMPAIGN = TK / "examples" / "data" / "docking_campaign"
LIBRARY = TK / "examples" / "data" / "ligand_library.csv"

print("== completion audit ==")
completion = check_docking_completion(CAMPAIGN)
print(completion["status"].value_counts().to_dict())
print("folders:", len(completion))

print("\n== campaign scan ==")
scan = scan_docking_directory(CAMPAIGN)
print("rows:", len(scan))
print("columns:", list(scan.columns))
print("status counts:", scan["status"].value_counts().to_dict())
ok = scan[scan["status"] == "OK"].reset_index(drop=True)
print("OK rows:", len(ok))
print(ok[["folder", "n_clusters", "best_rank", "best_lowest_energy", "best_num_in_cluster"]].to_string())

print("\n== heavy atoms (campaign ligands.csv) ==")
lig_names = pd.read_csv(CAMPAIGN / "ligands.csv")
heavy: dict[str, int] = {}
for _, row in lig_names.iterrows():
    mol = Chem.MolFromSmiles(str(row["SMILES"]))
    heavy[str(row["name"])] = 0 if mol is None else mol.GetNumHeavyAtoms()
print(heavy)

ok["ligand"] = ok["folder"].str.split("_", n=1).str[1]
ok["num_heavy_atoms"] = ok["ligand"].map(heavy)
print("\nany unmapped heavy atoms:", ok["num_heavy_atoms"].isna().sum())

print("\n== ligand efficiency series ==")
le = ligand_efficiency_series(ok, affinity_col="best_lowest_energy", atoms_col="num_heavy_atoms")
ok["ligand_efficiency"] = le
print(le.to_string())

print("\n== consensus rescore (3 evidence columns) ==")
cols = ["best_lowest_energy", "best_num_in_cluster", "ligand_efficiency"]
score = consensus_rescore(ok, cols)
ok["consensus"] = score
ranked = ok.sort_values("consensus", ascending=False).reset_index(drop=True)
print(ranked[["ligand", "best_lowest_energy", "best_num_in_cluster", "ligand_efficiency", "consensus"]].to_string())
top = ranked.iloc[0]
print(f"\nTOP: {top['ligand']} consensus={float(top['consensus']):.4f}")

print("\n== cluster table (01_aspirin) ==")
clusters = parse_dlg_cluster_table(CAMPAIGN / "01_aspirin" / "r.dlg")
for c in clusters:
    print(f"rank={c.rank} lowest={c.lowest_energy} mean={c.mean_energy} n={c.num_in_cluster}")
print("n_clusters:", len(clusters))
best = max(clusters, key=lambda c: c.num_in_cluster)
print(f"most populated: rank {best.rank} (n={best.num_in_cluster}), lowest {clusters[0].lowest_energy}")

print("\n== cluster table (05_caffeine) ==")
clusters_caf = parse_dlg_cluster_table(CAMPAIGN / "05_caffeine" / "r.dlg")
for c in clusters_caf:
    print(f"rank={c.rank} lowest={c.lowest_energy} mean={c.mean_energy} n={c.num_in_cluster}")

print("\n== DOCKSTRING registry ==")
frame = target_frame()
print("shape:", frame.shape)
print("objectives:", DOCKSTRING_OBJECTIVES)
print("n targets:", len(DOCKSTRING_TARGETS))
drd2 = get_target("DRD2")
print("DRD2 config:\n" + drd2.config_text())
first = frame.iloc[0]["target"]
print("first target:", first, "| last:", frame.iloc[-1]["target"])

print("\n== custom search box validation ==")
bad = "center_x = 1.0\nsize_y = 2.0\n"
try:
    parse_search_box(bad)
except Exception as exc:  # noqa: BLE001
    print(f"bad config error: {type(exc).__name__}: {exc}")
good = parse_search_box(drd2.config_text())
print("round-trip parse of DRD2 config:", good.center, good.size)

print("\n== ligand library (the prep card's input) ==")
lib = load_ligand_csv(LIBRARY)
print("rows:", len(lib), "valid:", len(lib.valid_frame), "invalid:", len(lib.invalid_frame))
print("smiles col:", lib.smiles_column, "| name col:", lib.name_column, "| validated:", lib.validated)
print("invalid rows:")
print(lib.invalid_frame[["name", "status", "error"]].to_string())
lib_ok = lib.valid_frame.copy()
lib_ok["num_heavy_atoms"] = [
    Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in lib_ok["smiles"]
]
lib_ok["n_rotatable"] = [
    Lipinski.NumRotatableBonds(Chem.MolFromSmiles(s)) for s in lib_ok["smiles"]
]
print(lib_ok[["name", "num_heavy_atoms", "n_rotatable"]].to_string())
print("\nmax heavy atoms:", lib_ok["num_heavy_atoms"].max(), "at", lib_ok.loc[lib_ok["num_heavy_atoms"].idxmax(), "name"])
