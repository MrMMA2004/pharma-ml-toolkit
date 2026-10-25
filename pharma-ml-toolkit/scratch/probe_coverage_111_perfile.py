#!/usr/bin/env python3
"""Per-file GUI coverage audit: every file of the rar -> playground.py surface.

Task ID: 100x-111 — the user asked to "check that playground.py covers the
entire contents (each file) of the zip and rar files".

Method (the standing contract from docs/COVERAGE.md):
  1. fresh-extract the pristine rar (done -> /tmp/rar_audit)
  2. for EACH of the 118 files, use the established library mapping
     (COVERAGE.md matrix, audited one-by-one in 100x-101)
  3. grep playground.py for the mapped GUI symbol(s)
  4. report every file: COVERED (hit count) / ZERO-HIT (gap) / ABSORBED (documented non-goal)

100x-118 extension — the uploads axis: the 2026-10 user upload
Applicability_domain_analysis.py joins the contract as file #119 (the
user's standing rule extends to every file they hand over, not just the
two archives).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

RAR_ROOT = Path("/tmp/rar_audit/کدهای ماشین لرنینگ")
PLAYGROUND = Path("playground.py")

src = PLAYGROUND.read_text(encoding="utf-8")

# ---------------------------------------------------------------
# per-file -> expected playground.py symbol(s)
# (mapping per docs/COVERAGE.md, the 100x-101 one-by-one audit;
#  every file's row from the rar matrix, family surface per 100x-110)
# ---------------------------------------------------------------
# model-registry scripts: the dynamic combobox + train/compare/persist
REG = ["list_models", "build_model", "train_model"]
ENSEMBLE = ["build_stacking", "build_voting"]
REUSE = ["load_model"]
PERSIST = ["save_model"]
DESC = ["compute_descriptors", "compute_fingerprints"]

FILE_MAP: dict[str, list[str]] = {
    # --- Classification/ ---
    "Classification/RF.py": REG,
    "Classification/SVM.py": REG,
    "Classification/XGboost.py": REG,
    "Classification/LightGBM.py": REG,
    "Classification/CATboost.py": REG,
    "Classification/ADAboost.py": REG,
    "Classification/KNN.py": REG,
    "Classification/Decision tree.py": REG,
    "Classification/Logistic regression.py": REG,
    "Classification/Ridge.py": REG,
    "Classification/Lasso.py": REG,
    "Classification/Elastic net.py": REG,
    "Classification/ANN.py": REG,
    "Classification/Stacking.py": ENSEMBLE,
    "Classification/Voting.py": ENSEMBLE,
    "Classification/ANN/ANN class.py": REG,
    "Classification/ANN/reuse.py": REUSE,
    "Classification/Logestic Regression/Logistic regression class done.py": REG + PERSIST,
    "Classification/Logestic Regression/reuse.py": REUSE,
    "Classification/Stacking/stacking class.py": ENSEMBLE,
    "Classification/Stacking/reuse.py": REUSE,
    "Classification/Descriptor/Descriptor classification.py": DESC,
    "Classification/Descriptor/Descriptor reuse.py": DESC,
    "Classification/Descriptor/Descriptor reuse(اگه اونیکی کار نکرد).py": DESC,
    "Classification/BBB/BBB_Prediction_model.ipynb": REG + ["compare_models"],
    "Classification/BBB/tuned_random_forest_bbb_prediction_model.pkl": ["save_model"],  # absorbed: dataset-specific weights
    "Descriptor classification.py": DESC,
    # --- Regression/ ---
    "Regression/RF.py": REG,
    "Regression/SVM.py": REG,
    "Regression/SVM/SVM reg.py": REG,
    "Regression/SVM/reuse.py": REUSE,
    "Regression/XGboost.py": REG,
    "Regression/LightGBM.py": REG,
    "Regression/CATboost.py": REG,
    "Regression/ADAboost.py": REG,
    "Regression/KNN.py": REG,
    "Regression/Decision tree.py": REG,
    "Regression/Linear regression.py": REG,
    "Regression/Ridge.py": REG,
    "Regression/Lasso.py": REG,
    "Regression/Elastic net.py": REG,
    "Regression/Huber.py": REG,
    "Regression/ANN.py": REG,
    "Regression/ANN2.py": REG,
    "Regression/Stacking.py": ENSEMBLE,
    "Regression/Stacking/Stacking reg.py": ENSEMBLE,
    "Regression/Stacking/reuse.py": REUSE,
    "Regression/Voting.py": ENSEMBLE,
    "Regression/Voting/Voting reg.py": ENSEMBLE,
    "Regression/Voting/reuse.py": REUSE,
    "Regression/Descriptor/Descriptor regression.py": DESC,
    "Regression/Descriptor/Descriptor reuse.py": DESC,
    "Descriptor regression.py": DESC,
    # --- root chemistry/data utilities ---
    "Convert CSV to SDF.py": ["table_to_sdf"],
    "Convert IC50(nM) to pIC50.py": ["ic50_to_pic50", "add_pic50_column"],
    "Convert SMI to CSV.py": ["smi_to_csv"],
    "Non-numerical data remover.py": ["clean_numeric_frame"],
    "Split CSV file.py": ["split_table_into_chunks"],
    "Tanimoto calculator.py": ["tanimoto_matrix", "similarity_scan", "top_matches"],
    "Tanimoto-edited.py": ["similarity_scan"],
    # --- others/ ---
    "others/adder.py": ["distribute_files"],
    "others/compare/notcommon.py": ["compare_tables"],
    "others/compare/smilesgetter.py": ["annotate_with_smiles"],
    "others/dlg_extracter.py": ["collect_files"],
    "others/Prettier_extracted_dlgs.py": ["reindex_by_number"],
    "others/reader_at_same_folder.py": ["scan_docking_directory"],
    "others/gpfer.py": ["write_gpf", "write_config_batch"],
    "others/dpfer.py": ["write_dpf", "write_config_batch"],
    "others/main.py": ["batch_run_directories"],
    "others/ligand_preparation_autodock4.py": ["prepare_ligand_library"],
    "others/Readme.txt": [],  # absorbed: README.md + home guide
    "others/Docking/checker.py": ["check_docking_completion"],
    "others/Docking/dpfer.py": ["write_dpf", "write_config_batch"],
    "others/Docking/gpfer.py": ["write_gpf", "write_config_batch"],
    "others/Docking/gpferreceptorni.py": ["write_gpf", "write_config_batch", "receptor_types"],
    "others/Docking/gpfreader.py": ["read_gpf", "scan_gpf_ligand_types"],
    "others/Docking/main.py": ["batch_run_directories"],
    "others/Docking/reader.py": ["parse_dlg_cluster_table"],
    "others/Docking/reader_at_same_folder.py": ["scan_docking_directory"],
    "others/Docking/readerwithdetails.py": ["parse_dlg_cluster_table", "parse_dlg_summary"],
    "others/Docking/smilestopdbqt.py": ["smiles_to_pdbqt"],
    "others/Docking/smiqt.py": ["smiles_to_pdbqt"],
    "others/Docking/Prettier_extracted_dlgs.py": ["reindex_by_number"],
    "others/Docking/ligand_pdbqt_extracter.py": ["collect_files"],
    "others/Docking/ligand_preparation_autodock4.py": ["prepare_ligand_library"],
    "others/Docking/New Text Document.txt": [],  # empty placeholder
    "others/Docking/ligand generation/mismatch2.py": ["check_torsion_consistency", "scan_torsion_consistency"],
    "others/Docking/ligand generation/nigga.py": ["prepare_ligand_library", "table_to_sdf"],
    "others/Docking/ligand generation/nigga2.py": ["prepare_ligand_library", "table_to_sdf"],
    "others/Docking/ligand generation/niggamatch.py": ["check_torsion_consistency", "scan_torsion_consistency"],
    "others/Docking/ligand generation/niggaqt.py": ["prepare_ligand_library"],  # absorbed: MGLTools deprecated
    "others/General/CSV_merger.py": ["merge_tables"],
    "others/General/Convert SMI to CSV.py": ["smi_to_csv"],
    "others/General/Nameextracter.py": ["list_file_names"],
    "others/General/Prettier_extracted_dlgs.py": ["reindex_by_number"],
    "others/General/Split CSV file.py": ["split_table_into_chunks"],
    "others/General/adder.py": ["distribute_files"],
    "others/General/approved.py": ["filter_approved_drugs", "export_docx"],
    "others/General/common_between_two_excel.py": ["compare_tables"],
    "others/General/delete.py": ["cleanup_tree"],
    "others/General/deleter.py": ["delete_named_files"],
    "others/General/extaacter.py": ["collect_files"],
    "others/General/filemerger.py": ["merge_table_folder"],
    "others/General/reader_at_same_folder.py": ["scan_docking_directory"],
    "others/Ml/Descriptor calculator.py": DESC,
    "others/Ml/Descriptoredited.py": DESC,
    "others/Ml/ECFP_calculation.py": ["compute_fingerprints"],
    "others/Ml/Mordred_calculation.py": DESC,
    "others/Ml/Morgan_calculator.py": ["compute_fingerprints"],
    "others/Ml/New Text Document.txt": [],  # empty placeholder
    "others/Ml/Sdf reading.ipynb": ["sdf_to_table"],
    "others/Ml/Tanimoto calculator.py": ["tanimoto_matrix"],
    "others/Ml/Tanimotoedited.py": ["similarity_scan"],
    "others/Reader for alphabetic folders/dlg_extracter_not_numeral.py": ["scan_docking_directory"],
    "others/Reader for alphabetic folders/readerwithdetails.py": ["parse_dlg_cluster_table"],
    "others/Vina/vina_reader.py": ["parse_vina_output", "scan_vina_directory"],
    "others/Vina/vinar.py": ["run_vina"],
    "others/Vina/vinar_multiple.py": ["run_vina_batch"],
}

ABSORBED = {
    "Classification/BBB/tuned_random_forest_bbb_prediction_model.pkl",
    "others/Readme.txt",
    "others/Docking/New Text Document.txt",
    "others/Ml/New Text Document.txt",
    "others/Docking/ligand generation/niggaqt.py",  # MGLTools deprecated -> Meeko path
}

# ---------------------------------------------------------------
# 100x-118: the uploads axis — user files outside the two archives.
# The 2026-10 upload Applicability_domain_analysis.py maps to the AD
# card's GUI surface (the card, its run/save handlers, the library
# functions the card calls, the figure, and the CLI twin).
# ---------------------------------------------------------------
UPLOAD_ROOT = Path(str(Path(__file__).resolve().parents[2] / "upload"))
UPLOAD_MAP: dict[str, list[str]] = {
    "Applicability_domain_analysis.py": [
        "applicability_domain",          # the composed workflow (card work())
        "plot_applicability_domain",     # the combined two-panel figure
        "_ml_ad_run",                    # the Analyse button
        "_ml_ad_save",                   # the results CSV button
        "_ml_ad_refresh_source",         # the source line + mode switch
        "_ad_source_combo",              # last-run vs loaded-bundle modes
        "_ad_boot_spin",                 # the bootstrap-ensemble knob
        "Applicability_Domain_Results",  # the upload's output name
    ],
}


def count_hits(symbol: str) -> int:
    return len(re.findall(r"\b" + re.escape(symbol) + r"\b", src))


def main() -> int:
    # 1. inventory: every file actually inside the fresh extraction
    actual = sorted(
        str(p.relative_to(RAR_ROOT)) for p in RAR_ROOT.rglob("*") if p.is_file()
    )
    print(f"files in fresh rar extraction: {len(actual)}")

    mapped = set(FILE_MAP)
    missing_map = [f for f in actual if f not in mapped]
    stale_map = [f for f in mapped if f not in actual]

    zero_hit: list[tuple[str, list[str]]] = []
    rows: list[tuple[str, str, str]] = []

    for rel in actual:
        syms = FILE_MAP.get(rel)
        if syms is None:
            rows.append((rel, "NO-MAPPING", ""))
            continue
        if not syms:
            note = "absorbed (documented non-goal)"
            rows.append((rel, "ABSORVED" if rel in ABSORBED else "EMPTY", note))
            continue
        hits = {s: count_hits(s) for s in syms}
        if all(h == 0 for h in hits.values()):
            zero_hit.append((rel, syms))
            rows.append((rel, "ZERO-HIT", ",".join(syms)))
        else:
            detail = " ".join(f"{s}:{h}" for s, h in hits.items())
            rows.append((rel, "COVERED", detail))

    # 100x-118: the uploads axis — every uploaded file must be covered too
    upload_rows: list[tuple[str, str, str]] = []
    upload_zero: list[tuple[str, list[str]]] = []
    for rel, syms in sorted(UPLOAD_MAP.items()):
        if not (UPLOAD_ROOT / rel).is_file():
            upload_rows.append((rel, "MISSING-UPLOAD", ""))
            upload_zero.append((rel, syms))
            continue
        hits = {s: count_hits(s) for s in syms}
        if all(h == 0 for h in hits.values()):
            upload_zero.append((rel, syms))
            upload_rows.append((rel, "ZERO-HIT", ",".join(syms)))
        else:
            detail = " ".join(f"{s}:{h}" for s, h in hits.items())
            upload_rows.append((rel, "COVERED", detail))

    print("\n=== PER-FILE RESULT ===")
    for rel, status, detail in rows:
        print(f"{status:9s} {rel}  {detail}")
    print("\n=== UPLOADS AXIS (100x-118) ===")
    for rel, status, detail in upload_rows:
        print(f"{status:16s} {rel}  {detail}")

    n_cov = sum(1 for r in rows if r[1] == "COVERED")
    n_abs = sum(1 for r in rows if r[1] in ("ABSORVED", "EMPTY"))
    n_zero = sum(1 for r in rows if r[1] == "ZERO-HIT")
    n_nomap = sum(1 for r in rows if r[1] == "NO-MAPPING")
    u_cov = sum(1 for r in upload_rows if r[1] == "COVERED")
    u_zero = sum(1 for r in upload_rows if r[1] in ("ZERO-HIT", "MISSING-UPLOAD"))
    print(f"\ntotal={len(rows)} covered={n_cov} absorbed/empty={n_abs} "
          f"zero-hit={n_zero} no-mapping={n_nomap}")
    print(f"uploads: covered={u_cov} zero-hit/missing={u_zero}")
    if missing_map:
        print(f"\nFILES WITHOUT MAPPING ({len(missing_map)}):")
        for f in missing_map:
            print(f"  - {f}")
    if stale_map:
        print(f"\nSTALE MAP ENTRIES ({len(stale_map)}):")
        for f in stale_map:
            print(f"  - {f}")
    if zero_hit or upload_zero:
        print(f"\nZERO-HIT FILES ({len(zero_hit) + len(upload_zero)}):")
        for f, syms in zero_hit + upload_zero:
            print(f"  - {f} (expected: {syms})")
        return 1
    print("\nVERDICT: every rar + upload file has a playground.py surface "
          "(or a documented absorption).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
