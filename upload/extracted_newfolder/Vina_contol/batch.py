#!/usr/bin/env python3
import os
import subprocess

# === USER SETTINGS ===
RECEPTOR = "r.pdbqt"
CONFIG   = "conf.txt"
LIGAND_ROOT = "ligands"
RESULTS_ROOT = "results"
LOGS_ROOT = "logs"

# === PREPARE DIRECTORIES ===
os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(LOGS_ROOT, exist_ok=True)

# === FIND NUMBERED FOLDERS ===
for folder in sorted(os.listdir(LIGAND_ROOT), key=lambda x: int(x) if x.isdigit() else x):
    subdir = os.path.join(LIGAND_ROOT, folder)
    if not os.path.isdir(subdir) or not folder.isdigit():
        continue  # skip non-numeric folders

    print(f"📂 Processing folder: {folder}")

    # Make matching results directory
    result_subdir = os.path.join(RESULTS_ROOT, folder)
    os.makedirs(result_subdir, exist_ok=True)

    # Loop through ligands inside
    for lig in os.listdir(subdir):
        if not lig.endswith(".pdbqt"):
            continue

        lig_path = os.path.join(subdir, lig)
        out_path = os.path.join(result_subdir, lig.replace(".pdbqt", "_out.pdbqt"))
        log_path = os.path.join(LOGS_ROOT, f"{folder}_{lig.replace('.pdbqt', '.log')}")

        # Skip if output already exists
        if os.path.exists(out_path):
            print(f"⏩ Skipping {lig} (already docked)")
            continue

        # Run vina docking
        cmd = [
            "vina",
            "--receptor", RECEPTOR,
            "--ligand", lig_path,
            "--config", CONFIG,
            "--out", out_path,
            "--log", log_path
        ]

        print(f"🚀 Docking {lig} ...")
        subprocess.run(cmd)

print("✅ All docking runs completed.")
