#!/usr/bin/env python3
import os
import subprocess
import time

CONF_FILE = "conf.txt"
LIGAND_DIR = "ligands"
OUT_DIR = "results"
SIGNAL_DIR = "signals"
MAX_PARALLEL = 1

# Make sure directories exist
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SIGNAL_DIR, exist_ok=True)

# Clean old signal files
for f in os.listdir(SIGNAL_DIR):
    os.remove(os.path.join(SIGNAL_DIR, f))

# Collect ligands
ligands = []
for root, _, files in os.walk(LIGAND_DIR):
    for f in files:
        if f.endswith(".pdbqt"):
            ligands.append(os.path.join(root, f))

total = len(ligands)
print(f"📦 Found {total} ligands. Running with up to {MAX_PARALLEL} terminals.\n")

completed = 0
launched = 0

while completed < total:
    # Launch new CMDs if we have capacity
    while launched < total and (launched - completed) < MAX_PARALLEL:
        ligand_path = ligands[launched]
        ligand_name = os.path.splitext(os.path.basename(ligand_path))[0]
        out_pdbqt = os.path.join(OUT_DIR, f"{ligand_name}_out.pdbqt")
        log_file = os.path.join(OUT_DIR, f"{ligand_name}.log")
        signal_file = os.path.join(SIGNAL_DIR, f"{ligand_name}.done")

        # Build the CMD command
        cmd = (
            f'vina --config "{CONF_FILE}" --ligand "{ligand_path}" '
            f'--out "{out_pdbqt}" --log "{log_file}" && echo done > "{signal_file}"'
        )

        # Debug print so you see what’s launching
        print(f"🚀 Launching docking {launched+1}/{total}: {ligand_name}")

        # Open a new CMD that runs vina, writes a done signal, then closes
        subprocess.Popen(f'start cmd /c "{cmd}"', shell=True)

        launched += 1

    # Check how many jobs are finished by counting .done files
    completed = len(os.listdir(SIGNAL_DIR))
    print(f"📊 Progress: {completed}/{total} completed", end="\r")

    time.sleep(5)

print("\n🎉 All dockings completed successfully!")
