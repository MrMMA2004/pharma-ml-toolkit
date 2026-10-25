#!/usr/bin/env python3
import os
import subprocess
import time

CONF_FILE = "conf.txt"
LIGAND_DIR = "ligands"
OUT_DIR = "results"
SIGNAL_DIR = "signals"
MAX_PARALLEL = 6

# Make sure directories exist
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SIGNAL_DIR, exist_ok=True)

# Step 1: Remove empty log files (crashed jobs)
for f in os.listdir(OUT_DIR):
    if f.endswith(".log"):
        path = os.path.join(OUT_DIR, f)
        if os.path.getsize(path) == 0:  # empty log
            print(f"🗑 Removing empty log: {f}")
            os.remove(path)
            sigfile = os.path.join(SIGNAL_DIR, f.replace(".log", ".done"))
            if os.path.exists(sigfile):
                os.remove(sigfile)

# Step 2: Collect ligands
ligands = []
for root, _, files in os.walk(LIGAND_DIR):
    for f in files:
        if f.endswith(".pdbqt"):
            ligands.append(os.path.join(root, f))

total = len(ligands)

# Step 3: Filter unfinished ligands
unfinished = []
for ligand_path in ligands:
    ligand_name = os.path.splitext(os.path.basename(ligand_path))[0]
    log_file = os.path.join(OUT_DIR, f"{ligand_name}.log")
    signal_file = os.path.join(SIGNAL_DIR, f"{ligand_name}.done")

    if not os.path.exists(signal_file) or not os.path.exists(log_file):
        unfinished.append(ligand_path)

print(f"📦 Found {total} ligands, {len(unfinished)} unfinished. Running with up to {MAX_PARALLEL} parallel jobs.\n")

completed = total - len(unfinished)
launched = 0

# Step 4: Run only unfinished ligands
while completed < total:
    # Launch new jobs up to MAX_PARALLEL
    while launched < len(unfinished) and (launched + completed) - completed < MAX_PARALLEL:
        ligand_path = unfinished[launched]
        ligand_name = os.path.splitext(os.path.basename(ligand_path))[0]
        out_pdbqt = os.path.join(OUT_DIR, f"{ligand_name}_out.pdbqt")
        log_file = os.path.join(OUT_DIR, f"{ligand_name}.log")
        signal_file = os.path.join(SIGNAL_DIR, f"{ligand_name}.done")

        cmd = (
            f'vina --config "{CONF_FILE}" --ligand "{ligand_path}" '
            f'--out "{out_pdbqt}" --log "{log_file}" && echo done > "{signal_file}"'
        )

        print(f"🚀 Launching docking {completed+launched+1}/{total}: {ligand_name}")
        # Launch in a new terminal window (Windows style)
        subprocess.Popen(f'start cmd /c "{cmd}"', shell=True)

        launched += 1

    # Update completed count
    completed = len([f for f in os.listdir(SIGNAL_DIR) if f.endswith(".done")])
    print(f"📊 Progress: {completed}/{total} completed", end="\r")
    time.sleep(5)

print("\n🎉 All dockings completed successfully!")
