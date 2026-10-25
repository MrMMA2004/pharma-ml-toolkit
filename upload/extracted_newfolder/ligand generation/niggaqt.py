#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import subprocess
import time
from datetime import datetime

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Output folder
pdbqt_out_dir = os.path.join(script_dir, 'output_pdbqt')
if not os.path.exists(pdbqt_out_dir):
    os.makedirs(pdbqt_out_dir)

# Path to MGLTools pythonsh (Windows)
mgl_python = r"C:\Program Files (x86)\MGLTools-1.5.7\python.exe"

# Path to prepare_ligand4.py (Windows)
adt_script = r"C:\Program Files (x86)\MGLTools-1.5.7\Lib\site-packages\AutoDockTools\Utilities24\prepare_ligand4.py"

failed_files = []
timeout_files = []
success_files = []

# Get all PDB files and sort them
pdb_files = sorted([f for f in os.listdir(script_dir) if f.endswith(".pdb")])

# Create numbered subfolders and convert each PDB file
for i, pdb_file in enumerate(pdb_files, 1):
    name = os.path.splitext(pdb_file)[0]
    pdb_path = os.path.join(script_dir, pdb_file)

    # Create numbered subfolder
    numbered_folder = os.path.join(pdbqt_out_dir, str(i))
    if not os.path.exists(numbered_folder):
        os.makedirs(numbered_folder)

    pdbqt_path = os.path.join(numbered_folder, "%s.pdbqt" % name)

    try:
        # Launch process
        proc = subprocess.Popen([
            mgl_python, adt_script,
            "-l", pdb_path,
            "-o", pdbqt_path,
            "-A", "hydrogens"
        ])

        # Manual timeout loop (30s)
        start = time.time()
        while proc.poll() is None:
            if time.time() - start > 30:
                proc.terminate()
                print "[%s] conversion timed out after 30 seconds." % name
                timeout_files.append(name)
                break
            time.sleep(1)
        else:
            # Process finished normally
            if proc.returncode == 0:
                print "[%s] converted to PDBQT." % name
                success_files.append(name)
            else:
                print "[%s] conversion failed (script error)." % name
                failed_files.append(name)

    except OSError as e:
        print "[%s] conversion failed (OS error): %s" % (name, str(e))
        failed_files.append(name)

# ------------------- LOG FAILURES -------------------
log_path = os.path.join(script_dir, "conversion_log.txt")
with open(log_path, 'w') as f:
    f.write("PDB to PDBQT Conversion Log - %s\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    f.write("=" * 50 + "\n\n")

    if success_files:
        f.write("SUCCESSFUL CONVERSIONS:\n")
        for name in success_files:
            f.write(name + "\n")
        f.write("\n")

    if timeout_files:
        f.write("TIMED OUT FILES (30+ seconds):\n")
        for name in timeout_files:
            f.write(name + "\n")
        f.write("\n")

    if failed_files:
        f.write("FAILED CONVERSIONS:\n")
        for name in failed_files:
            f.write(name + "\n")
        f.write("\n")

    if not timeout_files and not failed_files:
        f.write("All PDB files converted successfully.\n")

# Print summary
if timeout_files:
    print "Timed out files: %d" % len(timeout_files)
if failed_files:
    print "Failed conversions: %d" % len(failed_files)
if not timeout_files and not failed_files:
    print "All PDB files converted successfully."

print "Conversion log saved to: %s" % log_path
