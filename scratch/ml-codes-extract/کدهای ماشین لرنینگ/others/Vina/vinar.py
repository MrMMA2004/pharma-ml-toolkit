#!/usr/bin/env python3
import os
import subprocess
import argparse
import re

def get_receptor_from_config(conf_file):
    """Read receptor = ... from the config file"""
    with open(conf_file) as f:
        for line in f:
            m = re.match(r'\s*receptor\s*=\s*(\S+)', line)
            if m:
                return m.group(1)
    raise ValueError("❌ No receptor found in config file (missing 'receptor = ...')")

def run_vina(conf, ligand, outdir):
    name = os.path.splitext(os.path.basename(ligand))[0]
    out_pdbqt = os.path.join(outdir, f"{name}_out.pdbqt")
    log_file = os.path.join(outdir, f"{name}.log")

    cmd = [
        "vina",
        "--config", conf,
        "--ligand", ligand,
        "--out", out_pdbqt,
        "--log", log_file
    ]

    print(f"🔬 Docking {name} ...")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Batch dock ligands with AutoDock Vina (config-driven)")
    parser.add_argument("--conf", required=True, help="Path to config file containing receptor and grid params")
    parser.add_argument("--ligands", required=True, help="Folder containing ligand PDBQTs")
    parser.add_argument("--outdir", default="vina_results", help="Output folder for docking results")
    args = parser.parse_args()

    receptor = get_receptor_from_config(args.conf)
    if not os.path.isfile(receptor):
        raise FileNotFoundError(f"❌ Receptor '{receptor}' not found. Make sure it's in the same directory as the config or give full path.")

    os.makedirs(args.outdir, exist_ok=True)

    # Loop through all ligand PDBQT files (including subfolders)
    for root, dirs, files in os.walk(args.ligands):
        for f in files:
            if f.endswith(".pdbqt"):
                ligand_path = os.path.join(root, f)
                run_vina(args.conf, ligand_path, args.outdir)

    print("✅ Docking complete! Results saved in:", args.outdir)

if __name__ == "__main__":
    main()
