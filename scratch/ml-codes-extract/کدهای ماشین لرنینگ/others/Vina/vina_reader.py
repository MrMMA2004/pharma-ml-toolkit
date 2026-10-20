#!/usr/bin/env python3
import os
import re
import csv

LOG_DIR = "results"           # folder with .log files
OUTPUT_CSV = "vina_results.csv"

def get_best_affinity_from_log(filepath):
    best_affinity = None
    with open(filepath, "r") as f:
        for line in f:
            m = re.match(r"\s*\d+\s+(-?\d+\.\d+)", line)
            if m:
                val = float(m.group(1))
                if best_affinity is None or val < best_affinity:
                    best_affinity = val
    return best_affinity

def main():
    rows = []
    for root, _, files in os.walk(LOG_DIR):
        for file in files:
            if file.endswith(".log"):
                path = os.path.join(root, file)
                affinity = get_best_affinity_from_log(path)
                rows.append({"Ligand": os.path.splitext(file)[0], "Best_Affinity (kcal/mol)": affinity})

    # Write to CSV
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Ligand", "Best_Affinity (kcal/mol)"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Results saved to {OUTPUT_CSV} ({len(rows)} ligands parsed)")

if __name__ == "__main__":
    main()
