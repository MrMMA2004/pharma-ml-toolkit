import os
import pandas as pd
import re

# Current folder (where the script lives)
base_dir = os.path.dirname(os.path.abspath(__file__))
dlg_dir = base_dir   # change if needed

results = []

# Regex for extracting docking table info (your original)
pattern = re.compile(
    r"^\s*(\d+)\s+\|\s+([+-]?\d+\.\d+)\s+\|\s+\d+\s+\|\s+([+-]?\d+\.\d+)\s+\|\s+(\d+)\s+\|"
)


def extract_ligand_name(lines):
    # try line 68 (index 67)
    if len(lines) >= 68:
        m = re.search(r'([\w\-.]+\.pdbqt)', lines[67], flags=re.IGNORECASE)
        if m:
            return m.group(1)
    # try explicit "Ligand PDBQT file"
    for ln in lines:
        m = re.search(r'Ligand\s+PDBQT\s+file\s*=\s*["\']?([\w\-.]+\.pdbqt)', ln, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    # fallback: first occurrence of .pdbqt
    for ln in lines:
        m = re.search(r'([\w\-.]+\.pdbqt)', ln, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return ""

# Loop over .dlg files
for file in os.listdir(dlg_dir):
    if not file.lower().endswith(".dlg"):
        continue

    dlg_path = os.path.join(dlg_dir, file)
    dlg_base = os.path.splitext(file)[0]

    try:
        with open(dlg_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # NEW: ligand name from inside file
        ligand_pdbqt = extract_ligand_name(lines)

        cluster_data = []
        table_started = False

        for line in lines:
            if not table_started:
                if "Clus" in line and "Histogram" in line:
                    table_started = True
                continue

            match = pattern.match(line)
            if match:
                cluster_rank = int(match.group(1))
                lowest_energy = float(match.group(2))
                mean_energy = float(match.group(3))
                num_in_cluster = int(match.group(4))
                cluster_data.append((cluster_rank, lowest_energy, mean_energy, num_in_cluster))

        if not cluster_data:
            results.append({
                "DLG Filename": file,
                "DLG Base Name": dlg_base,
                "Ligand PDBQT": ligand_pdbqt,
                "Cluster": "جدول یافت نشد",
                "Num in Cluster": "",
                "Mean Binding Energy": "",
                "Lowest Binding Energy": "",
                "Top Cluster": "",
                "Top Num in Cluster": "",
                "Top Mean Binding Energy": "",
                "Top Lowest Binding Energy": ""
            })
            continue

        # Best cluster = the one with the largest number in cluster
        best_cluster = max(cluster_data, key=lambda x: x[3])
        # Top cluster = the first one listed
        top_cluster = cluster_data[0]

        results.append({
            "DLG Filename": file,
            "DLG Base Name": dlg_base,
            "Ligand PDBQT": ligand_pdbqt,
            "Cluster": best_cluster[0],
            "Num in Cluster": best_cluster[3],
            "Mean Binding Energy": best_cluster[2],
            "Lowest Binding Energy": best_cluster[1],
            "Top Cluster": top_cluster[0],
            "Top Num in Cluster": top_cluster[3],
            "Top Mean Binding Energy": top_cluster[2],
            "Top Lowest Binding Energy": top_cluster[1]
        })

    except Exception as e:
        results.append({
            "DLG Filename": file,
            "DLG Base Name": dlg_base,
            "Ligand PDBQT": "",
            "Cluster": f"خطا: {e}",
            "Num in Cluster": "",
            "Mean Binding Energy": "",
            "Lowest Binding Energy": "",
            "Top Cluster": "",
            "Top Num in Cluster": "",
            "Top Mean Binding Energy": "",
            "Top Lowest Binding Energy": ""
        })

# Save results to Excel
df = pd.DataFrame(results)
output_path = os.path.join(base_dir, "docking_summary_with_ligand.xlsx")
df.to_excel(output_path, index=False)

print(f"✅ فایل {output_path} ساخته شد و شامل خلاصه docking است.")
