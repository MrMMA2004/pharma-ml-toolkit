import os
import re
import pandas as pd

def parse_dlg_file(filepath):
    with open(filepath, 'r', errors='ignore') as file:
        lines = file.readlines()

    cluster_data = []
    pdbqt_name = None
    histogram_section = False
    histogram_lines = []

    for line in lines:
        # گرفتن نام فایل pdbqt
        if "NEWDPF move" in line and not pdbqt_name:
            match = re.search(r'NEWDPF move\s+(\S+)', line)
            if match:
                pdbqt_name = match.group(1)

        # پیدا کردن شروع بخش CLUSTERING HISTOGRAM
        if 'CLUSTERING HISTOGRAM' in line:
            histogram_section = True
            continue

        # جمع‌آوری خطوط جدول هیستوگرام
        if histogram_section:
            if re.match(r'^\s*\d+\s*\|', line):
                histogram_lines.append(line)
            elif len(histogram_lines) > 0:
                break

    # عبارت باقاعده اصلاح شده: گرفتن انرژی‌های مثبت و منفی
    for line in histogram_lines:
        match = re.match(r'\s*(\d+)\s*\|\s*([+-]?\d+\.\d+)\s*\|\s*(\d+)\s*\|\s*([+-]?\d+\.\d+)\s*\|\s*(\d+)', line)
        if match:
            cluster_data.append({
                "rank": int(match.group(1)),
                "lowest": float(match.group(2)),
                "run": int(match.group(3)),
                "mean": float(match.group(4)),
                "num": int(match.group(5)),
            })

    return cluster_data, pdbqt_name

def analyze_all_dlgs():
    current_folder = os.getcwd()
    output_excel = os.path.join(current_folder, "cluster_summary.xlsx")
    rows = []

    for file in os.listdir(current_folder):
        if file.endswith(".dlg"):
            full_path = os.path.join(current_folder, file)
            clusters, pdbqt = parse_dlg_file(full_path)

            if not clusters:
                print(f"⚠️ هیچ کلاستری در فایل {file} پیدا نشد.")
                continue

            most_crowded = max(clusters, key=lambda x: x["num"])
            first = clusters[0]

            rows.append({
                "DLG File": file,
                "PDBQT Name": pdbqt,
                "MostCrowded_MeanEnergy": most_crowded["mean"],
                "MostCrowded_LowestEnergy": most_crowded["lowest"],
                "MostCrowded_Num": most_crowded["num"],
                "FirstCluster_MeanEnergy": first["mean"],
                "FirstCluster_LowestEnergy": first["lowest"],
                "FirstCluster_Num": first["num"],
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_excel(output_excel, index=False)
        print(f"\n✅ فایل اکسل ساخته شد: {output_excel}")
    else:
        print("❌ هیچ داده‌ای پردازش نشد.")

if __name__ == "__main__":
    analyze_all_dlgs()
