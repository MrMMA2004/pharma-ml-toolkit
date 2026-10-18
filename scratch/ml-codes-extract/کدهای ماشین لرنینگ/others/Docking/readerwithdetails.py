import os
import pandas as pd
import re

base_dir = os.path.dirname(os.path.abspath(__file__))
results = []

# Regex برای استخراج اطلاعات جدول
pattern = re.compile(
    r"^\s*(\d+)\s+\|\s+([-]?\d+\.\d+)\s+\|\s+\d+\s+\|\s+([-]?\d+\.\d+)\s+\|\s+(\d+)\s+\|"
)

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    dlg_path = os.path.join(folder_path, "r.dlg")

    if not os.path.isdir(folder_path):
        continue
    if not os.path.isfile(dlg_path):
        results.append({
            "Folder": folder,
            "Cluster": "فایل r.dlg یافت نشد",
            "Num in Cluster": "",
            "Mean Binding Energy": "",
            "Lowest Binding Energy": "",
            "Top Cluster": "",
            "Top Num in Cluster": "",
            "Top Mean Binding Energy": "",
            "Top Lowest Binding Energy": ""
        })
        continue

    try:
        with open(dlg_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

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
                "Folder": folder,
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

        # انتخاب پرتعدادترین کلاستر
        best_cluster = max(cluster_data, key=lambda x: x[3])

        # اطلاعات خط اول جدول
        top_cluster = cluster_data[0]

        results.append({
            "Folder": folder,
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
            "Folder": folder,
            "Cluster": f"خطا: {e}",
            "Num in Cluster": "",
            "Mean Binding Energy": "",
            "Lowest Binding Energy": "",
            "Top Cluster": "",
            "Top Num in Cluster": "",
            "Top Mean Binding Energy": "",
            "Top Lowest Binding Energy": ""
        })

# ذخیره در فایل اکسل
df = pd.DataFrame(results)
df.to_excel("docking_summary.xlsx", index=False)
print("✅ فایل docking_summary.xlsx با اطلاعات کامل ساخته شد.")
