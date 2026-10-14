import os
import pandas as pd

main_folder = os.getcwd()
data = []

for subfolder in os.listdir(main_folder):
    sub_path = os.path.join(main_folder, subfolder)
    if os.path.isdir(sub_path):
        gpf_path = os.path.join(sub_path, "r.gpf")
        ligand_types = "⛔ Not found"
        if os.path.exists(gpf_path):
            with open(gpf_path, 'r') as f:
                for line in f:
                    if line.strip().startswith("ligand_types"):
                        ligand_types = line.strip().replace("ligand_types", "").strip()
                        break
        data.append({
            "Folder": subfolder,
            "Ligand Types": ligand_types
        })

# خروجی اکسل
df = pd.DataFrame(data)
df.to_excel("ligand_types_summary.xlsx", index=False)
print("✅ فایل ligand_types_summary.xlsx با موفقیت ذخیره شد.")
