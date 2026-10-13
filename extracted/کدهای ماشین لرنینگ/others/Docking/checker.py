import os
import pandas as pd

# مسیر پوشه فعلی (جایی که اسکریپت رو اجرا می‌کنی)
base_dir = os.getcwd()

issues = []

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    dlg_path = os.path.join(folder_path, "r.dlg")
    status = {"Folder": folder_name, "Issue": ""}

    if not os.path.isfile(dlg_path):
        status["Issue"] = "r.dlg NOT FOUND"
        issues.append(status)
    else:
        try:
            with open(dlg_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "Successful Completion" not in content:
                    status["Issue"] = "Docking NOT Completed"
                    issues.append(status)
        except Exception as e:
            status["Issue"] = f"Error reading file: {str(e)}"
            issues.append(status)

# خروجی اکسل
if issues:
    df = pd.DataFrame(issues)
    output_path = os.path.join(base_dir, "docking_issues.xlsx")
    df.to_excel(output_path, index=False)
    print(f"✅ Excel file created: {output_path}")
else:
    print("🎉 All r.dlg files are present and successfully completed.")
