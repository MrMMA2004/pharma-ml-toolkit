# mismatch_check.py (Python 3)
import os
from openpyxl import Workbook
from openpyxl.styles import PatternFill

# Parent folder containing numbered subfolders
parent_folder = r"output_pdb/output_pdbqt"

# Prepare Excel workbook
wb = Workbook()
ws = wb.active
ws.title = "Mismatch Report"

# Header row
headers = ["Ligand", "TORSDOF", "Active REMARKs", "BRANCHes", "Status"]
ws.append(headers)

# Define style for mismatches
red_fill = PatternFill(start_color="FF9999", end_color="FF9999", fill_type="solid")

# Walk through numbered subfolders
for subfolder in sorted(os.listdir(parent_folder), key=lambda x: int(x) if x.isdigit() else x):
    folder_path = os.path.join(parent_folder, subfolder)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdbqt"):
            continue

        filepath = os.path.join(folder_path, filename)
        tosdof_count = 0
        active_remarks = 0
        branch_count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Count TORSDOF
                if line.startswith("TORSDOF"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            tosdof_count = int(parts[1])
                        except ValueError:
                            pass
                # Count active REMARKs
                if line.startswith("REMARK") and " A " in line:
                    active_remarks += 1
                # Count BRANCH lines
                if line.startswith("BRANCH"):
                    branch_count += 1

        # Determine status
        status = "OK" if (tosdof_count == active_remarks == branch_count) else "MISMATCH"

        # Append to Excel
        row = [os.path.join(subfolder, filename), tosdof_count, active_remarks, branch_count, status]
        ws.append(row)

        # Highlight mismatches
        if status == "MISMATCH":
            for cell in ws[ws.max_row]:
                cell.fill = red_fill

# Auto-size columns
for col in ws.columns:
    max_length = 0
    col_letter = col[0].column_letter
    for cell in col:
        try:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        except:
            pass
    ws.column_dimensions[col_letter].width = max_length + 2

# Save Excel file
output_file = "mismatch_report.xlsx"
wb.save(output_file)

print(f"✅ Mismatch check finished. Results saved to: {output_file}")
