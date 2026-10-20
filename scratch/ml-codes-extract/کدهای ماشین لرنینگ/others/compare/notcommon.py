import pandas as pd
import os

# === CONFIGURATION ===
file1 = 'd_summary.csv'   # Can also be .xlsx
file2 = 'ligands.csv'   # Can also be .xlsx
output_csv = 'differences_output.csv'

# Optional: column to compare (common column containing ligand names or IDs)
# Set to None to compare all values
compare_column = 'Ligand'   # e.g., 'Ligand PDBQT' or None for full content

# === LOAD FILES ===
def load_file(path):
    if path.lower().endswith('.csv'):
        return pd.read_csv(path, dtype=str)
    elif path.lower().endswith('.xlsx') or path.lower().endswith('.xls'):
        return pd.read_excel(path, dtype=str)
    else:
        raise ValueError("Unsupported file type.")

df1 = load_file(file1)
df2 = load_file(file2)

# === EXTRACT COMPARISON COLUMNS ===
if compare_column:
    col1 = df1[compare_column].dropna().astype(str)
    col2 = df2[compare_column].dropna().astype(str)
else:
    col1 = df1.astype(str).values.ravel()
    col2 = df2.astype(str).values.ravel()

# === NORMALIZE NAMES (remove .pdbqt extension) ===
def normalize_name(name):
    return os.path.splitext(name.strip())[0]  # removes .pdbqt or any extension

set1 = set(normalize_name(x) for x in col1)
set2 = set(normalize_name(x) for x in col2)

# === FIND DIFFERENCES ===
only_in_file1 = sorted(set1 - set2)
only_in_file2 = sorted(set2 - set1)

# === SAVE TO CSV ===
df_out = pd.DataFrame({
    f"Only in {os.path.basename(file1)}": pd.Series(only_in_file1),
    f"Only in {os.path.basename(file2)}": pd.Series(only_in_file2)
})

df_out.to_csv(output_csv, index=False)

print(f"✅ Differences saved to '{output_csv}'")
