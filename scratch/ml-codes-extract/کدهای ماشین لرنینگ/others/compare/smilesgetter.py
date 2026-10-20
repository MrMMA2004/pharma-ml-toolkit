import pandas as pd
import os

# === CONFIGURATION ===
differences_file = 'differences_output.csv'     # from previous step
reference_smiles_file = 'ligands.csv' # contains 'Ligand' and 'SMILES'
output_file = 'differences_with_smiles.csv'

# === Load files ===
df_diff = pd.read_csv(differences_file)
df_ref = pd.read_csv(reference_smiles_file, dtype=str)

# Ensure no NaNs and normalize names
df_ref['Ligand'] = df_ref['Ligand'].astype(str).str.strip()
df_ref['Normalized'] = df_ref['Ligand'].apply(lambda x: os.path.splitext(x)[0].lower())

# Create a lookup dict: {normalized_ligand_name: SMILES}
ligand_to_smiles = dict(zip(df_ref['Normalized'], df_ref['SMILES']))

# Normalize function
def normalize(name):
    if pd.isna(name):
        return ''
    return os.path.splitext(name.strip())[0].lower()

# Add SMILES columns
df_diff['Ligand1_SMILES'] = df_diff.iloc[:, 0].apply(lambda x: ligand_to_smiles.get(normalize(x), ''))
df_diff['Ligand2_SMILES'] = df_diff.iloc[:, 1].apply(lambda x: ligand_to_smiles.get(normalize(x), ''))

# Save the result
df_diff.to_csv(output_file, index=False)
print(f"✅ SMILES added. Output saved to '{output_file}'")
