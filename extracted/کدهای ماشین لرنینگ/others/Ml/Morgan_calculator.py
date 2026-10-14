import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# ===== CONFIGURATION =====
INPUT_FILE = "Descriptoredkinm_5HT2A.csv"   # This file already has RDKit descriptors
SMILES_COLUMN = "SMILES"               # SMILES column
FP_SIZE = 2048                         # Morgan fingerprint size
BATCH_SIZE = 1000                      # You can lower this if needed
OUTPUT_FILE = "Morgan_RDKIT_KI_5HT2A.csv"
PROGRESS_FILE = f"progress_morgan_append_{FP_SIZE}.log"

# ===== FUNCTION DEFINITIONS =====

def load_data():
    df = pd.read_csv(INPUT_FILE)
    if SMILES_COLUMN not in df.columns:
        raise ValueError(f"Missing SMILES column: {SMILES_COLUMN}")
    return df

def compute_fingerprint(smiles, radius=3, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return list(fp)

def write_batch_to_csv(batch_df, file_path, header_written):
    batch_df.to_csv(file_path, mode='a', index=False, header=not header_written)

def load_progress():
    return int(open(PROGRESS_FILE).read().strip()) if os.path.exists(PROGRESS_FILE) else 0

def save_progress(batch_index):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(batch_index))

# ===== MAIN SCRIPT =====

def main():
    df = load_data()
    total = len(df)
    start_batch = load_progress()
    print(f"Total rows: {total}, starting from batch {start_batch}")

    header_written = os.path.exists(OUTPUT_FILE)

    for batch_index in range(start_batch, (total + BATCH_SIZE - 1) // BATCH_SIZE):
        start = batch_index * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        print(f"Processing batch {batch_index}: rows {start} to {end - 1}")

        df_batch = df.iloc[start:end].copy()
        fps = []

        for smiles in df_batch[SMILES_COLUMN]:
            fp = compute_fingerprint(smiles, radius=3, nBits=FP_SIZE)
            if fp is None:
                fps.append([None] * FP_SIZE)
            else:
                fps.append(fp)

        fp_df = pd.DataFrame(fps, columns=[f"Bit_{i}" for i in range(FP_SIZE)])
        df_combined = pd.concat([df_batch.reset_index(drop=True), fp_df], axis=1)

        write_batch_to_csv(df_combined, OUTPUT_FILE, header_written)
        header_written = True

        save_progress(batch_index + 1)

    print(f"✅ Done! Merged data with Morgan fingerprints saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
