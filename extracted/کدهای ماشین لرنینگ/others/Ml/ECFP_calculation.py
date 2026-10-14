import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import shutil

def is_valid_smiles(smi):
    try:
        return Chem.MolFromSmiles(smi) is not None
    except:
        return False

def calc_ecfp_batch(smiles_batch, radius=2, nBits=1024):
    fingerprints = []
    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
    df_fp = pd.DataFrame(fingerprints, columns=[f"ECFP_{i}" for i in range(nBits)])
    df_fp.insert(0, "SMILES", smiles_batch)
    return df_fp

def process_ecfp_batches(smiles_list, batch_size=10000, temp_dir="ecfp_batches", radius=2, nBits=1024):
    os.makedirs(temp_dir, exist_ok=True)
    batch_files = []
    total_batches = (len(smiles_list) + batch_size - 1) // batch_size

    for i in range(0, len(smiles_list), batch_size):
        batch_idx = i // batch_size + 1
        batch = smiles_list[i:i+batch_size]
        batch_file = os.path.join(temp_dir, f"ecfp_batch_{batch_idx}.csv")

        if os.path.exists(batch_file):
            print(f"✅ Batch {batch_idx} already exists. Skipping...")
        else:
            print(f"🚀 Processing ECFP batch {batch_idx}/{total_batches}...")
            df_fp = calc_ecfp_batch(batch, radius=radius, nBits=nBits)
            df_fp.to_csv(batch_file, index=False)
            print(f"✅ Saved: {batch_file}")
        
        batch_files.append(batch_file)

    print("📦 Merging ECFP batches...")
    return pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)

if __name__ == "__main__":
    input_file = "pIC50_D5.csv"
    output_file = "ECFP_Fingerprints_Batched.csv"
    temp_dir = "ecfp_batches"
    cleanup = True
    radius = 2
    nBits = 1024

    df = pd.read_csv(input_file)

    if "SMILES" not in df.columns:
        raise ValueError("❌ ستون SMILES پیدا نشد.")

    df = df[df["SMILES"].notna()]
    df = df[df["SMILES"].apply(is_valid_smiles)].reset_index(drop=True)
    smiles_list = df["SMILES"].tolist()

    print(f"🧠 محاسبه ECFP (radius={radius}, nBits={nBits}) برای {len(smiles_list)} مولکول...")

    df_ecfp = process_ecfp_batches(smiles_list, batch_size=10000, temp_dir=temp_dir, radius=radius, nBits=nBits)

    df_out = pd.concat([df.reset_index(drop=True), df_ecfp.drop(columns=["SMILES"])], axis=1)
    df_out.to_csv(output_file, index=False)

    print(f"\n✅ ذخیره شد: {output_file}")

    if cleanup and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"🧹 پاکسازی: پوشه {temp_dir} حذف شد.")
