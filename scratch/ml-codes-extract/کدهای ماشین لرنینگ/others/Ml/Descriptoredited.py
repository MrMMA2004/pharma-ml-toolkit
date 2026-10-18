# pip install rdkit-pypi pandas tqdm

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import descList
from tqdm import tqdm
import shutil  # for optional cleanup

def is_valid_smiles(smi):
    try:
        return Chem.MolFromSmiles(smi) is not None
    except:
        return False

def calc_descriptors_batch(smiles_batch):
    desc_names = [name for name, _ in descList]
    results = []

    for smi in tqdm(smiles_batch, desc="Calculating batch", unit="mol", leave=False):
        mol = Chem.MolFromSmiles(smi)
        values = [func(mol) for _, func in descList]
        results.append([smi] + values)

    return pd.DataFrame(results, columns=["SMILES"] + desc_names)

def calc_descriptors_in_batches(smiles_list, batch_size=20_000, temp_dir="temp_batches"):
    os.makedirs(temp_dir, exist_ok=True)
    total_batches = (len(smiles_list) + batch_size - 1) // batch_size
    batch_files = []

    for i in range(0, len(smiles_list), batch_size):
        batch_index = i // batch_size + 1
        batch = smiles_list[i:i + batch_size]
        batch_file = os.path.join(temp_dir, f"batch_{batch_index}.csv")

        if os.path.exists(batch_file):
            print(f"✅ Batch {batch_index} already completed. Skipping...")
        else:
            print(f"🚀 Processing batch {batch_index}/{total_batches}...")
            df_batch = calc_descriptors_batch(batch)
            df_batch.to_csv(batch_file, index=False)
            print(f"✅ Saved batch {batch_index} to {batch_file}")
        
        batch_files.append(batch_file)

    print("\n📦 Merging all completed batches...")
    df_all = pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)
    return df_all

if __name__ == "__main__":
    input_file = "pIC50_D5.csv"
    output_csv = "DescriptoredpIC50_D5.csv"
    temp_dir = "temp_batches"

    # 1. Read CSV input (fix engine issue)
    df_in = pd.read_csv(input_file)

    if "SMILES" not in df_in.columns:
        raise ValueError("❌ ستون 'SMILES' در فایل ورودی یافت نشد.")

    # 2. Validate SMILES
    valid_mask = df_in["SMILES"].dropna().astype(str).apply(is_valid_smiles)
    df_valid = df_in.loc[valid_mask].reset_index(drop=True)
    print(f"📦 فیلتر شد: {len(df_in) - len(df_valid)} SMILES نامعتبر | باقی‌مانده: {len(df_valid)}")

    smiles_list = df_valid["SMILES"].tolist()

    # 3. Calculate descriptors in resumable batches
    df_desc = calc_descriptors_in_batches(smiles_list, batch_size=20_000, temp_dir=temp_dir)

    # 4. Merge back with valid data (order-safe)
    df_out = pd.concat([df_valid.reset_index(drop=True), df_desc.drop(columns=["SMILES"])], axis=1)

    # 5. Save result to CSV
    df_out.to_csv(output_csv, index=False)
    print(f"\n✅ دسکریپتورها با موفقیت در '{output_csv}' ذخیره شدند.")

    # 6. Optional: Clean temp folder
    cleanup = True
    if cleanup and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"🧹 پاکسازی فایل‌های موقت: پوشه '{temp_dir}' حذف شد.")
