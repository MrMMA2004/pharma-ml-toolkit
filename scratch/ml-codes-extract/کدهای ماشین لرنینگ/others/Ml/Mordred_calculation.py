import os
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from tqdm import tqdm
import shutil

def is_valid_smiles(smi):
    try:
        return Chem.MolFromSmiles(smi) is not None
    except:
        return False

def calc_mordred_batch(smiles_batch):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_batch]
    calc = Calculator(descriptors, ignore_3D=True)
    df_desc = calc.pandas(mols)
    return df_desc.dropna(axis=1, how='all')

def process_batches(smiles_list, batch_size=10000, temp_dir="mordred_batches"):
    os.makedirs(temp_dir, exist_ok=True)
    batch_files = []
    total_batches = (len(smiles_list) + batch_size - 1) // batch_size

    for i in range(0, len(smiles_list), batch_size):
        batch_idx = i // batch_size + 1
        batch = smiles_list[i:i+batch_size]
        batch_file = os.path.join(temp_dir, f"batch_{batch_idx}.csv")

        if os.path.exists(batch_file):
            print(f"✅ Batch {batch_idx} already done. Skipping.")
        else:
            print(f"🚀 Calculating batch {batch_idx}/{total_batches}...")
            df_desc = calc_mordred_batch(batch)
            df_desc.insert(0, "SMILES", batch)
            df_desc.to_csv(batch_file, index=False)
            print(f"✅ Saved: {batch_file}")
        
        batch_files.append(batch_file)

    print("📦 Merging batches...")
    return pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)

if __name__ == "__main__":
    input_file = "pIC50_D5.csv"
    output_file = "MordredDescriptors_Batched.csv"
    temp_dir = "mordred_batches"
    cleanup = True

    df = pd.read_csv(input_file)

    if "SMILES" not in df.columns:
        raise ValueError("❌ ستون SMILES پیدا نشد.")

    df = df[df["SMILES"].notna()]
    df = df[df["SMILES"].apply(is_valid_smiles)].reset_index(drop=True)
    smiles_list = df["SMILES"].tolist()

    print(f"🔢 شروع محاسبه‌ی Mordred برای {len(smiles_list)} مولکول...")

    df_desc = process_batches(smiles_list, batch_size=10000, temp_dir=temp_dir)

    df_out = pd.concat([df.reset_index(drop=True), df_desc.drop(columns=["SMILES"])], axis=1)
    df_out.to_csv(output_file, index=False)

    print(f"\n✅ ذخیره شد: {output_file}")

    if cleanup and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"🧹 پاکسازی: پوشه {temp_dir} حذف شد.")
