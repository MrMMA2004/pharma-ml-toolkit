# pip install rdkit-pypi pandas openpyxl tqdm

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import descList
from tqdm import tqdm

def calc_descriptors(smiles_list):
    """
    محاسبه‌ی تمامی دسکریپتورهای RDKit برای لیست SMILES
    """
    desc_names = [name for name, _ in descList] # فقط توابع تک‌ورودی
    results = []

    for smi in tqdm(smiles_list, desc="Calculating descriptors", unit="mol"):
        mol = Chem.MolFromSmiles(smi)
        # اینجا مطمئنیم همه SMILESهای ورودی معتبر هستند
        values = [func(mol) for _, func in descList]
        results.append([smi] + values)

    df = pd.DataFrame(results, columns=["SMILES"] + desc_names)
    return df

if __name__ == "__main__":
    # --- ۱. خواندن فایل اکسل ورودی ---
    input_file = "BBB.xlsx"
    df_in = pd.read_excel(input_file, engine="openpyxl")

    if "SMILES" not in df_in.columns:
        raise ValueError("ستونی با نام 'SMILES' در فایل ورودی یافت نشد.")

    # --- ۲. حذف SMILESهای نامعتبر ---
    # تابعی برای چک اعتبار SMILES
    def is_valid_smiles(smi):
        try:
            return Chem.MolFromSmiles(smi) is not None
        except:
            return False

    valid_mask = df_in["SMILES"].dropna().astype(str).apply(is_valid_smiles)
    df_valid = df_in.loc[valid_mask].reset_index(drop=True)
    print(f"Filtered out {len(df_in) - len(df_valid)} invalid SMILES. {len(df_valid)} remain.")

    smiles_list = df_valid["SMILES"].tolist()

    # --- ۳. محاسبه دسکریپتورها با tqdm ---
    df_desc = calc_descriptors(smiles_list)

    # --- ۴. ادغام با اطلاعات اولیه معتبر ---
    df_out = pd.merge(df_valid, df_desc, on="SMILES", how="left")

    # --- ۵. ذخیره به فایل اکسل خروجی ---
    output_file = "BBB_descriptors.xlsx"
    df_out.to_excel(output_file, index=False, engine="openpyxl")

    print(f"\n✅ دسکریپتورها با موفقیت در '{output_file}' ذخیره شدند.")