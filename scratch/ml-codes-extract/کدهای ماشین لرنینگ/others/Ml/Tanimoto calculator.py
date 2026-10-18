import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.DataStructs import TanimotoSimilarity
from tqdm import tqdm

# ————— ۱. خواندن لیست مراجع (References) —————
# فایل اکسل حاوی مرجع‌ها را باز کنید (ستونی به نام "SMILES_REF" داشته باشد):
#df_ref = pd.read_excel('1234.xlsx')
# اگر CSV دارید:
df_ref = pd.read_csv(r"C:\Users\lenovo\new\Drugs.csv", usecols=["SMILES_REF"])

# حذف فاصله‌های اضافی از نام ستون (در صورت وجود)
df_ref.columns = df_ref.columns.str.strip()
if "SMILES_REF" not in df_ref.columns:
    raise KeyError("ستونی با نام 'SMILES_REF' در فایل مراجع یافت نشد! فهرست ستون‌ها را بررسی کنید.")

# تبدیل ستونی به لیست و حذف مقادیر NaN
reference_list = df_ref["SMILES_REF"].dropna().astype(str).tolist()

# ————— ۲. خواندن لیست اهداف (Targets) —————
#df_target = pd.read_excel('Smiles  FDA approved drugs.xlsx')
# اگر CSV:
df_target = pd.read_csv(r"C:\Users\lenovo\new\Analogues.csv", usecols=["SMILES"])

df_target.columns = df_target.columns.str.strip()
if "SMILES" not in df_target.columns:
    raise KeyError("ستونی با نام 'SMILES' در فایل اهداف یافت نشد! فهرست ستون‌ها را بررسی کنید.")

# حذف NaN و تبدیل به لیست
target_list = df_target["SMILES"].dropna().astype(str).tolist()

# ————— ۳. محاسبهٔ Fingerprint برای تمام اهداف —————
# بهتر است ابتدا Fingerprint همهٔ "Targets" را یک‌جا بسازیم تا در حلقهٔ دوتایی،
# هر بار دوباره مولکول‌سازی و FP نکنیم. اگر تعداد خیلی زیادی هدف دارید،
# ممکن است به حافظه فشار بیاید؛ ولی اگر در محدودهٔ معمول (چند هزار) هستید،
# این کار باعث تسریع زیادی خواهد شد.

print("در حال ساخت Fingerprint برای SMILESهای اهداف…")
target_fps = []  # لیستی که در همان اندیسِ مطابق با target_list قرار می‌گیرد.
valid_target_idx = []  # نگه‌داشتن اندیس‌هایی که مولکول ساختند.

for idx, smi in enumerate(tqdm(target_list, desc="Targets FP", unit="mol")):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        # SMILES نامعتبر → گنجاندن None در همان اندیس
        target_fps.append(None)
    else:
        fp = RDKFingerprint(mol)
        target_fps.append(fp)
        valid_target_idx.append(idx)

# ————— ۴. محاسبهٔ شباهت‌ها (Tanimoto) برای هر مرجع و هر هدف —————
# ما می‌خواهیم یک DataFrame طولی بسازیم که هر سطر به این شکل باشد:
# | SMILES_REF       | SMILES_TARGET   | Tanimoto  |
# و سپس آن را در Excel ذخیره کنیم.

records = []
print("\nدر حال محاسبهٔ شباهت Tanimoto بین مراجع و اهداف…")
for ref_smi in tqdm(reference_list, desc="References", unit="ref"):
    ref_mol = Chem.MolFromSmiles(ref_smi)
    if ref_mol is None:
        # اگر SMILES مرجع نامعتبر بود، برای همهٔ اهداف None بگذاریم:
        for tgt_idx, tgt_smi in enumerate(target_list):
            records.append({
                "SMILES_REF": ref_smi,
                "SMILES_TARGET": tgt_smi,
                "Tanimoto": None
            })
        continue

    ref_fp = RDKFingerprint(ref_mol)

    # برای هر هدف:
    for tgt_idx, tgt_smi in enumerate(target_list):
        fp_tgt = target_fps[tgt_idx]
        if fp_tgt is None:
            # هدف نامعتبر یا FP ساخته نشده
            sim = None
        else:
            sim = TanimotoSimilarity(ref_fp, fp_tgt)
        records.append({
            "SMILES_REF": ref_smi,
            "SMILES_TARGET": tgt_smi,
            "Tanimoto": sim
        })

# ۵. ساخت DataFrame طولی
df_results = pd.DataFrame(records)

# ۶. تبدیل ستون Tanimoto به عددی (float) و حذف مقادیر None/NaN برای اطمینان
# اگر ستون از نوع object باشد و شامل None یا اشیای غیرعدد باشد، ابتدا آن را به float تبدیل می‌کنیم:
df_results["Tanimoto"] = pd.to_numeric(df_results["Tanimoto"], errors="coerce")

# ۷. فیلتر: فقط نگه داشتن اعدادی که ≥ 0.9 هستند
df_filtered = df_results[df_results["Tanimoto"] >= 0.9].reset_index(drop=True)

# ۸. ذخیرهٔ خروجی فیلترشده در اکسل
output_excel = r"C:\Users\lenovo\new\tanimatoesanalogues.xlsx"
df_filtered.to_excel(output_excel, index=False)

print(f"\n✅ تنها جفت‌هایی که Tanimoto ≥ 0.۹ داشتند در '{output_excel}' ذخیره شدند.")