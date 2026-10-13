import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.AtomPairs import Pairs      # <-- added

# -------------------------
# تابع کمکی برای محاسبه فقط فیچرهای ذخیره‌شده روی یک CSV جدید
# -------------------------
def compute_selected_features_for_new_csv(new_csv_path,
                                          out_csv_path,
                                          selected_features_json='selected_features.json',
                                          smiles_col='SMILES',
                                          target_col=None,
                                          verbose=True):
    """
    new_csv_path: مسیر فایل CSV جدید حاوی ستون SMILES (و در صورت وجود target_col)
    out_csv_path: مسیر خروجی CSV که شامل SMILES و (در صورت وجود) target و فیچرهای انتخاب‌شده خواهد بود
    selected_features_json: فایل metadata که قبلاً ذخیره شده
    smiles_col: نام ستون SMILES در فایل ورودی
    target_col: در صورت وجود نام ستون هدف/Target را وارد کن (مثلاً 'pIC50') - اگر None باشد، ستون Target اضافه نمیشود
    """
    import pandas as pd
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Avalon import pyAvalonTools
    from rdkit.Chem.AtomPairs import Pairs    # <-- برای AtomPair
    import datamol as dm   # ممکن است برای بخش‌های دیگر نیاز باشد

    # helper (در صورت نبود، از نسخه‌های موجود در اسکریپت استفاده می‌کنیم)
    def safe_mol_local(smi):
        try:
            m = Chem.MolFromSmiles(smi, sanitize=False)
            Chem.SanitizeMol(m)
            return m
        except:
            return None

    def get_avalon_count_fp_dense_local(mol, fp_size=512):
        if mol is None:
            return np.zeros(fp_size, dtype=int)
        fp_sparse = pyAvalonTools.GetAvalonCountFP(mol)
        fp_dense = np.zeros(fp_size, dtype=int)
        for idx, count in fp_sparse.GetNonzeroElements().items():
            if idx < fp_size:
                fp_dense[idx] = count
        return fp_dense

    def get_morgan_fp_local(mol, radius=2, nBits=2048):
        if mol is None:
            return np.zeros(nBits, dtype=int)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits))

    def get_murcko_scaffold_mol_local(mol):
        if mol is None:
            return None
        try:
            return MurckoScaffold.GetMurckoScaffold(mol)
        except Exception:
            return None

    # ========== اضافه شده: AtomPair count fingerprint با سایز دلخواه ==========
    def get_atompair_count_fp_local(mol, size):
        """Return a dense numpy array of length `size` with atom‑pair counts."""
        if mol is None:
            return np.zeros(size, dtype=int)
        fp_sparse = Pairs.GetAtomPairFingerprint(mol)   # SparseIntVect
        dense = np.zeros(size, dtype=int)
        for idx, cnt in fp_sparse.GetNonzeroElements().items():
            if idx < size:
                dense[idx] = cnt
        return dense
    # ========================================================================

    # بارگزاری متادیتا
    with open(selected_features_json, 'r') as f:
        meta = json.load(f)

    selected_features = meta.get('selected_features', [])
    descriptor_names_saved = meta.get('descriptor_names', [])
    fp_sizes = meta.get('fingerprint_sizes', {})

    # خواندن CSV جدید
    df_new = pd.read_csv(new_csv_path, dtype={smiles_col: str})
    if smiles_col not in df_new.columns:
        raise ValueError(f"ستون '{smiles_col}' در فایل ورودی پیدا نشد.")

    # مول‌ها
    df_new['mol'] = df_new[smiles_col].apply(safe_mol_local)
    n_invalid = df_new['mol'].isna().sum()
    if verbose:
        print(f"مجموع مول‌ها: {len(df_new)} — مول‌های نامعتبر: {n_invalid}")

    # آماده‌سازی لیست نیازمندی‌ها بر اساس prefix
    desc_needed = [s.replace('desc_', '') for s in selected_features if s.startswith('desc_')]
    apc_needed_indices = sorted([int(s.replace('apc_bit_', '')) for s in selected_features if s.startswith('apc_bit_')])
    ecfp_needed_indices = sorted([int(s.replace('ecfp_bit_', '')) for s in selected_features if s.startswith('ecfp_bit_')])
    avalon_needed_indices = sorted([int(s.replace('avalon_count_', '')) for s in selected_features if s.startswith('avalon_count_')])
    rdkit_needed_indices = sorted([int(s.replace('rdkit_bit_', '')) for s in selected_features if s.startswith('rdkit_bit_')])
    morgan_full_needed_indices = sorted([int(s.replace('morgan_full_bit_', '')) for s in selected_features if s.startswith('morgan_full_bit_')])
    morgan_scaffold_needed_indices = sorted([int(s.replace('morgan_scaffold_bit_', '')) for s in selected_features if s.startswith('morgan_scaffold_bit_')])

    # ---------- محاسبه descriptors ----------
    df_out_parts = []
    if desc_needed:
        if not descriptor_names_saved:
            if verbose: print("⚠️ نام descriptorها ذخیره نشده؛ تلاش می‌کنم از RDKit خواص موجود را استخراج کنم.")
            descriptor_names_saved = list(rdMolDescriptors.Properties.GetAvailableProperties())
        prop_getter = rdMolDescriptors.Properties(descriptor_names_saved)
        props_list = []
        for smi in df_new[smiles_col].fillna('').tolist():
            try:
                m = Chem.MolFromSmiles(smi)
                if m:
                    Chem.SanitizeMol(m)
                    props_list.append(prop_getter.ComputeProperties(m))
                else:
                    props_list.append([np.nan]*len(descriptor_names_saved))
            except:
                props_list.append([np.nan]*len(descriptor_names_saved))
        df_props = pd.DataFrame(props_list, columns=[f'desc_{n}' for n in descriptor_names_saved])
        have_desc_cols = [f'desc_{d}' for d in desc_needed if f'desc_{d}' in df_props.columns]
        df_out_parts.append(df_props[have_desc_cols])
        if verbose: print(f"  → {len(have_desc_cols)} مورد descriptor محاسبه و اضافه شد.")

    # ---------- محاسبه AtomPair-Count (اصلاح شده) ----------
    if apc_needed_indices:
        # سایز مورد نیاز = آخرین ایندکس مورد نظر + 1
        apc_size = max(apc_needed_indices) + 1
        # محاسبه برای همه مولکول‌ها
        apc_array = np.vstack([
            get_atompair_count_fp_local(mol, apc_size) for mol in df_new['mol']
        ])
        # انتخاب فقط ستون‌هایی که در selected_features ذخیره شده‌اند
        selected_apc_indices = [i for i in apc_needed_indices if i < apc_size]
        df_apc_selected = pd.DataFrame(
            apc_array[:, selected_apc_indices],
            columns=[f'apc_bit_{i}' for i in selected_apc_indices]
        )
        df_out_parts.append(df_apc_selected)
        if verbose:
            print(f"  → {df_apc_selected.shape[1]} بیت AtomPair-Count اضافه شد (از {len(apc_needed_indices)} درخواست‌شده).")

    # ---------- محاسبه ECFP (Morgan radius=2) ----------
    if ecfp_needed_indices:
        nbits = fp_sizes.get('ecfp', fp_sizes.get('morgan_full', 2048))
        ecfp_array = np.vstack([get_morgan_fp_local(m, radius=2, nBits=nbits) for m in df_new['mol']])
        max_idx = ecfp_array.shape[1]-1
        selected_ecfp_indices = [i for i in ecfp_needed_indices if i <= max_idx]
        df_ecfp_selected = pd.DataFrame(ecfp_array[:, selected_ecfp_indices], columns=[f'ecfp_bit_{i}' for i in selected_ecfp_indices])
        df_out_parts.append(df_ecfp_selected)
        if verbose: print(f"  → {df_ecfp_selected.shape[1]} بیت ECFP اضافه شد (از {len(ecfp_needed_indices)} درخواست‌شده).")

    # ---------- محاسبه Avalon-Count ----------
    if avalon_needed_indices:
        avalon_size = fp_sizes.get('avalon', 512)
        avalon_array = np.vstack([get_avalon_count_fp_dense_local(m, fp_size=avalon_size) for m in df_new['mol']])
        max_idx = avalon_array.shape[1]-1
        selected_avalon_indices = [i for i in avalon_needed_indices if i <= max_idx]
        df_avalon_selected = pd.DataFrame(avalon_array[:, selected_avalon_indices], columns=[f'avalon_count_{i}' for i in selected_avalon_indices])
        df_out_parts.append(df_avalon_selected)
        if verbose: print(f"  → {df_avalon_selected.shape[1]} بیت Avalon-Count اضافه شد (از {len(avalon_needed_indices)} درخواست‌شده).")

    # ---------- محاسبه RDKit FP ----------
    if rdkit_needed_indices:
        rdkit_n = fp_sizes.get('rdkit', 2048)
        rdkit_array = np.vstack([np.array(Chem.RDKFingerprint(m, fpSize=rdkit_n)) if m is not None else np.zeros(rdkit_n, dtype=int) for m in df_new['mol']])
        max_idx = rdkit_array.shape[1]-1
        selected_rdkit_indices = [i for i in rdkit_needed_indices if i <= max_idx]
        df_rdkit_selected = pd.DataFrame(rdkit_array[:, selected_rdkit_indices], columns=[f'rdkit_bit_{i}' for i in selected_rdkit_indices])
        df_out_parts.append(df_rdkit_selected)
        if verbose: print(f"  → {df_rdkit_selected.shape[1]} بیت RDKit-FP اضافه شد (از {len(rdkit_needed_indices)} درخواست‌شده).")

    # ---------- محاسبه Morgan Full ----------
    if morgan_full_needed_indices:
        nbits = fp_sizes.get('morgan_full', 2048)
        morgan_full_array = np.vstack([get_morgan_fp_local(m, radius=2, nBits=nbits) for m in df_new['mol']])
        max_idx = morgan_full_array.shape[1]-1
        selected_morgan_full_indices = [i for i in morgan_full_needed_indices if i <= max_idx]
        df_morgan_full_selected = pd.DataFrame(morgan_full_array[:, selected_morgan_full_indices], columns=[f'morgan_full_bit_{i}' for i in selected_morgan_full_indices])
        df_out_parts.append(df_morgan_full_selected)
        if verbose: print(f"  → {df_morgan_full_selected.shape[1]} بیت Morgan-Full اضافه شد (از {len(morgan_full_needed_indices)} درخواست‌شده).")

    # ---------- محاسبه Morgan از Murcko Scaffolds ----------
    if morgan_scaffold_needed_indices:
        nbits = fp_sizes.get('morgan_scaffold', 2048)
        scaffold_mols = [get_murcko_scaffold_mol_local(m) for m in df_new['mol']]
        scaffold_array = np.vstack([get_morgan_fp_local(sm, radius=2, nBits=nbits) if sm is not None else np.zeros(nbits, dtype=int) for sm in scaffold_mols])
        max_idx = scaffold_array.shape[1]-1
        selected_morgan_scaffold_indices = [i for i in morgan_scaffold_needed_indices if i <= max_idx]
        df_morgan_scaffold_selected = pd.DataFrame(scaffold_array[:, selected_morgan_scaffold_indices], columns=[f'morgan_scaffold_bit_{i}' for i in selected_morgan_scaffold_indices])
        df_out_parts.append(df_morgan_scaffold_selected)
        if verbose: print(f"  → {df_morgan_scaffold_selected.shape[1]} بیت Morgan-Scaffold اضافه شد (از {len(morgan_scaffold_needed_indices)} درخواست‌شده).")

    # ---------- ترکیب نهایی ----------
    df_features_combined = pd.concat(df_out_parts, axis=1) if df_out_parts else pd.DataFrame(index=df_new.index)
    df_result = pd.DataFrame()
    df_result[smiles_col] = df_new[smiles_col]
    if target_col and target_col in df_new.columns:
        df_result[target_col] = df_new[target_col]

    for feat in selected_features:
        if feat in df_features_combined.columns:
            df_result[feat] = df_features_combined[feat].values
        else:
            df_result[feat] = 0

    df_result.to_csv(out_csv_path, index=False)
    if verbose:
        print(f"✅ فایل خروجی ساخته شد: {out_csv_path} — ابعاد: {df_result.shape}")

# -------------------------
# مثال استفاده:
# -------------------------
if __name__ == "__main__":
    compute_selected_features_for_new_csv('prediction_class.csv', 'newfile_selected_features.csv', selected_features_json='selected_features.json')