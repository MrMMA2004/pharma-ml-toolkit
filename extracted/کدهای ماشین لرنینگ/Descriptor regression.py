import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Avalon import pyAvalonTools
import numpy as np
import datamol as dm
from tqdm.auto import tqdm
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostRegressor  # CatBoostRegressor remains
import warnings

warnings.filterwarnings('ignore')

tqdm.pandas()

# --- Data Loading and Initial Cleaning ---
print("\n--- Data Loading and Initial Cleaning ---")
# 2) Load & clean SMILES column
df = pd.read_csv('newclass.csv', dtype={'SMILES': str})
df['SMILES'].replace({'nan': None}, inplace=True)
df = df.dropna(subset=['SMILES']).reset_index(drop=True)
print(f"Cleaned data size: {len(df)}")

# 3) Safe mol builder
tqdm.pandas()
def safe_mol(smi):
    try:
        m = Chem.MolFromSmiles(smi, sanitize=False)
        Chem.SanitizeMol(m)
        return m
    except:
        return None

df['mol'] = df.SMILES.progress_apply(safe_mol)
df = df.dropna(subset=['mol']).reset_index(drop=True)
print(f"Valid molecules after sanitization: {len(df)}")


# Define a function to evaluate models for reusability
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    print(f"\n--- Evaluating {model_name} ---")

    if y_train.isnull().any() or np.isinf(y_train).any():
        print(f"  Skipping {model_name} training due to NaNs/Infs in target variable (y_train).")
        return np.nan, np.nan, np.nan  # Return 3 values (even if nan)

    if X_train.isnull().any().any() or np.isinf(X_train).any().any():
        print(
            f"  Warning: NaNs or Infs found in X_train for {model_name}. Attempting to train but this can cause issues.")

    # CatBoost specific check for NaN in features:
    # CatBoost can handle NaNs by default if specified (nan_mode),
    # but for consistency with other models, we ensure no NaNs/Infs here.
    if isinstance(model, CatBoostRegressor):
        # Convert to numpy array for CatBoost, it handles NaNs internally if specified
        # but our previous scaling and fillna(0) should handle this.
        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"  {model_name} R2 Score: {r2:.4f}")
    print(f"  {model_name} MSE: {mse:.4f}")
    print(f"  {model_name} MAE: {mae:.4f}")
    return r2, mse, mae  # Ensure all 3 values are returned


# Define KFold for consistent cross-validation splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Step: Feature Generation (Descriptors, Atompair-Count, ECFP, Avalon-Count, RDKit, Morgan (Full), Morgan (Murcko)) ---
print(
    "\n--- Generating All Feature Types: Descriptors, Atompair-Count, ECFP, Avalon-Count, RDKit, Morgan (Full), Morgan (Murcko) ---")

# Descriptors
property_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
property_getter = rdMolDescriptors.Properties(property_names)


def smi2props(smi):
    mol = Chem.MolFromSmiles(smi)
    props = None
    if mol:
        Chem.SanitizeMol(mol)
        props = np.array(property_getter.ComputeProperties(mol))
    return props


df_temp_props = df.query("~mol.isna()").reset_index(drop=True).copy()
df_temp_props['props'] = df_temp_props.SMILES.progress_apply(lambda s: smi2props(s))
df_desc_valid = df_temp_props.query("~props.isna()").reset_index(drop=True)
try:
    df_props = pd.DataFrame(df_desc_valid['props'].to_list(), columns=property_names)
    df_props.columns = [f'desc_{col}' for col in df_props.columns]
except ValueError as e:
    print(f"Error expanding descriptor properties: {e}. Exiting.")
    exit()
print(f"  Generated {len(property_names)} descriptors.")

# Prepare base dataframe for fingerprints (ensure it has valid molecules)
df_current_mols = df_desc_valid.copy()
initial_num_mols = len(df_current_mols)

# Atompair-Count Fingerprints
fps_apc_gen = df_current_mols.mol.progress_apply(lambda m: dm.to_fp(m, fp_type='atompair-count', as_array=True))
valid_indices_apc = [i for i, x in enumerate(fps_apc_gen) if x is not None and isinstance(x, np.ndarray) and x.ndim > 0]
df_current_mols = df_current_mols.iloc[valid_indices_apc].reset_index(drop=True)
fps_apc_gen = fps_apc_gen.iloc[valid_indices_apc]
df_apc_fps = pd.DataFrame(fps_apc_gen.to_list())
df_apc_fps.columns = [f'apc_bit_{i}' for i in range(df_apc_fps.shape[1])]
print(f"  Generated {df_apc_fps.shape[1]} atompair-count fingerprints. Molecules remaining: {len(df_current_mols)}")

# ECFP Fingerprints (Morgan Fingerprint with radius 2, as a bit vector)
fps_ecfp_gen = df_current_mols.mol.progress_apply(
    lambda m: np.array(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)))
valid_indices_ecfp = [i for i, x in enumerate(fps_ecfp_gen) if
                      x is not None and isinstance(x, np.ndarray) and x.ndim > 0]
df_current_mols = df_current_mols.iloc[valid_indices_ecfp].reset_index(drop=True)
fps_ecfp_gen = fps_ecfp_gen.iloc[valid_indices_ecfp]
df_ecfp_fps = pd.DataFrame(fps_ecfp_gen.to_list())
df_ecfp_fps.columns = [f'ecfp_bit_{i}' for i in range(df_ecfp_fps.shape[1])]
print(f"  Generated {df_ecfp_fps.shape[1]} ECFP fingerprints. Molecules remaining: {len(df_current_mols)}")

# Avalon-Count Fingerprints
AVALON_FINGERPRINT_SIZE = 512


def get_avalon_count_fp_dense(mol, fp_size=AVALON_FINGERPRINT_SIZE):
    if mol is None: return np.zeros(fp_size, dtype=int)
    fp_sparse = pyAvalonTools.GetAvalonCountFP(mol)
    fp_dense = np.zeros(fp_size, dtype=int)
    for idx, count in fp_sparse.GetNonzeroElements().items():
        if idx < fp_size: fp_dense[idx] = count
    return fp_dense


fps_avalon_gen = df_current_mols.mol.progress_apply(lambda m: get_avalon_count_fp_dense(m))
df_avalon_fps = pd.DataFrame(fps_avalon_gen.to_list())
df_avalon_fps.columns = [f'avalon_count_{i}' for i in range(df_avalon_fps.shape[1])]
print(f"  Generated {df_avalon_fps.shape[1]} Avalon count fingerprints. Molecules remaining: {len(df_current_mols)}")

# RDKit Fingerprints
fps_rdkit_gen = df_current_mols.mol.progress_apply(lambda m: np.array(Chem.RDKFingerprint(m, fpSize=2048)))
valid_indices_rdkit = [i for i, x in enumerate(fps_rdkit_gen) if
                       x is not None and isinstance(x, np.ndarray) and x.ndim > 0]
df_current_mols = df_current_mols.iloc[valid_indices_rdkit].reset_index(drop=True)
fps_rdkit_gen = fps_rdkit_gen.iloc[valid_indices_rdkit]
df_rdkit_fps = pd.DataFrame(fps_rdkit_gen.to_list())
df_rdkit_fps.columns = [f'rdkit_bit_{i}' for i in range(df_rdkit_fps.shape[1])]
print(f"  Generated {df_rdkit_fps.shape[1]} RDKit Fingerprints. Molecules remaining: {len(df_current_mols)}")

# Morgan Fingerprints (from full molecule)
print("  Generating Morgan Fingerprints (from full molecules)...")
radius_morgan = 2
nBits_morgan = 2048


def get_morgan_fp(mol, radius=radius_morgan, nBits=nBits_morgan):
    if mol is None:
        return np.full(nBits, 0, dtype=np.int8)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits))


fps_morgan_full_gen = df_current_mols['mol'].progress_apply(get_morgan_fp)
valid_indices_morgan_full = [i for i, x in enumerate(fps_morgan_full_gen) if
                             x is not None and isinstance(x, np.ndarray) and x.ndim > 0]
df_current_mols = df_current_mols.iloc[valid_indices_morgan_full].reset_index(drop=True)
fps_morgan_full_gen = fps_morgan_full_gen.iloc[valid_indices_morgan_full]
df_morgan_fps_full = pd.DataFrame(fps_morgan_full_gen.tolist())
df_morgan_fps_full.columns = [f'morgan_full_bit_{i}' for i in range(df_morgan_fps_full.shape[1])]
print(
    f"  Generated {df_morgan_fps_full.shape[1]} Morgan Fingerprints (full molecule). Molecules remaining: {len(df_current_mols)}")

# Murcko Scaffolds and Morgan Fingerprints from Scaffolds
print("  Generating Murcko Scaffolds and Morgan Fingerprints from them...")


def get_murcko_scaffold_mol(mol):
    if mol is None:
        return None
    try:
        return MurckoScaffold.GetMurckoScaffold(mol)
    except Exception:
        return None


df_current_mols['murcko_mol'] = df_current_mols['mol'].progress_apply(get_murcko_scaffold_mol)

df_with_scaffolds = df_current_mols.dropna(subset=['murcko_mol']).copy()
print(f"  Generated Murcko Scaffolds for {len(df_with_scaffolds)} molecules.")
if len(df_current_mols) - len(df_with_scaffolds) > 0:
    print(
        f"  Warning: {len(df_current_mols) - len(df_with_scaffolds)} molecules had no Murcko scaffold (e.g., linear structures or issues).")

fps_morgan_scaffold_gen = df_with_scaffolds['murcko_mol'].progress_apply(get_morgan_fp)
df_morgan_fps_scaffold = pd.DataFrame(fps_morgan_scaffold_gen.tolist())
df_morgan_fps_scaffold.columns = [f'morgan_scaffold_bit_{i}' for i in range(df_morgan_fps_scaffold.shape[1])]
print(f"  Generated {df_morgan_fps_scaffold.shape[1]} Morgan Fingerprints (from Murcko Scaffolds).")

print(
    f"\n--- Initial Molecules: {initial_num_mols}, Molecules after all FP generation and filtering: {len(df_current_mols)} ---")
print("  The number of molecules might decrease if any molecules failed to generate specific fingerprints.")

# Align all feature dataframes to the final set of common molecules
common_pIC50 = df_current_mols.pIC50.astype(float).reset_index(drop=True)

print("\n--- Re-generating and aligning all feature DataFrames for consistency ---")

df_props_aligned = pd.DataFrame(df_current_mols.SMILES.progress_apply(lambda s: smi2props(s)).to_list(),
                                columns=property_names)
df_props_aligned.columns = [f'desc_{col}' for col in df_props_aligned.columns]

df_apc_fps_aligned = pd.DataFrame(
    df_current_mols.mol.progress_apply(lambda m: dm.to_fp(m, fp_type='atompair-count', as_array=True)).to_list())
df_apc_fps_aligned.columns = [f'apc_bit_{i}' for i in range(df_apc_fps_aligned.shape[1])]

df_ecfp_fps_aligned = pd.DataFrame(df_current_mols.mol.progress_apply(
    lambda m: np.array(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048))).to_list())
df_ecfp_fps_aligned.columns = [f'ecfp_bit_{i}' for i in range(df_ecfp_fps_aligned.shape[1])]

df_avalon_fps_aligned = pd.DataFrame(
    df_current_mols.mol.progress_apply(lambda m: get_avalon_count_fp_dense(m)).to_list())
df_avalon_fps_aligned.columns = [f'avalon_count_{i}' for i in range(df_avalon_fps_aligned.shape[1])]

df_rdkit_fps_aligned = pd.DataFrame(
    df_current_mols.mol.progress_apply(lambda m: np.array(Chem.RDKFingerprint(m, fpSize=2048))).to_list())
df_rdkit_fps_aligned.columns = [f'rdkit_bit_{i}' for i in range(df_rdkit_fps_aligned.shape[1])]

df_morgan_fps_full_aligned = pd.DataFrame(df_current_mols.mol.progress_apply(lambda m: get_morgan_fp(m)).to_list())
df_morgan_fps_full_aligned.columns = [f'morgan_full_bit_{i}' for i in range(df_morgan_fps_full_aligned.shape[1])]

temp_scaffold_mol_series = df_current_mols.mol.progress_apply(get_murcko_scaffold_mol)
temp_scaffold_fp_list = temp_scaffold_mol_series.apply(get_morgan_fp).tolist()
df_morgan_fps_scaffold_aligned = pd.DataFrame(temp_scaffold_fp_list)
df_morgan_fps_scaffold_aligned.columns = [f'morgan_scaffold_bit_{i}' for i in
                                          range(df_morgan_fps_scaffold_aligned.shape[1])]

# --- First Layer Feature Selection (Per Type, using median importance) ---
print("\n--- Performing First Layer Feature Selection (Per Feature Type) ---")

selected_feature_dfs = []
feature_sets_for_selection = {
    "Descriptors": df_props_aligned,
    "AtomPair-Count": df_apc_fps_aligned,
    "ECFP": df_ecfp_fps_aligned,
    "Avalon-Count": df_avalon_fps_aligned,
    "RDKit Fingerprints": df_rdkit_fps_aligned,
    "Morgan_Full": df_morgan_fps_full_aligned,
    "Morgan_Scaffold": df_morgan_fps_scaffold_aligned,
}

for name, X_features_df in feature_sets_for_selection.items():
    print(f"\n  Applying feature selection on {name}...")

    X_features_cleaned = X_features_df.astype(float).fillna(0)

    lgbm_selector_model = LGBMRegressor(random_state=42, n_jobs=3, n_estimators=200)

    if X_features_cleaned.empty or X_features_cleaned.shape[1] == 0:
        print(f"  Skipping feature selection for {name} as it has no features or is empty.")
        continue

    lgbm_selector_model.fit(X_features_cleaned, common_pIC50)

    selector = SelectFromModel(lgbm_selector_model, prefit=True, threshold='median')

    X_selected_array = selector.transform(X_features_cleaned)

    selected_cols_indices = selector.get_support(indices=True)
    selected_column_names = X_features_df.columns[selected_cols_indices]

    df_selected_features = pd.DataFrame(X_selected_array, columns=selected_column_names)
    selected_feature_dfs.append(df_selected_features)
    print(f"  {len(selected_column_names)} features selected from {name}.")

# Concatenate all first-layer selected features into one combined DataFrame
X_data_combined = pd.concat(selected_feature_dfs, axis=1)
print(f"\n  Combined feature matrix after first layer selection: {X_data_combined.shape}")

# --- Second Layer Feature Selection (Top K features from Combined Set) ---
print("\n--- Performing Second Layer Feature Selection (Top K features from Combined Set) ---")

k_features_to_select = 200 #200/500/1000/1500

if X_data_combined.shape[1] > k_features_to_select:
    print(
        f"  Selecting top {k_features_to_select} features from the combined set ({X_data_combined.shape[1]} available)...")

    overall_lgbm_model = LGBMRegressor(random_state=42, n_jobs=3, n_estimators=300)
    overall_lgbm_model.fit(X_data_combined.astype(float).fillna(0), common_pIC50)

    feature_importances = pd.Series(overall_lgbm_model.feature_importances_, index=X_data_combined.columns)
    top_k_features = feature_importances.nlargest(k_features_to_select).index

    X_data_final_optimized = X_data_combined[top_k_features]
    print(
        f"  Final optimized feature matrix after selecting top {k_features_to_select} features: {X_data_final_optimized.shape}")
else:
    print(
        f"  Number of combined features ({X_data_combined.shape[1]}) is already less than or equal to desired k ({k_features_to_select}). No further selection applied.")
    X_data_final_optimized = X_data_combined.copy()
# ذخیره خروجی
X_data_final_optimized.insert(0, 'Activity', common_pIC50)  # اضافه کردن ستون pIC50
X_data_final_optimized.insert(0, 'SMILES', df_current_mols.SMILES)
X_data_final_optimized.to_csv('200selected_features_regression.csv', index=False)

print("\n✅ خروجی ذخیره شد در فایل: selected_features.csv")
