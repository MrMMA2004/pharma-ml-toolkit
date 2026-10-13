# predict_from_bundle.py (regression version)
import pandas as pd
import joblib
import json
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------
# 🔧 تنظیم مسیرهای فایل‌ها
# --------------------------
NEW_CSV_PATH = "newfile_selected_features.csv"       # مسیر فایل CSV جدید
BUNDLE_PREFIX = "svr_bundle"     # بدون پسوند (فقط prefix)
OUTPUT_PREFIX = "predictions"       # مسیر و نام فایل خروجی بدون پسوند

# -----------------------------------------------------
# از اینجا به پایین نیازی به تغییر نیست
# -----------------------------------------------------
def load_bundle(bundle_prefix=BUNDLE_PREFIX):
    pipeline_path = f'{bundle_prefix}_pipeline.joblib'
    metadata_path = f'{bundle_prefix}_metadata.json'

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    pipeline = joblib.load(pipeline_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    if 'feature_columns' not in metadata:
        raise KeyError("metadata must contain 'feature_columns' key.")
    # metadata['target_col'] is optional but helpful if present
    return pipeline, metadata


def _predict_with_optional_std(pipeline, X):
    """
    Try to get (preds, std) if estimator supports return_std=True.
    Otherwise return preds and stds=None.
    """
    # First try predict with return_std=True (for GPR-like estimators)
    try:
        res = pipeline.predict(X, return_std=True)
        # sklearn convention: (preds, std) or (preds, sigma)
        if isinstance(res, tuple) and len(res) == 2:
            preds, std = res
            std = np.asarray(std)
            preds = np.asarray(preds)
            # ensure shape
            return preds, std
    except TypeError:
        # pipeline.predict may not accept return_std
        pass
    except Exception:
        # if any other error, ignore and fallback to normal predict
        pass

    # fallback: only predict
    preds = np.asarray(pipeline.predict(X))
    return preds, None


def predict_new_csv(new_csv_path=NEW_CSV_PATH, bundle_prefix=BUNDLE_PREFIX, output_prefix=OUTPUT_PREFIX):
    pipeline, metadata = load_bundle(bundle_prefix)
    print(f"✅ Loaded bundle from {bundle_prefix}")

    df_new = pd.read_csv(new_csv_path)
    print(f"✅ Loaded new CSV: {new_csv_path} | shape = {df_new.shape}")

    feature_cols = metadata['feature_columns']
    smiles_col = metadata.get('smiles_col', 'SMILES')
    # optional: name of true target column in new csv
    target_col = metadata.get('target_col') or metadata.get('y_col') or metadata.get('target')  # try several common keys

    missing = [c for c in feature_cols if c not in df_new.columns]
    if missing:
        raise ValueError(f"❌ Missing required feature columns: {missing}")

    X_new = df_new[feature_cols].copy()
    X_new = X_new.apply(pd.to_numeric, errors='coerce')
    if X_new.isna().any().any():
        # give more detail which columns have NaNs
        nan_cols = X_new.columns[X_new.isna().any()].tolist()
        raise ValueError(f"❌ Some features contain NaN or non-numeric values. Columns with NaN: {nan_cols}")

    print("🔹 Running predictions (regression)...")
    y_pred, y_std = _predict_with_optional_std(pipeline, X_new)

    df_out = pd.DataFrame()
    if smiles_col in df_new.columns:
        df_out['SMILES'] = df_new[smiles_col]

    df_out['Predicted'] = y_pred

    if y_std is not None:
        # If std is a single value per sample, add it
        if y_std.shape[0] == df_out.shape[0]:
            df_out['Pred_STD'] = y_std
        else:
            # fallback: try to flatten
            df_out['Pred_STD'] = np.asarray(y_std).reshape(-1)[: df_out.shape[0]]

    # If true target column exists in CSV (or metadata), include it and compute metrics
    true_present = False
    if target_col and target_col in df_new.columns:
        df_out['True'] = pd.to_numeric(df_new[target_col], errors='coerce')
        true_present = True
    else:
        # try some common names
        for cand in ['target', 'y', 'Value', 'value', 'LABEL', 'label']:
            if cand in df_new.columns:
                df_out['True'] = pd.to_numeric(df_new[cand], errors='coerce')
                target_col = cand
                true_present = True
                print(f"ℹ️ Found target column '{cand}' in new CSV and using it for metrics.")
                break

    if true_present:
        # remove rows with NaN in True to compute metrics reliably
        valid_mask = ~df_out['True'].isna()
        if valid_mask.sum() == 0:
            print("⚠️ True target column exists but all values are NaN. Skipping metrics.")
        else:
            y_true = df_out.loc[valid_mask, 'True'].values
            y_p = df_out.loc[valid_mask, 'Predicted'].values
            df_out['Residual'] = df_out['Predicted'] - df_out['True']
            mae = mean_absolute_error(y_true, y_p)
            mse = mean_squared_error(y_true, y_p)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_p)
            print("📊 Regression metrics (on rows where True is present):")
            print(f"    MAE  = {mae:.6f}")
            print(f"    MSE  = {mse:.6f}")
            print(f"    RMSE = {rmse:.6f}")
            print(f"    R2   = {r2:.6f}")
    else:
        print("ℹ️ No true target column found — only predictions will be saved.")

    out_csv = f'{output_prefix}_results.csv'
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Saved results to:\n  {out_csv}")

    # return dataframe for possible further use
    return df_out


if __name__ == "__main__":
    predict_new_csv()
