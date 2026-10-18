# predict_from_bundle.py
import pandas as pd
import joblib
import json
import numpy as np
import os

# --------------------------
# 🔧 تنظیم مسیرهای فایل‌ها
# --------------------------
NEW_CSV_PATH = "newfile_selected_features.csv"       # مسیر فایل CSV جدید
BUNDLE_PREFIX = "logreg_bundle"     # بدون پسوند (فقط prefix)
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
    if 'classes' not in metadata:
        raise KeyError("metadata must contain 'classes' key.")

    return pipeline, metadata


def _safe_predict_proba(pipeline, X):
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(X))
    elif hasattr(pipeline, "decision_function"):
        dec = np.asarray(pipeline.decision_function(X))
        if dec.ndim == 1:
            probs_pos = 1 / (1 + np.exp(-dec))
            return np.vstack([1 - probs_pos, probs_pos]).T
        expd = np.exp(dec - np.max(dec, axis=1, keepdims=True))
        return expd / np.sum(expd, axis=1, keepdims=True)
    else:
        raise RuntimeError("Model has neither predict_proba nor decision_function.")


def predict_new_csv(new_csv_path=NEW_CSV_PATH, bundle_prefix=BUNDLE_PREFIX, output_prefix=OUTPUT_PREFIX):
    pipeline, metadata = load_bundle(bundle_prefix)
    print(f"✅ Loaded bundle from {bundle_prefix}")

    df_new = pd.read_csv(new_csv_path)
    print(f"✅ Loaded new CSV: {new_csv_path} | shape = {df_new.shape}")

    feature_cols = metadata['feature_columns']
    smiles_col = metadata.get('smiles_col', 'SMILES')
    classes = metadata['classes']

    missing = [c for c in feature_cols if c not in df_new.columns]
    if missing:
        raise ValueError(f"❌ Missing required feature columns: {missing}")

    X_new = df_new[feature_cols].copy()
    X_new = X_new.apply(pd.to_numeric, errors='coerce')
    if X_new.isna().any().any():
        raise ValueError("❌ Some features contain NaN or non-numeric values.")

    print("🔹 Running predictions...")
    y_pred = pipeline.predict(X_new)
    y_proba = _safe_predict_proba(pipeline, X_new)

    df_out = pd.DataFrame()
    if smiles_col in df_new.columns:
        df_out['SMILES'] = df_new[smiles_col]
    df_out['Predicted'] = y_pred

    for idx, cls in enumerate(classes):
        df_out[f'Prob_Class_{cls}'] = y_proba[:, idx]

    out_csv = f'{output_prefix}_results.csv'
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Saved results to:\n  {out_csv}")

    return df_out


if __name__ == "__main__":
    predict_new_csv()
