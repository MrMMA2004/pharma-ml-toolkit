# train_and_save_stacking.py
import pandas as pd
import joblib
import json
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

warnings.filterwarnings("ignore")
sns.set()

# -----------------------------
# Data Loading & Preparation
# -----------------------------
def load_data(file_path, target_col='Activity'):
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(f"ستون '{target_col}' یافت نشد.")
    return df

def prepare_X_y(df, target_col='Activity', smiles_col='SMILES', drop_cols=None):
    drop_list = [target_col] + (drop_cols or [])
    if smiles_col in df.columns and smiles_col not in (drop_cols or []):
        drop_list.append(smiles_col)
    X = df.drop(columns=drop_list)
    y = df[target_col]
    return X, y

def split_train_test_with_df(df, X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    df_test = df.loc[X_test.index].copy().reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    return X_train, X_test, y_train, y_test, df_test

# -----------------------------
# Stacking Regressor
# -----------------------------
def build_stacking_model():
    base_models = [
        ('ridge', Ridge(alpha=100.0)),
        ('knn', KNeighborsRegressor(n_neighbors=3)),
        ('svr', SVR(kernel='rbf', C=100, epsilon=0.1)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=30, random_state=42))
    ]
    meta_model = Ridge(alpha=100.0)

    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        passthrough=True
    )

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('stack', stack)
    ])
    return pipeline

# -----------------------------
# Evaluation & Saving
# -----------------------------
def evaluate_and_save(model, X_test, y_test, df_test, prefix='stacking', save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R2 (test): {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")

    df_res = X_test.copy().reset_index(drop=True)
    df_res['Actual'] = y_test.reset_index(drop=True)
    df_res['Predicted'] = pd.Series(y_pred)
    df_res['Residual'] = df_res['Actual'] - df_res['Predicted']
    if 'SMILES' in df_test.columns:
        df_res['SMILES'] = df_test['SMILES'].values

    cols = []
    if 'SMILES' in df_res.columns:
        cols.append('SMILES')
    cols += ['Actual', 'Predicted', 'Residual']
    cols = [c for c in cols if c in df_res.columns]
    df_res = df_res[cols + [c for c in df_res.columns if c not in cols]]

    out_xlsx = os.path.join(save_dir, f'{prefix}_results.xlsx')
    df_res.to_excel(out_xlsx, index=False)
    print(f"✅ نتایج نمونه‌ای در '{out_xlsx}' ذخیره شد.")

    # Plots
    images = []

    # Parity plot
    plt.figure(figsize=(6, 6))
    plt.scatter(df_res['Actual'], df_res['Predicted'], alpha=0.6)
    minv = min(df_res['Actual'].min(), df_res['Predicted'].min())
    maxv = max(df_res['Actual'].max(), df_res['Predicted'].max())
    plt.plot([minv, maxv], [minv, maxv], '--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Parity plot')
    parity_file = os.path.join(save_dir, f'{prefix}_parity.jpg')
    plt.savefig(parity_file, format='jpg', dpi=300)
    plt.close()
    images.append(parity_file)

    # Residual histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals distribution')
    res_hist = os.path.join(save_dir, f'{prefix}_residuals_hist.jpg')
    plt.savefig(res_hist, format='jpg', dpi=300)
    plt.close()
    images.append(res_hist)

    # PDF report
    pdf_file = os.path.join(save_dir, f'{prefix}_report.pdf')
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Stacking Regression Report', ln=True, align='C')
    pdf.ln(5)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, f'R2: {r2:.6f}')
    pdf.multi_cell(0, 8, f'MSE: {mse:.6f}')
    pdf.multi_cell(0, 8, f'RMSE: {rmse:.6f}')
    pdf.multi_cell(0, 8, f'MAE: {mae:.6f}')
    pdf.ln(4)
    for img in images:
        pdf.add_page()
        pdf.image(img, x=15, y=30, w=180)
    pdf.output(pdf_file)
    print(f"📄 گزارش PDF در '{pdf_file}' ذخیره شد.")

    return {
        'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae,
        'results_xlsx': out_xlsx, 'images': images, 'pdf': pdf_file
    }

# -----------------------------
# Bundle Saving
# -----------------------------
def _to_python_native(x):
    import numpy as _np
    import pandas as _pd
    if isinstance(x, (list, tuple)):
        return [_to_python_native(v) for v in x]
    if isinstance(x, _pd.Index):
        return [ _to_python_native(v) for v in list(x) ]
    if isinstance(x, _np.ndarray):
        return [_to_python_native(v) for v in x.tolist()]
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.bool_ , bool)):
        return bool(x)
    try:
        if hasattr(x, 'item'):
            py = x.item()
            if isinstance(py, (int, float, bool, str)):
                return py
    except Exception:
        pass
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    return str(x)

def save_bundle(model_pipeline, X_columns, target_col='Activity', smiles_col='SMILES',
                bundle_prefix='stacking', save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)
    pipeline_path = os.path.join(save_dir, f'{bundle_prefix}_bundle_pipeline.joblib')
    metadata_path = os.path.join(save_dir, f'{bundle_prefix}_bundle_metadata.json')

    joblib.dump(model_pipeline, pipeline_path)
    metadata = {
        'feature_columns': _to_python_native(list(X_columns)),
        'target_col': _to_python_native(target_col),
        'smiles_col': _to_python_native(smiles_col),
        'bundle_version': '1.0',
        'notes': 'Pipeline = SimpleImputer(median) -> StandardScaler -> StackingRegressor'
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("✅ Bundle saved:")
    print(f" - {pipeline_path}")
    print(f" - {metadata_path}")
    return pipeline_path, metadata_path

# -----------------------------
# Main
# -----------------------------
def main():
    input_csv = '1000selected_features_reg.csv'
    target_col = 'Activity'
    smiles_col = 'SMILES'
    bundle_prefix = 'stacking'
    save_dir = 'outputs'

    print("Loading data...")
    df = load_data(input_csv, target_col=target_col)

    print("Preparing X and y...")
    X, y = prepare_X_y(df, target_col=target_col, smiles_col=smiles_col)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test, df_test = split_train_test_with_df(df, X, y)

    print("Building stacking model...")
    model_pipeline = build_stacking_model()
    model_pipeline.fit(X_train, y_train)
    print("Model training completed.")

    print("Evaluating on test set and saving reports...")
    eval_info = evaluate_and_save(model_pipeline, X_test, y_test, df_test,
                                  prefix=bundle_prefix, save_dir=save_dir)

    pipeline_path, metadata_path = save_bundle(model_pipeline, X.columns,
                                              target_col=target_col, smiles_col=smiles_col,
                                              bundle_prefix=bundle_prefix, save_dir=save_dir)

    print("All done.")

if __name__ == '__main__':
    main()
