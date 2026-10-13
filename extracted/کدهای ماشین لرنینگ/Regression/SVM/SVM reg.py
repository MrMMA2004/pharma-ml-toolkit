# train_and_save_combined_svr.py
import pandas as pd
import joblib
import json
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)

warnings.filterwarnings("ignore")
sns.set()


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


def split_train_test_with_df(df, X, y, test_size=0.2, random_state=42, stratify=False):
    # برای رگرسیون stratify معمول نیست؛ بنابراین پارام stratify=False استفاده می‌شود
    strat = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    df_test = df.loc[X_test.index].copy().reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    return X_train, X_test, y_train, y_test, df_test


def tune_model(X_train, y_train, param_grid, cv=5, n_jobs=-1, scoring='r2'):
    """
    بازگرداند: best_estimator_ (pipeline)، best_params و best_score
    برای SVR از KFold استفاده می‌کنیم
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('reg', SVR())
    ])

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # map param grid keys to reg__ اگر لازم
    mapped_grid = {}
    for k, v in param_grid.items():
        if k.startswith('reg__'):
            mapped_grid[k] = v
        else:
            mapped_grid[f'reg__{k}'] = v

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=mapped_grid,
        cv=kf,
        n_jobs=n_jobs,
        verbose=1,
        scoring=scoring
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def evaluate_and_save(model, X_test, y_test, df_test,
                      prefix='svr_bundle', save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)

    # پیش‌بینی
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    # متریک‌ها
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R2 (test): {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")

    # DataFrame خروجی: شامل SMILES (اگر موجود)، Actual، Predicted، Residual
    df_res = X_test.copy().reset_index(drop=True)
    df_res['Actual'] = y_test.reset_index(drop=True)
    df_res['Predicted'] = pd.Series(y_pred)
    df_res['Residual'] = df_res['Actual'] - df_res['Predicted']
    if 'SMILES' in df_test.columns:
        df_res['SMILES'] = df_test['SMILES'].values

    # ترتیب ستون‌ها: SMILES (اگر باشد) -> Actual -> Predicted -> Residual
    cols = []
    if 'SMILES' in df_res.columns:
        cols.append('SMILES')
    cols += ['Actual', 'Predicted', 'Residual']
    cols = [c for c in cols if c in df_res.columns]
    df_res = df_res[cols + [c for c in df_res.columns if c not in cols]]

    out_xlsx = os.path.join(save_dir, f'{prefix}_results.xlsx')
    df_res.to_excel(out_xlsx, index=False)
    print(f"✅ نتایج نمونه‌ای در '{out_xlsx}' ذخیره شد.")

    images = []

    # Parity plot (Predicted vs Actual)
    try:
        plt.figure(figsize=(6, 6))
        plt.scatter(df_res['Actual'], df_res['Predicted'], alpha=0.6)
        minv = min(df_res['Actual'].min(), df_res['Predicted'].min())
        maxv = max(df_res['Actual'].max(), df_res['Predicted'].max())
        plt.plot([minv, maxv], [minv, maxv], '--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Parity plot')
        plt.tight_layout()
        parity_file = os.path.join(save_dir, f'{prefix}_parity.jpg')
        plt.savefig(parity_file, format='jpg', dpi=300)
        plt.close()
        images.append(parity_file)
    except Exception as e:
        print("Warning: failed to draw/save parity plot:", e)

    # Residuals histogram
    try:
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True)
        plt.title('Residuals distribution')
        plt.xlabel('Residual')
        plt.tight_layout()
        res_hist = os.path.join(save_dir, f'{prefix}_residuals_hist.jpg')
        plt.savefig(res_hist, format='jpg', dpi=300)
        plt.close()
        images.append(res_hist)
    except Exception as e:
        print("Warning: failed to draw/save residuals histogram:", e)

    print("📊 نمودارها ذخیره شدند (JPG).")

    # PDF report (متن گزارش + تصاویر JPG)
    try:
        pdf_file = os.path.join(save_dir, f'{prefix}_report.pdf')
        pdf = FPDF()
        pdf.set_auto_page_break(True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Regression Report (SVR)', ln=True, align='C')
        pdf.ln(5)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 8, f'R2: {r2:.6f}')
        pdf.multi_cell(0, 8, f'MSE: {mse:.6f}')
        pdf.multi_cell(0, 8, f'RMSE: {rmse:.6f}')
        pdf.multi_cell(0, 8, f'MAE: {mae:.6f}')
        pdf.ln(4)
        for img in images:
            if img and os.path.exists(img):
                pdf.add_page()
                pdf.image(img, x=15, y=30, w=180)
        pdf.output(pdf_file)
        print(f"📄 گزارش PDF در '{pdf_file}' ذخیره شد.")
    except Exception as e:
        print("Warning: failed to create PDF report:", e)

    return {
        'r2': float(r2),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'results_xlsx': out_xlsx,
        'images': images,
        'pdf': pdf_file if 'pdf_file' in locals() else None
    }


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
                best_params=None, best_score=None, bundle_prefix='svr_bundle', save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)
    pipeline_path = os.path.join(save_dir, f'{bundle_prefix}_pipeline.joblib')
    metadata_path = os.path.join(save_dir, f'{bundle_prefix}_metadata.json')

    joblib.dump(model_pipeline, pipeline_path)

    metadata = {
        'feature_columns': _to_python_native(list(X_columns)),
        'target_col': _to_python_native(target_col),
        'smiles_col': _to_python_native(smiles_col),
        'best_params': _to_python_native(best_params),
        'best_cv_score': _to_python_native(best_score),
        'bundle_version': '1.0',
        'notes': 'Pipeline = SimpleImputer(median) -> StandardScaler -> SVR'
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("✅ Bundle saved:")
    print(f" - {pipeline_path}")
    print(f" - {metadata_path}")
    return pipeline_path, metadata_path

# ==========================================================
# TOP-10 FEATURE CORRELATION HEATMAP (+ pIC50 / Activity)
# ==========================================================

def plot_top10_feature_heatmap(df, target_col='Activity', save_dir='outputs', prefix='svr_bundle'):
    """
    Selects top 10 correlation-based features,
    plots heatmap,
    RETURNS the list of top-10 features.
    """

    numeric_df = df.select_dtypes(include=[np.number])

    # Correlation ranking
    corrs = numeric_df.corr()[target_col].abs().sort_values(ascending=False)

    top10 = corrs.drop(target_col).head(10).index.tolist()

    df_plot = numeric_df[top10 + [target_col]]

    print("Top 10 correlation-selected features:")
    for f in top10:
        print(" -", f)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df_plot.corr(),
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title(f"Correlation Heatmap: Top 10 Features + {target_col}", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{prefix}_top10_heatmap.jpg")
    plt.savefig(out_path, dpi=300, format='jpg')
    plt.close()

    print(f"Top-10 feature heatmap saved to: {out_path}")

    return top10

from sklearn.inspection import permutation_importance

def permutation_for_corr_top10_separately(top10_corr, X_train, X_test, y_train, y_test, output_folder):
    """
    Fits a NEW pipeline using ONLY the correlation-top10 features,
    then computes permutation importance safely.
    """

    print("\n=== Running permutation importance for CORRELATION TOP-10 (separate model) ===")

    # Subset data
    X_train_top10 = X_train[top10_corr].copy()
    X_test_top10 = X_test[top10_corr].copy()

    # New pipeline (same hyperparameters)
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.inspection import permutation_importance

    model_top10 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", SVR(C=10, epsilon=0.1, gamma="auto", kernel="rbf")),  
    ])

    # Fit on TOP 10 features only
    model_top10.fit(X_train_top10, y_train)

    # Permutation importance
    result = permutation_importance(
        model_top10,
        X_test_top10,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="r2"
    )

    importances_mean = result.importances_mean
    importances_std = result.importances_std

    perm_df = pd.DataFrame({
        "feature": top10_corr,
        "importance_mean": importances_mean,
        "importance_std": importances_std
    }).sort_values("importance_mean", ascending=False)

    save_path = os.path.join(output_folder, "svr_permutation_top10_corr.xlsx")
    perm_df.to_excel(save_path, index=False)

    print("Permutation importance (top10-corr) saved to:", save_path)

    # Plot
    plt.figure(figsize=(8, 6))
    perm_df.plot(
        x="feature", 
        y="importance_mean", 
        kind="bar", 
        yerr=perm_df["importance_std"],
        figsize=(10,5)
    )
    plt.title("Permutation Importance for Top-10 Features")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig_path = os.path.join(output_folder, "svr_permutation_top10_corr.jpg")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("Permutation importance plot saved:", fig_path)
    
    
def evaluate_train_test_and_plots(
    model,
    X_train, y_train,
    X_test, y_test,
    target_name="Activity",
    model_name="SVR (RBF)",
    prefix="svr_bundle",
    save_dir="outputs"
):
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = {
        "train": {
            "R2": r2_score(y_train, y_train_pred),
            "MSE": mean_squared_error(y_train, y_train_pred),
            "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "MAE": mean_absolute_error(y_train, y_train_pred),
        },
        "test": {
            "R2": r2_score(y_test, y_test_pred),
            "MSE": mean_squared_error(y_test, y_test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "MAE": mean_absolute_error(y_test, y_test_pred),
        }
    }

    # Console report
    print("\n================ MODEL PERFORMANCE REPORT ================")
    for split in ["train", "test"]:
        print(f"\n{split.upper()} SET:")
        for k, v in metrics[split].items():
            print(f"  {k}: {v:.6f}")

    # Save metrics to JSON
    metrics_path = os.path.join(save_dir, f"{prefix}_train_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to: {metrics_path}")

    # =========================================================
    # 1. PARITY PLOT (TRAIN + TEST)
    # =========================================================
    plt.figure(figsize=(7, 7))

    plt.scatter(
        y_train, y_train_pred,
        alpha=0.6, label="Train", edgecolor="k"
    )
    plt.scatter(
        y_test, y_test_pred,
        alpha=0.8, label="Test", edgecolor="k"
    )

    min_val = min(y_train.min(), y_test.min(), y_train_pred.min(), y_test_pred.min())
    max_val = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)

    plt.xlabel(f"Observed {target_name}", fontsize=12)
    plt.ylabel(f"Predicted {target_name}", fontsize=12)

    plt.title(
        f"Parity Plot – {model_name}\n"
        f"Train $R^2$ = {metrics['train']['R2']:.3f} | "
        f"Test $R^2$ = {metrics['test']['R2']:.3f}",
        fontsize=13
    )

    plt.legend()
    plt.tight_layout()

    parity_path = os.path.join(save_dir, f"{prefix}_parity_train_test.jpg")
    plt.savefig(parity_path, dpi=300)
    plt.close()

    # =========================================================
    # 2. RESIDUAL DISTRIBUTION (TRAIN vs TEST)
    # =========================================================
    plt.figure(figsize=(8, 5))

    sns.kdeplot(
        y_train - y_train_pred,
        label="Train residuals",
        fill=True,
        alpha=0.4
    )
    sns.kdeplot(
        y_test - y_test_pred,
        label="Test residuals",
        fill=True,
        alpha=0.4
    )

    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel(f"Residual ({target_name} observed − predicted)", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    plt.title(
        f"Residual Distribution – {model_name}\n"
        f"Test RMSE = {metrics['test']['RMSE']:.3f}",
        fontsize=13
    )

    plt.legend()
    plt.tight_layout()

    resid_path = os.path.join(save_dir, f"{prefix}_residuals_train_test.jpg")
    plt.savefig(resid_path, dpi=300)
    plt.close()

    print("📊 Labeled figures saved:")
    print(" -", parity_path)
    print(" -", resid_path)

    return {
        "metrics": metrics,
        "parity_plot": parity_path,
        "residual_plot": resid_path
    }


def main():
    input_csv = '1000selected_features_reg.csv'  # نام فایل ورودی را اینجا بگذار
    target_col = 'Activity'
    smiles_col = 'SMILES'
    bundle_prefix = 'svr_bundle'
    save_dir = 'outputs'

    print("Loading data...")
    df = load_data(input_csv, target_col=target_col)

    print("Preparing X and y...")
    X, y = prepare_X_y(df, target_col=target_col, smiles_col=smiles_col)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test, df_test = split_train_test_with_df(
        df, X, y, test_size=0.2, random_state=42, stratify=False
    )

    param_grid = {
        'kernel': ['rbf'],
        'C': [1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5],
        'gamma': ['scale', 'auto']  # می‌توانید اضافه کنید اگر دیتاست بزرگ باشد
    }

    print("Tuning model (GridSearchCV)...")
    model_pipeline, best_params, best_score = tune_model(X_train, y_train, param_grid, cv=5, n_jobs=-1,
                                                         scoring='r2')
    print("Tuning finished.")
    print("Best params:", best_params)
    print("Best CV score:", best_score)

    print("Evaluating on test set and saving reports...")
    eval_info = evaluate_and_save(model_pipeline, X_test, y_test, df_test,
                                  prefix=bundle_prefix, save_dir=save_dir)

    pipeline_path, metadata_path = save_bundle(model_pipeline, X.columns,
                                              target_col=target_col, smiles_col=smiles_col,
                                              best_params=best_params, best_score=best_score,
                                              bundle_prefix=bundle_prefix, save_dir=save_dir)

    # 1. Select top 10 correlation features and plot heatmap
    top10_corr = plot_top10_feature_heatmap(
        df,
        target_col=target_col,
        save_dir=save_dir,
        prefix=bundle_prefix
    )

    # 2. Run permutation importance ONLY for those 10
    permutation_for_corr_top10_separately(
        top10_corr=top10_corr,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_folder=save_dir
    )
    
    print("\nRunning full TRAIN vs TEST evaluation with labeled plots...")
    train_test_report = evaluate_train_test_and_plots(
        model=model_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        target_name=target_col,
        model_name="SVR (RBF kernel)",
        prefix=bundle_prefix,
        save_dir=save_dir
    )




if __name__ == '__main__':
    main()
