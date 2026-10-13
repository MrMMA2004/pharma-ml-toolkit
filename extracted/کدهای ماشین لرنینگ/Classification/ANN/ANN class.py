# train_and_save_ann.py
import pandas as pd
import joblib
import json
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)

warnings.filterwarnings("ignore")
sns.set()  # nicer plots


# ---------------------- Data Loading ----------------------
def load_data(file_path, target_col='Target'):
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(f"ستون '{target_col}' یافت نشد.")
    return df


def prepare_X_y(df, target_col='Target', smiles_col='SMILES', drop_cols=None):
    drop_list = [target_col] + (drop_cols or [])
    if smiles_col in df.columns and smiles_col not in (drop_cols or []):
        drop_list.append(smiles_col)
    X = df.drop(columns=drop_list)
    y = df[target_col]
    return X, y


def split_train_test_with_df(df, X, y, test_size=0.2, random_state=42, stratify=True):
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    df_test = df.loc[X_test.index].copy().reset_index(drop=True)
    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test, df_test


# ---------------------- Model Tuning ----------------------
def tune_model(X_train, y_train, param_grid, cv=5, n_jobs=-1, scoring='roc_auc_ovr_weighted'):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(max_iter=2000, random_state=42))
    ])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    mapped_grid = {f'clf__{k}' if not k.startswith('clf__') else k: v for k, v in param_grid.items()}

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=mapped_grid,
        cv=skf,
        n_jobs=n_jobs,
        verbose=1,
        scoring=scoring
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ---------------------- Evaluation & Saving ----------------------
def evaluate_and_save(model, X_test, y_test, df_test, classes,
                      prefix='ann_bundle', save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)
    classes = list(classes)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy (test): {acc:.4f}")
    print(report)

    df_res = X_test.copy().reset_index(drop=True)
    df_res['Actual'] = y_test.reset_index(drop=True)
    df_res['Predicted'] = pd.Series(y_pred)
    if 'SMILES' in df_test.columns:
        df_res['SMILES'] = df_test['SMILES'].values

    for idx, cls in enumerate(classes):
        df_res[f'Prob_Class_{cls}'] = y_proba[:, idx]

    cols = ['SMILES'] if 'SMILES' in df_res.columns else []
    cols += ['Actual', 'Predicted'] + [c for c in df_res.columns if c.startswith('Prob_Class_')]
    df_res = df_res[cols]

    out_xlsx = os.path.join(save_dir, f'{prefix}_results.xlsx')
    df_res.to_excel(out_xlsx, index=False)
    print(f"✅ نتایج نمونه‌ای در '{out_xlsx}' ذخیره شد.")

    images = []

    # Confusion Matrix -> JPG
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_file = os.path.join(save_dir, f'{prefix}_confusion.jpg')
    plt.savefig(cm_file, format='jpg', dpi=300)
    plt.close()
    images.append(cm_file)

    # ROC curves per class -> JPG
    y_bin = label_binarize(y_test, classes=classes)
    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC {cls}={roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], '--')
        plt.title(f'ROC for {cls}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        roc_file = os.path.join(save_dir, f'{prefix}_roc_{cls}.jpg')
        plt.savefig(roc_file, format='jpg', dpi=300)
        plt.close()
        images.append(roc_file)

    print("📊 نمودارها ذخیره شدند (JPG).")

    # PDF report
    pdf_file = os.path.join(save_dir, f'{prefix}_report.pdf')
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Classification Report', ln=True, align='C')
    pdf.ln(5)
    pdf.set_font('Arial', '', 12)
    for line in report.split('\n'):
        pdf.multi_cell(0, 8, line)
    for img in images:
        if img and os.path.exists(img):
            pdf.add_page()
            pdf.image(img, x=15, y=25, w=180)
    pdf.output(pdf_file)
    print(f"📄 گزارش PDF در '{pdf_file}' ذخیره شد.")

    return {'accuracy': float(acc), 'report_text': report, 'results_xlsx': out_xlsx,
            'images': images, 'pdf': pdf_file}


# ---------------------- Bundle Saving ----------------------
def _to_python_native(x):
    import numpy as _np
    import pandas as _pd
    if isinstance(x, (list, tuple)):
        return [_to_python_native(v) for v in x]
    if isinstance(x, _pd.Index):
        return [_to_python_native(v) for v in list(x)]
    if isinstance(x, _np.ndarray):
        return [_to_python_native(v) for v in x.tolist()]
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.bool_, bool)):
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


def save_bundle(model_pipeline, X_columns, classes, target_col='Target', smiles_col='SMILES',
                best_params=None, best_score=None, bundle_prefix='ann_bundle', save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)
    pipeline_path = os.path.join(save_dir, f'{bundle_prefix}_pipeline.joblib')
    metadata_path = os.path.join(save_dir, f'{bundle_prefix}_metadata.json')

    joblib.dump(model_pipeline, pipeline_path)

    metadata = {
        'feature_columns': _to_python_native(list(X_columns)),
        'classes': _to_python_native(list(classes)),
        'target_col': _to_python_native(target_col),
        'smiles_col': _to_python_native(smiles_col),
        'best_params': _to_python_native(best_params),
        'best_cv_score': _to_python_native(best_score),
        'bundle_version': '1.0',
        'notes': 'Pipeline = SimpleImputer(median) -> StandardScaler -> MLPClassifier'
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("✅ Bundle saved:")
    print(f" - {pipeline_path}")
    print(f" - {metadata_path}")
    return pipeline_path, metadata_path


# ---------------------- Main ----------------------
def main():
    input_csv = '1000selected_features_class.csv'
    target_col = 'Target'
    smiles_col = 'SMILES'
    bundle_prefix = 'ann_bundle'
    save_dir = 'outputs'

    print("Loading data...")
    df = load_data(input_csv, target_col=target_col)

    print("Preparing X and y...")
    X, y = prepare_X_y(df, target_col=target_col, smiles_col=smiles_col)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test, df_test = split_train_test_with_df(
        df, X, y, test_size=0.2, random_state=42, stratify=True
    )

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50,50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }

    print("Tuning model (GridSearchCV)...")
    model_pipeline, best_params, best_score = tune_model(X_train, y_train, param_grid, cv=5, n_jobs=-1,
                                                         scoring='roc_auc_ovr_weighted')
    print("Tuning finished.")
    print("Best params:", best_params)
    print("Best CV score:", best_score)

    print("Evaluating on test set and saving reports...")
    eval_info = evaluate_and_save(model_pipeline, X_test, y_test, df_test,
                                  classes=sorted(y.unique()),
                                  prefix=bundle_prefix, save_dir=save_dir)

    pipeline_path, metadata_path = save_bundle(model_pipeline, X.columns, sorted(y.unique()),
                                              target_col=target_col, smiles_col=smiles_col,
                                              best_params=best_params, best_score=best_score,
                                              bundle_prefix=bundle_prefix, save_dir=save_dir)

    print("All done.")


if __name__ == '__main__':
    main()
