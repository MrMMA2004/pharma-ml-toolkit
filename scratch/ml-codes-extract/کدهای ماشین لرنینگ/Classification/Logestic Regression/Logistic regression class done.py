# train_and_save_combined.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)

warnings.filterwarnings("ignore")
sns.set()  # for nicer plots


def load_data(file_path, target_col='Target'):
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(f"ستون '{target_col}' یافت نشد.")
    return df


def prepare_X_y(df, target_col='Target', smiles_col='SMILES', drop_cols=None):
    """
    X: فقط فیچرهای عددی/عددی-قابل‌استفاده (SMILES از X حذف می‌شود اما در df نگه‌داشته می‌شود)
    y: ستون هدف
    """
    drop_list = [target_col] + (drop_cols or [])
    if smiles_col in df.columns and smiles_col not in (drop_cols or []):
        drop_list.append(smiles_col)
    X = df.drop(columns=drop_list)
    y = df[target_col]
    return X, y


def split_train_test_with_df(df, X, y, test_size=0.2, random_state=42, stratify=True):
    """
    تقسیم داده‌ها؛ همچنین df_test (سطرهای متناظر با X_test) را برمی‌گرداند
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    # نگه داشتن ردیف‌های متناظر برای SMILES و هر ستون غیرفیچری دیگر
    df_test = df.loc[X_test.index].copy().reset_index(drop=True)
    # Reset index for X_test so later concat راحت باشد
    X_test = X_test.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    return X_train, X_test, y_train, y_test, df_test


def tune_model(X_train, y_train, param_grid, cv=5, n_jobs=-1, scoring='roc_auc_ovr_weighted'):
    """
    برمی‌گرداند: best_estimator_ (pipeline)، best_params و best_score
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, random_state=42))
    ])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # map param grid keys to clf__ if لازم
    mapped_grid = {}
    for k, v in param_grid.items():
        if k.startswith('clf__'):
            mapped_grid[k] = v
        else:
            mapped_grid[f'clf__{k}'] = v

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


def evaluate_and_save(model, X_test, y_test, df_test, classes,
                      prefix='logreg_bundle', save_dir='outputs'):
    """
    ارزیابی مدل روی X_test؛ ذخیره نتایج نمونه‌ای (شامل SMILES)، تصاویر (JPG) و PDF گزارش
    """
    os.makedirs(save_dir, exist_ok=True)
    classes = list(classes)

    # پیش‌بینی و احتمالات
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X_test)
        if dec.ndim == 1:
            probs_pos = 1 / (1 + np.exp(-dec))
            y_proba = np.vstack([1 - probs_pos, probs_pos]).T
        else:
            expd = np.exp(dec - np.max(dec, axis=1, keepdims=True))
            y_proba = expd / np.sum(expd, axis=1, keepdims=True)
    else:
        raise RuntimeError("Model has neither predict_proba nor decision_function.")

    # متریک‌ها
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy (test): {acc:.4f}")
    print(report)

    # DataFrame خروجی: شامل SMILES (اگر موجود)، Actual، Predicted، Prob_Class...
    df_res = X_test.copy().reset_index(drop=True)
    df_res['Actual'] = y_test.reset_index(drop=True)
    df_res['Predicted'] = pd.Series(y_pred)
    if 'SMILES' in df_test.columns:
        df_res['SMILES'] = df_test['SMILES'].values

    # مطابقت ابعاد احتمال‌ها و ترتیب کلاس‌ها؛ اگر mismatch بود سعی کن از model.classes_ استفاده کنی
    if y_proba.shape[1] != len(classes):
        if hasattr(model, "classes_"):
            model_classes = list(model.classes_)
            if len(model_classes) == y_proba.shape[1]:
                classes = model_classes
            else:
                raise RuntimeError("Mismatch between provided classes and model output shape.")
        else:
            raise RuntimeError("Mismatch between provided classes and model output shape.")

    for idx, cls in enumerate(classes):
        df_res[f'Prob_Class_{cls}'] = y_proba[:, idx]

    # ترتیب ستون‌ها: SMILES (اگر باشد) -> Actual -> Predicted -> Prob_*
    cols = []
    if 'SMILES' in df_res.columns:
        cols.append('SMILES')
    cols += ['Actual', 'Predicted'] + [c for c in df_res.columns if c.startswith('Prob_Class_')]
    cols = [c for c in cols if c in df_res.columns]
    df_res = df_res[cols]

    out_xlsx = os.path.join(save_dir, f'{prefix}_results.xlsx')
    df_res.to_excel(out_xlsx, index=False)
    print(f"✅ نتایج نمونه‌ای در '{out_xlsx}' ذخیره شد.")

    images = []

    # Confusion Matrix -> JPG
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        cm_file = os.path.join(save_dir, f'{prefix}_confusion.jpg')
        plt.savefig(cm_file, format='jpg', dpi=300)
        plt.close()
        images.append(cm_file)
    except Exception as e:
        print("Warning: failed to draw/save confusion matrix:", e)

    # ROC curves per class -> JPG
    try:
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
    except Exception as e:
        print("Warning: failed to draw/save ROC curves:", e)

    print("📊 نمودارها ذخیره شدند (JPG).")

    # PDF report (متن گزارش + تصاویر JPG)
    try:
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
    except Exception as e:
        print("Warning: failed to create PDF report:", e)

    return {
        'accuracy': float(acc),
        'report_text': report,
        'results_xlsx': out_xlsx,
        'images': images,
        'pdf': pdf_file if 'pdf_file' in locals() else None
    }


def _to_python_native(x):
    """
    تبدیل مقادیر numpy/pandas به انواع پایهٔ پایتون
    - اگر لیست/tuple -> بازگشتی از تبدیل عناصر
    - اگر numpy scalar -> تبدیل به int/float/bool
    - اگر pandas/Numpy index -> تبدیل به لیست از str
    - در غیر اینصورت str fallback
    """
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
    # pandas nullable integers may be pandas._libs.maybe... fallback:
    try:
        if hasattr(x, 'item'):
            py = x.item()
            if isinstance(py, (int, float, bool, str)):
                return py
    except Exception:
        pass
    # اگر رشته یا عدد پایتون است، برگردان
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    # fallback: رشته‌ای کن
    return str(x)


def save_bundle(model_pipeline, X_columns, classes, target_col='Target', smiles_col='SMILES',
                best_params=None, best_score=None, bundle_prefix='logreg_bundle', save_dir='outputs'):
    """
    ذخیره pipeline (.joblib) و metadata (.json) — این نسخه مقادیر را به انواع پایه پایتون تبدیل می‌کند
    """
    os.makedirs(save_dir, exist_ok=True)
    pipeline_path = os.path.join(save_dir, f'{bundle_prefix}_pipeline.joblib')
    metadata_path = os.path.join(save_dir, f'{bundle_prefix}_metadata.json')

    joblib.dump(model_pipeline, pipeline_path)

    # تبدیل امن به انواع پایهٔ پایتون
    metadata = {
        'feature_columns': _to_python_native(list(X_columns)),
        'classes': _to_python_native(list(classes)),
        'target_col': _to_python_native(target_col),
        'smiles_col': _to_python_native(smiles_col),
        'best_params': _to_python_native(best_params),
        'best_cv_score': _to_python_native(best_score),
        'bundle_version': '1.0',
        'notes': 'Pipeline = SimpleImputer(median) -> StandardScaler -> LogisticRegression'
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("✅ Bundle saved:")
    print(f" - {pipeline_path}")
    print(f" - {metadata_path}")
    return pipeline_path, metadata_path


def main():
    input_csv = '1000selected_features_class.csv'  # نام فایل ورودی را اینجا بگذار
    target_col = 'Target'
    smiles_col = 'SMILES'
    bundle_prefix = 'logreg_bundle'
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
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs'],
        'class_weight': [None, 'balanced']
    }

    print("Tuning model (GridSearchCV)...")
    model_pipeline, best_params, best_score = tune_model(X_train, y_train, param_grid, cv=5, n_jobs=-1,
                                                         scoring='roc_auc_ovr_weighted')
    print("Tuning finished.")
    print("Best params:", best_params)
    print("Best CV score:", best_score)

    # ارزیابی روی تست و ذخیره تصاویر/گزارش
    print("Evaluating on test set and saving reports...")
    eval_info = evaluate_and_save(model_pipeline, X_test, y_test, df_test,
                                  classes=sorted(y.unique()),
                                  prefix=bundle_prefix, save_dir=save_dir)

    # ذخیره bundle (pipeline + metadata)
    pipeline_path, metadata_path = save_bundle(model_pipeline, X.columns, sorted(y.unique()),
                                              target_col=target_col, smiles_col=smiles_col,
                                              best_params=best_params, best_score=best_score,
                                              bundle_prefix=bundle_prefix, save_dir=save_dir)

    print("All done.")


if __name__ == '__main__':
    main()
