# train_and_save_combined_stacking.py
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)

warnings.filterwarnings("ignore")
sns.set()


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
    X_test = X_test.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    return X_train, X_test, y_train, y_test, df_test


def build_stacking_pipeline(random_state=42):
    """
    ساخت یک pipeline ساده:
    preprocessing = SimpleImputer(median) -> StandardScaler
    clf = StackingClassifier( estimators=[('lr',LogReg),('rf',RF),('gb',GB)], final_estimator=LogReg )
    """
    preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # پایه‌ها (base estimators)
    estimators = [
        ('lr_base', LogisticRegression(max_iter=2000, random_state=random_state)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=random_state))
    ]

    final_estimator = LogisticRegression(max_iter=2000, random_state=random_state)

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=False,  # می‌تونید True کنید تا ویژگی‌های اصلی هم به متا داده شود
        n_jobs=-1,
        cv=5
    )

    pipeline = Pipeline([
        ('preproc', preproc),
        ('clf', stacking)
    ])
    return pipeline


def tune_model(X_train, y_train, param_grid, cv=5, n_jobs=-1, scoring='roc_auc_ovr_weighted', random_state=42):
    """
    این تابع pipeline stacking را می‌سازد و GridSearchCV را اجرا می‌کند.
    نگاشت ساده پارامترها:
      - اگر کلید 'final_C' در param_grid باشد -> mapped to 'clf__final_estimator__C'
      - اگر کلید از قبل با 'clf__' شروع شده، نگه داشته می‌شود
      - در غیر اینصورت پیشوند 'clf__' اضافه می‌شود
    بازگشت: best_estimator_, best_params_, best_score_
    """
    pipeline = build_stacking_pipeline(random_state=random_state)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    mapped_grid = {}
    for k, v in param_grid.items():
        if k == 'final_C':
            mapped_grid['clf__final_estimator__C'] = v
        elif k.startswith('clf__'):
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
                      prefix='stacking_bundle', save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)
    classes = list(classes)

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


def save_bundle(model_pipeline, X_columns, classes, target_col='Target', smiles_col='SMILES',
                best_params=None, best_score=None, bundle_prefix='stacking_bundle', save_dir='outputs'):
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
        'bundle_version': '1.0-stacking',
        'notes': 'Pipeline = SimpleImputer(median) -> StandardScaler -> StackingClassifier(lr,rf,gb) -> final LogisticRegression'
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("✅ Bundle saved:")
    print(f" - {pipeline_path}")
    print(f" - {metadata_path}")
    return pipeline_path, metadata_path

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def evaluate_train_test_classification(
    model,
    X_train, y_train,
    X_test, y_test,
    classes,
    model_name="StackingClassifier",
    prefix="stacking_bundle",
    save_dir="outputs"
):
    os.makedirs(save_dir, exist_ok=True)
    classes = list(classes)

    # ============================
    # Predictions
    # ============================
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)
    else:
        raise RuntimeError("Model must support predict_proba for ROC-AUC.")

    # ============================
    # Metrics
    # ============================
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    try:
        auc_train = roc_auc_score(y_train, y_train_proba, multi_class="ovr")
        auc_test = roc_auc_score(y_test, y_test_proba, multi_class="ovr")
    except Exception:
        auc_train = None
        auc_test = None

    metrics = {
        "train": {
            "accuracy": float(acc_train),
            "roc_auc_ovr": float(auc_train) if auc_train is not None else None
        },
        "test": {
            "accuracy": float(acc_test),
            "roc_auc_ovr": float(auc_test) if auc_test is not None else None
        }
    }

    print("\n================ CLASSIFICATION PERFORMANCE ================")
    for split in ["train", "test"]:
        print(f"\n{split.upper()} SET:")
        for k, v in metrics[split].items():
            print(f"  {k}: {v}")

    # Save metrics
    metrics_path = os.path.join(save_dir, f"{prefix}_train_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to: {metrics_path}")

    # =========================================================
    # 1. CONFUSION MATRICES (TRAIN vs TEST)
    # =========================================================
    for split, y_true, y_pred in [
        ("train", y_train, y_train_pred),
        ("test", y_test, y_test_pred)
    ]:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(
            f"{model_name} – {split.capitalize()} Confusion Matrix\n"
            f"Accuracy = {metrics[split]['accuracy']:.3f}"
        )
        plt.tight_layout()

        out = os.path.join(save_dir, f"{prefix}_confusion_{split}.jpg")
        plt.savefig(out, dpi=300)
        plt.close()

    # =========================================================
    # 2. ROC CURVES (TEST SET – ONE-vs-REST)
    # =========================================================
    y_test_bin = label_binarize(y_test, classes=classes)

    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_test_proba[:, idx])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve – Class {cls} (Test Set)")
        plt.legend(loc="lower right")
        plt.tight_layout()

        roc_path = os.path.join(save_dir, f"{prefix}_roc_test_class_{cls}.jpg")
        plt.savefig(roc_path, dpi=300)
        plt.close()

    print("📊 Train/Test confusion matrices and ROC curves saved.")

    return metrics


def main():
    input_csv = '1000selected_features_class.csv'
    target_col = 'Target'
    smiles_col = 'SMILES'
    bundle_prefix = 'stacking_bundle'
    save_dir = 'outputs'

    print("Loading data...")
    df = load_data(input_csv, target_col=target_col)

    print("Preparing X and y...")
    X, y = prepare_X_y(df, target_col=target_col, smiles_col=smiles_col)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test, df_test = split_train_test_with_df(
        df, X, y, test_size=0.2, random_state=42, stratify=True
    )

    # پارامترهایی که می‌خواهیم جست‌وجو کنیم
    param_grid = {
        # mapping helper: 'final_C' -> clf__final_estimator__C
        'final_C': [0.01, 0.1, 1],
        # امکان اضافه کردن clf__passthrough یا سایر پارامترهای سطح بالای Stacking
        'clf__passthrough': [False]
    }

    print("Tuning stacking model (GridSearchCV)...")
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

    
    print("\nRunning TRAIN vs TEST evaluation with labeled plots...")
    train_test_metrics = evaluate_train_test_classification(
        model=model_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classes=sorted(y.unique()),
        model_name="StackingClassifier (LR + RF + GB)",
        prefix=bundle_prefix,
        save_dir=save_dir
    )
    
    print("All done.")


if __name__ == '__main__':
    main()
