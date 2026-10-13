import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.neighbors import KNeighborsClassifier


def load_data(file_path, target_col='Target'):
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(f"ستون '{target_col}' یافت نشد.")
    return df


def preprocess(df, target_col='Target', drop_cols=None):
    drop_list = [target_col] + (drop_cols or [])
    X = df.drop(columns=drop_list)
    y = df[target_col]

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)


def tune_model(X_train, y_train, param_grid, cv=3, n_jobs=3):
    print("Progress: Hyperparameter tuning...")

    pipeline = Pipeline([
        ('clf', KNeighborsClassifier())
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={f'clf__{k}': v for k, v in param_grid.items()},
        cv=cv, n_jobs=n_jobs, verbose=1
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    return grid.best_estimator_


def evaluate_and_save(model, X_test, y_test, classes, prefix='knn_class'):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(report)

    df_res = X_test.copy()
    df_res['Actual'] = y_test.values
    df_res['Predicted'] = y_pred
    for idx, cls in enumerate(classes):
        df_res[f'Prob_Class_{cls}'] = y_proba[:, idx]
    df_res.to_excel(f'{prefix}_results.xlsx', index=False)
    print(f"✅ نتایج در '{prefix}_results.xlsx' ذخیره شد.")

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_file = f'{prefix}_confusion.png'
    plt.savefig(cm_file)
    plt.close()

    images = [cm_file]
    for idx, cls in enumerate(classes):
        y_bin = label_binarize(y_test, classes=classes)[:, idx]
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f'AUC {cls}={roc_auc:.3f}')
        plt.plot([0,1],[0,1],'--')
        plt.title(f'ROC for {cls}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        roc_file = f'{prefix}_roc_{cls}.png'
        plt.savefig(roc_file)
        plt.close()
        images.append(roc_file)
    print("📊 نمودارها ذخیره شدند.")

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'KNN Classification Report',ln=True,align='C')
    pdf.ln(5)
    pdf.set_font('Arial','',12)
    for line in report.split('\n'):
        pdf.multi_cell(0,8,line)
    for img in images:
        pdf.add_page()
        pdf.image(img, x=15, y=25, w=180)
    pdf.output(f'{prefix}_report.pdf')
    print(f"📄 گزارش PDF در '{prefix}_report.pdf' ذخیره شد.")

    joblib.dump(model, f'{prefix}_model.joblib')
    print(f"✅ مدل در '{prefix}_model.joblib' ذخیره شد.")


def main():
    df = load_data('1000selected_features_classification.csv', target_col='Target')
    X, y = preprocess(df, target_col='Target')
    X_train, X_test, y_train, y_test = split_data(X, y)

    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski'],  # Euclidean
        'p': [1,2]  # p=2 → Euclidean, p=1 → Manhattan
    }

    model = tune_model(X_train, y_train, param_grid)

    classes = sorted(y.unique())
    evaluate_and_save(model, X_test, y_test, classes)


if __name__ == '__main__':
    main()
# Best params: {'clf__metric': 'minkowski', 'clf__n_neighbors': 5, 'clf__p': 1, 'clf__weights': 'distance'}