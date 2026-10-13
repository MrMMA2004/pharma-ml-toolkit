import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np


def load_data(file_path, target_col='Target'):
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(f"ستون '{target_col}' یافت نشد.")
    return df


def preprocess(df, target_col='Target', drop_cols=None):
    drop_list = [target_col] + (drop_cols or [])
    X = df.drop(columns=drop_list)
    y = df[target_col]

    # تبدیل y به برچسب‌های صفرمبنا برای XGBoost
    le = LabelEncoder()
    y = le.fit_transform(y)

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)


def build_voting_model():
    model = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(solver='saga', max_iter=5000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
            ('xgb', XGBClassifier(tree_method='hist', n_estimators=100, learning_rate=0.1, random_state=42)),
            ('cat', CatBoostClassifier(verbose=0, iterations=200, random_state=42))
        ],
        voting='soft',  # hard,soft
        weights=[1, 2, 2, 2],  # وزن‌دهی بر اساس قدرت مدل
        n_jobs=3
    )
    return model


def evaluate_and_save(model, X_test, y_test, classes, prefix='voting_classifier_ensemble'):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(report)

    df_res = X_test.copy()
    df_res['Actual'] = y_test
    df_res['Predicted'] = y_pred
    df_res.to_excel(f'{prefix}_results.xlsx', index=False)
    print(f"📁 نتایج در '{prefix}_results.xlsx' ذخیره شد.")

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    cm_file = f'{prefix}_confusion.png'
    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()

    images = [cm_file]

    # اگر 'soft' voting بود ROC قابل محاسبه بود
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC {cls}={roc_auc:.3f}')
            plt.plot([0,1],[0,1],'--')
            plt.title(f'ROC Curve: {cls}')
            plt.legend()
            roc_file = f'{prefix}_roc_{cls}.png'
            plt.savefig(roc_file)
            plt.close()
            images.append(roc_file)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'Voting Classifier Report (XGB + LGBM + RF)', ln=True, align='C')
    pdf.ln(10)
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

    model = build_voting_model()
    model.fit(X_train, y_train)

    classes = sorted(np.unique(y))
    evaluate_and_save(model, X_test, y_test, classes)


if __name__ == '__main__':
    main()
