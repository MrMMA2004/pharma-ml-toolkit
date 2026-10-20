import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)


def load_data(file_path, target_col='Target'):
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(f"ستون '{target_col}' یافت نشد.")
    return df


def preprocess_split(df, target_col='Target', drop_cols=None):
    drop_list = [target_col] + (drop_cols or [])
    X = df.drop(columns=drop_list)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)

    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train, X_test, y_train, y_test



def build_stacking_model():
    base_estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svc', SVC(kernel='rbf', probability=True)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=8))
    ]
    final_estimator = LogisticRegression()
    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5, n_jobs=3, passthrough=False
    )
    return model


def evaluate_and_save(model, X_test, y_test, classes, prefix='stacking_clf'):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(report)

    # Save results
    df_res = X_test.copy()
    df_res['Actual'] = y_test.values
    df_res['Predicted'] = y_pred
    for i, cls in enumerate(classes):
        df_res[f'Prob_{cls}'] = y_proba[:, i]
    df_res.to_excel(f'{prefix}_results.xlsx', index=False)
    print(f"📁 نتایج در '{prefix}_results.xlsx' ذخیره شد.")

    # Confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_file = f'{prefix}_confusion.png'
    plt.savefig(cm_file)
    plt.close()

    # ROC curves
    images = [cm_file]
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

    # PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'Stacking Classifier Report', ln=True, align='C')
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
    X_train, X_test, y_train, y_test = preprocess_split(df, target_col='Target')

    model = build_stacking_model()
    model.fit(X_train, y_train)

    classes = sorted(y_test.unique())
    evaluate_and_save(model, X_test, y_test, classes)


if __name__ == '__main__':
    main()
