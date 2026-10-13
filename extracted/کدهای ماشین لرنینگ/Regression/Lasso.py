import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso  # Changed import
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from fpdf import FPDF


def load_data(file_path, target_col='Activity'):
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    return df


def split_data(df, target_col='Activity', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def tune_model(X_train, y_train, param_grid, cv=3, n_jobs=2):
    print("🔧 Hyperparameter tuning for Lasso...")

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', Lasso(random_state=42))  # Changed model
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={f'clf__{k}': v for k, v in param_grid.items()},
        cv=cv, n_jobs=n_jobs, verbose=1
    )
    grid.fit(X_train, y_train)
    print(f"✅ Best params: {grid.best_params_}")
    return grid.best_estimator_


def evaluate_and_save(model, X_test, y_test, prefix='lasso_reg'):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

    df_res = X_test.copy()
    df_res['Actual'] = y_test.values
    df_res['Predicted'] = y_pred
    df_res.to_excel(f'{prefix}_results.xlsx', index=False)
    print(f"📁 Results saved to '{prefix}_results.xlsx'")

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual (Lasso)") # Changed title
    plt.tight_layout()
    img_file = f'{prefix}_scatter.png'
    plt.savefig(img_file)
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Lasso Regression Report', ln=True, align='C') # Changed title
    pdf.ln(5)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"MAE: {mae:.4f}", ln=True)
    pdf.cell(0, 10, f"MSE: {mse:.4f}", ln=True)
    pdf.cell(0, 10, f"R²: {r2:.4f}", ln=True)
    pdf.add_page()
    pdf.image(img_file, x=15, y=25, w=180)
    pdf.output(f'{prefix}_report.pdf')
    print(f"📄 PDF report saved to '{prefix}_report.pdf'")

    joblib.dump(model, f'{prefix}_model.joblib')
    print(f"✅ Model saved to '{prefix}_model.joblib'")


def main():
    df = load_data('1000selected_features_regression.csv', target_col='Activity')
    X_train, X_test, y_train, y_test = split_data(df)

    # Hyperparameter grid for Lasso
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
    }

    model = tune_model(X_train, y_train, param_grid)
    evaluate_and_save(model, X_test, y_test)


if __name__ == '__main__':
    main()
#✅ Best params: {'clf__alpha': 0.01}