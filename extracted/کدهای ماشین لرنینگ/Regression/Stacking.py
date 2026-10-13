import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from fpdf import FPDF


def load_data(file_path, target_col='Activity'):
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(f"ستون '{target_col}' یافت نشد.")
    return df


def preprocess_split(df, target_col='Activity', drop_cols=None):
    drop_list = [target_col] + (drop_cols or [])
    X = df.drop(columns=drop_list)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)

    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train, X_test, y_train, y_test


def build_stacking_model():
    base_models = [
        ('ridge', Ridge(alpha=100.0)),
        ('knn', KNeighborsRegressor(n_neighbors=3)),
        ('svr', SVR(kernel='rbf', C=100, epsilon=0.1)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=30, random_state=42))
    ]
    meta_model = Ridge(alpha=100.0)
    model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=3, n_jobs=2, passthrough=False)
    return model


def evaluate_and_save(model, X_test, y_test, prefix='stacking_reg'):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

    df_res = X_test.copy()
    df_res['Actual'] = y_test.values
    df_res['Predicted'] = y_pred
    df_res.to_excel(f'{prefix}_results.xlsx', index=False)
    print(f"📁 نتایج در '{prefix}_results.xlsx' ذخیره شد.")

    plt.figure(figsize=(7,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    img_file = f'{prefix}_scatter.png'
    plt.savefig(img_file)
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'Stacking Regression Report',ln=True,align='C')
    pdf.ln(5)
    pdf.set_font('Arial','',12)
    pdf.cell(0,10,f"MAE: {mae:.4f}", ln=True)
    pdf.cell(0,10,f"MSE: {mse:.4f}", ln=True)
    pdf.cell(0,10,f"R²: {r2:.4f}", ln=True)
    pdf.add_page()
    pdf.image(img_file, x=15, y=25, w=180)
    pdf.output(f'{prefix}_report.pdf')
    print(f"📄 گزارش PDF در '{prefix}_report.pdf' ذخیره شد.")

    joblib.dump(model, f'{prefix}_model.joblib')
    print(f"✅ مدل در '{prefix}_model.joblib' ذخیره شد.")


def main():
    df = load_data('1000selected_features_regression.csv', target_col='Activity')
    X_train, X_test, y_train, y_test = preprocess_split(df, target_col='Activity')

    model = build_stacking_model()
    model.fit(X_train, y_train)
    evaluate_and_save(model, X_test, y_test)


if __name__ == '__main__':
    main()
