import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from fpdf import FPDF


def load_data(file_path, target_col='Activity'):
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(f"ستون '{target_col}' یافت نشد.")
    return df


def preprocess(df, target_col='Activity', drop_cols=None):
    drop_list = [target_col] + (drop_cols or [])
    X = df.drop(columns=drop_list)
    y = df[target_col]

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def tune_model(X_train, y_train, param_dist, cv=3, n_iter=5, n_jobs=2):
    print("🔧 Randomized hyperparameter tuning for ANN...")

    pipeline = Pipeline([
        ('clf', MLPRegressor(max_iter=1000, random_state=42, early_stopping=True))
    ])

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={f'clf__{k}': v for k, v in param_dist.items()},
        n_iter=n_iter,
        cv=cv, n_jobs=n_jobs, verbose=2,
        scoring='r2', random_state=42
    )
    random_search.fit(X_train, y_train)
    print(f"✅ Best params: {random_search.best_params_}")
    return random_search.best_estimator_


def evaluate_and_save(model, X_test, y_test, prefix='ann_reg'):
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
    pdf.cell(0,10,'ANN Regression Report',ln=True,align='C')
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
    X, y = preprocess(df, target_col='Activity')
    X_train, X_test, y_train, y_test = split_data(X, y)

    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (100,50), (100,100,50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01,0.1],
        'batch_size': ['auto', 32, 64,128],
        'learning_rate': ['constant', 'adaptive','invscaling'],
        'learning_rate_init': [0.001, 0.01,0.1],
        'momentum': [0.0,0.5,0.9],
        'validation_fraction': [0.1,0.2]
    }

    model = tune_model(X_train, y_train, param_dist)
    evaluate_and_save(model, X_test, y_test)


if __name__ == '__main__':
    main()
# ✅ Best params: {'clf__validation_fraction': 0.2, 'clf__solver': 'sgd', 'clf__momentum': 0.0, 'clf__learning_rate_init': 0.1, 'clf__learning_rate': 'constant', 'clf__hidden_layer_sizes': (100, 50), 'clf__batch_size': 'auto', 'clf__alpha': 0.001, 'clf__activation': 'logistic'}