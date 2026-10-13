import pandas as pd
import seaborn as sns
import joblib
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import numpy as np


def load_data(file_path, target_col='Activity'):
    """Loads data and separates features and target."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(f"Error: The target column '{target_col}' was not found in the CSV.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def tune_model_optuna(X_train, y_train, n_trials=50, cv=3):
    """Tunes SVR hyperparameters using Optuna."""
    print(f"🔧 Starting Optuna hyperparameter tuning with {n_trials} trials...")

    # Define the objective function for Optuna to optimize
    def objective(trial):
        # Define the search space for hyperparameters
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
        c = trial.suggest_float('C', 1e-2, 1e3, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-2, 1.0)

        # Additional parameters based on kernel
        gamma = 'scale'
        degree = 3
        if kernel == 'rbf':
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)

        # Create the Scikit-learn pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', SVR(kernel=kernel, C=c, epsilon=epsilon, gamma=gamma, degree=degree))
        ])

        # Evaluate the pipeline using cross-validation
        # Optuna maximizes the return value, so R² is a good metric
        score = cross_val_score(pipeline, X_train, y_train, n_jobs=-1, cv=cv, scoring='r2')
        return score.mean()

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print(f"\n✅ Optuna tuning finished.")
    print(f"✅ Best CV R² score: {study.best_value:.4f}")
    print(f"✅ Best params: {study.best_params}")

    # Build and train the final model with the best hyperparameters
    print("\n⚙️ Training final model with best parameters...")
    best_params = study.best_params
    final_model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', SVR(**best_params))
    ])
    final_model.fit(X_train, y_train)
    print("✅ Final model trained successfully.")
    return final_model


def evaluate_and_save(model, X_test, y_test, prefix='svm_optuna_reg'):
    """Evaluates the model and saves results, plots, and a detailed PDF report."""
    print("\n📊 Evaluating model on the test set...")
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test Set Metrics -> MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

    # --- Create Visualizations ---
    plot_files = []

    # 1. Actual vs. Predicted Plot
    plt.figure(figsize=(8, 7))
    sns.regplot(x=y_test, y=y_pred, line_kws={"color": "#E63946", "lw": 2}, scatter_kws={'alpha': 0.6})
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title("Actual vs. Predicted Values", fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    img_file = f'{prefix}_scatter.png'
    plt.savefig(img_file)
    plt.close()
    plot_files.append(('Actual vs. Predicted Plot', img_file))

    # 2. Residuals Plot
    plt.figure(figsize=(8, 7))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='#E63946', linestyle='--', lw=2)
    plt.xlabel("Predicted Values", fontsize=12)
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
    plt.title("Residuals vs. Predicted Values", fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    img_file = f'{prefix}_residuals.png'
    plt.savefig(img_file)
    plt.close()
    plot_files.append(('Residuals Plot', img_file))

    # 3. Error Distribution Plot
    plt.figure(figsize=(8, 7))
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel("Prediction Error", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Prediction Errors", fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    img_file = f'{prefix}_error_dist.png'
    plt.savefig(img_file)
    plt.close()
    plot_files.append(('Error Distribution', img_file))

    # --- Generate PDF Report ---
    try:
        pdf = FPDF()
        # Title Page
        pdf.add_page()
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 20, 'SVM Regression Report', ln=True, align='C')
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 15, 'Model Performance Metrics', ln=True, align='L')
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f"  - Mean Absolute Error (MAE): {mae:.4f}", ln=True)
        pdf.cell(0, 8, f"  - Mean Squared Error (MSE): {mse:.4f}", ln=True)
        pdf.cell(0, 8, f"  - R-squared (R²): {r2:.4f}", ln=True)
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 15, 'Optimal Hyperparameters (from Optuna)', ln=True, align='L')
        pdf.set_font('Arial', '', 10)
        for param, value in model.named_steps['clf'].get_params().items():
            if param in ['C', 'kernel', 'epsilon', 'degree', 'gamma']:
                pdf.cell(0, 7, f"  - {param}: {value}", ln=True)

        # Add plot pages
        for title, file in plot_files:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 15, title, ln=True, align='C')
            pdf.image(file, x=15, y=30, w=180)  # Center image

        report_file = f'{prefix}_report.pdf'
        pdf.output(report_file)
        print(f"📄 PDF report saved to '{report_file}'")
    except Exception as e:
        print(f"Could not generate PDF report. Error: {e}")

    # --- Save Model and Results ---
    model_file = f'{prefix}_model.joblib'
    joblib.dump(model, model_file)
    print(f"✅ Final model saved to '{model_file}'")

    df_res = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Residuals': residuals})
    results_file = f'{prefix}_results.xlsx'
    df_res.to_excel(results_file, index=False)
    print(f"📁 Full results with residuals saved to '{results_file}'")


def main():
    """Main function to run the ML workflow."""
    # 1. Load data
    X, y = load_data('1000selected_features_regression.csv', target_col='Activity')

    # 2. Split data (BEFORE any preprocessing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Tune model using Optuna
    best_model = tune_model_optuna(X_train, y_train, n_trials=50)  # Increase trials for better results

    # 4. Evaluate the final model and save all artifacts
    if best_model:
        evaluate_and_save(best_model, X_test, y_test)


if __name__ == '__main__':
    # To run Optuna, you might need to install it first:
    # pip install optuna
    main()