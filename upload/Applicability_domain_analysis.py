
# Applicability Domain Analysis

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from catboost import CatBoostRegressor

# 1. LOAD DATA
train_df = pd.read_csv("Optimized_Train_set_with_catboost_predictions.csv")
test_df  = pd.read_csv("Optimized_Test_set_with_catboost_predictions.csv")

model = joblib.load("optimized_model.pkl")

# 2. PREPARE FEATURE MATRICES
# First column = SMILES
# Assume target column is named 'pIC50' or similar
# Remove non-feature columns manually if needed

non_feature_cols = ["SMILES", "pIC50", "prediction"]  # adjust if needed

# 3. PREPARE FEATURE MATRICES (ROBUST VERSION)
# Keep only numeric columns
X_train = train_df.select_dtypes(include=[np.number])
X_test  = test_df.select_dtypes(include=[np.number])

# Make sure target column is not included
if "pIC50" in X_train.columns:
    X_train = X_train.drop(columns=["pIC50"])
if "pIC50" in X_test.columns:
    X_test = X_test.drop(columns=["pIC50"])

print("Training feature shape:", X_train.shape)
print("Test feature shape:", X_test.shape)

compound_names = test_df.iloc[:, 0].values  # SMILES column

# 4. STANDARDIZE DESCRIPTOR SPACE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 5. kNN DISTANCE-BASED AD
k = 5
nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
nbrs.fit(X_train_scaled)

# Training distances
train_distances, _ = nbrs.kneighbors(X_train_scaled)
train_knn_mean = train_distances.mean(axis=1)

threshold = np.percentile(train_knn_mean, 95)
print("Distance threshold (95th percentile):", threshold)

# Test distances
test_distances, _ = nbrs.kneighbors(X_test_scaled)
test_knn_mean = test_distances.mean(axis=1)

inside_ad = test_knn_mean <= threshold

# 6. BOOTSTRAP UNCERTAINTY
n_models = 30
boot_predictions = []

y_train = train_df["pIC50"].values  # adjust if different

for i in range(n_models):

    X_boot, y_boot = resample(X_train, y_train)

    model_boot = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        verbose=0
    )

    model_boot.fit(X_boot, y_boot)

    preds = model_boot.predict(X_test)
    boot_predictions.append(preds)

boot_predictions = np.array(boot_predictions)

pred_mean = boot_predictions.mean(axis=0)
pred_std  = boot_predictions.std(axis=0)

# Determine uncertainty threshold from training data
train_boot_predictions = []

for i in range(n_models):

    X_boot, y_boot = resample(X_train, y_train)

    model_boot = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        verbose=0
    )

    model_boot.fit(X_boot, y_boot)

    train_preds = model_boot.predict(X_train)
    train_boot_predictions.append(train_preds)

train_boot_predictions = np.array(train_boot_predictions)
train_std = train_boot_predictions.std(axis=0)

uncertainty_threshold = np.percentile(train_std, 95)
print("Uncertainty threshold:", uncertainty_threshold)
low_uncertainty = pred_std <= uncertainty_threshold

# 7. FINAL RELIABILITY TABLE
final_reliable = inside_ad & low_uncertainty
results_df = pd.DataFrame({
    "SMILES": compound_names,
    "Predicted_pIC50": pred_mean,
    "Prediction_SD": pred_std,
    "kNN_distance": test_knn_mean,
    "Inside_AD": inside_ad,
    "Low_Uncertainty": low_uncertainty,
    "Reliable_Prediction": final_reliable
})
results_df.to_csv("Applicability_Domain_Results.csv", index=False)
print("Applicability Domain analysis complete.")

# 8. VISUALIZATION FOR APPLICABILITY DOMAIN (COMBINED FIGURE)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): Histogram of training kNN distances
axes[0].hist(train_knn_mean, bins=40, color='skyblue', edgecolor='black')
axes[0].axvline(threshold, color='red', linestyle='--', label=f"95th percentile = {threshold:.2f}")
axes[0].set_xlabel("Mean 5-NN Distance (Training Set)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("(a) Training kNN Distance Distribution")
axes[0].legend()

# Panel (b): Scatter plot of kNN distance vs Predicted pIC50
# Highlight outside AD compounds
axes[1].scatter(
    test_knn_mean[inside_ad],
    pred_mean[inside_ad],
    s=15,
    alpha=0.7,
    label="Inside AD",
    color='green'
)
axes[1].scatter(
    test_knn_mean[~inside_ad],
    pred_mean[~inside_ad],
    s=25,
    alpha=0.9,
    label="Outside AD",
    color='red'
)
axes[1].axvline(threshold, color='blue', linestyle='--', label=f"95th percentile = {threshold:.2f}")
axes[1].set_xlabel("Mean 5-NN Distance")
axes[1].set_ylabel("Predicted pIC50")
axes[1].set_title("(b) kNN Distance vs Predicted pIC50")
axes[1].legend()

plt.tight_layout()
plt.savefig("AD_Combined_Figure.jpeg", dpi=500)

print("Combined AD figure saved as 'AD_Combined_Figure.jpeg'")