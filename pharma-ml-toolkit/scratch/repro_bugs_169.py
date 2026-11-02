"""Reproduce the four user-reported bugs (round 183, tag 100x-169).

1. heatmap not generated for classes with string names
2. SHAP chart errors
Run headless: pure library calls, no Tk.
"""
import sys
import traceback

sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit/src")

import numpy as np
import pandas as pd

print("=" * 60)
print("BUG 1: confusion heatmap with STRING class names")
print("=" * 60)
rng = np.random.default_rng(42)
n = 120
frame = pd.DataFrame({
    "f1": rng.normal(size=n),
    "f2": rng.normal(size=n),
    "f3": rng.normal(size=n),
    "activity": rng.choice(["active", "inactive"], size=n),  # STRING classes
})

from pharmaml.ml import evaluate_model, train_model
from pharmaml.ml.splits import make_train_test_split

target = "activity"
y = frame[target]
x_train, x_test, y_train, y_test = make_train_test_split(
    frame[["f1", "f2", "f3"]], y, strategy="random", random_state=42,
)
print(f"y_train dtype: {y_train.dtype!r}; classes: {sorted(set(y_train))}")

trained = train_model("logistic_regression", "classification",
                      x_train, y_train, tune=False)
report = evaluate_model(trained.model, x_test, y_test, "classification")
print(f"report.labels: {report.labels!r}")
print(f"report.confusion:\n{report.confusion}")

try:
    from pharmaml.viz import plot_confusion_matrix
    fig = plot_confusion_matrix(report.confusion, labels=report.labels)
    print("HEATMAP OK:", type(fig).__name__)
except Exception as exc:
    print(f"HEATMAP FAILED: {type(exc).__name__}: {exc}")
    traceback.print_exc()

print()
print("=" * 60)
print("BUG 1b: multiclass string labels via the GUI's exact call")
print("=" * 60)
frame2 = pd.DataFrame({
    "f1": rng.normal(size=150),
    "f2": rng.normal(size=150),
    "potency": rng.choice(["low", "mid", "high"], size=150),
})
y2 = frame2["potency"]
x2tr, x2te, y2tr, y2te = make_train_test_split(
    frame2[["f1", "f2"]], y2, strategy="random", random_state=42,
)
trained2 = train_model("random_forest", "classification", x2tr, y2tr, tune=False)
report2 = evaluate_model(trained2.model, x2te, y2te, "classification")
print(f"report2.labels: {report2.labels!r}")
try:
    fig2 = plot_confusion_matrix(report2.confusion, labels=report2.labels)
    print("HEATMAP2 OK:", type(fig2).__name__)
except Exception as exc:
    print(f"HEATMAP2 FAILED: {type(exc).__name__}: {exc}")
    traceback.print_exc()

print()
print("=" * 60)
print("BUG 2: SHAP — the GUI passes the TrainingResult itself")
print("=" * 60)
try:
    from pharmaml.ml import shap_summary
    summary = shap_summary(trained2, x2te)
    print("SHAP(trained) OK")
except Exception as exc:
    print(f"SHAP(trained) FAILED: {type(exc).__name__}: {exc}")
try:
    summary = shap_summary(trained2.model, x2te)
    print("SHAP(trained.model) OK")
except Exception as exc:
    print(f"SHAP(trained.model) FAILED: {type(exc).__name__}: {exc}")
try:
    inner = trained2.model.steps[-1][1]
    print(f"inner estimator: {type(inner).__name__}")
    summary = shap_summary(inner, x2te)
    print("SHAP(inner) OK:", summary.model_name, "n_features:", len(summary.names))
except Exception as exc:
    print(f"SHAP(inner) FAILED: {type(exc).__name__}: {exc}")
    traceback.print_exc()
