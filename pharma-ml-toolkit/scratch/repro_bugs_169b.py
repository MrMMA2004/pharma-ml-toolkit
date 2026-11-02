"""Library repro (100x-169b): classifier figures with STRING class names."""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from pharmaml.ml import evaluate_model, train_model
from pharmaml.ml.splits import make_train_test_split

rng = np.random.default_rng(42)
n = 200
frame = pd.DataFrame({
    "f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n),
    "potency": rng.choice(["low", "mid", "high"], size=n),
})
y = frame["potency"]
xtr, xte, ytr, yte = make_train_test_split(frame[["f1", "f2", "f3"]], y, strategy="random", random_state=42)
trained = train_model("random_forest", "classification", xtr, ytr, tune=False)
report = evaluate_model(trained.model, xte, yte, "classification")
print("labels:", report.labels)

from pharmaml.viz import plot_confusion_matrix, plot_roc_curves, plot_calibration_curve
for label, fn in [
    ("confusion", lambda: plot_confusion_matrix(report.confusion, labels=report.labels)),
    ("roc", lambda: plot_roc_curves(trained.model, xte, yte)),
    ("calibration", lambda: plot_calibration_curve(trained.model, xte, yte)),
]:
    try:
        fig = fn()
        print(f"{label}: OK")
    except Exception as exc:
        print(f"{label}: FAILED {type(exc).__name__}: {exc}")

# PR curves?
try:
    from pharmaml.viz import plot_pr_curves
    fig = plot_pr_curves(trained.model, xte, yte)
    print("pr: OK")
except ImportError:
    print("pr: no plot_pr_curves export")
except Exception as exc:
    print(f"pr: FAILED {type(exc).__name__}: {exc}")

# binary strings too
frame2 = pd.DataFrame({
    "f1": rng.normal(size=150), "f2": rng.normal(size=150),
    "ok": rng.choice(["yes", "no"], size=150),
})
y2 = frame2["ok"]
x2tr, x2te, y2tr, y2te = make_train_test_split(frame2[["f1", "f2"]], y2, strategy="random", random_state=1)
t2 = train_model("logistic_regression", "classification", x2tr, y2tr, tune=False)
r2 = evaluate_model(t2.model, x2te, y2te, "classification")
for label, fn in [
    ("bin-confusion", lambda: plot_confusion_matrix(r2.confusion, labels=r2.labels)),
    ("bin-roc", lambda: plot_roc_curves(t2.model, x2te, y2te)),
    ("bin-calibration", lambda: plot_calibration_curve(t2.model, x2te, y2te)),
]:
    try:
        fig = fn()
        print(f"{label}: OK")
    except Exception as exc:
        print(f"{label}: FAILED {type(exc).__name__}: {exc}")
