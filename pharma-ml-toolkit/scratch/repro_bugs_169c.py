import sys
sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit/src")
import numpy as np, pandas as pd
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

# the correlation heatmap's exact work flow
from pharmaml.chem.interpret import signed_feature_correlations
try:
    corr = signed_feature_correlations(xte, yte, method="pearson")
    print("signed_feature_correlations OK:")
    print(corr.head())
except Exception as exc:
    print(f"signed_feature_correlations FAILED: {type(exc).__name__}: {exc}")

from pharmaml.viz import plot_correlation_heatmap
target = "potency"
f = xte.copy()
f[target] = yte.loc[f.index].to_numpy()
try:
    fig = plot_correlation_heatmap(f, columns=["f1", "f2", "f3", target], title="t")
    print("plot_correlation_heatmap OK")
except Exception as exc:
    print(f"plot_correlation_heatmap FAILED: {type(exc).__name__}: {exc}")
