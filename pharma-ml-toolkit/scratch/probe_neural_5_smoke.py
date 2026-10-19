"""End-to-end smoke for 100x-18-b (outside the test suite).

DeepMLPClassifier on make_classification(200, 20) + ResidualMLPRegressor
on a synthetic sine function; prints accuracy / R2 / determinism check.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from pharmaml.ml.neural import DeepMLPClassifier, ResidualMLPRegressor

# ---- classification -------------------------------------------------------
X, y = make_classification(n_samples=200, n_features=20, n_informative=12, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

clf = DeepMLPClassifier(hidden_layers=(32, 16), epochs=10, patience=4, random_state=42)
clf.fit(X_tr, y_tr)
acc = accuracy_score(y_te, clf.predict(X_te))
print(f"DeepMLPClassifier  make_classification(200, 20): accuracy = {acc:.3f}  "
      f"(n_iter_={clf.n_iter_}, val_loss final={clf.history_[-1]['val_loss']:.4f})")

clf2 = DeepMLPClassifier(hidden_layers=(32, 16), epochs=10, patience=4, random_state=42)
clf2.fit(X_tr, y_tr)
print(f"determinism (same seed, refit): identical predictions = "
      f"{np.array_equal(clf.predict(X_te), clf2.predict(X_te))}")

# ---- regression on a sine surface ----------------------------------------
rng = np.random.default_rng(7)
Xr = rng.uniform(-3.0, 3.0, size=(200, 2))
yr = np.sin(Xr[:, 0]) * np.cos(Xr[:, 1]) + 0.05 * rng.normal(size=200)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, random_state=7)

reg = ResidualMLPRegressor(hidden_layers=(32, 32, 32), epochs=10, patience=4, random_state=42)
reg.fit(Xr_tr, yr_tr)
r2 = r2_score(yr_te, reg.predict(Xr_te))
print(f"ResidualMLPRegressor  sine(200x2): R^2 = {r2:.3f}  (n_iter_={reg.n_iter_})")
print(f"score() (sklearn r2_score semantics) = {reg.score(Xr_te, yr_te):.3f}")
preds = reg.predict(Xr_te)
print(f"predictions finite: {np.isfinite(preds).all()}")
