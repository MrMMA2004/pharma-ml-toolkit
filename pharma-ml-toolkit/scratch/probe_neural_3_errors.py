"""Probe 3: error contracts, input handling, predict_proba alignment.

100x-18-b review of ml/neural.py — verify-before-fix probe.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.utils.validation import check_is_fitted

from pharmaml.ml.neural import DeepMLPClassifier, DeepMLPRegressor
from pharmaml.exceptions import DependencyError

def header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


X, y = make_blobs(n_samples=90, centers=[(-4.0, -4.0), (0.0, 4.0), (4.0, -4.0)],
                  cluster_std=0.6, random_state=2)

# ---------------------------------------------------------------- (1) feature count mismatch at predict
header("(1) predict with wrong number of features")

clf = DeepMLPClassifier(hidden_layers=(8,), epochs=2, random_state=1).fit(X, y)
try:
    clf.predict(X[:, :1])  # 1 feature vs 2 fitted
except Exception as exc:
    print(f"RAW {type(exc).__name__}: {str(exc)[:160]}")
    print(">>> DEFECT if raw torch RuntimeError (should be a typed pharmaml error)")

# ---------------------------------------------------------------- (2) y containing None
header("(2) classifier y containing None")

Xb = X[:20]
y_none = ["a"] * 10 + ["b"] * 9 + [None]
try:
    DeepMLPClassifier(hidden_layers=(4,), epochs=1, random_state=1).fit(Xb, y_none)
except Exception as exc:
    print(f"{type(exc).__name__}: {str(exc)[:200]}")
    print(">>> DEFECT if raw TypeError (should be a typed pharmaml DataValidationError)")

# ---------------------------------------------------------------- (3) generator hidden_layers
header("(3) hidden_layers passed as a GENERATOR")

gen = (w for w in (16, 8))
est = DeepMLPClassifier(hidden_layers=gen, epochs=2, random_state=1).fit(X, y)
print(f"_net_spec_['hidden'] = {est._net_spec_['hidden']}   (user asked for (16, 8)!)")
print(">>> DEFECT if () — the validation loop consumed the generator and the "
      "model silently degenerated to a single linear layer")

# list input (should be fine) as control:
est2 = DeepMLPClassifier(hidden_layers=[16, 8], epochs=1, random_state=1).fit(X, y)
print(f"control list input: _net_spec_['hidden'] = {est2._net_spec_['hidden']}")

# ---------------------------------------------------------------- (4) failed fit (torch missing) leaves half-fitted state?
header("(4) attributes set BEFORE torch import -> half-fitted after DependencyError")

saved = sys.modules.get("torch")
sys.modules["torch"] = None  # simulate torch-less machine
try:
    est3 = DeepMLPClassifier(hidden_layers=(4,), epochs=1, random_state=1)
    try:
        est3.fit(X, y)
    except DependencyError as exc:
        print(f"fit raised DependencyError as documented: {str(exc)[:80]}")
    attrs = [a for a in ("classes_", "n_features_in_", "_mean_", "_std_") if hasattr(est3, a)]
    print(f"fitted-looking attributes present after FAILED fit: {attrs}")
    try:
        check_is_fitted(est3)
        print("sklearn check_is_fitted: CONSIDERS THE ESTIMATOR FITTED")
        print(">>> DEFECT: failed fit left classes_/n_features_in_/_mean_ behind "
              "(sklearn contract: exceptions leave the estimator unfitted)")
    except Exception as exc:
        print(f"check_is_fitted raises {type(exc).__name__} (correct)")
finally:
    if saved is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = saved

# ---------------------------------------------------------------- (5) classes_ / proba column alignment, shuffled labels
header("(5) string labels ['b','a','c']: classes_ sorted? proba columns aligned?")

y_str = np.array(["b", "a", "c"])[y]
clf2 = DeepMLPClassifier(hidden_layers=(8,), epochs=20, patience=5, random_state=2).fit(X, y_str)
print(f"classes_ = {list(clf2.classes_)}")
# a point deep in the 'a'(-4,-4) cluster must put its max proba in COLUMN 0
a_point = np.array([[-4.2, -4.1]])
proba = clf2.predict_proba(a_point)
print(f"P(a)={proba[0, 0]:.3f} P(b)={proba[0, 1]:.3f} P(c)={proba[0, 2]:.3f} "
      f"-> argmax col={int(np.argmax(proba))}, predict={clf2.predict(a_point)}")
aligned = (list(clf2.classes_) == ["a", "b", "c"]) and np.argmax(proba) == 0
print(f"classes_ sorted + column-0 == 'a' alignment: {aligned}")

# ---------------------------------------------------------------- (6) mixed-dtype DataFrame / int arrays
header("(6) X input handling: mixed dtypes, int arrays, float32")

df_ok = pd.DataFrame({"a": np.arange(90.0), "b": np.arange(90.0) * 2 - 40})
print("numeric mixed int/float DataFrame fits:",
      DeepMLPRegressor(hidden_layers=(4,), epochs=1, random_state=1).fit(df_ok, np.arange(90.0) * 1.0).n_iter_ == 1)
df_bad = pd.DataFrame({"a": np.arange(90.0), "b": [f"s{i}" for i in range(90)]})
try:
    DeepMLPRegressor(hidden_layers=(4,), epochs=1, random_state=1).fit(df_bad, np.arange(90.0))
except Exception as exc:
    print(f"string column -> {type(exc).__name__}: {str(exc)[:90]}")
print("int X array:", DeepMLPClassifier(hidden_layers=(4,), epochs=1, random_state=1).fit(X.astype(int), y).n_iter_ == 1)
print("float32 X:", DeepMLPClassifier(hidden_layers=(4,), epochs=1, random_state=1).fit(X.astype(np.float32), y).n_iter_ == 1)

# ---------------------------------------------------------------- (7) dropout disabled at predict
header("(7) dropout eval-mode at predict (dropout=0.5)")

clf3 = DeepMLPClassifier(hidden_layers=(32, 32), dropout=0.5, epochs=3, random_state=4).fit(X, y)
p1, p2 = clf3.predict_proba(X[:10]), clf3.predict_proba(X[:10])
print(f"two predict_proba calls identical (dropout OFF): {np.array_equal(p1, p2)}")

# ---------------------------------------------------------------- (8) batch_size > n_samples
header("(8) batch_size larger than n_samples")

small = DeepMLPClassifier(hidden_layers=(8,), epochs=2, batch_size=1000, random_state=1).fit(X[:12], y[:12])
print(f"batch_size=1000 on 12 rows: n_iter_={small.n_iter_}, proba sums ok: "
      f"{np.allclose(small.predict_proba(X[:12]).sum(axis=1), 1.0)}")
