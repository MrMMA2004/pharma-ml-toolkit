"""Probe 4: registry wiring (param_grid validity, determinism across GridSearchCV).

100x-18-b review of ml/neural.py + models.py dnn entries — verify-before-fix.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from pharmaml.ml.models import MODEL_REGISTRY, build_model, resolve_model_name
from pharmaml.ml.neural import DeepMLPClassifier

def header(t): print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")

X, y = make_classification(n_samples=150, n_features=10, n_informative=6, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# ---------------------------------------------------------------- (1) param_grid values valid
header("(1) registry param_grid values are valid constructor values")

for key in ("dnn", "dnn_residual"):
    spec = MODEL_REGISTRY[key]
    grid = spec.param_grid
    est = build_model(key, "classification")
    # every single value must be settable
    for param, values in grid.items():
        for value in values:
            est.set_params(**{param: value})
    print(f"{key}: all {sum(len(v) for v in grid.values())} grid values settable OK; "
          f"keys={sorted(grid)}")
    combos = int(np.prod([len(v) for v in grid.values()]))
    print(f"   full grid = {combos} combinations (x folds) — the default tune=True path")

# ---------------------------------------------------------------- (2) aliases
header("(2) aliases resolve")
for alias in ("dnn", "deep_mlp", "DNN", "dnn_residual", "resnet", "Residual-DNN", "residual_dnn"):
    print(f"  {alias!r} -> {resolve_model_name(alias)}")

# ---------------------------------------------------------------- (3) GridSearchCV determinism (n_jobs=1)
header("(3) GridSearchCV determinism: same seed twice")

def run_search():
    search = GridSearchCV(
        DeepMLPClassifier(hidden_layers=(8,), epochs=8, patience=3, random_state=7),
        {"hidden_layers": [(8,), (12,)], "learning_rate": [1e-3, 3e-3]},
        cv=3, scoring="accuracy", n_jobs=1,
    )
    search.fit(X_tr, y_tr)
    return search.best_score_, search.best_params_, search.predict(X_te)

s1 = run_search()
s2 = run_search()
print(f"run1: score={s1[0]:.6f} params={s1[1]}")
print(f"run2: score={s2[0]:.6f} params={s2[1]}")
print(f"identical best_score_/best_params_/predictions: "
      f"{s1[0] == s2[0] and s1[1] == s2[1] and np.array_equal(s1[2], s2[2])}")
print(f"test accuracy: {accuracy_score(y_te, s1[2]):.3f}")

# ---------------------------------------------------------------- (4) classifier float y with NaN
header("(4) classifier y float containing NaN -> typed error?")

try:
    DeepMLPClassifier(hidden_layers=(4,), epochs=1, random_state=1).fit(
        X[:12], np.array([0.0, 1.0, 0.0, 1.0] * 3, dtype=float)
    )
    print("control (no NaN) fit OK")
except Exception as exc:
    print(f"unexpected: {type(exc).__name__}: {exc}")

y_nan = np.array([0.0, 1.0, np.nan] * 4)
try:
    DeepMLPClassifier(hidden_layers=(4,), epochs=1, random_state=1).fit(X[:12], y_nan)
    print("NO ERROR?! (defect)")
except Exception as exc:
    print(f"{type(exc).__name__}: {str(exc)[:110]}")

# ---------------------------------------------------------------- (5) epochs=1 with val split
header("(5) epochs=1 + val split (best snapshot must be epoch 1)")
m = DeepMLPClassifier(hidden_layers=(4,), epochs=1, val_split=0.2, random_state=1).fit(X_tr, y_tr)
print(f"n_iter_={m.n_iter_}, history={m.history_}, proba finite="
      f"{np.isfinite(m.predict_proba(X_te[:5])).all()}")

# ---------------------------------------------------------------- (6) val_split near 1
header("(6) val_split validation boundary")
for vs in (0.0, 0.999, 1.0, -0.1):
    try:
        DeepMLPClassifier(hidden_layers=(4,), epochs=1, val_split=vs, random_state=1).fit(X_tr, y_tr)
        print(f"  val_split={vs}: fit OK")
    except Exception as exc:
        print(f"  val_split={vs}: {type(exc).__name__}: {str(exc)[:70]}")
