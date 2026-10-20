"""Probe 1: early stopping, best-weight restore, NaN on extreme LR, refit state.

100x-18-b review of ml/neural.py — demonstrates (or refutes) suspected
defects BEFORE any code change. Run:

  LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 python scratch/probe_neural_1_earlystop.py
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from pharmaml.ml.neural import DeepMLPClassifier, DeepMLPRegressor

ENV = "LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"


def header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------- (1) best-weight restore
header("(1) best-weight restore: weights == epoch-of-best-val-loss, not last?")

X, y = make_classification(n_samples=150, n_features=10, n_informative=6, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# A diverging-ish schedule: big lr makes val loss dip early then worsen.
common = dict(hidden_layers=(16, 8), epochs=60, patience=60, val_split=0.2,
              random_state=7, learning_rate=0.05)
model_a = DeepMLPClassifier(**common).fit(X_tr, y_tr)
hist = model_a.history_
val_losses = [h["val_loss"] for h in hist]
best_epoch = int(np.argmin(val_losses)) + 1  # 1-based
print(f"epochs run={model_a.n_iter_}  best epoch={best_epoch}  "
      f"val[best]={val_losses[best_epoch - 1]:.4f}  val[last]={val_losses[-1]:.4f}")
print(f"val loss strictly worse at last epoch than best? {val_losses[-1] > val_losses[best_epoch - 1]}")

# Train a second model that stops exactly at the best epoch (identical
# deterministic loop up to that epoch) -> its final weights are the
# epoch-best weights.
model_b = DeepMLPClassifier(**{**common, "epochs": best_epoch, "patience": best_epoch}).fit(X_tr, y_tr)
same = all(
    np.array_equal(model_a._weights[k], model_b._weights[k]) for k in model_a._weights
)
print(f"stored weights == weights at best epoch {best_epoch}? {same}")
if not same:
    # Maybe they equal the LAST epoch's weights instead (the defect)?
    model_c = DeepMLPClassifier(**{**common, "epochs": model_a.n_iter_, "patience": model_a.n_iter_}).fit(X_tr, y_tr)
    same_last = all(
        np.array_equal(model_a._weights[k], model_c._weights[k]) for k in model_a._weights
    )
    print(f"stored weights == weights at LAST epoch {model_a.n_iter_}? {same_last}")
    print(">>> DEFECT if same_last and not same")

# n_iter_ semantics: with patience=1 and a worsening curve, stop soon after best.
model_p1 = DeepMLPClassifier(**{**common, "patience": 1}).fit(X_tr, y_tr)
print(f"patience=1 -> epochs run={model_p1.n_iter_} (best epoch was {best_epoch}); "
      f"n_iter_ == epochs run (documented): {model_p1.n_iter_ == len(model_p1.history_)}")

# val loss never improves after epoch 1 -> best snapshot must be epoch 1
never_improves = all(v >= val_losses[0] for v in val_losses[1:])
print(f"never-improves-after-epoch-1 case present in this run? {never_improves}")
if never_improves:
    model_d = DeepMLPClassifier(**{**common, "epochs": 1, "patience": 1}).fit(X_tr, y_tr)
    print("weights == epoch-1 weights?",
          all(np.array_equal(model_a._weights[k], model_d._weights[k]) for k in model_a._weights))

# ---------------------------------------------------------------- (2) NaN on extreme learning rate
header("(2) extreme learning_rate: silent NaN predictions?")

rng = np.random.default_rng(0)
Xr = rng.normal(size=(120, 6))
yr = 3.0 * Xr[:, 0] - 2.0 * Xr[:, 1] + 0.1 * rng.normal(size=120)

for lr in (1e3, 1e6, 1e9):
    reg = DeepMLPRegressor(hidden_layers=(16, 8), epochs=12, patience=12,
                            val_split=0.15, random_state=3, learning_rate=lr).fit(Xr, yr)
    preds = reg.predict(Xr[:5])
    weights_finite = all(np.isfinite(w).all() for w in reg._weights.values())
    hist_nan = any(
        (h["train_loss"] is not None and not np.isfinite(h["train_loss"]))
        or (h["val_loss"] is not None and not np.isfinite(h["val_loss"]))
        for h in reg.history_
    )
    print(f"lr={lr:>8}: predictions finite={np.isfinite(preds).all()}  "
          f"weights finite={weights_finite}  history has NaN/inf={hist_nan}  "
          f"preds[:3]={np.round(preds[:3], 3)}")
    if not np.isfinite(preds).all():
        print(f">>> DEFECT: fit() SUCCEEDED silently; predict returns NaN "
              f"(preds={preds[:3]}) — no error, no warning")

# ---------------------------------------------------------------- (3) refit state reset
header("(3) refit: history_ reset? state leaked?")

m = DeepMLPClassifier(hidden_layers=(8,), epochs=6, patience=6, val_split=0.2, random_state=5).fit(X_tr, y_tr)
print(f"first fit:  n_iter_={m.n_iter_}  len(history_)={len(m.history_)}")
m2 = DeepMLPClassifier(hidden_layers=(8,), epochs=3, patience=3, val_split=0.2, random_state=5)
m2.fit(X_tr, y_tr)
m2.fit(X_tr, y_tr)  # refit same instance
print(f"refit:      n_iter_={m2.n_iter_}  len(history_)={len(m2.history_)}  "
      f"(DEFECT if 6: history appended/leaked)")
proba = m2.predict_proba(X_te[:4])
print(f"post-refit predict_proba rows sum to 1: {np.allclose(proba.sum(axis=1), 1.0)}")
