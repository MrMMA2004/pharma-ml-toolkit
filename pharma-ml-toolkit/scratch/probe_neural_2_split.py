"""Probe 2: validation-split edges + standardization leakage.

100x-18-b review of ml/neural.py — verify-before-fix probe.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

from pharmaml.ml.neural import DeepMLPClassifier, DeepMLPRegressor

def header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------- (1) single-member class + big val_split
header("(1) stratified split: single-member class with val_split >= 0.5")

# class 0: 6 rows, class 1: 6 rows, class 2: exactly ONE row.
X0 = np.random.default_rng(0).normal(size=(13, 4))
y0 = np.array([0] * 6 + [1] * 6 + [2])

for val_split in (0.15, 0.5, 0.6):
    est = DeepMLPClassifier(hidden_layers=(4,), epochs=1, val_split=val_split, random_state=7)
    rng = np.random.default_rng(7)
    train_idx, val_idx = est._train_val_split(y0, len(y0), rng)
    train_classes = set(y0[train_idx].tolist())
    print(f"val_split={val_split}: train rows={len(train_idx)} val rows={len(val_idx)} "
          f"classes present in TRAIN={sorted(train_classes)}")
    if 2 not in train_classes:
        print(f">>> DEFECT: class 2 (single member) is MISSING from the training fold "
              f"— docstring promises 'every class keeps at least one training row'")

# what actually happens end-to-end: the class can never be learned/predicted
est = DeepMLPClassifier(hidden_layers=(4,), epochs=8, val_split=0.6, random_state=7).fit(X0, y0)
print(f"fitted with val_split=0.6: predictions={est.predict(X0)}")
print(f"class 2 ever predicted? {2 in set(est.predict(X0).tolist())}")

# ---------------------------------------------------------------- (2) tiny n regression split
header("(2) regression split with tiny n")

Xr = np.random.default_rng(1).normal(size=(2, 3))
yr = np.array([0.5, -0.5])
est = DeepMLPRegressor(hidden_layers=(4,), epochs=3, val_split=0.15, random_state=3)
rng = np.random.default_rng(3)
tr, va = est._train_val_split(yr, 2, rng)
print(f"n=2, val_split=0.15 -> train={tr.tolist()} val={va.tolist()}")
reg = DeepMLPRegressor(hidden_layers=(4,), epochs=3, val_split=0.15, random_state=3).fit(Xr, yr)
print(f"fit on n=2 works, history val_loss entries: {[h['val_loss'] is not None for h in reg.history_]}")

Xr1 = np.random.default_rng(2).normal(size=(1, 3))
yr1 = np.array([0.5])
reg1 = DeepMLPRegressor(hidden_layers=(4,), epochs=2, val_split=0.3, random_state=3).fit(Xr1, yr1)
print(f"n=1 fit works, val_loss=None (no split): {[h['val_loss'] for h in reg1.history_]}")

# ---------------------------------------------------------------- (3) standardization leakage
header("(3) standardization: computed on train-only or on ALL fit rows (incl. val)?")

# Crafted: an extreme outlier row; with val_split>0 the outlier should land
# in the VAL fold (rng-seeded); if _mean_ still moves toward the outlier,
# the scaler saw the val rows.
rng_data = np.random.default_rng(5)
X = rng_data.normal(size=(40, 3))
y = (X[:, 0] > 0).astype(int)
X[0, 0] = 500.0  # giant outlier -> hugely shifts column-0 mean if included

est = DeepMLPClassifier(hidden_layers=(4,), epochs=2, val_split=0.15, random_state=11).fit(X, y)
# replicate the split to know which rows were train vs val
rng2 = np.random.default_rng(11)
train_idx, val_idx = est._train_val_split(y, len(y), rng2)
outlier_in_val = 0 in set(val_idx.tolist())
full_mean = X[:, 0].mean()
train_mean = X[train_idx][:, 0].mean()
print(f"outlier row in VAL fold? {outlier_in_val}")
print(f"_mean_[0]={est._mean_[0]:.4f}")
print(f"mean over ALL rows  ={full_mean:.4f}  (includes val+outlier)")
print(f"mean over TRAIN rows={train_mean:.4f}")
match_full = np.isclose(est._mean_[0], full_mean)
match_train = np.isclose(est._mean_[0], train_mean)
print(f"_mean_ == FULL-data mean (leak)? {match_full}   _mean_ == train-only mean? {match_train}")
if match_full and outlier_in_val:
    print(">>> DEFECT: comment says 'fitted on the training block only' but the "
          "validation rows (incl. the outlier) leaked into mean/std — the "
          "early-stopping metric is computed on rows the scaler saw")
