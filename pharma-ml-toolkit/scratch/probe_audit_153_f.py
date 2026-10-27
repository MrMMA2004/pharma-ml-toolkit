"""Probe F+G: evaluate_model's train side + cross_model_importances seed discipline.

Round-36 audit (100x-153). Deterministic (seed everything), no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from pharmaml.exceptions import ModelError
from pharmaml.ml.evaluation import cross_model_importances, evaluate_model

rng = np.random.default_rng(42)
x = pd.DataFrame(rng.uniform(size=(60, 3)), columns=["a", "b", "c"])
y = pd.Series(3.0 * x["a"] + rng.normal(0.0, 0.05, 60))
model = LinearRegression().fit(x, y)

# ---------------------------------------------------------------- F1: full train side
rep = evaluate_model(
    model, x.iloc[:20], y.iloc[:20], "regression",
    X_train=x.iloc[20:], y_train=y.iloc[20:],
)
print("F1 train_rmse/r2/pearson:", rep.train_rmse, round(rep.train_r2, 4), rep.train_pearson_r)
print("F1 r2_gap:", round(rep.r2_gap, 4), "== train_r2 - r2:",
      abs(rep.r2_gap - (rep.train_r2 - rep.r2)) < 1e-12)
d = rep.as_dict()
print("F1 as_dict train keys:", sorted(k for k in d if "train" in k or "gap" in k))

# ---------------------------------------------------------------- F2: absent train side
rep2 = evaluate_model(model, x.iloc[:20], y.iloc[:20], "regression")
d2 = rep2.as_dict()
print("F2 absent: no train keys:", not any("train" in k for k in d2), "| no gap:", "r2_gap" not in d2)
print("F2 train fields None:", rep2.train_rmse is None and rep2.train_r2 is None)
print("F2 r2_gap None:", rep2.r2_gap is None)
print("F2 str has no train line:", "train_r2" not in str(rep2))

# ---------------------------------------------------------------- F3: half pairs
for kwargs in ({"X_train": x}, {"y_train": y}):
    try:
        evaluate_model(model, x.iloc[:20], y.iloc[:20], "regression", **kwargs)  # type: ignore[arg-type]
        print("F3 FAIL half pair accepted")
    except ModelError as exc:
        print("F3 half pair typed:", str(exc)[:80])

# ---------------------------------------------------------------- F4: classification rejection
clf_y = (y > y.median()).astype(int)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(x, clf_y)
try:
    evaluate_model(clf, x.iloc[:20], clf_y.iloc[:20], "classification",
                   X_train=x.iloc[20:], y_train=clf_y.iloc[20:])
    print("F4 FAIL classification train side accepted")
except ModelError as exc:
    print("F4 classification typed:", str(exc)[:100])
# and the HALF pair under classification: which error fires?
try:
    evaluate_model(clf, x.iloc[:20], clf_y.iloc[:20], "classification", X_train=x.iloc[20:])
    print("F4b FAIL")
except ModelError as exc:
    print("F4b classification half-pair typed:", str(exc)[:80])

# ---------------------------------------------------------------- F5: length mismatch on train side
try:
    evaluate_model(model, x.iloc[:20], y.iloc[:20], "regression",
                   X_train=x.iloc[20:], y_train=y.iloc[30:])
    print("F5 FAIL mismatch accepted")
except ModelError as exc:
    print("F5 mismatch typed:", str(exc)[:90])

# ---------------------------------------------------------------- F6: NaN / inf train targets
y_bad = y.copy()
y_bad.iloc[25] = np.nan
try:
    evaluate_model(model, x.iloc[:20], y.iloc[:20], "regression",
                   X_train=x.iloc[20:], y_train=y_bad.iloc[20:])
    print("F6 FAIL NaN y_train accepted")
except ModelError as exc:
    print("F6 NaN y_train typed:", str(exc)[:90])
y_inf = y.copy()
y_inf.iloc[25] = np.inf
try:
    evaluate_model(model, x.iloc[:20], y.iloc[:20], "regression",
                   X_train=x.iloc[20:], y_train=y_inf.iloc[20:])
    print("F6b FAIL inf y_train accepted")
except ModelError as exc:
    print("F6b inf y_train typed:", str(exc)[:90])

# ---------------------------------------------------------------- F7: NaN train PREDICTIONS
class NanPredictor(LinearRegression):
    def predict(self, X):  # noqa: N802
        out = super().predict(X)
        out = np.array(out, dtype=float, copy=True)
        out[0] = np.nan
        return out

nan_model = NanPredictor().fit(x, y)
try:
    evaluate_model(nan_model, x.iloc[:20], y.iloc[:20], "regression",
                   X_train=x.iloc[20:], y_train=y.iloc[20:])
    print("F7 FAIL NaN train preds accepted")
except ModelError as exc:
    print("F7 NaN train preds typed:", str(exc)[:90])

# ---------------------------------------------------------------- F8: constant train preds -> pearson None
class ConstPredictor:
    def predict(self, X):  # noqa: N802
        return np.full(len(X), 5.0)

rep8 = evaluate_model(
    ConstPredictor(), x.iloc[:20], y.iloc[:20], "regression",
    X_train=x.iloc[20:], y_train=y.iloc[20:],
)
print("F8 train_pearson_r None:", rep8.train_pearson_r is None)
d8 = rep8.as_dict()
print("F8 as_dict carries train_pearson_r=None explicitly:", "train_pearson_r" in d8, d8.get("train_pearson_r"))

# ---------------------------------------------------------------- F9: constant test target -> pearson None, dict omits
rep9 = evaluate_model(
    ConstPredictor(), x.iloc[:20], pd.Series([3.0] * 20), "regression",
    X_train=x.iloc[20:], y_train=y.iloc[20:],
)
print("F9 pearson_r None (constant y_test):", rep9.pearson_r is None)
d9 = rep9.as_dict()
print("F9 pearson_r omitted from dict:", "pearson_r" not in d9,
      "| train keys still present:", "train_r2" in d9)

# ---------------------------------------------------------------- F10: as_dict gate asymmetry (hand-built)
from pharmaml.ml.evaluation import RegressionReport

hand = RegressionReport(rmse=1.0, mae=1.0, r2=0.5, max_error=2.0, n_samples=10, train_r2=0.9)
print("F10 hand-built r2_gap property:", hand.r2_gap)
print("F10 hand-built as_dict keys:", sorted(hand.as_dict()))

# ---------------------------------------------------------------- G1: same-seed shuffle alignment
class Recorder:
    """Records every X matrix predict() sees (the shuffled columns)."""

    def __init__(self, value):
        self.value = value
        self.seen: list[np.ndarray] = []

    def predict(self, X):  # noqa: N802
        arr = np.asarray(X)
        self.seen.append(arr.copy())
        return np.full(len(arr), self.value, dtype=float)

r_a = Recorder(1.0)
r_b = Recorder(2.0)
xx = rng.uniform(size=(30, 4))
yy = np.arange(30, dtype=float)
cross_model_importances({"a": r_a, "b": r_b}, xx, yy, n_repeats=3)
seq_a = [m[:, i] for m in r_a.seen for i in range(m.shape[1])]
seq_b = [m[:, i] for m in r_b.seen for i in range(m.shape[1])]
aligned = all(np.array_equal(u, v) for u, v in zip(seq_a, seq_b))
print("G1 same-seed shuffles aligned across models:", aligned, len(seq_a), "column shuffles")

# ---------------------------------------------------------------- G2: determinism
panel = {"ols": LinearRegression().fit(x, y), "ridge": Ridge(alpha=0.5).fit(x, y)}
one = cross_model_importances(panel, x, y)
two = cross_model_importances(panel, x, y)
print("G2 deterministic:", all(np.array_equal(one[k].means, two[k].means) for k in panel))

# ---------------------------------------------------------------- G3: failing model NAMED
class Broken:
    def predict(self, X):  # noqa: N802
        raise RuntimeError("boom inside predict")

try:
    cross_model_importances({"good": model, "the_bad_one": Broken()}, x, y)
    print("G3 FAIL broken model accepted")
except ModelError as exc:
    print("G3 broken model typed:", str(exc)[:110])

# ---------------------------------------------------------------- G4: empty + non-str keys
try:
    cross_model_importances({}, x, y)
    print("G4 FAIL empty accepted")
except ModelError as exc:
    print("G4 empty typed:", str(exc)[:80])
recs = cross_model_importances({42: model}, x, y)
print("G4 non-str key stringified:", list(recs), type(list(recs)[0]))

# ---------------------------------------------------------------- G5: classification default scoring
from sklearn.linear_model import LogisticRegression

clf2 = LogisticRegression().fit(x, clf_y)
recs5 = cross_model_importances({"clf": clf2}, x, clf_y, task="classification")
print("G5 classification scoring:", recs5["clf"].scoring)

print("PROBE_F_G_DONE")
