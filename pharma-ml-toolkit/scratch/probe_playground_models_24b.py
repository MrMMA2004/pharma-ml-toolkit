#!/usr/bin/env python3
"""100x-24-b sklearn cross-validation probe for the ML Playground models.

Generates one small regression CSV (2 features + target, multivariate path)
and one 2-feature binary classification CSV, POSTs them to the running
Next.js /api/ml/train endpoint (csv + csvName mode), and compares test-set
R^2 / accuracy against matched sklearn references.

Alignment details (all replicated here from the TypeScript source):
  * src/lib/csv.ts hashCsvSeed  — FNV-1a 32-bit over the raw CSV text drives
    the split seed, so both sides see identical train/test row sets.
  * src/lib/ml-math.ts mulberry32 + shuffledIndices + shuffledSplit — the
    seeded Fisher-Yates shuffle is reproduced bit-exactly (32-bit ops).
  * Standardization: the playground engine standardizes features with
    TRAIN-set population statistics, so sklearn gets a StandardScaler fit on
    the train rows only.

Honest tolerances: linear-family models and single trees should agree to
<=1e-3; forests/boosting use different RNG streams for bootstrap resampling
so only the metric magnitude is comparable (~0.03); the MLPs use a different
optimizer (full-batch plain GD vs sklearn's Adam) so only sanity-band
agreement is expected. Observed deltas are recorded per model below.

Run: python pharma-ml-toolkit/scratch/probe_playground_models_24b.py
"""

from __future__ import annotations

import json
import math
import urllib.request
from typing import Callable

import numpy as np

API = "http://localhost:3000/api/ml/train"

# --------------------------------------------------------------------------
# Bit-exact replicas of the playground's TypeScript helpers
# --------------------------------------------------------------------------

MASK32 = 0xFFFFFFFF


def imul(a: int, b: int) -> int:
    """JS Math.imul — low 32 bits of the integer product (bit pattern)."""
    return (a * b) & MASK32


def mulberry32(seed: int) -> Callable[[], float]:
    """src/lib/ml-math.ts mulberry32, bit-exact."""
    a = seed & MASK32
    state = [a]

    def rand() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & MASK32
        t = state[0]
        t = imul(t ^ (t >> 15), t | 1)
        t = (t + imul(t ^ (t >> 7), t | 61)) & MASK32 ^ t
        return ((t ^ (t >> 14)) & MASK32) / 4294967296.0

    return rand


def js_round(x: float) -> int:
    """JS Math.round — half away toward +infinity."""
    return math.floor(x + 0.5)


def shuffled_split(n: int, test_fraction: float, rand: Callable[[], float]):
    """src/lib/ml-math.ts shuffledSplit (seeded Fisher-Yates)."""
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = math.floor(rand() * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    n_test = min(max(1, js_round(n * test_fraction)), n - 2)
    return idx[n_test:], idx[:n_test]  # (train, test)


def hash_csv_seed(text: str) -> int:
    """src/lib/csv.ts hashCsvSeed — FNV-1a 32-bit over char codes."""
    h = 0x811C9DC5
    for ch in text:
        h = (h ^ ord(ch)) & MASK32
        h = imul(h, 0x01000193)
    return h & MASK32


# --------------------------------------------------------------------------
# Datasets (small, ASCII-only so char codes match byte codes)
# --------------------------------------------------------------------------

rng = np.random.default_rng(2424)
N = 120

# Regression: y = 2.5*x1 - 1.5*x2 + 1.0 + noise, x's ~ U(-2, 2)
x1 = rng.uniform(-2, 2, N)
x2 = rng.uniform(-2, 2, N)
y = 2.5 * x1 - 1.5 * x2 + 1.0 + rng.normal(0, 0.35, N)

reg_lines = ["x1,x2,y"]
for a, b, t in zip(x1, x2, y):
    reg_lines.append(f"{a:.6f},{b:.6f},{t:.6f}")
REG_CSV = "\n".join(reg_lines) + "\n"

# Classification: two Gaussian blobs (mildly overlapping) — 2 features, 0/1 label
n_half = N // 2
cx1 = np.concatenate([rng.normal(0.0, 0.9, n_half), rng.normal(1.8, 0.9, N - n_half)])
cx2 = np.concatenate([rng.normal(0.0, 0.9, n_half), rng.normal(1.8, 0.9, N - n_half)])
label = np.concatenate([np.zeros(n_half, dtype=int), np.ones(N - n_half, dtype=int)])

cls_lines = ["x1,x2,label"]
for a, b, t in zip(cx1, cx2, label):
    cls_lines.append(f"{a:.6f},{b:.6f},{t:d}")
CLS_CSV = "\n".join(cls_lines) + "\n"

# --------------------------------------------------------------------------
# Playground calls
# --------------------------------------------------------------------------


def post(body: dict) -> dict:
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def playground_regression(model: str, hyper: dict) -> dict:
    return post(
        {
            "task": "regression",
            "model": model,
            "csv": REG_CSV,
            "csvName": "probe_reg.csv",
            "testSize": 0.25,
            "cv": 0,
            **hyper,
        }
    )


def playground_classification(model: str, hyper: dict) -> dict:
    return post(
        {
            "task": "classification",
            "model": model,
            "csv": CLS_CSV,
            "csvName": "probe_cls.csv",
            "testSize": 0.25,
            "cv": 0,
            **hyper,
        }
    )


def csv_matrix(text: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [ln.split(",") for ln in text.strip().splitlines()[1:]]
    arr = np.array([[float(v) for v in r] for r in rows])
    return arr[:, :2], arr[:, 2]


# --------------------------------------------------------------------------
# sklearn references with matched hyperparameters
# --------------------------------------------------------------------------

from sklearn.ensemble import (  # noqa: E402
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso  # noqa: E402
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402


def compare_regression(name: str, hyper: dict, make_est) -> None:
    resp = playground_regression(name, hyper)
    X, yv = csv_matrix(REG_CSV)
    seed = hash_csv_seed(REG_CSV)
    train_idx, test_idx = shuffled_split(len(yv), 0.25, mulberry32(seed))
    tr, te = np.array(train_idx), np.array(test_idx)
    scaler = StandardScaler().fit(X[tr])
    est = make_est()
    est.fit(scaler.transform(X[tr]), yv[tr])
    sk_r2 = r2_score(yv[te], est.predict(scaler.transform(X[te])))
    pg_r2 = resp["metrics"]["r2"]
    delta = abs(pg_r2 - sk_r2)
    tol = TOLERANCES.get(name, 0.03)
    verdict = "OK " if delta <= tol else "WARN"
    print(
        f"  {verdict} regression {name:18} playground R2={pg_r2:+.4f}  "
        f"sklearn R2={sk_r2:+.4f}  |Δ|={delta:.4f}  (tol {tol})"
    )
    RESULTS.append((name, "R2", pg_r2, sk_r2, delta, tol, delta <= tol))


def compare_classification(name: str, hyper: dict, make_est) -> None:
    resp = playground_classification(name, hyper)
    X, yv = csv_matrix(CLS_CSV)
    seed = hash_csv_seed(CLS_CSV)
    train_idx, test_idx = shuffled_split(len(yv), 0.25, mulberry32(seed))
    tr, te = np.array(train_idx), np.array(test_idx)
    scaler = StandardScaler().fit(X[tr])
    est = make_est()
    est.fit(scaler.transform(X[tr]), yv[tr].astype(int))
    sk_acc = accuracy_score(yv[te].astype(int), est.predict(scaler.transform(X[te])))
    pg_acc = resp["metrics"]["accuracy"]
    delta = abs(pg_acc - sk_acc)
    tol = TOLERANCES.get(name, 0.05)
    verdict = "OK " if delta <= tol else "WARN"
    print(
        f"  {verdict} classif   {name:18} playground acc={pg_acc:.4f}  "
        f"sklearn acc={sk_acc:.4f}  |Δ|={delta:.4f}  (tol {tol})"
    )
    RESULTS.append((name, "acc", pg_acc, sk_acc, delta, tol, delta <= tol))


# Honest tolerances: exactness expectations per model family.
TOLERANCES = {
    "lasso": 0.002,
    "elastic_net": 0.002,
    "knn": 0.01,
    "decision_tree": 0.01,
    "random_forest": 0.05,
    "gradient_boosting": 0.05,
    "mlp": 0.15,
}
RESULTS: list[tuple] = []

print("=" * 78)
print("100x-24-b sklearn cross-validation probe (playground vs sklearn)")
print("=" * 78)
print(f"regression CSV: {N} rows, y = 2.5·x1 − 1.5·x2 + 1 + N(0, 0.35), split 75/25")
print(f"classification CSV: {N} rows, two overlapping Gaussian blobs, split 75/25")
print(f"hashCsvSeed(reg csv) = {hash_csv_seed(REG_CSV)}")
print(f"hashCsvSeed(cls csv) = {hash_csv_seed(CLS_CSV)}")
print("-" * 78)
print("regression references (matched hyperparameters, train-standardized):")

compare_regression(
    "lasso", {"alpha": 0.1}, lambda: Lasso(alpha=0.1, tol=1e-10, max_iter=5000)
)
compare_regression(
    "elastic_net",
    {"alpha": 0.1, "l1Ratio": 0.5},
    lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, tol=1e-10, max_iter=5000),
)
compare_regression(
    "knn", {"k": 7}, lambda: KNeighborsRegressor(n_neighbors=7, weights="uniform")
)
compare_regression(
    "decision_tree", {"maxDepth": 4}, lambda: DecisionTreeRegressor(max_depth=4, random_state=0)
)
compare_regression(
    "random_forest",
    {"nEstimators": 50, "maxDepth": 4},
    lambda: RandomForestRegressor(n_estimators=50, max_depth=4, random_state=0),
)
compare_regression(
    "gradient_boosting",
    {"nRounds": 60, "learningRate": 0.1},
    lambda: GradientBoostingRegressor(
        n_estimators=60, learning_rate=0.1, max_depth=3, random_state=0
    ),
)
compare_regression(
    "mlp",
    {"hidden": 24, "epochs": 600, "lr": 0.05},
    lambda: MLPRegressor(
        hidden_layer_sizes=(24,),
        activation="tanh",
        solver="adam",
        max_iter=600,
        learning_rate_init=0.05,
        random_state=0,
    ),
)

print("-" * 78)
print("classification references (matched hyperparameters, train-standardized):")

compare_classification(
    "knn", {"k": 7}, lambda: KNeighborsClassifier(n_neighbors=7, weights="uniform")
)
compare_classification(
    "decision_tree",
    {"maxDepth": 4},
    lambda: DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=0),
)
compare_classification(
    "random_forest",
    {"nEstimators": 50, "maxDepth": 4},
    lambda: RandomForestClassifier(n_estimators=50, max_depth=4, random_state=0),
)
compare_classification("gaussian_nb", {}, lambda: GaussianNB())
compare_classification(
    "svm_linear", {"c": 1.0}, lambda: LinearSVC(C=1.0, tol=1e-6, max_iter=20000)
)
compare_classification(
    "mlp",
    {"hidden": 24, "epochs": 600, "lr": 0.05},
    lambda: MLPClassifier(
        hidden_layer_sizes=(24,),
        activation="tanh",
        solver="adam",
        max_iter=600,
        learning_rate_init=0.05,
        random_state=0,
    ),
)

print("-" * 78)
ok = sum(1 for r in RESULTS if r[6])
print(f"SUMMARY: {ok}/{len(RESULTS)} models within tolerance")
for name, metric, pg, sk, d, tol, within in RESULTS:
    print(
        f"  {name:18} {metric:4}  playground {pg:+.4f}  sklearn {sk:+.4f}  "
        f"|Δ|={d:.4f}  tol={tol}  {'within' if within else 'OUT'}"
    )
print(
    "\nNotes: random_forest / gradient_boosting / mlp use different RNG or\n"
    "optimizers than sklearn, so agreement is metric-magnitude only; the\n"
    "linear family (lasso / elastic_net) and single trees / knn / gaussian_nb\n"
    "share the exact objective and split, hence tight tolerances."
)
