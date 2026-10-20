"""Cross-validation probe: ML Playground model library vs scikit-learn.

Task 26-R6 artifact. Rebuild-era probe (successor of the lost
probe_playground_models_24b.py): it replicates the playground's seeded
pipeline bit-for-bit in Python — mulberry32, the Box–Muller gaussian, the
synthetic generators (makeRegressionData / makeMoons), the Fisher–Yates
shuffledSplit, and the FNV-1a hashCsvSeed — then trains the scikit-learn
equivalent of every playground model on the *same* train/test split the API
used, and compares the headline metrics.

Run (dev server on :3000 must be up):

    python pharma-ml-toolkit/scratch/probe_playground_models_26.py

Agreement expectations (documented, not asserted):
  ~exact   lasso, elastic_net, knn (reg+cls), decision_tree (reg+cls),
           gaussian_nb, svm_linear — same deterministic algorithm /
           strongly-convex optimum, only optimizer-rounding differences.
  tolerance random_forest, gradient_boosting, mlp — the TS and sklearn
           implementations draw from different RNG streams (bagging / init)
           and use different optimizers, so metrics agree only to tolerance.

This file is a documentation/probe artifact only — no toolkit source, tests
or README depend on it.
"""

from __future__ import annotations

import json
import math
import urllib.request
from typing import Callable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

API = "http://127.0.0.1:3000/api/ml/train"

# --------------------------------------------------------------------- #
# Bit-exact replicas of src/lib/ml-math.ts + src/lib/csv.ts             #
# --------------------------------------------------------------------- #


def mulberry32(seed: int) -> Callable[[], float]:
    """Deterministic 32-bit PRNG returning floats in [0, 1) — same bit
    pattern arithmetic as the TypeScript original (all ops mod 2^32)."""
    a = seed & 0xFFFFFFFF

    def rand() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rand


def gaussian(rand: Callable[[], float]) -> float:
    """Box–Muller standard normal, consuming two uniforms (TS consumption
    order: reject zeros on u first, then v)."""
    u = 0.0
    v = 0.0
    while u == 0:
        u = rand()
    while v == 0:
        v = rand()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def js_round(x: float) -> int:
    """JavaScript Math.round (half away from −inf → toward +inf)."""
    return math.floor(x + 0.5)


def shuffled_indices(n: int, rand: Callable[[], float]) -> List[int]:
    """Seeded Fisher–Yates shuffle of [0, n)."""
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rand() * (i + 1))  # Math.floor
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def shuffled_split(n: int, test_fraction: float, rand: Callable[[], float]):
    """The API's train/test split (test rows first in the shuffled order)."""
    idx = shuffled_indices(n, rand)
    n_test = min(max(1, js_round(n * test_fraction)), n - 2)
    return idx[:n_test], idx[n_test:]


def hash_csv_seed(text: str) -> int:
    """FNV-1a string hash → uint32 (deterministic split seed for CSV mode).
    JS charCodeAt is a UTF-16 code unit — identical to ord() for the BMP
    characters a CSV carries."""
    h = 0x811C9DC5
    for ch in text:
        h = (h ^ ord(ch)) & 0xFFFFFFFF
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def make_regression_data(n: int, noise: float, rand: Callable[[], float]):
    """y = 0.8x + 0.5 + noise·N(0,1), x ~ U(-3, 3) — draw order: x, y."""
    xs: List[float] = []
    ys: List[float] = []
    for _ in range(n):
        x = rand() * 6 - 3
        y = 0.8 * x + 0.5 + noise * gaussian(rand)
        xs.append(x)
        ys.append(y)
    return xs, ys


def make_moons(n: int, noise: float, rand: Callable[[], float]):
    """Two-moons generator: outer arc first, inner arc second, then noise
    added point-by-point (x draw before y draw)."""
    n_out = n // 2
    n_in = n - n_out
    pts: List[Tuple[float, float, int]] = []
    for i in range(n_out):
        t = math.pi * i / max(n_out - 1, 1)
        pts.append((math.cos(t), math.sin(t), 0))
    for i in range(n_in):
        t = math.pi * i / max(n_in - 1, 1)
        pts.append((1 - math.cos(t), 0.5 - math.sin(t), 1))
    if noise > 0:
        pts = [
            (x + noise * gaussian(rand), y + noise * gaussian(rand), label)
            for (x, y, label) in pts
        ]
    return pts


# --------------------------------------------------------------------- #
# Metric replicas (src/lib/ml-math.ts)                                  #
# --------------------------------------------------------------------- #


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    n = len(y_true)
    se = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ae = sum(abs(t - p) for t, p in zip(y_true, y_pred))
    ym = sum(y_true) / n
    sst = sum((t - ym) ** 2 for t in y_true)
    return {
        "rmse": math.sqrt(se / n),
        "mae": ae / n,
        "r2": (1 - se / sst) if sst > 1e-12 else 0.0,
    }


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return {"accuracy": (tp + tn) / len(y_true), "precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------- #
# API plumbing                                                          #
# --------------------------------------------------------------------- #


def call_api(body: dict) -> dict:
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.loads(res.read().decode())


def std_pipeline(estimator):
    """StandardScaler + estimator — the TS models standardize internally
    with train-set population statistics (StandardScaler uses ddof=0)."""
    return make_pipeline(StandardScaler(), estimator)


# --------------------------------------------------------------------- #
# Probe cases                                                           #
# --------------------------------------------------------------------- #

BASE = {
    "samples": 300,
    "noise": 0.4,
    "testSize": 0.2,
    "cv": 0,
    "seed": 42,
}


def regression_cases() -> List[Tuple[str, dict, Callable[[], object]]]:
    """(label, api-extra-body, sklearn-factory) for the regression library."""
    return [
        ("lasso", {"alpha": 1.0}, lambda: std_pipeline(Lasso(alpha=1.0, max_iter=50000, tol=1e-12))),
        (
            "elastic_net",
            {"alpha": 1.0, "l1Ratio": 0.5},
            lambda: std_pipeline(ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=50000, tol=1e-12)),
        ),
        ("knn", {"k": 5}, lambda: std_pipeline(KNeighborsRegressor(n_neighbors=5))),
        (
            "decision_tree",
            {"maxDepth": 4},
            lambda: std_pipeline(
                DecisionTreeRegressor(criterion="squared_error", max_depth=4, random_state=42)
            ),
        ),
        (
            "random_forest",
            {"nEstimators": 50, "maxDepth": 6},
            lambda: std_pipeline(
                RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
            ),
        ),
        (
            "gradient_boosting",
            {"rounds": 50, "learningRate": 0.1},
            lambda: GradientBoostingRegressor(
                n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42
            ),
        ),
        (
            "mlp",
            {"hidden": 16, "epochs": 200, "lr": 0.05},
            lambda: std_pipeline(
                MLPRegressor(
                    hidden_layer_sizes=(16,),
                    activation="tanh",
                    solver="sgd",
                    learning_rate_init=0.05,
                    max_iter=200,
                    batch_size=240,
                    momentum=0.0,
                    nesterovs_momentum=False,
                    early_stopping=False,
                    shuffle=False,
                )
            ),
        ),
    ]


def classification_cases() -> List[Tuple[str, dict, Callable[[], object]]]:
    """(label, api-extra-body, sklearn-factory) for the classification library."""
    return [
        ("knn", {"k": 5}, lambda: std_pipeline(KNeighborsClassifier(n_neighbors=5))),
        (
            "decision_tree",
            {"maxDepth": 4},
            lambda: std_pipeline(
                DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
            ),
        ),
        (
            "random_forest",
            {"nEstimators": 50, "maxDepth": 6},
            lambda: std_pipeline(
                RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
            ),
        ),
        ("gaussian_nb", {}, lambda: std_pipeline(GaussianNB())),
        (
            "svm_linear",
            {"C": 1.0},
            lambda: std_pipeline(
                LinearSVC(C=1.0, loss="squared_hinge", penalty="l2", dual=False, tol=1e-9, max_iter=200000)
            ),
        ),
        (
            "mlp",
            {"hidden": 16, "epochs": 200, "lr": 0.05},
            lambda: std_pipeline(
                MLPClassifier(
                    hidden_layer_sizes=(16,),
                    activation="tanh",
                    solver="sgd",
                    learning_rate_init=0.05,
                    max_iter=200,
                    batch_size=240,
                    momentum=0.0,
                    nesterovs_momentum=False,
                    early_stopping=False,
                    shuffle=False,
                )
            ),
        ),
    ]


# Exact-agreement ceiling per model (metrics are O(1) quantities).
EXACT_TOL = 1e-6
TOL_TOLERANCE = {
    "random_forest": 0.05,
    "gradient_boosting": 0.08,
    "mlp": 0.10,
    "svm_linear": 0.02,  # GD vs liblinear both converge to the same optimum
}


def compare(model: str, api: dict, ours: dict, keys: Sequence[str]) -> Tuple[str, float, str]:
    worst = max(abs(api[k] - ours[k]) for k in keys)
    tol = TOL_TOLERANCE.get(model, EXACT_TOL)
    status = "EXACT" if worst <= EXACT_TOL else ("within-tol" if worst <= tol else "DIFF")
    return status, worst, tol


def main() -> None:
    rows: List[Tuple[str, str, float, float, float, str]] = []

    # ---------------- synthetic regression ---------------- #
    gen = mulberry32(BASE["seed"])
    xs, ys = make_regression_data(BASE["samples"], BASE["noise"], gen)
    split_rand = mulberry32(BASE["seed"])  # fresh stream, as in the route
    test_idx, train_idx = shuffled_split(len(xs), BASE["testSize"], split_rand)
    x_train = np.array([[xs[i]] for i in train_idx])
    y_train = np.array([ys[i] for i in train_idx])
    x_test = np.array([[xs[i]] for i in test_idx])
    y_test = np.array([ys[i] for i in test_idx])

    print("=" * 96)
    print("REGRESSION — synthetic data, seed 42, n=300, noise 0.4, test 20%")
    print("=" * 96)
    for label, extra, factory in regression_cases():
        api = call_api({"task": "regression", "model": label, **BASE, **extra})
        sk = factory()
        sk.fit(x_train, y_train)
        ours = regression_metrics(y_test, sk.predict(x_test))
        status, worst, tol = compare(label, api["metrics"], ours, ["r2", "rmse", "mae"])
        rows.append((label, "regression", api["metrics"]["r2"], ours["r2"], worst, status))
        print(
            f"{label:20s} API r2={api['metrics']['r2']:+.6f}  sklearn r2={ours['r2']:+.6f}"
            f"  Δrmse={abs(api['metrics']['rmse'] - ours['rmse']):.2e}  "
            f"[{status} ≤{tol:g}]"
        )

    # ---------------- synthetic classification ---------------- #
    gen = mulberry32(BASE["seed"])
    pts = make_moons(BASE["samples"], BASE["noise"], gen)
    split_rand = mulberry32(BASE["seed"])
    test_idx, train_idx = shuffled_split(len(pts), BASE["testSize"], split_rand)
    x_train = np.array([[pts[i][0], pts[i][1]] for i in train_idx])
    y_train = np.array([pts[i][2] for i in train_idx])
    x_test = np.array([[pts[i][0], pts[i][1]] for i in test_idx])
    y_test = np.array([pts[i][2] for i in test_idx])

    print()
    print("=" * 96)
    print("CLASSIFICATION — two moons, seed 42, n=300, noise 0.4, test 20%")
    print("=" * 96)
    for label, extra, factory in classification_cases():
        api = call_api({"task": "classification", "model": label, **BASE, **extra})
        sk = factory()
        sk.fit(x_train, y_train)
        ours = classification_metrics(y_test, sk.predict(x_test))
        status, worst, tol = compare(label, api["metrics"], ours, ["accuracy", "f1"])
        rows.append((label, "classification", api["metrics"]["accuracy"], ours["accuracy"], worst, status))
        print(
            f"{label:20s} API acc={api['metrics']['accuracy']:.6f}  sklearn acc={ours['accuracy']:.6f}"
            f"  Δf1={abs(api['metrics']['f1'] - ours['f1']):.2e}  "
            f"[{status} ≤{tol:g}]"
        )

    # ---------------- CSV mode: split seed via hashCsvSeed ---------------- #
    print()
    print("=" * 96)
    print("CSV MODE — pasted CSV, split seed = hashCsvSeed(text) replica")
    print("=" * 96)
    rng = mulberry32(1234)
    csv_lines = ["dose,response"]
    for _ in range(120):
        d = rng() * 10
        r = 0.5 * d + 2.0 + 0.3 * gaussian(rng)
        csv_lines.append(f"{d:.6f},{r:.6f}")
    csv_text = "\n".join(csv_lines) + "\n"

    seed = hash_csv_seed(csv_text)
    split_rand = mulberry32(seed)
    test_idx, train_idx = shuffled_split(120, 0.2, split_rand)
    api = call_api(
        {
            "task": "regression",
            "model": "knn",
            "k": 5,
            "testSize": 0.2,
            "cv": 0,
            "csv": csv_text,
            "csvName": "probe.csv",
        }
    )
    # The route z-scores the feature on ALL rows before training, so the
    # API's test x values are global z-scores — compare against the same
    # transform of the replicated split's raw doses.
    zx_all = np.array([[float(line.split(",")[0])] for line in csv_lines[1:]])
    zy_all = np.array([float(line.split(",")[1]) for line in csv_lines[1:]])
    mu = float(zx_all.mean())
    sigma = float(zx_all.std())  # population σ, as zscore()
    zx = (zx_all - mu) / sigma
    api_test_x = sorted(round(p["x"], 6) for p in api["series"]["test"])
    replica_x = sorted(round(float(zx[i][0]), 6) for i in test_idx)
    split_match = api_test_x == replica_x
    print(
        f"hashCsvSeed replica: 0x{seed:08X}; API test rows = replicated shuffledSplit"
        f" rows → {'MATCH' if split_match else 'MISMATCH'}"
    )
    # sklearn agreement on the same split: the route z-scores the feature
    # globally, then the model re-standardizes on train rows —
    # StandardScaler on the train rows of the global z-scores is the same
    # composition of affine maps.
    x_train = zx[train_idx]
    y_train = zy_all[train_idx]
    x_test = zx[test_idx]
    y_test = zy_all[test_idx]
    sk = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    sk.fit(x_train, y_train)
    ours = regression_metrics(y_test, sk.predict(x_test))
    status, worst, _ = compare("knn", api["metrics"], ours, ["r2", "rmse", "mae"])
    print(
        f"knn (csv)            API r2={api['metrics']['r2']:+.6f}  sklearn r2={ours['r2']:+.6f}"
        f"  Δ={worst:.2e}  [{status}]"
    )

    # ---------------- summary ---------------- #
    print()
    print("=" * 96)
    print("AGREEMENT SUMMARY")
    print("=" * 96)
    n_exact = sum(1 for r in rows if r[5] == "EXACT")
    n_tol = sum(1 for r in rows if r[5] == "within-tol")
    n_diff = sum(1 for r in rows if r[5] == "DIFF")
    for label, task, api_m, sk_m, worst, status in rows:
        print(f"  {task:16s} {label:20s} Δ={worst:.2e}  {status}")
    print()
    print(
        f"  {n_exact} exact (≤{EXACT_TOL:g}) · {n_tol} within tolerance · {n_diff} DIFF"
    )
    print(
        "  Expected: lasso/elastic_net/knn/decision_tree/gaussian_nb/svm_linear ~exact;"
        " random_forest/gradient_boosting/mlp within tolerance (RNG + optimizer differences)."
    )


if __name__ == "__main__":
    main()
