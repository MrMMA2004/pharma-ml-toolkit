#!/usr/bin/env python3
"""Stdin -> stdout JSON bridge for the Next.js ``/api/ml/shap`` route.

Runs inside the toolkit virtualenv (real pharmaml + shap) and wires ONLY
public ``pharmaml.ml`` APIs:

    build_model (ml/models.py) -> fit -> shap_summary (ml/explain.py)

This is the web twin's compact read of the desktop GUI's fourth button row
(round 37's SHAP surface): the full beeswarm needs a canvas, but the
mean-|SHAP| ranking is pure numbers — the route computes them here and the
card renders them in TypeScript. The estimator is the toolkit's own Python
twin (the registry's tree family, ``random_state=42`` threaded in by the
factory), fitted on the SAME rows and split the playground's TypeScript run
used (the route re-derives them with the same seeded generators and sends
them here), so the attribution names the very rows the run's parity chart
drew — the same discipline as the applicability-domain route.

Request (stdin, JSON)::

    {
      "task": "regression" | "classification",
      "model": "gradient_boosting" | "random_forest" | "extra_trees" | "decision_tree",
      "params": {"nEstimators": 100, "rounds": 120, "maxDepth": 6,
                 "learningRate": 0.1},              # optional, TS-style keys
      "featureNames": ["x1", "x2", ...],
      "xTrain": [[...], ...], "yTrain": [...],
      "xTest":  [[...], ...],
      "topN": 10
    }

Response (stdout, JSON, camelCase keys for the TypeScript route)::

    {
      "task": "regression", "model": "gradient_boosting",
      "pythonTwin": "GradientBoostingRegressor",
      "top": [{"rank": 1, "feature": "x1", "meanAbs": 0.834}, ...],
      "baseValue": 0.417, "nSamples": 60, "nFeatures": 2, "nTrain": 240,
      "shapVersion": "0.54.1", "elapsedSeconds": 1.2
    }

Errors print a one-line ``{"error": "..."}`` JSON on stdout with exit code 2
so the route can surface them as HTTP 400/500 cleanly.
"""
from __future__ import annotations

import json
import math
import sys
import time
from typing import Any

EXIT_OK = 0
EXIT_ERROR = 2

#: The web playground's tree families that have a Python registry twin.
#: Every other model family (linear / svm / knn / mlp / ...) is refused
#: honestly upstream — TreeExplainer explains trees (the desktop twin's
#: own scope; its model-agnostic sibling is permutation importance).
TREE_TWINS = ("gradient_boosting", "random_forest", "extra_trees", "decision_tree")

#: TypeScript hyper keys -> sklearn constructor kwargs (validated by
#: ``build_model`` against the estimator's own signature; unknown names
#: raise the typed ModelError, which the route surfaces).
PARAM_MAP = {
    "nEstimators": "n_estimators",
    "rounds": "n_estimators",
    "maxDepth": "max_depth",
    "learningRate": "learning_rate",
}


def _clean(value: Any) -> Any:
    """Recursively convert numpy scalars + NaN/inf to strict-JSON values."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)  # numpy scalars, Paths, etc.


def _fail(message: str) -> int:
    print(json.dumps({"error": message}))
    return EXIT_ERROR


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _fail(f"bridge request is not valid JSON: {exc}")

    if not isinstance(request, dict):
        return _fail("bridge request must be a JSON object")

    task = request.get("task")
    model = request.get("model")
    params_in = request.get("params", {})
    feature_names = request.get("featureNames")
    x_train = request.get("xTrain")
    y_train = request.get("yTrain")
    x_test = request.get("xTest")
    top_n = request.get("topN", 10)

    if task not in ("regression", "classification"):
        return _fail("task must be 'regression' or 'classification'")
    if model not in TREE_TWINS:
        return _fail(
            f"model must be one of the tree twins ({', '.join(TREE_TWINS)}) — "
            f"TreeExplainer explains trees"
        )
    if not isinstance(params_in, dict):
        return _fail("params must be an object when present")
    if not isinstance(feature_names, list) or not all(
        isinstance(n, str) and n for n in feature_names
    ):
        return _fail("featureNames must be a non-empty list of strings")
    if not isinstance(x_train, list) or not x_train:
        return _fail("xTrain must be a non-empty list of rows")
    if not isinstance(y_train, list) or len(y_train) != len(x_train):
        return _fail("yTrain must parallel xTrain (same row count)")
    if not isinstance(x_test, list) or not x_test:
        return _fail("xTest must be a non-empty list of rows")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        return _fail("topN must be a positive integer")

    width = len(feature_names)

    def _as_matrix(rows: Any, label: str) -> list[list[float]]:
        if not isinstance(rows, list):
            raise ValueError(f"{label} must be a list of rows")
        out: list[list[float]] = []
        for i, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != width:
                raise ValueError(
                    f"{label} row {i} must carry {width} values (one per feature)"
                )
            vals: list[float] = []
            for v in row:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"{label} row {i} carries a non-numeric value")
                fv = float(v)
                if not math.isfinite(fv):
                    raise ValueError(f"{label} row {i} carries a non-finite value")
                vals.append(fv)
            out.append(vals)
        return out

    try:
        xt = _as_matrix(x_train, "xTrain")
        xe = _as_matrix(x_test, "xTest")
        yt: list[float] = []
        for i, v in enumerate(y_train):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"yTrain[{i}] is not numeric")
            fv = float(v)
            if not math.isfinite(fv):
                raise ValueError(f"yTrain[{i}] is not finite")
            yt.append(fv)
    except ValueError as exc:
        return _fail(str(exc))

    # map the TS hyper keys onto the sklearn twin's constructor kwargs
    params: dict[str, Any] = {}
    for key, value in params_in.items():
        sklearn_name = PARAM_MAP.get(str(key))
        if sklearn_name is None or value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue  # the route's zod schema already typed these; skip noise
        params[sklearn_name] = int(value) if isinstance(value, int) else float(value)

    try:  # pragma: no cover - import guard for a broken environment
        from pharmaml.ml.explain import shap_summary
        from pharmaml.ml.models import build_model
        from pharmaml.exceptions import DependencyError, ModelError
    except ImportError as exc:
        return _fail(f"pharmaml unavailable: {exc}")

    import pandas as pd
    import shap

    started = time.perf_counter()
    try:
        estimator = build_model(str(model), str(task), params or None)
    except (ModelError, DependencyError) as exc:
        return _fail(f"the Python twin could not be built: {exc}")

    frame_train = pd.DataFrame(xt, columns=list(feature_names))
    frame_test = pd.DataFrame(xe, columns=list(feature_names))
    try:
        estimator.fit(frame_train, yt)
        summary = shap_summary(estimator, frame_test)
    except DependencyError as exc:
        return _fail(f"shap is unavailable: {exc}")
    except Exception as exc:  # ModelError + sklearn fit errors -> honest 400s
        return _fail(f"the Python twin could not be explained: {exc}")

    ranking = summary.ranking(top_n=top_n)
    response = {
        "task": task,
        "model": model,
        "pythonTwin": summary.model_name,
        "top": [
            {"rank": rank + 1, "feature": name, "meanAbs": value}
            for rank, (name, value) in enumerate(ranking)
        ],
        "baseValue": summary.base_value,
        "nSamples": summary.n_samples,
        "nFeatures": len(feature_names),
        "nTrain": len(xt),
        "shapVersion": getattr(shap, "__version__", "unknown"),
        "elapsedSeconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(_clean(response)))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
