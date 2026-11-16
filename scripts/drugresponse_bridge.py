#!/usr/bin/env python3
"""Stdin -> stdout JSON bridge for the Next.js ``/api/ml/drugresponse`` route.

Runs inside the toolkit virtualenv (real pharmaml + numpy + pandas) and
wires ONLY the public ``pharmaml.ml.drugresponse`` APIs the web twin
mirrors — the same surface the desktop Studio's Response AI page drives
through its eight tabs.  The web card's four analyses map to the page's
headline tabs:

    hill    — ``fit_hill_curve`` + ``dose_response_curve`` +
              ``auc_log_dose`` (the page's Hill-curve tab): the seeded
              21-point 4PL demo (the page's exact seed-42 recipe) or a
              pasted (dose, response) series, the four fitted
              parameters, r², the log-dose AUC both ways, and the fitted
              curve on the chart grid.
    metrics — ``response_metrics`` + ``rank_correlations`` (the page's
              response-metrics and rank-correlations tabs in one run):
              Pearson / Spearman / RMSE / MAE of a predicted column
              against an observed one, plus the Kendall tau-b trio, on
              the seeded noisy pair (the page's exact seed-42 recipe) or
              two pasted series (NaN tokens allowed — pairs where
              either side is NaN drop before every metric, the untested
              pairs of a screen).
    model   — ``DrugResponseModel`` (the page's response-model tab):
              the seeded 12x4 screen with its 15% NaN mask (the page's
              ``_seeded_screen`` verbatim), full-batch gradient descent
              over the observed pairs only, the honest in-sample
              pearson, the predicted-response head for CELL_00/CELL_07,
              and the observed-vs-predicted pairs.
    cv      — ``cross_validate_response`` (the page's CV tab): folds
              split CELL LINES (no cell line straddles a fold), each
              fold training a fresh model with the page's exact factory
              (n_factors=2, lr=0.1, epochs=120, l2=0.001, seed=42), the
              per-fold pearson/rmse and the honest across-fold mean.

The page's remaining three tabs (informative regression, the response
VAE, multi-omics fusion) stay desktop-side — the card says so.

Request (stdin, JSON)::

    {"analysis": "hill", "dose": "0.001 0.01 ...", "response": "...",
     "noise": 0.01, "yTrue": "...", "yPred": "...", "epochs": 120,
     "factors": 2, "folds": 4, "seed": 42}

(``dose``/``response`` feed hill (either BOTH present or both empty for
the seeded demo); ``yTrue``/``yPred`` feed metrics (both empty = the
seeded noisy pair); ``epochs``/``factors`` feed model; ``folds`` feeds
cv; ``seed`` varies the seeded demos and the model's weight seed.)

Response (stdout, JSON, camelCase keys for the TypeScript route) — one
block per analysis; every block carries ``elapsedSeconds`` and
``pharmamlVersion``.

Errors print a one-line ``{"error": "..."}`` JSON on stdout with exit
code 2 so the route can surface them as HTTP 400/500 cleanly.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

EXIT_OK = 0
EXIT_ERROR = 2

#: the web card's input budgets — the pasted series are capped the way
#: every web textarea is; the seeded demos ride the page's own sizes.
MAX_SERIES_CHARS = 4000
MAX_HILL_POINTS = 200
MAX_METRICS_POINTS = 400

MIN_EPOCHS, MAX_EPOCHS = 20, 400
MIN_FACTORS, MAX_FACTORS = 1, 4
MIN_FOLDS, MAX_FOLDS = 2, 8
MIN_NOISE, MAX_NOISE = 0.0, 0.5
MAX_SEED = 999_999

ANALYSES = ("hill", "metrics", "model", "cv")

_NAN_TOKENS = frozenset({"nan", "na", "n/a"})


def _fail(message: str) -> int:
    print(json.dumps({"error": message}))
    return EXIT_ERROR


def _parse_series(text: str, name: str, *, min_points: int, allow_nan: bool) -> list[float]:
    """Parse one pasted series: comma/whitespace/semicolon separated."""
    tokens = [t for t in text.replace(";", " ").replace(",", " ").split()]
    if not tokens:
        raise ValueError(f"{name}: the series is empty")
    if len(tokens) > (MAX_METRICS_POINTS if allow_nan else MAX_HILL_POINTS):
        raise ValueError(
            f"{name}: at most {MAX_METRICS_POINTS if allow_nan else MAX_HILL_POINTS}"
            " values fit the web budget"
        )
    values: list[float] = []
    for token in tokens:
        if token.lower() in _NAN_TOKENS:
            if not allow_nan:
                raise ValueError(f"{name}: NaN is not valid here ({token!r})")
            values.append(float("nan"))
            continue
        try:
            values.append(float(token))
        except ValueError:
            raise ValueError(f"{name}: not a number ({token!r})") from None
    finite = [v for v in values if v == v]  # NaN != NaN
    if len(finite) < min_points:
        raise ValueError(
            f"{name}: at least {min_points} finite values are needed "
            f"(got {len(finite)})"
        )
    return values


def _pharmaml_version() -> str:
    from pharmaml import __version__

    return str(__version__)


def _run_hill(request: dict[str, Any]) -> dict[str, Any]:
    from pharmaml.ml.drugresponse import (
        auc_log_dose,
        dose_response_curve,
        fit_hill_curve,
    )

    started = time.perf_counter()
    dose_text = request.get("dose", "")
    response_text = request.get("response", "")
    if not isinstance(dose_text, str) or not isinstance(response_text, str):
        raise ValueError("dose and response must be strings of numbers")
    if len(dose_text) > MAX_SERIES_CHARS or len(response_text) > MAX_SERIES_CHARS:
        raise ValueError(
            f"dose and response must be at most {MAX_SERIES_CHARS} characters"
        )
    noise = _number(request.get("noise", 0.01), "noise", MIN_NOISE, MAX_NOISE)
    seed = request.get("seed", 42)

    import numpy as np

    if not dose_text.strip() and not response_text.strip():
        # the page's exact seeded demo (seed 42 recipe; the card's seed
        # knob varies the noise draw, the generating truth is fixed)
        rng = np.random.default_rng(seed)
        hill_true = {"hill_coef": 1.2, "ic50": 0.8, "top": 1.0, "bottom": 0.05}
        doses = np.logspace(-3, 3, 21)
        clean = np.asarray(
            dose_response_curve(
                doses,
                hill_true["hill_coef"],
                hill_true["ic50"],
                top=hill_true["top"],
                bottom=hill_true["bottom"],
            ),
            dtype=float,
        )
        response = clean + rng.normal(0.0, noise, doses.size)
        source = "seeded 4PL demo"
    elif not dose_text.strip() or not response_text.strip():
        raise ValueError(
            "hill needs BOTH a dose series and a response series "
            "(or neither — the seeded demo)"
        )
    else:
        dose_values = _parse_series(
            dose_text, "dose", min_points=6, allow_nan=False
        )
        response_values = _parse_series(
            response_text, "response", min_points=6, allow_nan=False
        )
        if len(dose_values) != len(response_values):
            raise ValueError(
                f"dose and response must be the same length "
                f"({len(dose_values)} vs {len(response_values)})"
            )
        if any(v <= 0.0 for v in dose_values):
            raise ValueError("dose values must be strictly positive")
        doses = np.asarray(dose_values, dtype=float)
        response = np.asarray(response_values, dtype=float)
        hill_true = None
        source = "pasted series"

    fit = fit_hill_curve(doses, response)
    lo = float(np.log10(doses.min()))
    hi = float(np.log10(doses.max()))
    grid = np.logspace(lo - 0.25, hi + 0.25, 200)
    curve = np.asarray(fit.predict(grid), dtype=float)
    measured_auc = float(auc_log_dose(doses, response))

    payload: dict[str, Any] = {
        "analysis": "hill",
        "source": source,
        "parameters": {
            "hill_coef": float(fit.parameters["hill_coef"]),
            "ic50": float(fit.parameters["ic50"]),
            "top": float(fit.parameters["top"]),
            "bottom": float(fit.parameters["bottom"]),
        },
        "rSquared": float(fit.r_squared),
        "converged": bool(fit.converged),
        "nPoints": int(fit.n_points),
        "fittedAuc": float(fit.fitted_auc),
        "measuredAuc": measured_auc,
        "doses": [float(v) for v in doses],
        "response": [float(v) for v in response],
        "curveDose": [float(v) for v in grid],
        "curveResponse": [float(v) for v in curve],
        "pharmamlVersion": _pharmaml_version(),
    }
    if hill_true is not None:
        payload["truth"] = hill_true
        payload["noise"] = noise
        payload["cleanAuc"] = float(auc_log_dose(doses, clean))
        payload["summary"] = (
            f"{source} (seed {seed}, noise {noise:g}) · r² "
            f"{fit.r_squared:.4f} · fitted AUC {fit.fitted_auc:.4f} vs "
            f"generating {payload['cleanAuc']:.4f} (log-dose)"
        )
    else:
        payload["summary"] = (
            f"{source} · {int(fit.n_points)} points · r² "
            f"{fit.r_squared:.4f} · fitted AUC {fit.fitted_auc:.4f} vs "
            f"measured-points {measured_auc:.4f} (log-dose)"
        )
    payload["elapsedSeconds"] = round(time.perf_counter() - started, 3)
    return payload


def _run_metrics(request: dict[str, Any]) -> dict[str, Any]:
    from pharmaml.ml.drugresponse import response_metrics, rank_correlations

    started = time.perf_counter()
    y_true_text = request.get("yTrue", "")
    y_pred_text = request.get("yPred", "")
    if not isinstance(y_true_text, str) or not isinstance(y_pred_text, str):
        raise ValueError("yTrue and yPred must be strings of numbers")
    if len(y_true_text) > MAX_SERIES_CHARS or len(y_pred_text) > MAX_SERIES_CHARS:
        raise ValueError(
            f"yTrue and yPred must be at most {MAX_SERIES_CHARS} characters"
        )
    seed = request.get("seed", 42)

    import numpy as np

    if not y_true_text.strip() and not y_pred_text.strip():
        rng = np.random.default_rng(seed)
        y_true = rng.uniform(0.0, 1.0, 60)
        y_pred = np.clip(y_true + rng.normal(0.0, 0.08, 60), 0.0, 1.0)
        source = "seeded noisy pair"
    elif not y_true_text.strip() or not y_pred_text.strip():
        raise ValueError(
            "metrics needs BOTH an observed series and a predicted series "
            "(or neither — the seeded noisy pair)"
        )
    else:
        y_true = np.asarray(
            _parse_series(y_true_text, "yTrue", min_points=4, allow_nan=True),
            dtype=float,
        )
        y_pred = np.asarray(
            _parse_series(y_pred_text, "yPred", min_points=4, allow_nan=True),
            dtype=float,
        )
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"yTrue and yPred must be the same length "
                f"({y_true.size} vs {y_pred.size})"
            )
        source = "pasted pair"

    metrics = response_metrics(y_true, y_pred)
    # rank_correlations refuses NaN (its own contract: "x and y must be
    # finite"), while response_metrics drops NaN pairs before every
    # metric — so the rank trio rides the SURVIVING finite pairs, the
    # same pairs the metrics layer scored, and the drop is reported.
    import numpy as _np

    finite = _np.isfinite(y_true) & _np.isfinite(y_pred)
    n_dropped = int(y_true.size - finite.sum())
    rank = rank_correlations(y_true[finite], y_pred[finite])
    n = int(metrics.get("n", rank.n))
    drop_note = (
        f" · {n_dropped} NaN pair(s) dropped before every metric"
        if n_dropped
        else ""
    )
    payload = {
        "analysis": "metrics",
        "source": source,
        "n": n,
        "nDropped": n_dropped,
        "pearson": float(metrics["pearson"]),
        "spearman": float(metrics["spearman"]),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "rank": {
            "n": int(rank.n),
            "pearson": float(rank.pearson),
            "spearman": float(rank.spearman),
            "kendall": float(rank.kendall),
        },
        "yTrue": [float(v) for v in y_true],
        "yPred": [float(v) for v in y_pred],
        "summary": (
            f"{source} · n={n} · pearson {float(metrics['pearson']):.4f} · "
            f"spearman {float(metrics['spearman']):.4f} · kendall "
            f"{float(rank.kendall):.4f} · RMSE {float(metrics['rmse']):.4f}"
            f"{drop_note}"
        ),
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "pharmamlVersion": _pharmaml_version(),
    }
    return payload


def _seeded_screen(rng: Any) -> tuple[Any, Any]:
    """The page's ``_seeded_screen`` verbatim (12x4, 15% NaN mask)."""
    import pandas as pd

    n_cells, n_features = 12, 8
    X = pd.DataFrame(
        rng.normal(0.0, 1.0, (n_cells, n_features)),
        index=[f"CELL_{i:02d}" for i in range(n_cells)],
        columns=[f"feat_{j}" for j in range(n_features)],
    )
    true_factor = rng.normal(0.0, 1.0, n_cells)
    drugs = ["DRUG_A", "DRUG_B", "DRUG_C", "DRUG_D"]
    potency = [0.9, 0.7, 0.5, 0.3]
    import numpy as np

    table = pd.DataFrame(
        {
            d: 1.0
            / (
                1.0
                + np.exp(
                    -(potency[k] * true_factor + rng.normal(0.0, 0.05, n_cells))
                )
            )
            for k, d in enumerate(drugs)
        },
        index=X.index,
    )
    mask = rng.random(table.shape) < 0.15
    return X, table.mask(mask)


def _run_model(request: dict[str, Any]) -> dict[str, Any]:
    from pharmaml.ml.drugresponse import DrugResponseModel

    started = time.perf_counter()
    epochs = _integer(request.get("epochs", 120), "epochs", MIN_EPOCHS, MAX_EPOCHS)
    factors = _integer(request.get("factors", 2), "factors", MIN_FACTORS, MAX_FACTORS)
    seed = request.get("seed", 42)

    import numpy as np

    rng = np.random.default_rng(42)  # the screen seed is fixed, as on the page
    X, table = _seeded_screen(rng)
    model = DrugResponseModel(n_factors=factors, lr=0.1, epochs=epochs, l2=0.001, seed=seed)
    history = model.fit(X, table)
    head = model.predict(["CELL_00", "CELL_07"])
    predicted_all = model.predict(list(table.index)).to_numpy(dtype=float)
    observed_all = table.to_numpy(dtype=float)
    mask = np.isfinite(observed_all)
    observed = observed_all[mask]
    predicted = predicted_all[mask]

    head_rows = [
        {
            "cell": str(index),
            "values": [float(v) for v in row],
        }
        for index, row in head.iterrows()
    ]
    payload = {
        "analysis": "model",
        "epochs": int(history.n_epochs),
        "factors": factors,
        "seed": seed,
        "shape": {"cellLines": int(table.shape[0]), "drugs": int(table.shape[1])},
        "drugs": [str(c) for c in table.columns],
        "trainLosses": [float(v) for v in history.train_losses],
        "finalLoss": float(history.final_loss),
        "nObserved": int(history.n_observed),
        "pearsonR": float(history.pearson_r),
        "predictedHead": head_rows,
        "observed": [float(v) for v in observed],
        "predicted": [float(v) for v in predicted],
        "nPairs": int(observed.size),
        "summary": (
            f"seeded 12x4 screen (15% NaN mask) · {int(history.n_observed)} "
            f"observed pairs · in-sample pearson {float(history.pearson_r):.4f} "
            f"after {int(history.n_epochs)} epochs"
        ),
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "pharmamlVersion": _pharmaml_version(),
    }
    return payload


def _run_cv(request: dict[str, Any]) -> dict[str, Any]:
    from pharmaml.ml.drugresponse import DrugResponseModel, cross_validate_response

    started = time.perf_counter()
    folds = _integer(request.get("folds", 4), "folds", MIN_FOLDS, MAX_FOLDS)

    import numpy as np

    rng = np.random.default_rng(42)
    X, table = _seeded_screen(rng)
    report = cross_validate_response(
        lambda: DrugResponseModel(n_factors=2, lr=0.1, epochs=120, l2=0.001, seed=42),
        X,
        table,
        n_folds=folds,
        seed=42,
    )
    fold_rows = [
        {
            "fold": int(fold.fold),
            "nTestCells": int(fold.n_test_cells),
            "nObserved": int(fold.n_observed),
            "pearson": float(fold.metrics["pearson"]),
            "rmse": float(fold.metrics["rmse"]),
        }
        for fold in report.folds
    ]
    mean_pearson = float(report.summary["pearson"])
    payload = {
        "analysis": "cv",
        "nFolds": int(report.n_folds),
        "folds": fold_rows,
        "meanPearson": mean_pearson,
        "summary": (
            f"{int(report.n_folds)} folds grouped by cell line (no cell "
            f"straddles a fold) · mean pearson {mean_pearson:.4f} · the "
            "honest read: the toy screen barely carries signal"
        ),
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "pharmamlVersion": _pharmaml_version(),
    }
    return payload


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _fail(f"bridge request is not valid JSON: {exc}")

    if not isinstance(request, dict):
        return _fail("bridge request must be a JSON object")

    analysis = request.get("analysis")
    if analysis not in ANALYSES:
        return _fail("analysis must be one of " + ", ".join(ANALYSES))
    if request.get("seed", 42) is not None:
        seed = request.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            return _fail(f"seed must be an integer between 0 and {MAX_SEED}")
    for field in ("dose", "response", "yTrue", "yPred", "noise"):
        if field in request and request[field] is not None and field != "noise":
            if not isinstance(request[field], str):
                return _fail(f"{field} must be a string of numbers")
    if "noise" in request and request["noise"] is not None:
        if isinstance(request["noise"], bool) or not isinstance(
            request["noise"], (int, float)
        ):
            return _fail("noise must be a number")

    try:  # pragma: no cover - import guard for a broken environment
        import pharmaml  # noqa: F401
    except ImportError as exc:
        return _fail(f"pharmaml unavailable: {exc}")

    runner = {
        "hill": _run_hill,
        "metrics": _run_metrics,
        "model": _run_model,
        "cv": _run_cv,
    }[analysis]
    try:
        payload = runner(request)
    except ValueError as exc:  # client errors: honest, named
        return _fail(str(exc))
    except Exception as exc:  # engine errors: the route maps to 500
        return _fail(f"the drug-response analysis failed: {exc}")

    print(json.dumps(payload))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
