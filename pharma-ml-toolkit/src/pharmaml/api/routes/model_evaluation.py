"""``/api/v1/model-evaluation`` — the conformal + coverage-diagnostics
surface over the REAL backend (part13 Phase 4's queued row: the
Model-Evaluation/Applicability-Domain API connection — r311's audit
recorded zero conformal surfacing in the API; this family closes it).

* ``GET /api/v1/model-evaluation/catalog`` — the three entry points +
  their defaults (one source: the backend signatures).
* ``POST /api/v1/model-evaluation/conformal-regression`` —
  :func:`pharmaml.ml.conformal.calibrate_split_conformal` +
  ``SplitConformalRegressor.predict_interval`` over a session-trained
  regression model (``/api/v1/ml/train`` or ``/api/v1/dl/train``):
  the calibration radius at ``alpha``, intervals on optional query
  rows, and — when independent query labels arrive — the empirical
  coverage, optionally summarized ``groups``/``smiles``-wise through
  :func:`pharmaml.ml.coverage_diagnostics.coverage_by_group` /
  :func:`pharmaml.ml.scaffold_coverage.scaffold_aware_coverage` with
  the interval widths as sizes.
* ``POST /api/v1/model-evaluation/conformal-classification`` —
  :func:`pharmaml.ml.conformal_classification.\
calibrate_split_conformal_classifier` (marginal) or
  :func:`pharmaml.ml.conformal_classification.\
calibrate_label_conditional_classifier` (``conditional="label"``)
  + ``predict_set`` over a session-trained classification model:
  the threshold(s), label sets on optional query rows, and — with
  query labels — the empirical coverage + per-class coverage rows.
* ``POST /api/v1/model-evaluation/coverage-diagnostics`` — the
  standalone diagnostics surface: a held-out ``covered`` vector +
  ``groups`` (or ``smiles`` for the scaffold-aware mode, r311's
  module riding the API for the first time) + optional sizes → the
  Wilson-bounded report.

All routes are sync-only: one predict pass + a quantile — the
``/applicability-domain/knn`` / ``domain-box`` precedent for light
compute. Empirical coverage is reported as empirical, never as the
conformal guarantee; the calibration/test split discipline is the
caller's (the backend's own docstrings say so and the 422 lanes
repeat it).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.conformal import calibrate_split_conformal
from pharmaml.ml.conformal_classification import (
    LabelConditionalConformalClassifier,
    SplitConformalClassifier,
    calibrate_label_conditional_classifier,
    calibrate_split_conformal_classifier,
)
from pharmaml.ml.coverage_diagnostics import (
    CoverageGroup,
    CoverageReport,
    coverage_by_group,
)
from pharmaml.ml.scaffold_coverage import scaffold_aware_coverage

from ..stores import json_safe

router = APIRouter(prefix="/model-evaluation", tags=["model-evaluation"])

_MAX_CAL_ROWS = 2000
_MAX_TEST_ROWS = 1000
_MAX_FEATURES = 50


class ConformalRegressionRequest(BaseModel):
    # unique OpenAPI component names (the applicability-domain
    # family's anti-collision rule: same-named models elsewhere make
    # the exported contract hash-seed-unstable)
    model_id: str
    x_cal: list[list[float]] = Field(min_length=2, max_length=_MAX_CAL_ROWS)
    y_cal: list[float] = Field(min_length=2, max_length=_MAX_CAL_ROWS)
    x_test: list[list[float]] | None = Field(
        default=None, min_length=1, max_length=_MAX_TEST_ROWS
    )
    y_test: list[float] | None = Field(
        default=None, min_length=1, max_length=_MAX_TEST_ROWS
    )
    alpha: float = Field(default=0.1, gt=0.0, lt=1.0)
    groups: list[str | int] | None = Field(
        default=None, min_length=1, max_length=_MAX_TEST_ROWS
    )
    smiles: list[str] | None = Field(
        default=None, min_length=1, max_length=_MAX_TEST_ROWS
    )


class ConformalClassificationRequest(BaseModel):
    model_id: str
    x_cal: list[list[float]] = Field(min_length=2, max_length=_MAX_CAL_ROWS)
    y_cal: list[int | str] = Field(min_length=2, max_length=_MAX_CAL_ROWS)
    x_test: list[list[float]] | None = Field(
        default=None, min_length=1, max_length=_MAX_TEST_ROWS
    )
    y_test: list[int | str] | None = Field(
        default=None, min_length=1, max_length=_MAX_TEST_ROWS
    )
    alpha: float = Field(default=0.1, gt=0.0, lt=1.0)
    conditional: Literal["split", "label"] = "split"


class CoverageDiagnosticsRequest(BaseModel):
    covered: list[bool] = Field(min_length=1, max_length=5000)
    groups: list[str | int] | None = Field(
        default=None, min_length=1, max_length=5000
    )
    smiles: list[str] | None = Field(
        default=None, min_length=1, max_length=5000
    )
    sizes: list[float] | None = Field(
        default=None, min_length=1, max_length=5000
    )
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)


@router.get("/catalog")
def catalog(request: Request) -> dict:
    return {
        "entry_points": [
            {
                "route": "/conformal-regression",
                "backend": "calibrate_split_conformal + predict_interval",
                "defaults": {"alpha": 0.1},
            },
            {
                "route": "/conformal-classification",
                "backend": (
                    "calibrate_split_conformal_classifier | "
                    "calibrate_label_conditional_classifier + predict_set"
                ),
                "defaults": {"alpha": 0.1, "conditional": "split"},
            },
            {
                "route": "/coverage-diagnostics",
                "backend": "coverage_by_group | scaffold_aware_coverage",
                "defaults": {"confidence": 0.95},
            },
        ],
        "interval_columns": ["prediction", "lower", "upper"],
        "set_columns": ["prediction_set", "set_size", "covered"],
        "coverage_columns": [
            "group", "n_test", "n_covered", "coverage",
            "lower", "upper", "mean_size",
        ],
        "notes": [
            (
                "sync-only: one predict pass + a quantile (the knn/domain-box "
                "precedent for light compute)"
            ),
            "empirical coverage is empirical, never the conformal guarantee",
            (
                "calibration rows must never have trained the model — the "
                "finite-sample guarantee needs exchangeable held-out rows"
            ),
        ],
    }


def _matrix(rows: list[list[float]], what: str) -> np.ndarray:
    try:
        matrix = np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"{what} must be numeric rows: {exc}"
        ) from exc
    if matrix.ndim != 2:
        raise HTTPException(
            status_code=422, detail=f"{what} must be a 2-D rows x features array"
        )
    if matrix.shape[1] > _MAX_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"{what} has {matrix.shape[1]} features; the limit is {_MAX_FEATURES}",
        )
    return matrix


def _group_row(row: CoverageGroup) -> dict:
    return json_safe({
        "group": row.group,
        "n_test": row.n_test,
        "n_covered": row.n_covered,
        "coverage": row.coverage,
        "lower": row.lower,
        "upper": row.upper,
        "mean_size": row.mean_size,
    })


def _coverage_payload(report: CoverageReport) -> dict:
    return {
        "confidence": report.confidence,
        "overall": _group_row(report.overall),
        "groups": [_group_row(g) for g in report.groups],
    }


def _coverage_diagnostics(
    covered: np.ndarray,
    groups: list[str | int] | None,
    smiles: list[str] | None,
    sizes: np.ndarray | None,
    confidence: float,
) -> dict:
    if groups is not None and smiles is not None:
        raise HTTPException(
            status_code=422,
            detail="pass either groups or smiles, not both",
        )
    try:
        if smiles is not None:
            report = scaffold_aware_coverage(
                covered, smiles, sizes=sizes, confidence=confidence
            )
            return {"mode": "scaffold", **_coverage_payload(report)}
        report = coverage_by_group(
            covered, groups, sizes=sizes, confidence=confidence
        )
        return {"mode": "groups", **_coverage_payload(report)}
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _get_model(request: Request, model_id: str, task: str):
    model_record = request.app.state.model_store.get(model_id)
    if model_record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    if model_record.task != task:
        raise HTTPException(
            status_code=422,
            detail=(
                f"this route needs a {task} model; model {model_id} was "
                f"trained for {model_record.task}"
            ),
        )
    return model_record


def _check_width(matrix: np.ndarray, n_features: int, what: str) -> None:
    if matrix.shape[1] != n_features:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the model was trained on {n_features} features; {what} has "
                f"{matrix.shape[1]}"
            ),
        )


@router.post("/conformal-regression")
def conformal_regression(req: ConformalRegressionRequest, request: Request) -> dict:
    model_record = _get_model(request, req.model_id, "regression")
    x_cal = _matrix(req.x_cal, "x_cal")
    y_cal = np.asarray(req.y_cal, dtype=float).reshape(-1)
    if x_cal.shape[0] != y_cal.size:
        raise HTTPException(
            status_code=422,
            detail=(
                f"x_cal has {x_cal.shape[0]} rows; y_cal has {y_cal.size} "
                "(they must match)"
            ),
        )
    _check_width(x_cal, len(model_record.features), "x_cal")
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    if req.x_test is not None:
        x_test = _matrix(req.x_test, "x_test")
        _check_width(x_test, len(model_record.features), "x_test")
        if req.y_test is not None:
            y_test = np.asarray(req.y_test, dtype=float).reshape(-1)
            if x_test.shape[0] != y_test.size:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"x_test has {x_test.shape[0]} rows; y_test has "
                        f"{y_test.size} (they must match)"
                    ),
                )
        elif req.groups is not None or req.smiles is not None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "coverage diagnostics need y_test — empirical coverage is "
                    "measured against independent labels"
                ),
            )
    if (req.groups is not None or req.smiles is not None) and req.y_test is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "coverage diagnostics need y_test — empirical coverage is "
                "measured against independent labels"
            ),
        )
    if req.groups is not None and req.smiles is not None:
        raise HTTPException(
            status_code=422, detail="pass either groups or smiles, not both"
        )
    try:
        calibrated = calibrate_split_conformal(
            model_record.model, x_cal, y_cal, alpha=req.alpha
        )
        intervals = (
            None
            if x_test is None
            else calibrated.predict_interval(x_test, y_test)
        )
    except (DataValidationError, ModelError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload: dict[str, Any] = {
        "alpha": float(calibrated.alpha),
        "radius": float(calibrated.radius),
        "n_calibration": int(calibrated.n_calibration),
        "rank": int(calibrated.rank),
        "mean_width": float(2.0 * calibrated.radius),
        "feature_names": list(model_record.features),
        "intervals": None,
        "empirical": None,
        "coverage_diagnostics": None,
    }
    if intervals is not None:
        payload["intervals"] = json_safe(
            intervals.to_dataframe().to_dict(orient="records")
        )
        if y_test is not None:
            payload["empirical"] = {
                "n_test": int(intervals.n_test),
                "n_covered": int(intervals.n_covered or 0),
                "coverage": (
                    None if intervals.coverage is None
                    else float(intervals.coverage)
                ),
            }
            covered = intervals.covered
            if covered is not None and (req.groups is not None or req.smiles is not None):
                sizes = np.full(covered.size, 2.0 * calibrated.radius)
                payload["coverage_diagnostics"] = _coverage_diagnostics(
                    covered, req.groups, req.smiles, sizes, 0.95
                )
    return payload


@router.post("/conformal-classification")
def conformal_classification(
    req: ConformalClassificationRequest, request: Request
) -> dict:
    model_record = _get_model(request, req.model_id, "classification")
    x_cal = _matrix(req.x_cal, "x_cal")
    if x_cal.shape[0] != len(req.y_cal):
        raise HTTPException(
            status_code=422,
            detail=(
                f"x_cal has {x_cal.shape[0]} rows; y_cal has {len(req.y_cal)} "
                "(they must match)"
            ),
        )
    _check_width(x_cal, len(model_record.features), "x_cal")
    x_test: np.ndarray | None = None
    y_test: list[int | str] | None = None
    if req.x_test is not None:
        x_test = _matrix(req.x_test, "x_test")
        _check_width(x_test, len(model_record.features), "x_test")
        if req.y_test is not None:
            y_test = req.y_test
            if x_test.shape[0] != len(y_test):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"x_test has {x_test.shape[0]} rows; y_test has "
                        f"{len(y_test)} (they must match)"
                    ),
                )
    try:
        calibrated: SplitConformalClassifier | LabelConditionalConformalClassifier
        label_calibrated: LabelConditionalConformalClassifier | None = None
        if req.conditional == "label":
            label_calibrated = calibrated = calibrate_label_conditional_classifier(
                model_record.model, x_cal, req.y_cal, alpha=req.alpha
            )
        else:
            calibrated = calibrate_split_conformal_classifier(
                model_record.model, x_cal, req.y_cal, alpha=req.alpha
            )
        sets = (
            None if x_test is None else calibrated.predict_set(x_test, y_test)
        )
    except (DataValidationError, ModelError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    threshold = getattr(calibrated, "threshold", None)
    payload: dict[str, Any] = {
        "alpha": float(calibrated.alpha),
        "conditional": req.conditional,
        "classes": [json_safe(c) for c in calibrated.classes],
        "n_calibration": int(calibrated.n_calibration),
        "threshold": None if threshold is None else float(threshold),
        "class_thresholds": None,
        "calibration_summary": None,
        "sets": None,
        "empirical": None,
        "per_class": None,
    }
    if label_calibrated is not None:
        summary = label_calibrated.calibration_summary().to_dict(orient="records")
        payload["class_thresholds"] = {
            json_safe(label): float(t)
            for label, t in zip(
                label_calibrated.classes, label_calibrated.thresholds, strict=True
            )
        }
        payload["calibration_summary"] = json_safe(summary)
    if sets is not None:
        payload["sets"] = json_safe(sets.to_dataframe().to_dict(orient="records"))
        payload["empirical"] = {
            "n_test": int(sets.n_test),
            "n_covered": (
                None if sets.n_covered is None else int(sets.n_covered)
            ),
            "coverage": (
                None if sets.coverage is None else float(sets.coverage)
            ),
            "mean_set_size": float(sets.mean_set_size),
        }
        if y_test is not None and sets.covered is not None:
            payload["per_class"] = _coverage_diagnostics(
                sets.covered, list(y_test), None, sets.set_sizes, 0.95
            )
    return payload


@router.post("/coverage-diagnostics")
def coverage_diagnostics(req: CoverageDiagnosticsRequest, request: Request) -> dict:
    if req.groups is None and req.smiles is None:
        raise HTTPException(
            status_code=422,
            detail="coverage diagnostics need either groups or smiles",
        )
    if req.groups is not None and req.smiles is not None:
        raise HTTPException(
            status_code=422, detail="pass either groups or smiles, not both"
        )
    n = len(req.covered)
    for name, values in (("groups", req.groups), ("smiles", req.smiles),
                         ("sizes", req.sizes)):
        if values is not None and len(values) != n:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"covered has {n} rows; {name} has {len(values)} "
                    "(they must match)"
                ),
            )
    covered = np.asarray(req.covered, dtype=bool)
    sizes = (
        None
        if req.sizes is None
        else np.asarray(req.sizes, dtype=float).reshape(-1)
    )
    return _coverage_diagnostics(
        covered, req.groups, req.smiles, sizes, req.confidence
    )
