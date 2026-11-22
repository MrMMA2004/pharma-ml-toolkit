"""The step runners of the lightweight Scientific Workflow Engine
(part13 Phase 6, §8 — the round-322 API lane).

Every runner delegates to a REAL backend callable that already lives
in this tree — the JobManager's own rule ("no algorithm is
reimplemented here") applied to the workflow layer.  Heavy imports
(chemistry stacks, the explain lane) happen INSIDE the runner bodies,
so importing this module stays cheap.

Runner contract (uniform, engine-facing):

* input  — ``inputs: dict[str, Any]`` keyed by the step's input-port
  names (the engine has already resolved the wiring);
* tune   — ``params: dict[str, Any]`` (validated against the step's
  ``ParamSpec`` default vocabulary by the caller);
* output — ``dict[str, Any]`` keyed by the step's OUTPUT-port names.

Failures propagate: a backend's typed error becomes the node's
``FAILED`` record verbatim — the engine's partial-failure semantics
take over downstream.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import STEP_REGISTRY

__all__ = ["STEP_RUNNERS"]

_MAX_REPORT_ROWS = 12


def _runner(step_id: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    return STEP_RUNNERS[step_id]


# -- dataset.import ---------------------------------------------------


def _run_dataset_import(
    inputs: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    import pandas as pd

    rows = inputs["records"]
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError(
            "dataset.import needs a non-empty list of record rows."
        )
    smiles_column = str(params.get("smiles_column", "smiles"))
    target_column = str(params.get("target_column", "pIC50"))
    frame = pd.DataFrame(list(rows))
    for column in (smiles_column, target_column):
        if column not in frame.columns:
            raise ValueError(
                f"dataset.import: records carry no {column!r} column "
                f"(found {list(frame.columns)})"
            )
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    if frame[target_column].isna().any():
        raise ValueError(
            "dataset.import: the target column holds non-numeric or "
            "missing values -- every row needs a numeric target"
        )
    return {"dataset": frame}


# -- chem.standardize -------------------------------------------------


def _run_chem_standardize(
    inputs: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    import pandas as pd

    from pharmaml.chem.csv_standardize import standardize_smiles

    frame: pd.DataFrame = inputs["dataset"]
    smiles_column = str(params.get("smiles_column", "smiles"))
    if smiles_column not in frame.columns:
        raise ValueError(
            f"chem.standardize: no {smiles_column!r} column to standardize"
        )
    outcomes = [standardize_smiles(str(s)) for s in frame[smiles_column]]
    statuses = [o.status for o in outcomes]
    keep = [i for i, o in enumerate(outcomes) if o.canonical is not None]
    standardized = frame.iloc[keep].copy()
    standardized[smiles_column] = [outcomes[i].canonical for i in keep]
    counts: dict[str, float] = {"total": float(len(outcomes))}
    for status in sorted(set(statuses)):
        counts[status] = float(statuses.count(status))
    errors = [
        {"row": i, "error": o.error or ""}
        for i, o in enumerate(outcomes)
        if o.canonical is None and o.error
    ]
    if errors:
        counts["invalid_rows_reported"] = float(len(errors))
    return {"dataset": standardized, "standardization": {
        "status_counts": counts,
        "invalid_rows": errors[:_MAX_REPORT_ROWS],
    }}


# -- chem.descriptors --------------------------------------------------


def _run_chem_descriptors(
    inputs: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    from typing import Literal, cast

    import pandas as pd

    from pharmaml.chem.descriptors import compute_descriptors

    frame: pd.DataFrame = inputs["dataset"]
    smiles_column = str(params.get("smiles_column", "smiles"))
    descriptor_set = str(params.get("descriptor_set", "rdkit_all"))
    allowed = ("rdkit_all", "morgan", "mordred")
    if descriptor_set not in allowed:
        raise ValueError(
            f"chem.descriptors: unknown descriptor_set "
            f"{descriptor_set!r} (have: {', '.join(allowed)})"
        )
    descriptors = compute_descriptors(
        [str(s) for s in frame[smiles_column]],
        descriptor_set=cast(
            "Literal['rdkit_all', 'morgan', 'mordred']", descriptor_set
        ),
        progress=False,
    ).reset_index(drop=True)
    # the backend echoes the input molecules back as a 'SMILES'
    # column -- the pipeline's frame contract is numeric features
    # + the carried target, so the echo is dropped here (never a
    # silent string column leaking into the design matrix)
    if "SMILES" in descriptors.columns:
        descriptors = descriptors.drop(columns=["SMILES"])
    carried = frame.drop(columns=[smiles_column]).reset_index(drop=True)
    out = pd.concat([descriptors, carried], axis=1)
    return {"dataset": out}


# -- ml.split -----------------------------------------------------------


def _run_ml_split(
    inputs: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    import pandas as pd

    from pharmaml.ml.preprocessing import make_train_test_split

    frame: pd.DataFrame = inputs["dataset"]
    target_column = str(params.get("target_column", "pIC50"))
    if target_column not in frame.columns:
        raise ValueError(
            f"ml.split: no {target_column!r} target column in the frame"
        )
    from pharmaml.ml.preprocessing import clean_numeric_frame

    # the REAL backend hygiene: non-numeric columns (a SMILES echo,
    # identifier columns) never enter the design matrix — the
    # backend's own warning names them; the drop rides the bundle
    cleaned = clean_numeric_frame(frame, target_col=target_column)
    dropped_non_numeric = [
        c for c in frame.columns if c not in cleaned.columns
    ]
    y = cleaned[target_column]
    X = cleaned.drop(columns=[target_column])
    if X.shape[1] == 0:
        raise ValueError(
            "ml.split: the frame carries no numeric feature columns "
            "beyond the target — featurize (chem.descriptors) before "
            "splitting (a SMILES-only column is never a feature)"
        )
    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=float(params.get("test_size", 0.2)),
        random_state=int(params.get("random_state", 42)),
        stratify=False,
    )
    return {"split": {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train if isinstance(y_train, pd.Series)
        else pd.Series(y_train),
        "y_test": y_test if isinstance(y_test, pd.Series)
        else pd.Series(y_test),
        "columns": list(X.columns),
        "target_column": target_column,
        "dropped_non_numeric_columns": dropped_non_numeric,
    }}


# -- ml.feature_selection ----------------------------------------------


def _run_ml_feature_selection(
    inputs: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    from pharmaml.ml.preprocessing import select_kbest_union

    split_in: dict[str, Any] = inputs["split"]
    methods = params.get(
        "methods", ("f_regression", "mutual_info_regression")
    )
    if isinstance(methods, (list, tuple)):
        methods = tuple(str(m) for m in methods)
    else:
        methods = (str(methods),)
    X_train, X_test, kept = select_kbest_union(
        split_in["X_train"],
        split_in["y_train"],
        split_in["X_test"],
        k=int(params.get("k", 8)),
        methods=methods,
    )
    # Rank hygiene over the kept set — the leverage/Williams
    # backend's own full-rank contract, executed exactly as its
    # guard prescribes ("drop the collinear descriptors"): a kept
    # column survives only if it ADDS rank to [1 | X_kept].  This
    # catches exact duplicate columns (MaxAbsEStateIndex vs
    # MaxEStateIndex coincide on all-positive E-State sets), zero-
    # variance columns, and any exact linear combination — every
    # dropped name + reason rides the artifact, nothing is silently
    # dropped.
    import numpy.linalg as la

    current_rank = 1
    final: list[str] = []
    dropped: list[dict[str, str]] = []
    for idx, col in enumerate(kept):
        candidate = np.column_stack([
            np.ones(len(X_train)),
            X_train[final + [col]].to_numpy(dtype=float),
        ])
        if la.matrix_rank(candidate) > current_rank:
            final.append(col)
            current_rank += 1
        else:
            reason = (
                "exact duplicate of an earlier kept column"
                if float(
                    X_train[col].to_numpy(dtype=float).var()
                ) == 0.0
                else "exact linear combination of the kept columns"
            )
            if len(final) and np.allclose(
                X_train[col].to_numpy(dtype=float),
                X_train[final[-1]].to_numpy(dtype=float),
            ):
                reason = f"exact duplicate of {final[-1]!r}"
            dropped.append({"column": str(col), "reason": reason})
    X_train = X_train[final]
    X_test = X_test[final]
    return {"split": {
        **split_in,
        "X_train": X_train,
        "X_test": X_test,
        "columns": list(final),
        "selected_columns": list(final),
        "selection_methods": list(methods),
        "n_selected": len(final),
        "rank_hygiene": {
            "dropped": dropped,
            "note": (
                "the leverage/Williams backend's full-rank contract "
                "executed: a kept column survives only if it adds rank "
                "to [1 | X] — every dropped name + reason is recorded"
            ),
        } if dropped else None,
    }}


# -- ml.train ------------------------------------------------------------


def _run_ml_train(
    inputs: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    from pharmaml.ml.models import build_model

    split: dict[str, Any] = inputs["split"]
    task = str(params.get("task", "regression"))
    model_name = str(params.get("model", "random_forest"))
    random_state = int(params.get("random_state", 42))
    estimator = build_model(
        model_name,
        task,
        {"random_state": random_state},
    )
    estimator.fit(split["X_train"], split["y_train"])
    return {"model": estimator}


# -- ml.evaluate ----------------------------------------------------------


def _run_ml_evaluate(
    inputs: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    from pharmaml.ml.evaluation import evaluate_model

    split: dict[str, Any] = inputs["split"]
    task = str(params.get("task", "regression"))
    report = evaluate_model(
        inputs["model"],
        split["X_test"],
        split["y_test"],
        task,
        X_train=split["X_train"],
        y_train=split["y_train"],
    )
    fields = (
        "rmse", "mae", "r2", "max_error", "pearson_r",
        "train_rmse", "train_r2", "train_pearson_r", "r2_gap",
        "accuracy", "f1_macro",
    )
    metrics: dict[str, float] = {}
    for name in fields:
        value = getattr(report, name, None)
        if isinstance(value, (int, float)):
            metrics[name] = float(value)
    if not metrics:
        raise ValueError(
            "ml.evaluate: the report carried none of the expected "
            "numeric metric fields -- refusing to fabricate any"
        )
    return {"metrics": metrics}


# -- ml.applicability ------------------------------------------------------


def _run_ml_applicability(
    inputs: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    from pharmaml.ml.applicability import leverage_williams

    split: dict[str, Any] = inputs["split"]
    result = leverage_williams(
        inputs["model"],
        split["X_train"],
        split["y_train"],
        split["X_test"],
        residual_limit=float(params.get("residual_limit", 3.0)),
    )

    def _bool_list(values: Any) -> list[bool]:
        return [bool(v) for v in np.asarray(values).tolist()]

    block: dict[str, Any] = {
        "n_train": int(result.n_train),
        "n_features": int(result.n_features),
        "leverage_threshold": float(result.leverage_threshold),
        "residual_limit": float(result.residual_limit),
        "train_inside": _bool_list(result.train_inside),
        "train_inside_count": int(np.asarray(result.train_inside).sum()),
        "train_leverage": [float(v) for v in
                           np.asarray(result.train_leverage).tolist()],
    }
    if result.test_leverage is not None:
        block["test_leverage"] = [
            float(v) for v in np.asarray(result.test_leverage).tolist()
        ]
    if result.test_inside is not None:
        block["test_inside"] = _bool_list(result.test_inside)
        block["test_inside_count"] = int(
            np.asarray(result.test_inside).sum()
        )
    return {"ad_report": block}


# -- ml.shap ----------------------------------------------------------------


def _run_ml_shap(
    inputs: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    from pharmaml.ml.explain import shap_summary

    split: dict[str, Any] = inputs["split"]
    summary = shap_summary(inputs["model"], split["X_train"])
    mean_abs = np.asarray(summary.mean_abs).ravel().tolist()
    return {"shap_summary": {
        "names": list(summary.names),
        "mean_abs": [float(v) for v in mean_abs],
        "n_features": len(summary.names),
        "n_samples": int(np.asarray(summary.values).shape[0])
        if summary.values is not None else None,
        "note": (
            "the full SHAP values matrix stays in-process; the "
            "serialised answer carries the bar chart's own numbers"
        ),
    }}


# -- report.generate ---------------------------------------------------------


def _run_report_generate(
    inputs: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    import uuid

    import pandas as pd

    from pharmaml.viz.report import save_pdf_report

    metrics: dict[str, float] = dict(inputs["metrics"])
    tables: dict[str, pd.DataFrame] = {
        "metrics": pd.DataFrame(
            sorted(metrics.items(), key=lambda kv: kv[0]),
            columns=["metric", "value"],
        )
    }
    shap = inputs.get("shap_summary")
    if isinstance(shap, dict) and shap.get("names"):
        frame = pd.DataFrame({
            "feature": shap["names"],
            "mean_abs_shap": shap["mean_abs"],
        }).sort_values("mean_abs_shap", ascending=False)
        tables["shap_top"] = frame.head(_MAX_REPORT_ROWS).reset_index(
            drop=True
        )
    ad = inputs.get("ad_report")
    if isinstance(ad, dict):
        tables["applicability_domain"] = pd.DataFrame(
            [
                ("n_train", ad.get("n_train")),
                ("n_features", ad.get("n_features")),
                ("leverage_threshold h*", ad.get("leverage_threshold")),
                ("train_inside_count", ad.get("train_inside_count")),
                ("test_inside_count", ad.get("test_inside_count")),
            ],
            columns=["field", "value"],
        )
    out_dir = Path(tempfile.gettempdir()) / "pharmaml-workflow-reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / f"build-{uuid.uuid4().hex}.pdf"
    written = save_pdf_report(
        figures={},
        tables=tables,
        metrics=metrics,
        out_path=scratch,
        title=str(params.get("title", "PharmaML workflow report")),
        subtitle="the lightweight workflow engine's run record",
    )
    digest = hashlib.sha256(written.read_bytes()).hexdigest()
    # content-addressed: the backend's PDF builds are byte-
    # deterministic for equal inputs, so the same content lands at
    # the same path — the artifact digest is stable across same-seed
    # runs and the bytes PERSIST (a temp-dir context would leave a
    # ghost path pointing at deleted bytes)
    final = out_dir / f"workflow-report-{digest[:16]}.pdf"
    if final.exists():
        final.unlink()
    scratch.rename(final)
    size = final.stat().st_size
    return {"report": {
        "path": str(final),
        "sha256": digest,
        "bytes": int(size),
        "note": (
            "the artifact's bytes persist at a content-addressed path "
            "under the session's tempdir (the local/offline guarantee "
            "keeps the file off the wire); the digest + size ride the "
            "lineage"
        ),
    }}


#: The runner table (step_id -> the uniform runner contract).
STEP_RUNNERS: dict[
    str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
] = {
    "dataset.import": _run_dataset_import,
    "chem.standardize": _run_chem_standardize,
    "chem.descriptors": _run_chem_descriptors,
    "ml.split": _run_ml_split,
    "ml.feature_selection": _run_ml_feature_selection,
    "ml.train": _run_ml_train,
    "ml.evaluate": _run_ml_evaluate,
    "ml.applicability": _run_ml_applicability,
    "ml.shap": _run_ml_shap,
    "report.generate": _run_report_generate,
}

# keep the registry <-> runner tables honest against each other
assert set(STEP_RUNNERS) == set(STEP_REGISTRY), (
    "every registered step must carry exactly one runner"
)
