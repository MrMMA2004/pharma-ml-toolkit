"""Dataset quality -- the part16 §15.4 scientific-validation report.

The §15.4 checklist ("Data Quality and Scientific Validation") rides
on top of :mod:`pharmaml.data_profiling`: the profiler owns the
descriptive statistics and the generic alert thresholds; this module
owns the *scientific* gate -- the SMILES and molecular-structure
validation (two-stage: grammar parse, then sanitization), the
chemistry-aware duplicate detection (canonical-form duplicates), the
leakage family (target leakage, duplicate columns, ID-like
advisories, the train/test overlap risk), the class-imbalance
warning, and the pinned scientific range vocabulary (unit
validation).  Every check is read-only: the report describes the
frame, it never alters a single user-provided cell.

The honest scope, stated once: this is the §15.4 surface over a
loaded table.  Cross-workflow checks that need the docking/NCA/PK
domain objects (their coordinate and unit contracts) stay in those
workflows' own validators, where the domain lives -- the report links
the vocabulary, it does not duplicate their rules.

Examples
--------
>>> import pandas as pd
>>> from pharmaml.data_quality import data_quality_report
>>> frame = pd.DataFrame({
...     "smiles": ["CCO", "CCO", "not_a_smiles", "CCN"],
...     "activity": [1.0, 1.0, 0.0, 0.0],
... })
>>> report = data_quality_report(frame, target="activity")
>>> codes = {finding["code"] for finding in report["findings"]}
>>> {"invalid_smiles", "duplicate_rows", "duplicate_molecules",
...  "train_test_overlap_risk"} <= codes
True
>>> report["counts"]["error"]
1
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "SMILES_DETECTION_THRESHOLD",
    "data_quality_report",
]

#: A column is "the SMILES column" when at least this fraction of its
#: sampled non-null string values parses as valid SMILES (or the name
#: carries "smiles" and parses at all).
SMILES_DETECTION_THRESHOLD = 0.8

#: The duplicate-molecule advisory's canonicalization: the two-stage
#: parse + sanitize pipeline, then ``MolToSmiles`` (the identity a
#: duplicate check can honestly use).
SMILES_SAMPLE_LIMIT = 200

#: The pinned scientific ranges (§15.4 "unit validation").  The
#: vocabulary is deliberately small and exact (case-insensitive
#: full-name match): a range fires only when a numeric column's
#: lowercased name equals the anchor, the bounds ride the finding's
#: evidence, and the check never edits the values it flags.
SCIENTIFIC_RANGES: dict[str, tuple[float, float, str]] = {
    "mw": (0.0, 10000.0, "molecular weight in [0, 10000]"),
    "molecular_weight": (0.0, 10000.0, "molecular weight in [0, 10000]"),
    "mass": (0.0, 10000.0, "molecular weight in [0, 10000]"),
    "logp": (-20.0, 20.0, "logP in [-20, 20]"),
    "logd": (-20.0, 20.0, "logD in [-20, 20]"),
    "conc": (0.0, float("inf"), "concentration >= 0"),
    "concentration": (0.0, float("inf"), "concentration >= 0"),
    "dose": (0.0, float("inf"), "dose >= 0"),
    "time": (0.0, float("inf"), "time >= 0"),
    "hour": (0.0, float("inf"), "time >= 0"),
}

#: The ID-like advisory's floor: below this row count every column
#: is trivially unique-per-row, and the advisory is noise, not signal.
ID_LIKE_MIN_ROWS = 10

#: The class-imbalance warning's majority fraction (§15.4 "class
#: imbalance warnings"): the dominant class holds at least 90% of the
#: target's values.
CLASS_IMBALANCE_FRACTION = 0.9

#: The target-leakage correlation bound (|corr| at or above this is
#: a leakage finding; 1.0-exact catches the identical-column class).
TARGET_LEAKAGE_CORRELATION = 0.999

#: The §15.4 checklist row each check answers (the executed-checks
#: list mirrors this vocabulary, so the API report and the directive
#: read the same).
CHECKLIST: list[tuple[str, str]] = [
    ("smiles_validation", "SMILES validation"),
    ("molecular_structure_validation", "Molecular structure validation"),
    ("duplicate_detection", "Duplicate detection"),
    ("invalid_descriptor_detection", "Invalid descriptor detection"),
    ("missing_value_report", "Missing-value reports"),
    ("data_leakage_warnings", "Data leakage warnings"),
    ("target_leakage_detection", "Target leakage detection"),
    ("class_imbalance_warnings", "Class imbalance warnings"),
    ("train_test_separation_check", "Train/test separation checks"),
    ("unit_validation", "Unit validation"),
    ("scientific_input_validation", "Scientific input validation"),
]


def _finding(
    code: str,
    severity: str,
    category: str,
    message: str,
    columns: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One report finding: the code, the severity, the honest evidence."""
    return {
        "code": code,
        "severity": severity,
        "category": category,
        "message": message,
        "columns": columns or [],
        "evidence": evidence or {},
    }


def _json_safe(value: Any) -> Any:
    """numpy scalars to their JSON-safe Python primitives."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _detect_smiles_column(frame: pd.DataFrame) -> str | None:
    """The honest SMILES auto-detection: name anchor or parse rate.

    A column qualifies when its name carries "smiles" (any case) and
    any sampled value parses, or -- name aside -- at least
    :data:`SMILES_DETECTION_THRESHOLD` of a sample of its non-null
    string values parses as a valid molecule.  The best parse rate
    wins ties; ``None`` when no column qualifies.
    """
    try:
        from rdkit import Chem  # the deferred import the chem modules use
    except Exception:  # noqa: BLE001 -- a broken runtime is the health
        # route's state to report, not this module's crash
        return None
    best: tuple[float, str] | None = None
    from rdkit.rdBase import DisableLog, EnableLog

    DisableLog("rdApp.error")
    try:
        for column in frame.columns:
            series = frame[column]
            if series.dtype != object and not pd.api.types.is_string_dtype(series):
                continue
            values = series.dropna().astype(str)
            if values.empty:
                continue
            sample = values.head(SMILES_SAMPLE_LIMIT)
            parsed = sum(
                1 for value in sample if Chem.MolFromSmiles(value) is not None
            )
            rate = parsed / len(sample)
            name_anchor = "smiles" in str(column).lower()
            if parsed and name_anchor:
                return str(column)  # the named column wins outright
            if rate >= SMILES_DETECTION_THRESHOLD and (best is None or rate > best[0]):
                best = (rate, str(column))
    finally:
        EnableLog("rdApp.error")
    return best[1] if best is not None else None


def _chemistry_findings(
    frame: pd.DataFrame, column: str
) -> list[dict[str, Any]]:
    """The two-stage SMILES/structure validation over one column.

    Stage one is the grammar parse (``MolFromSmiles`` returning
    ``None`` = invalid SMILES); stage two is sanitization of a
    grammatically-valid molecule (``SanitizeMol`` raising = the
    structure itself is invalid -- valence, aromaticity, stereo).
    Both stages count and cite up to five examples; the third
    finding is the canonical-form duplicate census.
    """
    from rdkit import Chem
    from rdkit.rdBase import DisableLog, EnableLog

    findings: list[dict[str, Any]] = []
    series = frame[column].dropna().astype(str)
    invalid: list[str] = []
    unsanitizable: list[str] = []
    canonical: dict[str, int] = {}
    # RDKit logs parse errors to stderr before returning None; the
    # findings below are the honest report, not the raw parser trace.
    DisableLog("rdApp.error")
    try:
        for value in series:
            mol = Chem.MolFromSmiles(value, sanitize=False)
            if mol is None:
                invalid.append(value)
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:  # noqa: BLE001 -- RDKit raises its own
                # exception types (AtomValenceException &c); the honest
                # report counts the class, it does not re-raise it
                unsanitizable.append(value)
                continue
            try:
                form = Chem.MolToSmiles(mol)
            except Exception:  # noqa: BLE001
                unsanitizable.append(value)
                continue
            canonical[form] = canonical.get(form, 0) + 1
    finally:
        EnableLog("rdApp.error")
    n = len(series)
    if invalid:
        findings.append(_finding(
            "invalid_smiles", "error", "chemistry",
            f"{len(invalid)} of {n} SMILES values failed the grammar parse.",
            columns=[column],
            evidence={"invalid_count": len(invalid), "checked": n,
                     "examples": invalid[:5]},
        ))
    if unsanitizable:
        findings.append(_finding(
            "unsanitizable_molecules", "error", "chemistry",
            f"{len(unsanitizable)} of {n} values parsed but failed sanitization.",
            columns=[column],
            evidence={"unsanitizable_count": len(unsanitizable),
                      "checked": n, "examples": unsanitizable[:5]},
        ))
    duplicates = {form: count for form, count in canonical.items()
                  if count > 1}
    if duplicates:
        total = sum(duplicates.values())
        findings.append(_finding(
            "duplicate_molecules", "warning", "chemistry",
            f"{total} molecules share {len(duplicates)} canonical forms.",
            columns=[column],
            evidence={"duplicate_forms": len(duplicates),
                      "duplicate_rows": total,
                      "top": sorted(duplicates.items(),
                                     key=lambda kv: -kv[1])[:5]},
        ))
    return findings


def _leakage_findings(
    frame: pd.DataFrame, target: str | None
) -> list[dict[str, Any]]:
    """The leakage family: ID-like, duplicate columns, target leakage."""
    findings: list[dict[str, Any]] = []
    n = len(frame)
    for column in frame.columns:
        series = frame[column].dropna()
        if (n > ID_LIKE_MIN_ROWS and series.size == n
                and series.nunique() == n):
            role = "target" if column == target else "feature"
            findings.append(_finding(
                "id_like_column", "info", "leakage",
                f"Column '{column}' is unique per row ({role}); carrying it "
                "as an identifier into a model leaks row identity.",
                columns=[str(column)],
                evidence={"distinct": int(n), "rows": int(n)},
            ))
    pairs: list[tuple[str, str]] = []
    columns = [str(c) for c in frame.columns]
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            if frame[a].equals(frame[b]):
                pairs.append((a, b))
    if pairs:
        findings.append(_finding(
            "duplicate_columns", "warning", "leakage",
            f"{len(pairs)} column pairs are identical (redundant copies).",
            columns=sorted({c for pair in pairs for c in pair}),
            evidence={"pairs": [[a, b] for a, b in pairs[:10]]},
        ))
    if target is not None and target in frame.columns:
        the_target = pd.to_numeric(frame[target], errors="coerce")
        for column in frame.columns:
            if str(column) == str(target):
                continue
            feature = pd.to_numeric(frame[column], errors="coerce")
            if feature.notna().sum() == 0:
                continue
            identical = frame[column].astype(str).equals(frame[target].astype(str))
            correlation: float | None = None
            if the_target.notna().sum() > 1 and feature.notna().sum() > 1:
                aligned = pd.concat([feature, the_target], axis=1).dropna()
                if len(aligned) > 1 and aligned.iloc[:, 0].nunique() > 1:
                    correlation = float(
                        aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                    )
            leaking = identical or (
                correlation is not None
                and np.isfinite(correlation)
                and abs(correlation) >= TARGET_LEAKAGE_CORRELATION
            )
            if leaking:
                if identical:
                    relation = "is identical to"
                elif correlation is not None:
                    relation = f"correlates at {correlation:.4f} with"
                else:
                    relation = "mirrors"
                findings.append(_finding(
                    "target_leakage", "error", "leakage",
                    f"Column '{column}' {relation} the target "
                    f"'{target}'.",
                    columns=[str(column), str(target)],
                    evidence={"identical": bool(identical),
                              "correlation": _json_safe(correlation)},
                ))
    return findings


def _distribution_findings(
    frame: pd.DataFrame, target: str | None
) -> list[dict[str, Any]]:
    """The class-imbalance warning + the constant-column census."""
    findings: list[dict[str, Any]] = []
    constant = [
        str(column) for column in frame.columns
        if frame[column].nunique(dropna=False) <= 1
    ]
    if constant:
        findings.append(_finding(
            "constant_columns", "warning", "distribution",
            f"{len(constant)} columns carry a single value "
            "(zero variance).",
            columns=constant,
            evidence={"columns": constant},
        ))
    if target is not None and target in frame.columns:
        counts = frame[target].value_counts(dropna=False)
        if 2 <= len(counts) <= 20:
            majority = float(counts.iloc[0] / counts.sum())
            if majority >= CLASS_IMBALANCE_FRACTION:
                findings.append(_finding(
                    "class_imbalance", "warning", "distribution",
                    f"Target '{target}': the dominant class holds "
                    f"{majority:.1%} of the values.",
                    columns=[str(target)],
                    evidence={"classes": len(counts),
                              "majority_fraction": round(majority, 4),
                              "top": [(str(k), int(v))
                                      for k, v in counts.head(5).items()]},
                ))
    return findings


def _range_findings(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The pinned unit-validation vocabulary over numeric columns."""
    findings: list[dict[str, Any]] = []
    lowered = {str(c).lower(): str(c) for c in frame.columns}
    for anchor, (low, high, rule) in SCIENTIFIC_RANGES.items():
        column = lowered.get(anchor)
        if column is None:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        outside = numeric[(numeric < low) | (numeric > high)]
        if not outside.empty:
            findings.append(_finding(
                "out_of_range_values", "warning", "range",
                f"Column '{column}': {len(outside)} values outside "
                f"{rule}.",
                columns=[column],
                evidence={"rule": rule, "violations": len(outside),
                          "examples": [float(v) for v in outside.head(5)]},
            ))
    return findings


def data_quality_report(
    frame: pd.DataFrame,
    target: str | None = None,
    smiles_column: str | None = None,
) -> dict[str, Any]:
    """The §15.4 report -- every check read-only, never an edit.

    Examples
    --------
    >>> import pandas as pd
    >>> from pharmaml.data_quality import data_quality_report
    >>> frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    >>> report = data_quality_report(frame)
    >>> report["rows"], report["columns"]
    (2, 2)
    >>> report["counts"]
    {'error': 0, 'warning': 0, 'info': 0}
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError("data_quality_report needs a non-empty DataFrame")
    n, m = frame.shape
    findings: list[dict[str, Any]] = []

    # 1. missing-value reports (per column with any gap)
    missing = {
        str(column): int(frame[column].isna().sum())
        for column in frame.columns
        if frame[column].isna().any()
    }
    if missing:
        findings.append(_finding(
            "missing_values", "warning", "completeness",
            f"{len(missing)} columns carry missing values.",
            columns=sorted(missing),
            evidence={"per_column": missing,
                      "cells_missing": int(frame.isna().sum().sum())},
        ))

    # 2. duplicate rows (the exact-row census)
    duplicated_mask = frame.duplicated(keep=False)
    n_duplicated = int(duplicated_mask.sum())
    if n_duplicated:
        findings.append(_finding(
            "duplicate_rows", "warning", "integrity",
            f"{n_duplicated} rows are exact duplicates "
            f"({n_duplicated / n:.1%}).",
            evidence={"duplicate_rows": n_duplicated,
                      "fraction": round(n_duplicated / n, 4)},
        ))

    # 3. non-finite numeric values (invalid descriptors)
    non_finite: dict[str, int] = {}
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = frame[column].dropna().to_numpy(dtype=float)
            infinite = int(np.isinf(values).sum())
            if infinite:
                non_finite[str(column)] = infinite
    if non_finite:
        findings.append(_finding(
            "non_finite_values", "error", "integrity",
            f"{sum(non_finite.values())} infinite values across "
            f"{len(non_finite)} numeric columns.",
            columns=sorted(non_finite),
            evidence={"per_column": non_finite},
        ))

    # 4. the chemistry pair over the SMILES column: ``None`` = the
    #    honest auto-detection, ``""`` = the explicit disable, any
    #    other string = the explicit override
    if smiles_column is None:
        column = _detect_smiles_column(frame)
    else:
        column = smiles_column or None
    chemistry: list[dict[str, Any]] = []
    if column is not None and column in frame.columns:
        chemistry = _chemistry_findings(frame, column)
    findings.extend(chemistry)

    # 5. the leakage family + 6. the distribution family + 7. ranges
    findings.extend(_leakage_findings(frame, target))
    findings.extend(_distribution_findings(frame, target))
    findings.extend(_range_findings(frame))

    # 8. the train/test overlap advisory (data-level §15.4 "separation")
    overlap_codes = {"duplicate_rows", "duplicate_molecules"}
    if any(finding["code"] in overlap_codes for finding in findings):
        findings.append(_finding(
            "train_test_overlap_risk", "warning", "leakage",
            "Duplicate rows/molecules present: a random train/test split "
            "can place identical records in both folds (evaluation "
            "leakage). De-duplicate before splitting.",
            evidence={"see": ["duplicate_rows", "duplicate_molecules"]},
        ))

    # 9. the non-numeric advisory (the loader's own dropped-column
    #    policy; the SMILES column itself is chemistry surface, not
    #    dropped-column noise)
    object_columns = [
        str(name) for name in frame.columns
        if frame[name].dtype == object and str(name) != column
    ]
    if object_columns:
        findings.append(_finding(
            "non_numeric_column", "info", "integrity",
            f"{len(object_columns)} non-numeric columns will be dropped "
            "by the ML loader's feature policy.",
            columns=object_columns,
            evidence={"columns": object_columns},
        ))

    order = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda finding: order[finding["severity"]])
    counts = {
        severity: sum(1 for finding in findings
                      if finding["severity"] == severity)
        for severity in order
    }
    executed = {finding["code"] for finding in findings}
    checks: list[dict[str, Any]] = []
    for key, label in CHECKLIST:
        if key in {"smiles_validation", "molecular_structure_validation"}:
            detail = (f"over column '{column}'"
                      if column is not None else
                      "skipped: no SMILES column detected")
            checks.append({"check": key, "label": label,
                          "status": "executed" if column else "skipped",
                          "detail": detail})
        elif key == "train_test_separation_check":
            checks.append({"check": key, "label": label,
                          "status": "executed", "detail": (
                "overlap advisory issued" if "train_test_overlap_risk"
                in executed else "no duplicates to straddle folds")})
        else:
            checks.append({"check": key, "label": label,
                          "status": "executed",
                          "detail": "over the loaded frame"})
    return {
        "rows": int(n),
        "columns": int(m),
        "target": target,
        "smiles_column": column,
        "checks": checks,
        "findings": findings,
        "counts": counts,
        "read_only": True,
        "advisory": (
            "Advisory report only -- PharmaML never modifies "
            "user-provided scientific data."
        ),
    }
