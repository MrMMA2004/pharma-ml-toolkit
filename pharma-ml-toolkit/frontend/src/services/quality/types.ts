/**
 * The data-quality DTOs — part16 §15.4 (the Phase-12 frontend lane).
 *
 * Field-for-field mirrors of the backend's `data_quality_report`
 * (`pharmaml.data_quality`) plus the route's `{"quality": …}`
 * wrapper — the Python drift guard in `tests/test_frontend_quality.py`
 * pins the vocabulary (codes, severities, categories, checks), so a
 * backend rename turns that suite red before the frontend drifts.
 *
 * The report is advisory by construction: the backend sets
 * `read_only: true` on every payload and this surface renders
 * findings — it never offers a mutation, per §15.4's "never silently
 * alter user-provided scientific data".
 */

import { ApiSchemaError } from "@/services/runs/types";

/** The finding severities, in display order (mirrors the backend's sort). */
export const QUALITY_SEVERITIES = ["error", "warning", "info"] as const;

export type QualitySeverity = (typeof QUALITY_SEVERITIES)[number];

/** The finding categories (the §15.4 families). */
export const QUALITY_CATEGORIES = [
  "chemistry",
  "completeness",
  "distribution",
  "integrity",
  "leakage",
  "range",
] as const;

export type QualityCategory = (typeof QUALITY_CATEGORIES)[number];

/** The finding codes — the backend's pinned vocabulary. */
export const QUALITY_CODES = [
  "invalid_smiles",
  "unsanitizable_molecules",
  "duplicate_molecules",
  "duplicate_rows",
  "duplicate_columns",
  "missing_values",
  "non_finite_values",
  "constant_columns",
  "id_like_column",
  "target_leakage",
  "class_imbalance",
  "train_test_overlap_risk",
  "out_of_range_values",
  "non_numeric_column",
] as const;

export type QualityCode = (typeof QUALITY_CODES)[number];

/** One §15.4 finding: the code, the severity, the honest evidence. */
export interface QualityFinding {
  readonly code: QualityCode;
  readonly severity: QualitySeverity;
  readonly category: QualityCategory;
  readonly message: string;
  readonly columns: readonly string[];
  readonly evidence: Readonly<Record<string, unknown>>;
}

/** One row of the executed-check list (status executed/skipped). */
export interface QualityCheckEntry {
  readonly check: string;
  readonly label: string;
  readonly status: "executed" | "skipped";
  readonly detail: string;
}

/** `POST /api/v1/datasets/{id}/quality` — the full report. */
export interface DataQualityReport {
  readonly rows: number;
  readonly columns: number;
  readonly target: string | null;
  readonly smiles_column: string | null;
  readonly checks: readonly QualityCheckEntry[];
  readonly findings: readonly QualityFinding[];
  readonly counts: Readonly<Record<QualitySeverity, number>>;
  readonly read_only: boolean;
  readonly advisory: string;
}

/** The route wrapper (`{"mode": "sync", "quality": …}`). */
export interface DatasetQualityResponse {
  readonly mode: "sync";
  readonly quality: DataQualityReport;
}

const SEVERITY_SET = new Set<string>(QUALITY_SEVERITIES);
const CATEGORY_SET = new Set<string>(QUALITY_CATEGORIES);
const CODE_SET = new Set<string>(QUALITY_CODES);

function severityOf(value: unknown, ctx: string): QualitySeverity {
  if (typeof value !== "string" || !SEVERITY_SET.has(value)) {
    throw new ApiSchemaError(`${ctx}: unknown severity ${String(value)}`);
  }
  return value as QualitySeverity;
}

function codeOf(value: unknown, ctx: string): QualityCode {
  if (typeof value !== "string" || !CODE_SET.has(value)) {
    throw new ApiSchemaError(`${ctx}: unknown finding code ${String(value)}`);
  }
  return value as QualityCode;
}

function categoryOf(value: unknown, ctx: string): QualityCategory {
  if (typeof value !== "string" || !CATEGORY_SET.has(value)) {
    throw new ApiSchemaError(`${ctx}: unknown category ${String(value)}`);
  }
  return value as QualityCategory;
}

function stringArray(value: unknown, ctx: string): readonly string[] {
  if (!Array.isArray(value) || value.some((v) => typeof v !== "string")) {
    throw new ApiSchemaError(`${ctx}: expected a string array`);
  }
  return value as readonly string[];
}

function parseFinding(raw: unknown, ctx: string): QualityFinding {
  if (typeof raw !== "object" || raw === null) {
    throw new ApiSchemaError(`${ctx}: finding is not an object`);
  }
  const r = raw as Record<string, unknown>;
  if (typeof r["message"] !== "string") {
    throw new ApiSchemaError(`${ctx}: message is not a string`);
  }
  return {
    code: codeOf(r["code"], ctx),
    severity: severityOf(r["severity"], ctx),
    category: categoryOf(r["category"], ctx),
    message: r["message"],
    columns: stringArray(r["columns"] ?? [], ctx),
    evidence:
      typeof r["evidence"] === "object" && r["evidence"] !== null
        ? (r["evidence"] as Readonly<Record<string, unknown>>)
        : {},
  };
}

function parseCheckEntry(raw: unknown, ctx: string): QualityCheckEntry {
  if (typeof raw !== "object" || raw === null) {
    throw new ApiSchemaError(`${ctx}: check entry is not an object`);
  }
  const r = raw as Record<string, unknown>;
  const status = r["status"];
  if (status !== "executed" && status !== "skipped") {
    throw new ApiSchemaError(`${ctx}: unknown check status ${String(status)}`);
  }
  return {
    check: typeof r["check"] === "string" ? r["check"] : "",
    label: typeof r["label"] === "string" ? r["label"] : "",
    status,
    detail: typeof r["detail"] === "string" ? r["detail"] : "",
  };
}

/** Narrow the backend payload into the typed report (ApiSchemaError on drift). */
export function parseDataQualityReport(raw: unknown): DataQualityReport {
  if (typeof raw !== "object" || raw === null) {
    throw new ApiSchemaError("quality report is not an object");
  }
  const r = raw as Record<string, unknown>;
  const ctx = "quality report";
  const findings = r["findings"];
  const checks = r["checks"];
  const counts = r["counts"];
  if (!Array.isArray(findings) || !Array.isArray(checks)) {
    throw new ApiSchemaError(`${ctx}: findings/checks are not arrays`);
  }
  if (typeof counts !== "object" || counts === null) {
    throw new ApiSchemaError(`${ctx}: counts block missing`);
  }
  const parsedCounts: Record<QualitySeverity, number> = {
    error: 0,
    warning: 0,
    info: 0,
  };
  for (const severity of QUALITY_SEVERITIES) {
    const value = (counts as Record<string, unknown>)[severity];
    if (typeof value === "number") {
      parsedCounts[severity] = value;
    }
  }
  if (r["read_only"] !== true) {
    throw new ApiSchemaError(`${ctx}: read_only flag missing`);
  }
  return {
    rows: typeof r["rows"] === "number" ? r["rows"] : 0,
    columns: typeof r["columns"] === "number" ? r["columns"] : 0,
    target: typeof r["target"] === "string" ? r["target"] : null,
    smiles_column:
      typeof r["smiles_column"] === "string" ? r["smiles_column"] : null,
    checks: checks.map((entry, index) =>
      parseCheckEntry(entry, `${ctx}: checks[${index}]`),
    ),
    findings: findings.map((finding, index) =>
      parseFinding(finding, `${ctx}: findings[${index}]`),
    ),
    counts: parsedCounts,
    read_only: true,
    advisory:
      typeof r["advisory"] === "string"
        ? r["advisory"]
        : "Advisory report only — PharmaML never modifies user-provided scientific data.",
  };
}
