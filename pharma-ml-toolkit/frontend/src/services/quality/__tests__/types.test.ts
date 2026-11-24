/**
 * part16 §15.4 (Phase 12) — the data-quality service's own contract.
 *
 * The `POST /api/v1/datasets/{id}/quality` surface narrowed through
 * the house parse pattern: the full report field-for-field (the
 * severity counts, the §15.4 checks list with the honest executed/
 * skipped statuses, the findings with their pinned codes), the
 * read-only flag's absence as an ApiSchemaError (the advisory
 * contract is not optional), the unknown-code drift verdict, and the
 * client's request shape (the QualityRequest body straight from the
 * generated types). The vocabulary pins here mirror the Python drift
 * guard (`tests/test_frontend_quality.py`) on the other side of the
 * wire.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

const postMock = vi.hoisted(() => vi.fn());
vi.mock("@/services/api/client", () => ({
  api: { post: (path: string, body?: unknown) => postMock(path, body) },
}));

import { qualityApi } from "@/services/quality/client";
import { ApiSchemaError } from "@/services/runs/types";
import {
  parseDataQualityReport,
  QUALITY_CATEGORIES,
  QUALITY_CODES,
  QUALITY_SEVERITIES,
} from "@/services/quality/types";

function reportPayload(): Record<string, unknown> {
  return {
    rows: 10,
    columns: 3,
    target: "activity",
    smiles_column: "smiles",
    checks: [
      {
        check: "smiles_validation",
        label: "SMILES validation",
        status: "executed",
        detail: "over column 'smiles'",
      },
      {
        check: "molecular_structure_validation",
        label: "Molecular structure validation",
        status: "skipped",
        detail: "skipped: no SMILES column detected",
      },
    ],
    findings: [
      {
        code: "invalid_smiles",
        severity: "error",
        category: "chemistry",
        message: "1 of 10 SMILES values failed the grammar parse.",
        columns: ["smiles"],
        evidence: { invalid_count: 1, examples: ["not_a_smiles"] },
      },
      {
        code: "class_imbalance",
        severity: "warning",
        category: "distribution",
        message: "Target 'activity': the dominant class holds 90.0%.",
        columns: ["activity"],
        evidence: { majority_fraction: 0.9 },
      },
    ],
    counts: { error: 1, warning: 1, info: 0 },
    read_only: true,
    advisory:
      "Advisory report only -- PharmaML never modifies user-provided scientific data.",
  };
}

describe("parseDataQualityReport", () => {
  it("narrows the full §15.4 report field-for-field", () => {
    const report = parseDataQualityReport(reportPayload());
    expect(report.rows).toBe(10);
    expect(report.columns).toBe(3);
    expect(report.target).toBe("activity");
    expect(report.smiles_column).toBe("smiles");
    expect(report.read_only).toBe(true);
    expect(report.counts).toEqual({ error: 1, warning: 1, info: 0 });
    expect(report.checks).toHaveLength(2);
    expect(report.checks[1]?.status).toBe("skipped");
    expect(report.findings).toHaveLength(2);
    expect(report.findings[0]?.code).toBe("invalid_smiles");
    expect(report.findings[0]?.evidence).toEqual({
      invalid_count: 1,
      examples: ["not_a_smiles"],
    });
  });

  it("rejects a payload without the read-only flag", () => {
    const payload = reportPayload();
    delete (payload as Record<string, unknown>)["read_only"];
    expect(() => parseDataQualityReport(payload)).toThrow(ApiSchemaError);
  });

  it("rejects an unknown finding code (the drift verdict)", () => {
    const payload = reportPayload();
    const findings = payload["findings"] as Array<Record<string, unknown>>;
    findings[0]!["code"] = "made_up_code";
    expect(() => parseDataQualityReport(payload)).toThrow(/unknown finding code/);
  });

  it("rejects an unknown severity", () => {
    const payload = reportPayload();
    const findings = payload["findings"] as Array<Record<string, unknown>>;
    findings[0]!["severity"] = "fatal";
    expect(() => parseDataQualityReport(payload)).toThrow(/unknown severity/);
  });

  it("rejects a non-array findings block", () => {
    const payload = reportPayload();
    payload["findings"] = { not: "an array" };
    expect(() => parseDataQualityReport(payload)).toThrow(ApiSchemaError);
  });
});

describe("the pinned §15.4 vocabulary", () => {
  it("carries the fourteen finding codes", () => {
    expect(QUALITY_CODES).toHaveLength(14);
    expect(QUALITY_CODES).toContain("target_leakage");
    expect(QUALITY_CODES).toContain("train_test_overlap_risk");
    expect(QUALITY_CODES).toContain("unsanitizable_molecules");
  });

  it("carries the three severities and six categories", () => {
    expect(QUALITY_SEVERITIES).toEqual(["error", "warning", "info"]);
    expect(QUALITY_CATEGORIES).toHaveLength(6);
  });
});

describe("qualityApi.run", () => {
  beforeEach(() => {
    postMock.mockReset();
  });

  it("POSTs the QualityRequest body and returns the parsed report", async () => {
    postMock.mockResolvedValue({ mode: "sync", quality: reportPayload() });
    const report = await qualityApi.run("ds-001");
    expect(postMock).toHaveBeenCalledWith(
      "/api/v1/datasets/ds-001/quality",
      {
        smiles_column: null,
        target: null,
        mode: "sync",
      },
    );
    expect(report.read_only).toBe(true);
  });

  it("passes the explicit knobs through", async () => {
    postMock.mockResolvedValue({ mode: "sync", quality: reportPayload() });
    await qualityApi.run("ds-001", {
      smilesColumn: "",
      target: "y",
    });
    expect(postMock).toHaveBeenCalledWith(
      "/api/v1/datasets/ds-001/quality",
      {
        smiles_column: "",
        target: "y",
        mode: "sync",
      },
    );
  });

  it("surfaces a report-less response as an error", async () => {
    postMock.mockResolvedValue({ mode: "sync" });
    await expect(qualityApi.run("ds-001")).rejects.toThrow(
      /no report/,
    );
  });
});
