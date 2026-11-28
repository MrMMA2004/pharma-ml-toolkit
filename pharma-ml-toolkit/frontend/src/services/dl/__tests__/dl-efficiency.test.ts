import { describe, expect, it } from "vitest";

import { parseDLEfficiencyResult } from "@/services/dl/client";

/**
 * DL-3 (the WORKFLOW_ENHANCEMENT_ROADMAP's row 15) — the efficiency
 * client pair: the strict parser over the route's own answers.
 *
 * The honest clauses the roadmap pins:
 *
 *   the int8 answer — the per-layer engine report fields + the
 *   memory line, mirrored field-for-field;
 *   the prune answer — the magnitude counts + the metric block
 *   (nulls stay null, never implied zeros);
 *   the artifact view rides parsed (or stays null when the lane
 *   was asked to skip it);
 *   a malformed answer is the parser's own refusal — a missing
 *   total, a missing per-layer field, a layers-length disagreement.
 */

const INT8_ANSWER = {
  mode: "sync",
  model_id: "mdl-dltorch000333",
  network: "deep_mlp",
  efficiency_mode: "quantize_int8",
  n_layers: 2,
  n_weights: 64,
  layers: [
    {
      layer: "0.weight",
      shape: [8, 6],
      n_weights: 48,
      method: "int8",
      bits_per_weight: 8,
      rmse: 0.0021,
      relative_error: 0.0052,
    },
    {
      layer: "3.weight",
      shape: [2, 8],
      n_weights: 16,
      method: "int8",
      bits_per_weight: 8,
      rmse: 0.0017,
      relative_error: 0.0048,
    },
  ],
  artifact: {
    artifact_id: "art-eff000444555",
    kind: "table",
    filename: "dl-efficiency-quantize_int8.csv",
    content_type: "text/csv; charset=utf-8",
    size_bytes: 412,
    sha256: "a".repeat(64),
    created_at: "2026-11-28T00:00:00Z",
    meta: { model_id: "mdl-dltorch000333" },
    download_url: "/api/v1/artifacts/art-eff000444555/download",
  },
  method: "int8",
  bits_per_weight: 8,
  original_bits: 2048,
  quantized_bits: 512,
  memory_savings: 0.75,
  note: "the per-layer rows ride the REAL quantization engine's own report",
};

const PRUNE_ANSWER = {
  mode: "sync",
  model_id: "mdl-dltorch000333",
  network: "deep_mlp",
  efficiency_mode: "prune",
  n_layers: 2,
  n_weights: 64,
  layers: [
    {
      layer: "0.weight",
      shape: [8, 6],
      n_weights: 48,
      n_zeroed: 24,
      n_kept: 24,
      sparsity: 0.5,
    },
    {
      layer: "3.weight",
      shape: [2, 8],
      n_weights: 16,
      n_zeroed: 8,
      n_kept: 8,
      sparsity: 0.5,
    },
  ],
  artifact: null,
  target_sparsity: 0.5,
  scope: "local",
  n_zeroed: 32,
  achieved_sparsity: 0.5,
  metric: {
    kind: "classification",
    before: 0.917,
    after: 0.875,
    delta: -0.042,
    note: "scored through a pruned COPY",
  },
  note: "the masks ride the REAL magnitude-pruning engine",
};

describe("parseDLEfficiencyResult — the DL-3 client pair", () => {
  it("mirrors the int8 answer field-for-field (the engine's own report)", () => {
    const parsed = parseDLEfficiencyResult(INT8_ANSWER, "dl.efficiency");
    expect(parsed.mode).toEqual("sync");
    expect(parsed.modelId).toEqual("mdl-dltorch000333");
    expect(parsed.network).toEqual("deep_mlp");
    expect(parsed.efficiencyMode).toEqual("quantize_int8");
    expect(parsed.nLayers).toEqual(2);
    expect(parsed.nWeights).toEqual(64);
    expect(parsed.layers).toHaveLength(2);
    const first = parsed.layers[0];
    expect(first?.kind).toEqual("quantize");
    expect(first?.layer).toEqual("0.weight");
    expect(first?.shape).toEqual([8, 6]);
    expect(first?.nWeights).toEqual(48);
    if (first?.kind === "quantize") {
      expect(first.method).toEqual("int8");
      expect(first.bitsPerWeight).toEqual(8);
      expect(first.rmse).toEqual(0.0021);
      expect(first.relativeError).toEqual(0.0052);
    }
    // the quantize totals + the artifact view ride parsed
    expect(parsed.method).toEqual("int8");
    expect(parsed.bitsPerWeight).toEqual(8);
    expect(parsed.originalBits).toEqual(2048);
    expect(parsed.quantizedBits).toEqual(512);
    expect(parsed.memorySavings).toEqual(0.75);
    expect(parsed.artifact?.artifact_id).toEqual("art-eff000444555");
    expect(parsed.artifact?.filename).toEqual(
      "dl-efficiency-quantize_int8.csv",
    );
    // the prune lanes' fields stay honestly null
    expect(parsed.targetSparsity).toBeNull();
    expect(parsed.metric).toBeNull();
    expect(parsed.nZeroed).toBeNull();
  });

  it("mirrors the prune answer (the counts + the metric block + the null artifact)", () => {
    const parsed = parseDLEfficiencyResult(PRUNE_ANSWER, "dl.efficiency");
    expect(parsed.efficiencyMode).toEqual("prune");
    const first = parsed.layers[0];
    expect(first?.kind).toEqual("prune");
    if (first?.kind === "prune") {
      expect(first.nZeroed).toEqual(24);
      expect(first.nKept).toEqual(24);
      expect(first.sparsity).toEqual(0.5);
    }
    expect(parsed.targetSparsity).toEqual(0.5);
    expect(parsed.scope).toEqual("local");
    expect(parsed.nZeroed).toEqual(32);
    expect(parsed.achievedSparsity).toEqual(0.5);
    expect(parsed.metric).toEqual({
      kind: "classification",
      before: 0.917,
      after: 0.875,
      delta: -0.042,
      note: "scored through a pruned COPY",
    });
    // the quantize totals stay honestly null on the prune lane
    expect(parsed.method).toBeNull();
    expect(parsed.bitsPerWeight).toBeNull();
    expect(parsed.memorySavings).toBeNull();
    // the artifact stays null when the lane skipped it
    expect(parsed.artifact).toBeNull();
  });

  it("refuses a layers-length disagreement", () => {
    const broken = {
      ...INT8_ANSWER,
      n_layers: 3,
    };
    expect(() =>
      parseDLEfficiencyResult(broken, "dl.efficiency"),
    ).toThrow(/layers is not the answer's own per-layer table/);
  });

  it("refuses a quantize row that lost the engine's own report fields", () => {
    const broken = {
      ...INT8_ANSWER,
      layers: [
        { ...INT8_ANSWER.layers[0], rmse: null },
        INT8_ANSWER.layers[1],
      ],
    };
    expect(() =>
      parseDLEfficiencyResult(broken, "dl.efficiency"),
    ).toThrow(/a quantize row lost the engine's own report fields/);
  });

  it("refuses a prune answer without its metric block", () => {
    const broken = { ...PRUNE_ANSWER, metric: undefined };
    expect(() =>
      parseDLEfficiencyResult(broken, "dl.efficiency"),
    ).toThrow(/the prune lane lost its metric block/);
  });

  it("refuses a prune row that lost its counts (never implied zeros)", () => {
    const broken = {
      ...PRUNE_ANSWER,
      layers: [
        { ...PRUNE_ANSWER.layers[0], n_zeroed: undefined },
        PRUNE_ANSWER.layers[1],
      ],
    };
    expect(() =>
      parseDLEfficiencyResult(broken, "dl.efficiency"),
    ).toThrow(/a prune row lost its counts/);
  });

  it("refuses a non-object answer outright", () => {
    expect(() =>
      parseDLEfficiencyResult("not-an-object", "dl.efficiency"),
    ).toThrow();
  });
});
