import { describe, expect, it } from "vitest";

import {
  parseTrainResult,
  parseTrainSplitBlock,
  type TrainSplitBlock,
  type TrainedModelView,
} from "@/services/ml/client";

/**
 * ML-3 — the train-lane split section's runtime parsers: the carve
 * block narrows field-by-field (the strategy validated against the
 * engine's own literal set), an unknown strategy or a non-numeric side
 * throws instead of rendering as if it were honest, and the train
 * result payload answers `{model, split}` with a null split when the
 * request carried no split section (the legacy full-frame lane).
 */

const MODEL: TrainedModelView = {
  model_id: "mdl-abc123",
  name: "random_forest",
  task: "classification",
  dataset_id: "ds-1",
  target: "y",
  features: ["x"],
  n_train_rows: 8,
  best_params: {},
  cv_scores: [1.0, 1.0],
  cv_mean: 1.0,
  cv_std: 0.0,
  fit_seconds: 0.02,
  created_at: "2026-11-28T00:00:00Z",
};

const BLOCK: TrainSplitBlock = {
  strategy: "scaffold",
  smiles_column: "smiles",
  group_column: null,
  test_size: 0.25,
  random_state: 42,
  n_rows: 12,
  sides: { train: { rows: 8 }, heldout: { rows: 4 } },
  group_census: {
    train: {
      n_groups: 1,
      top_groups: [["c1ccccc1", 8]],
    },
    heldout: { n_groups: 1, top_groups: [["", 4]] },
    shared_groups: [],
  },
};

describe("the ML-3 train split block parser", () => {
  it("narrows a full carve block with the census and the verdict", () => {
    const parsed = parseTrainSplitBlock(
      JSON.parse(JSON.stringify(BLOCK)),
      "train split",
    );
    expect(parsed).not.toBeNull();
    expect(parsed?.strategy).toBe("scaffold");
    expect(parsed?.smiles_column).toBe("smiles");
    expect(parsed?.group_column).toBeNull();
    expect(parsed?.test_size).toBe(0.25);
    expect(parsed?.n_rows).toBe(12);
    expect(parsed?.sides.train.rows).toBe(8);
    expect(parsed?.sides.heldout.rows).toBe(4);
    expect(parsed?.group_census?.train.top_groups[0]).toEqual([
      "c1ccccc1",
      8,
    ]);
    expect(parsed?.group_census?.shared_groups).toEqual([]);
  });

  it("answers null for an absent or null block (the legacy lane)", () => {
    expect(parseTrainSplitBlock(undefined, "x")).toBeNull();
    expect(parseTrainSplitBlock(null, "x")).toBeNull();
  });

  it("throws on a strategy outside the engine's literals", () => {
    const hostile = { ...(BLOCK as unknown as Record<string, unknown>) };
    hostile["strategy"] = "nope";
    expect(() =>
      parseTrainSplitBlock(hostile, "train split"),
    ).toThrow(/engine's literals/);
  });

  it("throws when a side's row count is not a number", () => {
    const hostile = JSON.parse(JSON.stringify(BLOCK));
    hostile["sides"]["train"]["rows"] = "8";
    expect(() =>
      parseTrainSplitBlock(hostile, "train split"),
    ).toThrow(/side rows are not numbers/);
  });

  it("throws when n_rows is not a number", () => {
    const hostile = JSON.parse(JSON.stringify(BLOCK));
    hostile["n_rows"] = "12";
    expect(() =>
      parseTrainSplitBlock(hostile, "train split"),
    ).toThrow(/n_rows is not a number/);
  });
});

describe("the ML-3 train result payload parser", () => {
  it("returns the model view and a null split without the key", () => {
    const parsed = parseTrainResult(
      { model: JSON.parse(JSON.stringify(MODEL)) },
      "train result",
    );
    expect(parsed.model.model_id).toBe("mdl-abc123");
    expect(parsed.model.n_train_rows).toBe(8);
    expect(parsed.split).toBeNull();
  });

  it("narrows both halves when the carve block rides the payload", () => {
    const parsed = parseTrainResult(
      {
        model: JSON.parse(JSON.stringify(MODEL)),
        split: JSON.parse(JSON.stringify(BLOCK)),
      },
      "train result",
    );
    expect(parsed.model.model_id).toBe("mdl-abc123");
    expect(parsed.split?.strategy).toBe("scaffold");
    expect(parsed.split?.sides.heldout.rows).toBe(4);
    expect(parsed.split?.group_census?.shared_groups).toEqual([]);
  });
});
