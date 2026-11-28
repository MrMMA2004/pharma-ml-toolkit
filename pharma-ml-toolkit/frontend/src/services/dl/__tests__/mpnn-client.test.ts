import { describe, expect, it } from "vitest";

import {
  parseMpnnTrainResult,
  parseMpnnPredictResult,
} from "@/services/dl/client";

/**
 * DL-4 (the WORKFLOW_ENHANCEMENT_ROADMAP's row 16) — the molecular
 * client pair: the strict parsers over the route's own answers.
 *
 * The honest clauses the roadmap pins:
 *
 *   the train answer — the held-out carve's own verdicts (val_rmse a
 *   float; val_correlation null when the carve was too small or
 *   degenerate — never an implied zero) + the honest rejected-rows
 *   accounting;
 *   the predict answer — one prediction per VALID row aligned with
 *   the served index, the rejected rows riding their own list, and
 *   a count disagreement the parser's own refusal;
 *   a malformed answer is the parser's own refusal.
 */

const TRAIN_ANSWER = {
  model: {
    model_id: "mdl-mpnn000222",
    name: "dmpnn_molecular_regressor",
    task: "regression",
    dataset_id: "ds-mpnn0001",
    target: "y",
    features: ["smiles"],
    n_train_rows: 40,
    best_params: { network: "dmpnn", hidden_size: 32 },
    cv_scores: null,
    cv_mean: null,
    cv_std: null,
    fit_seconds: 6.2,
    created_at: "2026-11-28T00:00:00Z",
    engine: "numpy",
    molecular_model: true,
    predict_url: "/api/v1/dl/mpnn-predict",
  },
  val_rmse: 0.109,
  val_n: 8,
  val_correlation: 0.991,
  train_history_tail: [0.61, 0.42, 0.3, 0.18, 0.11],
  refit_history_tail: [0.55, 0.36, 0.25, 0.14, 0.09],
  n_rows: 40,
  n_rejected: 0,
  rejected: [],
  n_parameters: 3193,
  held_out_note: "the carve is seeded and held out",
};

const PREDICT_ANSWER = {
  model_id: "mdl-mpnn000222",
  n_valid: 2,
  predictions: [
    { index: 0, smiles: "CCO", prediction: 3.04 },
    { index: 2, smiles: "c1ccccc1", prediction: 6.02 },
  ],
  n_rejected: 1,
  rejected: [
    { index: 1, smiles: "not-a-smiles", error: "SMILES syntax problem" },
  ],
  note: "the predictions are in the original target units",
};

describe("parseMpnnTrainResult — the DL-4 client pair", () => {
  it("mirrors the train answer field-for-field (the carve's verdicts)", () => {
    const parsed = parseMpnnTrainResult(TRAIN_ANSWER, "dl.mpnn-train");
    expect(parsed.model.model_id).toEqual("mdl-mpnn000222");
    expect(parsed.model.molecular_model).toEqual(true);
    expect(parsed.model.predict_url).toEqual("/api/v1/dl/mpnn-predict");
    expect(parsed.val_rmse).toEqual(0.109);
    expect(parsed.val_n).toEqual(8);
    expect(parsed.val_correlation).toEqual(0.991);
    expect(parsed.train_history_tail).toEqual([
      0.61, 0.42, 0.3, 0.18, 0.11,
    ]);
    expect(parsed.n_rows).toEqual(40);
    expect(parsed.n_rejected).toEqual(0);
    expect(parsed.rejected).toEqual([]);
    expect(parsed.n_parameters).toEqual(3193);
  });

  it("keeps a degenerate correlation honestly null (never an implied zero)", () => {
    const parsed = parseMpnnTrainResult(
      { ...TRAIN_ANSWER, val_correlation: null },
      "dl.mpnn-train",
    );
    expect(parsed.val_correlation).toBeNull();
  });

  it("carries the rejected rows' own accounting", () => {
    const parsed = parseMpnnTrainResult(
      {
        ...TRAIN_ANSWER,
        n_rejected: 2,
        rejected: [
          { index: 3, smiles: null, error: "a null cell" },
          { index: 7, smiles: "C(", error: "ring label not closed" },
        ],
      },
      "dl.mpnn-train",
    );
    expect(parsed.n_rejected).toEqual(2);
    expect(parsed.rejected).toEqual([
      { index: 3, smiles: null, error: "a null cell" },
      { index: 7, smiles: "C(", error: "ring label not closed" },
    ]);
  });

  it("refuses an answer that lost the carve's verdicts", () => {
    expect(() =>
      parseMpnnTrainResult(
        { ...TRAIN_ANSWER, val_rmse: null },
        "dl.mpnn-train",
      ),
    ).toThrow(/val_rmse\/val_n\/n_rows\/n_rejected\/n_parameters/);
  });
});

describe("parseMpnnPredictResult — the DL-4 client pair", () => {
  it("mirrors the predict answer with the served-index alignment", () => {
    const parsed = parseMpnnPredictResult(PREDICT_ANSWER, "dl.mpnn-predict");
    expect(parsed.model_id).toEqual("mdl-mpnn000222");
    expect(parsed.n_valid).toEqual(2);
    expect(parsed.predictions).toEqual([
      { index: 0, smiles: "CCO", prediction: 3.04 },
      { index: 2, smiles: "c1ccccc1", prediction: 6.02 },
    ]);
    expect(parsed.n_rejected).toEqual(1);
    expect(parsed.rejected).toEqual([
      { index: 1, smiles: "not-a-smiles", error: "SMILES syntax problem" },
    ]);
  });

  it("refuses a predictions/n_valid count disagreement", () => {
    const broken = {
      ...PREDICT_ANSWER,
      n_valid: 3, // two rows served — the honest parser refuses
    };
    expect(() =>
      parseMpnnPredictResult(broken, "dl.mpnn-predict"),
    ).toThrow(/disagree with the served n_valid/);
  });

  it("refuses a prediction row that lost its prediction", () => {
    const broken = {
      ...PREDICT_ANSWER,
      predictions: [
        { index: 0, smiles: "CCO", prediction: null },
        { index: 2, smiles: "c1ccccc1", prediction: 6.02 },
      ],
    };
    expect(() =>
      parseMpnnPredictResult(broken, "dl.mpnn-predict"),
    ).toThrow(/a prediction row lost its index\/smiles\/prediction/);
  });

  it("refuses a rejected row that lost its error", () => {
    const broken = {
      ...PREDICT_ANSWER,
      rejected: [{ index: 1, smiles: "C(", error: null }],
    };
    expect(() =>
      parseMpnnPredictResult(broken, "dl.mpnn-predict"),
    ).toThrow(/a rejected row lost its index\/error/);
  });
});
