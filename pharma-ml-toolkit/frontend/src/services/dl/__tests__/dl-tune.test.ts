import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/services/api/client";
import { dlApi, parseDLTuneReport } from "@/services/dl/client";

/**
 * DL-1 (the roadmap's row 13) — the train operation's search lane as
 * the roadmap's own vitest pair: `dlApi.train` with a tune block over
 * the REAL client (the network boundary alone is faked, so the
 * strict parsers ride the actual code):
 *
 *   the tune pair     — `tune: true` + the search block + n_trials +
 *                      sampler/seed POST verbatim; the sync answer
 *                      narrows into {model, tune} (the trial table,
 *                      the best configuration, the refit score);
 *   the fixed pair    — no tune block: the request stays the legacy
 *                      shape and the sync answer's tune is null (the
 *                      absence stays honest);
 *   malformed-throw   — a trial row that lost its number/value pair
 *                      and a report without its refit score throw
 *                      instead of rendering as if it were honest;
 *   the honest 422    — the backend's own 422 (a doomed search) rides
 *                      through untouched — the page's describeError
 *                      renders the detail verbatim.
 */

const apiMocks = vi.hoisted(() => ({ post: vi.fn() }));

vi.mock("@/services/api/client", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("@/services/api/client")
  >();
  return {
    ...actual,
    api: { ...actual.api, post: apiMocks.post },
  };
});

const TUNE_ANSWER = {
  mode: "sync",
  model: {
    model_id: "mdl-dltorch000222",
    name: "deep_mlp_classifier",
    task: "classification",
    dataset_id: "ds-1",
    target: "y",
    features: ["x1", "x2"],
    n_train_rows: 60,
    best_params: {
      network: "deep_mlp_classifier",
      learning_rate: 0.2,
      epochs: 15,
    },
    cv_scores: null,
    cv_mean: null,
    cv_std: null,
    fit_seconds: 4.5,
    created_at: "2026-11-28T00:00:00Z",
    engine: "torch",
    predict_url: "/api/v1/ml/predict",
  },
  tune: {
    best_params: { learning_rate: 0.2 },
    best_score: 0.9167,
    n_trials: 4,
    task: "classification",
    trials: [
      { number: 0, params: { learning_rate: 0.031 }, value: -0.75 },
      { number: 1, params: { learning_rate: 0.107 }, value: -0.8333 },
      { number: 2, params: { learning_rate: 0.2 }, value: -0.9167 },
      { number: 3, params: { learning_rate: 0.455 }, value: -0.6667 },
    ],
    refit_score: 0.9167,
    sampler: "random",
    seed: 0,
    searched: { learning_rate: { low: 0.001, high: 0.5, log: true } },
  },
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe("dlApi.train — the DL-1 search lane", () => {
  it("POSTs tune: true with the search block verbatim and narrows the answer", async () => {
    apiMocks.post.mockResolvedValueOnce(TUNE_ANSWER);

    const result = await dlApi.train({
      datasetId: "ds-1",
      network: "deep_mlp_classifier",
      hyperparams: { epochs: 15, batch_size: 16 },
      tune: {
        search: {
          learning_rate: { low: 0.001, high: 0.5, log: true },
        },
        nTrials: 4,
        sampler: "random",
        seed: 0,
      },
      mode: "sync",
    });

    expect(apiMocks.post).toHaveBeenCalledTimes(1);
    expect(apiMocks.post).toHaveBeenCalledWith("/api/v1/dl/train", {
      dataset_id: "ds-1",
      network: "deep_mlp_classifier",
      hyperparams: { epochs: 15, batch_size: 16 },
      tune: true,
      search: { learning_rate: { low: 0.001, high: 0.5, log: true } },
      n_trials: 4,
      sampler: "random",
      seed: 0,
      mode: "sync",
    });

    expect(result.mode).toBe("sync");
    if (result.mode !== "sync") {
      throw new Error("unreachable");
    }
    expect(result.model.engine).toBe("torch");
    expect(result.model.best_params).toEqual({
      network: "deep_mlp_classifier",
      learning_rate: 0.2,
      epochs: 15,
    });
    expect(result.tune).not.toBeNull();
    expect(result.tune?.n_trials).toBe(4);
    expect(result.tune?.best_score).toBe(0.9167);
    expect(result.tune?.refit_score).toBe(0.9167);
    expect(result.tune?.trials).toHaveLength(4);
    expect(result.tune?.trials[2]?.params).toEqual({ learning_rate: 0.2 });
    expect(result.tune?.searched).toEqual({
      learning_rate: { low: 0.001, high: 0.5, log: true },
    });
  });

  it("keeps the fixed-config lane's request legacy and the tune absence honest", async () => {
    apiMocks.post.mockResolvedValueOnce({
      mode: "sync",
      model: TUNE_ANSWER.model,
    });

    const result = await dlApi.train({
      datasetId: "ds-1",
      network: "deep_mlp_classifier",
      hyperparams: { epochs: 15 },
      mode: "sync",
    });

    // the generated DTO marks the search fields required — the
    // fixed-config lane always sends them, with the tune lane off
    expect(apiMocks.post).toHaveBeenCalledWith("/api/v1/dl/train", {
      dataset_id: "ds-1",
      network: "deep_mlp_classifier",
      hyperparams: { epochs: 15 },
      mode: "sync",
      tune: false,
      search: null,
      n_trials: 12,
      sampler: "tpe",
      seed: 0,
    });
    expect(result.mode).toBe("sync");
    if (result.mode !== "sync") {
      throw new Error("unreachable");
    }
    expect(result.tune).toBeNull();
  });

  it("surfaces the backend's own 422 verbatim, never a silent failure", async () => {
    apiMocks.post.mockRejectedValueOnce(
      new ApiError(
        422,
        "the tune lane needs a search block: {key: {choices: [...]} or {key: {low: ..., high: ..., log: bool}}} over the network 'deep_mlp_classifier' constructor vocabulary",
      ),
    );

    const thrown = await dlApi
      .train({
        datasetId: "ds-1",
        network: "deep_mlp_classifier",
        hyperparams: { epochs: 15 },
        tune: {
          search: {},
          nTrials: 4,
          sampler: "tpe",
          seed: 0,
        },
        mode: "sync",
      })
      .catch((error) => error);

    // the typed error rides through untouched — the page's
    // describeError renders the detail verbatim
    expect(thrown).toBeInstanceOf(ApiError);
    expect((thrown as ApiError).status).toBe(422);
    expect((thrown as ApiError).detail).toContain(
      "the tune lane needs a search block",
    );
  });
});

describe("parseDLTuneReport — the strict parser", () => {
  it("returns null for the fixed-config lane's absence", () => {
    expect(parseDLTuneReport(null, "tune")).toBeNull();
    expect(parseDLTuneReport(undefined, "tune")).toBeNull();
  });

  it("throws on a trial row that lost its number/value pair", () => {
    const broken = {
      ...TUNE_ANSWER.tune,
      trials: [
        { number: 0, params: { learning_rate: 0.031 } },
      ],
    };
    expect(() => parseDLTuneReport(broken, "tune")).toThrow(
      /lost its number\/value pair/,
    );
  });

  it("throws on a report without its refit score", () => {
    const broken = { ...TUNE_ANSWER.tune } as Record<string, unknown>;
    delete broken["refit_score"];
    expect(() => parseDLTuneReport(broken, "tune")).toThrow(
      /are not the report's own types/,
    );
  });
});
