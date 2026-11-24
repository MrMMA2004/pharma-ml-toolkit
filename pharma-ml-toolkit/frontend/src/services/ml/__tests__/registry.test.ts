/**
 * part16 §15.3 (Phase 12) — the model-registry client operations'
 * own contract.
 *
 * The registry projection's three additions to the ML client (the
 * catalog's full listing, the evict lane, the export lane) narrowed
 * through the house parse pattern: the transport is mocked at the
 * module boundary only — the request shapes, the parse guards, and
 * the honest error verdicts for drifted payloads are the real ones.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

const getMock = vi.hoisted(() => vi.fn());
const postMock = vi.hoisted(() => vi.fn());
const deleteMock = vi.hoisted(() => vi.fn());
vi.mock("@/services/api/client", () => ({
  api: {
    get: (path: string) => getMock(path),
    post: (path: string, body?: unknown) => postMock(path, body),
    delete: (path: string) => deleteMock(path),
  },
}));

import { mlApi } from "@/services/ml/client";

const CATALOG_ENTRY = {
  name: "logistic_regression",
  task: "classification",
  available: true,
  requires: null,
  notes: "the linear baseline",
};

const TRAINED_VIEW = {
  model_id: "mdl-4f0a9c21b7d3",
  name: "logistic_regression",
  task: "classification",
  dataset_id: "ds-20250601-0001",
  target: "activity",
  features: ["molwt", "logp"],
  n_train_rows: 96,
  best_params: { C: 0.5 },
  cv_scores: [0.91, 0.92],
  cv_mean: 0.915,
  cv_std: 0.01,
  fit_seconds: 0.21,
  created_at: "2025-06-01T10:00:03+00:00",
};

const ARTIFACT_VIEW = {
  artifact_id: "art-20250601-0001",
  kind: "model",
  filename: "logistic_regression-model.joblib",
  content_type: "application/octet-stream",
  size_bytes: 1837,
  sha256:
    "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f",
  created_at: "2025-06-01T10:01:00+00:00",
  meta: { model_name: "logistic_regression", task: "classification" },
  download_url: "/api/v1/artifacts/art-20250601-0001/download",
};

beforeEach(() => {
  getMock.mockReset();
  postMock.mockReset();
  deleteMock.mockReset();
});

describe("mlApi.catalog — the §15.3 full listing (part16 Phase 12)", () => {
  it("omits the task parameter when the whole catalog is asked for", async () => {
    getMock.mockResolvedValue({ models: [CATALOG_ENTRY] });
    const answer = await mlApi.catalog(undefined, false);
    expect(answer.models).toHaveLength(1);
    expect(getMock).toHaveBeenCalledWith(
      "/api/v1/ml/models?only_available=false",
    );
  });

  it("keeps the task parameter on the filtered listing", async () => {
    getMock.mockResolvedValue({ models: [CATALOG_ENTRY] });
    await mlApi.catalog("classification", true);
    expect(getMock).toHaveBeenCalledWith(
      "/api/v1/ml/models?task=classification&only_available=true",
    );
  });

  it("narrows the entries — the five catalog keys, the honest requires null", async () => {
    getMock.mockResolvedValue({
      models: [
        CATALOG_ENTRY,
        {
          name: "xgboost",
          task: "both",
          available: false,
          requires: "xgboost",
          notes: "the boosted lane",
        },
      ],
    });
    const answer = await mlApi.catalog(undefined, false);
    expect(answer.models[0]).toEqual(CATALOG_ENTRY);
    expect(answer.models[1]?.available).toBe(false);
    expect(answer.models[1]?.requires).toBe("xgboost");
  });

  it("degrades a missing models array to the empty listing", async () => {
    getMock.mockResolvedValue({});
    const answer = await mlApi.catalog(undefined, false);
    expect(answer.models).toEqual([]);
  });
});

describe("mlApi.evictTrained — the registry's deletion lane (§15.3)", () => {
  it("targets the trained-model path and narrows the evict answer", async () => {
    deleteMock.mockResolvedValue({ evicted: TRAINED_VIEW.model_id, remaining: 2 });
    const answer = await mlApi.evictTrained(TRAINED_VIEW.model_id);
    expect(deleteMock).toHaveBeenCalledWith(
      `/api/v1/ml/trained/${TRAINED_VIEW.model_id}`,
    );
    expect(answer).toEqual({ evicted: TRAINED_VIEW.model_id, remaining: 2 });
  });

  it("URL-encodes the model id", async () => {
    deleteMock.mockResolvedValue({ evicted: "a b", remaining: 0 });
    await mlApi.evictTrained("a b");
    expect(deleteMock).toHaveBeenCalledWith("/api/v1/ml/trained/a%20b");
  });

  it("refuses a drifted evict answer (no silent undefined)", async () => {
    deleteMock.mockResolvedValue({ evicted: 7 });
    await expect(mlApi.evictTrained("mdl-x")).rejects.toThrow(
      /evicted field is not a string/,
    );
  });

  it("refuses an answer missing the remaining count", async () => {
    deleteMock.mockResolvedValue({ evicted: "mdl-x" });
    await expect(mlApi.evictTrained("mdl-x")).rejects.toThrow(
      /remaining field is not a number/,
    );
  });
});

describe("mlApi.exportTrained — the registry's export lane (§15.3)", () => {
  it("posts the model id to the artifacts route and narrows the artifact", async () => {
    postMock.mockResolvedValue({ artifact: ARTIFACT_VIEW });
    const artifact = await mlApi.exportTrained(TRAINED_VIEW.model_id);
    expect(postMock).toHaveBeenCalledWith("/api/v1/artifacts/model", {
      model_id: TRAINED_VIEW.model_id,
    });
    expect(artifact.artifact_id).toBe(ARTIFACT_VIEW.artifact_id);
    expect(artifact.kind).toBe("model");
    expect(artifact.filename.endsWith(".joblib")).toBe(true);
  });

  it("carries the optional filename through the request body", async () => {
    postMock.mockResolvedValue({ artifact: ARTIFACT_VIEW });
    await mlApi.exportTrained(TRAINED_VIEW.model_id, "baseline");
    expect(postMock).toHaveBeenCalledWith("/api/v1/artifacts/model", {
      model_id: TRAINED_VIEW.model_id,
      filename: "baseline",
    });
  });

  it("refuses a drifted export answer (the artifact block missing)", async () => {
    postMock.mockResolvedValue({});
    await expect(mlApi.exportTrained("mdl-x")).rejects.toThrow();
  });
});
