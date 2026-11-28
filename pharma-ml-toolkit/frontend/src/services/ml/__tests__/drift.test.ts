import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/services/api/client";
import { mlApi, parseDriftReport } from "@/services/ml/client";

/**
 * ML-5 (the roadmap's row 12) — the drift client operation as the
 * roadmap's own vitest pair: `drift` over the REAL client (the
 * network boundary alone is faked, so the strict parser rides the
 * actual code):
 *
 *   the dataset pair — the reference + current ids POST verbatim
 *                      (dataset_id + new_dataset_id), the answer
 *                      narrows field-by-field (payload narrowing);
 *   the model pair   — the model reference lane POSTs model_id +
 *                      new_dataset_id, the reference block narrows
 *                      with its kind/model_id/name;
 *   malformed-throw  — a feature row that lost its psi/verdict pair
 *                      and a report without its threshold constants
 *                      throw instead of rendering as if it were
 *                      honest;
 *   the honest 422   — the backend's own 422 (the exactly-one
 *                      reference contract, an unknown id) surfaces
 *                      verbatim through the client, never swallowed;
 *   the pre-flight   — both/neither reference ids are refused
 *                      client-side before the request fires (the
 *                      route's own contract mirrored).
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

const DRIFT_ANSWER = {
  mode: "sync",
  reference: {
    kind: "dataset",
    model_id: null,
    name: "training-batch.csv",
    dataset_id: "ds-ref",
    n_reference_rows: 400,
  },
  current: {
    dataset_id: "ds-cur",
    name: "fresh-batch.csv",
    n_rows: 400,
  },
  warn_psi: 0.1,
  shift_psi: 0.2,
  n_features: 2,
  n_drifted: 1,
  drift_share: 0.5,
  features: [
    {
      feature: "molwt",
      psi: 1.742,
      ks_statistic: 0.83,
      ks_pvalue: 3.1e-131,
      reference_mean: 300.1,
      current_mean: 455.2,
      mean_direction: "up",
      verdict: "shifted",
    },
    {
      feature: "tpsa",
      psi: 0.021,
      ks_statistic: 0.06,
      ks_pvalue: 0.42,
      reference_mean: 74.1,
      current_mean: 74.4,
      mean_direction: "flat",
      verdict: "stable",
    },
  ],
  drifted: [
    {
      feature: "molwt",
      psi: 1.742,
      ks_statistic: 0.83,
      ks_pvalue: 3.1e-131,
      reference_mean: 300.1,
      current_mean: 455.2,
      mean_direction: "up",
      verdict: "shifted",
    },
  ],
  summary:
    "drift: 1/2 feature(s) shifted vs the training frame (PSI >= 0.2)\n  molwt (PSI 1.742, mean up)",
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe("mlApi.drift — the drift client operation", () => {
  it("POSTs the dataset reference verbatim and narrows the answer", async () => {
    apiMocks.post.mockResolvedValueOnce(DRIFT_ANSWER);

    const result = await mlApi.drift({
      modelId: null,
      datasetId: "ds-ref",
      newDatasetId: "ds-cur",
    });

    expect(apiMocks.post).toHaveBeenCalledTimes(1);
    expect(apiMocks.post).toHaveBeenCalledWith(
      "/api/v1/ml/drift",
      {
        new_dataset_id: "ds-cur",
        dataset_id: "ds-ref",
      },
    );
    expect(result.mode).toBe("sync");
    expect(result.warn_psi).toBe(0.1);
    expect(result.shift_psi).toBe(0.2);
    expect(result.n_features).toBe(2);
    expect(result.n_drifted).toBe(1);
    expect(result.drift_share).toBe(0.5);
    expect(result.reference.kind).toBe("dataset");
    expect(result.reference.n_reference_rows).toBe(400);
    expect(result.current.dataset_id).toBe("ds-cur");
    expect(result.current.n_rows).toBe(400);
    expect(result.features).toHaveLength(2);
    expect(result.features[0]?.feature).toBe("molwt");
    expect(result.features[0]?.verdict).toBe("shifted");
    expect(result.features[0]?.ks_statistic).toBe(0.83);
    expect(result.features[1]?.verdict).toBe("stable");
    expect(result.drifted).toHaveLength(1);
    expect(result.drifted[0]?.feature).toBe("molwt");
    expect(result.summary).toContain("1/2 feature(s) shifted");
  });

  it("POSTs the model reference verbatim (model_id + new_dataset_id only)", async () => {
    apiMocks.post.mockResolvedValueOnce({
      ...DRIFT_ANSWER,
      reference: {
        kind: "model",
        model_id: "tm-drift",
        name: "random_forest",
        dataset_id: "ds-ref",
        n_reference_rows: 400,
      },
    });

    const result = await mlApi.drift({
      modelId: "tm-drift",
      datasetId: null,
      newDatasetId: "ds-cur",
    });

    expect(apiMocks.post).toHaveBeenCalledWith(
      "/api/v1/ml/drift",
      {
        new_dataset_id: "ds-cur",
        model_id: "tm-drift",
      },
    );
    expect(result.reference.kind).toBe("model");
    expect(result.reference.model_id).toBe("tm-drift");
    expect(result.reference.name).toBe("random_forest");
  });

  it("refuses both/neither references pre-flight, no request fired", async () => {
    await expect(
      mlApi.drift({
        modelId: "tm-drift",
        datasetId: "ds-ref",
        newDatasetId: "ds-cur",
      }),
    ).rejects.toThrow(/give either modelId or datasetId/);
    await expect(
      mlApi.drift({
        modelId: null,
        datasetId: null,
        newDatasetId: "ds-cur",
      }),
    ).rejects.toThrow(/give either modelId or datasetId/);
    expect(apiMocks.post).not.toHaveBeenCalled();
  });

  it("surfaces the backend's own 422 verbatim, never swallowed", async () => {
    apiMocks.post.mockRejectedValueOnce(
      new ApiError(
        404,
        "the model's training dataset ds-ref is no longer registered in this session; re-upload it or pass dataset_id=... explicitly",
      ),
    );

    const thrown = await mlApi
      .drift({
        modelId: "tm-drift",
        datasetId: null,
        newDatasetId: "ds-cur",
      })
      .catch((error) => error);

    // the typed error rides through untouched — the page's
    // describeError renders the detail verbatim
    expect(thrown).toBeInstanceOf(ApiError);
    expect((thrown as ApiError).status).toBe(404);
    expect((thrown as ApiError).detail).toBe(
      "the model's training dataset ds-ref is no longer registered in this session; re-upload it or pass dataset_id=... explicitly",
    );
  });
});

describe("parseDriftReport — the strict parser", () => {
  it("throws on a feature row that lost its psi/verdict pair", () => {
    const broken = {
      ...DRIFT_ANSWER,
      features: [
        {
          feature: "molwt",
          psi: 1.742,
          ks_statistic: 0.83,
          ks_pvalue: 3.1e-131,
          reference_mean: 300.1,
          current_mean: 455.2,
          mean_direction: "up",
          // verdict lost — a numberless silence must throw
        },
      ],
    };
    expect(() => parseDriftReport(broken, "drift")).toThrow(
      /lost its psi\/verdict pair/,
    );
  });

  it("throws on a report without its threshold constants", () => {
    const broken = { ...DRIFT_ANSWER };
    delete (broken as Record<string, unknown>)["warn_psi"];
    expect(() => parseDriftReport(broken, "drift")).toThrow(
      /warn_psi\/shift_psi are not numbers/,
    );
  });

  it("narrows the honest no-shared-features zero answer", () => {
    const empty = {
      ...DRIFT_ANSWER,
      n_features: 0,
      n_drifted: 0,
      drift_share: 0.0,
      features: [],
      drifted: [],
      summary: "drift: no numeric features compared",
    };
    const result = parseDriftReport(empty, "drift");
    expect(result.n_features).toBe(0);
    expect(result.features).toHaveLength(0);
    expect(result.summary).toContain("no numeric features");
  });
});
