import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/services/api/client";
import { mlApi } from "@/services/ml/client";

/**
 * ML-2 (the roadmap's row 10) — the two new client operations as the
 * roadmap's own vitest pair: `knnDomain` and `domainBoxCheck` over the
 * REAL client (the network boundary alone is faked, so the strict
 * parsers ride the actual code):
 *
 *   the knn pair     — the two matrices POST verbatim, the answer
 *                      narrows field-by-field (payload narrowing);
 *   the box pair     — the metadata + rows POST verbatim, the checked
 *                      report narrows (counts + below/above + the out
 *                      mask), the unchecked answer carries the route's
 *                      own reason;
 *   malformed-throw  — a frame row that lost its verdict pair and a
 *                      report without its boolean throw instead of
 *                      rendering as if it were honest;
 *   the honest 422   — the backend's own 422 (a feature-count
 *                      mismatch, an empty-rows refusal) surfaces
 *                      verbatim through the client, never swallowed.
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

const X_TRAIN = [
  [1.2, 0.4, 38.5],
  [1.6, 0.7, 41.2],
  [2.1, 1.1, 44.9],
  [2.8, 1.4, 47.0],
  [3.3, 1.9, 49.8],
  [3.9, 2.2, 52.6],
  [4.4, 2.6, 55.1],
  [5.0, 3.0, 57.4],
  [5.7, 3.4, 59.9],
  [6.2, 3.7, 62.3],
];

const X_TEST = [
  [1.4, 0.5, 39.1],
  [12.9, 9.5, 118.6],
];

const KNN_ANSWER = {
  k: 5,
  percentile: 95,
  distance_threshold: 1.204,
  train_knn_mean: 0.612,
  n_train: 10,
  n_test: 2,
  n_inside: 1,
  frame: [
    {
      name: "compound_0",
      predicted: null,
      prediction_sd: null,
      knn_distance: 0.31,
      inside_ad: true,
      low_uncertainty: null,
      reliable: null,
    },
    {
      name: "compound_1",
      predicted: null,
      prediction_sd: null,
      knn_distance: 9.7,
      inside_ad: false,
      low_uncertainty: null,
      reliable: null,
    },
  ],
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe("mlApi.knnDomain — the distance-only client operation", () => {
  it("POSTs the two raw matrices verbatim and narrows the answer", async () => {
    apiMocks.post.mockResolvedValueOnce(KNN_ANSWER);

    const result = await mlApi.knnDomain({
      xTrain: X_TRAIN,
      xTest: X_TEST,
      k: 5,
      percentile: 95,
      metric: "euclidean",
    });

    expect(apiMocks.post).toHaveBeenCalledTimes(1);
    const [path, body] = apiMocks.post.mock.calls[0] as unknown[];
    expect(path).toBe("/api/v1/applicability-domain/knn");
    expect(body).toStrictEqual({
      x_train: X_TRAIN,
      x_test: X_TEST,
      k: 5,
      percentile: 95,
      metric: "euclidean",
    });
    // payload narrowing: the typed verdict, the uncertainty nulls
    // dropped from this lane's frame rows
    expect(result.k).toBe(5);
    expect(result.n_train).toBe(10);
    expect(result.n_test).toBe(2);
    expect(result.n_inside).toBe(1);
    expect(result.frame).toEqual([
      { name: "compound_0", knn_distance: 0.31, inside_ad: true },
      { name: "compound_1", knn_distance: 9.7, inside_ad: false },
    ]);
  });

  it("throws on a frame row that lost its verdict pair — never renders as if honest", async () => {
    apiMocks.post.mockResolvedValueOnce({
      ...KNN_ANSWER,
      frame: [{ name: "compound_0", knn_distance: 0.31 }],
    });

    await expect(
      mlApi.knnDomain({
        xTrain: X_TRAIN,
        xTest: X_TEST,
        k: 5,
        percentile: 95,
        metric: "euclidean",
      }),
    ).rejects.toThrow(/lost its knn_distance\/inside_ad verdict/);
  });

  it("surfaces the backend's own 422 verbatim — never a silent failure", async () => {
    apiMocks.post.mockRejectedValueOnce(
      new ApiError(422, {
        detail: "x_train has 3 features; x_test has 2 (they must match)",
      }),
    );

    const thrown = await mlApi
      .knnDomain({
        xTrain: X_TRAIN,
        xTest: [[1.0, 2.0]],
        k: 5,
        percentile: 95,
        metric: "euclidean",
      })
      .catch((error) => error);

    // the typed error rides through untouched — the page's
    // describeError renders the detail verbatim
    expect(thrown).toBeInstanceOf(ApiError);
    expect((thrown as ApiError).status).toBe(422);
    expect((thrown as ApiError).detail).toEqual({
      detail: "x_train has 3 features; x_test has 2 (they must match)",
    });
  });
});

describe("mlApi.domainBoxCheck — the box-check client operation", () => {
  const METADATA = {
    name: "rf-solubility",
    task: "regression",
    feature_ranges: {
      molwt: { min: 130.2, max: 614.6 },
      logp: { min: -2.1, max: 7.4 },
    },
  };

  const ROWS = [
    { molwt: 250.3, logp: 2.4 },
    { molwt: 720.9, logp: -3.5 },
  ];

  it("POSTs the metadata + rows verbatim and narrows the checked report", async () => {
    apiMocks.post.mockResolvedValueOnce({
      checked: true,
      n_rows: 2,
      n_out: 1,
      fraction_out: 0.5,
      below: { logp: 1 },
      above: { molwt: 1 },
      out: [false, true],
    });

    const report = await mlApi.domainBoxCheck(METADATA, ROWS);

    const [path, body] = apiMocks.post.mock.calls[0] as unknown[];
    expect(path).toBe("/api/v1/applicability-domain/domain-box");
    expect(body).toStrictEqual({ metadata: METADATA, rows: ROWS });
    expect(report.checked).toBe(true);
    expect(report.n_out).toBe(1);
    expect(report.fraction_out).toBe(0.5);
    expect(report.below).toEqual({ logp: 1 });
    expect(report.above).toEqual({ molwt: 1 });
    expect(report.out).toEqual([false, true]);
  });

  it("narrows the unchecked answer with the route's own reason", async () => {
    apiMocks.post.mockResolvedValueOnce({
      checked: false,
      reason: "the metadata carries no feature_ranges box",
      n_rows: 2,
    });

    const report = await mlApi.domainBoxCheck({ name: "old-bundle" }, ROWS);

    expect(report.checked).toBe(false);
    expect(report.reason).toBe("the metadata carries no feature_ranges box");
    expect(report.n_rows).toBe(2);
    expect(report.n_out).toBeNull();
    expect(report.out).toBeNull();
  });

  it("throws on a report that lost its boolean — never renders as if honest", async () => {
    apiMocks.post.mockResolvedValueOnce({ n_rows: 2, n_out: 1 });

    await expect(
      mlApi.domainBoxCheck(METADATA, ROWS),
    ).rejects.toThrow(/checked is not a boolean/);
  });
});
