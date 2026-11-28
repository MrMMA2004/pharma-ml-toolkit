import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/services/api/client";
import { dlApi, parseGnnExplainResult } from "@/services/dl/client";

/**
 * DL-2 (the roadmap's row 14) — the gnn-explain client operation as
 * the roadmap's own vitest pair: `gnnExplain` over the REAL client
 * (the network boundary alone is faked, so the strict parser rides
 * the actual code):
 *
 *   the request pair   — the model_id + the graphs + the method /
 *                        top_k / seed / target / labels / iters
 *                        POST verbatim (nulls dropped, never sent);
 *   the parser pair    — the answer narrows graph-by-graph (the
 *                        scores must align with the served
 *                        edge_index — a disagreement throws before
 *                        any scatter geometry could be fabricated);
 *   malformed-throw    — a graph row that lost its fidelity verdicts
 *                        and a scores/edge_index disagreement throw;
 *   the honest 422     — the engine's own refusal (the GAT
 *                        requirement) rides through untouched.
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

const GRAPHS = [
  {
    x: [[1, 0], [0, 1], [1, 1]],
    edge_index: [[0, 1], [1, 2], [2, 0]],
  },
];

const ANSWER = {
  mode: "sync",
  model_id: "mdl-gnnaa000111",
  method: "gnnexplainer",
  top_k: 4,
  seed: 0,
  graphs: [
    {
      index: 0,
      n_nodes: 3,
      n_edges: 3,
      target_class: 1,
      method: "gnnexplainer",
      scores: [0.9, 0.2, 0.75],
      top_edges: [0, 2],
      fidelity_plus: 0.12,
      fidelity_minus: -0.03,
      sparsity: 1 / 3,
      extra: { iters: 20, final_loss: 0.4 },
      edge_index: [[0, 1], [1, 2], [2, 0]],
      layout: {
        kind: "circular",
        nodes: [[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]],
      },
    },
  ],
  note: "the per-edge scores are aligned with each graph's own edge_index columns",
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe("dlApi.gnnExplain — the DL-2 explain client operation", () => {
  it("POSTs the graphs + the method controls verbatim and narrows the answer", async () => {
    apiMocks.post.mockResolvedValueOnce(ANSWER);

    const result = await dlApi.gnnExplain({
      modelId: "mdl-gnnaa000111",
      graphs: GRAPHS,
      method: "gnnexplainer",
      topK: 4,
      seed: 0,
      target: 1,
      labels: null,
      iters: 20,
    });

    expect(apiMocks.post).toHaveBeenCalledTimes(1);
    expect(apiMocks.post).toHaveBeenCalledWith("/api/v1/dl/gnn-explain", {
      model_id: "mdl-gnnaa000111",
      graphs: GRAPHS,
      method: "gnnexplainer",
      top_k: 4,
      seed: 0,
      target: 1,
      iters: 20,
    });

    expect(result.mode).toBe("sync");
    expect(result.model_id).toBe("mdl-gnnaa000111");
    expect(result.method).toBe("gnnexplainer");
    expect(result.graphs).toHaveLength(1);
    const row = result.graphs[0];
    expect(row?.n_nodes).toBe(3);
    expect(row?.n_edges).toBe(3);
    expect(row?.scores).toEqual([0.9, 0.2, 0.75]);
    expect(row?.top_edges).toEqual([0, 2]);
    expect(row?.fidelity_plus).toBe(0.12);
    expect(row?.fidelity_minus).toBe(-0.03);
    expect(row?.sparsity).toBeCloseTo(1 / 3);
    expect(row?.layout.nodes).toHaveLength(3);
    expect(row?.edge_index).toEqual([[0, 1], [1, 2], [2, 0]]);
  });

  it("drops the null target/labels/iters from the request verbatim", async () => {
    apiMocks.post.mockResolvedValueOnce(ANSWER);

    await dlApi.gnnExplain({
      modelId: "mdl-gnnaa000111",
      graphs: GRAPHS,
      method: "subgraphx",
      topK: 8,
      seed: 0,
      target: null,
      labels: null,
      iters: null,
    });

    expect(apiMocks.post).toHaveBeenCalledWith("/api/v1/dl/gnn-explain", {
      model_id: "mdl-gnnaa000111",
      graphs: GRAPHS,
      method: "subgraphx",
      top_k: 8,
      seed: 0,
    });
  });

  it("surfaces the engine's own honest refusal untouched", async () => {
    apiMocks.post.mockRejectedValueOnce(
      new ApiError(422, "AttentionExplainer: the model has no GAT layer"),
    );

    const thrown = await dlApi
      .gnnExplain({
        modelId: "mdl-gnnaa000111",
        graphs: GRAPHS,
        method: "attention",
        topK: 8,
        seed: 0,
        target: null,
        labels: null,
        iters: null,
      })
      .catch((error) => error);

    expect(thrown).toBeInstanceOf(ApiError);
    expect((thrown as ApiError).status).toBe(422);
    expect((thrown as ApiError).detail).toBe(
      "AttentionExplainer: the model has no GAT layer",
    );
  });
});

describe("parseGnnExplainResult — the strict parser", () => {
  it("throws on a graph row that lost its fidelity verdicts", () => {
    const broken = {
      ...ANSWER,
      graphs: [
        {
          ...ANSWER.graphs[0],
          fidelity_plus: null,
        },
      ],
    };
    expect(() => parseGnnExplainResult(broken, "gnn-explain")).toThrow(
      /lost its counts\/verdicts/,
    );
  });

  it("throws on a scores/edge_index disagreement (fabricated geometry)", () => {
    const broken = {
      ...ANSWER,
      graphs: [
        {
          ...ANSWER.graphs[0],
          scores: [0.9, 0.2],
        },
      ],
    };
    expect(() => parseGnnExplainResult(broken, "gnn-explain")).toThrow(
      /disagree/,
    );
  });

  it("throws on a layout that lost its node positions", () => {
    const broken = {
      ...ANSWER,
      graphs: [
        {
          ...ANSWER.graphs[0],
          layout: { kind: "circular", nodes: [[1.0, 0.0]] },
        },
      ],
    };
    expect(() => parseGnnExplainResult(broken, "gnn-explain")).toThrow(
      /lost its node positions/,
    );
  });
});
