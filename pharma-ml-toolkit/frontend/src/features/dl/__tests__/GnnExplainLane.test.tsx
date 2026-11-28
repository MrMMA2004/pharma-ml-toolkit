import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DLPage } from "@/features/dl/DLPage";
import { ApiError } from "@/services/api/client";
import type { JobView } from "@/services/jobs/types";

/**
 * DL-2 (the WORKFLOW_ENHANCEMENT_ROADMAP's row 14) — the GNN lane's
 * edge-attribution card, as a complete component-level user flow:
 *
 *   the armed lane        — the explain card arms off the trained
 *                          GNN model (no card before the train); the
 *                          graphs paste + the method/top_k/seed
 *                          controls ride the request;
 *   the edge table        — the explained rows render the per-edge
 *                          importance table with the top rows
 *                          highlighted (data-top="true") + the
 *                          PlotlyFigure sized-edge scatter stub;
 *   the pre-flight        — the pgexplainer labels lane (labels with
 *                          another method refused here, the route's
 *                          own contract mirrored, no request fired);
 *   the honest 422        — the backend's own refusal (the attention
 *                          lane's GAT requirement) surfaces verbatim,
 *                          never swallowed.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parser (parseGnnExplainResult —
 * the REAL one, spread from the actual module), and the DOM are the
 * real ones.
 */

const jobState = vi.hoisted(() => {
  const state: {
    snapshots: Record<string, { view: Record<string, unknown> } | null>;
    cancels: string[];
  } = { snapshots: {}, cancels: [] };
  return state;
});

vi.mock("@/hooks/useJobTracking", () => ({
  useJob: (jobId: string | null) => ({
    snapshot: jobId === null ? null : jobState.snapshots[jobId] ?? null,
    cancel: jobId === null
      ? vi.fn()
      : vi.fn(async () => {
          jobState.cancels.push(jobId);
        }),
    refresh: vi.fn(),
  }),
}));

const dlMocks = vi.hoisted(() => ({
  networks: vi.fn(),
  train: vi.fn(),
  gnnTrain: vi.fn(),
  gnnPredict: vi.fn(),
  gnnExplain: vi.fn(),
  lossCurveFigure: vi.fn(),
  saveCheckpoint: vi.fn(),
}));

vi.mock("@/services/dl/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/dl/client")>();
  // the REAL parsers ride the page under test; only the network
  // boundary is faked
  return {
    ...actual,
    dlApi: {
      networks: dlMocks.networks,
      train: dlMocks.train,
      gnnTrain: dlMocks.gnnTrain,
      gnnPredict: dlMocks.gnnPredict,
      gnnExplain: dlMocks.gnnExplain,
      lossCurveFigure: dlMocks.lossCurveFigure,
      saveCheckpoint: dlMocks.saveCheckpoint,
    },
  };
});

const datasetsMocks = vi.hoisted(() => ({ list: vi.fn(), view: vi.fn() }));
vi.mock("@/services/datasets/client", () => ({
  datasetsApi: { list: datasetsMocks.list, view: datasetsMocks.view },
}));

const jobsMocks = vi.hoisted(() => ({ result: vi.fn() }));
vi.mock("@/services/jobs/client", () => ({
  jobsApi: { result: jobsMocks.result },
}));

const mlMocks = vi.hoisted(() => ({ predict: vi.fn() }));
vi.mock("@/services/ml/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/ml/client")>();
  return {
    ...actual,
    mlApi: { ...actual.mlApi, predict: mlMocks.predict },
  };
});

vi.mock("@/services/websocket/client", () => ({
  getEventStream: () => ({
    subscribe: () => () => undefined,
  }),
}));

vi.mock("@/services/artifacts/client", () => ({
  artifactsApi: { download: vi.fn() },
  parseArtifactView: vi.fn(),
}));

vi.mock("@/components/charts/PlotlyFigure", () => ({
  PlotlyFigure: () => <div data-testid="plotly-stub" />,
}));

const CATALOG = {
  networks: [],
  predict_url: "/api/v1/ml/predict",
  note: "predictions for a DL-trained model run through the single session prediction path",
  gnn: {
    convs: ["gcn", "sage", "gin", "gat", "edge"],
    pools: ["mean", "sum", "max", "topk", "sag"],
    defaults: { hidden: 16, layers: 2, lr: 0.05, epochs: 200 },
    train_url: "/api/v1/dl/gnn-train",
    predict_url: "/api/v1/dl/gnn-predict",
    note: "graph models predict through their own path — never the tabular one",
  },
};

const GNN_TRAIN_RESULT = {
  model: {
    model_id: "mdl-gnnaa000111",
    name: "gnn_graph_classifier",
    task: "classification",
    dataset_id: "",
    target: "",
    features: [],
    n_train_rows: 18,
    best_params: { network: "gnn", conv: "gcn", pool: "mean" },
    cv_scores: null,
    cv_mean: null,
    cv_std: null,
    fit_seconds: 2.5,
    created_at: "2026-11-21T00:00:00Z",
    engine: "torch",
    graph_model: true,
    predict_url: "/api/v1/dl/gnn-predict",
  },
  train_accuracy: 1.0,
  loss_history_tail: [0.71, 0.52, 0.33, 0.14, 0.05],
  n_graphs: 18,
  classes: [0, 1, 2],
  conv: "gcn",
  pool: "mean",
};

const EXPLAIN_GRAPHS = JSON.stringify({
  graphs: [
    {
      x: [
        [1, 0], [0, 1], [1, 1], [0, 0], [1, 0.5],
      ],
      edge_index: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [1, 3]],
    },
    {
      x: [
        [0, 1], [1, 0], [0.5, 0.5],
      ],
      edge_index: [[0, 1], [1, 2], [2, 0]],
    },
  ],
});

const EXPLAIN_ANSWER = {
  mode: "sync",
  model_id: "mdl-gnnaa000111",
  method: "gnnexplainer",
  top_k: 4,
  seed: 0,
  graphs: [
    {
      index: 0,
      n_nodes: 5,
      n_edges: 6,
      target_class: 1,
      method: "gnnexplainer",
      scores: [0.9, 0.2, 0.75, 0.1, 0.5, 0.3],
      top_edges: [0, 2],
      fidelity_plus: 0.12,
      fidelity_minus: -0.03,
      sparsity: 1 - 2 / 6,
      extra: { iters: 20, final_loss: 0.4 },
      edge_index: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [1, 3]],
      layout: {
        kind: "circular",
        nodes: [
          [1.0, 0.0], [0.309, 0.951], [-0.809, 0.588],
          [-0.809, -0.588], [0.309, -0.951],
        ],
      },
    },
    {
      index: 1,
      n_nodes: 3,
      n_edges: 3,
      target_class: 0,
      method: "gnnexplainer",
      scores: [0.4, 0.8, 0.1],
      top_edges: [1],
      fidelity_plus: 0.05,
      fidelity_minus: -0.01,
      sparsity: 1 - 1 / 3,
      extra: { iters: 20, final_loss: 0.3 },
      edge_index: [[0, 1], [1, 2], [2, 0]],
      layout: {
        kind: "circular",
        nodes: [[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]],
      },
    },
  ],
  note: "the per-edge scores are aligned with each graph's own edge_index columns",
};

function completedJobView(jobId: string): JobView {
  return {
    job_id: jobId,
    operation: "dl.gnn-train",
    category: "deep-learning",
    state: "COMPLETED",
    progress: 100,
    stage: "done",
    error: null,
    created_at: "2026-11-21T00:00:00Z",
    started_at: "2026-11-21T00:00:01Z",
    finished_at: "2026-11-21T00:00:04Z",
    configuration: {},
    result_ready: true,
  } as unknown as JobView;
}

beforeEach(() => {
  vi.clearAllMocks();
  jobState.snapshots = {};
  jobState.cancels = [];
  dlMocks.networks.mockResolvedValue(CATALOG);
  datasetsMocks.list.mockResolvedValue({ datasets: [] });
  dlMocks.gnnTrain.mockResolvedValue({
    mode: "job",
    job: completedJobView("job-gnn-x"),
  });
  jobState.snapshots["job-gnn-x"] = {
    view: completedJobView("job-gnn-x") as unknown as Record<string, unknown>,
  };
  jobsMocks.result.mockResolvedValue({ result: GNN_TRAIN_RESULT });
});

async function trainAndArm(user: ReturnType<typeof userEvent.setup>) {
  render(<DLPage />);
  await user.click(
    await screen.findByRole("button", { name: "Train the graph model" }),
  );
  // the train job completes → the model card + the explain card arm
  await waitFor(() => {
    expect(
      screen.getByText("Edge attribution (GNN explainability)"),
    ).toBeInTheDocument();
  });
}

describe("DL-2 — the GNN lane's edge-attribution card", () => {
  it("rides the explain route with the parsed graphs and renders the edge table + the sized-edge scatter", async () => {
    const user = userEvent.setup();
    await trainAndArm(user);
    dlMocks.gnnExplain.mockResolvedValueOnce(EXPLAIN_ANSWER);

    fireEvent.change(screen.getByLabelText("the graphs to explain"), {
      target: { value: EXPLAIN_GRAPHS },
    });
    await user.click(
      screen.getByRole("button", { name: "Explain the edges" }),
    );

    expect(dlMocks.gnnExplain).toHaveBeenCalledTimes(1);
    const call = dlMocks.gnnExplain.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(call["modelId"]).toBe("mdl-gnnaa000111");
    expect(call["method"]).toBe("gnnexplainer");
    expect(call["topK"]).toBe(8);
    expect(call["seed"]).toBe(0);
    expect(call["iters"]).toBeNull();
    expect(call["target"]).toBeNull();
    expect(call["labels"]).toBeNull();
    expect((call["graphs"] as unknown[]).length).toBe(2);

    // the per-graph rows: the fidelity line + the edge table with the
    // top rows highlighted + the PlotlyFigure sized-edge scatter stub
    expect(
      await screen.findAllByText(/graph 0 · 5 nodes · 6 edges/),
    ).toHaveLength(1);
    expect(screen.getByText(/fidelity\+ 0\.12/)).toBeInTheDocument();
    // both graphs land the same sparsity (2/6 and 1/3) — two rows
    expect(screen.getAllByText(/sparsity 0\.6667/)).toHaveLength(2);
    // graph 0 renders first: its "0→1" row IS top (top_edges [0, 2]),
    // graph 1's "0→1" row is not — the same edge text in both tables
    const zeroOne = screen.getAllByText("0→1");
    expect(zeroOne[0]?.closest("tr")).toHaveAttribute("data-top", "true");
    expect(zeroOne[1]?.closest("tr")).not.toHaveAttribute("data-top");
    const oneTwo = screen.getAllByText("1→2");
    expect(oneTwo[0]?.closest("tr")).not.toHaveAttribute("data-top");
    // both graphs render their own sized-edge scatter (the Plotly stub)
    expect(screen.getAllByTestId("plotly-stub").length).toBeGreaterThanOrEqual(2);
  });

  it("refuses labels with a non-pgexplainer method pre-flight, no request fired", async () => {
    const user = userEvent.setup();
    await trainAndArm(user);

    fireEvent.change(screen.getByLabelText("the graphs to explain"), {
      target: { value: EXPLAIN_GRAPHS },
    });
    fireEvent.change(screen.getByLabelText("labels (pgexplainer only — JSON list)"), {
      target: { value: "[0, 1]" },
    });
    await user.click(
      screen.getByRole("button", { name: "Explain the edges" }),
    );

    expect(dlMocks.gnnExplain).not.toHaveBeenCalled();
    expect(
      await screen.findByText(/labels ride the pgexplainer lane only/),
    ).toBeInTheDocument();
  });

  it("surfaces the backend's own honest 422 verbatim, never swallowed", async () => {
    const user = userEvent.setup();
    await trainAndArm(user);

    fireEvent.change(screen.getByLabelText("the graphs to explain"), {
      target: { value: EXPLAIN_GRAPHS },
    });
    dlMocks.gnnExplain.mockRejectedValueOnce(
      new ApiError(
        422,
        "AttentionExplainer: the model has no GAT layer",
      ),
    );
    // pick the attention method (a non-GAT model — the engine's own
    // refusal rides verbatim)
    await user.selectOptions(
      screen.getByLabelText("the explain method"),
      "attention",
    );
    await user.click(
      screen.getByRole("button", { name: "Explain the edges" }),
    );

    expect(
      await screen.findByText(/the model has no GAT layer/),
    ).toBeInTheDocument();
  });
});
