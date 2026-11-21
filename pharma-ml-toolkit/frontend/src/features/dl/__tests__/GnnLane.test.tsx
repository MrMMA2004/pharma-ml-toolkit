import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DLPage } from "@/features/dl/DLPage";
import { ApiError } from "@/services/api/client";
import type { JobView } from "@/services/jobs/types";

/**
 * part13 Phase 5 — the GNN/MPNN lane of the Deep Learning workspace
 * as a complete component-level user flow:
 *
 *   the live vocabulary  — the conv/pool selects read the backend's
 *                         own CONV_NAMES/POOL_NAMES off the catalog's
 *                         gnn block (never re-declared);
 *   the train submission — the synthetic fixture's honest request
 *                         shape (motif count, seed, conv, pool, the
 *                         trainer's own controls, job mode);
 *   the job relay       — the COMPLETED job's authoritative REST
 *                         result relays into the graph model card
 *                         (model view + train accuracy + the honest
 *                         loss_history tail, the graph-model badge);
 *   the predict lane    — the graph models' OWN prediction path
 *                         (never the tabular /ml/predict), with the
 *                         class-probability columns;
 *   typed errors        — the backend's own 422s surface verbatim;
 *   client-side gating  — a malformed inline block never leaves the
 *                         page (validated BEFORE any request).
 *
 * Services are mocked at the network boundary only; the component, its
 * state machine, the runtime parsers (parseGnnInlineGraphs /
 * parseGnnTrainResult — the REAL ones, spread from the actual module),
 * and the DOM are the real ones.
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
  // the REAL dl client imports parseTrainedModelView from this
  // module — the mock keeps it (spread from the actual module) so
  // the page's runtime parsers run the real code, while only the
  // predict boundary is faked
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

const GNN_CATALOG = {
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
  classes: [0, 1],
  conv: "gcn",
  pool: "mean",
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
  dlMocks.networks.mockResolvedValue(GNN_CATALOG);
  datasetsMocks.list.mockResolvedValue({ datasets: [] });
  dlMocks.gnnTrain.mockResolvedValue({
    mode: "job",
    job: completedJobView("job-gnn-1"),
  });
  jobState.snapshots["job-gnn-1"] = {
    view: completedJobView("job-gnn-1") as unknown as Record<string, unknown>,
  };
  jobsMocks.result.mockResolvedValue({ result: GNN_TRAIN_RESULT });
});

describe("the GNN lane — the live vocabulary + the honest empty state", () => {
  it("renders the lane beside the tabular workflow, with the catalog's own conv/pool tokens", async () => {
    render(<DLPage />);
    expect(
      await screen.findByRole("heading", { name: "GNN / MPNN — the graph lane" }),
    ).toBeVisible();
    // the conv/pool vocabularies come from the SAME catalog call the
    // architecture stage rides — the backend's own CONV_NAMES /
    // POOL_NAMES, never re-declared in the page
    const convSelect = screen.getByLabelText("gnn conv");
    const convTokens = Array.from(
      convSelect.querySelectorAll("option"),
    ).map((option) => option.value);
    expect(convTokens).toEqual(["gcn", "sage", "gin", "gat", "edge"]);
    const poolSelect = screen.getByLabelText("gnn pool");
    const poolTokens = Array.from(
      poolSelect.querySelectorAll("option"),
    ).map((option) => option.value);
    expect(poolTokens).toEqual(["mean", "sum", "max", "topk", "sag"]);
  });

  it("states the no-fabricated-epoch-stream honesty where the lane lives", async () => {
    render(<DLPage />);
    await screen.findByRole("heading", { name: "GNN / MPNN — the graph lane" });
    // the torch GNN trainer exposes loss_history_ only AFTER the fit —
    // the lane never fabricates a per-epoch stream
    expect(
      screen.getByText(/never\s+fabricates a per-epoch stream/i),
    ).toBeVisible();
    expect(
      screen.getByText(/HONEST stage-level telemetry/i),
    ).toBeVisible();
  });
});

describe("the GNN train lane — submission, relay, the graph model card", () => {
  it("submits the synthetic fixture's honest request shape in job mode", async () => {
    const user = userEvent.setup();
    render(<DLPage />);
    const trainButton = await screen.findByRole("button", {
      name: "Train the graph model",
    });
    expect(trainButton).toBeEnabled();
    await user.click(trainButton);
    await waitFor(() => {
      expect(dlMocks.gnnTrain).toHaveBeenCalledTimes(1);
    });
    expect(dlMocks.gnnTrain).toHaveBeenCalledWith({
      syntheticN: 18,
      seed: 11,
      inline: null,
      conv: "gcn",
      pool: "mean",
      hidden: 16,
      layers: 2,
      lr: 0.08,
      epochs: 120,
      weightDecay: 0,
      heads: 1,
      ratio: 0.5,
      mode: "job",
    });
  });

  it("relays the COMPLETED job's authoritative REST result into the graph model card", async () => {
    const user = userEvent.setup();
    render(<DLPage />);
    await user.click(
      await screen.findByRole("button", { name: "Train the graph model" }),
    );
    // the COMPLETED job's result relay (the REST lane, not the WS
    // payload) lands the full GnnTrainResult in the card
    const card = await screen.findByText("mdl-gnnaa000111");
    expect(card).toBeVisible();
    expect(screen.getByText("graph model")).toBeVisible();
    expect(screen.getByText("18 graphs")).toBeVisible();
    expect(screen.getByText(/train accuracy 1\.0000/)).toBeVisible();
    expect(
      screen.getByText(/loss history \(tail\) 0\.710 → 0\.520 → 0\.330 → 0\.140 → 0\.050/),
    ).toBeVisible();
    expect(screen.getByText("conv gcn · pool mean")).toBeVisible();
    expect(screen.getByText(/classes 0 \/ 1/)).toBeVisible();
    expect(jobsMocks.result).toHaveBeenCalledWith("job-gnn-1");
  });

  it("surfaces the backend's own 422 verbatim — never a silent failure", async () => {
    const user = userEvent.setup();
    dlMocks.gnnTrain.mockRejectedValueOnce(
      new ApiError(
        422,
        "the trainer needs inline graphs+labels or synthetic_n",
      ),
    );
    render(<DLPage />);
    await user.click(
      await screen.findByRole("button", { name: "Train the graph model" }),
    );
    expect(
      await screen.findByText(/the trainer needs inline graphs\+labels or synthetic_n/),
    ).toBeVisible();
  });
});

describe("the GNN predict lane — the graph models' own path", () => {
  it("predicts on inline graphs through /dl/gnn-predict with the class probabilities", async () => {
    const user = userEvent.setup();
    render(<DLPage />);
    await user.click(
      await screen.findByRole("button", { name: "Train the graph model" }),
    );
    await screen.findByText("mdl-gnnaa000111");
    // the predict lane's graphs block (same shape as the train
    // block) — fireEvent.change: the JSON braces are literal text,
    // never user-event key descriptors
    fireEvent.change(
      screen.getByLabelText(/graphs to predict \(JSON/),
      {
        target: {
          value:
            '{"graphs": [{"x": [[1, 0], [0, 1]], "edge_index": [[0, 1], [1, 0]]}, {"x": [[1, 0], [0, 1]], "edge_index": [[0, 1], [1, 0]]}]}',
        },
      },
    );
    dlMocks.gnnPredict.mockResolvedValueOnce({
      model_id: "mdl-gnnaa000111",
      n_graphs: 2,
      predictions: [0, 1],
      classes: [0, 1],
      probabilities: [
        [0.9, 0.1],
        [0.2, 0.8],
      ],
    });
    await user.click(
      screen.getByRole("button", { name: "Predict on the graphs" }),
    );
    expect(
      await screen.findByRole("columnheader", { name: "p(0)" }),
    ).toBeVisible();
    expect(screen.getByRole("columnheader", { name: "p(1)" })).toBeVisible();
    expect(screen.getByText("0.9000")).toBeVisible();
    expect(screen.getByText("0.8000")).toBeVisible();
    await waitFor(() => {
      expect(dlMocks.gnnPredict).toHaveBeenCalledWith({
        modelId: "mdl-gnnaa000111",
        graphs: [
          {
            x: [[1, 0], [0, 1]],
            edge_index: [[0, 1], [1, 0]],
          },
          {
            x: [[1, 0], [0, 1]],
            edge_index: [[0, 1], [1, 0]],
          },
        ],
        probabilities: true,
      });
    });
    // the tabular prediction path is NEVER wired into the graph lane
    expect(dlMocks.train).not.toHaveBeenCalled();
  });

  it("keeps a malformed inline train block client-side — the request never fires", async () => {
    const user = userEvent.setup();
    render(<DLPage />);
    await screen.findByRole("heading", { name: "GNN / MPNN — the graph lane" });
    await user.selectOptions(screen.getByLabelText("gnn source"), "inline");
    fireEvent.change(
      screen.getByLabelText(/inline graphs \+ labels \(JSON\)/),
      { target: { value: '{"graphs": "not an array"}' } },
    );
    await user.click(
      screen.getByRole("button", { name: "Train the graph model" }),
    );
    expect(
      await screen.findByText(/the inline graphs block needs a "graphs" array/),
    ).toBeVisible();
    expect(dlMocks.gnnTrain).not.toHaveBeenCalled();
  });
});
