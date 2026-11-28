import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DLPage } from "@/features/dl/DLPage";
import { ApiError } from "@/services/api/client";
import type { JobView } from "@/services/jobs/types";

/**
 * DL-3 (the WORKFLOW_ENHANCEMENT_ROADMAP's row 15) — the efficiency
 * lab's stage-11c card, as a complete component-level user flow:
 *
 *   the armed lane        — the card arms off the trained tabular
 *                          model (no run before the train); the
 *                          mode/level/scope controls ride the request;
 *   the report card       — the per-layer table renders the engine's
 *                          own fields (bits/weight, RMSE, relative
 *                          error on the quantize lanes; zeroed/kept/
 *                          sparsity on the prune lane) + the totals
 *                          line + the artifact download button;
 *   the pre-flight        — the prune lane's level must land in
 *                          [0, 1) client-side (the route's own
 *                          guard mirrored, no request fired);
 *   the honest 422        — the backend's own refusal (a non-tabular
 *                          model) surfaces verbatim, never swallowed.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, and the DOM are the real ones.
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
  efficiency: vi.fn(),
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
      efficiency: dlMocks.efficiency,
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
  networks: [
    {
      network: "deep_mlp",
      class: "DeepMLPClassifier",
      module: "pharmaml.ml.neural",
      task: "classification",
      engine: "torch",
      defaults: { hidden_layers: [64, 32], lr: 0.01, epochs: 50 },
      train_url: "/api/v1/dl/train",
    },
  ],
  predict_url: "/api/v1/ml/predict",
  note: "predictions for a DL-trained model run through the single session prediction path",
  gnn: {
    convs: [],
    pools: [],
    defaults: {},
    train_url: "/api/v1/dl/gnn-train",
    predict_url: "/api/v1/dl/gnn-predict",
    note: "",
  },
};

const TRAIN_RESULT = {
  model: {
    model_id: "mdl-dltorch000333",
    name: "deep_mlp",
    task: "classification",
    dataset_id: "ds-binary",
    target: "y",
    features: ["x1", "x2"],
    n_train_rows: 24,
    best_params: { network: "deep_mlp", epochs: 50 },
    cv_scores: null,
    cv_mean: null,
    cv_std: null,
    fit_seconds: 3.5,
    created_at: "2026-11-28T00:00:00Z",
    engine: "torch",
    predict_url: "/api/v1/ml/predict",
  },
};

const INT8_RESULT = {
  mode: "sync",
  modelId: "mdl-dltorch000333",
  network: "deep_mlp",
  efficiencyMode: "quantize_int8",
  nLayers: 2,
  nWeights: 64,
  layers: [
    {
      kind: "quantize",
      layer: "0.weight",
      shape: [8, 6],
      nWeights: 48,
      method: "int8",
      bitsPerWeight: 8,
      rmse: 0.0021,
      relativeError: 0.0052,
    },
    {
      kind: "quantize",
      layer: "3.weight",
      shape: [2, 8],
      nWeights: 16,
      method: "int8",
      bitsPerWeight: 8,
      rmse: 0.0017,
      relativeError: 0.0048,
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
    meta: {},
    download_url: "/api/v1/artifacts/art-eff000444555/download",
  },
  method: "int8",
  bitsPerWeight: 8,
  originalBits: 2048,
  quantizedBits: 512,
  memorySavings: 0.75,
  targetSparsity: null,
  scope: null,
  nZeroed: null,
  achievedSparsity: null,
  metric: null,
  note: "the per-layer rows ride the REAL quantization engine's own report",
};

const PRUNE_RESULT = {
  mode: "sync",
  modelId: "mdl-dltorch000333",
  network: "deep_mlp",
  efficiencyMode: "prune",
  nLayers: 2,
  nWeights: 64,
  layers: [
    {
      kind: "prune",
      layer: "0.weight",
      shape: [8, 6],
      nWeights: 48,
      nZeroed: 24,
      nKept: 24,
      sparsity: 0.5,
    },
    {
      kind: "prune",
      layer: "3.weight",
      shape: [2, 8],
      nWeights: 16,
      nZeroed: 8,
      nKept: 8,
      sparsity: 0.5,
    },
  ],
  artifact: {
    artifact_id: "art-eff000444666",
    kind: "table",
    filename: "dl-efficiency-prune.csv",
    content_type: "text/csv; charset=utf-8",
    size_bytes: 388,
    sha256: "b".repeat(64),
    created_at: "2026-11-28T00:00:00Z",
    meta: {},
    download_url: "/api/v1/artifacts/art-eff000444666/download",
  },
  method: null,
  bitsPerWeight: null,
  originalBits: null,
  quantizedBits: null,
  memorySavings: null,
  targetSparsity: 0.5,
  scope: "local",
  nZeroed: 32,
  achievedSparsity: 0.5,
  metric: {
    kind: "classification",
    before: 0.917,
    after: 0.875,
    delta: -0.042,
    note: "scored on the model's own training dataset through a pruned COPY",
  },
  note: "the masks ride the REAL magnitude-pruning engine",
};

function completedJobView(jobId: string): JobView {
  return {
    job_id: jobId,
    operation: "dl.train",
    category: "deep-learning",
    state: "COMPLETED",
    progress: 100,
    stage: "done",
    error: null,
    created_at: "2026-11-28T00:00:00Z",
    started_at: "2026-11-28T00:00:01Z",
    finished_at: "2026-11-28T00:00:04Z",
    configuration: {},
    result_ready: true,
  } as unknown as JobView;
}

beforeEach(() => {
  vi.clearAllMocks();
  jobState.snapshots = {};
  jobState.cancels = [];
  dlMocks.networks.mockResolvedValue(CATALOG);
  datasetsMocks.list.mockResolvedValue({
    datasets: [
      {
        dataset_id: "ds-binary",
        name: "binary.csv",
        shape: [24, 3],
        target: "y",
        task: "classification",
        created_at: "2026-11-28T00:00:00Z",
      },
    ],
  });
  datasetsMocks.view.mockResolvedValue({
    dataset_id: "ds-binary",
    name: "binary.csv",
    source_filename: "binary.csv",
    size_bytes: 256,
    shape: [24, 3],
    columns: ["x1", "x2", "y"],
    target: "y",
    task: "classification",
    n_features: 2,
    feature_names: ["x1", "x2"],
    classes: [0, 1],
    class_counts: { 0: 12, 1: 12 },
    dropped_columns: [],
    categorical_columns: [],
    summary: {},
    created_at: "2026-11-28T00:00:00Z",
  });
  dlMocks.train.mockResolvedValue({
    mode: "job",
    job: completedJobView("job-eff-1"),
  });
  jobState.snapshots["job-eff-1"] = {
    view: completedJobView("job-eff-1") as unknown as Record<string, unknown>,
  };
  jobsMocks.result.mockResolvedValue({ result: TRAIN_RESULT });
});

async function armTheTrainedModel(user: ReturnType<typeof userEvent.setup>) {
  render(<DLPage />);
  // stage 1: select the dataset (the registry row click)
  await waitFor(() => {
    expect(screen.getByText("binary.csv")).toBeInTheDocument();
  });
  await user.click(screen.getByText("binary.csv"));
  // stage 2: select the network (the catalog row click)
  await waitFor(() => {
    expect(screen.getByText("deep_mlp")).toBeInTheDocument();
  });
  await user.click(screen.getByText("deep_mlp"));
  // stage 6: train (the job completes and arms the model)
  await user.click(screen.getByText("Train the network"));
  await waitFor(() => {
    expect(jobsMocks.result).toHaveBeenCalled();
  });
}

describe("DLPage stage 11c — the DL-3 efficiency lab", () => {
  it("runs the int8 lane off the trained model and renders the engine's own report", async () => {
    const user = userEvent.setup();
    await armTheTrainedModel(user);

    dlMocks.efficiency.mockResolvedValueOnce(INT8_RESULT);
    await user.click(screen.getByText("Run the efficiency pass"));

    await waitFor(() => {
      expect(dlMocks.efficiency).toHaveBeenCalledTimes(1);
    });
    const call = dlMocks.efficiency.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(call["modelId"]).toEqual("mdl-dltorch000333");
    expect(call["mode"]).toEqual("quantize_int8");
    expect(call["level"]).toBeNull();
    expect(call["scope"]).toEqual("local");
    expect(call["artifact"]).toEqual(true);

    // the per-layer table renders the engine's own report fields
    expect(
      await screen.findByText(/bits\/weight · memory 75\.0% saved/),
    ).toBeInTheDocument();
    expect(screen.getByText("0.weight")).toBeInTheDocument();
    expect(screen.getByText("3.weight")).toBeInTheDocument();
    expect(screen.getByText("8×6")).toBeInTheDocument();
    expect(screen.getByText("0.00210")).toBeInTheDocument();
    // the artifact's own download button renders
    expect(
      screen.getByText(/Download dl-efficiency-quantize_int8\.csv/),
    ).toBeInTheDocument();
  });

  it("rides the prune lane with the parsed level and the scope, rendering the counts + the metric delta", async () => {
    const user = userEvent.setup();
    await armTheTrainedModel(user);

    // switch to the prune mode (the level + scope controls appear)
    fireEvent.change(screen.getByLabelText("the efficiency mode"), {
      target: { value: "prune" },
    });
    const levelInput = screen.getByLabelText(/level \(sparsity\)/);
    fireEvent.change(levelInput, { target: { value: "0.5" } });
    fireEvent.change(screen.getByLabelText("the pruning scope"), {
      target: { value: "global" },
    });

    dlMocks.efficiency.mockResolvedValueOnce(PRUNE_RESULT);
    await user.click(screen.getByText("Run the efficiency pass"));

    await waitFor(() => {
      expect(dlMocks.efficiency).toHaveBeenCalledTimes(1);
    });
    const call = dlMocks.efficiency.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(call["mode"]).toEqual("prune");
    expect(call["level"]).toEqual(0.5);
    expect(call["scope"]).toEqual("global");

    // the totals line + the metric delta + the per-layer counts
    expect(
      await screen.findByText(/achieved 0\.500 \(32 of 64 weights zeroed\)/),
    ).toBeInTheDocument();
    expect(screen.getByText(/score 0\.917 → 0\.875/)).toBeInTheDocument();
    expect(screen.getAllByText("24").length).toBeGreaterThan(0);
    expect(screen.getByText(/never mutated/)).toBeInTheDocument();
    expect(
      screen.getByText(/Download dl-efficiency-prune\.csv/),
    ).toBeInTheDocument();
  });

  it("refuses a level outside [0, 1) pre-flight, no request fired", async () => {
    const user = userEvent.setup();
    await armTheTrainedModel(user);

    fireEvent.change(screen.getByLabelText("the efficiency mode"), {
      target: { value: "prune" },
    });
    fireEvent.change(screen.getByLabelText(/level \(sparsity\)/), {
      target: { value: "1.4" },
    });
    await user.click(screen.getByText("Run the efficiency pass"));

    expect(dlMocks.efficiency).not.toHaveBeenCalled();
    expect(
      await screen.findByText(
        /the prune lane's level must land in \[0, 1\) — "1\.4" does not parse there/,
      ),
    ).toBeInTheDocument();
  });

  it("surfaces the backend's own 422 verbatim, never swallowed", async () => {
    const user = userEvent.setup();
    await armTheTrainedModel(user);

    dlMocks.efficiency.mockRejectedValueOnce(
      new ApiError(
        422,
        "model mdl-dltorch000333 is a 'gnn_graph_classifier' model; the efficiency lab serves the tabular DL families trained through POST /api/v1/dl/train (the six torch networks whose weight matrices the lab reads)",
      ),
    );
    await user.click(screen.getByText("Run the efficiency pass"));

    expect(
      await screen.findByText(/the efficiency lab serves the tabular DL families/),
    ).toBeInTheDocument();
  });

  it("stays unarmed before a model exists (the honest gate)", async () => {
    render(<DLPage />);
    const button = await screen.findByText("Run the efficiency pass");
    expect(button).toBeDisabled();
    expect(
      screen.getByText(/Train a network in stage 6 first/),
    ).toBeInTheDocument();
  });
});
