/**
 * part21 §3.2 — the DL workspace's Evaluate stage adopts the SHARED
 * evaluation-visualization layer, as a complete component-level user
 * flow over the full DLPage:
 *
 *   the tabular train   — job mode, the result relayed through the
 *                         job tracker (the model card renders the
 *                         torch model registered in the shared store);
 *   the predict lane    — the SINGLE session prediction path
 *                         (`POST /api/v1/ml/predict`, dataset mode,
 *                         sync) — the SAME contract the ML workspace
 *                         rides, answered with a real evaluation block
 *                         + probabilities;
 *   the shared charts   — the shared EvaluationCharts renders the
 *                         classification family (the confusion matrix,
 *                         the per-class metrics, the ROC/PR curves, the
 *                         reliability diagram) inside the DL Evaluate
 *                         stage — one component, two workspaces, zero
 *                         duplicate implementation;
 *   the explanatory state — an answer without an evaluation block
 *                         states it, never invents charts.
 *
 * Services are mocked at the network boundary only (the GnnLane
 * convention); the component, its state machine, and the REAL runtime
 * parsers (parseDLTrainResult / parsePredictResponse /
 * parseEvaluationBlock — spread from the actual modules) run the
 * real code.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DLPage } from "@/features/dl/DLPage";
import type { JobView } from "@/services/jobs/types";

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
}));
vi.mock("@/services/dl/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/dl/client")>();
  return {
    ...actual,
    dlApi: {
      networks: dlMocks.networks,
      train: dlMocks.train,
      gnnTrain: vi.fn(),
      gnnPredict: vi.fn(),
      cnnTrain: vi.fn(),
      cnnPredict: vi.fn(),
      autoencoderTrain: vi.fn(),
      autoencoderEncode: vi.fn(),
      lossCurveFigure: vi.fn(),
      saveCheckpoint: vi.fn(),
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
    model_id: "mdl-dltorch000111",
    name: "deep_mlp",
    task: "classification",
    dataset_id: "ds-binary",
    target: "y",
    features: ["x1", "x2"],
    n_train_rows: 24,
    best_params: { network: "deep_mlp" },
    cv_scores: null,
    cv_mean: null,
    cv_std: null,
    fit_seconds: 1.25,
    created_at: "2026-11-28T00:00:00Z",
    engine: "torch",
    predict_url: "/api/v1/ml/predict",
  },
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

const PREDICT_WITH_EVALUATION = {
  mode: "sync",
  model_id: "mdl-dltorch000111",
  n_rows: 6,
  predictions: [1, 1, 0, 0, 1, 0],
  probabilities: {
    classes: [0, 1],
    rows: [
      [0.1, 0.9],
      [0.2, 0.8],
      [0.85, 0.15],
      [0.75, 0.25],
      [0.3, 0.7],
      [0.9, 0.1],
    ],
  },
  evaluation: {
    target: "y",
    task: "classification",
    smiles_col: null,
    n_rows_available: 6,
    n_rows_evaluated: 6,
    rows: [
      { name: 0, actual: 1, predicted: 1, residual: null, smiles: null },
      { name: 1, actual: 1, predicted: 1, residual: null, smiles: null },
      { name: 2, actual: 0, predicted: 0, residual: null, smiles: null },
      { name: 3, actual: 0, predicted: 0, residual: null, smiles: null },
      { name: 4, actual: 1, predicted: 1, residual: null, smiles: null },
      { name: 5, actual: 0, predicted: 0, residual: null, smiles: null },
    ],
    residual_diagnostics: null,
    class_metrics: {
      rows: [
        { label: "0", precision: 1, recall: 1, f1: 1, support: 3 },
        { label: "1", precision: 1, recall: 1, f1: 1, support: 3 },
      ],
      macro: { precision: 1, recall: 1, f1: 1 },
      accuracy: 1,
      n: 6,
    },
    roc_curves: [
      {
        label: "0",
        fpr: [0, 0.1, 1],
        tpr: [0, 0.9, 1],
        thresholds: [0.9, 0.5, 0.1],
        auc: 0.95,
        n_pos: 3,
        n_neg: 3,
      },
      {
        label: "1",
        fpr: [0, 0.1, 1],
        tpr: [0, 0.9, 1],
        thresholds: [0.9, 0.5, 0.1],
        auc: 0.95,
        n_pos: 3,
        n_neg: 3,
      },
    ],
    pr_curves: [
      {
        label: "1",
        recall: [0, 0.5, 1],
        precision: [1, 1, 1],
        thresholds: [0.9, 0.7, 0.5],
        average_precision: 1,
        prevalence: 0.5,
        n_pos: 3,
      },
    ],
    calibration: {
      bins: [{ mean_predicted: 0.9, observed: 1, count: 3 }],
      n: 6,
      positive_class: "1",
    },
  },
};

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
    job: completedJobView("job-dl-1"),
  });
  jobState.snapshots["job-dl-1"] = {
    view: completedJobView("job-dl-1") as unknown as Record<string, unknown>,
  };
  jobsMocks.result.mockResolvedValue({ result: TRAIN_RESULT });
  mlMocks.predict.mockResolvedValue(PREDICT_WITH_EVALUATION);
});

async function driveTabularPredict() {
  const user = userEvent.setup();
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
  // stage: train (job mode — the relayed result sets the model card)
  const trainButton = await screen.findByText("Train the network");
  await user.click(trainButton);
  // stage 9: predict on the dataset
  const predictButton = await screen.findByText("Predict on the dataset");
  await waitFor(() => {
    expect(predictButton).toBeEnabled();
  });
  await user.click(predictButton);
}

describe("part21 §3.2 — the DL Evaluate stage rides the shared charts", () => {
  it("renders the shared classification family inside the Evaluate stage", async () => {
    await driveTabularPredict();
    // the shared evaluation charts rendered (the one testid the shared
    // component carries) with the classification family's panels
    expect(
      await screen.findByTestId("evaluation-charts"),
    ).toBeInTheDocument();
    expect(screen.getByText(/classification diagnostics/)).toBeInTheDocument();
    expect(screen.getByText(/Per-class metrics over the sample/)).toBeInTheDocument();
    expect(screen.getByText(/ROC curves \(one-vs-rest, per class\)/)).toBeInTheDocument();
    expect(screen.getByText(/Precision–Recall curves \(one-vs-rest, per class\)/)).toBeInTheDocument();
    expect(screen.getByText(/Reliability diagram \(positive class\)/)).toBeInTheDocument();
    // the metrics table rides inside its panel (its real class rows
    // follow in the DOM under the shared panel's caption)
    const panel = screen.getByText(/Per-class metrics over the sample/).closest("div");
    expect(panel).not.toBeNull();
    expect(panel?.textContent).toContain("precision");
    expect(panel?.textContent).toContain("macro avg");
  });

  it("states the honest explanatory state when no evaluation block rides the answer", async () => {
    mlMocks.predict.mockResolvedValue({
      ...PREDICT_WITH_EVALUATION,
      evaluation: null,
    });
    await driveTabularPredict();
    expect(
      await screen.findByText(/No evaluation block in this answer/),
    ).toBeInTheDocument();
    // and no chart panels fabricate themselves
    expect(screen.queryByTestId("evaluation-charts")).toBeNull();
    expect(screen.queryByText(/Per-class metrics over the sample/)).toBeNull();
  });
});
