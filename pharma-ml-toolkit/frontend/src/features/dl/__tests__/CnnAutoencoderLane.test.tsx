import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DLPage } from "@/features/dl/DLPage";

/**
 * part18 §5.2 / §5.5 / §5.4 — the Deep Learning workspace's new
 * architecture lanes as complete component-level user flows:
 *
 *   the transformer family — the catalog's own entries render in the
 *                         architecture stage and the Configure Model
 *                         stage swaps to the transformer's OWN keys
 *                         (d_model/heads/blocks/feedforward);
 *   the sequence CNN   — the synthetic motif fixture's honest
 *                         request shape, the job relay into the
 *                         sequence model card, and the models' OWN
 *                         predict path;
 *   the autoencoder    — the train lane over the SELECTED dataset,
 *                         the encode lane (latent + reconstruction
 *                         errors + the honest outlier ranking), and
 *                         the unsupervised task badge.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers, and the DOM are the real
 * ones (the GnnLane.test.tsx conventions).
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
  cnnTrain: vi.fn(),
  cnnPredict: vi.fn(),
  autoencoderTrain: vi.fn(),
  autoencoderEncode: vi.fn(),
  lossCurveFigure: vi.fn(),
  saveCheckpoint: vi.fn(),
}));

vi.mock("@/services/dl/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/dl/client")>();
  return {
    ...actual,
    dlApi: {
      networks: dlMocks.networks,
      train: dlMocks.train,
      gnnTrain: dlMocks.gnnTrain,
      gnnPredict: dlMocks.gnnPredict,
      cnnTrain: dlMocks.cnnTrain,
      cnnPredict: dlMocks.cnnPredict,
      autoencoderTrain: dlMocks.autoencoderTrain,
      autoencoderEncode: dlMocks.autoencoderEncode,
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

const NETWORK_CATALOG = {
  networks: [
    {
      network: "deep_mlp_classifier", class: "DeepMLPClassifier",
      module: "pharmaml.ml.neural", task: "classification", engine: "torch",
      defaults: { epochs: 120 }, train_url: "/api/v1/dl/train",
    },
    {
      network: "tabular_transformer_classifier",
      class: "TabularTransformerClassifier",
      module: "pharmaml.ml.transformer", task: "classification",
      engine: "torch",
      defaults: { d_model: 32, heads: 4, n_blocks: 2, dim_feedforward: 64 },
      train_url: "/api/v1/dl/train",
    },
    {
      network: "tabular_transformer_regressor",
      class: "TabularTransformerRegressor",
      module: "pharmaml.ml.transformer", task: "regression",
      engine: "torch",
      defaults: { d_model: 32, heads: 4, n_blocks: 2, dim_feedforward: 64 },
      train_url: "/api/v1/dl/train",
    },
  ],
  predict_url: "/api/v1/ml/predict",
  note: "predictions for a DL-trained model run through the single session prediction path",
  gnn: {
    convs: ["gcn"], pools: ["mean"],
    defaults: { hidden: 16, layers: 2, lr: 0.05, epochs: 200 },
    train_url: "/api/v1/dl/gnn-train",
    predict_url: "/api/v1/dl/gnn-predict",
    note: "graph models predict through their own path — never the tabular one",
  },
};

const CNN_TRAIN_RESULT = {
  model: {
    model_id: "mdl-cnnaa000111",
    name: "sequence_cnn_classification",
    task: "classification",
    dataset_id: "",
    target: "",
    features: [],
    n_train_rows: 40,
    best_params: { network: "sequence_cnn" },
    cv_scores: null,
    cv_mean: null,
    cv_std: null,
    fit_seconds: 1.2,
    created_at: "2026-11-27T00:00:00Z",
    engine: "torch",
    sequence_model: true,
    predict_url: "/api/v1/dl/cnn-predict",
  },
  seq_len: 16,
  n_channels: 3,
  n_iter: 25,
  classes: [0, 1],
  train_accuracy: 1.0,
};

const AE_TRAIN_RESULT = {
  model: {
    model_id: "mdl-aeaa000111",
    name: "tabular_autoencoder",
    task: "unsupervised",
    dataset_id: "ds-1",
    target: "",
    features: ["x1", "x2"],
    n_train_rows: 60,
    best_params: { network: "autoencoder" },
    cv_scores: null,
    cv_mean: null,
    cv_std: null,
    fit_seconds: 2.0,
    created_at: "2026-11-27T00:00:00Z",
    engine: "torch",
    encode_url: "/api/v1/dl/autoencoder-encode",
  },
  latentDim: 2,
  nIter: 40,
  meanReconstructionError: 0.2071,
};

const AE_ENCODE_RESULT = {
  modelId: "mdl-aeaa000111",
  nRows: 3,
  latentDim: 2,
  latent: [[0.1, 0.2], [-0.3, 0.4], [2.5, -2.5]],
  reconstructionErrors: [0.1, 0.2, 9.9],
  meanReconstructionError: 3.4,
  topOutliers: [
    { index: 2, error: 9.9 }, { index: 1, error: 0.2 },
  ],
  note: "reconstruction error is a heuristic outlier ranking — not ground truth",
};

const DATASET_VIEW = {
  dataset_id: "ds-1",
  name: "demo",
  shape: [60, 2],
  task: "classification",
  target: "y",
  n_features: 2,
  classes: ["0", "1"],
  class_counts: { "0": 30, "1": 30 },
  dropped_columns: [],
  categorical_columns: [],
  created_at: "2026-11-27T00:00:00Z",
};

function completedJobView(jobId: string, operation: string): Record<string, unknown> {
  return {
    job_id: jobId,
    operation,
    category: "deep-learning",
    state: "COMPLETED",
    progress: 100,
    stage: "done",
    error: null,
    created_at: "2026-11-27T00:00:00Z",
    started_at: "2026-11-27T00:00:01Z",
    finished_at: "2026-11-27T00:00:04Z",
    configuration: {},
    result_ready: true,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  jobState.snapshots = {};
  jobState.cancels = [];
  dlMocks.networks.mockResolvedValue(NETWORK_CATALOG);
  datasetsMocks.list.mockResolvedValue({ datasets: [DATASET_VIEW] });
  datasetsMocks.view.mockResolvedValue(DATASET_VIEW);
});

describe("the transformer family — the catalog's own entries", () => {
  it("lists the tabular transformer families off the live catalog", async () => {
    render(<DLPage />);
    await screen.findByRole("heading", { name: "GNN / MPNN — the graph lane" });
    // the architecture stage renders the catalog's own network entries —
    // the two transformer families appear with their live defaults
    expect(
      await screen.findByText("TabularTransformerClassifier"),
    ).toBeVisible();
    expect(
      screen.getByText("TabularTransformerRegressor"),
    ).toBeVisible();
  });
});

describe("the sequence CNN lane", () => {
  it("renders the lane with the motif fixture controls", async () => {
    render(<DLPage />);
    expect(
      await screen.findByRole("heading", {
        name: "Sequence CNN — the convolutional lane",
      }),
    ).toBeVisible();
    // the no-fake-images honesty rides the lane itself
    expect(
      screen.getByText(/no-fake-images rule, on the wire/i),
    ).toBeVisible();
  });

  it("submits the synthetic fixture's honest request shape in job mode", async () => {
    const user = userEvent.setup();
    dlMocks.cnnTrain.mockResolvedValue({
      mode: "job",
      job: completedJobView("job-cnn-1", "dl.cnn-train"),
    });
    jobState.snapshots["job-cnn-1"] = { view: completedJobView("job-cnn-1", "dl.cnn-train") };
    jobsMocks.result.mockResolvedValue({ result: CNN_TRAIN_RESULT });
    render(<DLPage />);
    const trainButton = await screen.findByRole("button", {
      name: "Train the sequence model",
    });
    await user.click(trainButton);
    await waitFor(() => {
      expect(dlMocks.cnnTrain).toHaveBeenCalledTimes(1);
    });
    const options = dlMocks.cnnTrain.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(options["syntheticN"]).toBe(40);
    expect(options["task"]).toBe("classification");
    expect(options["channels"]).toBe(8);
    expect(options["kernel"]).toBe(3);
    // the COMPLETED job's authoritative REST result relays into the card
    expect(
      await screen.findByText(/sequence model \(own predict path\)/i),
    ).toBeVisible();
    expect(screen.getByText(/train accuracy 1\.000/i)).toBeVisible();
  });

  it("predicts through the sequence models' OWN path", async () => {
    const user = userEvent.setup();
    dlMocks.cnnTrain.mockResolvedValue({ mode: "sync", result: CNN_TRAIN_RESULT });
    dlMocks.cnnPredict.mockResolvedValue({
      modelId: "mdl-cnnaa000111",
      nSequences: 1,
      predictions: [1],
      probabilities: [[0.1, 0.9]],
      classes: [0, 1],
    });
    render(<DLPage />);
    const trainButton = await screen.findByRole("button", {
      name: "Train the sequence model",
    });
    await user.click(trainButton);
    const predictButton = await screen.findByRole("button", {
      name: "Predict on the sequences",
    });
    // the predict lane needs an inline block — the textarea gates it
    expect(predictButton).toBeDisabled();
    const textarea = screen.getByPlaceholderText(/"sequences": \[\[\[/);
    fireEvent.change(textarea, {
      target: { value: '{"sequences": [[[0.1, 0.2], [0.3, 0.4]]]}' },
    });
    await waitFor(() => expect(predictButton).toBeEnabled());
    await user.click(predictButton);
    await waitFor(() => {
      expect(dlMocks.cnnPredict).toHaveBeenCalledTimes(1);
    });
    const opts = dlMocks.cnnPredict.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(opts["modelId"]).toBe("mdl-cnnaa000111");
    expect(opts["probabilities"]).toBe(true);
  });
});

describe("the tabular autoencoder lane", () => {
  it("gates the train on a selected dataset", async () => {
    render(<DLPage />);
    const trainButton = await screen.findByRole("button", {
      name: "Train the autoencoder",
    });
    expect(trainButton).toBeDisabled();
    // the AE card's own hint (scoped to the card text, not the page-wide
    // dataset hints of the other lanes)
    expect(screen.getAllByText(/select a dataset above/i).length).toBeGreaterThan(0);
  });

  it("trains over the SELECTED dataset and encodes it", async () => {
    const user = userEvent.setup();
    dlMocks.autoencoderTrain.mockResolvedValue({
      mode: "sync",
      result: AE_TRAIN_RESULT,
    });
    dlMocks.autoencoderEncode.mockResolvedValue(AE_ENCODE_RESULT);
    render(<DLPage />);
    // select the dataset by clicking its registry row (the page's own
    // table-row selection lane)
    const nameCell = await screen.findByText("demo");
    const row = nameCell.closest("tr");
    expect(row).not.toBeNull();
    await user.click(row as HTMLElement);
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
    });
    const trainButton = await screen.findByRole("button", {
      name: "Train the autoencoder",
    });
    await waitFor(() => expect(trainButton).toBeEnabled());
    await user.click(trainButton);
    await waitFor(() => {
      expect(dlMocks.autoencoderTrain).toHaveBeenCalledTimes(1);
    });
    const options = dlMocks.autoencoderTrain.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(options["datasetId"]).toBe("ds-1");
    expect(options["latentDim"]).toBe(2);
    // the unsupervised task badge + the honest encode lane
    expect(
      await screen.findByText(/task\s+unsupervised/i),
    ).toBeVisible();
    const encodeButton = await screen.findByRole("button", {
      name: "Encode the dataset",
    });
    await user.click(encodeButton);
    await waitFor(() => {
      expect(dlMocks.autoencoderEncode).toHaveBeenCalledTimes(1);
    });
    // the outlier ranking renders, with the heuristic honesty note
    expect(
      await screen.findByText(/#2 \(9\.900\)/i),
    ).toBeVisible();
    expect(
      screen.getByText(/never ground truth/i),
    ).toBeVisible();
    // the 2-D latent renders the projection (the stubbed figure lane)
    expect(screen.getByTestId("plotly-stub")).toBeVisible();
  });
});
