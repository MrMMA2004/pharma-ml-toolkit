import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DLPage } from "@/features/dl/DLPage";
import { ApiError } from "@/services/api/client";
import type { JobView } from "@/services/jobs/types";

/**
 * DL-1 (the WORKFLOW_ENHANCEMENT_ROADMAP's row 13) — the
 * hyperparameter search lane of the Deep Learning workspace, as a
 * complete component-level user flow:
 *
 *   the armed search     — the checkbox arms the SAME train route with
 *                          tune: true and the parsed search block (the
 *                          n_trials/sampler/seed controls ride the
 *                          request);
 *   the trial table      — the relayed job result renders the trial
 *                          rows with the best (argmin objective)
 *                          highlighted + its own data-best attribute,
 *                          the best configuration + the refit score
 *                          named in the summary line;
 *   the pre-flight       — a malformed search block (an unknown key)
 *                          is refused here, the route's own message
 *                          mirrored, no request fired;
 *   the honest 422       — the backend's own 422 (a doomed search)
 *                          surfaces verbatim in the lane's own banner,
 *                          never swallowed.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers (parseDLTuneResult — the
 * REAL one, spread from the actual module), and the DOM are the real
 * ones.
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

const TUNE_RESULT = {
  model: {
    model_id: "mdl-dltorch000222",
    name: "deep_mlp",
    task: "classification",
    dataset_id: "ds-binary",
    target: "y",
    features: ["x1", "x2"],
    n_train_rows: 24,
    best_params: {
      network: "deep_mlp",
      learning_rate: 0.2,
      epochs: 50,
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
      {
        number: 0,
        params: { learning_rate: 0.031 },
        value: -0.75,
      },
      {
        number: 1,
        params: { learning_rate: 0.107 },
        value: -0.8333,
      },
      {
        number: 2,
        params: { learning_rate: 0.2 },
        value: -0.9167,
      },
      {
        number: 3,
        params: { learning_rate: 0.455 },
        value: -0.6667,
      },
    ],
    refit_score: 0.9167,
    sampler: "random",
    seed: 0,
    searched: {
      learning_rate: { low: 0.001, high: 0.5, log: true },
    },
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
    job: completedJobView("job-tune-1"),
  });
  jobState.snapshots["job-tune-1"] = {
    view: completedJobView("job-tune-1") as unknown as Record<string, unknown>,
  };
  jobsMocks.result.mockResolvedValue({ result: TUNE_RESULT });
});

async function armTheSearchLane(user: ReturnType<typeof userEvent.setup>) {
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
  // DL-1: arm the search lane
  await user.click(
    screen.getByLabelText("arm the hyperparameter search"),
  );
  return screen.getByLabelText("the tune search block");
}

describe("DLPage stage 7b — the DL-1 hyperparameter search lane", () => {
  it("rides the SAME train route with tune: true and the parsed search block", async () => {
    const user = userEvent.setup();
    await armTheSearchLane(user);

    // the default sample block rides the request as parsed JSON
    await user.click(screen.getByText("Train the network"));

    await waitFor(() => {
      expect(dlMocks.train).toHaveBeenCalledTimes(1);
    });
    const call = dlMocks.train.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(call["tune"]).toEqual({
      search: {
        learning_rate: { low: 0.001, high: 0.3, log: true },
        dropout: { low: 0.05, high: 0.5 },
        hidden_layers: { choices: [[32], [64, 32], [128, 64]] },
      },
      nTrials: 12,
      sampler: "tpe",
      seed: 0,
    });

    // the relayed job result renders the trial table with the best
    // (argmin objective) row highlighted
    expect(
      await screen.findByText(/best of 4 trial\(s\)/),
    ).toBeInTheDocument();
    expect(screen.getByText(/refit 0\.9167/)).toBeInTheDocument();
    const bestRow = screen
      .getByText('{"learning_rate":0.2}')
      .closest("tr");
    expect(bestRow).not.toBeNull();
    expect(bestRow).toHaveAttribute("data-best", "true");
    const otherRow = screen
      .getByText('{"learning_rate":0.031}')
      .closest("tr");
    expect(otherRow).not.toHaveAttribute("data-best");
    // the honest bigger-is-better score column + the study's own
    // smaller-is-better objective column both ride the table
    expect(screen.getByText("0.9167")).toBeInTheDocument();
    expect(screen.getByText("-0.9167")).toBeInTheDocument();
  });

  it("refuses a doomed search block pre-flight, no request fired", async () => {
    const user = userEvent.setup();
    const area = await armTheSearchLane(user);

    // an unknown key — the route's own vocabulary message mirrored
    // (fireEvent.change: the JSON braces are not a keyboard path)
    fireEvent.change(area, {
      target: { value: '{"nonsense_key": {"low": 1, "high": 5}}' },
    });
    await user.click(screen.getByText("Train the network"));

    expect(dlMocks.train).not.toHaveBeenCalled();
    expect(
      await screen.findByText(/unknown search key "nonsense_key"/),
    ).toBeInTheDocument();
  });

  it("refuses an inverted bounds block pre-flight, no request fired", async () => {
    const user = userEvent.setup();
    const area = await armTheSearchLane(user);

    fireEvent.change(area, {
      target: { value: '{"dropout": {"low": 0.9, "high": 0.1}}' },
    });
    await user.click(screen.getByText("Train the network"));

    expect(dlMocks.train).not.toHaveBeenCalled();
    expect(
      await screen.findByText(/must satisfy low < high/),
    ).toBeInTheDocument();
  });

  it("surfaces the backend's own 422 verbatim, never swallowed", async () => {
    const user = userEvent.setup();
    await armTheSearchLane(user);

    dlMocks.train.mockRejectedValueOnce(
      new ApiError(
        422,
        "the tune lane needs a search block: {key: {choices: [...]} or {key: {low: ..., high: ..., log: bool}}} over the network 'deep_mlp' constructor vocabulary",
      ),
    );
    await user.click(screen.getByText("Train the network"));

    expect(
      await screen.findByText(/the tune lane needs a search block/),
    ).toBeInTheDocument();
  });
});
