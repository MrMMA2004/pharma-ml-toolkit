import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DLPage } from "@/features/dl/DLPage";
import { ApiError } from "@/services/api/client";
import type { JobView } from "@/services/jobs/types";

/**
 * DL-4 (the WORKFLOW_ENHANCEMENT_ROADMAP's row 16) — the D-MPNN's
 * molecular lane, as a complete component-level user flow:
 *
 *   the armed lane        — the molecular lane's own dataset pick
 *                          (fetching its own view) + the SMILES
 *                          column select riding the request; the
 *                          task gate refuses a classification
 *                          dataset before any request fires;
 *   the result card       — the held-out carve's own verdicts (the
 *                          roadmap's own validation step) + the
 *                          honest rejected-rows accounting;
 *   the predict lane      — raw SMILES (one per line) through the
 *                          molecular models' OWN prediction path,
 *                          the rejected rows rendering inline;
 *   the honest 422        — the backend's own refusal (an unknown
 *                          SMILES column) surfaces verbatim, never
 *                          swallowed.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parser (parseMpnnTrainResult — the
 * REAL one, spread from the actual module), and the DOM are the
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
  mpnnTrain: vi.fn(),
  mpnnPredict: vi.fn(),
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
      mpnnTrain: dlMocks.mpnnTrain,
      mpnnPredict: dlMocks.mpnnPredict,
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

const MPNN_VIEW = {
  dataset_id: "ds-mpnn0001",
  name: "molecules.csv",
  source_filename: "molecules.csv",
  size_bytes: 512,
  shape: [40, 2],
  columns: ["smiles", "y"],
  target: "y",
  task: "regression",
  n_features: 1,
  feature_names: ["smiles"],
  classes: null,
  class_counts: null,
  dropped_columns: [],
  categorical_columns: ["smiles"],
  summary: {},
  created_at: "2026-11-28T00:00:00Z",
};

const MPNN_TRAIN_RESULT = {
  model: {
    model_id: "mdl-mpnn000222",
    name: "dmpnn_molecular_regressor",
    task: "regression",
    dataset_id: "ds-mpnn0001",
    target: "y",
    features: ["smiles"],
    n_train_rows: 40,
    best_params: { network: "dmpnn", hidden_size: 32, depth: 3 },
    cv_scores: null,
    cv_mean: null,
    cv_std: null,
    fit_seconds: 6.2,
    created_at: "2026-11-28T00:00:00Z",
    engine: "numpy",
    molecular_model: true,
    predict_url: "/api/v1/dl/mpnn-predict",
  },
  val_rmse: 0.109,
  val_n: 8,
  val_correlation: 0.991,
  train_history_tail: [0.61, 0.42, 0.3, 0.18, 0.11],
  refit_history_tail: [0.55, 0.36, 0.25, 0.14, 0.09],
  n_rows: 40,
  n_rejected: 0,
  rejected: [],
  n_parameters: 3193,
  held_out_note: "the carve is seeded and held out",
};

const MPNN_PREDICT_RESULT = {
  model_id: "mdl-mpnn000222",
  n_valid: 2,
  predictions: [
    { index: 0, smiles: "CCO", prediction: 3.04 },
    { index: 2, smiles: "c1ccccc1", prediction: 6.02 },
  ],
  n_rejected: 1,
  rejected: [
    { index: 1, smiles: "not-a-smiles", error: "SMILES syntax problem" },
  ],
  note: "the predictions are in the original target units",
};

function completedJobView(jobId: string): JobView {
  return {
    job_id: jobId,
    operation: "dl.mpnn-train",
    category: "deep-learning",
    state: "COMPLETED",
    progress: 100,
    stage: "done",
    error: null,
    created_at: "2026-11-28T00:00:00Z",
    started_at: "2026-11-28T00:00:01Z",
    finished_at: "2026-11-28T00:00:06Z",
    configuration: {},
    result_ready: true,
  } as unknown as JobView;
}

beforeEach(() => {
  vi.clearAllMocks();
  jobState.snapshots = {};
  jobState.cancels = [];
  dlMocks.networks.mockResolvedValue({
    networks: [],
    predict_url: "/api/v1/ml/predict",
    note: "",
    gnn: {
      convs: [], pools: [], defaults: {},
      train_url: "/api/v1/dl/gnn-train",
      predict_url: "/api/v1/dl/gnn-predict",
      note: "",
    },
  });
  datasetsMocks.list.mockResolvedValue({
    datasets: [
      {
        dataset_id: "ds-mpnn0001",
        name: "molecules.csv",
        shape: [40, 2],
        target: "y",
        task: "regression",
        created_at: "2026-11-28T00:00:00Z",
      },
    ],
  });
  datasetsMocks.view.mockResolvedValue(MPNN_VIEW);
  dlMocks.mpnnTrain.mockResolvedValue({
    mode: "job",
    job: completedJobView("job-mpnn-1"),
  });
  jobState.snapshots["job-mpnn-1"] = {
    view: completedJobView("job-mpnn-1") as unknown as Record<string, unknown>,
  };
  jobsMocks.result.mockResolvedValue({ result: MPNN_TRAIN_RESULT });
});

async function armTheMolecularLane(_user: ReturnType<typeof userEvent.setup>) {
  render(<DLPage />);
  // the molecular lane's own dataset pick (its own select — the
  // tabular registry stands untouched)
  const datasetSelect = await screen.findByLabelText(
    "the molecular lane's dataset",
  );
  fireEvent.change(datasetSelect, { target: { value: "ds-mpnn0001" } });
  // the SMILES column select populates from the lane's own view
  await waitFor(() => {
    const columnSelect = screen.getByLabelText("the SMILES column");
    expect(columnSelect).toBeInTheDocument();
  });
  fireEvent.change(screen.getByLabelText("the SMILES column"), {
    target: { value: "smiles" },
  });
}

describe("DLPage — the DL-4 molecular (D-MPNN) lane", () => {
  it("trains over the picked SMILES column and renders the held-out verdicts", async () => {
    const user = userEvent.setup();
    await armTheMolecularLane(user);

    await user.click(screen.getByText("Train the molecular model"));

    await waitFor(() => {
      expect(dlMocks.mpnnTrain).toHaveBeenCalledTimes(1);
    });
    const call = dlMocks.mpnnTrain.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(call["datasetId"]).toEqual("ds-mpnn0001");
    expect(call["smilesColumn"]).toEqual("smiles");
    expect(call["hiddenSize"]).toEqual(32);
    expect(call["depth"]).toEqual(3);
    expect(call["mode"]).toEqual("job");

    // the job's relayed result renders the held-out carve's verdicts
    expect(
      await screen.findByText(/held-out RMSE 0\.109 over 8 rows/),
    ).toBeInTheDocument();
    expect(screen.getByText(/correlation 0\.991/)).toBeInTheDocument();
    expect(screen.getByText(/40 molecules/)).toBeInTheDocument();
    expect(screen.getByText(/3193 parameters/)).toBeInTheDocument();
    expect(screen.getByText(/the carve is seeded and held out/)).toBeInTheDocument();
  });

  it("predicts over raw SMILES with the rejected rows rendered inline", async () => {
    const user = userEvent.setup();
    await armTheMolecularLane(user);
    await user.click(screen.getByText("Train the molecular model"));
    await waitFor(() => {
      expect(jobsMocks.result).toHaveBeenCalled();
    });

    dlMocks.mpnnPredict.mockResolvedValueOnce(MPNN_PREDICT_RESULT);
    fireEvent.change(screen.getByLabelText("the SMILES to predict"), {
      target: { value: "CCO\nnot-a-smiles\nc1ccccc1" },
    });
    await user.click(screen.getByText("Predict the properties"));

    await waitFor(() => {
      expect(dlMocks.mpnnPredict).toHaveBeenCalledTimes(1);
    });
    const call = dlMocks.mpnnPredict.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(call["modelId"]).toEqual("mdl-mpnn000222");
    expect(call["smiles"]).toEqual(["CCO", "not-a-smiles", "c1ccccc1"]);

    // the valid rows render with their served indexes; the rejected
    // row renders inline with its own error
    expect(await screen.findByText(/2 prediction/)).toBeInTheDocument();
    expect(screen.getByText(/1 rejected/)).toBeInTheDocument();
    expect(screen.getAllByText("CCO").length).toBeGreaterThan(0);
    expect(screen.getByText("3.040")).toBeInTheDocument();
    expect(screen.getByText(/rejected: SMILES syntax problem/)).toBeInTheDocument();
  });

  it("refuses a classification dataset before any request fires", async () => {
    render(<DLPage />);
    // the view answers a classification task — the honest gate
    datasetsMocks.view.mockResolvedValue({
      ...MPNN_VIEW,
      task: "classification",
    });
    const datasetSelect = await screen.findByLabelText(
      "the molecular lane's dataset",
    );
    fireEvent.change(datasetSelect, { target: { value: "ds-mpnn0001" } });
    await waitFor(() => {
      expect(
        screen.getByText(/the D-MPNN is a regressor/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText("Train the molecular model"),
    ).toBeDisabled();
    expect(dlMocks.mpnnTrain).not.toHaveBeenCalled();
  });

  it("surfaces the backend's own 422 verbatim, never swallowed", async () => {
    const user = userEvent.setup();
    await armTheMolecularLane(user);

    dlMocks.mpnnTrain.mockRejectedValueOnce(
      new ApiError(
        422,
        "smiles_column 'smiles' is not one of the dataset's columns: ['structure', 'y']",
      ),
    );
    await user.click(screen.getByText("Train the molecular model"));

    expect(
      await screen.findByText(
        /is not one of the dataset's columns: \['structure', 'y'\]/,
      ),
    ).toBeInTheDocument();
  });

  it("stays unarmed before a dataset + column pick", async () => {
    render(<DLPage />);
    expect(
      await screen.findByText("Train the molecular model"),
    ).toBeDisabled();
    expect(screen.getByText(/— pick a dataset —/)).toBeInTheDocument();
  });
});
