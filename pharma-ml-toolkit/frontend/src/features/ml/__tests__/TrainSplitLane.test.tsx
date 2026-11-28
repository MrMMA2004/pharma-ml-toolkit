import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import type { DatasetView } from "@/services/datasets/client";

/**
 * ML-3 — the Configure Split/CV stage's strategy lane of the Machine
 * Learning workspace as a complete component-level user flow:
 *
 *   the selector        — "none" keeps the legacy full-frame request
 *                         (no split key at all); picking a strategy
 *                         sends the split section through with the
 *                         honest column defaults (scaffold auto-detect,
 *                         group never guessed);
 *   the group gate       — the Train button stays disabled until an
 *                         explicit group column is picked;
 *   the carve readout   — the completed job's relayed result renders
 *                         the carve block: the strategy, the two sides,
 *                         the per-side group counts and the
 *                         shared-groups leakage verdict.
 *
 * Services are mocked at the network boundary only; the component, its
 * state machine, the runtime parsers (parseTrainResult /
 * parseTrainSplitBlock — the REAL ones, spread from the actual
 * module), and the DOM are the real ones.
 */

const jobState = vi.hoisted(() => {
  const state: {
    snapshots: Record<string, { view: Record<string, unknown> } | null>;
  } = { snapshots: {} };
  return state;
});

vi.mock("@/hooks/useJobTracking", () => ({
  useJob: (jobId: string | null) => ({
    snapshot: jobId === null ? null : jobState.snapshots[jobId] ?? null,
    cancel: jobId === null ? vi.fn() : vi.fn(),
    refresh: jobId === null ? vi.fn() : vi.fn(),
  }),
}));

const datasetsMocks = vi.hoisted(() => ({ list: vi.fn(), view: vi.fn() }));
vi.mock("@/services/datasets/client", () => ({
  datasetsApi: { list: datasetsMocks.list, view: datasetsMocks.view },
}));

const mlMocks = vi.hoisted(() => ({
  trained: vi.fn(),
  catalog: vi.fn(),
  train: vi.fn(),
  admetEndpoints: vi.fn(),
  dtaRegistry: vi.fn(),
}));
vi.mock("@/services/ml/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/ml/client")>();
  return {
    ...actual,
    mlApi: {
      ...actual.mlApi,
      trained: mlMocks.trained,
      catalog: mlMocks.catalog,
      train: mlMocks.train,
      admetEndpoints: mlMocks.admetEndpoints,
      dtaRegistry: mlMocks.dtaRegistry,
    },
  };
});

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

const DATASET: DatasetView = {
  dataset_id: "ds-1",
  name: "scaffold-fixture.csv",
  source_filename: "scaffold-fixture.csv",
  size_bytes: 512,
  shape: [12, 4],
  columns: ["smiles", "grp", "x"],
  target: "y",
  task: "classification",
  n_features: 1,
  feature_names: ["x"],
  classes: [0, 1],
  class_counts: { "0": 6, "1": 6 },
  dropped_columns: [],
  categorical_columns: ["smiles", "grp"],
  summary: {},
  created_at: "2026-11-28T00:00:00Z",
};

const CATALOG = {
  models: [
    {
      name: "random_forest",
      task: "classification",
      available: true,
      requires: null,
      notes: "",
    },
  ],
};

const JOB = {
  job_id: "job-1",
  operation: "ml.train",
  workspace: "ml",
  state: "RUNNING",
  created_at: "2026-11-28T00:00:00Z",
  started_at: "2026-11-28T00:00:00Z",
  completed_at: null,
  progress: 5,
  stage: "queued",
  configuration: {},
  cancellation: "not_requested",
  error: null,
};

const TRAIN_RESULT = {
  model: {
    model_id: "mdl-ml300000001",
    name: "random_forest",
    task: "classification",
    dataset_id: "ds-1",
    target: "y",
    features: ["x"],
    n_train_rows: 8,
    best_params: {},
    cv_scores: [1.0, 1.0],
    cv_mean: 1.0,
    cv_std: 0.0,
    fit_seconds: 0.02,
    created_at: "2026-11-28T00:00:01Z",
  },
  split: {
    strategy: "scaffold",
    smiles_column: "smiles",
    group_column: null,
    test_size: 0.25,
    random_state: 42,
    n_rows: 12,
    sides: { train: { rows: 8 }, heldout: { rows: 4 } },
    group_census: {
      train: { n_groups: 1, top_groups: [["c1ccccc1", 8]] },
      heldout: { n_groups: 1, top_groups: [["", 4]] },
      shared_groups: [],
    },
  },
};

beforeEach(() => {
  jobState.snapshots = {};
  datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
  datasetsMocks.view.mockResolvedValue(DATASET);
  mlMocks.trained.mockResolvedValue({ models: [] });
  mlMocks.catalog.mockResolvedValue(CATALOG);
  mlMocks.train.mockReset();
  jobsMocks.result.mockReset();
  mlMocks.admetEndpoints.mockResolvedValue({
    endpoints: [],
    categories: [],
    radial_axes: [],
    low_confidence_endpoints: [],
    provenance: {
      backend: "pharmaml.ml.admet",
      source: "",
      n_endpoints: 0,
      n_learned_endpoints: 0,
      shipped_ensembles: 0,
      shipped_ensembles_note: "",
      low_performance_threshold: 0.6,
      disclaimer: "",
    },
    score_url: "",
    note: "",
  });
  mlMocks.dtaRegistry.mockResolvedValue({
    alphabets: {
      protein: {
        n_symbols: 25,
        padding_code: 0,
        symbols: { A: 1 },
        encoding: "",
        source: "",
      },
      smiles_isomeric: {
        n_symbols: 64,
        padding_code: 0,
        symbols: { C: 42 },
        encoding: "",
        source: "",
      },
      smiles_canonical: {
        n_symbols: 62,
        padding_code: 0,
        symbols: { "#": 1 },
        encoding: "",
        source: "",
      },
    },
    amino_acids: Array.from("ACDEFGHIKLMNPQRSTVWY"),
    protein_features: {
      n_base: 23,
      base_columns: ["prot_length"],
      n_dipeptide_extras: 401,
      n_with_dipeptides: 424,
      note: "",
    },
    ligand_features: {
      family: "ecfp",
      n_bits: 2048,
      source: "",
    },
    paired_width: { base: 2071, with_dipeptides: 2472 },
    metrics: [{ name: "deepdta_rm2", description: "" }],
    demo_frame: { rows: 12, note: "" },
    provenance: {
      backend: "pharmaml.ml.dta",
      source: "",
      n_alphabets: 3,
      shipped_checkpoints: 0,
      shipped_checkpoints_note: "",
      disclaimer: "",
    },
    features_url: "/api/v1/dta/features",
    note: "",
  });
});

async function driveToTrainReady() {
  render(<MLPage />);
  // stage 1: the dataset row selection loads the view + the catalog
  // (the name renders in the registry row and the selection summary —
  // always click a freshly queried node: the registry re-renders)
  const rows = await screen.findAllByText("scaffold-fixture.csv");
  await userEvent.click(rows[0]!);
  // stage 5: the model row selection — wait for the catalog row to
  // mount (the fetch follows the selection), then click the CURRENT
  // node and gate on the Train button actually arming (the summary's
  // model slot + the enabled button — the drive's real contract)
  await waitFor(() => {
    expect(screen.getAllByText("random_forest").length).toBeGreaterThan(0);
  });
  await userEvent.click(screen.getAllByText("random_forest")[0]!);
  await waitFor(() => {
    expect(
      screen.getByRole("button", { name: /Train the model/ }),
    ).toBeEnabled();
  });
}

describe("the ML-3 split strategy lane of the Machine Learning workspace", () => {
  it("sends the scaffold section through and renders the carve + the verdict", async () => {
    await driveToTrainReady();
    await userEvent.selectOptions(
      screen.getByLabelText("split strategy"),
      "scaffold",
    );
    mlMocks.train.mockResolvedValue({ mode: "job", job: JOB });
    jobState.snapshots["job-1"] = {
      view: {
        ...JOB,
        state: "COMPLETED",
        completed_at: "2026-11-28T00:00:02Z",
        progress: 100,
        stage: "completed",
        result_ready: true,
      },
    };
    jobsMocks.result.mockResolvedValue({ result: TRAIN_RESULT });

    await userEvent.click(screen.getByRole("button", { name: /Train the model/ }));

    await waitFor(() => {
      expect(mlMocks.train).toHaveBeenCalledTimes(1);
    });
    const options = (mlMocks.train.mock.calls[0] as unknown[])[0] as Record<
      string,
      unknown
    >;
    expect(options["split"]).toEqual({
      strategy: "scaffold",
      smilesColumn: null,
      groupColumn: null,
      testSize: 0.2,
    });

    // the completed relay renders the carve: the strategy line, the
    // sides, the census counts, and the leakage verdict
    await waitFor(() => {
      expect(
        screen.getByText(/0 shared groups across the sides/),
      ).toBeInTheDocument();
    });
    expect(screen.getByText(/strategy=scaffold/)).toBeInTheDocument();
    expect(screen.getByText(/Bemis–Murcko families kept whole/)).toBeInTheDocument();
    expect(
      screen.getByText(/train 8 of 12 rows · held-out 4/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/1 train groups · 1 held-out groups/),
    ).toBeInTheDocument();
    expect(jobsMocks.result).toHaveBeenCalledWith("job-1");
  });

  it("keeps the group strategy gated until an explicit column is picked", async () => {
    await driveToTrainReady();
    await userEvent.selectOptions(
      screen.getByLabelText("split strategy"),
      "group",
    );
    // the group column select appears with the dataset's own columns
    const groupSelect = screen.getByLabelText("group column");
    expect(screen.getByText(/group labels are never guessed/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Train the model/ }),
    ).toBeDisabled();

    await userEvent.selectOptions(groupSelect, "grp");
    // the gate opens — re-query the button at click time (the re-render
    // after the selection re-creates the node)
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /Train the model/ }),
      ).toBeEnabled();
    });
    mlMocks.train.mockResolvedValue({ mode: "job", job: JOB });
    await userEvent.click(
      screen.getByRole("button", { name: /Train the model/ }),
    );

    await waitFor(() => {
      expect(mlMocks.train).toHaveBeenCalledTimes(1);
    });
    const options = (mlMocks.train.mock.calls[0] as unknown[])[0] as Record<
      string,
      unknown
    >;
    expect(options["split"]).toEqual({
      strategy: "group",
      smilesColumn: null,
      groupColumn: "grp",
      testSize: 0.2,
    });
  });

  it("keeps the none lane on the legacy full-frame request", async () => {
    await driveToTrainReady();
    mlMocks.train.mockResolvedValue({ mode: "job", job: JOB });
    jobState.snapshots["job-1"] = {
      view: {
        ...JOB,
        state: "COMPLETED",
        completed_at: "2026-11-28T00:00:02Z",
        progress: 100,
        stage: "completed",
        result_ready: true,
      },
    };
    // the legacy result carries no split key — the parser answers null
    jobsMocks.result.mockResolvedValue({
      result: { model: TRAIN_RESULT["model"] },
    });

    await userEvent.click(screen.getByRole("button", { name: /Train the model/ }));

    await waitFor(() => {
      expect(mlMocks.train).toHaveBeenCalledTimes(1);
    });
    const options = (mlMocks.train.mock.calls[0] as unknown[])[0] as Record<
      string,
      unknown
    >;
    expect(options["split"]).toBeNull();
    // the config summary states the legacy lane, no carve readout
    expect(screen.getByText(/split full frame/)).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText(/mdl-ml300000001/)).toBeInTheDocument();
    });
    expect(screen.queryByText(/shared groups across the sides/)).toBeNull();
  });
});
