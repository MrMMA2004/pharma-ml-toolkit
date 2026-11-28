import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import { ApiError } from "@/services/api/client";
import type {
  DatasetView,
} from "@/services/datasets/client";

/**
 * ML-5 (the WORKFLOW_ENHANCEMENT_ROADMAP's row 12) — the data-drift
 * lane (stage 10f, the batch-level AD companion), as a complete
 * component-level user flow:
 *
 *   the PSI table        — the drifted rows highlight (the destructive
 *                          row class + data-verdict="shifted"), the
 *                          moderate row carries its own milder class,
 *                          the stable row stays plain — the roadmap's
 *                          own page-test acceptance, verbatim;
 *   the model reference  — the default kind selects a session-trained
 *                          model's own training dataset as the
 *                          reference (modelId rides, datasetId null);
 *   the honest 422       — the backend's own 422 (the exactly-one
 *                          reference contract) surfaces verbatim in
 *                          the lane's own banner, never swallowed;
 *   the pre-flight       — a missing current dataset is refused here
 *                          (the route's own contract mirrored), no
 *                          request fired;
 *   the empty overlap    — a current frame sharing no numeric feature
 *                          answers the honest empty line, never a
 *                          fabricated verdict.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, and the DOM are the real ones.
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

const datasetsMocks = vi.hoisted(() => ({
  list: vi.fn(),
  view: vi.fn(),
  column: vi.fn(),
}));
vi.mock("@/services/datasets/client", () => ({
  datasetsApi: {
    list: datasetsMocks.list,
    view: datasetsMocks.view,
    column: datasetsMocks.column,
  },
}));

const mlMocks = vi.hoisted(() => ({
  trained: vi.fn(),
  catalog: vi.fn(),
  train: vi.fn(),
  admetEndpoints: vi.fn(),
  dtaRegistry: vi.fn(),
  drift: vi.fn(),
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
      drift: mlMocks.drift,
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

const REFERENCE: DatasetView = {
  dataset_id: "ds-ref",
  name: "training-batch.csv",
  source_filename: "training-batch.csv",
  size_bytes: 512,
  shape: [400, 3],
  columns: ["a", "b", "y"],
  target: "y",
  task: "regression",
  n_features: 2,
  feature_names: ["a", "b"],
  classes: null,
  class_counts: null,
  dropped_columns: [],
  categorical_columns: [],
  summary: {},
  created_at: "2026-11-28T00:00:00Z",
};

const CURRENT: DatasetView = {
  ...REFERENCE,
  dataset_id: "ds-cur",
  name: "fresh-batch.csv",
  source_filename: "fresh-batch.csv",
};

const TRAINED = {
  models: [
    {
      model_id: "tm-drift",
      name: "random_forest",
      task: "regression",
      dataset_id: "ds-ref",
      target: "y",
      features: ["a", "b"],
      n_train_rows: 400,
      best_params: {},
      cv_scores: [0.8, 0.81],
      cv_mean: 0.805,
      cv_std: 0.005,
      fit_seconds: 1.2,
      created_at: "2026-11-28T00:00:00Z",
    },
  ],
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

const DRIFT_ANSWER = {
  mode: "sync",
  reference: {
    kind: "dataset",
    model_id: null,
    name: "training-batch.csv",
    dataset_id: "ds-ref",
    n_reference_rows: 400,
  },
  current: {
    dataset_id: "ds-cur",
    name: "fresh-batch.csv",
    n_rows: 400,
  },
  warn_psi: 0.1,
  shift_psi: 0.2,
  n_features: 3,
  n_drifted: 1,
  drift_share: 0.333333,
  features: [
    {
      feature: "molwt",
      psi: 1.742,
      ks_statistic: 0.83,
      ks_pvalue: 3.1e-131,
      reference_mean: 300.1,
      current_mean: 455.2,
      mean_direction: "up",
      verdict: "shifted",
    },
    {
      feature: "logp",
      psi: 0.146,
      ks_statistic: 0.19,
      ks_pvalue: 0.0007,
      reference_mean: 2.4,
      current_mean: 2.9,
      mean_direction: "up",
      verdict: "moderate",
    },
    {
      feature: "tpsa",
      psi: 0.021,
      ks_statistic: 0.06,
      ks_pvalue: 0.42,
      reference_mean: 74.1,
      current_mean: 74.4,
      mean_direction: "flat",
      verdict: "stable",
    },
  ],
  drifted: [
    {
      feature: "molwt",
      psi: 1.742,
      ks_statistic: 0.83,
      ks_pvalue: 3.1e-131,
      reference_mean: 300.1,
      current_mean: 455.2,
      mean_direction: "up",
      verdict: "shifted",
    },
  ],
  summary:
    "drift: 1/3 feature(s) shifted vs the training frame (PSI >= 0.2)\n  molwt (PSI 1.742, mean up)",
};

beforeEach(() => {
  jobState.snapshots = {};
  vi.clearAllMocks();
  datasetsMocks.list.mockResolvedValue({
    datasets: [REFERENCE, CURRENT],
  });
  datasetsMocks.view.mockResolvedValue(REFERENCE);
  mlMocks.trained.mockResolvedValue(TRAINED);
  mlMocks.catalog.mockResolvedValue(CATALOG);
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

async function selectLane() {
  // the drift lane lives inside stage 10's card; the selects are
  // labelled and the lane renders unconditionally (no trained-model
  // gate — the roadmap's own unsupervised case)
  const kindSelect = await screen.findByLabelText("the drift reference kind");
  return { kindSelect };
}

describe("MLPage stage 10f — the ML-5 data-drift lane (the batch-level AD companion)", () => {
  it("renders the PSI table with the drifted rows highlighted", async () => {
    mlMocks.drift.mockResolvedValueOnce(DRIFT_ANSWER);

    render(<MLPage />);

    await selectLane();
    // the dataset-reference kind (flip away from the default model
    // kind so the dataset select renders)
    await userEvent.selectOptions(
      screen.getByLabelText("the drift reference kind"),
      "dataset",
    );
    await userEvent.selectOptions(
      screen.getByLabelText("the drift reference dataset"),
      "ds-ref",
    );
    await userEvent.selectOptions(
      screen.getByLabelText("the drift current dataset"),
      "ds-cur",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Score the drift" }),
    );

    expect(mlMocks.drift).toHaveBeenCalledTimes(1);
    expect(mlMocks.drift).toHaveBeenCalledWith({
      modelId: null,
      datasetId: "ds-ref",
      newDatasetId: "ds-cur",
    });

    // the summary line names the shifted share + the engine's own
    // first summary line
    expect(
      await screen.findByText(/1 of 3 feature\(s\) shifted/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/1\/3 feature\(s\) shifted vs the training frame/),
    ).toBeInTheDocument();

    // the PSI table: the drifted row highlighted with its own
    // data-verdict attribute, the stable row plain
    const shiftedRow = screen.getByText("molwt").closest("tr");
    expect(shiftedRow).not.toBeNull();
    expect(shiftedRow).toHaveAttribute("data-verdict", "shifted");
    expect(shiftedRow?.className).toContain("font-medium");
    const moderateRow = screen.getByText("logp").closest("tr");
    expect(moderateRow).toHaveAttribute("data-verdict", "moderate");
    expect(moderateRow?.className).not.toContain("font-medium");
    const stableRow = screen.getByText("tpsa").closest("tr");
    expect(stableRow).toHaveAttribute("data-verdict", "stable");
    expect(stableRow?.className).not.toContain("font-medium");

    // the reference/current provenance line rides the payload's own
    // fields
    expect(
      screen.getByText(/dataset training-batch.csv \(400 rows\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/current: fresh-batch.csv \(400 rows\)/),
    ).toBeInTheDocument();
  });

  it("rides the model reference kind with the trained model's own id", async () => {
    mlMocks.drift.mockResolvedValueOnce({
      ...DRIFT_ANSWER,
      reference: {
        kind: "model",
        model_id: "tm-drift",
        name: "random_forest",
        dataset_id: "ds-ref",
        n_reference_rows: 400,
      },
    });

    render(<MLPage />);

    await selectLane();
    // the default kind is the model lane — the trained registry
    // carries the session's one model
    await userEvent.selectOptions(
      screen.getByLabelText("the drift reference model"),
      "tm-drift",
    );
    await userEvent.selectOptions(
      screen.getByLabelText("the drift current dataset"),
      "ds-cur",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Score the drift" }),
    );

    expect(mlMocks.drift).toHaveBeenCalledWith({
      modelId: "tm-drift",
      datasetId: null,
      newDatasetId: "ds-cur",
    });
    expect(
      await screen.findByText(/model random_forest \(400 training rows\)/),
    ).toBeInTheDocument();
  });

  it("surfaces the backend's own 422 verbatim, never swallowed", async () => {
    mlMocks.drift.mockRejectedValueOnce(
      new ApiError(
        422,
        "give either model_id=... or dataset_id=... as the drift reference, not both/neither",
      ),
    );

    render(<MLPage />);

    await selectLane();
    await userEvent.selectOptions(
      screen.getByLabelText("the drift reference kind"),
      "dataset",
    );
    await userEvent.selectOptions(
      screen.getByLabelText("the drift reference dataset"),
      "ds-ref",
    );
    await userEvent.selectOptions(
      screen.getByLabelText("the drift current dataset"),
      "ds-cur",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Score the drift" }),
    );

    expect(
      await screen.findByText(/give either model_id=... or dataset_id=.../),
    ).toBeInTheDocument();
  });

  it("refuses a missing current dataset pre-flight, no request fired", async () => {
    render(<MLPage />);

    await selectLane();
    await userEvent.selectOptions(
      screen.getByLabelText("the drift reference kind"),
      "dataset",
    );
    await userEvent.selectOptions(
      screen.getByLabelText("the drift reference dataset"),
      "ds-ref",
    );
    // no current dataset selected — the lane's own refusal
    await userEvent.click(
      screen.getByRole("button", { name: "Score the drift" }),
    );

    expect(mlMocks.drift).not.toHaveBeenCalled();
    expect(
      await screen.findByText(/select the current dataset \(the new batch\) first/),
    ).toBeInTheDocument();
  });

  it("answers the no-shared-features overlap honestly, never a verdict", async () => {
    mlMocks.drift.mockResolvedValueOnce({
      ...DRIFT_ANSWER,
      n_features: 0,
      n_drifted: 0,
      drift_share: 0.0,
      features: [],
      drifted: [],
      summary: "drift: no numeric features compared",
    });

    render(<MLPage />);

    await selectLane();
    await userEvent.selectOptions(
      screen.getByLabelText("the drift reference kind"),
      "dataset",
    );
    await userEvent.selectOptions(
      screen.getByLabelText("the drift reference dataset"),
      "ds-ref",
    );
    await userEvent.selectOptions(
      screen.getByLabelText("the drift current dataset"),
      "ds-cur",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Score the drift" }),
    );

    expect(
      await screen.findByText(/no shared numeric features/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/never a verdict/),
    ).toBeInTheDocument();
  });
});
