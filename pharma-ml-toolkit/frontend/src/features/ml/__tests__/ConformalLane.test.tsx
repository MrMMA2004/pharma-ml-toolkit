import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import type { DatasetView } from "@/services/datasets/client";

/**
 * ML-1 — the Explainability & Uncertainty stage's conformal panel as a
 * complete component-level user flow over the real page mount:
 *
 *   the trained-model row click arms the panel (the same
 *                         selectTrainedModel lane the SHAP card uses);
 *   the calibration run  — the bounded column bridge assembles the
 *                         calibration + test slices over the mocked
 *                         column endpoint, the request reaches
 *                         modelEvaluationApi.conformalRegression with
 *                         the exact wire fields, and the answer renders
 *                         the interval bounds + the empirical coverage +
 *                         the Wilson-bounded scaffold coverage table;
 *   the family catalog   — fetched once and its honesty note rendered.
 *
 * Services are mocked at the network boundary only; the component, the
 * bridge handler, the runtime parsers (the REAL ones, spread from the
 * actual module), and the DOM are the real ones.
 */

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

const conformalMocks = vi.hoisted(() => ({
  catalog: vi.fn(),
  conformalRegression: vi.fn(),
  conformalClassification: vi.fn(),
  coverageDiagnostics: vi.fn(),
}));

const mlMocks = vi.hoisted(() => ({
  trained: vi.fn(),
  trainedView: vi.fn(),
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
      trainedView: mlMocks.trainedView,
      catalog: mlMocks.catalog,
      train: mlMocks.train,
      admetEndpoints: mlMocks.admetEndpoints,
      dtaRegistry: mlMocks.dtaRegistry,
    },
    modelEvaluationApi: {
      ...actual.modelEvaluationApi,
      catalog: conformalMocks.catalog,
      conformalRegression: conformalMocks.conformalRegression,
      conformalClassification: conformalMocks.conformalClassification,
      coverageDiagnostics: conformalMocks.coverageDiagnostics,
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
  name: "conformal-fixture.csv",
  source_filename: "conformal-fixture.csv",
  size_bytes: 512,
  shape: [12, 3],
  columns: ["smiles", "x", "y"],
  target: "y",
  task: "regression",
  n_features: 1,
  feature_names: ["x"],
  classes: null,
  class_counts: null,
  dropped_columns: [],
  categorical_columns: ["smiles"],
  summary: {},
  created_at: "2026-11-29T00:00:00Z",
};

const TRAINED = {
  model_id: "mdl-cf1",
  name: "ridge",
  task: "regression",
  dataset_id: "ds-1",
  target: "y",
  features: ["x"],
  n_train_rows: 12,
  best_params: {},
  cv_scores: [0.9, 0.91],
  cv_mean: 0.905,
  cv_std: 0.005,
  fit_seconds: 0.02,
  created_at: "2026-11-29T00:00:00Z",
};

const REGRESSION_PAYLOAD = {
  alpha: 0.1,
  radius: 0.25,
  n_calibration: 8,
  rank: 7,
  mean_width: 0.5,
  feature_names: ["x"],
  intervals: [
    { prediction: 1.2, lower: 0.95, upper: 1.45 },
    { prediction: 2.4, lower: 2.15, upper: 2.65 },
    { prediction: 3.6, lower: 3.35, upper: 3.85 },
    { prediction: 4.8, lower: 4.55, upper: 5.05 },
  ],
  empirical: { n_test: 4, n_covered: 4, coverage: 1.0 },
  coverage_diagnostics: {
    mode: "scaffold",
    confidence: 0.95,
    overall: {
      group: "overall",
      n_test: 4,
      n_covered: 4,
      coverage: 1.0,
      lower: 0.51,
      upper: 1.0,
      mean_size: 0.5,
    },
    groups: [
      {
        group: "c1ccccc1",
        n_test: 2,
        n_covered: 2,
        coverage: 1.0,
        lower: 0.34,
        upper: 1.0,
        mean_size: 0.5,
      },
      {
        group: "CCO",
        n_test: 2,
        n_covered: 2,
        coverage: 1.0,
        lower: 0.34,
        upper: 1.0,
        mean_size: 0.5,
      },
    ],
  },
};

const CATALOG_PAYLOAD = {
  entry_points: [
    {
      route: "/conformal-regression",
      backend: "calibrate_split_conformal + predict_interval",
      defaults: { alpha: 0.1 },
    },
  ],
  interval_columns: ["prediction", "lower", "upper"],
  set_columns: ["prediction_set", "set_size", "covered"],
  coverage_columns: ["group", "n_test", "n_covered", "coverage"],
  notes: [
    "sync-only: one predict pass + a quantile (the knn/domain-box precedent for light compute)",
    "empirical coverage is empirical, never the conformal guarantee",
  ],
};

beforeEach(() => {
  datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
  datasetsMocks.view.mockResolvedValue(DATASET);
  datasetsMocks.column.mockImplementation(
    async (_datasetId: string, column: string, _limit: number) => {
      if (column === "x") {
        return {
          column,
          dtype: "float64",
          n_rows: 12,
          returned: 12,
          truncated: false,
          values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        };
      }
      if (column === "y") {
        return {
          column,
          dtype: "float64",
          n_rows: 12,
          returned: 12,
          truncated: false,
          values: [0.9, 2.1, 3.0, 4.2, 5.1, 5.9, 7.2, 8.1, 8.8, 10.2, 11.1, 11.9],
        };
      }
      return {
        column,
        dtype: "object",
        n_rows: 12,
        returned: 12,
        truncated: false,
        values: [
          "c1ccccc1",
          "c1ccccc1",
          "CCO",
          "CCO",
          "c1ccccc1",
          "CCO",
          "c1ccccc1",
          "CCO",
          "c1ccccc1",
          "CCO",
          "c1ccccc1",
          "CCO",
        ],
      };
    },
  );
  mlMocks.trained.mockResolvedValue({ models: [TRAINED] });
  mlMocks.trainedView.mockResolvedValue(TRAINED);
  mlMocks.catalog.mockResolvedValue({ models: [] });
  conformalMocks.catalog.mockResolvedValue(CATALOG_PAYLOAD);
  conformalMocks.conformalRegression.mockReset();
  conformalMocks.conformalClassification.mockReset();
  conformalMocks.coverageDiagnostics.mockReset();
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

async function driveToConformalReady() {
  render(<MLPage />);
  // stage 1: select the dataset
  const rows = await screen.findAllByText("conformal-fixture.csv");
  await userEvent.click(rows[0]!);
  await waitFor(() => {
    expect(datasetsMocks.view).toHaveBeenCalled();
  });
  // the trained registry row click selects the model (the Compare
  // stage's trained table — the same lane the SHAP card rides)
  const trainedRows = await screen.findAllByText("mdl-cf1");
  await userEvent.click(trainedRows[0]!);
  await waitFor(() => {
    expect(mlMocks.trainedView).toHaveBeenCalledWith("mdl-cf1");
  });
}

describe("the ML-1 conformal panel of the Machine Learning workspace", () => {
  it("calibrates over the bridge and renders bounds, coverage, and the scaffold table", async () => {
    await driveToConformalReady();

    // the panel is armed by the trained-model selection
    const runButton = await screen.findByRole("button", {
      name: /Run the conformal calibration/,
    });
    expect(runButton).toBeEnabled();

    // the family catalog rides along once, its honesty note rendered
    await waitFor(() => {
      expect(conformalMocks.catalog).toHaveBeenCalledTimes(1);
    });
    expect(
      await screen.findByText(/never the conformal guarantee/),
    ).toBeInTheDocument();

    conformalMocks.conformalRegression.mockResolvedValue(REGRESSION_PAYLOAD);
    // re-query at click time: the catalog fetch's re-render may detach
    // the earlier button node (the detached-node hazard)
    await userEvent.click(
      screen.getByRole("button", {
        name: /Run the conformal calibration/,
      }),
    );

    await waitFor(() => {
      expect(conformalMocks.conformalRegression).toHaveBeenCalledTimes(1);
    });
    const request = conformalMocks.conformalRegression.mock.calls[0]![0] as {
      modelId: string;
      xCal: number[][];
      yCal: number[];
      xTest: number[][];
      yTest: number[];
      alpha: number;
      smiles: string[];
    };
    // the wire contract: the model id, the two slices, the alpha, the
    // test-slice SMILES for the scaffold coverage table
    expect(request.modelId).toBe("mdl-cf1");
    expect(request.alpha).toBe(0.1);
    expect(request.xCal).toHaveLength(8);
    expect(request.yCal[0]).toBe(0.9);
    expect(request.xTest).toHaveLength(4);
    expect(request.yTest).toEqual([8.8, 10.2, 11.1, 11.9]);
    expect(request.smiles).toEqual([
      "c1ccccc1",
      "CCO",
      "c1ccccc1",
      "CCO",
    ]);

    // the interval bounds table renders the parsed rows
    const intervalsTable = await screen.findByLabelText(
      "conformal interval rows",
    );
    const intervalCells = within(intervalsTable).getAllByText("2.1500");
    expect(intervalCells).toHaveLength(1);
    expect(within(intervalsTable).getAllByText("2.6500")).toHaveLength(1);

    // the empirical coverage line
    expect(await screen.findByText(/4\/4 covered/)).toBeInTheDocument();

    // the Wilson-bounded scaffold coverage table (overall + two groups)
    const coverageTable = await screen.findByLabelText(
      "conformal coverage table",
    );
    expect(
      within(coverageTable).getAllByText("c1ccccc1").length,
    ).toBeGreaterThanOrEqual(1);
    expect(within(coverageTable).getAllByText("CCO").length).toBe(
      1,
    );
    expect(
      within(coverageTable).getAllByText("overall").length,
    ).toBeGreaterThanOrEqual(1);
  });

  it("keeps the no-SMILES lane honest (no coverage table, null smiles)", async () => {
    await driveToConformalReady();

    await userEvent.selectOptions(
      screen.getByLabelText("conformal smiles column"),
      "",
    );
    const payload = JSON.parse(JSON.stringify(REGRESSION_PAYLOAD));
    payload["coverage_diagnostics"] = null;
    conformalMocks.conformalRegression.mockResolvedValue(payload);

    await userEvent.click(
      screen.getByRole("button", { name: /Run the conformal calibration/ }),
    );

    await waitFor(() => {
      expect(conformalMocks.conformalRegression).toHaveBeenCalledTimes(1);
    });
    const request = conformalMocks.conformalRegression.mock.calls[0]![0] as {
      smiles: string[] | null;
    };
    expect(request.smiles).toBeNull();
    await waitFor(() => {
      expect(
        screen.queryByLabelText("conformal coverage table"),
      ).toBeNull();
    });
    // the interval table still renders
    expect(
      await screen.findByLabelText("conformal interval rows"),
    ).toBeInTheDocument();
  });

  it("surfaces the bridge's honest refusal when the slices cannot fill", async () => {
    await driveToConformalReady();

    // a degenerate 2-row dataset: the ratio cap leaves the test slice
    // empty — the bridge refuses honestly and the request never fires
    datasetsMocks.column.mockImplementation(
      async (_datasetId: string, column: string, _limit: number) => {
        if (column === "x") {
          return {
            column,
            dtype: "float64",
            n_rows: 2,
            returned: 2,
            truncated: false,
            values: [1, 2],
          };
        }
        if (column === "y") {
          return {
            column,
            dtype: "float64",
            n_rows: 2,
            returned: 2,
            truncated: false,
            values: [0.9, 2.1],
          };
        }
        return {
          column,
          dtype: "object",
          n_rows: 2,
          returned: 2,
          truncated: false,
          values: ["c1ccccc1", "CCO"],
        };
      },
    );

    await userEvent.click(
      screen.getByRole("button", { name: /Run the conformal calibration/ }),
    );

    // the request never fires — the bridge refuses first
    await waitFor(() => {
      expect(
        screen.getByText(/assembled only \d+ calibration rows/),
      ).toBeInTheDocument();
    });
    expect(conformalMocks.conformalRegression).not.toHaveBeenCalled();
  });
});
