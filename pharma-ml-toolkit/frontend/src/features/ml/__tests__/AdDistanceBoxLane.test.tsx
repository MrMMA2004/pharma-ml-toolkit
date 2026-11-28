import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import { ApiError } from "@/services/api/client";
import type {
  ColumnValues,
  DatasetView,
} from "@/services/datasets/client";

/**
 * ML-2 (the WORKFLOW_ENHANCEMENT_ROADMAP's row 10) — the two AD
 * sub-lanes rendered beside the analyze/Williams cards, as a
 * complete component-level user flow:
 *
 *   the matrix-only verdict — a user with NO trained model reaches
 *                            a distance verdict: the two raw matrices
 *                            POST through knnDomain, the verdict
 *                            line + the per-row table render;
 *   the honest 422        — the backend's own 422 (a feature-count
 *                            mismatch) surfaces verbatim in the lane's
 *                            own banner, never swallowed;
 *   the pre-flight        — a feature-count disagreement the page can
 *                            see coming is refused here (the route's
 *                            own message mirrored), no request fired;
 *   the box verdict       — the persisted bundle's metadata + the
 *                            pasted rows produce the checked report
 *                            (the out counts + the below/above axes);
 *   the unchecked answer  — a metadata block without a box answers
 *                            `not checked` with the route's own
 *                            reason, never an implied verdict;
 *   the bridged rows     — the box features' columns pulled through
 *                            the SAME bounded column() lane fill the
 *                            rows textarea verbatim (JSON lines).
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
  knnDomain: vi.fn(),
  domainBoxCheck: vi.fn(),
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
      knnDomain: mlMocks.knnDomain,
      domainBoxCheck: mlMocks.domainBoxCheck,
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
  dataset_id: "ds-box",
  name: "solubility-box.csv",
  source_filename: "solubility-box.csv",
  size_bytes: 512,
  shape: [3, 4],
  columns: ["molwt", "logp", "tpsa"],
  target: null,
  task: "regression",
  n_features: 3,
  feature_names: ["molwt", "logp", "tpsa"],
  classes: null,
  class_counts: null,
  dropped_columns: [],
  categorical_columns: [],
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

function columnPayload(
  column: string,
  values: (string | number | null)[],
): ColumnValues {
  return {
    column,
    dtype: "float64",
    n_rows: values.length,
    returned: values.length,
    truncated: false,
    values,
  };
}

const KNN_ANSWER = {
  k: 5,
  percentile: 95,
  distance_threshold: 1.204,
  train_knn_mean: 0.612,
  n_train: 12,
  n_test: 3,
  n_inside: 2,
  frame: [
    { name: "compound_0", knn_distance: 0.31, inside_ad: true },
    { name: "compound_1", knn_distance: 0.44, inside_ad: true },
    { name: "compound_2", knn_distance: 9.7, inside_ad: false },
  ],
};

beforeEach(() => {
  jobState.snapshots = {};
  vi.clearAllMocks();
  datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
  datasetsMocks.view.mockResolvedValue(DATASET);
  mlMocks.trained.mockResolvedValue({ models: [] });
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

describe("MLPage stage 10 — the ML-2 sub-lanes (the distance-only kNN + the domain-box check)", () => {
  it("lets a matrix-only user reach a verdict without a trained model", async () => {
    mlMocks.knnDomain.mockResolvedValueOnce(KNN_ANSWER);

    render(<MLPage />);

    // the sub-lane renders with NO trained model selected — the
    // unsupervised case the roadmap names
    expect(
      screen.getByText("Distance-only AD (kNN)"),
    ).toBeInTheDocument();
    expect(screen.getByText("no trained model required")).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("button", { name: "Run the distance-only AD" }),
    );

    await waitFor(() => {
      expect(mlMocks.knnDomain).toHaveBeenCalledTimes(1);
    });
    const options = mlMocks.knnDomain.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    // the seeded sample matrices parsed verbatim: 12 train rows, 3
    // query rows, both 3-wide
    expect((options["xTrain"] as unknown[]).length).toBe(12);
    expect((options["xTest"] as unknown[]).length).toBe(3);
    expect(options["k"]).toBe(5);
    expect(options["metric"]).toBe("euclidean");

    // the verdict line + the per-row table
    expect(
      await screen.findByText(/inside 2\/3 of 12\+3 rows/),
    ).toBeInTheDocument();
    expect(await screen.findByText("compound_2")).toBeInTheDocument();
    expect(screen.getByText(/9\.7000|9\.7/)).toBeInTheDocument();
  });

  it("surfaces the backend's own 422 verbatim — never swallowed", async () => {
    mlMocks.knnDomain.mockRejectedValueOnce(
      new ApiError(
        422,
        "x_train has 3 features; x_test has 2 (they must match)",
      ),
    );

    render(<MLPage />);
    await userEvent.click(
      screen.getByRole("button", { name: "Run the distance-only AD" }),
    );

    expect(
      await screen.findByText(
        /API 422: x_train has 3 features; x_test has 2 \(they must match\)/,
      ),
    ).toBeInTheDocument();
  });

  it("refuses a feature-count disagreement the page can see coming — no request fired", async () => {
    render(<MLPage />);

    const train = screen.getByLabelText("the x_train matrix");
    await userEvent.clear(train);
    await userEvent.type(train, "1 2 3\n4 5 6\n7 8 9\n1 2 3\n4 5 6\n7 8 9\n1 2 3\n4 5 6\n7 8 9\n1 2 3");
    const test = screen.getByLabelText("the x_test matrix");
    await userEvent.clear(test);
    await userEvent.type(test, "1 2\n3 4");

    await userEvent.click(
      screen.getByRole("button", { name: "Run the distance-only AD" }),
    );

    expect(
      await screen.findByText(
        /x_train has 3 features; x_test has 2 \(they must match\)/,
      ),
    ).toBeInTheDocument();
    expect(mlMocks.knnDomain).not.toHaveBeenCalled();
  });

  it("renders the box verdict — the out counts + the below/above axes", async () => {
    mlMocks.domainBoxCheck.mockResolvedValueOnce({
      checked: true,
      n_rows: 3,
      n_out: 2,
      fraction_out: 0.6666666666666666,
      below: { logp: 1 },
      above: { molwt: 1 },
      out: [false, true, true],
    });

    render(<MLPage />);

    // the default metadata + rows samples ride the lane verbatim
    await userEvent.click(
      screen.getByRole("button", { name: "Check the domain box" }),
    );

    await waitFor(() => {
      expect(mlMocks.domainBoxCheck).toHaveBeenCalledTimes(1);
    });
    const [metadata, rows] = mlMocks.domainBoxCheck.mock.calls[0] as unknown[];
    expect(metadata).toHaveProperty("feature_ranges");
    expect((rows as unknown[]).length).toBe(3);

    expect(
      await screen.findByText(/2 of 3 rows outside the box/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/below the training minimum: logp ×1/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/above the training maximum: molwt ×1/),
    ).toBeInTheDocument();
  });

  it("renders the unchecked answer with the route's own reason — never an implied verdict", async () => {
    mlMocks.domainBoxCheck.mockResolvedValueOnce({
      checked: false,
      reason: "the metadata carries no feature_ranges box",
      n_rows: 3,
    });

    render(<MLPage />);

    const metadata = screen.getByLabelText(
      "the persisted bundle's metadata JSON",
    );
    fireEvent.change(metadata, {
      target: { value: '{"name": "old-bundle"}' },
    });

    await userEvent.click(
      screen.getByRole("button", { name: "Check the domain box" }),
    );

    expect(
      await screen.findByText(
        /not checked — the metadata carries no feature_ranges box \(3 rows submitted\)/,
      ),
    ).toBeInTheDocument();
  });

  it("bridges the box features' columns from a registered dataset — the rows textarea fills verbatim", async () => {
    datasetsMocks.column.mockImplementation(
      async (_datasetId: string, column: string) => {
        if (column === "molwt") {
          return columnPayload("molwt", [250.3, 720.9, 180.2]);
        }
        if (column === "logp") {
          return columnPayload("logp", [2.4, 1.1, -3.5]);
        }
        return columnPayload("tpsa", [46.5, 88.0, 21.6]);
      },
    );

    render(<MLPage />);

    await waitFor(() => {
      expect(
        screen.getByRole("option", { name: /solubility-box\.csv · 3 rows/ }),
      ).toBeInTheDocument();
    });
    await userEvent.selectOptions(
      screen.getByLabelText("the box-bridge dataset"),
      "ds-box",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Assemble rows from dataset" }),
    );

    await waitFor(() => {
      expect(datasetsMocks.column).toHaveBeenCalledWith("ds-box", "molwt", 500);
      expect(datasetsMocks.column).toHaveBeenCalledWith("ds-box", "logp", 500);
      expect(datasetsMocks.column).toHaveBeenCalledWith("ds-box", "tpsa", 500);
    });
    await waitFor(() => {
      expect(screen.getByLabelText("the rows to check")).toHaveValue(
        [
          '{"molwt":250.3,"logp":2.4,"tpsa":46.5}',
          '{"molwt":720.9,"logp":1.1,"tpsa":88}',
          '{"molwt":180.2,"logp":-3.5,"tpsa":21.6}',
        ].join("\n"),
      );
    });
    expect(
      await screen.findByText(/3 rows assembled from the box's 3 feature columns/),
    ).toBeInTheDocument();
  });
});
