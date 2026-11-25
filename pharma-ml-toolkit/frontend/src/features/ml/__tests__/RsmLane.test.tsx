import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import { ApiError } from "@/services/api/client";

/**
 * part17 §6 (the R1 lane) — the Experimental Design & Response Surface
 * lane of the Machine Learning workspace as a complete component-level
 * user flow:
 *
 *   the registry line      — the stage renders the backend's own
 *                            vocabulary (the three design generators,
 *                            the desirability goals, the demo
 *                            experiment descriptor) off the mocked
 *                            registry answer, and the factor editor
 *                            pre-fills from the demo's declared
 *                            factors;
 *   the design round trip  — the factor editor + the design selects
 *                            become the honest request, and the
 *                            generated run table renders (the coded
 *                            AND actual levels, the run-type census,
 *                            the honest assumptions — the rotatable
 *                            CCD's axial disclosure included);
 *   the fit round trip     — the demo experiment becomes the honest
 *                            request (no dataset selected), and the
 *                            answer renders the REAL parsed payload
 *                            (the per-response coefficient table with
 *                            estimate / std error / t / p, the R2 /
 *                            adj / pred / RMSE statistics line, the
 *                            machine-precision diagnostics note, the
 *                            SYNTHETIC provenance line, the models
 *                            comparison table);
 *   the prediction lane    — the JSON points become the honest
 *                            request, and the per-response predictions
 *                            render with the PI bounds and the honest
 *                            zone vocabulary (the extrapolation row
 *                            flagged, never silent);
 *   the contour lane      — the pair selects + the response select
 *                            become the honest request, and the
 *                            Plotly contour lane renders off the
 *                            grid payload;
 *   the optimisation lane  — the goal editor's defaults become the
 *                            honest request, and the candidate block
 *                            renders (the combined desirability, the
 *                            per-response predictions + individual
 *                            desirabilities, the sensitivity line, the
 *                            Pareto alternatives table);
 *   typed errors           — the backend's own hard 422 surfaces
 *                            verbatim through the banner;
 *   client-side gating     — an invalid factor row never leaves the
 *                            page (validated BEFORE any request fires).
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers (parseRsmRegistryResponse /
 * parseRsmDesignResponse / parseRsmFitResponse / parseRsmPredictResponse
 * / parseRsmContourResponse / parseRsmOptimizeResponse — the REAL ones,
 * spread from the actual module), and the DOM are the real ones.
 */

const datasetsMocks = vi.hoisted(() => ({ list: vi.fn() }));
vi.mock("@/services/datasets/client", () => ({
  datasetsApi: { list: datasetsMocks.list },
}));

const mlMocks = vi.hoisted(() => ({
  trained: vi.fn(),
  admetEndpoints: vi.fn(),
  rsmRegistry: vi.fn(),
  rsmDesign: vi.fn(),
  rsmFit: vi.fn(),
  rsmPredict: vi.fn(),
  rsmContour: vi.fn(),
  rsmOptimize: vi.fn(),
  rsmModels: vi.fn(),
  rsmExport: vi.fn(),
}));
vi.mock("@/services/ml/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/ml/client")>();
  // the REAL parsers ride the page under test; only the network
  // boundary is faked
  return {
    ...actual,
    mlApi: {
      ...actual.mlApi,
      trained: mlMocks.trained,
      admetEndpoints: mlMocks.admetEndpoints,
      rsmRegistry: mlMocks.rsmRegistry,
      rsmDesign: mlMocks.rsmDesign,
      rsmFit: mlMocks.rsmFit,
      rsmPredict: mlMocks.rsmPredict,
      rsmContour: mlMocks.rsmContour,
      rsmOptimize: mlMocks.rsmOptimize,
      rsmModels: mlMocks.rsmModels,
      rsmExport: mlMocks.rsmExport,
    },
  };
});

vi.mock("@/services/jobs/client", () => ({
  jobsApi: { result: vi.fn() },
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

const DEMO_FACTORS = [
  { name: "temperature_c", low: 50.0, high: 90.0, unit: "degC" },
  { name: "ph", low: 5.0, high: 9.0, unit: "pH" },
  { name: "substrate_mM", low: 10.0, high: 50.0, unit: "mM" },
];

const RSM_REGISTRY = {
  designs: [
    {
      key: "ccd",
      label: "Central Composite Design",
      alpha_types: ["rotatable", "faced", "custom"],
      min_factors: 2,
      max_factors: 10,
      runs_formula: "2^k + 2k + centers (x replicates)",
      blocking: true,
      note: "the classic CCD: the axial runs sit OUTSIDE the declared ranges",
    },
    {
      key: "box_behnken",
      label: "Box-Behnken Design",
      alpha_types: null,
      min_factors: 3,
      max_factors: 10,
      runs_formula: "4 * k(k-1)/2 + centers (x replicates)",
      blocking: false,
      note: "all levels inside [-1, +1]",
    },
    {
      key: "full_factorial",
      label: "Full 2^k Factorial",
      alpha_types: null,
      min_factors: 2,
      max_factors: 8,
      runs_formula: "2^k + centers (x replicates)",
      blocking: false,
      note: "the first-order screen",
    },
  ],
  goals: ["maximize", "minimize", "target", "range"],
  model: {
    order: "second-order polynomial in coded factors",
    terms: "1 intercept + k linear + k(k-1)/2 interaction + k quadratic",
    term_count_formula: "1 + 2k + k(k-1)/2",
  },
  bounds: {
    min_factors: 2,
    max_factors: 10,
    max_design_runs: 512,
    max_responses: 8,
    grid_points: [5, 60],
    confidence: "(0.5, 1.0)",
    max_constraints: 16,
    seed: [0, 2147483647],
  },
  demo_dataset: {
    name: "synthetic-rsm-demo",
    synthetic: true,
    factors: DEMO_FACTORS.map((f) => ({
      ...f,
      center: (f.low + f.high) / 2,
      half_range: (f.high - f.low) / 2,
    })),
    responses: ["yield_pct", "purity_pct"],
    runs: 17,
    design: "ccd rotatable (2^3 + 2*3 axial + 3 centers, un-randomised)",
    noise: false,
    note: "the KNOWN noise-free quadratic surfaces",
  },
  scope_statement:
    "RSM is a designed-experiment modelling discipline — NOT the generic multi-response ML",
};

function designResponse() {
  return {
    factors: RSM_REGISTRY.demo_dataset.factors,
    design: {
      type: "ccd",
      alpha: 1.681792830507429,
      alpha_type: "rotatable",
      n_factors: 3,
      runs: 17,
      center_points: 3,
      replicates: 1,
      randomized: false,
      seed: 42,
      blocks: false,
    },
    runs: [
      {
        run_id: "r001",
        run_order: 1,
        block: 1,
        run_type: "factorial",
        coded: { temperature_c: -1, ph: -1, substrate_mM: -1 },
        actual: { temperature_c: 50, ph: 5, substrate_mM: 10 },
      },
      {
        run_id: "r004",
        run_order: 2,
        block: 1,
        run_type: "axial",
        coded: { temperature_c: 1.681793, ph: 0, substrate_mM: 0 },
        actual: { temperature_c: 103.63586, ph: 7, substrate_mM: 30 },
      },
      {
        run_id: "r015",
        run_order: 3,
        block: 1,
        run_type: "center",
        coded: { temperature_c: 0, ph: 0, substrate_mM: 0 },
        actual: { temperature_c: 70, ph: 7, substrate_mM: 30 },
      },
    ],
    run_type_counts: { factorial: 8, axial: 6, center: 3 },
    assumptions: [
      "the rotatable alpha (1.682 = (2*3)**(1/4)) places the axial runs OUTSIDE the declared factor ranges",
      "the coded levels are exact transforms of the declared ranges",
    ],
    export_note:
      "the design table exports through the EXISTING generic table lane",
  };
}

const COEF_ROWS = (response: string) => [
  {
    term: "intercept",
    label: "Intercept",
    kind: "intercept",
    estimate: response === "yield_pct" ? 78.0 : 92.0,
    std_error: 0.01,
    t: 7800.0,
    p: 1e-20,
    ci_low: 77.98,
    ci_high: 78.02,
  },
  {
    term: "x1",
    label: "temperature_c (linear)",
    kind: "linear",
    estimate: response === "yield_pct" ? 4.0 : -3.0,
    std_error: 0.01,
    t: 400.0,
    p: 1e-16,
    ci_low: 3.98,
    ci_high: 4.02,
  },
  {
    term: "x1_sq",
    label: "temperature_c^2",
    kind: "quadratic",
    estimate: response === "yield_pct" ? -4.0 : -2.0,
    std_error: 0.01,
    t: -400.0,
    p: 1e-16,
    ci_low: -4.02,
    ci_high: -3.98,
  },
];

const FIT_STATS = {
  n_runs: 17,
  df_residual: 7,
  rank: 10,
  r2: 1.0,
  adj_r2: 1.0,
  pred_r2: 1.0,
  rmse: 1e-13,
  press: 1e-24,
  confidence: 0.95,
  outliers_flagged: 0,
  influential_flagged: 0,
  cooks_threshold: 0.235,
  machine_precision_fit: true,
  diagnostics_note:
    "the fit is at machine precision (rmse at the float noise floor)",
  thresholds_note:
    "outlier flag: |internal studentised residual| > 3; influential flag: Cook's distance > max(1, 4/n)",
};

function fitResponse() {
  return {
    model: {
      model_id: "rsm-test000001",
      name: "rsm-yield_pct-purity_pct",
      task: "rsm-fit",
      dataset_id: null,
      target: "yield_pct|purity_pct",
      features: ["temperature_c", "ph", "substrate_mM"],
      n_train_rows: 17,
      best_params: {},
      cv_scores: null,
      cv_mean: null,
      cv_std: null,
      fit_seconds: 0.05,
      created_at: "2026-01-01T00:00:00Z",
      rsm: {
        kind: "rsm",
        factors: ["temperature_c", "ph", "substrate_mM"],
        responses: ["yield_pct", "purity_pct"],
        n_terms: 10,
        design_meta: { type: "ccd", alpha: 1.681793, alpha_type: "rotatable" },
        df_residual: 7,
        rank: 10,
      },
    },
    run_id: "run-test000001",
    provenance: {
      source: "demo",
      dataset_id: null,
      dataset_name: "synthetic-rsm-demo",
      synthetic: true,
      note: "the seeded SYNTHETIC demonstration experiment",
    },
    configuration: {
      factors: RSM_REGISTRY.demo_dataset.factors,
      response_columns: ["yield_pct", "purity_pct"],
      design_meta: { type: "ccd", alpha: 1.681793, alpha_type: "rotatable" },
      confidence: 0.95,
    },
    data: {
      n_runs: 17,
      dropped_rows: 0,
      terms: [
        { key: "intercept", kind: "intercept", label: "Intercept" },
        { key: "x1", kind: "linear", label: "temperature_c (linear)" },
        { key: "x1_sq", kind: "quadratic", label: "temperature_c^2" },
      ],
      hierarchy_notes: [],
      aliasing: null,
    },
    coefficients: {
      yield_pct: COEF_ROWS("yield_pct"),
      purity_pct: COEF_ROWS("purity_pct"),
    },
    anova: {
      yield_pct: {
        regression: { ss: 1200.0, df: 9, ms: 133.3, f: 1e12, p: 1e-20 },
        residual: { ss: 0.0, df: 7, ms: 0.0 },
        total: { ss: 1200.0, df: 16 },
        pure_error: { ss: 0.0, df: 2, ms: 0.0 },
        lack_of_fit: {
          ss: 0.0,
          df: 5,
          ms: 0.0,
          f: null,
          p: null,
          note: "the pure-error variance is exactly zero — UNDEFINED",
        },
        terms: [],
      },
      purity_pct: {},
    },
    diagnostics: {
      yield_pct: [
        {
          run: 1,
          observed: 78.0,
          fitted: 78.0,
          residual: 0.0,
          studentized: 0.0,
          leverage: 0.2,
          cooks_distance: 0.0,
          outlier_flag: false,
          influential_flag: false,
        },
      ],
      purity_pct: [],
    },
    fit_statistics: {
      yield_pct: FIT_STATS,
      purity_pct: FIT_STATS,
    },
    fit_seconds: 0.05,
  };
}

function predictResponse() {
  return {
    model_id: "rsm-test000001",
    rows: [
      {
        run: 1,
        zone: "inside_ranges",
        coded: { temperature_c: 0, ph: 0, substrate_mM: 0 },
        actual: { temperature_c: 70, ph: 7, substrate_mM: 30 },
        yield_pct: {
          fitted: 78.0,
          ci_low: 77.9,
          ci_high: 78.1,
          pi_low: 77.8,
          pi_high: 78.2,
        },
        purity_pct: {
          fitted: 92.0,
          ci_low: 91.9,
          ci_high: 92.1,
          pi_low: 91.8,
          pi_high: 92.2,
        },
      },
      {
        run: 2,
        zone: "outside_design",
        coded: { temperature_c: 2.5, ph: 0, substrate_mM: 0 },
        actual: { temperature_c: 120, ph: 7, substrate_mM: 30 },
        yield_pct: {
          fitted: 70.0,
          ci_low: 69.9,
          ci_high: 70.1,
          pi_low: 69.8,
          pi_high: 70.2,
        },
        purity_pct: {
          fitted: 84.0,
          ci_low: 83.9,
          ci_high: 84.1,
          pi_low: 83.8,
          pi_high: 84.2,
        },
      },
    ],
    responses: ["yield_pct", "purity_pct"],
    confidence: 0.95,
    df_residual: 7,
    extrapolation: {
      count: 1,
      runs: [2],
      vocabulary: {
        inside_ranges: "inside the declared factor ranges",
        axial_region: "inside the CCD's axial region",
        outside_design: "outside the design region entirely",
      },
    },
    note: "the zone vocabulary flags extrapolated points",
  };
}

function contourResponse() {
  const grid = [-1, 0, 1];
  return {
    model_id: "rsm-test000001",
    factor_x: {
      name: "temperature_c",
      unit: "degC",
      coded: grid,
      actual: [50, 70, 90],
    },
    factor_y: { name: "ph", unit: "pH", coded: grid, actual: [5, 7, 9] },
    fixed: { substrate_mM: { actual: 30, coded: 0 } },
    grid_points: 3,
    z_name: "yield_pct",
    z: [[60.0, 70.0, 78.0], [70.0, 78.0, 79.5], [78.0, 79.5, 76.0]],
    note: "a MODEL surface, not measured data",
    units_note: "the z values ride the response's own units",
  };
}

function optimizeResponsePayload() {
  return {
    model_id: "rsm-test000001",
    candidate: {
      x_coded: { temperature_c: 0.25, ph: 0.4, substrate_mM: 0.1 },
      x_actual: { temperature_c: 75, ph: 7.8, substrate_mM: 32 },
      combined_desirability: 0.6789,
      responses: {
        yield_pct: {
          goal: "maximize",
          prediction: 79.2,
          individual_desirability: 0.85,
          weight: 1,
          exponent: 1,
          bounds: { low: 68.0, high: 79.5 },
          bounds_source: "observed",
        },
        purity_pct: {
          goal: "minimize",
          prediction: 86.5,
          individual_desirability: 0.6,
          weight: 1,
          exponent: 1,
          bounds: { low: 84.1, high: 92.0 },
          bounds_source: "observed",
        },
      },
    },
    weights: { yield_pct: 1, purity_pct: 1 },
    total_weight: 2,
    constraints: [],
    sensitivity: [
      {
        factor: "temperature_c",
        step_coded: 0.05,
        low: { coded: 0.2, combined_desirability: 0.6712, delta: -0.0077 },
        high: { coded: 0.3, combined_desirability: 0.6811, delta: 0.0022 },
      },
      {
        factor: "ph",
        step_coded: 0.05,
        low: { coded: 0.35, combined_desirability: 0.674, delta: -0.0049 },
        high: { coded: 0.45, combined_desirability: 0.681, delta: 0.0021 },
      },
      {
        factor: "substrate_mM",
        step_coded: 0.05,
        low: { coded: 0.05, combined_desirability: 0.6788, delta: -0.0001 },
        high: { coded: 0.15, combined_desirability: 0.679, delta: 0.0001 },
      },
    ],
    sensitivity_note: "each factor stepped +-0.05 in CODED units",
    pareto_alternatives: [
      {
        kind: "individual",
        response: "yield_pct",
        goal: "maximize",
        combined_desirability: null,
        note: "the single-response optimum for yield_pct alone",
        x_coded: { temperature_c: 0.3, ph: 0.5, substrate_mM: 0.1 },
        x_actual: { temperature_c: 76, ph: 8, substrate_mM: 32 },
        predictions: { yield_pct: 79.5, purity_pct: 86.8 },
      },
      {
        kind: "weight_tilted",
        response: "purity_pct",
        goal: "minimize",
        combined_desirability: 0.72,
        note: "the combined optimum with purity_pct's weight x4",
        x_coded: { temperature_c: 0.2, ph: 0.4, substrate_mM: 0.1 },
        x_actual: { temperature_c: 74, ph: 7.8, substrate_mM: 32 },
        predictions: { yield_pct: 78.9, purity_pct: 85.5 },
      },
    ],
    pareto_note:
      "the curated deterministic alternatives — a SAMPLE, NOT the exhaustive Pareto front",
    feasible_grid: null,
    region_note:
      "the search is confined to the coded box [-1, +1]^k so it CANNOT extrapolate",
    bounds_source_note:
      "desirability bounds the request did not set fall back to the OBSERVED design min/max",
    goals_vocabulary: ["maximize", "minimize", "target", "range"],
  };
}

beforeEach(() => {
  datasetsMocks.list.mockResolvedValue({ datasets: [] });
  mlMocks.trained.mockResolvedValue({ models: [] });
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
  mlMocks.rsmRegistry.mockResolvedValue(RSM_REGISTRY);
  mlMocks.rsmModels.mockResolvedValue({ models: [] });
  vi.clearAllMocks();
  // re-apply the standing stubs after clearAllMocks
  datasetsMocks.list.mockResolvedValue({ datasets: [] });
  mlMocks.trained.mockResolvedValue({ models: [] });
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
  mlMocks.rsmRegistry.mockResolvedValue(RSM_REGISTRY);
  mlMocks.rsmModels.mockResolvedValue({ models: [] });
});

describe("the Experimental Design & Response Surface lane of the Machine Learning workspace", () => {
  it("renders the stage with the registry vocabulary and the factor editor", async () => {
    render(<MLPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Experimental Design & Response Surface"),
      ).toBeInTheDocument();
    });
    // the design select carries the backend's own labels
    const designSelect = screen.getByLabelText("rsm design") as HTMLSelectElement;
    expect(designSelect.value).toBe("ccd");
    const labels = Array.from(designSelect.options).map((o) => o.textContent);
    expect(labels).toContain("Central Composite Design");
    expect(labels).toContain("Box-Behnken Design");
    expect(labels).toContain("Full 2^k Factorial");
    // the registry line off the REAL parsed registry
    await waitFor(() => {
      expect(
        screen.getByText(
          /Central Composite Design · Box-Behnken Design · Full 2\^k Factorial/,
        ),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/goals: maximize \/ minimize \/ target \/ range/),
    ).toBeInTheDocument();
    // the factor editor pre-filled from the demo's declared factors
    expect(
      (screen.getByLabelText("rsm factor 1 name") as HTMLInputElement).value,
    ).toBe("temperature_c");
    expect(
      (screen.getByLabelText("rsm factor 1 low") as HTMLInputElement).value,
    ).toBe("50");
    expect(
      (screen.getByLabelText("rsm factor 2 high") as HTMLInputElement).value,
    ).toBe("9");
    expect(mlMocks.rsmRegistry).toHaveBeenCalledTimes(1);
  });

  it("generates a design and renders the run table with the assumptions", async () => {
    mlMocks.rsmDesign.mockResolvedValue(designResponse());
    render(<MLPage />);
    await userEvent.click(await screen.findByRole("button", {
      name: "Generate design",
    }));
    // the demo factors + the defaults become the honest request
    await waitFor(() => {
      expect(mlMocks.rsmDesign).toHaveBeenCalledWith({
        factors: [
          { name: "temperature_c", low: 50, high: 90, unit: "degC" },
          { name: "ph", low: 5, high: 9, unit: "pH" },
          { name: "substrate_mM", low: 10, high: 50, unit: "mM" },
        ],
        design: "ccd",
        alpha_type: "rotatable",
        center_points: 3,
        randomize: true,
        seed: 42,
        blocks: false,
      });
    });
    // the run table renders the coded AND actual levels
    expect(await screen.findByText("r001")).toBeInTheDocument();
    expect(screen.getByText("factorial")).toBeInTheDocument();
    expect(screen.getByText("axial")).toBeInTheDocument();
    expect(
      screen.getByText(/ccd · 17 runs · centers 3 · replicates 1/),
    ).toBeInTheDocument();
    expect(screen.getByText(/factorial: 8 · axial: 6 · center: 3/)).toBeInTheDocument();
    // the honest assumptions — the rotatable CCD's axial disclosure
    expect(
      screen.getByText(/places the axial runs OUTSIDE the declared factor ranges/),
    ).toBeInTheDocument();
    // the export note rides the generated table
    expect(
      screen.getByText(/exports through the EXISTING generic table lane/),
    ).toBeInTheDocument();
  });

  it("runs the demo fit round trip and renders the coefficient surface", async () => {
    const fit = fitResponse();
    mlMocks.rsmFit.mockResolvedValue(fit);
    mlMocks.rsmModels.mockResolvedValue({ models: [fit.model] });
    render(<MLPage />);
    await userEvent.click(await screen.findByRole("button", {
      name: "Fit response surface",
    }));
    // the demo mode becomes the honest request
    await waitFor(() => {
      expect(mlMocks.rsmFit).toHaveBeenCalledWith({
        response_columns: ["yield_pct", "purity_pct"],
      });
    });
    // the per-response statistics line (the REAL parsed payload)
    expect(
      await screen.findByText(/SYNTHETIC demo \(labelled\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/temperature_c · ph · substrate_mM → yield_pct · purity_pct/),
    ).toBeInTheDocument();
    // the coefficient table rows
    expect(
      screen.getAllByText("temperature_c (linear)").length,
    ).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("Intercept").length).toBeGreaterThanOrEqual(2);
    // the machine-precision diagnostics note rides the stats
    expect(
      screen.getAllByText(/the fit is at machine precision/).length,
    ).toBeGreaterThanOrEqual(2);
    // the ANOVA nulls honesty line
    expect(
      screen.getAllByText(/nulls where the data cannot support the test/).length,
    ).toBeGreaterThanOrEqual(2);
    // the models comparison table fills after the fit
    expect(await screen.findByText("rsm-test000001")).toBeInTheDocument();
  });

  it("renders the prediction round trip with the zone vocabulary", async () => {
    const fit = fitResponse();
    mlMocks.rsmFit.mockResolvedValue(fit);
    mlMocks.rsmPredict.mockResolvedValue(predictResponse());
    mlMocks.rsmModels.mockResolvedValue({ models: [fit.model] });
    render(<MLPage />);
    await userEvent.click(await screen.findByRole("button", {
      name: "Fit response surface",
    }));
    await screen.findByText("rsm-test000001");
    // the model id auto-fills from the fit
    const modelInput = screen.getByLabelText("rsm predict model id") as HTMLInputElement;
    expect(modelInput.value).toBe("rsm-test000001");
    const points = screen.getByLabelText("rsm predict points");
    fireEvent.change(points, {
      target: {
        value:
          '[{"temperature_c": 70, "ph": 7, "substrate_mM": 30}, {"temperature_c": 120, "ph": 7, "substrate_mM": 30}]',
      },
    });
    await userEvent.click(
      await screen.findByRole("button", { name: "Predict the surface" }),
    );
    await waitFor(() => {
      expect(mlMocks.rsmPredict).toHaveBeenCalledWith({
        model_id: "rsm-test000001",
        points: [
          { temperature_c: 70, ph: 7, substrate_mM: 30 },
          { temperature_c: 120, ph: 7, substrate_mM: 30 },
        ],
      });
    });
    // the per-response predictions with the PI bounds
    expect(await screen.findByText(/78\.00 \[77\.80, 78\.20\]/)).toBeInTheDocument();
    // the zone vocabulary — the extrapolation row flagged, never silent
    expect(screen.getByText("inside_ranges")).toBeInTheDocument();
    expect(screen.getByText("outside_design")).toBeInTheDocument();
    expect(screen.getByText(/extrapolation flagged on 1 point\(s\) — 2/)).toBeInTheDocument();
  });

  it("renders the contour round trip through the Plotly lane", async () => {
    const fit = fitResponse();
    mlMocks.rsmFit.mockResolvedValue(fit);
    mlMocks.rsmContour.mockResolvedValue(contourResponse());
    mlMocks.rsmModels.mockResolvedValue({ models: [fit.model] });
    render(<MLPage />);
    await userEvent.click(await screen.findByRole("button", {
      name: "Fit response surface",
    }));
    await screen.findByText("rsm-test000001");
    // the pair + response selects (pre-filled from the registry)
    const factorX = screen.getByLabelText("rsm contour factor x") as HTMLSelectElement;
    expect(factorX.value).toBe("temperature_c");
    const factorY = screen.getByLabelText("rsm contour factor y") as HTMLSelectElement;
    expect(factorY.value).toBe("ph");
    const response = screen.getByLabelText("rsm contour response") as HTMLSelectElement;
    expect(response.value).toBe("yield_pct");
    await userEvent.click(
      await screen.findByRole("button", { name: "Explore the surface" }),
    );
    await waitFor(() => {
      expect(mlMocks.rsmContour).toHaveBeenCalledWith({
        model_id: "rsm-test000001",
        factor_x: "temperature_c",
        factor_y: "ph",
        grid_points: 21,
        mode: "response",
        response: "yield_pct",
        goals: null,
      });
    });
    // the Plotly contour lane renders off the grid payload
    expect(await screen.findByTestId("plotly-stub")).toBeInTheDocument();
    // the fixed factor line displays in BOTH units
    expect(screen.getByText(/substrate_mM=30/)).toBeInTheDocument();
    // the honest note rides the contour
    expect(screen.getByText(/a MODEL surface, not measured data/)).toBeInTheDocument();
  });

  it("renders the desirability optimisation round trip", async () => {
    const fit = fitResponse();
    mlMocks.rsmFit.mockResolvedValue(fit);
    mlMocks.rsmOptimize.mockResolvedValue(optimizeResponsePayload());
    mlMocks.rsmModels.mockResolvedValue({ models: [fit.model] });
    render(<MLPage />);
    await userEvent.click(await screen.findByRole("button", {
      name: "Fit response surface",
    }));
    await screen.findByText("rsm-test000001");
    // the goal editor pre-filled one row per response (the registry's
    // defaults: maximize for the first, minimize for the second)
    const goalSelect = screen.getByLabelText("rsm goal for yield_pct") as HTMLSelectElement;
    expect(goalSelect.value).toBe("maximize");
    const goalSelect2 = screen.getByLabelText("rsm goal for purity_pct") as HTMLSelectElement;
    expect(goalSelect2.value).toBe("minimize");
    await userEvent.click(
      await screen.findByRole("button", { name: "Optimise the responses" }),
    );
    // the goal defaults become the honest request (bounds fall back to
    // the observed min/max when left empty)
    await waitFor(() => {
      expect(mlMocks.rsmOptimize).toHaveBeenCalledWith({
        model_id: "rsm-test000001",
        goals: [
          {
            response: "yield_pct",
            goal: "maximize",
            low: null,
            high: null,
            target: null,
            weight: 1,
          },
          {
            response: "purity_pct",
            goal: "minimize",
            low: null,
            high: null,
            target: null,
            weight: 1,
          },
        ],
      });
    });
    // the candidate block renders
    expect(
      await screen.findByText(/combined desirability 0\.678900/),
    ).toBeInTheDocument();
    expect(screen.getByText(/temperature_c=75\.00/)).toBeInTheDocument();
    // the per-response blocks with the individual desirabilities
    expect(screen.getByText(/0\.8500/)).toBeInTheDocument();
    expect(screen.getAllByText(/\(observed\)/).length).toBeGreaterThanOrEqual(2);
    // the sensitivity line + the Pareto alternatives table
    expect(
      screen.getByText(/temperature_c: ΔD -0\.0077 \/ 0\.0022/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/individual · yield_pct/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/weight_tilted · purity_pct/),
    ).toBeInTheDocument();
    // the honest notes ride the answer
    expect(
      screen.getByText(/NOT the exhaustive Pareto front/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/CANNOT extrapolate/),
    ).toBeInTheDocument();
  });

  it("surfaces the backend's own hard 422 verbatim through the banner", async () => {
    mlMocks.rsmFit.mockRejectedValue(
      new ApiError(
        422,
        "the design has fewer runs (8) than model terms (10) -- the second-order model is not estimable; use a bigger design",
      ),
    );
    render(<MLPage />);
    await userEvent.click(await screen.findByRole("button", {
      name: "Fit response surface",
    }));
    expect(
      await screen.findByText(
        /the design has fewer runs \(8\) than model terms \(10\)/,
      ),
    ).toBeInTheDocument();
  });

  it("gates an invalid factor row client-side — no request fires", async () => {
    mlMocks.rsmDesign.mockResolvedValue(designResponse());
    render(<MLPage />);
    // blank the first factor's name (wait for the editor to pre-fill)
    const nameInput = await screen.findByLabelText("rsm factor 1 name");
    await userEvent.clear(nameInput);
    await userEvent.click(
      await screen.findByRole("button", { name: "Generate design" }),
    );
    expect(
      await screen.findByText(
        /the design needs at least 2 valid factors/,
      ),
    ).toBeInTheDocument();
    expect(mlMocks.rsmDesign).not.toHaveBeenCalled();
  });

  it("exports the fitted bundle through the joblib artifact lane", async () => {
    const fit = fitResponse();
    mlMocks.rsmFit.mockResolvedValue(fit);
    mlMocks.rsmModels.mockResolvedValue({ models: [fit.model] });
    mlMocks.rsmExport.mockResolvedValue({
      artifact_id: "art-rsm-1",
      kind: "model",
      filename: "rsm-rsm-test000001.joblib",
      content_type: "application/octet-stream",
      size_bytes: 4096,
      sha256: "ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12",
      created_at: "2026-01-01T00:00:00Z",
      meta: {},
      download_url: "/api/v1/artifacts/art-rsm-1/download",
    });
    render(<MLPage />);
    await userEvent.click(await screen.findByRole("button", {
      name: "Fit response surface",
    }));
    await screen.findByText("rsm-test000001");
    await userEvent.click(
      await screen.findByRole("button", { name: "Export the bundle (joblib)" }),
    );
    await waitFor(() => {
      expect(mlMocks.rsmExport).toHaveBeenCalledWith("rsm-test000001");
    });
    expect(
      await screen.findByText(/rsm-rsm-test000001\.joblib/),
    ).toBeInTheDocument();
    expect(screen.getByText(/4096 B/)).toBeInTheDocument();
  });
});
