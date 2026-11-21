import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import { ApiError } from "@/services/api/client";

/**
 * part13 Phase 5's next remainder item — the cancer cell-line
 * drug-response lane of the Machine Learning workspace as a complete
 * component-level user flow:
 *
 *   the registry line      — the stage renders the backend's own
 *                            vocabulary summary (the five-key metric
 *                            contract, the sensitivity-ranking schema +
 *                            ordering, the cross-cell-line fold rule)
 *                            off the mocked registry answer;
 *   the analyze round trip — the default parameter block becomes the
 *                            honest request, and the answer renders the
 *                            REAL parsed payload (the screen summary,
 *                            the fit telemetry, the metric row, the
 *                            per-cell ranking tables, the per-drug
 *                            biomarker weight tables, the CV summary);
 *   the honest labels      — a null pearson and a null CV mean read as
 *                            null, never zero, and the model-estimate
 *                            column header labels every score;
 *   typed errors           — the backend's own hard 422 surfaces
 *                            verbatim through the banner;
 *   client-side gating     — a below-bound screen, an over-cap screen
 *                            and a folds-exceed-cells request never
 *                            leave the page (validated BEFORE any
 *                            request fires);
 *   honest provenance      — the disclaimer + the "0 shipped
 *                            checkpoints" line ride every answer.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers
 * (parseDrugResponseRegistryResponse /
 * parseDrugResponseAnalyzeResponse — the REAL ones, spread from the
 * actual module), and the DOM are the real ones.
 */

const datasetsMocks = vi.hoisted(() => ({ list: vi.fn() }));
vi.mock("@/services/datasets/client", () => ({
  datasetsApi: { list: datasetsMocks.list },
}));

const mlMocks = vi.hoisted(() => ({
  trained: vi.fn(),
  catalog: vi.fn(),
  admetEndpoints: vi.fn(),
  admetScore: vi.fn(),
  dtaRegistry: vi.fn(),
  dtaFeatures: vi.fn(),
  repurposingRegistry: vi.fn(),
  repurposingScreen: vi.fn(),
  drugResponseRegistry: vi.fn(),
  drugResponseAnalyze: vi.fn(),
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
      catalog: mlMocks.catalog,
      admetEndpoints: mlMocks.admetEndpoints,
      admetScore: mlMocks.admetScore,
      dtaRegistry: mlMocks.dtaRegistry,
      dtaFeatures: mlMocks.dtaFeatures,
      repurposingRegistry: mlMocks.repurposingRegistry,
      repurposingScreen: mlMocks.repurposingScreen,
      drugResponseRegistry: mlMocks.drugResponseRegistry,
      drugResponseAnalyze: mlMocks.drugResponseAnalyze,
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

const DR_PROVENANCE = {
  backend: "pharmaml.ml.drugresponse",
  source:
    "CSOgroup/CaDRReS + SmritiChawla/Precily + " +
    "burakcan-izmirli/DeepResponse",
  n_repositories: 3,
  shipped_checkpoints: 0,
  shipped_checkpoints_note:
    "no pretrained checkpoints ship — DeepCDR's multi-omics CNN and " +
    "the external VAE weights stay external",
  screen: "SYNTHETIC — the stand-in for a CCLE/GDSC-sized table",
  disclaimer:
    "predicted response scores are model estimates over the synthetic " +
    "screen — sensitivity rankings and biomarker weights are triage " +
    "hypotheses, never clinical recommendations",
};

const DR_REGISTRY = {
  model: {
    family: "CaDRReS-style bilinear factorisation with a sigmoid link",
    formulation: "gamma = X.W; the score is the sigmoid of the bilinear form",
    direction:
      "the score is the sigmoid-linked response — higher = stronger " +
      "predicted response; the sensitivity ranking is most-responsive " +
      "first: highest predicted score = rank 1",
    telemetry_fields: [
      "train_losses",
      "final_loss",
      "n_observed",
      "pearson_r",
      "n_epochs",
    ],
    history_note: "the trainer has no live per-epoch callback",
  },
  hyperparameters: {
    n_factors: { default: 3, max: 10 },
    lr: { default: 0.1, min: 0.001, max: 0.5 },
    epochs: { default: 200, max: 1000 },
    l2: { default: 0.001, max: 1.0 },
    seed: { default: 42, max: 2147483647 },
    note: "the backend's own defaults are n_factors=10 / lr=0.1",
  },
  screen: {
    defaults: {
      n_cells: 12,
      n_drugs: 4,
      n_features: 8,
      noise: 0.05,
      observed: 0.85,
      seed: 42,
    },
    bounds: {
      n_cells: "4..40",
      n_drugs: "2..30",
      n_features: "2..20",
      observed: "0.5..1.0",
    },
    label: "SYNTHETIC — a seeded response screen",
    scale: "one response scale only — never mixed",
  },
  sensitivity_ranking: {
    schema: ["cell_line", "drug", "predicted_score", "rank"],
    ordering: "most-responsive first — highest predicted score = rank 1",
    source: "the model's whole drug axis scored per cell profile",
    response_limits: { profiles: "1..8", top_k_per_profile: "1..10" },
  },
  biomarkers: {
    schema: ["feature", "weight"],
    ordering: "sorted by |weight| descending then feature name ascending",
    source: "DrugResponseModel.explain_drug — the composed W w_d readout",
    response_limits: { top_k_per_drug: "1..10" },
  },
  cross_validation: {
    n_folds: "2..5",
    rule:
      "folds split cell lines, never (cell line, drug) pairs — " +
      "n_folds must not exceed the cell count (the backend's own 422)",
    summary: "the across-fold means of the metric contract",
  },
  metrics: {
    keys: ["pearson", "spearman", "rmse", "mae", "n"],
    note: "the four numbers every response paper reports, plus n",
  },
  provenance: DR_PROVENANCE,
  analyze_url: "/api/v1/drugresponse/analyze",
  note: "sync-only light compute — dense bounded matrices",
};

function rankRow(drug: string, predicted: number, rank: number) {
  return { drug, predicted_score: predicted, rank };
}

function biomarkerRow(feature: string, weight: number) {
  return { feature, weight };
}

function analyzeResponse(overrides: Record<string, unknown> = {}) {
  return {
    request: {
      n_cells: 12,
      n_drugs: 4,
      n_features: 8,
      n_factors: 3,
      lr: 0.1,
      epochs: 200,
      l2: 0.001,
      noise: 0.05,
      observed: 0.85,
      seed: 42,
      top_k: 5,
      profiles: 3,
      n_folds: 3,
    },
    screen: {
      cells: ["CELL_00", "CELL_01", "CELL_02"],
      drugs: ["DRUG_00", "DRUG_01", "DRUG_02", "DRUG_03"],
      features: [
        "gene_00", "gene_01", "gene_02", "gene_03",
        "gene_04", "gene_05", "gene_06", "gene_07",
      ],
      n_cells: 12,
      n_drugs: 4,
      n_features: 8,
      n_observed: 40,
      n_entries: 48,
      observed_fraction: 0.833,
      label: "SYNTHETIC seeded screen — not real pharmacology",
    },
    training: {
      epochs_run: 200,
      max_epochs: 200,
      final_loss: 0.045,
      first_loss: 0.121,
      n_observed: 40,
      pearson_r: 0.93,
      n_factors: 3,
      note: "never a fabricated epoch stream",
    },
    metrics: {
      pearson: 0.93,
      spearman: 0.9,
      rmse: 0.087,
      mae: 0.069,
      n: 40,
    },
    ranking: {
      n_profiles: 3,
      n_ranked_rows_total: 12,
      top_k: 5,
      per_cell: {
        CELL_00: [
          rankRow("DRUG_00", 0.912, 1),
          rankRow("DRUG_01", 0.844, 2),
          rankRow("DRUG_02", 0.701, 3),
          rankRow("DRUG_03", 0.355, 4),
        ],
        CELL_01: [
          rankRow("DRUG_01", 0.887, 1),
          rankRow("DRUG_00", 0.796, 2),
          rankRow("DRUG_03", 0.412, 3),
          rankRow("DRUG_02", 0.389, 4),
        ],
        CELL_02: [
          rankRow("DRUG_02", 0.901, 1),
          rankRow("DRUG_01", 0.733, 2),
          rankRow("DRUG_00", 0.552, 3),
          rankRow("DRUG_03", 0.301, 4),
        ],
      },
      ordering: "highest predicted response score = rank 1",
    },
    biomarkers: {
      n_drugs: 4,
      top_k: 5,
      per_drug: {
        DRUG_00: [
          biomarkerRow("gene_03", 0.412),
          biomarkerRow("gene_07", -0.365),
          biomarkerRow("gene_00", 0.211),
          biomarkerRow("gene_05", 0.184),
          biomarkerRow("gene_01", -0.152),
        ],
        DRUG_01: [
          biomarkerRow("gene_07", 0.388),
          biomarkerRow("gene_02", 0.341),
          biomarkerRow("gene_04", -0.275),
          biomarkerRow("gene_01", 0.198),
          biomarkerRow("gene_06", 0.161),
        ],
        DRUG_02: [
          biomarkerRow("gene_02", 0.355),
          biomarkerRow("gene_05", 0.298),
          biomarkerRow("gene_00", -0.241),
          biomarkerRow("gene_03", 0.187),
          biomarkerRow("gene_07", 0.154),
        ],
        DRUG_03: [
          biomarkerRow("gene_01", 0.322),
          biomarkerRow("gene_06", -0.287),
          biomarkerRow("gene_04", 0.226),
          biomarkerRow("gene_02", 0.173),
          biomarkerRow("gene_05", 0.149),
        ],
      },
      ordering: "sorted by |weight| descending then feature name",
    },
    cross_validation: {
      n_folds: 3,
      folds: [
        {
          fold: 1,
          n_test_cells: 4,
          n_observed: 13,
          metrics: { pearson: 0.89, spearman: 0.87, rmse: 0.108, mae: 0.088, n: 13 },
        },
        {
          fold: 2,
          n_test_cells: 4,
          n_observed: 14,
          metrics: { pearson: 0.86, spearman: 0.84, rmse: 0.121, mae: 0.096, n: 14 },
        },
        {
          fold: 3,
          n_test_cells: 4,
          n_observed: 13,
          metrics: { pearson: 0.9, spearman: 0.88, rmse: 0.107, mae: 0.086, n: 13 },
        },
      ],
      summary: { pearson: 0.883, spearman: 0.863, rmse: 0.112, mae: 0.09, n: 13.3 },
      note: "folds split cell lines, never (cell line, drug) pairs",
    },
    provenance: DR_PROVENANCE,
    note: "triage hypotheses, never clinical recommendations",
    ...overrides,
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
  // structurally complete DTA registry stub — the DTA stage renders the
  // registry line off these fields, so the lane under test never crashes
  // its neighbour while this suite focuses on the drug-response stage
  mlMocks.dtaRegistry.mockResolvedValue({
    alphabets: {
      protein: {
        n_symbols: 25,
        padding_code: 0,
        symbols: { A: 1 },
        encoding: "1-based labels",
        source: "DeepDTA CHARPROTSET verbatim",
      },
      smiles_isomeric: {
        n_symbols: 64,
        padding_code: 0,
        symbols: { C: 42 },
        encoding: "encoded as-is",
        source: "DeepDTA CHARISOSMISET verbatim",
      },
      smiles_canonical: {
        n_symbols: 62,
        padding_code: 0,
        symbols: { "#": 1 },
        encoding: "the 62-symbol canonical set",
        source: "DeepDTA CHARCANSMISET verbatim",
      },
    },
    amino_acids: Array.from("ACDEFGHIKLMNPQRSTVWY"),
    protein_features: {
      n_base: 23,
      base_columns: [],
      n_dipeptide_extras: 401,
      n_with_dipeptides: 424,
      note: "",
    },
    ligand_features: { family: "ecfp", n_bits: 2048, source: "" },
    paired_width: { base: 2071, with_dipeptides: 2472 },
    metrics: [],
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
  // the repurposing registry stub — the same neighbour contract
  mlMocks.repurposingRegistry.mockResolvedValue({
    split_strategies: {
      within_table: ["random", "cell_stratified"],
      cross_domain: "cross_domain_split",
      source: "DeepResponse's five split strategies, verbatim",
    },
    model: {
      family: "caDRReS-style matrix factorisation",
      formulation: "S = mu + b_p + b_q + P.Q^T",
      direction: "the matrix reads IC50 (lower = more sensitive)",
      telemetry_fields: ["history", "observed_rmse", "stopped_early"],
      history_note: "the trainer has no live per-epoch callback",
    },
    hyperparameters: {
      n_factors: { default: 3, max: 10 },
      lr: { default: 0.05, min: 0.001, max: 0.5 },
      max_epochs: { default: 200, max: 1000 },
      seed: { default: 42, max: 2147483647 },
      note: "",
    },
    screen: {
      defaults: { n_cells: 12, n_drugs: 8, n_features: 6 },
      bounds: { n_cells: "4..40" },
      label: "SYNTHETIC — a seeded IC50 screen",
    },
    repurposing: {
      schema: ["cell_line", "drug", "predicted_ic50", "rank"],
      ordering: "most-sensitive first",
      source: "",
      response_limits: { profiles: "1..8", top_k_per_profile: "1..10" },
    },
    metrics: {
      keys: ["rmse", "mae", "r2"],
      threshold: 6.0,
      note: "",
    },
    provenance: {
      backend: "pharmaml.ml.response_model",
      source: "",
      n_repositories: 3,
      shipped_checkpoints: 0,
      shipped_checkpoints_note: "",
      screen: "",
      disclaimer: "",
    },
    screen_url: "/api/v1/repurposing/screen",
    note: "",
  });
  mlMocks.drugResponseRegistry.mockResolvedValue(DR_REGISTRY);
  mlMocks.drugResponseAnalyze.mockReset();
});

describe("the cancer cell-line drug-response lane of the Machine Learning workspace", () => {
  it("renders the stage with the registry vocabulary line", async () => {
    render(<MLPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Cancer Cell-Line Drug Response"),
      ).toBeInTheDocument();
    });
    // the backend's own registry summary, never re-declared
    await waitFor(() => {
      expect(
        screen.getByText(/The metric contract: 5 keys/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/most-responsive first — highest predicted score = rank 1/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/folds split cell lines, never \(cell line, drug\) pairs/),
    ).toBeInTheDocument();
    expect(mlMocks.drugResponseRegistry).toHaveBeenCalledTimes(1);
  });

  it("runs the analyze round trip and renders the ranking + biomarker tables", async () => {
    mlMocks.drugResponseAnalyze.mockResolvedValue(analyzeResponse());
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the analysis",
    });
    await userEvent.click(button);
    // the default parameter block becomes the honest request
    await waitFor(() => {
      expect(mlMocks.drugResponseAnalyze).toHaveBeenCalledWith({
        n_cells: 12,
        n_drugs: 4,
        n_features: 8,
        n_factors: 3,
        lr: 0.1,
        epochs: 200,
        l2: 0.001,
        noise: 0.05,
        observed: 0.85,
        seed: 42,
        top_k: 5,
        profiles: 3,
        n_folds: 3,
      });
    });
    // the summary line over the REAL parsed payload
    expect(
      await screen.findByText(/12×4 screen \(SYNTHETIC\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/40\/48 observed entries \(83%\)/),
    ).toBeInTheDocument();
    expect(screen.getByText(/200\/200 epochs/)).toBeInTheDocument();
    expect(screen.getByText(/final loss 0\.0450/)).toBeInTheDocument();
    expect(screen.getByText(/pearson r 0\.930/)).toBeInTheDocument();
    expect(screen.getByText(/rmse 0\.087/)).toBeInTheDocument();
    expect(
      screen.getByText(/cross-cell-line CV: 3 folds, mean rmse 0\.112/),
    ).toBeInTheDocument();
    // the per-cell ranking tables: the profile header carries the
    // ordering rule, and the rank-1 row renders the most-responsive
    // drug with its labelled model estimate
    expect(screen.getByText(/CELL_00 — top 4 of 4/)).toBeInTheDocument();
    expect(screen.getByText(/CELL_02 — top 4 of 4/)).toBeInTheDocument();
    expect(screen.getAllByText("DRUG_00").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("0.912").length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText(/predicted score \(model estimate\)/).length,
    ).toBeGreaterThanOrEqual(3);
    // the per-drug biomarker tables: the composed W w_d readout with
    // its logit-contribution weights
    expect(
      screen.getAllByText(/biomarker weights \(the composed W w_d readout\)/)
        .length,
    ).toBeGreaterThanOrEqual(4);
    expect(
      screen.getAllByText(/weight \(logit contribution\)/).length,
    ).toBeGreaterThanOrEqual(4);
    expect(screen.getAllByText("gene_03").length).toBeGreaterThanOrEqual(1);
  });

  it("labels the estimates and renders null telemetry honestly", async () => {
    mlMocks.drugResponseAnalyze.mockResolvedValue(
      analyzeResponse({
        training: {
          epochs_run: 200,
          max_epochs: 200,
          final_loss: 0.062,
          first_loss: 0.14,
          n_observed: 40,
          pearson_r: null,
          n_factors: 3,
          note: "never a fabricated epoch stream",
        },
        metrics: {
          pearson: null,
          spearman: null,
          rmse: 0.623,
          mae: 0.5,
          n: 40,
        },
        cross_validation: {
          n_folds: 3,
          folds: [],
          summary: { pearson: null, spearman: null, rmse: null, mae: null, n: 13.3 },
          note: "folds split cell lines",
        },
      }),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the analysis",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(screen.getByText(/pearson r null/)).toBeInTheDocument();
    });
    // a null metric reads as null, never zero — the fit line and the
    // CV summary both carry the honest absence
    expect(screen.getByText(/rmse 0\.623/)).toBeInTheDocument();
    expect(screen.getByText(/mean rmse null/)).toBeInTheDocument();
    // the model-estimate column header labels every predicted score
    expect(
      screen.getAllByText(/predicted score \(model estimate\)/).length,
    ).toBeGreaterThanOrEqual(3);
  });

  it("surfaces the backend's own hard 422 verbatim through the banner", async () => {
    mlMocks.drugResponseAnalyze.mockRejectedValue(
      new ApiError(422, {
        detail: "n_folds must not exceed the number of cell lines (4), got 5",
      }),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the analysis",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(screen.getByText(/API 422/)).toBeInTheDocument();
    });
    expect(
      screen.getByText(/n_folds must not exceed the number of cell lines/),
    ).toBeInTheDocument();
    expect(mlMocks.drugResponseAnalyze).toHaveBeenCalledTimes(1);
  });

  it("gates a below-bound screen client-side — no request fires", async () => {
    render(<MLPage />);
    const cells = await screen.findByLabelText("drugresponse n_cells");
    await userEvent.clear(cells);
    await userEvent.type(cells, "3");
    const button = screen.getByRole("button", { name: "Run the analysis" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/needs at least 4 cell lines/),
    ).toBeInTheDocument();
    expect(mlMocks.drugResponseAnalyze).not.toHaveBeenCalled();
  });

  it("gates an over-cap screen client-side — no request fires", async () => {
    render(<MLPage />);
    const cells = await screen.findByLabelText("drugresponse n_cells");
    await userEvent.clear(cells);
    await userEvent.type(cells, "41");
    const button = screen.getByRole("button", { name: "Run the analysis" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/caps the screen at 40 cells/),
    ).toBeInTheDocument();
    expect(mlMocks.drugResponseAnalyze).not.toHaveBeenCalled();
  });

  it("gates a folds-exceed-cells request client-side — no request fires", async () => {
    render(<MLPage />);
    const cells = await screen.findByLabelText("drugresponse n_cells");
    await userEvent.clear(cells);
    await userEvent.type(cells, "4");
    const folds = screen.getByLabelText("drugresponse n_folds");
    await userEvent.clear(folds);
    await userEvent.type(folds, "5");
    const button = screen.getByRole("button", { name: "Run the analysis" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/n_folds may not exceed the cell count/),
    ).toBeInTheDocument();
    expect(mlMocks.drugResponseAnalyze).not.toHaveBeenCalled();
  });

  it("carries the honest provenance and disclaimer on every answer", async () => {
    mlMocks.drugResponseAnalyze.mockResolvedValue(analyzeResponse());
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the analysis",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(
        screen.getByText(/0 shipped checkpoints \(the honest absence/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/never clinical recommendations/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/SYNTHETIC — the stand-in for a CCLE\/GDSC-sized table/),
    ).toBeInTheDocument();
  });
});
