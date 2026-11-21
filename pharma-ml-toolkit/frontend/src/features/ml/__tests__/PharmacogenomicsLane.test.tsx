import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import { ApiError } from "@/services/api/client";

/**
 * part13 Phase 5's next remainder item — the pharmacogenomics lane
 * of the Machine Learning workspace as a complete component-level
 * user flow:
 *
 *   the registry line      — the stage renders the backend's own
 *                            vocabulary summary (the feasibility
 *                            gates + the percent convention) off the
 *                            mocked registry answer;
 *   the analyze round trip — the default parameter block becomes the
 *                            honest request, and the answer renders
 *                            the REAL parsed payload (the screen
 *                            summary, the association census, the
 *                            top-association table with the
 *                            p/q/FDR columns, the volcano counts, the
 *                            guarded + normalized ruler scores);
 *   the honest labels      — a null guarded pearson and a null
 *                            normalized score read as null, never
 *                            zero, and the top-table header labels
 *                            every row a statistic over the
 *                            synthetic screen;
 *   typed errors           — the backend's own hard 422 surfaces
 *                            verbatim through the banner;
 *   client-side gating     — a negative seed, an over-range seed and
 *                            an out-of-band top-k request never
 *                            leave the page (validated BEFORE any
 *                            request fires);
 *   honest provenance      — the disclaimer + the "0 shipped tables"
 *                            + "0 shipped checkpoints" lines ride
 *                            every answer.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers
 * (parsePharmacogenomicsRegistryResponse /
 * parsePharmacogenomicsAnalyzeResponse — the REAL ones, spread from
 * the actual module), and the DOM are the real ones.
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
  pharmacogenomicsRegistry: vi.fn(),
  pharmacogenomicsAnalyze: vi.fn(),
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
      pharmacogenomicsRegistry: mlMocks.pharmacogenomicsRegistry,
      pharmacogenomicsAnalyze: mlMocks.pharmacogenomicsAnalyze,
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

const PG_PROVENANCE = {
  backend: "pharmaml.ml.pharmacogenomics",
  source:
    "five repositories reviewed file by file — CancerRxGene/gdsctools " +
    "+ TangYiChing/GDSC_DataParser + daisybio/drevalpy + " +
    "CBIIT/cellminercdb + cBioPortal/cbioportal",
  n_repositories: 5,
  shipped_tables: 0,
  shipped_tables_note:
    "no real GDSC, CCLE, CTRP or DepMap table ships with the toolkit",
  shipped_checkpoints: 0,
  shipped_checkpoints_note:
    "no pretrained predictor ships — this lane ships the statistical " +
    "association layer itself",
  screen:
    "SYNTHETIC — the backend's own seeded demo screen with planted " +
    "TP53/KRAS associations",
  disclaimer:
    "p-values, q-values, effect sizes and rankings are statistics " +
    "over the synthetic screen — associations are triage " +
    "hypotheses, never clinical recommendations",
};

const PG_REGISTRY = {
  associations: {
    family: "the classic GDSC ANOVA (gdsctools' anova.py)",
    design_matrix:
      "[1 | tissue dummies (reference = first sorted level) | msi | feature]",
    feasibility_gates: {
      drug: ">= 6 non-NA IC50 values",
      feature: ">= 3 positives AND >= 3 negatives",
      msi: ">= 2 lines each side",
      tissue: "dropped when a single tissue remains",
    },
    row_schema: ["drug", "feature", "pvalue", "qvalue", "fdr_percent"],
    ordering: "sorted by pvalue ascending, stable",
  },
  effect_sizes: {
    keys: ["effect_size", "glass_pos", "glass_neg", "delta_mean"],
    note: "Cohen's d on the pooled ddof=1 variance",
  },
  multiple_testing: {
    methods: ["bh_fdr", "qvalue"],
    bh_fdr: "step-up with the monotone back-substitution",
    qvalue: "the λ grid with the cubic spline",
    percent_convention:
      "gdsctools' convention: FDR_THRESHOLD = 25 means 25 %",
  },
  volcano: {
    transforms: "x = sign(delta_mean) · effect_size; y = -log10(p)",
    hit_rule: "pvalue <= 0.001 AND fdr_percent <= 25",
    verdicts: "sensitive (green) / resistant (red)",
    additional_fdr_lines: [0.01, 0.1, 10.0],
  },
  evaluation: {
    ruler: "overall + tissue + cell-residual + drug",
    guarded_metrics: ["pearson", "spearman", "kendall"],
    guards: "a constant prediction scores 0.0",
    normalized_metric: "the score on the residuals beyond the ruler",
  },
  vocabulary: {
    cna_values: {
      "-2": "deep loss",
      "-1": "single-copy loss",
      "0": "diploid",
      "1": "low-level gain",
      "2": "high-level amplification",
    },
    diploid_zscores: "the reference population is the diploid samples",
    coexpression: "t = r·sqrt((n-2)/(1-r²))",
    enrichment_dispatchers: "two groups → Fisher exact",
  },
  screen: {
    geometry: { n_cells: 40, n_drugs: 12, n_features: 10 },
    planted_associations: "TP53_mut (-0.8 on five of the twelve drugs)",
    label: "SYNTHETIC — the backend's own seeded demo screen",
    scale: "the natural-log µM scale",
    missing: "~15 % untested",
  },
  provenance: PG_PROVENANCE,
  analyze_url: "/api/v1/pharmacogenomics/analyze",
  note: "sync-only light compute — a 40x12 screen",
};

function associationRow(
  drug: string,
  feature: string,
  overrides: Record<string, unknown> = {},
) {
  return {
    drug,
    feature,
    n_pos: 16,
    n_neg: 14,
    mean_pos: 0.31,
    mean_neg: 1.12,
    delta_mean: -0.81,
    effect_size: -1.24,
    glass_pos: -1.31,
    glass_neg: -1.18,
    t_pvalue: 0.0004,
    pvalue: 0.00021,
    qvalue: 0.014,
    fdr_percent: 1.4,
    tissue_pvalue: 0.44,
    msi_pvalue: 0.71,
    volcano_x: -1.24,
    volcano_y: 3.68,
    verdict: "sensitive",
    ...overrides,
  };
}

function analyzeResponse(overrides: Record<string, unknown> = {}) {
  return {
    request: { seed: 42, top_k: 5, include_msi: true },
    screen: {
      n_cells: 40,
      n_drugs: 12,
      n_features: 10,
      drugs: ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D10", "D11", "D12"],
      features: [
        "TP53_mut", "KRAS_mut", "BRAF_mut", "ARID1A_mut", "PTEN_mut",
        "PIK3CA_mut", "EGFR_mut", "STK11_mut", "CDKN2A_mut", "NRAS_mut",
      ],
      tissues: { lung: 13, breast: 14, colon: 13 },
      msi: { msi_positive: 4, msi_negative: 36 },
      n_observed: 408,
      n_entries: 480,
      observed_fraction: 0.85,
      label: "SYNTHETIC seeded screen — not real pharmacology",
    },
    associations: {
      n_rows: 108,
      n_drugs_tested: 12,
      n_features_tested: 9,
      top_k: 5,
      top: [
        associationRow("D01", "TP53_mut"),
        associationRow("D02", "TP53_mut"),
        associationRow("D06", "KRAS_mut"),
        associationRow("D03", "TP53_mut"),
        associationRow("D07", "KRAS_mut"),
      ],
      ordering: "sorted by pvalue ascending, stable",
    },
    volcano: {
      n_sensitive: 11,
      n_resistant: 2,
      n_not_significant: 95,
      hit_rule: "pvalue <= 0.001 AND fdr_percent <= 25",
    },
    evaluation: {
      n_observed: 408,
      ruler: "the tissue-aware mean-effects ruler",
      guarded: { pearson: 0.86, spearman: 0.83, kendall: 0.68 },
      naive_drug_mean: { pearson: 0.31, spearman: 0.29, kendall: 0.22 },
      naive_cell_mean: { pearson: 0.44, spearman: 0.4, kendall: 0.3 },
      normalized_beyond_drug_mean: { pearson: 0.79, spearman: 0.74 },
      guards_note: "a constant prediction scores 0",
      normalized_note: "the score beyond the naive drug-mean baseline",
    },
    provenance: PG_PROVENANCE,
    note: "p-values are statistics over the synthetic screen",
    ...overrides,
  };
}

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
  mlMocks.pharmacogenomicsRegistry.mockResolvedValue(PG_REGISTRY);
  mlMocks.pharmacogenomicsAnalyze.mockReset();
});

describe("the pharmacogenomics lane of the Machine Learning workspace", () => {
  it("renders the stage with the registry vocabulary line", async () => {
    render(<MLPage />);
    await waitFor(() => {
      expect(screen.getByText("Pharmacogenomics")).toBeInTheDocument();
    });
    // the backend's own registry summary, never re-declared
    await waitFor(() => {
      expect(
        screen.getByText(/The feasibility gates: a drug needs ≥ 6 non-NA/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/a feature ≥ 3 positives AND ≥ 3 negatives/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/FDR_THRESHOLD = 25 means 25 %/),
    ).toBeInTheDocument();
    expect(mlMocks.pharmacogenomicsRegistry).toHaveBeenCalledTimes(1);
  });

  it("runs the analyze round trip and renders the association table", async () => {
    mlMocks.pharmacogenomicsAnalyze.mockResolvedValue(analyzeResponse());
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the ANOVA",
    });
    await userEvent.click(button);
    // the default parameter block becomes the honest request
    await waitFor(() => {
      expect(mlMocks.pharmacogenomicsAnalyze).toHaveBeenCalledWith({
        seed: 42,
        top_k: 5,
      });
    });
    // the summary line over the REAL parsed payload
    expect(
      await screen.findByText(/40×12 screen \(SYNTHETIC\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/408\/480 observed entries \(85%\)/),
    ).toBeInTheDocument();
    expect(screen.getByText(/108 associations over/)).toBeInTheDocument();
    expect(
      screen.getByText(/12 drugs × 9 features/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/volcano: 11 sensitive \/ 2 resistant \/ 95 not significant/),
    ).toBeInTheDocument();
    // the top-association table: the planted feature wins with its
    // labelled statistics
    expect(
      screen.getByText(/Top 5 associations/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /statistic over the synthetic screen, sorted by p/,
      ),
    ).toBeInTheDocument();
    expect(screen.getAllByText("TP53_mut").length).toBeGreaterThanOrEqual(3);
    expect(screen.getAllByText("KRAS_mut").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("-0.810").length).toBeGreaterThanOrEqual(1); // the Δ-mean cells
    expect(screen.getAllByText("0.00021").length).toBeGreaterThanOrEqual(1); // the p cells
    expect(screen.getAllByText("0.01400").length).toBeGreaterThanOrEqual(1); // the q cells
    expect(screen.getAllByText("1.40").length).toBeGreaterThanOrEqual(1); // the FDR % cells
    expect(screen.getAllByText("sensitive").length).toBeGreaterThanOrEqual(5);
    // the ruler scores line (guarded + normalized)
    expect(
      screen.getByText(/the mean-effects ruler \(guarded pearson 0\.860/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/normalized beyond the naive drug-mean ruler \(pearson 0\.790\)/),
    ).toBeInTheDocument();
  });

  it("labels the estimates and renders null telemetry honestly", async () => {
    mlMocks.pharmacogenomicsAnalyze.mockResolvedValue(
      analyzeResponse({
        evaluation: {
          n_observed: 408,
          ruler: "the tissue-aware mean-effects ruler",
          guarded: { pearson: null, spearman: null, kendall: 0.68 },
          naive_drug_mean: { pearson: 0.31, spearman: 0.29, kendall: 0.22 },
          naive_cell_mean: { pearson: 0.44, spearman: 0.4, kendall: 0.3 },
          normalized_beyond_drug_mean: { pearson: null, spearman: 0.74 },
          guards_note: "a constant prediction scores 0",
          normalized_note: "the score beyond the naive drug-mean baseline",
        },
        associations: {
          n_rows: 108,
          n_drugs_tested: 12,
          n_features_tested: 9,
          top_k: 5,
          top: [associationRow("D01", "TP53_mut", { delta_mean: null })],
          ordering: "sorted by pvalue ascending, stable",
        },
      }),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the ANOVA",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(screen.getByText(/guarded pearson null/)).toBeInTheDocument();
    });
    // a null metric reads as null, never zero — the ruler line and
    // the normalized line both carry the honest absence, and a null
    // Δ-mean cell reads as null, never 0.000
    expect(
      screen.getByText(/normalized beyond the naive drug-mean ruler \(pearson null\)/),
    ).toBeInTheDocument();
    expect(screen.getAllByText("null").length).toBeGreaterThanOrEqual(1);
    // the top-table header labels every row a statistic over the
    // synthetic screen
    expect(
      screen.getByText(/statistic over the synthetic screen/),
    ).toBeInTheDocument();
  });

  it("surfaces the backend's own hard 422 verbatim through the banner", async () => {
    mlMocks.pharmacogenomicsAnalyze.mockRejectedValue(
      new ApiError(422, {
        detail: "top_k must be within 1..20 (got 21)",
      }),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the ANOVA",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(screen.getByText(/API 422/)).toBeInTheDocument();
    });
    expect(
      screen.getByText(/top_k must be within 1\.\.20/),
    ).toBeInTheDocument();
    expect(mlMocks.pharmacogenomicsAnalyze).toHaveBeenCalledTimes(1);
  });

  it("gates a negative seed client-side — no request fires", async () => {
    render(<MLPage />);
    const seed = await screen.findByLabelText("pharmacogenomics seed");
    // one fireEvent.change instead of keystroke simulation (the
    // r316 lesson: the gate assertion is the point, the typing is not)
    fireEvent.change(seed, { target: { value: "-1" } });
    const button = screen.getByRole("button", { name: "Run the ANOVA" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/the seed must lie in 0\.\.2147483647/),
    ).toBeInTheDocument();
    expect(mlMocks.pharmacogenomicsAnalyze).not.toHaveBeenCalled();
  });

  it("gates an over-range seed client-side — no request fires", async () => {
    render(<MLPage />);
    const seed = await screen.findByLabelText("pharmacogenomics seed");
    await userEvent.clear(seed);
    await userEvent.type(seed, "2147483648");
    const button = screen.getByRole("button", { name: "Run the ANOVA" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/the seed must lie in 0\.\.2147483647/),
    ).toBeInTheDocument();
    expect(mlMocks.pharmacogenomicsAnalyze).not.toHaveBeenCalled();
  });

  it("gates an out-of-band top-k client-side — no request fires", async () => {
    render(<MLPage />);
    const topK = await screen.findByLabelText("pharmacogenomics top_k");
    await userEvent.clear(topK);
    await userEvent.type(topK, "21");
    const button = screen.getByRole("button", { name: "Run the ANOVA" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/the answer carries 1\.\.20 associations/),
    ).toBeInTheDocument();
    expect(mlMocks.pharmacogenomicsAnalyze).not.toHaveBeenCalled();
  });

  it("carries the honest provenance and disclaimer on every answer", async () => {
    mlMocks.pharmacogenomicsAnalyze.mockResolvedValue(analyzeResponse());
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the ANOVA",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(
        screen.getByText(/0 shipped tables, 0 shipped checkpoints/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/never clinical recommendations/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/SYNTHETIC — the backend's own seeded demo screen/),
    ).toBeInTheDocument();
  });
});
