import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import { ApiError } from "@/services/api/client";

/**
 * part13 Phase 5's LAST remainder item — the multi-objective
 * optimization lane of the Machine Learning workspace as a complete
 * component-level user flow:
 *
 *   the registry line      — the stage renders the backend's own
 *                            vocabulary summary (the minimize
 *                            convention + the tunable-objective ledger
 *                            keys) off the mocked registry answer;
 *   the study round trip   — the default parameter block becomes the
 *                            honest request (the model, the task, the
 *                            objectives, the two randint dimensions,
 *                            the trials/cv/seed/weights/keep dials),
 *                            and the answer renders the REAL parsed
 *                            payload (the study summary, the
 *                            non-dominated front table with the
 *                            objective-named value columns and the
 *                            crowding cells, the trade-off corners,
 *                            the crowding-pruned shortlist);
 *   the honest nulls       — a 3-objective study reports a null
 *                            hypervolume (the exact sweep is 2-D
 *                            only, never a fabricated indicator) and
 *                            the +inf boundary crowding reads as
 *                            "boundary", never as a zero spread;
 *   typed errors           — the backend's own hard 422 surfaces
 *                            verbatim through the banner;
 *   client-side gating     — an out-of-band n_trials, a negative seed
 *                            and a one-objective study never leave the
 *                            page (validated BEFORE any request
 *                            fires);
 *   honest provenance      — the "0 shipped checkpoints" absence and
 *                            the disclaimer ride every answer.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers
 * (parseMultiobjectiveRegistryResponse /
 * parseMultiobjectiveOptimizeResponse — the REAL ones, spread from
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
  multiobjectiveRegistry: vi.fn(),
  multiobjectiveOptimize: vi.fn(),
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
      multiobjectiveRegistry: mlMocks.multiobjectiveRegistry,
      multiobjectiveOptimize: mlMocks.multiobjectiveOptimize,
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

const MO_PROVENANCE = {
  backend: "pharmaml.ml.multiobjective",
  source:
    "the best-of-ml-python HPO/AutoML leaders reviewed file by file — " +
    "optuna/optuna (the multi-objective study); facebook/Ax + BoTorch " +
    "(the expected hypervolume improvement over a GP); " +
    "facebookresearch/nevergrad (the portfolio of gradient-free " +
    "optimizers per objective)",
  n_repositories: 3,
  transfers_exactly:
    "the Pareto order (dominates/pareto_indices), the exact 2-D " +
    "hypervolume (the staircase sweep), NSGA-II's crowding distance, " +
    "the MOEA/D Tchebycheff decomposition, and the study arithmetic " +
    "that drives BOTH existing samplers (random + TPE) as " +
    "front-chasers — pure NumPy",
  stays_upstream:
    "EHVI needs a GP posterior over a vector objective (this house's " +
    "GP is scalar); MOTPE's per-side multivariate Parzen estimators " +
    "and NSGA-II's genetic operators stay upstream; the exact " +
    "hypervolume beyond two objectives (WFG's sweep) is not vendored — " +
    "the 2-D sweep is the indicator reported",
  shipped_checkpoints: 0,
  shipped_checkpoints_note:
    "no pretrained anything ships — the study runs in-tree over the " +
    "seeded synthetic dataset on every request",
  dataset:
    "the served dataset is the route's own SYNTHETIC seeded stand-in " +
    "(80 rows x 6 features, default_rng(seed)) — no real dataset ships " +
    "with this route",
  disclaimer:
    "every objective value is a cross-validated statistic over the " +
    "synthetic dataset — front members are model estimates for triage, " +
    "never experimental facts and never a deployment recommendation",
};

const MO_REGISTRY = {
  order: {
    convention:
      "the optuna convention — every objective is MINIMIZED; 'score' " +
      "rides the negated mean CV score",
    dominates:
      "a dominates b iff a is at least as good on every objective and " +
      "strictly better on at least one",
    pareto_indices:
      "the non-dominated set, ascending input order — objective-space " +
      "duplicates survive together (mutual non-domination is the " +
      "definition)",
    complexity: "pairwise O(n^2 * k) at study scale — the honest cost",
  },
  indicator: {
    family: "the exact 2-D dominated hypervolume (the staircase sweep)",
    reference_contract:
      "the WFG contract — the reference point must dominate every " +
      "point; a non-dominating reference raises, never silently clips",
    report_reference:
      "the observed nadir padded by 10 % of the ideal-to-nadir range " +
      "per axis (the documented margin)",
    curve:
      "the per-trial indicator under that same FIXED reference — " +
      "monotonically non-decreasing by construction (a growing point " +
      "set grows the union)",
    two_d_only:
      "the exact sweep is 2-D only — a 3-objective study reports " +
      "null, never a fabricated indicator",
  },
  spread: {
    family: "NSGA-II's crowding distance (boundaries +inf)",
    rule:
      "the two extremes of every objective gain +inf and each interior " +
      "point gains the normalised neighbour gap; a degenerate " +
      "objective (zero span) contributes nothing",
    json_note:
      "boundary members' +inf reads as null in JSON — the crowding " +
      "note on every answer states the convention",
  },
  decomposition: {
    family: "the MOEA/D Tchebycheff decomposition",
    form: "max_j w_j |v_j - ideal_j|, minimized",
    rule:
      "a zero weight is the corner subproblem that optimizes its " +
      "objective alone (the front's endpoint); at least one weight " +
      "must be positive, negative weights are meaningless",
  },
  study: {
    samplers: ["random", "tpe"],
    samplers_note:
      "the decomposition drives BOTH existing samplers as " +
      "front-chasers — each trial's Tchebycheff distance to the " +
      "running ideal point is the single number the underlying " +
      "engine is told",
    weight_modes: {
      spread:
        "Das & Dennis's simplex-lattice weight set, deterministic " +
        "spread, corner-heavy prefix, cycled",
      random:
        "seeded uniform simplex draws — exponential variates " +
        "normalised to sum 1 (the Dirichlet(1,...,1) shortcut, " +
        "Optuna's own approach)",
    },
    n_objectives: "a Pareto study needs at least two objectives",
    drift_note:
      "the ideal point advances as the front does, so the scalar " +
      "landscape is not strictly stationary across trials — the " +
      "hypervolume curve is the convergence check that the drift " +
      "still descends",
  },
  tunable_objectives: {
    ledger: {
      score: { cross_validate_key: "test_score", sign: -1 },
      fit_seconds: { cross_validate_key: "fit_time", sign: 1 },
      score_seconds: { cross_validate_key: "score_time", sign: 1 },
    },
    note:
      "'score' = the negated mean CV score (bigger is better -> " +
      "minimize the negation); 'fit_seconds' / 'score_seconds' = the " +
      "mean fold times — the same numbers pharmaml.ml.cv_results " +
      "records",
    bounds: "2..3 distinct entries per request",
  },
  report: {
    schema: [
      "objective_names",
      "front (the non-dominated MOTrial members in study order)",
      "crowding (aligned with the front)",
      "ref",
      "hypervolume",
      "hypervolume_curve",
      "n_trials",
    ],
    best_for:
      "the front member minimizing one named objective (its " +
      "endpoint); an unknown name or an empty front raises",
    prune:
      "up to keep front members by full-front crowding, largest " +
      "first, ties in front order — a one-shot selection; the " +
      "shortlist never claims the full study's hypervolume as its own",
  },
  request: {
    defaults: {
      model: "random_forest",
      task: "classification",
      objectives: ["score", "fit_seconds"],
      n_trials: 20,
      cv: 3,
      sampler: "tpe",
      seed: 42,
      weight_mode: "spread",
      n_weight_vectors: 8,
      shortlist_keep: 5,
    },
    bounds: {
      n_trials: "1..60",
      cv: "2..5",
      dimensions: "1..4",
      n_weight_vectors: "2..32",
      shortlist_keep: "1..10",
      seed: "0..2147483647",
    },
    dimension_kinds: ["uniform", "loguniform", "randint"],
    model_note:
      "the model is any sklearn registry name — the backend's own " +
      "build_model validates it (unknown names and unknown " +
      "hyperparameter names raise its typed ModelError, a 422, never " +
      "a 500)",
  },
  dataset: {
    geometry: { n_samples: 80, n_features: 6 },
    label:
      "SYNTHETIC — the route's own seeded stand-in (default_rng(seed)); " +
      "no real dataset ships",
  },
  provenance: MO_PROVENANCE,
  optimize_url: "/api/v1/multiobjective/optimize",
  note:
    "sync-only light compute — bounded trials over a small dense " +
    "synthetic frame; the seed fixes the dataset and (under the " +
    "random sampler) the parameterization sequence, while the " +
    "fit/score-seconds objectives are wall-clock fold timings that " +
    "vary run-to-run and the TPE sampler conditions on the observed " +
    "scalars — the honest drift, stated",
};

function trialRow(
  number: number,
  params: Record<string, number>,
  values: number[],
  crowding?: number | null,
  boundary = false,
) {
  return {
    number,
    params,
    values,
    weights: values.map(() => 1 / values.length),
    scalar:
      values.length > 0
        ? values.reduce((acc, v) => acc + v / values.length, 0)
        : null,
    ...(crowding === undefined
      ? {}
      : { crowding, crowding_is_boundary: boundary }),
  };
}

function optimizeResponse(overrides: Record<string, unknown> = {}) {
  return {
    request: {
      model: "random_forest",
      task: "classification",
      objectives: ["score", "fit_seconds"],
      dimensions: [
        { name: "n_estimators", kind: "randint", low: 10, high: 80 },
        { name: "max_depth", kind: "randint", low: 2, high: 10 },
      ],
      n_trials: 20,
      cv: 3,
      sampler: "tpe",
      seed: 42,
      weight_mode: "spread",
      n_weight_vectors: 8,
      shortlist_keep: 5,
    },
    dataset: {
      n_samples: 80,
      n_features: 6,
      task: "classification",
      label: "SYNTHETIC — the route's own seeded stand-in, not a real dataset",
    },
    study: {
      n_trials: 20,
      n_objectives: 2,
      objective_names: ["score", "fit_seconds"],
      sampler: "tpe",
      weight_mode: "spread",
      n_weight_vectors: 8,
      n_front: 4,
      objectives_note:
        "every objective is minimized — 'score' is the negated mean " +
        "CV score, so a lower score value means a MORE accurate model",
    },
    front: [
      trialRow(3, { n_estimators: 74, max_depth: 9 }, [-0.94, 0.281], null, true),
      trialRow(7, { n_estimators: 41, max_depth: 6 }, [-0.91, 0.152], 0.31),
      trialRow(11, { n_estimators: 23, max_depth: 5 }, [-0.86, 0.081], 0.47),
      trialRow(16, { n_estimators: 12, max_depth: 3 }, [-0.79, 0.037], null, true),
    ],
    crowding_note:
      "NSGA-II crowding aligned with the front; boundary members " +
      "carry +inf and read as null here (crowding_is_boundary marks " +
      "them) — null is never a zero spread",
    indicator: {
      ref: [-0.5, 0.36],
      hypervolume: 0.1342,
      hypervolume_curve: [0.01, 0.04, 0.09, 0.12, 0.1342],
      two_d_only:
        "the exact sweep is 2-D only — null here means a 3-objective " +
        "study, never a fabricated indicator",
      curve_note:
        "the per-trial indicator under the FIXED reference — " +
        "monotonically non-decreasing; the convergence check for the " +
        "drifting ideal point",
    },
    best_for: {
      score: trialRow(3, { n_estimators: 74, max_depth: 9 }, [-0.94, 0.281]),
      fit_seconds: trialRow(
        16,
        { n_estimators: 12, max_depth: 3 },
        [-0.79, 0.037],
      ),
    },
    best_for_note:
      "the front endpoint minimizing each named objective — the " +
      "trade-off corners of the accuracy-vs-cost menu",
    shortlist: {
      keep: 5,
      n_front: 4,
      n_trials: 20,
      trials: [
        trialRow(11, { n_estimators: 23, max_depth: 5 }, [-0.86, 0.081], 0.47),
        trialRow(7, { n_estimators: 41, max_depth: 6 }, [-0.91, 0.152], 0.31),
      ],
      note:
        "selected by full-front crowding, largest first — a one-shot " +
        "selection that never claims the full study's hypervolume as " +
        "its own",
    },
    provenance: MO_PROVENANCE,
    timings_note:
      "the seed fixes the dataset and (under the random sampler) the " +
      "parameterization sequence; the fit/score-seconds columns are " +
      "wall-clock fold timings that vary run-to-run and the TPE " +
      "sampler conditions on the observed scalars — the honest drift, " +
      "stated where the timings ride",
    note:
      "every objective value is a cross-validated statistic over the " +
      "synthetic dataset — front members are model estimates for " +
      "triage, never experimental facts; no real dataset ships with " +
      "this route",
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
  // structurally complete DTA registry stub — the neighbour stages
  // render their registry lines off these fields, so the lane under
  // test never crashes its neighbours while this suite focuses on
  // the multi-objective stage
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
  // the drug-response registry stub — the same neighbour contract
  mlMocks.drugResponseRegistry.mockResolvedValue({
    model: {
      family: "CaDRReS-style bilinear factorisation with a sigmoid link",
      formulation:
        "gamma = X.W; the score is the sigmoid of the bilinear form",
      direction:
        "the score is the sigmoid-linked response — higher = " +
        "stronger predicted response",
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
    provenance: {
      backend: "pharmaml.ml.drugresponse",
      source: "",
      n_repositories: 3,
      shipped_checkpoints: 0,
      shipped_checkpoints_note: "",
      screen: "",
      disclaimer: "",
    },
    analyze_url: "/api/v1/drugresponse/analyze",
    note: "sync-only light compute — dense bounded matrices",
  });
  // the pharmacogenomics registry stub — the same neighbour contract
  mlMocks.pharmacogenomicsRegistry.mockResolvedValue({
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
      percent_convention: "gdsctools' convention: FDR_THRESHOLD = 25 means 25 %",
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
        0: "diploid",
        1: "low-level gain",
        2: "high-level amplification",
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
    provenance: {
      backend: "pharmaml.ml.pharmacogenomics",
      source: "",
      n_repositories: 5,
      shipped_tables: 0,
      shipped_tables_note: "",
      shipped_checkpoints: 0,
      shipped_checkpoints_note: "",
      screen: "",
      disclaimer: "",
    },
    analyze_url: "/api/v1/pharmacogenomics/analyze",
    note: "sync-only light compute — a 40x12 screen",
  });
  mlMocks.multiobjectiveRegistry.mockResolvedValue(MO_REGISTRY);
  mlMocks.multiobjectiveOptimize.mockReset();
});

describe("the multi-objective optimization lane of the Machine Learning workspace", () => {
  it("renders the stage with the registry vocabulary line", async () => {
    render(<MLPage />);
    await waitFor(() => {
      expect(screen.getByText("Multi-objective Optimization")).toBeInTheDocument();
    });
    // the backend's own registry summary, never re-declared
    await waitFor(() => {
      expect(
        screen.getByText(/Every objective is minimized — 'score' rides/),
      ).toBeInTheDocument();
    });
    // the tunable-objective ledger keys off the REAL parsed registry
    expect(
      screen.getByText(/score · fit_seconds · score_seconds/),
    ).toBeInTheDocument();
    expect(mlMocks.multiobjectiveRegistry).toHaveBeenCalledTimes(1);
  });

  it("runs the study round trip and renders the front table", async () => {
    mlMocks.multiobjectiveOptimize.mockResolvedValue(optimizeResponse());
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the study",
    });
    await userEvent.click(button);
    // the default parameter block becomes the honest request
    await waitFor(() => {
      expect(mlMocks.multiobjectiveOptimize).toHaveBeenCalledWith({
        model: "random_forest",
        task: "classification",
        objectives: ["score", "fit_seconds"],
        dimensions: [
          { name: "n_estimators", kind: "randint", low: 10, high: 80 },
          { name: "max_depth", kind: "randint", low: 2, high: 10 },
        ],
        n_trials: 20,
        cv: 3,
        sampler: "tpe",
        seed: 42,
        weight_mode: "spread",
        n_weight_vectors: 8,
        shortlist_keep: 5,
      });
    });
    // the summary line over the REAL parsed payload
    expect(
      await screen.findByText(/20 trials over 80×6 \(SYNTHETIC\)/),
    ).toBeInTheDocument();
    expect(screen.getByText(/front 4/)).toBeInTheDocument();
    expect(screen.getByText(/score vs fit_seconds/)).toBeInTheDocument();
    expect(screen.getByText(/hypervolume 0\.1342 under ref \[-0\.500, 0\.360\]/)).toBeInTheDocument();
    expect(screen.getByText(/curve 5 points/)).toBeInTheDocument();
    // the front table: the objective-named value columns, the params
    // cells, the crowding cells (boundary members marked, never a
    // zero spread)
    expect(
      screen.getByText(/The non-dominated front \(4 members/),
    ).toBeInTheDocument();
    expect(screen.getByText("n_estimators=74, max_depth=9")).toBeInTheDocument();
    expect(screen.getByText("n_estimators=12, max_depth=3")).toBeInTheDocument();
    expect(screen.getAllByText("boundary").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("0.310").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("0.470").length).toBeGreaterThanOrEqual(1);
    // the trade-off corners carry their own lines
    expect(screen.getByText(/best score: trial #3/)).toBeInTheDocument();
    expect(screen.getByText(/best fit_seconds: trial #16/)).toBeInTheDocument();
    // the crowding-pruned shortlist rides the same honest table
    expect(
      screen.getByText(/Shortlist — keep 5 by full-front crowding/),
    ).toBeInTheDocument();
  });

  it("renders the honest nulls — a 3-objective study reports a null indicator", async () => {
    mlMocks.multiobjectiveOptimize.mockResolvedValue(
      optimizeResponse({
        study: {
          n_trials: 20,
          n_objectives: 3,
          objective_names: ["score", "fit_seconds", "score_seconds"],
          sampler: "tpe",
          weight_mode: "spread",
          n_weight_vectors: 8,
          n_front: 3,
          objectives_note:
            "every objective is minimized — 'score' is the negated " +
            "mean CV score",
        },
        front: [
          trialRow(
            3,
            { n_estimators: 74, max_depth: 9 },
            [-0.94, 0.281, 0.09],
            null,
            true,
          ),
          trialRow(
            7,
            { n_estimators: 41, max_depth: 6 },
            [-0.91, 0.152, 0.06],
            0.31,
          ),
          trialRow(
            16,
            { n_estimators: 12, max_depth: 3 },
            [-0.79, 0.037, 0.02],
            null,
            true,
          ),
        ],
        indicator: {
          ref: null,
          hypervolume: null,
          hypervolume_curve: [],
          two_d_only:
            "the exact sweep is 2-D only — null here means a " +
            "3-objective study, never a fabricated indicator",
          curve_note: "",
        },
        best_for: {
          score: trialRow(
            3,
            { n_estimators: 74, max_depth: 9 },
            [-0.94, 0.281, 0.09],
          ),
          fit_seconds: trialRow(
            16,
            { n_estimators: 12, max_depth: 3 },
            [-0.79, 0.037, 0.02],
          ),
          score_seconds: trialRow(
            16,
            { n_estimators: 12, max_depth: 3 },
            [-0.79, 0.037, 0.02],
          ),
        },
      }),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the study",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(
        screen.getByText(/score vs fit_seconds vs score_seconds/),
      ).toBeInTheDocument();
    });
    // a null hypervolume reads as null, never zero — the exact sweep
    // is 2-D only and a 3-objective study never fabricates the
    // indicator; the empty curve rides the same honest absence
    expect(screen.getByText(/hypervolume null/)).toBeInTheDocument();
    expect(screen.getByText(/curve 0 points/)).toBeInTheDocument();
    // the +inf boundary crowding reads as "boundary", never as a
    // zero spread — the honest NSGA-II convention
    expect(screen.getAllByText("boundary").length).toBeGreaterThanOrEqual(2);
    // the third objective's trade-off corner carries its own line
    expect(
      screen.getByText(/best score_seconds: trial #16/),
    ).toBeInTheDocument();
  });

  it("surfaces the backend's own hard 422 verbatim through the banner", async () => {
    mlMocks.multiobjectiveOptimize.mockRejectedValue(
      new ApiError(422, {
        detail:
          "dimension 'max_depth' needs low < high, got 10 !< 3",
      }),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the study",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(screen.getByText(/API 422/)).toBeInTheDocument();
    });
    expect(
      screen.getByText(/dimension 'max_depth' needs low < high/),
    ).toBeInTheDocument();
    expect(mlMocks.multiobjectiveOptimize).toHaveBeenCalledTimes(1);
  });

  it("gates an out-of-band n_trials client-side — no request fires", async () => {
    render(<MLPage />);
    const trials = await screen.findByLabelText("multiobjective n_trials");
    fireEvent.change(trials, { target: { value: "61" } });
    const button = screen.getByRole("button", { name: "Run the study" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/n_trials must be an integer in \[1, 60\]/),
    ).toBeInTheDocument();
    expect(mlMocks.multiobjectiveOptimize).not.toHaveBeenCalled();
  });

  it("gates a one-objective study client-side — no request fires", async () => {
    render(<MLPage />);
    // uncheck fit_seconds: the default pair drops to a single
    // objective, which is not a Pareto study
    const fitSeconds = await screen.findByLabelText(
      "multiobjective objective fit_seconds",
    );
    await userEvent.click(fitSeconds);
    const button = screen.getByRole("button", { name: "Run the study" });
    await userEvent.click(button);
    expect(
      await screen.findByText(
        /a Pareto study needs 2\.\.3 distinct objectives/,
      ),
    ).toBeInTheDocument();
    expect(mlMocks.multiobjectiveOptimize).not.toHaveBeenCalled();
  });

  it("gates a negative seed client-side — no request fires", async () => {
    render(<MLPage />);
    const seed = await screen.findByLabelText("multiobjective seed");
    fireEvent.change(seed, { target: { value: "-1" } });
    const button = screen.getByRole("button", { name: "Run the study" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/the seed must lie in 0\.\.2147483647/),
    ).toBeInTheDocument();
    expect(mlMocks.multiobjectiveOptimize).not.toHaveBeenCalled();
  });

  it("carries the honest provenance and disclaimer on every answer", async () => {
    mlMocks.multiobjectiveOptimize.mockResolvedValue(optimizeResponse());
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the study",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(
        screen.getByText(/0 shipped checkpoints/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/never a deployment recommendation/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/SYNTHETIC — the route's own seeded stand-in, not a real dataset/),
    ).toBeInTheDocument();
  });
});
