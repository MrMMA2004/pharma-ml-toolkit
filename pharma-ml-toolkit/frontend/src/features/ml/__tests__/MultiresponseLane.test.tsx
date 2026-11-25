import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import { ApiError } from "@/services/api/client";

/**
 * part17 §5 (the M1 lane) — the generic multi-response regression
 * lane of the Machine Learning workspace as a complete
 * component-level user flow:
 *
 *   the registry line      — the stage renders the backend's own
 *                            vocabulary (the mechanism per strategy,
 *                            the sample-weight capability) off the
 *                            mocked registry answer;
 *   the train round trip   — the demo frame + the checked feature and
 *                            >=2 target columns + the strategy select
 *                            become the honest request, and the answer
 *                            renders the REAL parsed payload (the
 *                            per-target metric cards with every
 *                            target's own R2/RMSE/MAE/MSE/explained
 *                            variance and test-row count, the
 *                            uniform/weighted aggregates, the target
 *                            correlation matrix, the CV per-target
 *                            mean, the predicted-vs-actual chart off
 *                            the bounded residual sample, the
 *                            SYNTHETIC provenance line);
 *   the models surface     — the comparison table (the per-target
 *                            R2 side by side) and the models refetch
 *                            after a train lands;
 *   typed errors           — the backend's own hard 422 surfaces
 *                            verbatim through the banner;
 *   client-side gating     — a one-target selection never leaves the
 *                            page (validated BEFORE any request
 *                            fires);
 *   the honest bound       — the residual sample reports its bound,
 *                            never silent.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers
 * (parseMultiresponseRegistryResponse /
 * parseMultiresponseTrainResponse / parseMultiresponsePredictResponse
 * / parseMultiresponseModelsResponse — the REAL ones, spread from
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
  multiresponseRegistry: vi.fn(),
  multiresponseTrain: vi.fn(),
  multiresponsePredict: vi.fn(),
  multiresponseModels: vi.fn(),
  multiresponseExport: vi.fn(),
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
      multiresponseRegistry: mlMocks.multiresponseRegistry,
      multiresponseTrain: mlMocks.multiresponseTrain,
      multiresponsePredict: mlMocks.multiresponsePredict,
      multiresponseModels: mlMocks.multiresponseModels,
      multiresponseExport: mlMocks.multiresponseExport,
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

const MR_REGISTRY = {
  strategies: [
    {
      key: "ridge",
      label: "Ridge",
      mechanism: "native",
      native_multi_output: true,
      sample_weight_support: true,
      notes: "the closed-form baseline: natively 2-D Y",
      base_keys: [],
      default_base: null,
    },
    {
      key: "pls2",
      label: "PLSRegression (PLS2)",
      mechanism: "native",
      native_multi_output: true,
      sample_weight_support: false,
      notes: "the cross-decomposition classic",
      base_keys: [],
      default_base: null,
    },
    {
      key: "multi_output",
      label: "MultiOutputRegressor (wrapper)",
      mechanism: "wrapper",
      native_multi_output: false,
      sample_weight_support: true,
      notes: "one clone of the base estimator per target",
      base_keys: ["ridge", "svr", "gradient_boosting"],
      default_base: "ridge",
    },
  ],
  missing_strategies: {
    complete_case: {
      label: "Complete-case analysis",
      description: "rows missing ANY target column are dropped once",
      mechanisms: ["native", "wrapper", "chain"],
    },
    per_target: {
      label: "Per-target fitting (masked multi-task)",
      description: "each target's clone is fit on that target's rows",
      mechanisms: ["wrapper"],
    },
  },
  metrics: ["r2", "rmse", "mae", "mse", "explained_variance"],
  max_sample_rows: 500,
  demo_dataset: {
    features: ["x1", "x2", "x3", "x4", "x5"],
    targets: ["response_a", "response_b", "response_c"],
    n_rows: 240,
    synthetic: true,
    seed: 42,
    note: "the seeded synthetic demonstration frame",
  },
  bounds: {
    max_rows: 5000,
    max_feature_columns: 100,
    max_target_columns: 32,
    max_predict_rows: 500,
    cv_folds: [2, 10],
  },
  scope_statement: "the mechanism is stated per strategy, never implied",
};

function trainResponse() {
  return {
    model: {
      model_id: "mresp-test000001",
      name: "multiresponse-ridge",
      task: "multiresponse-regression",
      dataset_id: null,
      target: "response_a|response_b|response_c",
      features: ["x1", "x2", "x3", "x4", "x5"],
      n_train_rows: 180,
      best_params: {},
      cv_scores: [{ fold: 0 }, { fold: 1 }, { fold: 2 }],
      cv_mean: 0.995,
      cv_std: null,
      fit_seconds: 0.12,
      created_at: "2026-11-25T06:00:00Z",
      multiresponse: {
        target_names: ["response_a", "response_b", "response_c"],
        target_order: ["response_a", "response_b", "response_c"],
        mechanism: "native",
        missing_strategy: "complete_case",
        per_target_metrics: {
          response_a: {
            r2: 0.997,
            rmse: 0.104,
            mae: 0.082,
            mse: 0.0108,
            explained_variance: 0.997,
            n_test_rows: 60,
          },
          response_b: {
            r2: 0.993,
            rmse: 0.141,
            mae: 0.114,
            mse: 0.0199,
            explained_variance: 0.993,
            n_test_rows: 60,
          },
          response_c: {
            r2: 0.994,
            rmse: 0.126,
            mae: 0.1,
            mse: 0.0159,
            explained_variance: 0.994,
            n_test_rows: 60,
          },
        },
        aggregate_metrics: {
          uniform: { r2: 0.995, rmse: 0.124 },
          weighted: { r2: 0.996, rmse: 0.117 },
        },
        n_test_rows: 60,
      },
    },
    run_id: "run-test0000001",
    provenance: {
      source: "demo",
      dataset_id: null,
      dataset_name: "synthetic-multiresponse-demo",
      synthetic: true,
      note: "the seeded synthetic demonstration frame",
    },
    configuration: {
      strategy: "ridge",
      mechanism: "native",
      params: {},
      preprocess: true,
      target_scaling: false,
      test_size: 0.25,
      cv_folds: 3,
      missing: "complete_case",
      group_column: null,
      sample_weight_column: null,
      target_weights: null,
      random_state: 42,
    },
    data: {
      n_rows: 240,
      usable_rows_per_target: {
        response_a: 240,
        response_b: 200,
        response_c: 240,
      },
      missing_counts_per_target: { response_a: 0, response_b: 40, response_c: 0 },
      dropped_train_rows: 0,
      split: {
        strategy: "shuffle_split",
        train_rows: 180,
        test_rows: 60,
        n_train_groups: null,
        n_test_groups: null,
        group_overlap: [],
        test_size: 0.25,
        random_state: 42,
      },
    },
    per_target_metrics: {
      response_a: {
        r2: 0.997,
        rmse: 0.104,
        mae: 0.082,
        mse: 0.0108,
        explained_variance: 0.997,
        n_test_rows: 60,
      },
      response_b: {
        r2: 0.993,
        rmse: 0.141,
        mae: 0.114,
        mse: 0.0199,
        explained_variance: 0.993,
        n_test_rows: 60,
      },
      response_c: {
        r2: 0.994,
        rmse: 0.126,
        mae: 0.1,
        mse: 0.0159,
        explained_variance: 0.994,
        n_test_rows: 60,
      },
    },
    aggregate_metrics: {
      uniform: { r2: 0.995, rmse: 0.124, mae: 0.099, mse: 0.0155, explained_variance: 0.995 },
      weighted: { r2: 0.996, rmse: 0.117, mae: 0.093, mse: 0.0138, explained_variance: 0.996 },
    },
    residuals: {
      per_target: {
        response_a: {
          actual: [0.1, 0.2, 0.3],
          predicted: [0.11, 0.19, 0.31],
          residual: [-0.01, 0.01, -0.01],
          n_rows: 60,
          n_sampled: 3,
        },
        response_b: {
          actual: [0.5, 0.4, 0.3],
          predicted: [0.49, 0.41, 0.29],
          residual: [0.01, -0.01, 0.01],
          n_rows: 60,
          n_sampled: 3,
        },
        response_c: {
          actual: [-0.2, 0.0, 0.2],
          predicted: [-0.19, 0.01, 0.21],
          residual: [-0.01, -0.01, -0.01],
          n_rows: 60,
          n_sampled: 3,
        },
      },
      max_rows: 500,
    },
    target_correlation_matrix: {
      response_a: { response_a: 1.0, response_b: 0.464, response_c: 0.219 },
      response_b: { response_a: 0.464, response_b: 1.0, response_c: 0.127 },
      response_c: { response_a: 0.219, response_b: 0.127, response_c: 1.0 },
    },
    cv: {
      folds: 3,
      per_target: {
        response_a: { r2_mean: 0.997, r2_std: 0.001, r2_folds: [0.996, 0.997, 0.998] },
        response_b: { r2_mean: 0.993, r2_std: 0.002, r2_folds: [0.991, 0.993, 0.995] },
        response_c: { r2_mean: 0.994, r2_std: 0.001, r2_folds: [0.993, 0.994, 0.995] },
      },
      aggregate: { r2_uniform: 0.995, r2_weighted: 0.996 },
      folds_table: [{ fold: 0 }, { fold: 1 }, { fold: 2 }],
    },
    contract: {
      target_names: ["response_a", "response_b", "response_c"],
      target_order: ["response_a", "response_b", "response_c"],
    },
    fit_seconds: 0.12,
    max_sample_rows: 500,
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

  mlMocks.multiresponseRegistry.mockResolvedValue(MR_REGISTRY);
  mlMocks.multiresponseModels.mockResolvedValue({ models: [], metrics: [] });
  mlMocks.multiresponseTrain.mockReset();
  mlMocks.multiresponsePredict.mockReset();
  mlMocks.multiresponseExport.mockReset();

describe("the Multi-Response regression lane of the Machine Learning workspace", () => {
  it("renders the stage with the registry vocabulary", async () => {
    render(<MLPage />);
    await waitFor(() => {
      expect(screen.getByText("Multi-Response Regression")).toBeInTheDocument();
    });
    // the strategy select carries the backend's own labels
    const strategySelect = screen.getByLabelText("multiresponse strategy") as HTMLSelectElement;
    expect(strategySelect.value).toBe("ridge");
    const labels = Array.from(strategySelect.options).map((o) => o.textContent);
    expect(labels).toContain("PLSRegression (PLS2)");
    expect(labels).toContain("MultiOutputRegressor (wrapper)");
    // the mechanism + capability line off the REAL parsed registry
    await waitFor(() => {
      expect(screen.getByText(/mechanism native · sample weights supported/)).toBeInTheDocument();
    });
    expect(mlMocks.multiresponseRegistry).toHaveBeenCalledTimes(1);
  });

  it("runs the demo train round trip and renders the per-target surface", async () => {
    const train = trainResponse();
    mlMocks.multiresponseTrain.mockResolvedValue(train);
    mlMocks.multiresponseModels.mockResolvedValue({
      models: [train.model],
      metrics: ["r2", "rmse"],
    });
    render(<MLPage />);
    // check the feature and target columns (the demo frame's own)
    await userEvent.click(await screen.findByLabelText("x1"));
    await userEvent.click(await screen.findByLabelText("x2"));
    await userEvent.click(await screen.findByLabelText("x3"));
    await userEvent.click(await screen.findByLabelText("x4"));
    await userEvent.click(await screen.findByLabelText("x5"));
    await userEvent.click(await screen.findByLabelText("response_a"));
    await userEvent.click(await screen.findByLabelText("response_b"));
    await userEvent.click(await screen.findByLabelText("response_c"));
    const button = await screen.findByRole("button", {
      name: /Train the multi-response model/,
    });
    await userEvent.click(button);
    // the demo selection + defaults become the honest request
    await waitFor(() => {
      expect(mlMocks.multiresponseTrain).toHaveBeenCalledWith({
        dataset_id: null,
        feature_columns: ["x1", "x2", "x3", "x4", "x5"],
        target_columns: ["response_a", "response_b", "response_c"],
        strategy: "ridge",
        params: null,
        preprocess: true,
        target_scaling: false,
        test_size: 0.25,
        cv_folds: 3,
        missing: "complete_case",
        group_column: null,
        sample_weight_column: null,
        target_weights: null,
        random_state: 42,
      });
    });
    // the per-target metric cards — every target's own block
    expect(
      await screen.findByText(/the seeded SYNTHETIC demo frame — every number is a model estimate/),
    ).toBeInTheDocument();
    const cardA = screen
  .getAllByText("response_a", { selector: "p.font-mono.text-xs" })
  .map((node) => node.closest("div"))
  .find(Boolean) as HTMLElement;
    expect(within(cardA).getAllByText(/0\.997/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/60 test rows/).length).toBeGreaterThanOrEqual(3);
    // the aggregate rides alongside, never instead
    expect(screen.getByText(/aggregates — uniform:/)).toBeInTheDocument();
    expect(screen.getAllByText("0.995").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("0.996").length).toBeGreaterThanOrEqual(1);
    // the CV per-target mean ± std
    expect(
      screen.getByText(/response_a: 0\.997±0\.001/),
    ).toBeInTheDocument();
    // the target correlation matrix renders
    expect(screen.getAllByText(/0\.46/).length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText(/0\.22/).length).toBeGreaterThanOrEqual(2);
    // the residual chart (the bounded sample + the bound reported)
    expect(screen.getByText(/3 of 60 test rows sampled/)).toBeInTheDocument();
    expect(screen.getAllByTestId("plotly-stub").length).toBeGreaterThanOrEqual(1);
    // the models comparison table fills after the train
    expect(await screen.findByText("mresp-test000001")).toBeInTheDocument();
    expect(mlMocks.multiresponseModels).toHaveBeenCalled();
  });

  it("renders the predict round trip in the stable target order", async () => {
    const train = trainResponse();
    mlMocks.multiresponseTrain.mockResolvedValue(train);
    mlMocks.multiresponseModels.mockResolvedValue({
      models: [train.model],
      metrics: ["r2"],
    });
    mlMocks.multiresponsePredict.mockResolvedValue({
      model_id: "mresp-test000001",
      target_names: ["response_a", "response_b", "response_c"],
      target_units: {},
      predictions: [
        { response_a: 1.007, response_b: 1.943, response_c: 0.199 },
        { response_a: -0.44, response_b: 0.12, response_c: 0.51 },
      ],
      n_rows: 2,
      note: "predictions arrive in the model's STABLE contract target order",
    });
    render(<MLPage />);
    await userEvent.click(await screen.findByLabelText("x1"));
    await userEvent.click(await screen.findByLabelText("response_a"));
    await userEvent.click(await screen.findByLabelText("response_b"));
    await userEvent.click(await screen.findByLabelText("response_c"));
    await userEvent.click(
      await screen.findByRole("button", { name: /Train the multi-response model/ }),
    );
    await screen.findByText(/the seeded SYNTHETIC demo frame — every number is a model estimate/);
    // the rows textarea + the model id (auto-filled by the train)
    const rowsField = screen.getByLabelText("multiresponse predict rows");
    fireEvent.change(rowsField, {
      target: { value: '[{"x1": 0.5}]' },
    });
    await userEvent.click(screen.getByRole("button", { name: "Predict" }));
    // the prediction table in the contract order
    expect(await screen.findByText(/1\.943/)).toBeInTheDocument();
    expect(screen.getByText(/0\.199/)).toBeInTheDocument();
    expect(mlMocks.multiresponsePredict).toHaveBeenCalledWith({
      model_id: "mresp-test000001",
      rows: [{ x1: 0.5 }],
    });
  });

  it("surfaces the backend's own hard 422 verbatim through the banner", async () => {
    mlMocks.multiresponseTrain.mockRejectedValue(
      new ApiError(
        422,
        "per_target missing handling rides the multi_output wrapper mechanism only",
      ),
    );
    render(<MLPage />);
    await userEvent.click(await screen.findByLabelText("x1"));
    await userEvent.click(await screen.findByLabelText("response_a"));
    await userEvent.click(await screen.findByLabelText("response_b"));
    await userEvent.click(
      await screen.findByRole("button", { name: /Train the multi-response model/ }),
    );
    expect(
      await screen.findByText(
        /per_target missing handling rides the multi_output wrapper mechanism only/,
      ),
    ).toBeInTheDocument();
  });

  it("gates a one-target selection client-side — no request fires", async () => {
    render(<MLPage />);
    await userEvent.click(await screen.findByLabelText("x1"));
    await userEvent.click(await screen.findByLabelText("response_a"));
    const button = await screen.findByRole("button", {
      name: /Train the multi-response model/,
    });
    expect(button).toBeDisabled();
    expect(mlMocks.multiresponseTrain).not.toHaveBeenCalled();
  });

  it("exports the trained model through the joblib artifact lane", async () => {
    const train = trainResponse();
    mlMocks.multiresponseTrain.mockResolvedValue(train);
    mlMocks.multiresponseModels.mockResolvedValue({
      models: [train.model],
      metrics: ["r2"],
    });
    mlMocks.multiresponseExport.mockResolvedValue({
      artifact_id: "art-test00001",
      kind: "model",
      filename: "multiresponse-mresp-test000001.joblib",
      content_type: "application/octet-stream",
      size_bytes: 4096,
      created_at: "2026-11-25T06:00:00Z",
      meta: {},
      digest_sha256: "abc",
    });
    render(<MLPage />);
    await userEvent.click(await screen.findByLabelText("x1"));
    await userEvent.click(await screen.findByLabelText("response_a"));
    await userEvent.click(await screen.findByLabelText("response_b"));
    await userEvent.click(
      await screen.findByRole("button", { name: /Train the multi-response model/ }),
    );
    await screen.findByText(/the seeded SYNTHETIC demo frame — every number is a model estimate/);
    await userEvent.click(
      screen.getByRole("button", { name: /Export the model/ }),
    );
    expect(await screen.findByText(/art-test00001/)).toBeInTheDocument();
    expect(mlMocks.multiresponseExport).toHaveBeenCalledWith(
      "mresp-test000001",
    );
  });
});
