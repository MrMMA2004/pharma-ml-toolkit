import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import { ApiError } from "@/services/api/client";
import type { JobView } from "@/services/jobs/types";

/**
 * part13 Phase 6 — the §8 Workflow Builder lane of the Machine
 * Learning workspace as a complete component-level user flow:
 *
 *   the registry line    — the stage renders the engine block's own
 *                          vocabulary summary (the step count, the
 *                          closed port-kind list, the kind-equality
 *                          rule) off the mocked steps answer;
 *   the saved templates  — the template select seeds the editable
 *                          spec from the REAL templates answer; the
 *                          chain renders the registry's own step
 *                          titles with the honest leakage note (the
 *                          split BEFORE selection — the toolkit's own
 *                          rule, stated in the metadata);
 *   the validate gate    — the editable spec becomes the honest
 *                          request; the answer renders the verdict,
 *                          the executor's topological order, and the
 *                          canonical digest;
 *   the sync run        — the honest request (spec + seed +
 *                          n_synthetic + mode) executes and the REAL
 *                          parsed run record renders — the per-node
 *                          states, the artifact lineage with its
 *                          honest digest_kinds, the reproducibility
 *                          block with the determinism scope;
 *   typed errors         — the backend's own hard 422 surfaces
 *                          verbatim through the banner;
 *   client-side gating   — a negative seed never leaves the page
 *                          (validated BEFORE any request fires);
 *   the job relay        — the job-mode COMPLETED result relays
 *                          through the authoritative REST lane into
 *                          the run record (§13, never the WS payload
 *                          alone);
 *   the runs ledger      — the bounded in-session execution history
 *                          renders with the honest lifetime note.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers
 * (parseWorkflowStepsResponse / parseWorkflowTemplatesResponse /
 * parseWorkflowValidateResponse / parseWorkflowRunResponse /
 * parseWorkflowRunRecord / parseWorkflowRunsResponse — the REAL
 * ones, spread from the actual module), and the DOM are the real
 * ones.
 */

const jobState = vi.hoisted(() => ({
  snapshots: {} as Record<string, { view: Record<string, unknown> }>,
  cancels: [] as string[],
}));
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
  workflowSteps: vi.fn(),
  workflowTemplates: vi.fn(),
  workflowValidate: vi.fn(),
  workflowRun: vi.fn(),
  workflowRuns: vi.fn(),
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
      workflowSteps: mlMocks.workflowSteps,
      workflowTemplates: mlMocks.workflowTemplates,
      workflowValidate: mlMocks.workflowValidate,
      workflowRun: mlMocks.workflowRun,
      workflowRuns: mlMocks.workflowRuns,
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

// -- the §8 engine's own vocabulary, stubbed field-for-field off the
// route's real shape ----------------------------------------------

const WF_PORT_KINDS = [
  "records",
  "dataframe",
  "split",
  "model",
  "metrics",
  "ad_report",
  "shap_summary",
  "report",
];

function stepStub(
  step_id: string,
  title: string,
  backend: string,
  workspace: string,
  inputs: { name: string; kind: string; required?: boolean }[],
  outputs: { name: string; kind: string }[],
  parameters: { name: string; default: unknown }[] = [],
) {
  return {
    step_id,
    title,
    description: "",
    backend,
    workspace,
    inputs: inputs.map((p) => ({
      name: p.name,
      kind: p.kind,
      required: p.required ?? true,
      description: "",
    })),
    outputs: outputs.map((p) => ({ name: p.name, kind: p.kind, description: "" })),
    parameters: parameters.map((p) => ({
      name: p.name,
      default: p.default,
      description: "",
    })),
    notes: "",
  };
}

const WF_STEPS = {
  engine: {
    layer: "pharmaml.workflow (the lightweight in-tree engine)",
    decision:
      "the existing JobManager + capability vocabulary support the " +
      "lightweight implementation -- the §8 assessment's own " +
      "conclusion (docs/ARCHITECTURE_REVIEW.md); no external " +
      "workflow engine adopted",
    port_kinds: WF_PORT_KINDS,
    job_integration:
      "job-mode runs ride the existing JobManager: progress per " +
      "node, cooperative cancellation between nodes, the SQLite job " +
      "history",
  },
  steps: [
    stepStub(
      "dataset.import",
      "Dataset Import",
      "pandas.DataFrame (the toolkit's own frame layer)",
      "datasets",
      [{ name: "records", kind: "records" }],
      [{ name: "dataset", kind: "dataframe" }],
      [
        { name: "smiles_column", default: "smiles" },
        { name: "target_column", default: "pIC50" },
      ],
    ),
    stepStub(
      "chem.standardize",
      "Standardization",
      "pharmaml.chem.csv_standardize.standardize_smiles",
      "datasets",
      [{ name: "dataset", kind: "dataframe" }],
      [
        { name: "dataset", kind: "dataframe" },
        { name: "standardization", kind: "metrics" },
      ],
      [{ name: "smiles_column", default: "smiles" }],
    ),
    stepStub(
      "chem.descriptors",
      "Descriptor Generation",
      "pharmaml.chem.descriptors.compute_descriptors",
      "datasets",
      [{ name: "dataset", kind: "dataframe" }],
      [{ name: "dataset", kind: "dataframe" }],
      [
        { name: "descriptor_set", default: "rdkit_all" },
        { name: "smiles_column", default: "smiles" },
      ],
    ),
    stepStub(
      "ml.split",
      "Data Split",
      "pharmaml.ml.preprocessing.make_train_test_split",
      "machine-learning",
      [{ name: "dataset", kind: "dataframe" }],
      [{ name: "split", kind: "split" }],
      [
        { name: "target_column", default: "pIC50" },
        { name: "test_size", default: 0.2 },
        { name: "random_state", default: 42 },
      ],
    ),
    stepStub(
      "ml.feature_selection",
      "Feature Selection",
      "pharmaml.ml.preprocessing.select_kbest_union",
      "machine-learning",
      [{ name: "split", kind: "split" }],
      [{ name: "split", kind: "split" }],
      [
        { name: "k", default: 8 },
        {
          name: "methods",
          default: ["f_regression", "mutual_info_regression"],
        },
      ],
    ),
    stepStub(
      "ml.train",
      "Model Training",
      "pharmaml.ml.models.build_model",
      "machine-learning",
      [{ name: "split", kind: "split" }],
      [{ name: "model", kind: "model" }],
      [
        { name: "model", default: "random_forest" },
        { name: "task", default: "regression" },
        { name: "random_state", default: 42 },
      ],
    ),
    stepStub(
      "ml.evaluate",
      "Model Evaluation",
      "pharmaml.ml.evaluation.evaluate_model",
      "machine-learning",
      [
        { name: "split", kind: "split" },
        { name: "model", kind: "model" },
      ],
      [{ name: "metrics", kind: "metrics" }],
      [{ name: "task", default: "regression" }],
    ),
    stepStub(
      "ml.applicability",
      "Applicability Domain",
      "pharmaml.ml.applicability.leverage_williams",
      "machine-learning",
      [
        { name: "split", kind: "split" },
        { name: "model", kind: "model" },
      ],
      [{ name: "ad_report", kind: "ad_report" }],
      [{ name: "residual_limit", default: 3.0 }],
    ),
    stepStub(
      "ml.shap",
      "SHAP",
      "pharmaml.ml.explain.shap_summary",
      "machine-learning",
      [
        { name: "split", kind: "split" },
        { name: "model", kind: "model" },
      ],
      [{ name: "shap_summary", kind: "shap_summary" }],
    ),
    stepStub(
      "report.generate",
      "Report Generation",
      "pharmaml.viz.report.save_pdf_report",
      "visualization",
      [
        { name: "metrics", kind: "metrics" },
        { name: "ad_report", kind: "ad_report", required: false },
        { name: "shap_summary", kind: "shap_summary", required: false },
      ],
      [{ name: "report", kind: "report" }],
      [{ name: "title", default: "PharmaML workflow report" }],
    ),
  ],
};

// -- the saved templates, stubbed off templates.py's own shapes ----

function guidedQsarSpec() {
  return {
    name: "guided-qsar",
    description:
      "part13 §8's own example chain, wired over the REAL backends",
    nodes: [
      {
        node_id: "import",
        step_id: "dataset.import",
        params: { smiles_column: "smiles", target_column: "pIC50" },
        wiring: {},
      },
      {
        node_id: "standardize",
        step_id: "chem.standardize",
        params: { smiles_column: "smiles" },
        wiring: {
          dataset: { node: "import", port: "dataset" },
        },
      },
      {
        node_id: "descriptors",
        step_id: "chem.descriptors",
        params: { descriptor_set: "rdkit_all", smiles_column: "smiles" },
        wiring: {
          dataset: { node: "standardize", port: "dataset" },
        },
      },
      {
        node_id: "split",
        step_id: "ml.split",
        params: { target_column: "pIC50", test_size: 0.25 },
        wiring: {
          dataset: { node: "descriptors", port: "dataset" },
        },
      },
      {
        node_id: "selection",
        step_id: "ml.feature_selection",
        params: { k: 8 },
        wiring: { split: { node: "split", port: "split" } },
      },
      {
        node_id: "train",
        step_id: "ml.train",
        params: { model: "random_forest", task: "regression" },
        wiring: { split: { node: "selection", port: "split" } },
      },
      {
        node_id: "evaluate",
        step_id: "ml.evaluate",
        params: { task: "regression" },
        wiring: {
          split: { node: "selection", port: "split" },
          model: { node: "train", port: "model" },
        },
      },
      {
        node_id: "applicability",
        step_id: "ml.applicability",
        params: { residual_limit: 3.0 },
        wiring: {
          split: { node: "selection", port: "split" },
          model: { node: "train", port: "model" },
        },
      },
      {
        node_id: "shap",
        step_id: "ml.shap",
        params: {},
        wiring: {
          split: { node: "selection", port: "split" },
          model: { node: "train", port: "model" },
        },
      },
      {
        node_id: "report",
        step_id: "report.generate",
        params: { title: "Guided QSAR workflow report" },
        wiring: {
          metrics: { node: "evaluate", port: "metrics" },
          ad_report: { node: "applicability", port: "ad_report" },
          shap_summary: { node: "shap", port: "shap_summary" },
        },
      },
    ],
    metadata: {
      lineage: "part13 §8 example chain",
      leakage_note:
        "feature selection is fitted on the training split only " +
        "and applied to the held-out frame as a pure column subset " +
        "-- the look.ipynb fix, kept as the contract even though " +
        "§8's illustrative order lists selection first",
    },
  };
}

function minimalRegressionSpec() {
  return {
    name: "minimal-regression",
    description: "The six-node core",
    nodes: [
      {
        node_id: "import",
        step_id: "dataset.import",
        params: { smiles_column: "smiles", target_column: "pIC50" },
        wiring: {},
      },
      {
        node_id: "descriptors",
        step_id: "chem.descriptors",
        params: { descriptor_set: "rdkit_all", smiles_column: "smiles" },
        wiring: { dataset: { node: "import", port: "dataset" } },
      },
      {
        node_id: "split",
        step_id: "ml.split",
        params: { target_column: "pIC50", test_size: 0.25 },
        wiring: { dataset: { node: "descriptors", port: "dataset" } },
      },
      {
        node_id: "train",
        step_id: "ml.train",
        params: { model: "random_forest", task: "regression" },
        wiring: { split: { node: "split", port: "split" } },
      },
      {
        node_id: "evaluate",
        step_id: "ml.evaluate",
        params: { task: "regression" },
        wiring: {
          split: { node: "split", port: "split" },
          model: { node: "train", port: "model" },
        },
      },
      {
        node_id: "report",
        step_id: "report.generate",
        params: { title: "Minimal regression workflow report" },
        wiring: { metrics: { node: "evaluate", port: "metrics" } },
      },
    ],
    metadata: { lineage: "the minimal core chain" },
  };
}

/** The serialisable spec shape (engine.py's own to_dict form). */
interface SpecStub {
  name: string;
  description: string;
  nodes: {
    node_id: string;
    step_id: string;
    params: Record<string, unknown>;
    wiring: Record<string, { node: string; port: string } | undefined>;
  }[];
  metadata: Record<string, unknown>;
}

function templateEntry(template_id: string, spec: SpecStub) {
  return {
    template_id,
    name: spec.name,
    description: spec.description,
    node_count: spec.nodes.length,
    nodes: spec.nodes.map((n) => ({
      node_id: n.node_id,
      step_id: n.step_id,
    })),
    metadata: spec.metadata,
    spec,
  };
}

const WF_TEMPLATES = {
  templates: [
    templateEntry("guided-qsar", guidedQsarSpec()),
    templateEntry("minimal-regression", minimalRegressionSpec()),
  ],
};

// -- the run record, stubbed off engine.py's public() shape --------

function runRecord() {
  return {
    run_id: "wf-run-abc123def456",
    workflow_name: "guided-qsar",
    spec_digest: "3f2a".repeat(16),
    state: "COMPLETED",
    seed: 42,
    started_at: "2026-11-23T00:00:00Z",
    completed_at: "2026-11-23T00:00:05Z",
    reproducibility: {
      toolkit_version: "1.2.240",
      python: "3.12.16",
      implementation: "CPython",
      platform: "Linux-6.8.0-x86_64",
      seed: 42,
      spec_digest: "3f2a".repeat(16),
      determinism_note:
        "seeded splits and estimators reproduce; the f_regression " +
        "selector is deterministic while the mutual-info member of " +
        "the default union pair is unseeded in the backend (its " +
        "selected column set can vary run-to-run on the same seed); " +
        "the report PDF embeds fpdf's own creation timestamp so its " +
        "bytes vary run-to-run even at the same seed; wall-clock " +
        "durations vary and are excluded from every digest",
    },
    node_counts: { COMPLETED: 10, FAILED: 0, SKIPPED: 0, CANCELLED: 0 },
    nodes: [
      {
        node_id: "import",
        step_id: "dataset.import",
        state: "COMPLETED",
        started_at: "2026-11-23T00:00:00Z",
        duration_ms: 1.2,
        error: null,
        skipped_reason: null,
        artifacts: [
          {
            node: "import",
            port: "dataset",
            kind: "dataframe",
            digest: "ab12".repeat(16),
            digest_kind: "dataframe-shape-rowhash",
            lineage: [],
          },
        ],
      },
      {
        node_id: "train",
        step_id: "ml.train",
        state: "COMPLETED",
        started_at: "2026-11-23T00:00:02Z",
        duration_ms: 812.5,
        error: null,
        skipped_reason: null,
        artifacts: [
          {
            node: "train",
            port: "model",
            kind: "model",
            digest: "cd34".repeat(16),
            digest_kind: "estimator-params",
            lineage: [
              { node: "import", port: "dataset" },
              { node: "standardize", port: "dataset" },
              { node: "descriptors", port: "dataset" },
              { node: "split", port: "split" },
              { node: "selection", port: "split" },
            ],
          },
        ],
      },
      {
        node_id: "report",
        step_id: "report.generate",
        state: "COMPLETED",
        started_at: "2026-11-23T00:00:04Z",
        duration_ms: 240.0,
        error: null,
        skipped_reason: null,
        artifacts: [
          {
            node: "report",
            port: "report",
            kind: "report",
            digest: "ef56".repeat(16),
            digest_kind: "json",
            lineage: [
              { node: "train", port: "model" },
              { node: "evaluate", port: "metrics" },
            ],
          },
        ],
      },
    ],
  };
}

function completedJobView(jobId: string): JobView {
  return {
    job_id: jobId,
    operation: "workflow.run",
    workspace: "workflow",
    state: "COMPLETED",
    progress: 100,
    stage: "node 10/10: report COMPLETED",
    error: null,
    created_at: "2026-11-23T00:00:00Z",
    started_at: "2026-11-23T00:00:01Z",
    completed_at: "2026-11-23T00:00:05Z",
    configuration: {},
    cancellation: "none",
    result_ready: true,
  } as unknown as JobView;
}

// -- the neighbour registry stubs (the same contract every lane
// suite carries so the page's mount effects never crash) ----------

const MO_REGISTRY_MIN = {
  order: {
    convention: "the optuna convention — every objective is MINIMIZED",
    dominates: "a dominates b iff a is at least as good everywhere",
    pareto_indices: "the non-dominated set, ascending input order",
    complexity: "pairwise O(n^2 * k) at study scale",
  },
  indicator: {
    family: "the exact 2-D dominated hypervolume (the staircase sweep)",
    reference_contract: "the WFG contract",
    report_reference: "the observed nadir padded by 10 %",
    curve: "the per-trial indicator under that same FIXED reference",
    two_d_only: "the exact sweep is 2-D only — null beyond, never fabricated",
  },
  spread: {
    family: "NSGA-II's crowding distance (boundaries +inf)",
    rule: "the two extremes of every objective gain +inf",
    json_note: "boundary members' +inf reads as null in JSON",
  },
  decomposition: {
    family: "the MOEA/D Tchebycheff decomposition",
    form: "max_j w_j |v_j - ideal_j|, minimized",
    rule: "a zero weight is the corner subproblem",
  },
  study: {
    samplers: ["random", "tpe"],
    samplers_note: "the decomposition drives BOTH existing samplers",
    weight_modes: { spread: "Das & Dennis", random: "seeded uniform draws" },
    n_objectives: "a Pareto study needs at least two objectives",
    drift_note: "the ideal point advances as the front does",
  },
  tunable_objectives: {
    ledger: {
      score: { cross_validate_key: "test_score", sign: -1 },
      fit_seconds: { cross_validate_key: "fit_time", sign: 1 },
      score_seconds: { cross_validate_key: "score_time", sign: 1 },
    },
    note: "'score' = the negated mean CV score",
    bounds: "2..3 distinct entries per request",
  },
  report: {
    schema: ["objective_names", "front", "crowding", "ref", "hypervolume"],
    best_for: "the front member minimizing one named objective",
    prune: "up to keep front members by full-front crowding",
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
    model_note: "the model is any sklearn registry name",
  },
  dataset: {
    geometry: { n_samples: 80, n_features: 6 },
    label: "SYNTHETIC — the route's own seeded stand-in",
  },
  provenance: {
    backend: "pharmaml.ml.multiobjective",
    source: "",
    n_repositories: 3,
    transfers_exactly: "",
    stays_upstream: "",
    shipped_checkpoints: 0,
    shipped_checkpoints_note: "",
    dataset: "",
    disclaimer: "",
  },
  optimize_url: "/api/v1/multiobjective/optimize",
  note: "sync-only light compute",
};

beforeEach(() => {
  vi.clearAllMocks();
  jobState.snapshots = {};
  jobState.cancels = [];
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
    metrics: { keys: ["rmse", "mae", "r2"], threshold: 6.0, note: "" },
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
  mlMocks.drugResponseRegistry.mockResolvedValue({
    model: {
      family: "CaDRReS-style bilinear factorisation with a sigmoid link",
      formulation: "gamma = X.W; the score is the sigmoid of the bilinear form",
      direction: "the score is the sigmoid-linked response",
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
  mlMocks.multiobjectiveRegistry.mockResolvedValue(MO_REGISTRY_MIN);
  // the §8 lane's own mount answers
  mlMocks.workflowSteps.mockResolvedValue(WF_STEPS);
  mlMocks.workflowTemplates.mockResolvedValue(WF_TEMPLATES);
  mlMocks.workflowValidate.mockReset();
  mlMocks.workflowRun.mockReset();
  mlMocks.workflowRuns.mockReset();
});

describe("the §8 Workflow Builder lane of the Machine Learning workspace", () => {
  it("renders the stage with the engine block's own vocabulary line", async () => {
    render(<MLPage />);
    await waitFor(() => {
      expect(screen.getByText("Workflow Builder")).toBeInTheDocument();
    });
    // the registry's own numbers, never re-declared
    await waitFor(() => {
      expect(
        screen.getByText(/The registry: 10 steps · 8 port kinds/),
      ).toBeInTheDocument();
    });
    // the closed port-kind vocabulary renders verbatim
    expect(
      screen.getByText(/records · dataframe · split · model · metrics · ad_report · shap_summary · report/),
    ).toBeInTheDocument();
    // the §8 assessment's own decision text rides the line
    expect(
      screen.getByText(/no external engine adopted/),
    ).toBeInTheDocument();
    expect(mlMocks.workflowSteps).toHaveBeenCalledTimes(1);
  });

  it("seeds the editable spec from the saved template and renders the chain with the honest leakage note", async () => {
    render(<MLPage />);
    // the select's options come off the templates answer
    const templateSelect = await screen.findByLabelText("workflow template");
    const options = Array.from(templateSelect.querySelectorAll("option")).map(
      (option) => option.value,
    );
    expect(options).toEqual(["guided-qsar", "minimal-regression"]);
    // the first entry seeds the construct canvas
    const specField = screen.getByLabelText("workflow spec") as HTMLTextAreaElement;
    expect(specField.value).toContain('"node_id": "import"');
    expect(specField.value).toContain('"step_id": "report.generate"');
    // the chain renders the registry's own step titles
    expect(
      screen.getByText(/import: Dataset Import → standardize: Standardization/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/report: Report Generation/),
    ).toBeInTheDocument();
    // the honest leakage note — the toolkit's own rule, stated
    expect(
      screen.getByText(/the honest leakage reorder: the split runs BEFORE feature selection/),
    ).toBeInTheDocument();
    expect(mlMocks.workflowTemplates).toHaveBeenCalledTimes(1);
  });

  it("validates the spec as edited and renders the verdict, the order, and the digest", async () => {
    const user = userEvent.setup();
    mlMocks.workflowValidate.mockResolvedValue({
      valid: true,
      errors: [],
      topological_order: [
        "import",
        "standardize",
        "descriptors",
        "split",
        "selection",
        "train",
        "evaluate",
        "applicability",
        "shap",
        "report",
      ],
      spec_digest: "3f2a".repeat(16),
      node_count: 10,
    });
    render(<MLPage />);
    await user.click(
      await screen.findByRole("button", { name: "Validate the spec" }),
    );
    await waitFor(() => {
      expect(mlMocks.workflowValidate).toHaveBeenCalledTimes(1);
    });
    // the honest request: the spec exactly as the canvas carries it
    expect(mlMocks.workflowValidate).toHaveBeenCalledWith(
      JSON.parse(JSON.stringify(guidedQsarSpec())),
    );
    await waitFor(() => {
      expect(screen.getByText(/valid — 10 nodes/)).toBeInTheDocument();
    });
    expect(
      screen.getByText(/the executor's order: import → standardize → descriptors/),
    ).toBeInTheDocument();
    expect(screen.getByText(/digest 3f2a3f2a3f2a3f2a/)).toBeInTheDocument();
    expect(
      screen.getByText(/the canonical digest is the spec's identity/),
    ).toBeInTheDocument();
  });

  it("runs the sync round trip and renders the REAL parsed run record", async () => {
    const user = userEvent.setup();
    mlMocks.workflowRun.mockResolvedValue({
      mode: "sync",
      run: runRecord(),
    });
    render(<MLPage />);
    await user.click(
      await screen.findByRole("button", { name: "Run the workflow" }),
    );
    await waitFor(() => {
      expect(mlMocks.workflowRun).toHaveBeenCalledTimes(1);
    });
    // the honest request: the spec as edited + the route's own dials
    expect(mlMocks.workflowRun).toHaveBeenCalledWith({
      spec: JSON.parse(JSON.stringify(guidedQsarSpec())),
      seed: 42,
      n_synthetic: 30,
      mode: "sync",
    });
    // the run record renders — the summary line, the per-node table,
    // the lineage, the reproducibility block
    await waitFor(() => {
      expect(
        screen.getByText(/run wf-run-abc123def456 · guided-qsar · COMPLETED/),
      ).toBeInTheDocument();
    });
    expect(screen.getByText(/10 completed \/ 0 failed/)).toBeInTheDocument();
    expect(
      screen.getByText(/model \(model, estimator-params, 5 upstream\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/report \(report, json, 2 upstream\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/reproducibility — toolkit 1\.2\.240 · python 3\.12\.16/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/the mutual-info member of the default union pair is unseeded/),
    ).toBeInTheDocument();
  });

  it("surfaces the backend's own hard 422 verbatim through the banner", async () => {
    const user = userEvent.setup();
    mlMocks.workflowRun.mockRejectedValue(
      new ApiError(422, {
        message: "the workflow failed validation",
        errors: [
          "node 'train': port 'split' needs 'split' but 'import'.dataset produces 'dataframe'",
        ],
      }),
    );
    render(<MLPage />);
    await user.click(
      await screen.findByRole("button", { name: "Run the workflow" }),
    );
    await waitFor(() => {
      expect(
        screen.getByText(/API 422/),
      ).toBeInTheDocument();
    });
    // the backend's own detail payload, verbatim
    expect(
      screen.getByText(/the workflow failed validation/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/port 'split' needs 'split' but 'import'\.dataset produces 'dataframe'/),
    ).toBeInTheDocument();
  });

  it("gates a negative seed client-side — the request never leaves the page", async () => {
    const user = userEvent.setup();
    render(<MLPage />);
    const seedField = await screen.findByLabelText("workflow seed");
    fireEvent.change(seedField, { target: { value: "-1" } });
    await user.click(
      screen.getByRole("button", { name: "Run the workflow" }),
    );
    // the route's own bound, stated before any request fires
    await waitFor(() => {
      expect(
        screen.getByText(
          /the seed must lie in 0\.\.2147483647 \(the route's own bound\)/,
        ),
      ).toBeInTheDocument();
    });
    expect(mlMocks.workflowRun).not.toHaveBeenCalled();
  });

  it("relays the job-mode COMPLETED result through the authoritative REST lane", async () => {
    const user = userEvent.setup();
    mlMocks.workflowRun.mockResolvedValue({
      mode: "job",
      job_id: "job-wf-1",
      job_state: "RUNNING",
      note: "poll /api/v1/jobs/{job_id} for progress",
    });
    jobState.snapshots["job-wf-1"] = {
      view: completedJobView("job-wf-1") as unknown as Record<string, unknown>,
    };
    jobsMocks.result.mockResolvedValue({ result: runRecord() });
    render(<MLPage />);
    // switch to job mode — the EXISTING JobManager lane
    await user.selectOptions(
      await screen.findByLabelText("workflow run mode"),
      "job",
    );
    await user.click(
      screen.getByRole("button", { name: "Run the workflow" }),
    );
    await waitFor(() => {
      expect(mlMocks.workflowRun).toHaveBeenCalledWith(
        expect.objectContaining({ mode: "job", seed: 42, n_synthetic: 30 }),
      );
    });
    // the COMPLETED job's result relay lands the run record
    await waitFor(() => {
      expect(jobsMocks.result).toHaveBeenCalledWith("job-wf-1");
    });
    await waitFor(() => {
      expect(
        screen.getByText(/run wf-run-abc123def456 · guided-qsar · COMPLETED/),
      ).toBeInTheDocument();
    });
  });

  it("renders the bounded runs ledger with the honest lifetime note", async () => {
    const user = userEvent.setup();
    mlMocks.workflowRuns.mockResolvedValue({
      runs: [
        {
          run_id: "wf-run-abc123def456",
          workflow_name: "guided-qsar",
          state: "COMPLETED",
          seed: 42,
          started_at: "2026-11-23T00:00:00Z",
          completed_at: "2026-11-23T00:00:05Z",
          node_counts: { COMPLETED: 10, FAILED: 0, SKIPPED: 0, CANCELLED: 0 },
        },
      ],
      note:
        "the in-session ledger lives as long as the process; job-mode " +
        "runs persist through the JobManager's own SQLite history",
    });
    render(<MLPage />);
    await user.click(
      await screen.findByRole("button", { name: "Refresh the runs ledger" }),
    );
    await waitFor(() => {
      expect(mlMocks.workflowRuns).toHaveBeenCalledTimes(1);
    });
    await waitFor(() => {
      expect(
        screen.getByText(/The execution history \(1 runs —/),
      ).toBeInTheDocument();
    });
    // the honest lifetime note rides the ledger
    expect(
      screen.getByText(/the in-session ledger lives as long as the process/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/10 completed \/ 0 failed \/ 0 skipped \/ 0 cancelled/),
    ).toBeInTheDocument();
  });
});
