/**
 * part16 §15.2 (Phase 12) — the experiment-run client's own contract.
 *
 * The `/api/v1/runs` surface (the RunStore's query projections)
 * narrowed through the house parse pattern: the lineage-table row
 * field-for-field, the full §15.2 record with every block joined,
 * the honest failed-run shape (the fit-only fields preserved as null,
 * the model artifact absent — never invented), and the ApiSchemaError
 * verdict for drifted payloads. The status/mode unions and the schema
 * pin are the drift guards the Python suite (`tests/api/test_runs.py`)
 * holds on the other side of the wire.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

const getMock = vi.hoisted(() => vi.fn());
vi.mock("@/services/api/client", () => ({
  api: { get: (path: string) => getMock(path) },
}));

import { runsApi } from "@/services/runs/client";
import {
  ApiSchemaError,
  parseExperimentRun,
  parseRunSummary,
} from "@/services/runs/types";

function summaryPayload(): Record<string, unknown> {
  return {
    run_id: "run-4f0a9c21b7d3",
    family: "ml.train",
    mode: "sync",
    status: "completed",
    job_id: null,
    dataset_id: "ds-20250601-0001",
    dataset_name: "Walkthrough classification",
    model_name: "logistic_regression",
    task: "classification",
    metric: 0.9123,
    cv_folds: 5,
    random_seed: 42,
    created_at: "2025-06-01T10:00:00+00:00",
    started_at: "2025-06-01T10:00:01+00:00",
    completed_at: "2025-06-01T10:00:03+00:00",
  };
}

function fullPayload(): Record<string, unknown> {
  return {
    ...summaryPayload(),
    dataset: {
      dataset_id: "ds-20250601-0001",
      name: "Walkthrough classification",
      source_filename: "walkthrough_classification.csv",
      size_bytes: 43883,
      shape: [120, 9],
      target: "activity",
      task: "classification",
      n_features: 8,
      created_at: "2025-06-01T09:00:00+00:00",
    },
    feature_configuration: {
      feature_names: ["molwt", "logp", "tpsa"],
      n_features: 8,
      target: "activity",
    },
    preprocessing: {
      enabled: true,
      pipeline:
        "median imputation -> StandardScaler (non-numeric columns: " +
        "most_frequent imputation + one-hot encoding, handle_unknown=ignore)",
    },
    algorithm: {
      name: "logistic_regression",
      task: "classification",
      toolkit_version: "1.2.240",
    },
    hyperparameters: {
      requested: { C: 0.5 },
      tuned: true,
      best_params: { C: 0.5 },
    },
    random_seed: 42,
    cross_validation: {
      cv_folds: 5,
      tuning: true,
      n_jobs_requested: 1,
      n_jobs_effective: 1,
      parallelism_advisory: {
        requested: 1,
        executed: 1,
        reason: null,
        estimated_bytes: null,
        available_bytes: 6_200_000_000,
      },
    },
    evaluation: {
      cv_scores: [0.91, 0.92, 0.9, 0.93, 0.89],
      cv_mean: 0.9123,
      cv_std: 0.0141,
      primary_metric: "roc_auc_ovo (balanced_accuracy without predict_proba)",
    },
    model_artifact: {
      model_id: "model-20250601-0001",
      name: "logistic_regression",
      task: "classification",
      n_train_rows: 96,
      fit_seconds: 0.21,
      created_at: "2025-06-01T10:00:03+00:00",
      fetch: "GET /api/v1/ml/trained/model-20250601-0001",
    },
    environment: {
      toolkit: "1.2.240",
      python: "3.11.9",
      platform: "Linux-6.5.0-x86_64",
      machine: "x86_64",
      packages: { numpy: "1.26.4", rdkit: null },
      numeric_environment: { OPENBLAS_NUM_THREADS: null },
    },
    logs: {
      job_id: null,
      stage: "completed",
      history: "sync mode (no job history)",
    },
    schema: "pharmaml-experiment-run/1",
  };
}

beforeEach(() => {
  getMock.mockReset();
});

describe("parseRunSummary (the lineage-table row)", () => {
  it("narrows the happy path field-for-field", () => {
    const view = parseRunSummary(summaryPayload(), "runs[0]");

    expect(view.run_id).toBe("run-4f0a9c21b7d3");
    expect(view.family).toBe("ml.train");
    expect(view.mode).toBe("sync");
    expect(view.status).toBe("completed");
    expect(view.job_id).toBeNull();
    expect(view.dataset_id).toBe("ds-20250601-0001");
    expect(view.dataset_name).toBe("Walkthrough classification");
    expect(view.model_name).toBe("logistic_regression");
    expect(view.task).toBe("classification");
    expect(view.metric).toBe(0.9123);
    expect(view.cv_folds).toBe(5);
    expect(view.random_seed).toBe(42);
    expect(view.created_at).toBe("2025-06-01T10:00:00+00:00");
    expect(view.started_at).toBe("2025-06-01T10:00:01+00:00");
    expect(view.completed_at).toBe("2025-06-01T10:00:03+00:00");
    // the error field is optional on the wire — absent stays null
    expect(view.error).toBeNull();
  });

  it("throws ApiSchemaError on drift — a missing run_id, never a silent any", () => {
    const payload = summaryPayload();
    delete payload["run_id"];
    expect(() => parseRunSummary(payload, "runs[0]")).toThrow(ApiSchemaError);
    expect(() => parseRunSummary(42, "runs[0]")).toThrow(ApiSchemaError);
    expect(() => parseRunSummary(null, "runs[0]")).toThrow(ApiSchemaError);
  });

  it("holds the status union — the three honest states pass, a typo refuses", () => {
    for (const status of ["completed", "failed", "cancelled"] as const) {
      const view = parseRunSummary({ ...summaryPayload(), status }, "runs[0]");
      expect(view.status).toBe(status);
    }
    expect(() =>
      parseRunSummary({ ...summaryPayload(), status: "queued" }, "runs[0]"),
    ).toThrow(ApiSchemaError);
    // the mode union rides the same guard
    for (const mode of ["job", "sync"] as const) {
      const view = parseRunSummary({ ...summaryPayload(), mode }, "runs[0]");
      expect(view.mode).toBe(mode);
    }
    expect(() =>
      parseRunSummary({ ...summaryPayload(), mode: "async" }, "runs[0]"),
    ).toThrow(ApiSchemaError);
  });

  it("keeps the error field optional and the failed-run absences honest", () => {
    const view = parseRunSummary(
      {
        ...summaryPayload(),
        status: "failed",
        mode: "job",
        job_id: "job-20250601-0007",
        metric: null,
        error: "n_estimators must be greater than 0, got 0",
      },
      "runs[0]",
    );
    expect(view.status).toBe("failed");
    expect(view.job_id).toBe("job-20250601-0007");
    expect(view.metric).toBeNull();
    expect(view.error).toBe("n_estimators must be greater than 0, got 0");
  });
});

describe("parseExperimentRun (the full §15.2 record)", () => {
  it("joins every §15.2 block field-for-field", () => {
    const run = parseExperimentRun(fullPayload(), "run run-4f0a9c21b7d3");

    // the summary skeleton rides through untouched
    expect(run.run_id).toBe("run-4f0a9c21b7d3");
    expect(run.metric).toBe(0.9123);
    // dataset identifiers and versions
    expect(run.dataset.dataset_id).toBe("ds-20250601-0001");
    expect(run.dataset.name).toBe("Walkthrough classification");
    expect(run.dataset.source_filename).toBe("walkthrough_classification.csv");
    expect(run.dataset.size_bytes).toBe(43883);
    expect(run.dataset.shape).toEqual([120, 9]);
    expect(run.dataset.target).toBe("activity");
    expect(run.dataset.n_features).toBe(8);
    // feature configuration
    expect(run.feature_configuration.feature_names).toEqual([
      "molwt",
      "logp",
      "tpsa",
    ]);
    // preprocessing pipeline
    expect(run.preprocessing.enabled).toBe(true);
    expect(run.preprocessing.pipeline).toContain("imputation");
    // algorithm + toolkit version
    expect(run.algorithm.name).toBe("logistic_regression");
    expect(run.algorithm.toolkit_version).toBe("1.2.240");
    // hyperparameters (requested AND tuned)
    expect(run.hyperparameters.requested).toEqual({ C: 0.5 });
    expect(run.hyperparameters.tuned).toBe(true);
    expect(run.hyperparameters.best_params).toEqual({ C: 0.5 });
    // random seed + cross-validation configuration
    expect(run.random_seed).toBe(42);
    expect(run.cross_validation.cv_folds).toBe(5);
    expect(run.cross_validation.n_jobs_requested).toBe(1);
    expect(run.cross_validation.n_jobs_effective).toBe(1);
    expect(run.cross_validation.parallelism_advisory?.available_bytes).toBe(
      6_200_000_000,
    );
    // evaluation metrics
    expect(run.evaluation.cv_scores).toEqual([0.91, 0.92, 0.9, 0.93, 0.89]);
    expect(run.evaluation.cv_mean).toBe(0.9123);
    expect(run.evaluation.primary_metric).toBe(
      "roc_auc_ovo (balanced_accuracy without predict_proba)",
    );
    // model artifact reference + the fetch hint
    expect(run.model_artifact?.model_id).toBe("model-20250601-0001");
    expect(run.model_artifact?.fetch).toBe(
      "GET /api/v1/ml/trained/model-20250601-0001",
    );
    // environment information
    expect(run.environment.toolkit).toBe("1.2.240");
    expect(run.environment.python).toBe("3.11.9");
    expect(run.environment.platform).toBe("Linux-6.5.0-x86_64");
    expect(run.environment.packages["numpy"]).toBe("1.26.4");
    expect(run.environment.packages["rdkit"]).toBeNull();
    // job reference
    expect(run.logs.history).toBe("sync mode (no job history)");
    // the schema pin
    expect(run.schema).toBe("pharmaml-experiment-run/1");
  });

  it("throws ApiSchemaError when a block drifts — the dataset block is required", () => {
    const payload = fullPayload();
    delete payload["dataset"];
    expect(() => parseExperimentRun(payload, "run x")).toThrow(ApiSchemaError);

    // the schema pin refuses a renamed record vocabulary
    expect(() =>
      parseExperimentRun(
        { ...fullPayload(), schema: "pharmaml-experiment-run/2" },
        "run x",
      ),
    ).toThrow(ApiSchemaError);
    // a non-object payload never narrows silently
    expect(() => parseExperimentRun("run", "run x")).toThrow(ApiSchemaError);
  });

  it("keeps the failed-run record honest — the fit-only fields absent, never invented", () => {
    const run = parseExperimentRun(
      {
        ...fullPayload(),
        status: "failed",
        mode: "job",
        job_id: "job-20250601-0007",
        metric: null,
        error: "n_estimators must be greater than 0",
        hyperparameters: {
          requested: { n_estimators: 0 },
          tuned: false,
          best_params: null,
        },
        cross_validation: {
          cv_folds: 5,
          tuning: false,
          n_jobs_requested: 1,
          n_jobs_effective: null,
          parallelism_advisory: null,
        },
        evaluation: {
          cv_scores: null,
          cv_mean: null,
          cv_std: null,
          primary_metric: null,
        },
        model_artifact: null,
        logs: {
          job_id: "job-20250601-0007",
          stage: "failed",
          history: "GET /api/v1/jobs/job-20250601-0007",
        },
      },
      "run x",
    );

    expect(run.status).toBe("failed");
    expect(run.error).toBe("n_estimators must be greater than 0");
    expect(run.hyperparameters.best_params).toBeNull();
    expect(run.cross_validation.parallelism_advisory).toBeNull();
    expect(run.evaluation.cv_mean).toBeNull();
    expect(run.model_artifact).toBeNull();
    expect(run.logs.history).toBe("GET /api/v1/jobs/job-20250601-0007");
  });
});

describe("runsApi (the REST surface)", () => {
  it("lists the lineage table at the pinned path and narrows every row", async () => {
    getMock.mockResolvedValue({
      runs: [summaryPayload(), { ...summaryPayload(), run_id: "run-2" }],
      count: 2,
    });

    const listing = await runsApi.list();

    expect(getMock).toHaveBeenCalledWith("/api/v1/runs");
    expect(listing.runs).toHaveLength(2);
    expect(listing.count).toBe(2);
    expect(listing.runs[0]?.run_id).toBe("run-4f0a9c21b7d3");
    expect(listing.runs[1]?.run_id).toBe("run-2");
  });

  it("builds the filter query string from every optional parameter", async () => {
    getMock.mockResolvedValue({ runs: [], count: 0 });

    await runsApi.list({
      family: "ml.train",
      datasetId: "ds-20250601-0001",
      model: "ridge",
      status: "failed",
      limit: 5,
    });

    expect(getMock).toHaveBeenCalledWith(
      "/api/v1/runs?family=ml.train&dataset_id=ds-20250601-0001&model=ridge&status=failed&limit=5",
    );
    // an omitted parameter stays off the wire
    await runsApi.list({ limit: 1 });
    expect(getMock).toHaveBeenCalledWith("/api/v1/runs?limit=1");
  });

  it("tolerates the empty ledger (the honest zero, never a fabricated row)", async () => {
    getMock.mockResolvedValue({ runs: [], count: 0 });

    const listing = await runsApi.list();

    expect(listing.runs).toEqual([]);
    expect(listing.count).toBe(0);
  });

  it("surfaces a drifted listing row as ApiSchemaError, never silent any", async () => {
    getMock.mockResolvedValue({ runs: [{ family: "ml.train" }], count: 1 });
    await expect(runsApi.list()).rejects.toBeInstanceOf(ApiSchemaError);
  });

  it("fetches the full §15.2 record with the encoded run_id", async () => {
    getMock.mockResolvedValue(fullPayload());

    const run = await runsApi.detail("run-4f0a9c21b7d3");

    expect(getMock).toHaveBeenCalledWith("/api/v1/runs/run-4f0a9c21b7d3");
    expect(run.dataset.name).toBe("Walkthrough classification");
    expect(run.schema).toBe("pharmaml-experiment-run/1");

    // the run_id rides the encodeURIComponent lane like every house id
    getMock.mockResolvedValue(fullPayload());
    await runsApi.detail("run needs encoding");
    expect(getMock).toHaveBeenCalledWith(
      "/api/v1/runs/run%20needs%20encoding",
    );
  });

  it("surfaces a drifted detail payload as ApiSchemaError", async () => {
    getMock.mockResolvedValue({ nope: true });
    await expect(runsApi.detail("run-x")).rejects.toBeInstanceOf(ApiSchemaError);
  });
});
