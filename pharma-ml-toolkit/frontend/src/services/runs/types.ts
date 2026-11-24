/**
 * The experiment-run DTOs — part16 §15.2 (the Phase-12 frontend lane).
 *
 * Field-for-field mirrors of the backend's `ExperimentRunRecord.summary()`
 * and `.view()` projections (`pharmaml.api.runs`) plus the route's
 * `{"runs": …, "count": n}` wrapper — the Python drift guards in
 * `tests/api/test_runs.py` pin the vocabulary, so a backend rename turns
 * that suite red before the frontend drifts.
 *
 * The strictness is on the required skeleton (the fields every record
 * builder sets concretely); the honestly-absent fields — the fit-only
 * numbers on a failed run, the job id in sync mode, the model artifact
 * before any fit landed — ride through as null and render as the honest
 * dash, never fabricated.
 */

/** The run store's three honest terminal states (mirrors `routes/runs.py`). */
export const RUN_STATUSES = [
  "completed",
  "failed",
  "cancelled",
] as const;

export type RunStatus = (typeof RUN_STATUSES)[number];

/** The two execution lanes that record runs (mirrors `routes/ml.py`). */
export const RUN_MODES = ["job", "sync"] as const;

export type RunMode = (typeof RUN_MODES)[number];

/** `GET /api/v1/runs` — one lineage-table row (`summary()`). */
export interface RunSummary {
  readonly run_id: string;
  readonly family: string;
  readonly mode: RunMode;
  readonly status: RunStatus;
  readonly job_id: string | null;
  readonly dataset_id: string;
  readonly dataset_name: string;
  readonly model_name: string;
  readonly task: string;
  /** The CV mean of the primary metric (null on a failed run). */
  readonly metric: number | null;
  readonly cv_folds: number;
  readonly random_seed: number;
  readonly created_at: string;
  readonly started_at: string;
  readonly completed_at: string;
  /** Present only on failed/cancelled runs (the typed error). */
  readonly error: string | null;
}

/** `GET /api/v1/runs` — the lineage table (newest first). */
export interface RunListResponse {
  readonly runs: readonly RunSummary[];
  readonly count: number;
}

/** The dataset lineage block (`_dataset_lineage` — the identifying fields). */
export interface RunDatasetBlock {
  readonly dataset_id: string;
  readonly name: string;
  readonly source_filename: string;
  readonly size_bytes: number;
  readonly shape: readonly [number, number];
  readonly target: string | null;
  readonly task: string | null;
  readonly n_features: number;
  readonly created_at: string;
}

/** The feature configuration (the dataset's own feature/target identity). */
export interface RunFeatureConfigurationBlock {
  readonly feature_names: readonly string[] | null;
  readonly n_features: number | null;
  readonly target: string | null;
}

/** The preprocessing pipeline block. */
export interface RunPreprocessingBlock {
  readonly enabled: boolean;
  readonly pipeline: string;
}

/** The algorithm + toolkit version block. */
export interface RunAlgorithmBlock {
  readonly name: string;
  readonly task: string;
  readonly toolkit_version: string;
}

/** The hyperparameters block (requested AND tuned). */
export interface RunHyperparametersBlock {
  readonly requested: Readonly<Record<string, unknown>>;
  readonly tuned: boolean;
  readonly best_params: Readonly<Record<string, unknown>> | null;
}

/** The §5.3 parallelism advisory (null on a failed run). */
export interface RunParallelismAdvisoryBlock {
  readonly requested: number;
  readonly executed: number;
  readonly reason: string | null;
  readonly estimated_bytes: number | null;
  readonly available_bytes: number | null;
}

/** The cross-validation configuration block. */
export interface RunCrossValidationBlock {
  readonly cv_folds: number;
  readonly tuning: boolean;
  readonly n_jobs_requested: number;
  readonly n_jobs_effective: number | null;
  readonly parallelism_advisory: RunParallelismAdvisoryBlock | null;
}

/** The evaluation metrics block (nulls = the honest absent on a failed run). */
export interface RunEvaluationBlock {
  readonly cv_scores: readonly number[] | null;
  readonly cv_mean: number | null;
  readonly cv_std: number | null;
  readonly primary_metric: string | null;
}

/** The model artifact reference (null when no fit ever landed). */
export interface RunModelArtifactBlock {
  readonly model_id: string;
  readonly name: string;
  readonly task: string;
  readonly n_train_rows: number | null;
  readonly fit_seconds: number | null;
  readonly created_at: string | null;
  readonly fetch: string;
}

/** The environment snapshot (`ml.provenance.environment_snapshot()`). */
export interface RunEnvironmentBlock {
  readonly toolkit: string;
  readonly python: string;
  readonly platform: string;
  readonly machine: string;
  readonly packages: Readonly<Record<string, string | null>>;
  readonly numeric_environment: Readonly<Record<string, string | null>>;
}

/** The job reference block (the full history rides the jobs route). */
export interface RunLogsBlock {
  readonly job_id: string | null;
  readonly stage: string;
  readonly history: string;
}

/** `GET /api/v1/runs/{run_id}` — the full §15.2 record (`view()`). */
export interface ExperimentRun extends RunSummary {
  readonly dataset: RunDatasetBlock;
  readonly feature_configuration: RunFeatureConfigurationBlock;
  readonly preprocessing: RunPreprocessingBlock;
  readonly algorithm: RunAlgorithmBlock;
  readonly hyperparameters: RunHyperparametersBlock;
  readonly cross_validation: RunCrossValidationBlock;
  readonly evaluation: RunEvaluationBlock;
  readonly model_artifact: RunModelArtifactBlock | null;
  readonly environment: RunEnvironmentBlock;
  readonly logs: RunLogsBlock;
  readonly schema: "pharmaml-experiment-run/1";
}

/**
 * Thrown when a response cannot be narrowed to the wire DTO (§8.1).
 * Defined beside the DTOs it guards — the same per-service pattern as
 * `services/jobs/types.ts` and `services/system/types.ts` (the shared
 * `@/services/api/client` module exports only the transport's own
 * `ApiError`).
 */
export class ApiSchemaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiSchemaError";
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function requireString(
  r: Record<string, unknown>,
  field: string,
  ctx: string,
): string {
  if (typeof r[field] !== "string") {
    throw new ApiSchemaError(`${ctx}: field ${field} is not a string`);
  }
  return r[field] as string;
}

function requireNumber(
  r: Record<string, unknown>,
  field: string,
  ctx: string,
): number {
  if (typeof r[field] !== "number") {
    throw new ApiSchemaError(`${ctx}: field ${field} is not a number`);
  }
  return r[field] as number;
}

function requireBoolean(
  r: Record<string, unknown>,
  field: string,
  ctx: string,
): boolean {
  if (typeof r[field] !== "boolean") {
    throw new ApiSchemaError(`${ctx}: field ${field} is not a boolean`);
  }
  return r[field] as boolean;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function numberArrayOrNull(
  value: unknown,
  ctx: string,
  field: string,
): readonly number[] | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (!Array.isArray(value) || !value.every((x) => typeof x === "number")) {
    throw new ApiSchemaError(`${ctx}: field ${field} is not a number array`);
  }
  return value as readonly number[];
}

function stringArrayOrNull(
  value: unknown,
  ctx: string,
  field: string,
): readonly string[] | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (!Array.isArray(value) || !value.every((x) => typeof x === "string")) {
    throw new ApiSchemaError(`${ctx}: field ${field} is not a string array`);
  }
  return value as readonly string[];
}

/** Require a named object block; a missing block is drift, not silence. */
function requireBlock(
  raw: unknown,
  field: string,
  ctx: string,
): Record<string, unknown> {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${ctx}: ${field} block missing`);
  }
  return raw;
}

function recordOfStringOrNull(
  value: unknown,
): Readonly<Record<string, string | null>> {
  if (!isObject(value)) {
    return {};
  }
  const out: Record<string, string | null> = {};
  for (const [key, entry] of Object.entries(value)) {
    out[key] = stringOrNull(entry);
  }
  return out;
}

/**
 * Validate an untrusted payload as a RunSummary (§8.1: validate API
 * responses where appropriate) — the lineage-table row.
 */
export function parseRunSummary(raw: unknown, ctx: string): RunSummary {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${ctx}: expected an object`);
  }
  const r = raw;
  const required: Array<readonly [string, "string" | "number"]> = [
    ["run_id", "string"],
    ["family", "string"],
    ["mode", "string"],
    ["status", "string"],
    ["dataset_id", "string"],
    ["dataset_name", "string"],
    ["model_name", "string"],
    ["task", "string"],
    ["cv_folds", "number"],
    ["random_seed", "number"],
    ["created_at", "string"],
    ["started_at", "string"],
    ["completed_at", "string"],
  ];
  for (const [field, kind] of required) {
    if (typeof r[field] !== kind) {
      throw new ApiSchemaError(`${ctx}: field ${field} is not a ${kind}`);
    }
  }
  const mode = r["mode"] as string;
  if (!(RUN_MODES as readonly string[]).includes(mode)) {
    throw new ApiSchemaError(`${ctx}: unknown run mode ${JSON.stringify(mode)}`);
  }
  const status = r["status"] as string;
  if (!(RUN_STATUSES as readonly string[]).includes(status)) {
    throw new ApiSchemaError(
      `${ctx}: unknown run status ${JSON.stringify(status)}`,
    );
  }
  return {
    run_id: r["run_id"] as string,
    family: r["family"] as string,
    mode: mode as RunMode,
    status: status as RunStatus,
    job_id: stringOrNull(r["job_id"]),
    dataset_id: r["dataset_id"] as string,
    dataset_name: r["dataset_name"] as string,
    model_name: r["model_name"] as string,
    task: r["task"] as string,
    metric: numberOrNull(r["metric"]),
    cv_folds: r["cv_folds"] as number,
    random_seed: r["random_seed"] as number,
    created_at: r["created_at"] as string,
    started_at: r["started_at"] as string,
    completed_at: r["completed_at"] as string,
    error: stringOrNull(r["error"]),
  };
}

/**
 * Validate an untrusted payload as the full ExperimentRun (§8.1) — the
 * §15.2 checklist joined in one record. The summary skeleton is parsed
 * first (the detail endpoint returns the summary fields too), then
 * every block is narrowed; a missing block is drift, never silence.
 */
export function parseExperimentRun(raw: unknown, ctx: string): ExperimentRun {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${ctx}: expected an object`);
  }
  const r = raw;
  const summary = parseRunSummary(raw, ctx);

  // dataset identifiers and versions
  const datasetRaw = requireBlock(r["dataset"], "dataset", ctx);
  const shapeRaw = datasetRaw["shape"];
  if (
    !Array.isArray(shapeRaw) ||
    shapeRaw.length !== 2 ||
    !shapeRaw.every((x) => typeof x === "number")
  ) {
    throw new ApiSchemaError(
      `${ctx}: dataset.shape is not the [rows, columns] pair`,
    );
  }
  const dataset: RunDatasetBlock = {
    dataset_id: requireString(datasetRaw, "dataset_id", ctx),
    name: requireString(datasetRaw, "name", ctx),
    source_filename: requireString(datasetRaw, "source_filename", ctx),
    size_bytes: requireNumber(datasetRaw, "size_bytes", ctx),
    shape: [shapeRaw[0] as number, shapeRaw[1] as number],
    target: stringOrNull(datasetRaw["target"]),
    task: stringOrNull(datasetRaw["task"]),
    n_features: requireNumber(datasetRaw, "n_features", ctx),
    created_at: requireString(datasetRaw, "created_at", ctx),
  };

  // feature configuration
  const featureRaw = requireBlock(
    r["feature_configuration"],
    "feature_configuration",
    ctx,
  );
  const feature_configuration: RunFeatureConfigurationBlock = {
    feature_names: stringArrayOrNull(
      featureRaw["feature_names"],
      ctx,
      "feature_configuration.feature_names",
    ),
    n_features: numberOrNull(featureRaw["n_features"]),
    target: stringOrNull(featureRaw["target"]),
  };

  // preprocessing pipeline
  const preprocessingRaw = requireBlock(r["preprocessing"], "preprocessing", ctx);
  const preprocessing: RunPreprocessingBlock = {
    enabled: requireBoolean(preprocessingRaw, "enabled", ctx),
    pipeline: requireString(preprocessingRaw, "pipeline", ctx),
  };

  // algorithm and version
  const algorithmRaw = requireBlock(r["algorithm"], "algorithm", ctx);
  const algorithm: RunAlgorithmBlock = {
    name: requireString(algorithmRaw, "name", ctx),
    task: requireString(algorithmRaw, "task", ctx),
    toolkit_version: requireString(algorithmRaw, "toolkit_version", ctx),
  };

  // hyperparameters (requested AND tuned)
  const hyperparametersRaw = requireBlock(
    r["hyperparameters"],
    "hyperparameters",
    ctx,
  );
  const hyperparameters: RunHyperparametersBlock = {
    requested: isObject(hyperparametersRaw["requested"])
      ? (hyperparametersRaw["requested"] as Record<string, unknown>)
      : {},
    tuned: requireBoolean(hyperparametersRaw, "tuned", ctx),
    best_params: isObject(hyperparametersRaw["best_params"])
      ? (hyperparametersRaw["best_params"] as Record<string, unknown>)
      : null,
  };

  // cross-validation configuration (the §5.3 advisory may be absent)
  const cvRaw = requireBlock(r["cross_validation"], "cross_validation", ctx);
  const advisoryRaw = cvRaw["parallelism_advisory"];
  let parallelism_advisory: RunParallelismAdvisoryBlock | null = null;
  if (advisoryRaw !== null && advisoryRaw !== undefined) {
    const advisory = requireBlock(
      advisoryRaw,
      "cross_validation.parallelism_advisory",
      ctx,
    );
    parallelism_advisory = {
      requested: requireNumber(advisory, "requested", ctx),
      executed: requireNumber(advisory, "executed", ctx),
      reason: stringOrNull(advisory["reason"]),
      estimated_bytes: numberOrNull(advisory["estimated_bytes"]),
      available_bytes: numberOrNull(advisory["available_bytes"]),
    };
  }
  const cross_validation: RunCrossValidationBlock = {
    cv_folds: requireNumber(cvRaw, "cv_folds", ctx),
    tuning: requireBoolean(cvRaw, "tuning", ctx),
    n_jobs_requested: requireNumber(cvRaw, "n_jobs_requested", ctx),
    n_jobs_effective: numberOrNull(cvRaw["n_jobs_effective"]),
    parallelism_advisory,
  };

  // evaluation metrics
  const evaluationRaw = requireBlock(r["evaluation"], "evaluation", ctx);
  const evaluation: RunEvaluationBlock = {
    cv_scores: numberArrayOrNull(evaluationRaw["cv_scores"], ctx, "evaluation.cv_scores"),
    cv_mean: numberOrNull(evaluationRaw["cv_mean"]),
    cv_std: numberOrNull(evaluationRaw["cv_std"]),
    primary_metric: stringOrNull(evaluationRaw["primary_metric"]),
  };

  // model artifact reference (absent when no fit landed — never invented)
  const artifactRaw = r["model_artifact"];
  let model_artifact: RunModelArtifactBlock | null = null;
  if (artifactRaw !== null && artifactRaw !== undefined) {
    const artifact = requireBlock(artifactRaw, "model_artifact", ctx);
    model_artifact = {
      model_id: requireString(artifact, "model_id", ctx),
      name: requireString(artifact, "name", ctx),
      task: requireString(artifact, "task", ctx),
      n_train_rows: numberOrNull(artifact["n_train_rows"]),
      fit_seconds: numberOrNull(artifact["fit_seconds"]),
      created_at: stringOrNull(artifact["created_at"]),
      fetch: requireString(artifact, "fetch", ctx),
    };
  }

  // environment information
  const environmentRaw = requireBlock(r["environment"], "environment", ctx);
  const environment: RunEnvironmentBlock = {
    toolkit: requireString(environmentRaw, "toolkit", ctx),
    python: requireString(environmentRaw, "python", ctx),
    platform: requireString(environmentRaw, "platform", ctx),
    machine: requireString(environmentRaw, "machine", ctx),
    packages: recordOfStringOrNull(environmentRaw["packages"]),
    numeric_environment: recordOfStringOrNull(
      environmentRaw["numeric_environment"],
    ),
  };

  // job reference
  const logsRaw = requireBlock(r["logs"], "logs", ctx);
  const logs: RunLogsBlock = {
    job_id: stringOrNull(logsRaw["job_id"]),
    stage: requireString(logsRaw, "stage", ctx),
    history: requireString(logsRaw, "history", ctx),
  };

  // the schema pin — a renamed vocabulary refuses, never drifts silently
  if (r["schema"] !== "pharmaml-experiment-run/1") {
    throw new ApiSchemaError(
      `${ctx}: unknown schema ${JSON.stringify(r["schema"])}`,
    );
  }

  return {
    ...summary,
    dataset,
    feature_configuration,
    preprocessing,
    algorithm,
    hyperparameters,
    cross_validation,
    evaluation,
    model_artifact,
    environment,
    logs,
    schema: "pharmaml-experiment-run/1",
  };
}
