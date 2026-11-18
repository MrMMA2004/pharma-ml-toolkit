/**
 * The ML client — part11 §11.2's Machine Learning workflow service
 * (the train / predict / applicability-domain / semi-supervised
 * surface over the REAL backend engines).
 *
 * The backend contract (mirrored field-for-field from the route
 * sources `routes/ml.py`, `routes/applicability_domain.py` and
 * `routes/semi_supervised.py`; the request bodies come from the
 * generated OpenAPI types — the same one-DTO-source rule the
 * Python drift guards in `tests/test_frontend_services.py` pin
 * for the datasets and chemistry clients):
 *
 * * `GET /api/v1/ml/models?task=…&only_available=…` — the registry
 *   catalog projection (`list_models`: name, task, available,
 *   requires, notes — one source of truth, no estimator field to
 *   invent).
 * * `GET /api/v1/ml/trained` / `GET /api/v1/ml/trained/{model_id}` —
 *   the session's trained-model registry (`TrainedModelRecord.view()`).
 * * `POST /api/v1/ml/train` — `TrainRequest` → `{mode: "job",
 *   job: JobView}` (the default lane here); the COMPLETED job's REST
 *   result payload is `{model: TrainedModelView}` — the page relays
 *   the authoritative REST result through the jobs client (§13),
 *   never the WS event payload alone.
 * * `POST /api/v1/ml/predict` — `PredictRequest` in dataset mode →
 *   the sync answer `{mode: "sync", model_id, n_rows, predictions,
 *   probabilities?}` (probabilities only when the estimator exposes
 *   predict_proba — a 422 states the honest refusal otherwise).
 * * `POST /api/v1/ml/explain` — `ExplainRequest` → the sync answer
 *   `{mode: "sync", explain: view}` (job mode when asked): the REAL
 *   `shap_summary` engine over a bounded sample of the model's
 *   dataset — the per-feature mean-|SHAP| ranking (all features),
 *   the top-50-column preview matrices for the beeswarm, and the
 *   optional importance + beeswarm figure artifacts. Non-tree
 *   estimators surface the engine's own typed refusal as a 422.
 * * `POST /api/v1/applicability-domain/analyze` —
 *   `ADAnalyzeRequest` (x_test required, min 1 row) → job mode; the
 *   COMPLETED result payload is `{analysis: {…as_dict()…, frame},
 *   artifact: ArtifactView | null}` — every key is surfaced as it
 *   arrives, validated at runtime.
 * * `POST /api/v1/semi-supervised/train` —
 *   `SemiSupervisedTrainRequest` (classification datasets only — the
 *   backend 422s otherwise) → job mode; the COMPLETED result payload
 *   is `{semi_supervised: {…view…}}`.
 *
 * The x_test assembly for the AD stage rides the datasets client's
 * bounded `column()` bridge (each feature column ≤ 500 values — the
 * backend's own request cap, §17.3: explicit, never silent).
 *
 * Request bodies come from the generated OpenAPI types
 * (`src/types/api.d.ts`, §12.2's one DTO source — no hand duplicates).
 */

import { api } from "@/services/api/client";
import {
  parseArtifactView,
  type ArtifactView,
} from "@/services/artifacts/client";
import { type JobView, parseJobView } from "@/services/jobs/types";
import type { components } from "@/types/api.d";

type TrainRequest = components["schemas"]["TrainRequest"];
type PredictRequest = components["schemas"]["PredictRequest"];
type ExplainRequest = components["schemas"]["ExplainRequest"];
type ADAnalyzeRequest = components["schemas"]["ADAnalyzeRequest"];
type SemiSupervisedTrainRequest =
  components["schemas"]["SemiSupervisedTrainRequest"];

/**
 * The AD analyze engine's own feature cap (applicability_domain.py's
 * `_MAX_FEATURES`) — mirrored once beside the route so the page can
 * state the doomed request honestly instead of firing it.
 */
export const AD_MAX_FEATURES = 50;

/**
 * The explain route's own row-sample cap (ml.py's
 * `_MAX_EXPLAIN_ROWS`) — mirrored once beside the route so the page
 * can bound its control honestly instead of firing the doomed
 * request.
 */
export const EXPLAIN_MAX_ROWS = 500;

/** `list_models` — one registry catalog entry (the catalog's own keys). */
export interface ModelCatalogEntry {
  readonly name: string;
  readonly task: string;
  readonly available: boolean;
  readonly requires: string | null;
  readonly notes: string;
}

export interface ModelCatalogResponse {
  readonly models: readonly ModelCatalogEntry[];
}

/** `TrainedModelRecord.view()` — the session's trained-model summary. */
export interface TrainedModelView {
  readonly model_id: string;
  readonly name: string;
  readonly task: string;
  readonly dataset_id: string;
  readonly target: string;
  readonly features: readonly string[];
  readonly n_train_rows: number;
  readonly best_params: Readonly<Record<string, unknown>>;
  readonly cv_scores: readonly number[] | null;
  readonly cv_mean: number | null;
  readonly cv_std: number | null;
  readonly fit_seconds: number;
  readonly created_at: string;
}

export interface TrainedModelListResponse {
  readonly models: readonly TrainedModelView[];
}

/** The job-mode submission answer (train / analyze / semi-supervised). */
export interface MLJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

/** The sync-mode train answer (`{mode: "sync", model: view}`). */
export interface TrainSyncResponse {
  readonly mode: "sync";
  readonly model: TrainedModelView;
}

export type TrainResponse = MLJobResponse | TrainSyncResponse;

/** The probabilities block of the predict answer. */
export interface ProbabilitiesBlock {
  readonly classes: readonly unknown[];
  readonly rows: readonly (readonly number[])[];
}

/** `POST /api/v1/ml/predict` — the sync dataset-mode answer. */
export interface PredictResponse {
  readonly mode: "sync";
  readonly model_id: string;
  readonly n_rows: number;
  readonly predictions: readonly unknown[];
  readonly probabilities: ProbabilitiesBlock | null;
}

/** One row of the AD reliability frame (`to_frame`, JSON-safe). */
export interface ADFrameRow {
  readonly name: string;
  readonly predicted: number | null;
  readonly prediction_sd: number | null;
  readonly knn_distance: number | null;
  readonly inside_ad: boolean | null;
  readonly low_uncertainty: boolean | null;
  readonly reliable: boolean | null;
}

/**
 * The `analysis` dict — `ApplicabilityDomainResult.as_dict()` plus the
 * route-added `frame` records. The count fields (`inside_ad`,
 * `low_uncertainty`, `reliable`) are COUNTS over the query rows, not
 * per-row flags; the nullables state plainly that the bootstrap half
 * was skipped.
 */
export interface ADAnalysis {
  readonly k: number;
  readonly percentile: number;
  readonly distance_threshold: number;
  readonly n_train: number;
  readonly n_test: number;
  readonly inside_ad: number;
  readonly n_models: number | null;
  readonly uncertainty_threshold: number | null;
  readonly low_uncertainty: number | null;
  readonly reliable: number | null;
  readonly uncertainty_kind: string | null;
  readonly fit_seconds: number;
  readonly frame: readonly ADFrameRow[];
}

/** The analyze result payload (job COMPLETED / the sync result half). */
export interface ADAnalyzeResult {
  readonly analysis: ADAnalysis;
  readonly artifact: ArtifactView | null;
}

/** The sync-mode analyze answer. */
export interface ADAnalyzeSyncResponse {
  readonly mode: "sync";
  readonly analysis: ADAnalysis;
  readonly artifact: ArtifactView | null;
}

export type ADAnalyzeResponse = MLJobResponse | ADAnalyzeSyncResponse;

/** The confidence summary (self-training only, as it arrives). */
export interface ConfidenceSummary {
  readonly min: number;
  readonly mean: number;
  readonly max: number;
}

/** `_semi_view` — the semi-supervised result view. */
export interface SemiSupervisedView {
  readonly method: string;
  readonly params: Readonly<Record<string, unknown>>;
  readonly dataset_id: string;
  readonly target: string;
  readonly n_labeled: number;
  readonly n_unlabeled: number;
  readonly n_pseudo_labeled: number;
  readonly classes: readonly unknown[];
  readonly fit_seconds: number;
  readonly labels_preview: readonly unknown[];
  readonly n_labels: number;
  readonly confidence_summary: ConfidenceSummary | null;
  readonly model_id: string | null;
  readonly predict_url: string | null;
}

/** The sync-mode semi-supervised train answer. */
export interface SemiSupervisedSyncResponse {
  readonly mode: "sync";
  readonly semi_supervised: SemiSupervisedView;
}

export type SemiSupervisedTrainResponse =
  | MLJobResponse
  | SemiSupervisedSyncResponse;

/** One row of the explain ranking (all features, descending). */
export interface ExplainRankingRow {
  readonly feature: string;
  readonly mean_abs_shap: number;
}

/** The bounded preview matrices (top-ranked feature columns). */
export interface ExplainPreview {
  readonly max_features: number;
  readonly features: readonly string[];
  readonly values: readonly (readonly number[])[];
  readonly feature_values: readonly (readonly number[])[];
}

/** The `_explain_body` view — the SHAP summary payload. */
export interface ExplainResult {
  readonly model_id: string;
  readonly model_name: string;
  readonly task: string;
  readonly engine: string;
  readonly n_rows_available: number;
  readonly n_samples: number;
  readonly n_features: number;
  readonly base_value: number | null;
  readonly ranking: readonly ExplainRankingRow[];
  readonly preview: ExplainPreview;
  readonly artifacts: readonly ArtifactView[];
}

/** The sync-mode explain answer (`{mode: "sync", explain: view}`). */
export interface ExplainSyncResponse {
  readonly mode: "sync";
  readonly explain: ExplainResult;
}

export type ExplainResponse = MLJobResponse | ExplainSyncResponse;

export interface TrainOptions {
  readonly datasetId: string;
  readonly model: string;
  readonly task: "classification" | "regression" | null;
  readonly tune: boolean;
  readonly cv: number;
  readonly nJobs: number;
  readonly preprocess: boolean;
  readonly randomState: number;
  readonly modelParams: Record<string, unknown> | null;
  readonly mode?: "job" | "sync";
}

export interface PredictOptions {
  readonly modelId: string;
  readonly datasetId: string;
  readonly predictProba: boolean;
}

export interface ADAnalyzeOptions {
  readonly modelId: string;
  readonly datasetId: string;
  readonly xTest: readonly (readonly number[])[];
  readonly k: number;
  readonly percentile: number;
  readonly nModels: number;
  readonly uncertainty: boolean;
  readonly figure: boolean;
  readonly mode?: "job" | "sync";
}

export interface SemiSupervisedOptions {
  readonly datasetId: string;
  readonly method: "self_training" | "label_spreading" | "label_propagation";
  readonly baseEstimator: string | null;
  readonly maskFraction: number | null;
  readonly threshold: number;
  readonly kBest: number;
  readonly maxIter: number;
  readonly kernel: "knn" | "rbf";
  readonly nNeighbors: number;
  readonly gamma: number;
  readonly randomState: number;
  readonly mode?: "job" | "sync";
}

export interface ExplainOptions {
  readonly modelId: string;
  readonly datasetId: string | null;
  readonly maxRows: number;
  readonly figure: boolean;
  readonly mode?: "job" | "sync";
}

export const mlApi = {
  /**
   * `GET /api/v1/ml/models?task=…&only_available=…` — the registry
   * catalog projection. `task` must stay synced with the selected
   * dataset's own task (the backend filters classification /
   * regression / both).
   */
  async catalog(
    task: "classification" | "regression",
    onlyAvailable: boolean = true,
  ): Promise<ModelCatalogResponse> {
    const raw = await api.get<Record<string, unknown>>(
      `/api/v1/ml/models?task=${encodeURIComponent(task)}&only_available=${onlyAvailable ? "true" : "false"}`,
    );
    const items = raw["models"];
    if (!Array.isArray(items)) {
      return { models: [] };
    }
    return {
      models: items.map((item, i) =>
        parseModelCatalogEntry(item, `models[${i}]`),
      ),
    };
  },

  /** `GET /api/v1/ml/trained` — the session's trained models. */
  async trained(): Promise<TrainedModelListResponse> {
    const raw = await api.get<Record<string, unknown>>("/api/v1/ml/trained");
    const items = raw["models"];
    if (!Array.isArray(items)) {
      return { models: [] };
    }
    return {
      models: items.map((item, i) =>
        parseTrainedModelView(item, `trained[${i}]`),
      ),
    };
  },

  /** `GET /api/v1/ml/trained/{model_id}` — one trained model's view. */
  async trainedView(modelId: string): Promise<TrainedModelView> {
    const raw = await api.get<Record<string, unknown>>(
      `/api/v1/ml/trained/${encodeURIComponent(modelId)}`,
    );
    return parseTrainedModelView(raw, `trained ${modelId}`);
  },

  /**
   * `POST /api/v1/ml/train` — one registry model trained by the REAL
   * train_model engine on a registered dataset. Job mode (the default)
   * answers `{mode: "job", job: JobView}`; the COMPLETED result
   * payload is `{model: TrainedModelView}` (parseTrainResult).
   */
  async train(options: TrainOptions): Promise<TrainResponse> {
    const request: TrainRequest = {
      dataset_id: options.datasetId,
      model: options.model,
      task: options.task,
      tune: options.tune,
      cv: options.cv,
      n_jobs: options.nJobs,
      preprocess: options.preprocess,
      random_state: options.randomState,
      model_params: options.modelParams,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post("/api/v1/ml/train", request)) as Record<
      string,
      unknown
    >;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        model: parseTrainedModelView(raw["model"], "train(sync)"),
      };
    }
    return { mode: "job", job: parseJobView(raw["job"], "train(job)") };
  },

  /**
   * `POST /api/v1/ml/predict` — dataset mode (the rows lane stays the
   * backend's own inline path; this workflow always predicts over a
   * registered dataset's feature matrix). Sync only: the route's own
   * default lane.
   */
  async predict(options: PredictOptions): Promise<PredictResponse> {
    const request: PredictRequest = {
      model_id: options.modelId,
      dataset_id: options.datasetId,
      predict_proba: options.predictProba,
      mode: "sync",
    };
    const raw = (await api.post("/api/v1/ml/predict", request)) as Record<
      string,
      unknown
    >;
    return parsePredictResponse(raw, "predict");
  },

  /**
   * `POST /api/v1/ml/explain` — the REAL shap_summary engine over a
   * bounded sample of the trained model's dataset (its own when
   * datasetId is null). Sync by default (the route's own lane); the
   * job-mode COMPLETED result payload is `{explain: view}` (parse
   * with parseExplainResult).
   */
  async explain(options: ExplainOptions): Promise<ExplainResponse> {
    const request: ExplainRequest = {
      model_id: options.modelId,
      dataset_id: options.datasetId,
      max_rows: options.maxRows,
      figure: options.figure,
      mode: options.mode ?? "sync",
    };
    const raw = (await api.post("/api/v1/ml/explain", request)) as Record<
      string,
      unknown
    >;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        explain: parseExplainView(raw["explain"], "explain(sync)"),
      };
    }
    return { mode: "job", job: parseJobView(raw["job"], "explain(job)") };
  },

  /**
   * `POST /api/v1/applicability-domain/analyze` — the REAL bootstrap
   * analysis over a session-trained model. Job mode (the default):
   * the COMPLETED result payload is `{analysis, artifact}` (parse
   * with parseADAnalyzeResult).
   */
  async analyze(options: ADAnalyzeOptions): Promise<ADAnalyzeResponse> {
    const request: ADAnalyzeRequest = {
      model_id: options.modelId,
      dataset_id: options.datasetId,
      x_test: options.xTest.map((row) => [...row]),
      k: options.k,
      percentile: options.percentile,
      n_models: options.nModels,
      uncertainty: options.uncertainty,
      random_state: 0,
      figure: options.figure,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/applicability-domain/analyze",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        ...parseADAnalyzeResult(raw, "analyze(sync)"),
      };
    }
    return { mode: "job", job: parseJobView(raw["job"], "analyze(job)") };
  },

  /**
   * `POST /api/v1/semi-supervised/train` — the REAL
   * train_semi_supervised engine on a registered dataset
   * (classification only; the backend 422s regression datasets).
   * Job mode (the default): the COMPLETED result payload is
   * `{semi_supervised: view}` (parseSemiSupervisedResult).
   */
  async semiSupervisedTrain(
    options: SemiSupervisedOptions,
  ): Promise<SemiSupervisedTrainResponse> {
    const request: SemiSupervisedTrainRequest = {
      dataset_id: options.datasetId,
      method: options.method,
      unlabeled_marker: -1,
      mask_fraction: options.maskFraction,
      random_state: options.randomState,
      base_estimator: options.baseEstimator,
      threshold: options.threshold,
      k_best: options.kBest,
      max_iter: options.maxIter,
      kernel: options.kernel,
      n_neighbors: options.nNeighbors,
      gamma: options.gamma,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/semi-supervised/train",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        semi_supervised: parseSemiSupervisedView(
          raw["semi_supervised"],
          "semi_supervised(sync)",
        ),
      };
    }
    return {
      mode: "job",
      job: parseJobView(raw["job"], "semi_supervised(job)"),
    };
  },
};

// -- validation helpers (§8.1) ------------------------------------------

function expectObject(raw: unknown, what: string): Record<string, unknown> {
  if (typeof raw !== "object" || raw === null) {
    throw new Error(`${what}: expected an object`);
  }
  return raw as Record<string, unknown>;
}

function asString(raw: unknown): string | null {
  return typeof raw === "string" ? raw : null;
}

function asNumber(raw: unknown): number | null {
  return typeof raw === "number" ? raw : null;
}

function asBoolean(raw: unknown): boolean | null {
  return typeof raw === "boolean" ? raw : null;
}

/** Narrow an untrusted payload to one catalog entry. */
function parseModelCatalogEntry(
  raw: unknown,
  what: string,
): ModelCatalogEntry {
  const r = expectObject(raw, what);
  const name = r["name"];
  const task = r["task"];
  if (typeof name !== "string" || typeof task !== "string") {
    throw new Error(`${what}: name/task are not strings`);
  }
  return {
    name,
    task,
    available: r["available"] === true,
    requires: asString(r["requires"]),
    notes: asString(r["notes"]) ?? "",
  };
}

/** Narrow an untrusted payload to the trained-model view. */
export function parseTrainedModelView(
  raw: unknown,
  what: string,
): TrainedModelView {
  const r = expectObject(raw, what);
  const modelId = r["model_id"];
  const name = r["name"];
  if (typeof modelId !== "string" || typeof name !== "string") {
    throw new Error(`${what}: model_id/name are not strings`);
  }
  const cvScores = r["cv_scores"];
  return {
    model_id: modelId,
    name,
    task: asString(r["task"]) ?? "",
    dataset_id: asString(r["dataset_id"]) ?? "",
    target: asString(r["target"]) ?? "",
    features: Array.isArray(r["features"])
      ? (r["features"] as unknown[]).map((f) =>
          typeof f === "string" ? f : String(f),
        )
      : [],
    n_train_rows: asNumber(r["n_train_rows"]) ?? 0,
    best_params:
      typeof r["best_params"] === "object" && r["best_params"] !== null
        ? (r["best_params"] as Record<string, unknown>)
        : {},
    cv_scores: Array.isArray(cvScores)
      ? (cvScores as unknown[]).map((s) =>
          typeof s === "number" ? s : Number.NaN,
        )
      : null,
    cv_mean: asNumber(r["cv_mean"]),
    cv_std: asNumber(r["cv_std"]),
    fit_seconds: asNumber(r["fit_seconds"]) ?? 0,
    created_at: asString(r["created_at"]) ?? "",
  };
}

/** Narrow the sync predict answer. */
export function parsePredictResponse(
  raw: unknown,
  what: string,
): PredictResponse {
  const r = expectObject(raw, what);
  const modelId = r["model_id"];
  const predictions = r["predictions"];
  if (typeof modelId !== "string" || !Array.isArray(predictions)) {
    throw new Error(`${what}: model_id/predictions have wrong shapes`);
  }
  const probabilitiesRaw = r["probabilities"];
  let probabilities: ProbabilitiesBlock | null = null;
  if (typeof probabilitiesRaw === "object" && probabilitiesRaw !== null) {
    const p = probabilitiesRaw as Record<string, unknown>;
    probabilities = {
      classes: Array.isArray(p["classes"]) ? (p["classes"] as unknown[]) : [],
      rows: Array.isArray(p["rows"])
        ? (p["rows"] as unknown[]).map((row) =>
            Array.isArray(row) ? (row as readonly number[]) : [],
          )
        : [],
    };
  }
  return {
    mode: "sync",
    model_id: modelId,
    n_rows: asNumber(r["n_rows"]) ?? 0,
    predictions: predictions as readonly unknown[],
    probabilities,
  };
}

/**
 * Narrow the analyze result payload (the job COMPLETED result, or the
 * sync answer's own halves): `{analysis: {…, frame}, artifact}`.
 */
export function parseADAnalyzeResult(
  raw: unknown,
  what: string,
): ADAnalyzeResult {
  const r = expectObject(raw, what);
  return {
    analysis: parseADAnalysis(r["analysis"], `${what} analysis`),
    artifact:
      typeof r["artifact"] === "object" && r["artifact"] !== null
        ? parseArtifactView(r["artifact"], `${what} artifact`)
        : null,
  };
}

/** Narrow the analysis dict — every key surfaced as it arrives. */
function parseADAnalysis(raw: unknown, what: string): ADAnalysis {
  const r = expectObject(raw, what);
  const k = r["k"];
  const nTest = r["n_test"];
  if (typeof k !== "number" || typeof nTest !== "number") {
    throw new Error(`${what}: k/n_test are not numbers`);
  }
  const frameRaw = r["frame"];
  const frame: ADFrameRow[] = [];
  if (Array.isArray(frameRaw)) {
    for (const rowRaw of frameRaw as unknown[]) {
      const row = expectObject(rowRaw, `${what} frame row`);
      frame.push({
        name: asString(row["name"]) ?? "",
        predicted: asNumber(row["predicted"]),
        prediction_sd: asNumber(row["prediction_sd"]),
        knn_distance: asNumber(row["knn_distance"]),
        inside_ad: asBoolean(row["inside_ad"]),
        low_uncertainty: asBoolean(row["low_uncertainty"]),
        reliable: asBoolean(row["reliable"]),
      });
    }
  }
  return {
    k,
    percentile: asNumber(r["percentile"]) ?? 0,
    distance_threshold: asNumber(r["distance_threshold"]) ?? 0,
    n_train: asNumber(r["n_train"]) ?? 0,
    n_test: nTest,
    inside_ad: asNumber(r["inside_ad"]) ?? 0,
    n_models: asNumber(r["n_models"]),
    uncertainty_threshold: asNumber(r["uncertainty_threshold"]),
    low_uncertainty: asNumber(r["low_uncertainty"]),
    reliable: asNumber(r["reliable"]),
    uncertainty_kind: asString(r["uncertainty_kind"]),
    fit_seconds: asNumber(r["fit_seconds"]) ?? 0,
    frame,
  };
}

/** Narrow the train job's result payload: `{model: TrainedModelView}`. */
export function parseTrainResult(raw: unknown, what: string): TrainedModelView {
  const r = expectObject(raw, what);
  return parseTrainedModelView(r["model"], `${what} model`);
}

/**
 * Narrow the semi-supervised train result payload:
 * `{semi_supervised: SemiSupervisedView}`.
 */
export function parseSemiSupervisedResult(
  raw: unknown,
  what: string,
): SemiSupervisedView {
  const r = expectObject(raw, what);
  return parseSemiSupervisedView(r["semi_supervised"], what);
}

/** Narrow the explain job result payload: `{explain: ExplainResult}`. */
export function parseExplainResult(
  raw: unknown,
  what: string,
): ExplainResult {
  const r = expectObject(raw, what);
  return parseExplainView(r["explain"], what);
}

/** Narrow the `_explain_body` view — optional keys surface as null. */
function parseExplainView(raw: unknown, what: string): ExplainResult {
  const r = expectObject(raw, what);
  const modelId = r["model_id"];
  const modelName = r["model_name"];
  const nSamples = r["n_samples"];
  if (
    typeof modelId !== "string" ||
    typeof modelName !== "string" ||
    typeof nSamples !== "number"
  ) {
    throw new Error(`${what}: model_id/model_name/n_samples have wrong shapes`);
  }
  const rankingRaw = r["ranking"];
  const ranking: ExplainRankingRow[] = [];
  if (Array.isArray(rankingRaw)) {
    for (const rowRaw of rankingRaw as unknown[]) {
      const row = expectObject(rowRaw, `${what} ranking row`);
      const feature = row["feature"];
      const meanAbs = row["mean_abs_shap"];
      if (typeof feature !== "string" || typeof meanAbs !== "number") {
        throw new Error(`${what} ranking row has wrong shapes`);
      }
      ranking.push({ feature, mean_abs_shap: meanAbs });
    }
  }
  const previewRaw = r["preview"];
  let preview: ExplainPreview = {
    max_features: 0,
    features: [],
    values: [],
    feature_values: [],
  };
  if (typeof previewRaw === "object" && previewRaw !== null) {
    const p = previewRaw as Record<string, unknown>;
    preview = {
      max_features: asNumber(p["max_features"]) ?? 0,
      features: Array.isArray(p["features"])
        ? (p["features"] as unknown[]).map((f) =>
            typeof f === "string" ? f : String(f),
          )
        : [],
      values: matrixOf(p["values"], `${what} preview.values`),
      feature_values: matrixOf(
        p["feature_values"],
        `${what} preview.feature_values`,
      ),
    };
  }
  const artifactsRaw = r["artifacts"];
  const artifacts: ArtifactView[] = [];
  if (Array.isArray(artifactsRaw)) {
    for (const a of artifactsRaw as unknown[]) {
      if (typeof a === "object" && a !== null) {
        artifacts.push(parseArtifactView(a, `${what} artifact`));
      }
    }
  }
  return {
    model_id: modelId,
    model_name: modelName,
    task: asString(r["task"]) ?? "",
    engine: asString(r["engine"]) ?? "",
    n_rows_available: asNumber(r["n_rows_available"]) ?? 0,
    n_samples: nSamples,
    n_features: asNumber(r["n_features"]) ?? 0,
    base_value: asNumber(r["base_value"]),
    ranking,
    preview,
    artifacts,
  };
}

/** A (rows × columns) numeric matrix, validated row-by-row. */
function matrixOf(
  raw: unknown,
  what: string,
): readonly (readonly number[])[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  return (raw as unknown[]).map((row, i) => {
    if (!Array.isArray(row)) {
      throw new Error(`${what}[${i}]: expected a numeric row`);
    }
    return (row as unknown[]).map((v) =>
      typeof v === "number" ? v : Number.NaN,
    );
  });
}

/** Narrow the `_semi_view` payload — optional keys surface as null. */
function parseSemiSupervisedView(
  raw: unknown,
  what: string,
): SemiSupervisedView {
  const r = expectObject(raw, what);
  const method = r["method"];
  const nLabeled = r["n_labeled"];
  if (typeof method !== "string" || typeof nLabeled !== "number") {
    throw new Error(`${what}: method/n_labeled have wrong shapes`);
  }
  const classes = r["classes"];
  const labelsPreview = r["labels_preview"];
  const confidenceRaw = r["confidence_summary"];
  let confidence_summary: ConfidenceSummary | null = null;
  if (typeof confidenceRaw === "object" && confidenceRaw !== null) {
    const c = confidenceRaw as Record<string, unknown>;
    confidence_summary = {
      min: asNumber(c["min"]) ?? 0,
      mean: asNumber(c["mean"]) ?? 0,
      max: asNumber(c["max"]) ?? 0,
    };
  }
  return {
    method,
    params:
      typeof r["params"] === "object" && r["params"] !== null
        ? (r["params"] as Record<string, unknown>)
        : {},
    dataset_id: asString(r["dataset_id"]) ?? "",
    target: asString(r["target"]) ?? "",
    n_labeled: nLabeled,
    n_unlabeled: asNumber(r["n_unlabeled"]) ?? 0,
    n_pseudo_labeled: asNumber(r["n_pseudo_labeled"]) ?? 0,
    classes: Array.isArray(classes) ? (classes as unknown[]) : [],
    fit_seconds: asNumber(r["fit_seconds"]) ?? 0,
    labels_preview: Array.isArray(labelsPreview)
      ? (labelsPreview as unknown[])
      : [],
    n_labels: asNumber(r["n_labels"]) ?? 0,
    confidence_summary,
    model_id: asString(r["model_id"]),
    predict_url: asString(r["predict_url"]),
  };
}
