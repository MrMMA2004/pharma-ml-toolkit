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
type WilliamsRequest = components["schemas"]["WilliamsRequest"];
type AdmetScoreRequest = components["schemas"]["AdmetScoreRequest"];
type SemiSupervisedTrainRequest =
  components["schemas"]["SemiSupervisedTrainRequest"];
type SemiSupervisedEvaluateRequest =
  components["schemas"]["SemiSupervisedEvaluateRequest"];

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
  /** part11 §22.5 — dataset mode only (null for inline rows). */
  readonly evaluation: EvaluationBlock | null;
}

/** One aligned sample row of the evaluation block (§22.5's identity). */
export interface EvaluationRow {
  readonly name: unknown;
  readonly actual: unknown;
  readonly predicted: unknown;
  readonly residual: number | null;
  readonly smiles: string | null;
}

/** The aligned actual/predicted sample the parity lane plots. */
export interface EvaluationBlock {
  readonly target: string;
  readonly task: string;
  readonly smiles_col: string | null;
  readonly n_rows_available: number;
  readonly n_rows_evaluated: number;
  readonly rows: readonly EvaluationRow[];
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

/** One row of the Williams training table (`WilliamsResult.to_frame()`). */
export interface WilliamsFrameRow {
  readonly name: string;
  readonly leverage: number;
  readonly std_residual: number;
  readonly inside: boolean;
}

/** One query row: leverage verdict + the model's own prediction. */
export interface WilliamsTestRow {
  readonly leverage: number;
  readonly inside: boolean;
  readonly predicted: number;
}

/**
 * The `analysis` dict — `WilliamsResult.as_dict()` plus the
 * route-added `frame` / `test_frame` records (B-4, part12 Phase 2).
 * The count fields are COUNTS over the training rows; `n_test` /
 * `test_inside` are null when no query rows were passed (the
 * training-only Williams plot lane).
 */
export interface WilliamsAnalysis {
  readonly n_train: number;
  readonly n_features: number;
  readonly leverage_threshold: number;
  readonly residual_limit: number;
  readonly rmse: number;
  readonly train_inside: number;
  readonly n_train_outlier_leverage: number;
  readonly n_train_outlier_residual: number;
  readonly n_test: number | null;
  readonly test_inside: number | null;
  readonly frame: readonly WilliamsFrameRow[];
  readonly test_frame: readonly WilliamsTestRow[];
  readonly feature_names: readonly string[];
}

/** The williams result payload (job COMPLETED / the sync result half). */
export interface WilliamsResultPayload {
  readonly analysis: WilliamsAnalysis;
  readonly artifact: ArtifactView | null;
}

/** The sync-mode williams answer. */
export interface WilliamsSyncResponse {
  readonly mode: "sync";
  readonly analysis: WilliamsAnalysis;
  readonly artifact: ArtifactView | null;
}

export type WilliamsResponse = MLJobResponse | WilliamsSyncResponse;

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

/**
 * The `/evaluate` body's `evaluation` view — the REAL held-out
 * comparison (`evaluate_semi_supervised`, r303's GUI lane over the
 * tested-but-non-GUI route): held-out accuracy/F1 against the
 * labelled-only baseline over seeded repeats.
 */
export interface SemiSupervisedEvaluation {
  readonly method: string;
  readonly dataset_id: string;
  readonly n_repeats: number;
  readonly heldout_accuracy: number | null;
  readonly heldout_f1_macro: number | null;
  readonly baseline_accuracy: number | null;
  readonly baseline_f1_macro: number | null;
  readonly train_accuracy_on_labeled: number | null;
  readonly delta_vs_baseline: number | null;
}

/** The sync-mode evaluate answer (`{mode: "sync", evaluation}`). */
export interface SemiSupervisedEvaluateSyncResponse {
  readonly mode: "sync";
  readonly evaluation: SemiSupervisedEvaluation;
}

export type SemiSupervisedEvaluateResponse =
  | MLJobResponse
  | SemiSupervisedEvaluateSyncResponse;

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

export interface WilliamsOptions {
  readonly modelId: string;
  readonly datasetId: string;
  readonly xTest: readonly (readonly number[])[] | null;
  readonly residualLimit: number;
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

export interface SemiSupervisedEvaluateOptions {
  readonly datasetId: string;
  readonly method: "self_training" | "label_spreading" | "label_propagation";
  readonly testSize: number;
  readonly nRepeats: number;
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

// -- the ADMET lane (part13 Phase 5's next remainder item) -------------

/** One vendored registry row — admet.csv's own cells + the quality join. */
export interface AdmetEndpointRow {
  readonly id: string;
  readonly category: string;
  readonly task_type: string;
  readonly units: string | null;
  readonly minimum: number | null;
  readonly maximum: number | null;
  readonly species: string | null;
  readonly size: number | null;
  readonly quality: number | null;
  readonly low_confidence: boolean;
}

/** One radial axis: the endpoint id, its display label, directionality. */
export interface AdmetRadialAxis {
  readonly endpoint: string;
  readonly label: string;
  readonly invert: boolean;
}

/** The honest provenance + limitations block (§11.1's own rule). */
export interface AdmetProvenance {
  readonly backend: string;
  readonly source: string;
  readonly n_endpoints: number;
  readonly n_learned_endpoints: number;
  readonly shipped_ensembles: number;
  readonly shipped_ensembles_note: string;
  readonly low_performance_threshold: number;
  readonly disclaimer: string;
}

/** `GET /api/v1/admet/endpoints` — the registry answer. */
export interface AdmetEndpointsResponse {
  readonly endpoints: readonly AdmetEndpointRow[];
  readonly categories: readonly string[];
  readonly radial_axes: readonly AdmetRadialAxis[];
  readonly low_confidence_endpoints: readonly string[];
  readonly provenance: AdmetProvenance;
  readonly score_url: string;
  readonly note: string;
}

/** The batch cap the route itself pins (`routes/admet.py`'s cap). */
export const ADMET_MAX_SMILES = 100;

/** One valid molecule's scorecard row. */
export interface AdmetMoleculeCard {
  readonly smiles: string;
  readonly valid: true;
  readonly physchem: Readonly<Record<string, number>>;
  readonly radial: Readonly<Record<string, number>>;
  readonly low_confidence_endpoints: readonly string[];
}

/** One invalid row — it reports itself, never dropped silently. */
export interface AdmetInvalidRow {
  readonly smiles: string;
  readonly valid: false;
  readonly error: string;
}

export type AdmetScoreRow = AdmetMoleculeCard | AdmetInvalidRow;

/** `POST /api/v1/admet/score` — the per-molecule scorecard answer. */
export interface AdmetScoreResponse {
  readonly n_smiles: number;
  readonly n_valid: number;
  readonly n_invalid: number;
  readonly molecules: readonly AdmetScoreRow[];
  readonly provenance: AdmetProvenance;
  readonly note: string;
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
   * `POST /api/v1/applicability-domain/leverage-williams` — the REAL
   * hat-matrix leverage / Williams analysis over a session-trained
   * REGRESSION model (B-4, part12 Phase 2; the route 422s a
   * classification target up front). `xTest: null` runs the
   * training-only lane. Job mode (the default): the COMPLETED result
   * payload is `{analysis, artifact}` (parse with
   * parseWilliamsResult).
   */
  async williams(options: WilliamsOptions): Promise<WilliamsResponse> {
    const request: WilliamsRequest = {
      model_id: options.modelId,
      dataset_id: options.datasetId,
      x_test: options.xTest === null ? null : options.xTest.map((row) => [...row]),
      residual_limit: options.residualLimit,
      figure: options.figure,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/applicability-domain/leverage-williams",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        ...parseWilliamsResult(raw, "williams(sync)"),
      };
    }
    return { mode: "job", job: parseJobView(raw["job"], "williams(job)") };
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

  /**
   * `POST /api/v1/semi-supervised/evaluate` — the REAL held-out
   * evaluation: the semi-supervised method's accuracy/F1 on a seeded
   * test split against the labelled-only baseline, over
   * `n_repeats` repeats (r303's GUI consumer of the tested route).
   * Job mode (the default): the COMPLETED result payload is
   * `{evaluation: view}` (parse with parseSemiSupervisedEvaluation).
   */
  async semiSupervisedEvaluate(
    options: SemiSupervisedEvaluateOptions,
  ): Promise<SemiSupervisedEvaluateResponse> {
    const request: SemiSupervisedEvaluateRequest = {
      dataset_id: options.datasetId,
      method: options.method,
      test_size: options.testSize,
      n_repeats: options.nRepeats,
      random_state: options.randomState,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/semi-supervised/evaluate",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        evaluation: parseSemiSupervisedEvaluation(
          raw["evaluation"],
          "semi_supervised evaluate(sync)",
        ),
      };
    }
    return {
      mode: "job",
      job: parseJobView(raw["job"], "semi_supervised evaluate(job)"),
    };
  },

  /**
   * `GET /api/v1/admet/endpoints` — the vendored 52-endpoint registry
   * (part13 Phase 5's ADMET lane): admet.csv's own rows + the quality
   * joins, the low-confidence warning set, the radial axes with
   * directionality, and the honest provenance block — every cell the
   * backend's own, never re-declared.
   */
  async admetEndpoints(): Promise<AdmetEndpointsResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/admet/endpoints",
    );
    return parseAdmetEndpointsResponse(raw, "admet endpoints");
  },

  /**
   * `POST /api/v1/admet/score` — the per-molecule scorecard over a
   * SMILES batch (sync-only light compute, the chemistry-descriptors
   * precedent): the 11-property physchem panel, the five-axis radial
   * summary on the DrugBank grid, the low-confidence list. Invalid
   * rows report themselves per-row — never dropped, never a 500.
   */
  async admetScore(smiles: readonly string[]): Promise<AdmetScoreResponse> {
    const request: AdmetScoreRequest = { smiles: [...smiles] };
    const raw = (await api.post(
      "/api/v1/admet/score",
      request,
    )) as Record<string, unknown>;
    return parseAdmetScoreResponse(raw, "admet score");
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
  const evaluationRaw = r["evaluation"];
  const evaluation: EvaluationBlock | null =
    typeof evaluationRaw === "object" && evaluationRaw !== null
      ? parseEvaluationBlock(evaluationRaw, `${what} evaluation`)
      : null;
  return {
    mode: "sync",
    model_id: modelId,
    n_rows: asNumber(r["n_rows"]) ?? 0,
    predictions: predictions as readonly unknown[],
    probabilities,
    evaluation,
  };
}

/** Narrow the §22.5 evaluation block — row-validated, never trusted. */
function parseEvaluationBlock(
  raw: unknown,
  what: string,
): EvaluationBlock {
  const r = expectObject(raw, what);
  const target = r["target"];
  const task = r["task"];
  const rowsRaw = r["rows"];
  if (
    typeof target !== "string" ||
    typeof task !== "string" ||
    !Array.isArray(rowsRaw)
  ) {
    throw new Error(`${what}: target/task/rows have wrong shapes`);
  }
  const rows: EvaluationRow[] = [];
  for (const rowRaw of rowsRaw) {
    if (typeof rowRaw !== "object" || rowRaw === null) {
      throw new Error(`${what}: a row is not an object`);
    }
    const row = rowRaw as Record<string, unknown>;
    rows.push({
      name: row["name"] ?? null,
      actual: row["actual"] ?? null,
      predicted: row["predicted"] ?? null,
      residual: asNumber(row["residual"]),
      smiles: asString(row["smiles"]),
    });
  }
  return {
    target,
    task,
    smiles_col: asString(r["smiles_col"]),
    n_rows_available: asNumber(r["n_rows_available"]) ?? 0,
    n_rows_evaluated: asNumber(r["n_rows_evaluated"]) ?? 0,
    rows,
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
 * Narrow the williams result payload (the job COMPLETED result, or the
 * sync answer's own halves): `{analysis: {…, frame, test_frame}, artifact}`.
 */
export function parseWilliamsResult(
  raw: unknown,
  what: string,
): WilliamsResultPayload {
  const r = expectObject(raw, what);
  return {
    analysis: parseWilliamsAnalysis(r["analysis"], `${what} analysis`),
    artifact:
      typeof r["artifact"] === "object" && r["artifact"] !== null
        ? parseArtifactView(r["artifact"], `${what} artifact`)
        : null,
  };
}

/** Narrow the williams analysis dict — every key surfaced as it arrives. */
function parseWilliamsAnalysis(
  raw: unknown,
  what: string,
): WilliamsAnalysis {
  const r = expectObject(raw, what);
  const nTrain = r["n_train"];
  const threshold = r["leverage_threshold"];
  if (typeof nTrain !== "number" || typeof threshold !== "number") {
    throw new Error(
      `${what}: n_train/leverage_threshold are not numbers`,
    );
  }
  const frameRaw = r["frame"];
  const frame: WilliamsFrameRow[] = [];
  if (Array.isArray(frameRaw)) {
    for (const rowRaw of frameRaw as unknown[]) {
      const row = expectObject(rowRaw, `${what} frame row`);
      const leverage = asNumber(row["leverage"]);
      const stdResidual = asNumber(row["std_residual"]);
      if (leverage === null || stdResidual === null) {
        throw new Error(`${what} frame row: leverage/std_residual missing`);
      }
      frame.push({
        name: asString(row["name"]) ?? "",
        leverage,
        std_residual: stdResidual,
        inside: row["inside"] === true,
      });
    }
  }
  const testFrameRaw = r["test_frame"];
  const testFrame: WilliamsTestRow[] = [];
  if (Array.isArray(testFrameRaw)) {
    for (const rowRaw of testFrameRaw as unknown[]) {
      const row = expectObject(rowRaw, `${what} test_frame row`);
      const leverage = asNumber(row["leverage"]);
      const predicted = asNumber(row["predicted"]);
      if (leverage === null || predicted === null) {
        throw new Error(`${what} test_frame row: leverage/predicted missing`);
      }
      testFrame.push({
        leverage,
        inside: row["inside"] === true,
        predicted,
      });
    }
  }
  const featureNamesRaw = r["feature_names"];
  return {
    n_train: nTrain,
    n_features: asNumber(r["n_features"]) ?? 0,
    leverage_threshold: threshold,
    residual_limit: asNumber(r["residual_limit"]) ?? 3.0,
    rmse: asNumber(r["rmse"]) ?? 0,
    train_inside: asNumber(r["train_inside"]) ?? 0,
    n_train_outlier_leverage:
      asNumber(r["n_train_outlier_leverage"]) ?? 0,
    n_train_outlier_residual:
      asNumber(r["n_train_outlier_residual"]) ?? 0,
    n_test: asNumber(r["n_test"]),
    test_inside: asNumber(r["test_inside"]),
    frame,
    test_frame: testFrame,
    feature_names: Array.isArray(featureNamesRaw)
      ? (featureNamesRaw as unknown[]).map((f) =>
          typeof f === "string" ? f : String(f),
        )
      : [],
  };
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

/**
 * Narrow the evaluate job's result payload / the sync `evaluation`
 * view: every key surfaced as it arrives, counts as numbers, nullables
 * stated honestly.
 */
export function parseSemiSupervisedEvaluation(
  raw: unknown,
  what: string,
): SemiSupervisedEvaluation {
  const r = expectObject(raw, what);
  const method = r["method"];
  const datasetId = r["dataset_id"];
  if (typeof method !== "string" || typeof datasetId !== "string") {
    throw new Error(`${what}: method/dataset_id are not strings`);
  }
  return {
    method,
    dataset_id: datasetId,
    n_repeats: asNumber(r["n_repeats"]) ?? 0,
    heldout_accuracy: asNumber(r["heldout_accuracy"]),
    heldout_f1_macro: asNumber(r["heldout_f1_macro"]),
    baseline_accuracy: asNumber(r["baseline_accuracy"]),
    baseline_f1_macro: asNumber(r["baseline_f1_macro"]),
    train_accuracy_on_labeled: asNumber(r["train_accuracy_on_labeled"]),
    delta_vs_baseline: asNumber(r["delta_vs_baseline"]),
  };
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

// -- the ADMET lane's §8.1 parsers -------------------------------------

/** Narrow the honest provenance block (runtime validation). */
function parseAdmetProvenance(
  raw: unknown,
  what: string,
): AdmetProvenance {
  const r = expectObject(raw, `${what} provenance`);
  const backend = r["backend"];
  const source = r["source"];
  if (typeof backend !== "string" || typeof source !== "string") {
    throw new Error(`${what} provenance: backend/source are not strings`);
  }
  return {
    backend,
    source,
    n_endpoints: asNumber(r["n_endpoints"]) ?? 0,
    n_learned_endpoints: asNumber(r["n_learned_endpoints"]) ?? 0,
    shipped_ensembles: asNumber(r["shipped_ensembles"]) ?? 0,
    shipped_ensembles_note: asString(r["shipped_ensembles_note"]) ?? "",
    low_performance_threshold:
      asNumber(r["low_performance_threshold"]) ?? 0.6,
    disclaimer: asString(r["disclaimer"]) ?? "",
  };
}

/** Narrow one registry row (admet.csv's own cells + the quality join). */
function parseAdmetEndpointRow(
  raw: unknown,
  what: string,
): AdmetEndpointRow {
  const r = expectObject(raw, what);
  const id = r["id"];
  const category = r["category"];
  const taskType = r["task_type"];
  if (
    typeof id !== "string" ||
    typeof category !== "string" ||
    typeof taskType !== "string"
  ) {
    throw new Error(`${what}: id/category/task_type are not strings`);
  }
  return {
    id,
    category,
    task_type: taskType,
    units: asString(r["units"]),
    minimum: asNumber(r["minimum"]),
    maximum: asNumber(r["maximum"]),
    species: asString(r["species"]),
    size: asNumber(r["size"]),
    quality: asNumber(r["quality"]),
    low_confidence: r["low_confidence"] === true,
  };
}

/** Narrow the registry answer. */
function parseAdmetEndpointsResponse(
  raw: unknown,
  what: string,
): AdmetEndpointsResponse {
  const r = expectObject(raw, what);
  const endpoints = r["endpoints"];
  const axes = r["radial_axes"];
  if (!Array.isArray(endpoints) || !Array.isArray(axes)) {
    throw new Error(`${what}: endpoints/radial_axes are malformed`);
  }
  return {
    endpoints: endpoints.map((row, i) =>
      parseAdmetEndpointRow(row, `${what} endpoints[${i}]`),
    ),
    categories: Array.isArray(r["categories"])
      ? (r["categories"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
    radial_axes: axes.map((axis, i) => {
      const a = expectObject(axis, `${what} radial_axes[${i}]`);
      const endpoint = a["endpoint"];
      const label = a["label"];
      if (typeof endpoint !== "string" || typeof label !== "string") {
        throw new Error(`${what} radial_axes[${i}]: malformed axis`);
      }
      return { endpoint, label, invert: a["invert"] === true };
    }),
    low_confidence_endpoints: Array.isArray(r["low_confidence_endpoints"])
      ? (r["low_confidence_endpoints"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
    provenance: parseAdmetProvenance(r["provenance"], what),
    score_url: asString(r["score_url"]) ?? "",
    note: asString(r["note"]) ?? "",
  };
}

/** Narrow one scorecard row — the valid/invalid union. */
function parseAdmetScoreRow(
  raw: unknown,
  what: string,
): AdmetScoreRow {
  const r = expectObject(raw, what);
  const smiles = r["smiles"];
  if (typeof smiles !== "string") {
    throw new Error(`${what}: smiles is not a string`);
  }
  if (r["valid"] !== true) {
    return {
      smiles,
      valid: false,
      error: asString(r["error"]) ?? "invalid SMILES",
    };
  }
  const physchemRaw = expectObject(r["physchem"], `${what} physchem`);
  const radialRaw = expectObject(r["radial"], `${what} radial`);
  const physchem: Record<string, number> = {};
  for (const [key, value] of Object.entries(physchemRaw)) {
    if (typeof value === "number") {
      physchem[key] = value;
    }
  }
  const radial: Record<string, number> = {};
  for (const [key, value] of Object.entries(radialRaw)) {
    if (typeof value === "number") {
      radial[key] = value;
    }
  }
  return {
    smiles,
    valid: true,
    physchem,
    radial,
    low_confidence_endpoints: Array.isArray(r["low_confidence_endpoints"])
      ? (r["low_confidence_endpoints"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
  };
}

/** Narrow the scorecard answer. */
function parseAdmetScoreResponse(
  raw: unknown,
  what: string,
): AdmetScoreResponse {
  const r = expectObject(raw, what);
  const molecules = r["molecules"];
  const nSmiles = r["n_smiles"];
  const nValid = r["n_valid"];
  const nInvalid = r["n_invalid"];
  if (
    !Array.isArray(molecules) ||
    typeof nSmiles !== "number" ||
    typeof nValid !== "number" ||
    typeof nInvalid !== "number"
  ) {
    throw new Error(
      `${what}: n_smiles/n_valid/n_invalid/molecules are malformed`,
    );
  }
  return {
    n_smiles: nSmiles,
    n_valid: nValid,
    n_invalid: nInvalid,
    molecules: molecules.map((row, i) =>
      parseAdmetScoreRow(row, `${what} molecules[${i}]`),
    ),
    provenance: parseAdmetProvenance(r["provenance"], what),
    note: asString(r["note"]) ?? "",
  };
}
