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
 *   invent). `task` omitted lists the whole catalog (the §15.3
 *   registry view).
 * * `GET /api/v1/ml/trained` / `GET /api/v1/ml/trained/{model_id}` —
 *   the session's trained-model registry (`TrainedModelRecord.view()`).
 * * `DELETE /api/v1/ml/trained/{model_id}` — the registry's deletion
 *   lane (part16 §15.3): the bounded store's evict (the in-memory
 *   record, the persisted row + vault payload when durable), the
 *   §15.2 run lineage and exported artifacts untouched.
 * * `POST /api/v1/artifacts/model` — the registry's export lane
 *   (§15.3: the trained estimator persisted through the REAL
 *   `save_model` as a digest-verified joblib artifact — the
 *   export is an independent copy that outlives the model).
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
 * * `GET /api/v1/repurposing/registry` — the drug-repurposing lane's
 *   vocabulary (part13 Phase 5): DeepResponse's split ladder, the
 *   CaDRReS formulation, the fit hyperparameters + the route's
 *   sync-light caps, the ten-key metric contract, the repurposing
 *   table schema, and the honest provenance block.
 * * `POST /api/v1/repurposing/screen` —
 *   `RepurposingScreenRequest` → the sync round trip over the real
 *   backend: the seeded SYNTHETIC screen, the CaDRReS fit, the
 *   predicted IC50 matrix, the metric contract on the observed
 *   entries, and the ranked candidates per profile — every predicted
 *   value a labelled model estimate, never real pharmacology.
 * * `GET /api/v1/drugresponse/registry` — the cancer cell-line
 *   drug-response lane's vocabulary (part13 Phase 5): the CaDRReS
 *   bilinear formulation with its sigmoid link, the fit
 *   hyperparameters + the route's sync-light caps, the five-key
 *   metric contract, the sensitivity-ranking and biomarker schemas,
 *   the cross-cell-line validation rule, and the honest provenance
 *   block.
 * * `POST /api/v1/drugresponse/analyze` —
 *   `DrugResponseAnalyzeRequest` → the sync round trip over the real
 *   backend: the seeded SYNTHETIC screen, the bilinear fit, the
 *   predicted response-score matrix, the metric contract on the
 *   observed entries, the per-cell sensitivity ranking, the per-drug
 *   biomarker weights, and the cross-cell-line validation folds —
 *   one response scale only, every predicted value a labelled model
 *   estimate, never real pharmacology.
 * * `GET /api/v1/pharmacogenomics/registry` — the pharmacogenomic-
 *   association lane's vocabulary (part13 Phase 5): the classic GDSC
 *   ANOVA design matrix with its sequential QR F-test and the
 *   feasibility gates, the effect-size contract, the BH FDR +
 *   Storey-Tibshirani q-value layer with gdsctools' percent
 *   convention, the volcano transforms with the default hit rule,
 *   the drevalpy evaluation guards + the mean-effects ruler with
 *   the normalized metric, the cBioPortal vocabularies, and the
 *   honest provenance block (five repositories, zero shipped
 *   tables).
 * * `POST /api/v1/pharmacogenomics/analyze` —
 *   `PharmacogenomicsAnalyzeRequest` → the sync round trip over the
 *   real backend: the seeded SYNTHETIC screen, the real GDSC ANOVA
 *   over every feasible drug x feature pair, the q-value/FDR
 *   columns, the volcano verdicts, the top associations table, and
 *   the mean-effects ruler evaluated through the guarded family
 *   with the normalized score beyond the naive drug-mean ruler —
 *   one response scale only (the natural-log µM scale), every
 *   p-value a statistic over the synthetic screen, never real
 *   pharmacology.
 * * `GET /api/v1/multiobjective/registry` — the multi-objective HPO
 *   lane's vocabulary (part13 Phase 5's last remainder item): the
 *   optuna minimize convention + the Pareto order, the exact 2-D
 *   hypervolume with the WFG reference contract, NSGA-II's crowding
 *   (+inf boundaries read as null), the MOEA/D Tchebycheff
 *   decomposition, the Das-Dennis/random weight supply, the
 *   tunable-objective ledger (score/fit_seconds/score_seconds with
 *   their cross_validate keys and sign flips), the report schema
 *   with the one-shot prune contract, and the honest scope block
 *   (optuna/Ax/nevergrad reviewed; EHVI/MOTPE/NSGA-II operators
 *   stay upstream; zero checkpoints; no real dataset ships).
 * * `POST /api/v1/multiobjective/optimize` —
 *   `MultiObjectiveOptimizeRequest` → the sync round trip over the
 *   real backend: the seeded SYNTHETIC 80x6 frame, the request's own
 *   search space, the real cross-validated study over the sklearn
 *   registry model, the non-dominated front with its aligned
 *   crowding, the 2-D indicator + the monotone curve (null for a
 *   3-objective study — the exact sweep is 2-D only, never
 *   fabricated), the per-objective trade-off corners, and the
 *   crowding-pruned shortlist — every objective value a labelled
 *   statistic over the synthetic dataset; the fold timings vary
 *   run-to-run (the honest drift note rides every answer).
 *
 * * `GET /api/v1/workflow/steps` — the §8 Scientific Workflow Engine
 *   lane's typed vocabulary (part13 Phase 6, the r322 API lane this
 *   client's GUI leg rides): the engine block's own decision text
 *   (the in-tree layer over the existing JobManager — Prefect /
 *   Knime / JupyterLab / marimo reviewed in
 *   docs/ARCHITECTURE_REVIEW.md, none adopted), the closed
 *   port-kind vocabulary, and every step's typed input/output ports,
 *   parameter dials, the REAL backend it delegates to, and its
 *   honest notes — zero new dependencies, zero algorithms
 *   reimplemented.
 * * `GET /api/v1/workflow/templates` — the saved workflow templates
 *   (§8's own example chain `guided-qsar` with its honest leakage
 *   reorder — the split runs before feature selection because the
 *   toolkit's own leakage rule is a contract — plus the 6-node
 *   minimal-regression core), each entry carrying the full
 *   serialisable spec the builder edits.
 * * `POST /api/v1/workflow/validate` — the pre-execution dependency
 *   validation: every finding at once (unknown steps/nodes/ports,
 *   contract mismatches, cycles) plus the topological order the
 *   executor would walk and the canonical spec digest.
 * * `POST /api/v1/workflow/run` — the real round trip over the
 *   engine's REAL backend runners: sync mode (bounded ≤ 12 nodes /
 *   ≤ 80 records) answers the run record inline — the per-node
 *   states, the artifact lineage with honest digest_kinds, the
 *   partial-failure ledger (the backend's own error verbatim,
 *   downstream SKIPPED, completed artifacts surviving), and the
 *   reproducibility block with the honest determinism scope; job
 *   mode (≤ 24 nodes / ≤ 200 records) rides the EXISTING
 *   JobManager (progress per node, cooperative cancellation between
 *   nodes, the SQLite job history). The default dataset is the
 *   route's own seeded SYNTHETIC 30-molecule stand-in — every
 *   metric a statistic over it, never an experimental fact.
 * * `GET /api/v1/workflow/runs` — the bounded in-session execution
 *   history + the honest lifetime note (job-mode runs persist
 *   through the JobManager's own SQLite history).
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
import type { SplitStrategy } from "@/services/datasets/client";
import type { components } from "@/types/api.d";

type TrainRequest = components["schemas"]["TrainRequest"];
type TrainSplitSpec = components["schemas"]["TrainSplitSpec"];
type PredictRequest = components["schemas"]["PredictRequest"];
type ExplainRequest = components["schemas"]["ExplainRequest"];
type ADAnalyzeRequest = components["schemas"]["ADAnalyzeRequest"];
type WilliamsRequest = components["schemas"]["WilliamsRequest"];
type AdmetScoreRequest = components["schemas"]["AdmetScoreRequest"];
type DtaFeaturesRequest = components["schemas"]["DtaFeaturesRequest"];
type RepurposingScreenRequest =
  components["schemas"]["RepurposingScreenRequest"];
type DrugResponseAnalyzeRequest =
  components["schemas"]["DrugResponseAnalyzeRequest"];
type PharmacogenomicsAnalyzeRequest =
  components["schemas"]["PharmacogenomicsAnalyzeRequest"];
type MultiobjectiveOptimizeRequest =
  components["schemas"]["MultiObjectiveOptimizeRequest"];
type WorkflowRunRequest = components["schemas"]["WorkflowRunRequest"];
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

/** `DELETE /api/v1/ml/trained/{model_id}` — the registry's evict answer. */
export interface TrainedModelEvicted {
  readonly evicted: string;
  readonly remaining: number;
}

/** The job-mode submission answer (train / analyze / semi-supervised). */
export interface MLJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

/** The sync-mode train answer (`{mode: "sync", model, split}`). */
export interface TrainSyncResponse {
  readonly mode: "sync";
  readonly model: TrainedModelView;
  /** ML-3 — null when the request carried no split section. */
  readonly split: TrainSplitBlock | null;
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
  /** part21 §3 — the sample's own residual diagnostics (regression). */
  readonly residual_diagnostics: ResidualDiagnostics | null;
  /** part21 §3 — per-class metrics over the sample (classification). */
  readonly class_metrics: ClassMetrics | null;
  /** part21 §3 — per-class OvR ROC curves (when proba was carried). */
  readonly roc_curves: readonly RocCurveSeries[] | null;
  /** part21 §3 — per-class OvR PR curves (when proba was carried). */
  readonly pr_curves: readonly PrCurveSeries[] | null;
  /** part21 §3 — the binary reliability diagram (when proba + 2 classes). */
  readonly calibration: CalibrationDiagram | null;
}

/** The regression sample's own residual diagnostics (part21 §3.1). */
export interface ResidualDiagnostics {
  readonly histogram: ResidualHistogram;
  readonly stats: ResidualStats;
  readonly qq: QQPoints | null;
}

export interface ResidualHistogram {
  readonly bin_edges: readonly number[];
  readonly counts: readonly number[];
}

export interface ResidualStats {
  readonly mean: number;
  readonly median: number;
  readonly std: number;
  readonly min: number;
  readonly max: number;
  readonly n: number;
}

/** Normal Q–Q points over the ordered residuals (Blom positions). */
export interface QQPoints {
  readonly theoretical: readonly number[];
  readonly empirical: readonly number[];
}

export interface ClassMetrics {
  readonly rows: readonly ClassMetricRow[];
  readonly macro: Readonly<Record<"precision" | "recall" | "f1", number>>;
  readonly accuracy: number;
  readonly n: number;
}

export interface ClassMetricRow {
  readonly label: string;
  readonly precision: number;
  readonly recall: number;
  readonly f1: number;
  readonly support: number;
}

/** One OvR ROC curve for a class over the evaluation sample. */
export interface RocCurveSeries {
  readonly label: string;
  readonly fpr: readonly number[];
  readonly tpr: readonly number[];
  readonly thresholds: readonly number[];
  readonly auc: number;
  readonly n_pos: number;
  readonly n_neg: number;
}

/** One OvR PR curve for a class over the evaluation sample. */
export interface PrCurveSeries {
  readonly label: string;
  readonly recall: readonly number[];
  readonly precision: readonly number[];
  readonly thresholds: readonly number[];
  readonly average_precision: number;
  readonly prevalence: number;
  readonly n_pos: number;
}

/** The binary reliability diagram over the positive-class score. */
export interface CalibrationDiagram {
  readonly bins: readonly CalibrationBin[];
  readonly n: number;
  readonly positive_class: string;
}

export interface CalibrationBin {
  readonly mean_predicted: number;
  readonly observed: number;
  readonly count: number;
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

/** The ML-3 split block's strategy literal set (the parser's own guard). */
const TRAIN_SPLIT_STRATEGY_LITERALS: readonly string[] = [
  "random",
  "stratified",
  "scaffold",
  "group",
  "kennard_stone",
  "temporal",
];

/** The ML-3 train-lane split section (the frontend-shaped knobs). */
export interface TrainSplitOptions {
  readonly strategy: SplitStrategy;
  /** null = the honest backend auto-detection (scaffold only). */
  readonly smilesColumn?: string | null;
  /** required by the group strategy, never guessed. */
  readonly groupColumn?: string | null;
  readonly testSize: number;
}

/** One side of the ML-3 carve (the fit saw the train side only). */
export interface TrainSplitSide {
  readonly rows: number;
}

/** One side's cohort census for the group-carrying strategies. */
export interface TrainSplitGroupCensusSide {
  readonly n_groups: number;
  readonly top_groups: ReadonlyArray<readonly [string, number]>;
}

/** The ML-3 carve block — sides + the census + the leakage verdict. */
export interface TrainSplitBlock {
  readonly strategy: SplitStrategy;
  readonly smiles_column: string | null;
  readonly group_column: string | null;
  readonly test_size: number;
  readonly random_state: number;
  readonly n_rows: number;
  readonly sides: {
    readonly train: TrainSplitSide;
    readonly heldout: TrainSplitSide;
  };
  readonly group_census: {
    readonly train: TrainSplitGroupCensusSide;
    readonly heldout: TrainSplitGroupCensusSide;
    readonly shared_groups: readonly string[];
  } | null;
  /** present only on the §15.2 failed-run record's refused echo. */
  readonly error?: string | null;
}

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
  /** ML-3 — the optional strategy-aware held-out carve before the fit. */
  readonly split?: TrainSplitOptions | null;
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

// -- the DTA lane (part13 Phase 5's next remainder item) -------------

/** One DeepDTA alphabet block — the vendored symbol→code table. */
export interface DtaAlphabetBlock {
  readonly n_symbols: number;
  readonly padding_code: number;
  readonly symbols: Readonly<Record<string, number>>;
  readonly encoding: string;
  readonly source: string;
}

/** The AAC panel arithmetic (the backend's own columns). */
export interface DtaProteinFeatures {
  readonly n_base: number;
  readonly base_columns: readonly string[];
  readonly n_dipeptide_extras: number;
  readonly n_with_dipeptides: number;
  readonly note: string;
}

/** The honest provenance + limitations block. */
export interface DtaProvenance {
  readonly backend: string;
  readonly source: string;
  readonly n_alphabets: number;
  readonly shipped_checkpoints: number;
  readonly shipped_checkpoints_note: string;
  readonly disclaimer: string;
}

/** `GET /api/v1/dta/registry` — the encoding registry answer. */
export interface DtaRegistryResponse {
  readonly alphabets: {
    readonly protein: DtaAlphabetBlock;
    readonly smiles_isomeric: DtaAlphabetBlock;
    readonly smiles_canonical: DtaAlphabetBlock;
  };
  readonly amino_acids: readonly string[];
  readonly protein_features: DtaProteinFeatures;
  readonly ligand_features: {
    readonly family: string;
    readonly n_bits: number;
    readonly source: string;
  };
  readonly paired_width: {
    readonly base: number;
    readonly with_dipeptides: number;
  };
  readonly metrics: readonly {
    readonly name: string;
    readonly description: string;
  }[];
  readonly demo_frame: {
    readonly rows: number;
    readonly note: string;
  };
  readonly provenance: DtaProvenance;
  readonly features_url: string;
  readonly note: string;
}

/** The batch cap the route itself pins (`routes/dta.py`'s cap). */
export const DTA_MAX_PAIRS = 100;

/** One row's protein composition — the backend's own columns. */
export interface DtaProteinRow {
  readonly prot_length: number;
  readonly prot_standard_fraction: number;
  readonly prot_nonstandard_fraction: number;
  readonly aac: Readonly<Record<string, number>>;
}

/** One pair card — the valid/invalid verdict rides the row itself. */
export interface DtaPairCard {
  readonly index: number;
  readonly smiles: string;
  readonly protein: string;
  readonly valid: boolean;
  readonly protein_row: DtaProteinRow;
  readonly affinity: number | null;
}

/** One dropped row — it reports itself, never dropped silently. */
export interface DtaDroppedRow {
  readonly index: number;
  readonly smiles: string;
  readonly reason: string;
}

/** `POST /api/v1/dta/features` — the paired featurization answer. */
export interface DtaFeaturesResponse {
  readonly n_pairs: number;
  readonly n_kept: number;
  readonly n_dropped: number;
  readonly kept_indices: readonly number[];
  readonly dropped: readonly DtaDroppedRow[];
  readonly n_ligand_features: number;
  readonly n_protein_features: number;
  readonly n_features: number;
  readonly dipeptides: boolean;
  readonly rows: readonly DtaPairCard[];
  readonly affinity: {
    readonly present: boolean;
    readonly n: number;
    readonly min: number;
    readonly max: number;
    readonly mean: number;
  } | null;
  readonly provenance: DtaProvenance;
  readonly note: string;
}

// -- the repurposing lane (part13 Phase 5's next remainder item) ------

/** The honest provenance + limitations block. */
export interface RepurposingProvenance {
  readonly backend: string;
  readonly source: string;
  readonly n_repositories: number;
  readonly shipped_checkpoints: number;
  readonly shipped_checkpoints_note: string;
  readonly screen: string;
  readonly disclaimer: string;
}

/** `GET /api/v1/repurposing/registry` — the vocabulary answer. */
export interface RepurposingRegistryResponse {
  readonly split_strategies: {
    readonly within_table: readonly string[];
    readonly cross_domain: string;
    readonly source: string;
  };
  readonly model: {
    readonly family: string;
    readonly formulation: string;
    readonly direction: string;
    readonly telemetry_fields: readonly string[];
    readonly history_note: string;
  };
  readonly hyperparameters: {
    readonly n_factors: Readonly<Record<string, number>>;
    readonly lr: Readonly<Record<string, number>>;
    readonly max_epochs: Readonly<Record<string, number>>;
    readonly seed: Readonly<Record<string, number>>;
    readonly note: string;
  };
  readonly screen: {
    readonly defaults: Readonly<Record<string, number>>;
    readonly bounds: Readonly<Record<string, string>>;
    readonly label: string;
  };
  readonly repurposing: {
    readonly schema: readonly string[];
    readonly ordering: string;
    readonly source: string;
    readonly response_limits: {
      readonly profiles: string;
      readonly top_k_per_profile: string;
    };
  };
  readonly metrics: {
    readonly keys: readonly string[];
    readonly threshold: number;
    readonly note: string;
  };
  readonly provenance: RepurposingProvenance;
  readonly screen_url: string;
  readonly note: string;
}

/** One ranked candidate row — the backend's own schema. */
export interface RepurposingCandidateRow {
  readonly drug: string;
  readonly predicted_ic50: number;
  readonly rank: number;
}

/** `POST /api/v1/repurposing/screen` — the round-trip answer. */
export interface RepurposingScreenResponse {
  readonly request: Readonly<Record<string, number>>;
  readonly screen: {
    readonly cells: readonly string[];
    readonly drugs: readonly string[];
    readonly features: readonly string[];
    readonly n_cells: number;
    readonly n_drugs: number;
    readonly n_features: number;
    readonly n_observed: number;
    readonly n_entries: number;
    readonly observed_fraction: number;
    readonly label: string;
  };
  readonly training: {
    readonly epochs_run: number;
    readonly max_epochs: number;
    readonly stopped_early: boolean;
    readonly observed_rmse: number | null;
    readonly first_epoch_rmse: number | null;
    readonly first_epoch: number;
    readonly first_epoch_ndcg: number | null;
    readonly n_factors: number;
    readonly note: string;
  };
  readonly metrics: Readonly<Record<string, number | null>>;
  readonly candidates: {
    readonly n_profiles: number;
    readonly n_ranked_rows_total: number;
    readonly top_k: number;
    readonly per_profile: Readonly<
      Record<string, readonly RepurposingCandidateRow[]>
    >;
    readonly ordering: string;
  };
  readonly provenance: RepurposingProvenance;
  readonly note: string;
}

// -- the drug-response lane (part13 Phase 5's next remainder item) ---

/** The honest provenance + limitations block (the drugresponse lane). */
export interface DrugResponseProvenance {
  readonly backend: string;
  readonly source: string;
  readonly n_repositories: number;
  readonly shipped_checkpoints: number;
  readonly shipped_checkpoints_note: string;
  readonly screen: string;
  readonly disclaimer: string;
}

/** `GET /api/v1/drugresponse/registry` — the vocabulary answer. */
export interface DrugResponseRegistryResponse {
  readonly model: {
    readonly family: string;
    readonly formulation: string;
    readonly direction: string;
    readonly telemetry_fields: readonly string[];
    readonly history_note: string;
  };
  readonly hyperparameters: {
    readonly n_factors: Readonly<Record<string, number>>;
    readonly lr: Readonly<Record<string, number>>;
    readonly epochs: Readonly<Record<string, number>>;
    readonly l2: Readonly<Record<string, number>>;
    readonly seed: Readonly<Record<string, number>>;
    readonly note: string;
  };
  readonly screen: {
    readonly defaults: Readonly<Record<string, number>>;
    readonly bounds: Readonly<Record<string, string>>;
    readonly label: string;
    readonly scale: string;
  };
  readonly sensitivity_ranking: {
    readonly schema: readonly string[];
    readonly ordering: string;
    readonly source: string;
    readonly response_limits: {
      readonly profiles: string;
      readonly top_k_per_profile: string;
    };
  };
  readonly biomarkers: {
    readonly schema: readonly string[];
    readonly ordering: string;
    readonly source: string;
    readonly response_limits: {
      readonly top_k_per_drug: string;
    };
  };
  readonly cross_validation: {
    readonly n_folds: string;
    readonly rule: string;
    readonly summary: string;
  };
  readonly metrics: {
    readonly keys: readonly string[];
    readonly note: string;
  };
  readonly provenance: DrugResponseProvenance;
  readonly analyze_url: string;
  readonly note: string;
}

/** One ranked sensitivity row — the backend's own schema. */
export interface DrugResponseRankRow {
  readonly drug: string;
  readonly predicted_score: number;
  readonly rank: number;
}

/** One biomarker weight row — the composed `W w_d` readout. */
export interface DrugResponseBiomarkerRow {
  readonly feature: string;
  readonly weight: number;
}

/** One cross-cell-line validation fold. */
export interface DrugResponseCvFold {
  readonly fold: number;
  readonly n_test_cells: number;
  readonly n_observed: number;
  readonly metrics: Readonly<Record<string, number | null>>;
}

/** `POST /api/v1/drugresponse/analyze` — the round-trip answer. */
export interface DrugResponseAnalyzeResponse {
  readonly request: Readonly<Record<string, number>>;
  readonly screen: {
    readonly cells: readonly string[];
    readonly drugs: readonly string[];
    readonly features: readonly string[];
    readonly n_cells: number;
    readonly n_drugs: number;
    readonly n_features: number;
    readonly n_observed: number;
    readonly n_entries: number;
    readonly observed_fraction: number;
    readonly label: string;
  };
  readonly training: {
    readonly epochs_run: number;
    readonly max_epochs: number;
    readonly final_loss: number | null;
    readonly first_loss: number | null;
    readonly n_observed: number;
    readonly pearson_r: number | null;
    readonly n_factors: number;
    readonly note: string;
  };
  readonly metrics: Readonly<Record<string, number | null>>;
  readonly ranking: {
    readonly n_profiles: number;
    readonly n_ranked_rows_total: number;
    readonly top_k: number;
    readonly per_cell: Readonly<
      Record<string, readonly DrugResponseRankRow[]>
    >;
    readonly ordering: string;
  };
  readonly biomarkers: {
    readonly n_drugs: number;
    readonly top_k: number;
    readonly per_drug: Readonly<
      Record<string, readonly DrugResponseBiomarkerRow[]>
    >;
    readonly ordering: string;
  };
  readonly cross_validation: {
    readonly n_folds: number;
    readonly folds: readonly DrugResponseCvFold[];
    readonly summary: Readonly<Record<string, number | null>>;
    readonly note: string;
  };
  readonly provenance: DrugResponseProvenance;
  readonly note: string;
}

// -- the pharmacogenomics lane (part13 Phase 5's next remainder item) ---

/** The honest provenance + limitations block (the pharmacogenomics lane). */
export interface PharmacogenomicsProvenance {
  readonly backend: string;
  readonly source: string;
  readonly n_repositories: number;
  readonly shipped_tables: number;
  readonly shipped_tables_note: string;
  readonly shipped_checkpoints: number;
  readonly shipped_checkpoints_note: string;
  readonly screen: string;
  readonly disclaimer: string;
}

/** `GET /api/v1/pharmacogenomics/registry` — the vocabulary answer. */
export interface PharmacogenomicsRegistryResponse {
  readonly associations: {
    readonly family: string;
    readonly design_matrix: string;
    readonly feasibility_gates: Readonly<Record<string, string>>;
    readonly row_schema: readonly string[];
    readonly ordering: string;
  };
  readonly effect_sizes: {
    readonly keys: readonly string[];
    readonly note: string;
  };
  readonly multiple_testing: {
    readonly methods: readonly string[];
    readonly bh_fdr: string;
    readonly qvalue: string;
    readonly percent_convention: string;
  };
  readonly volcano: {
    readonly transforms: string;
    readonly hit_rule: string;
    readonly verdicts: string;
    readonly additional_fdr_lines: readonly number[];
  };
  readonly evaluation: {
    readonly ruler: string;
    readonly guarded_metrics: readonly string[];
    readonly guards: string;
    readonly normalized_metric: string;
  };
  readonly vocabulary: {
    readonly cna_values: Readonly<Record<string, string>>;
    readonly diploid_zscores: string;
    readonly coexpression: string;
    readonly enrichment_dispatchers: string;
  };
  readonly screen: {
    readonly geometry: Readonly<Record<string, number>>;
    readonly planted_associations: string;
    readonly label: string;
    readonly scale: string;
    readonly missing: string;
  };
  readonly provenance: PharmacogenomicsProvenance;
  readonly analyze_url: string;
  readonly note: string;
}

/** One top-association row — the backend's own gdsctools schema. */
export interface PharmacogenomicsAssociationRow {
  readonly drug: string;
  readonly feature: string;
  readonly n_pos: number;
  readonly n_neg: number;
  readonly mean_pos: number | null;
  readonly mean_neg: number | null;
  readonly delta_mean: number | null;
  readonly effect_size: number | null;
  readonly glass_pos: number | null;
  readonly glass_neg: number | null;
  readonly t_pvalue: number | null;
  readonly pvalue: number | null;
  readonly qvalue: number | null;
  readonly fdr_percent: number | null;
  readonly tissue_pvalue: number | null;
  readonly msi_pvalue: number | null;
  readonly volcano_x: number | null;
  readonly volcano_y: number | null;
  readonly verdict: string | null;
}

/** `POST /api/v1/pharmacogenomics/analyze` — the round-trip answer. */
export interface PharmacogenomicsAnalyzeResponse {
  readonly request: {
    readonly seed: number;
    readonly top_k: number;
    readonly include_msi: boolean;
  };
  readonly screen: {
    readonly n_cells: number;
    readonly n_drugs: number;
    readonly n_features: number;
    readonly drugs: readonly string[];
    readonly features: readonly string[];
    readonly tissues: Readonly<Record<string, number>>;
    readonly msi: { readonly msi_positive: number; readonly msi_negative: number };
    readonly n_observed: number;
    readonly n_entries: number;
    readonly observed_fraction: number;
    readonly label: string;
  };
  readonly associations: {
    readonly n_rows: number;
    readonly n_drugs_tested: number;
    readonly n_features_tested: number;
    readonly top_k: number;
    readonly top: readonly PharmacogenomicsAssociationRow[];
    readonly ordering: string;
  };
  readonly volcano: {
    readonly n_sensitive: number;
    readonly n_resistant: number;
    readonly n_not_significant: number;
    readonly hit_rule: string;
  };
  readonly evaluation: {
    readonly n_observed: number;
    readonly ruler: string;
    readonly guarded: Readonly<Record<string, number | null>>;
    readonly naive_drug_mean: Readonly<Record<string, number | null>>;
    readonly naive_cell_mean: Readonly<Record<string, number | null>>;
    readonly normalized_beyond_drug_mean: Readonly<Record<string, number | null>>;
    readonly guards_note: string;
    readonly normalized_note: string;
  };
  readonly provenance: PharmacogenomicsProvenance;
  readonly note: string;
}

// -- the multi-objective HPO lane (part13 Phase 5's last remainder item) ---

/** The honest provenance + scope block (the multiobjective lane). */
export interface MultiobjectiveProvenance {
  readonly backend: string;
  readonly source: string;
  readonly n_repositories: number;
  readonly transfers_exactly: string;
  readonly stays_upstream: string;
  readonly shipped_checkpoints: number;
  readonly shipped_checkpoints_note: string;
  readonly dataset: string;
  readonly disclaimer: string;
}

/** `GET /api/v1/multiobjective/registry` — the vocabulary answer. */
export interface MultiobjectiveRegistryResponse {
  readonly order: {
    readonly convention: string;
    readonly dominates: string;
    readonly pareto_indices: string;
    readonly complexity: string;
  };
  readonly indicator: {
    readonly family: string;
    readonly reference_contract: string;
    readonly report_reference: string;
    readonly curve: string;
    readonly two_d_only: string;
  };
  readonly spread: {
    readonly family: string;
    readonly rule: string;
    readonly json_note: string;
  };
  readonly decomposition: {
    readonly family: string;
    readonly form: string;
    readonly rule: string;
  };
  readonly study: {
    readonly samplers: readonly string[];
    readonly samplers_note: string;
    readonly weight_modes: Readonly<Record<string, string>>;
    readonly n_objectives: string;
    readonly drift_note: string;
  };
  readonly tunable_objectives: {
    readonly ledger: Readonly<
      Record<string, { readonly cross_validate_key: string; readonly sign: number }>
    >;
    readonly note: string;
    readonly bounds: string;
  };
  readonly report: {
    readonly schema: readonly string[];
    readonly best_for: string;
    readonly prune: string;
  };
  readonly request: {
    readonly defaults: Readonly<Record<string, unknown>>;
    readonly bounds: Readonly<Record<string, string>>;
    readonly dimension_kinds: readonly string[];
    readonly model_note: string;
  };
  readonly dataset: {
    readonly geometry: Readonly<Record<string, number>>;
    readonly label: string;
  };
  readonly provenance: MultiobjectiveProvenance;
  readonly optimize_url: string;
  readonly note: string;
}

/**
 * One trial row — the backend's MOTrial projection. Front and
 * shortlist rows carry the aligned crowding (null for the +inf
 * boundary members, `crowding_is_boundary` marks them); the
 * `best_for` endpoint rows deliberately carry no crowding claim.
 */
export interface MultiobjectiveTrialRow {
  readonly number: number;
  readonly params: Readonly<Record<string, number>>;
  readonly values: readonly number[];
  readonly weights: readonly number[];
  readonly scalar: number | null;
  readonly crowding?: number | null;
  readonly crowding_is_boundary?: boolean;
}

/** `POST /api/v1/multiobjective/optimize` — the round-trip answer. */
export interface MultiobjectiveOptimizeResponse {
  readonly request: {
    readonly model: string;
    readonly task: string;
    readonly objectives: readonly string[];
    readonly dimensions: readonly {
      readonly name: string;
      readonly kind: string;
      readonly low: number;
      readonly high: number;
    }[];
    readonly n_trials: number;
    readonly cv: number;
    readonly sampler: string;
    readonly seed: number;
    readonly weight_mode: string;
    readonly n_weight_vectors: number;
    readonly shortlist_keep: number;
  };
  readonly dataset: {
    readonly n_samples: number;
    readonly n_features: number;
    readonly task: string;
    readonly label: string;
  };
  readonly study: {
    readonly n_trials: number;
    readonly n_objectives: number;
    readonly objective_names: readonly string[];
    readonly sampler: string;
    readonly weight_mode: string;
    readonly n_weight_vectors: number;
    readonly n_front: number;
    readonly objectives_note: string;
  };
  readonly front: readonly MultiobjectiveTrialRow[];
  readonly crowding_note: string;
  readonly indicator: {
    readonly ref: readonly number[] | null;
    readonly hypervolume: number | null;
    readonly hypervolume_curve: readonly number[];
    readonly two_d_only: string;
    readonly curve_note: string;
  };
  readonly best_for: Readonly<Record<string, MultiobjectiveTrialRow>>;
  readonly best_for_note: string;
  readonly shortlist: {
    readonly keep: number;
    readonly n_front: number;
    readonly n_trials: number;
    readonly trials: readonly MultiobjectiveTrialRow[];
    readonly note: string;
  };
  readonly provenance: MultiobjectiveProvenance;
  readonly timings_note: string;
  readonly note: string;
}

// -- the generic multi-response ML lane (part17 §5 — the M1 lane) ------

/** One strategy row of the multiresponse registry (the mechanism is
 * stated per strategy, never implied). */
export interface MultiresponseStrategy {
  readonly key: string;
  readonly label: string;
  readonly mechanism: "native" | "wrapper" | "chain";
  readonly native_multi_output: boolean;
  readonly sample_weight_support: boolean;
  readonly notes: string;
  readonly base_keys: readonly string[];
  readonly default_base: string | null;
}

/** `GET /api/v1/multiresponse/registry` — the vocabulary answer. */
export interface MultiresponseRegistryResponse {
  readonly strategies: readonly MultiresponseStrategy[];
  readonly missing_strategies: Readonly<
    Record<string, {
      readonly label: string;
      readonly description: string;
      readonly mechanisms: readonly string[];
    }>
  >;
  readonly metrics: readonly string[];
  readonly max_sample_rows: number;
  readonly demo_dataset: {
    readonly features: readonly string[];
    readonly targets: readonly string[];
    readonly n_rows: number;
    readonly synthetic: boolean;
    readonly seed: number;
    readonly note: string;
  };
  readonly bounds: Readonly<Record<string, number | readonly number[]>>;
  readonly scope_statement: string;
}

/** One per-target metric block (R2/RMSE/MAE/MSE/explained variance,
 * each with its own test-row count — never hidden behind the
 * aggregate). */
export interface MultiresponsePerTargetMetrics {
  readonly r2: number;
  readonly rmse: number;
  readonly mae: number;
  readonly mse: number;
  readonly explained_variance: number;
  readonly n_test_rows: number;
}

/** One target's residual sample (bounded, the bound reported). */
export interface MultiresponseResidualBlock {
  readonly actual: readonly number[];
  readonly predicted: readonly number[];
  readonly residual: readonly number[];
  readonly n_rows: number;
  readonly n_sampled: number;
}

/** `POST /api/v1/multiresponse/train` — the round-trip answer. */
export interface MultiresponseTrainResponse {
  readonly model: MultiresponseModelView;
  readonly run_id: string;
  readonly provenance: {
    readonly source: string;
    readonly dataset_id: string | null;
    readonly dataset_name: string;
    readonly synthetic: boolean;
    readonly note?: string;
  };
  readonly configuration: {
    readonly strategy: string;
    readonly mechanism: string;
    readonly params: Readonly<Record<string, unknown>>;
    readonly preprocess: boolean;
    readonly target_scaling: boolean;
    readonly test_size: number;
    readonly cv_folds: number;
    readonly missing: string;
    readonly group_column: string | null;
    readonly sample_weight_column: string | null;
    readonly target_weights: readonly number[] | null;
    readonly random_state: number;
  };
  readonly data: {
    readonly n_rows: number;
    readonly usable_rows_per_target: Readonly<Record<string, number>>;
    readonly missing_counts_per_target: Readonly<Record<string, number>>;
    readonly dropped_train_rows: number;
    readonly split: {
      readonly strategy: string;
      readonly train_rows: number;
      readonly test_rows: number;
      readonly n_train_groups: number | null;
      readonly n_test_groups: number | null;
      readonly group_overlap: readonly string[];
      readonly test_size: number;
      readonly random_state: number;
    };
  };
  readonly per_target_metrics: Readonly<
    Record<string, MultiresponsePerTargetMetrics>
  >;
  readonly aggregate_metrics: {
    readonly uniform: Readonly<Record<string, number>>;
    readonly weighted: Readonly<Record<string, number>>;
  };
  readonly residuals: {
    readonly per_target: Readonly<
      Record<string, MultiresponseResidualBlock>
    >;
    readonly max_rows: number;
  };
  readonly target_correlation_matrix: Readonly<
    Record<string, Readonly<Record<string, number | null>>>
  >;
  readonly cv: {
    readonly folds: number;
    readonly per_target: Readonly<
      Record<string, {
        readonly r2_mean: number;
        readonly r2_std: number;
        readonly r2_folds: readonly number[];
      }>
    >;
    readonly aggregate: {
      readonly r2_uniform: number;
      readonly r2_weighted: number;
    };
    readonly folds_table: readonly Record<string, number>[];
  };
  readonly contract: Readonly<Record<string, unknown>>;
  readonly fit_seconds: number;
  readonly max_sample_rows: number;
}

/** One trained multiresponse model's view (the comparison surface). */
export interface MultiresponseModelView {
  readonly model_id: string;
  readonly name: string;
  readonly task: string;
  readonly dataset_id: string | null;
  readonly target: string;
  readonly features: readonly string[];
  readonly n_train_rows: number;
  readonly cv_mean: number | null;
  readonly fit_seconds: number;
  readonly created_at: string;
  readonly multiresponse: {
    readonly target_names: readonly string[];
    readonly target_order: readonly string[];
    readonly mechanism: string;
    readonly missing_strategy: string;
    readonly per_target_metrics:
      | Readonly<Record<string, MultiresponsePerTargetMetrics>>
      | null;
    readonly aggregate_metrics: {
      readonly uniform: Readonly<Record<string, number>>;
      readonly weighted: Readonly<Record<string, number>>;
    } | null;
    readonly n_test_rows: number | null;
  };
}

/** `POST /api/v1/multiresponse/predict` — the ordered prediction
 * answer (the STABLE contract target order). */
export interface MultiresponsePredictResponse {
  readonly model_id: string;
  readonly target_names: readonly string[];
  readonly target_units: Readonly<Record<string, string>>;
  readonly predictions: readonly Readonly<Record<string, number>>[];
  readonly n_rows: number;
  readonly note: string;
}

// -- the Response Surface Methodology lane (part17 §6 — the R1 lane) --

/** One design generator row of the RSM registry (the honest notes stated
 * per design — the rotatable CCD's axial runs sit OUTSIDE the declared
 * ranges, disclosed). */
export interface RsmDesignInfo {
  readonly key: "ccd" | "box_behnken" | "full_factorial";
  readonly label: string;
  readonly alpha_types: readonly string[] | null;
  readonly min_factors: number;
  readonly max_factors: number;
  readonly runs_formula: string;
  readonly blocking: boolean;
  readonly note: string;
}

/** One declared factor's view (center + half-range ride the coding). */
export interface RsmFactorView {
  readonly name: string;
  readonly low: number;
  readonly high: number;
  readonly unit: string | null;
  readonly center: number;
  readonly half_range: number;
}

/** `GET /api/v1/rsm/registry` — the vocabulary answer. */
export interface RsmRegistryResponse {
  readonly designs: readonly RsmDesignInfo[];
  readonly goals: readonly string[];
  readonly model: {
    readonly order: string;
    readonly terms: string;
    readonly term_count_formula: string;
  };
  readonly bounds: Readonly<Record<string, unknown>>;
  readonly demo_dataset: {
    readonly name: string;
    readonly synthetic: boolean;
    readonly factors: readonly RsmFactorView[];
    readonly responses: readonly string[];
    readonly runs: number;
    readonly design: string;
    readonly noise: boolean;
    readonly note: string;
  };
  readonly scope_statement: string;
}

/** One run row of a generated design (coded AND actual levels). */
export interface RsmDesignRunRow {
  readonly run_id: string;
  readonly run_order: number;
  readonly block: number;
  readonly run_type: string;
  readonly coded: Readonly<Record<string, number>>;
  readonly actual: Readonly<Record<string, number>>;
}

/** `POST /api/v1/rsm/design` — the generated experiment sheet. */
export interface RsmDesignResponse {
  readonly factors: readonly RsmFactorView[];
  readonly design: {
    readonly type: string;
    readonly alpha: number | null;
    readonly alpha_type: string | null;
    readonly n_factors: number;
    readonly runs: number;
    readonly center_points: number;
    readonly replicates: number;
    readonly randomized: boolean;
    readonly seed: number;
    readonly blocks: boolean;
  };
  readonly runs: readonly RsmDesignRunRow[];
  readonly run_type_counts: Readonly<Record<string, number>>;
  readonly assumptions: readonly string[];
  readonly export_note: string;
}

/** One coefficient row of the second-order fit. */
export interface RsmCoefficientRow {
  readonly term: string;
  readonly label: string;
  readonly kind: string;
  readonly estimate: number;
  readonly std_error: number | null;
  readonly t: number | null;
  readonly p: number | null;
  readonly ci_low: number | null;
  readonly ci_high: number | null;
}

/** One fitted-vs-observed diagnostics row (leverage, studentised
 * residual, Cook's distance — flags reported, never removals). */
export interface RsmDiagnosticsRow {
  readonly run: number;
  readonly observed: number;
  readonly fitted: number;
  readonly residual: number;
  readonly studentized: number;
  readonly leverage: number;
  readonly cooks_distance: number;
  readonly outlier_flag: boolean;
  readonly influential_flag: boolean;
}

/** The per-response fit statistics block. */
export interface RsmFitStatistics {
  readonly n_runs: number;
  readonly df_residual: number;
  readonly rank: number;
  readonly r2: number;
  readonly adj_r2: number;
  readonly pred_r2: number;
  readonly rmse: number;
  readonly press: number;
  readonly confidence: number;
  readonly outliers_flagged: number;
  readonly influential_flagged: number;
  readonly cooks_threshold: number;
  readonly machine_precision_fit: boolean;
  readonly diagnostics_note: string | null;
  readonly thresholds_note: string;
}

/** One trained RSM model's view (the comparison surface). */
export interface RsmModelView {
  readonly model_id: string;
  readonly name: string;
  readonly task: string;
  readonly dataset_id: string | null;
  readonly target: string;
  readonly features: readonly string[];
  readonly n_train_rows: number;
  readonly cv_mean: number | null;
  readonly fit_seconds: number;
  readonly created_at: string;
  readonly rsm: {
    readonly kind: string;
    readonly factors: readonly string[];
    readonly responses: readonly string[];
    readonly n_terms: number;
    readonly design_meta: Readonly<Record<string, unknown>>;
    readonly df_residual: number;
    readonly rank: number;
  };
}

/** `POST /api/v1/rsm/fit` — the full second-order fit answer. */
export interface RsmFitResponse {
  readonly model: RsmModelView;
  readonly run_id: string;
  readonly provenance: {
    readonly source: string;
    readonly dataset_id: string | null;
    readonly dataset_name: string;
    readonly synthetic: boolean;
    readonly note?: string;
  };
  readonly configuration: {
    readonly factors: readonly RsmFactorView[];
    readonly response_columns: readonly string[];
    readonly design_meta: Readonly<Record<string, unknown>>;
    readonly confidence: number;
  };
  readonly data: {
    readonly n_runs: number;
    readonly dropped_rows: number;
    readonly terms: readonly { key: string; kind: string; label: string }[];
    readonly hierarchy_notes: readonly string[];
    readonly aliasing: {
      readonly rank: number;
      readonly columns: number;
      readonly aliased: readonly string[] | null;
      readonly note: string;
    } | null;
  };
  readonly coefficients: Readonly<Record<string, readonly RsmCoefficientRow[]>>;
  readonly anova: Readonly<
    Record<string, Readonly<Record<string, unknown>>>
  >;
  readonly diagnostics: Readonly<
    Record<string, readonly RsmDiagnosticsRow[]>
  >;
  readonly fit_statistics: Readonly<Record<string, RsmFitStatistics>>;
  readonly fit_seconds: number;
}

/** `POST /api/v1/rsm/predict` — the per-response predictions with the
 * CI/PI and the honest zone vocabulary per point. */
export interface RsmPredictResponse {
  readonly model_id: string;
  readonly rows: readonly {
    readonly run: number;
    readonly zone: "inside_ranges" | "axial_region" | "outside_design";
    readonly coded: Readonly<Record<string, number>>;
    readonly actual: Readonly<Record<string, number>>;
    readonly [response: string]:
      | number
      | string
      | Readonly<Record<string, number>>;
  }[];
  readonly responses: readonly string[];
  readonly confidence: number;
  readonly df_residual: number;
  readonly extrapolation: {
    readonly count: number;
    readonly runs: readonly number[];
    readonly vocabulary: Readonly<Record<string, string>> | null;
  };
  readonly note: string;
}

/** `POST /api/v1/rsm/contour` — the two-factor grid payload (the
 * Plotly contour / 3-D surface data; the rows index factor_x). */
export interface RsmContourResponse {
  readonly model_id: string;
  readonly factor_x: {
    readonly name: string;
    readonly unit: string | null;
    readonly coded: readonly number[];
    readonly actual: readonly number[];
  };
  readonly factor_y: {
    readonly name: string;
    readonly unit: string | null;
    readonly coded: readonly number[];
    readonly actual: readonly number[];
  };
  readonly fixed: Readonly<
    Record<string, { actual: number; coded: number }>
  >;
  readonly grid_points: number;
  readonly z_name: string;
  readonly z: readonly (readonly number[])[];
  readonly note: string;
  readonly units_note: string;
}

/** One desirability goal (the §6.6 contract). */
export interface RsmGoalSpec {
  readonly response: string;
  readonly goal: "maximize" | "minimize" | "target" | "range";
  readonly low: number | null;
  readonly high: number | null;
  readonly target: number | null;
  readonly weight: number;
  readonly exponent: number;
}

/** One per-response block of the optimisation answer. */
export interface RsmOptimumResponseBlock {
  readonly goal: string;
  readonly prediction: number;
  readonly individual_desirability: number;
  readonly weight: number;
  readonly exponent: number;
  readonly bounds: Readonly<Record<string, number>>;
  readonly bounds_source: string;
}

/** One Pareto alternative (the curated deterministic sample —
 * honestly NOT the exhaustive front). */
export interface RsmParetoAlternative {
  readonly kind: "individual" | "weight_tilted";
  readonly response: string;
  readonly goal: string;
  readonly combined_desirability: number | null;
  readonly note: string;
  readonly x_coded: Readonly<Record<string, number>>;
  readonly x_actual: Readonly<Record<string, number>>;
  readonly predictions: Readonly<Record<string, number>>;
}

/** `POST /api/v1/rsm/optimize` — the desirability optimisation answer. */
export interface RsmOptimizeResponse {
  readonly model_id: string;
  readonly candidate: {
    readonly x_coded: Readonly<Record<string, number>>;
    readonly x_actual: Readonly<Record<string, number>>;
    readonly combined_desirability: number;
    readonly responses: Readonly<Record<string, RsmOptimumResponseBlock>>;
  };
  readonly weights: Readonly<Record<string, number>>;
  readonly total_weight: number;
  readonly constraints: readonly {
    readonly coefficients: readonly number[];
    readonly sense: string;
    readonly bound: number;
  }[];
  readonly sensitivity: readonly {
    readonly factor: string;
    readonly step_coded: number;
    readonly low: { coded: number; combined_desirability: number; delta: number };
    readonly high: { coded: number; combined_desirability: number; delta: number };
  }[];
  readonly sensitivity_note: string;
  readonly pareto_alternatives: readonly RsmParetoAlternative[];
  readonly pareto_note: string;
  readonly feasible_grid: {
    readonly factor_x: string;
    readonly factor_y: string;
    readonly grid_points: number;
    readonly other_factors_at: Readonly<Record<string, number>>;
    readonly feasible: readonly (readonly boolean[])[];
    readonly note: string;
  } | null;
  readonly region_note: string;
  readonly bounds_source_note: string;
  readonly goals_vocabulary: readonly string[];
}

// -- the workflow engine lane (part13 Phase 6, §8 — the round-323 GUI
// leg over the r322 /api/v1/workflow family) --------------------------

/** One typed input/output port of a workflow step (the registry's own). */
export interface WorkflowPortView {
  readonly name: string;
  readonly kind: string;
  readonly required: boolean;
  readonly description: string;
}

/** One tunable parameter of a workflow step (the registry's own dial). */
export interface WorkflowParamView {
  readonly name: string;
  readonly default: unknown;
  readonly description: string;
}

/** One step of the registry: the closed typed contract + the REAL
 * backend it delegates to (never reimplemented here). */
export interface WorkflowStepView {
  readonly step_id: string;
  readonly title: string;
  readonly description: string;
  readonly backend: string;
  readonly workspace: string;
  readonly inputs: readonly WorkflowPortView[];
  readonly outputs: readonly WorkflowPortView[];
  readonly parameters: readonly WorkflowParamView[];
  readonly notes: string;
}

/** `GET /api/v1/workflow/steps` — the typed vocabulary, verbatim. */
export interface WorkflowStepsResponse {
  readonly engine: {
    readonly layer: string;
    readonly decision: string;
    readonly port_kinds: readonly string[];
    readonly job_integration: string;
  };
  readonly steps: readonly WorkflowStepView[];
}

/** One saved workflow template (§8's own example chain among them). */
export interface WorkflowTemplateView {
  readonly template_id: string;
  readonly name: string;
  readonly description: string;
  readonly node_count: number;
  readonly nodes: readonly {
    readonly node_id: string;
    readonly step_id: string;
  }[];
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly spec: Readonly<Record<string, unknown>>;
}

/** `GET /api/v1/workflow/templates` — the template index. */
export interface WorkflowTemplatesResponse {
  readonly templates: readonly WorkflowTemplateView[];
}

/** `POST /api/v1/workflow/validate` — the pre-execution gate's answer:
 * every finding at once, the topological order, the spec digest. */
export interface WorkflowValidateResponse {
  readonly valid: boolean;
  readonly errors: readonly string[];
  readonly topological_order: readonly string[] | null;
  readonly spec_digest: string;
  readonly node_count: number;
}

/** One output artifact + its honest lineage (§8's artifact lineage —
 * the digest_kind says what the digest is, never pretends). */
export interface WorkflowArtifactView {
  readonly node: string;
  readonly port: string;
  readonly kind: string;
  readonly digest: string;
  readonly digest_kind: string;
  readonly lineage: readonly {
    readonly node: string;
    readonly port: string;
  }[];
}

/** One node's execution verdict (the partial-failure ledger). */
export interface WorkflowNodeResultView {
  readonly node_id: string;
  readonly step_id: string;
  readonly state: string;
  readonly started_at: string | null;
  readonly duration_ms: number | null;
  readonly error: string | null;
  readonly skipped_reason: string | null;
  readonly artifacts: readonly WorkflowArtifactView[];
}

/** One execution's full ledger — the run record the engine answers. */
export interface WorkflowRunRecordView {
  readonly run_id: string;
  readonly workflow_name: string;
  readonly spec_digest: string;
  readonly state: string;
  readonly seed: number;
  readonly started_at: string;
  readonly completed_at: string | null;
  readonly reproducibility: {
    readonly toolkit_version: string;
    readonly python: string;
    readonly implementation: string;
    readonly platform: string;
    readonly seed: number;
    readonly spec_digest: string;
    readonly determinism_note: string;
  };
  readonly node_counts: Readonly<Record<string, number>>;
  readonly nodes: readonly WorkflowNodeResultView[];
}

/** `POST /api/v1/workflow/run` sync mode — the run record inline. */
export interface WorkflowRunSyncResponse {
  readonly mode: "sync";
  readonly run: WorkflowRunRecordView;
}

/** `POST /api/v1/workflow/run` job mode — the JobManager lane. */
export interface WorkflowRunJobResponse {
  readonly mode: "job";
  readonly job_id: string;
  readonly job_state: string;
  readonly note: string;
}

export type WorkflowRunResponse =
  | WorkflowRunSyncResponse
  | WorkflowRunJobResponse;

/** One row of the bounded in-session run ledger. */
export interface WorkflowRunsLedgerRow {
  readonly run_id: string;
  readonly workflow_name: string;
  readonly state: string;
  readonly seed: number;
  readonly started_at: string;
  readonly completed_at: string | null;
  readonly node_counts: Readonly<Record<string, number>>;
}

/** `GET /api/v1/workflow/runs` — the execution history + the honest
 * lifetime note. */
export interface WorkflowRunsResponse {
  readonly runs: readonly WorkflowRunsLedgerRow[];
  readonly note: string;
}

/**
 * The run route's own bounds (workflow.py's `_MAX_NODES_SYNC` /
 * `_MAX_NODES_JOB`) — mirrored once beside the client so the builder
 * states the doomed request honestly instead of firing it.
 */
export const WORKFLOW_SYNC_MAX_NODES = 12;
export const WORKFLOW_JOB_MAX_NODES = 24;
/**
 * The synthetic stand-in's own record bounds (workflow.py's
 * `_MAX_RECORDS_SYNC` / `_MAX_RECORDS_JOB` intersected with the
 * request dial `n_synthetic`'s 12..30 range).
 */
export const WORKFLOW_MIN_SYNTHETIC = 12;
export const WORKFLOW_MAX_SYNTHETIC = 30;
/** The route's own seed ceiling (`_MAX_SEED`, a signed 32-bit int). */
export const WORKFLOW_MAX_SEED = 2147483647;

export type CatalogTaskFilter = "classification" | "regression" | "both";

export const mlApi = {
  /**
   * `GET /api/v1/ml/models?task=…&only_available=…` — the registry
   * catalog projection. `task` must stay synced with the selected
   * dataset's own task (the backend filters classification /
   * regression / both); `undefined` lists the WHOLE catalog — the
   * §15.3 registry view over every entry, both tasks, the
   * availability stated per row.
   */
  async catalog(
    task?: CatalogTaskFilter,
    onlyAvailable: boolean = true,
  ): Promise<ModelCatalogResponse> {
    const query =
      task === undefined
        ? `only_available=${onlyAvailable ? "true" : "false"}`
        : `task=${encodeURIComponent(task)}&only_available=${
            onlyAvailable ? "true" : "false"
          }`;
    const raw = await api.get<Record<string, unknown>>(
      `/api/v1/ml/models?${query}`,
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
   * `DELETE /api/v1/ml/trained/{model_id}` — the registry's deletion
   * lane (part16 §15.3): the bounded store's own evict (the in-memory
   * record, the persisted row + vault payload when the durability
   * lane is on). The answer carries the remaining count so the GUI
   * refreshes honestly without a second listing call; the §15.2 run
   * lineage and exported artifacts are untouched by scope.
   */
  async evictTrained(modelId: string): Promise<TrainedModelEvicted> {
    const raw = await api.delete<Record<string, unknown>>(
      `/api/v1/ml/trained/${encodeURIComponent(modelId)}`,
    );
    if (typeof raw["evicted"] !== "string") {
      throw new Error("the evict answer's evicted field is not a string");
    }
    if (typeof raw["remaining"] !== "number") {
      throw new Error("the evict answer's remaining field is not a number");
    }
    return {
      evicted: raw["evicted"] as string,
      remaining: raw["remaining"] as number,
    };
  },

  /**
   * `POST /api/v1/artifacts/model` — the registry's export lane
   * (§15.3: export): the session-trained estimator persisted through
   * the REAL `save_model` (the backward-compatible joblib format) as
   * a registered artifact — the digest-verified download then rides
   * the artifacts client's own lane, and the exported copy outlives
   * the model row itself.
   */
  async exportTrained(
    modelId: string,
    filename?: string,
  ): Promise<ArtifactView> {
    const body: Record<string, unknown> = { model_id: modelId };
    if (filename !== undefined) {
      body["filename"] = filename;
    }
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/artifacts/model",
      body,
    );
    return parseArtifactView(raw["artifact"], `export of ${modelId}`);
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
      // ML-3 — the explicit null keeps the legacy full-frame lane the
      // pydantic-equal default (the DS-1 shape-consistency pattern);
      // a section replaces it below when one is requested
      split: null,
      mode: options.mode ?? "job",
    };
    const split = options.split ?? null;
    if (split !== null) {
      const section: TrainSplitSpec = {
        strategy: split.strategy,
        smiles_column: split.smilesColumn ?? null,
        group_column: split.groupColumn ?? null,
        test_size: split.testSize,
      };
      request.split = section;
    }
    const raw = (await api.post("/api/v1/ml/train", request)) as Record<
      string,
      unknown
    >;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        model: parseTrainedModelView(raw["model"], "train(sync)"),
        split: parseTrainSplitBlock(raw["split"], "train(sync) split"),
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

  /**
   * `GET /api/v1/dta/registry` — the DeepDTA encoding registry
   * (part13 Phase 5's DTA lane): the three vendored alphabets
   * (25/64/62 symbols), the 20 standard amino acids, the AAC +
   * dipeptide panel arithmetic, the 2048-bit ECFP pairing, the rm2
   * metric, the demo frame, and the honest provenance block — every
   * cell the backend's own, never re-declared.
   */
  async dtaRegistry(): Promise<DtaRegistryResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/dta/registry",
    );
    return parseDtaRegistryResponse(raw, "dta registry");
  },

  /**
   * `POST /api/v1/dta/features` — the paired featurization over a
   * (smiles, protein, optional affinity) batch (sync-only light
   * compute, the chemistry-descriptors precedent): the per-row
   * protein composition cards, the kept/dropped row accounting, and
   * the batch-level matrix shape. Sequence errors and non-finite
   * affinities are the backend's own hard 422s — surfaced verbatim,
   * never softened; invalid ligands are dropped together with their
   * protein and affinity and report themselves per-row.
   */
  async dtaFeatures(
    pairs: readonly {
      smiles: string;
      protein: string;
      affinity?: number | null;
    }[],
    dipeptides: boolean = false,
  ): Promise<DtaFeaturesResponse> {
    const request: DtaFeaturesRequest = {
      pairs: pairs.map((pair) => ({
        smiles: pair.smiles,
        protein: pair.protein,
        affinity: pair.affinity ?? null,
      })),
      dipeptides,
    };
    const raw = (await api.post(
      "/api/v1/dta/features",
      request,
    )) as Record<string, unknown>;
    return parseDtaFeaturesResponse(raw, "dta features");
  },

  /**
   * `GET /api/v1/repurposing/registry` — the drug-repurposing
   * lane's own vocabulary (part13 Phase 5): DeepResponse's split
   * ladder, the CaDRReS formulation, the fit hyperparameters + the
   * route's sync-light caps, the ten-key metric contract, the
   * repurposing table schema, and the honest provenance block —
   * every cell the backend's own, never re-declared.
   */
  async repurposingRegistry(): Promise<RepurposingRegistryResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/repurposing/registry",
    );
    return parseRepurposingRegistryResponse(raw, "repurposing registry");
  },

  /**
   * `POST /api/v1/repurposing/screen` — the full honest round trip
   * over the real backend (sync-only light compute, bounded dense
   * matrices): the seeded SYNTHETIC screen, the CaDRReS fit, the
   * predicted IC50 matrix, DeepResponse's metric contract on the
   * observed entries, and the ranked repurposing candidates per
   * profile. Every predicted value is a labelled model estimate over
   * the synthetic screen — never real pharmacology, never a clinical
   * recommendation.
   */
  async repurposingScreen(
    params: Partial<RepurposingScreenRequest> = {},
  ): Promise<RepurposingScreenResponse> {
    const raw = (await api.post(
      "/api/v1/repurposing/screen",
      params,
    )) as Record<string, unknown>;
    return parseRepurposingScreenResponse(raw, "repurposing screen");
  },

  /**
   * `GET /api/v1/drugresponse/registry` — the cancer cell-line
   * drug-response lane's own vocabulary (part13 Phase 5): the
   * CaDRReS bilinear formulation with its sigmoid link, the fit
   * hyperparameters + the route's sync-light caps, the five-key
   * metric contract, the sensitivity-ranking and biomarker schemas,
   * the cross-cell-line validation rule, and the honest provenance
   * block — every cell the backend's own, never re-declared.
   */
  async drugResponseRegistry(): Promise<DrugResponseRegistryResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/drugresponse/registry",
    );
    return parseDrugResponseRegistryResponse(raw, "drugresponse registry");
  },

  /**
   * `POST /api/v1/drugresponse/analyze` — the full honest round trip
   * over the real backend (sync-only light compute, bounded dense
   * matrices): the seeded SYNTHETIC screen, the bilinear fit, the
   * predicted response-score matrix, the metric contract on the
   * observed entries, the per-cell sensitivity ranking, the per-drug
   * biomarker weights, and the cross-cell-line validation folds. One
   * response scale only; every predicted value is a labelled model
   * estimate over the synthetic screen — never real pharmacology,
   * never a clinical recommendation.
   */
  async drugResponseAnalyze(
    params: Partial<DrugResponseAnalyzeRequest> = {},
  ): Promise<DrugResponseAnalyzeResponse> {
    const raw = (await api.post(
      "/api/v1/drugresponse/analyze",
      params,
    )) as Record<string, unknown>;
    return parseDrugResponseAnalyzeResponse(raw, "drugresponse analyze");
  },

  /**
   * `GET /api/v1/pharmacogenomics/registry` — the pharmacogenomic-
   * association lane's own vocabulary (part13 Phase 5): the classic
   * GDSC ANOVA design matrix with its QR F-test and feasibility
   * gates, the effect-size contract, the multiple-testing layer,
   * the volcano transforms, the evaluation guards + the
   * mean-effects ruler, the cBioPortal vocabularies, and the
   * honest provenance block — every cell the backend's own, never
   * re-declared.
   */
  async pharmacogenomicsRegistry(): Promise<PharmacogenomicsRegistryResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/pharmacogenomics/registry",
    );
    return parsePharmacogenomicsRegistryResponse(
      raw,
      "pharmacogenomics registry",
    );
  },

  /**
   * `POST /api/v1/pharmacogenomics/analyze` — the full honest round
   * trip over the real backend (sync-only light compute, a 40x12
   * screen): the seeded SYNTHETIC screen, the real GDSC ANOVA over
   * every feasible drug x feature pair, the q-value/FDR columns,
   * the volcano verdicts, the top associations table, and the
   * mean-effects ruler evaluated through the guarded family with
   * the normalized score beyond the naive drug-mean ruler. One
   * response scale only (the natural-log µM scale); every p-value
   * and effect size is a labelled statistic over the synthetic
   * screen — never real pharmacology, never a clinical
   * recommendation.
   */
  async pharmacogenomicsAnalyze(
    params: Partial<PharmacogenomicsAnalyzeRequest> = {},
  ): Promise<PharmacogenomicsAnalyzeResponse> {
    const raw = (await api.post(
      "/api/v1/pharmacogenomics/analyze",
      params,
    )) as Record<string, unknown>;
    return parsePharmacogenomicsAnalyzeResponse(
      raw,
      "pharmacogenomics analyze",
    );
  },

  /**
   * `GET /api/v1/multiobjective/registry` — the multi-objective HPO
   * lane's own vocabulary (part13 Phase 5's last remainder item): the
   * Pareto order + the optuna minimize convention, the exact 2-D
   * hypervolume with the WFG reference contract, NSGA-II's crowding
   * with the +inf-boundary-null convention, the MOEA/D Tchebycheff
   * decomposition, the study's weight supply, the tunable-objective
   * ledger with its cross_validate keys and sign flips, and the
   * honest scope block — every cell the backend's own, never
   * re-declared.
   */
  async multiobjectiveRegistry(): Promise<MultiobjectiveRegistryResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/multiobjective/registry",
    );
    return parseMultiobjectiveRegistryResponse(
      raw,
      "multiobjective registry",
    );
  },

  /**
   * `POST /api/v1/multiobjective/optimize` — the full honest round
   * trip over the real backend (sync-only light compute): the seeded
   * SYNTHETIC 80x6 frame, the request's own search space, the real
   * cross-validated study over the sklearn registry model, the
   * non-dominated front with its aligned crowding, the 2-D indicator
   * + the monotone hypervolume curve (null for a 3-objective study —
   * the exact sweep is 2-D only, never a fabricated indicator), the
   * per-objective trade-off corners, and the crowding-pruned
   * shortlist. Every objective value is a labelled statistic over
   * the synthetic dataset; the fold timings vary run-to-run (the
   * honest drift note rides every answer).
   */
  async multiobjectiveOptimize(
    params: Partial<MultiobjectiveOptimizeRequest> = {},
  ): Promise<MultiobjectiveOptimizeResponse> {
    const raw = (await api.post(
      "/api/v1/multiobjective/optimize",
      params,
    )) as Record<string, unknown>;
    return parseMultiobjectiveOptimizeResponse(
      raw,
      "multiobjective optimize",
    );
  },

  /**
   * `GET /api/v1/multiresponse/registry` — the generic multi-response
   * lane's own vocabulary (part17 §5, the M1 lane): the strategy
   * registry with each mechanism STATED (native 2-D Y / wrapper
   * per-target clones / chain's predicted-target feed), the
   * missing-value strategy lanes (complete_case always; per_target
   * the wrapper's masked multi-task), the metric names, the seeded
   * SYNTHETIC demo dataset descriptor, the bounds, and the honest
   * scope statement.
   */
  async multiresponseRegistry(): Promise<MultiresponseRegistryResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/multiresponse/registry",
    );
    return parseMultiresponseRegistryResponse(
      raw,
      "multiresponse registry",
    );
  },

  /**
   * `POST /api/v1/multiresponse/train` — the full round trip over the
   * real backend (sync-only light compute): a registered typed
   * dataset or the seeded SYNTHETIC demo frame, the request's own
   * feature and >=2 target column selections, the split-first
   * leakage-safe trainer, the per-target metrics (never hidden
   * behind the uniform/weighted aggregates reported alongside), the
   * bounded residual samples, the target correlation matrix, the
   * k-fold CV summary, and the complete §5.6 training contract; the
   * fitted bundle registers in the session model store and the run
   * records in the §15.2 lineage.
   */
  async multiresponseTrain(
    params: Record<string, unknown>,
  ): Promise<MultiresponseTrainResponse> {
    const raw = (await api.post(
      "/api/v1/multiresponse/train",
      params,
    )) as Record<string, unknown>;
    return parseMultiresponseTrainResponse(
      raw,
      "multiresponse train",
    );
  },

  /**
   * `POST /api/v1/multiresponse/predict` — multi-column predictions
   * in the STABLE contract target order with per-target inverse
   * scaling (raw units), from inline rows.
   */
  async multiresponsePredict(
    params: Record<string, unknown>,
  ): Promise<MultiresponsePredictResponse> {
    const raw = (await api.post(
      "/api/v1/multiresponse/predict",
      params,
    )) as Record<string, unknown>;
    return parseMultiresponsePredictResponse(
      raw,
      "multiresponse predict",
    );
  },

  /**
   * `GET /api/v1/multiresponse/models` — the session's multi-response
   * models (the comparison surface: per-target metric blocks side
   * by side).
   */
  async multiresponseModels(): Promise<{
    models: readonly MultiresponseModelView[];
    metrics: readonly string[];
  }> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/multiresponse/models",
    );
    return parseMultiresponseModelsResponse(raw, "multiresponse models");
  },

  /**
   * `POST /api/v1/artifacts/model` — the multiresponse lane's export:
   * the trained MultiResponseBundle persisted through the REAL
   * `save_model` as a digest-verified joblib artifact (an independent
   * copy that outlives the model; the reload path is the house
   * `load_model`).
   */
  async multiresponseExport(
    modelId: string,
  ): Promise<ArtifactView> {
    const raw = (await api.post(
      "/api/v1/artifacts/model",
      { model_id: modelId, filename: `multiresponse-${modelId}` },
    )) as Record<string, unknown>;
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the model lane answered without an artifact");
    }
    return parseArtifactView(artifact, "multiresponse model artifact");
  },

  /**
   * `GET /api/v1/rsm/registry` — the Response Surface Methodology
   * lane's own vocabulary (part17 §6, the R1 lane): the three design
   * generators (each with its honest notes — the rotatable CCD's
   * axial runs sit OUTSIDE the declared ranges, disclosed), the
   * desirability goals, the second-order term formula, the bounds,
   * the seeded SYNTHETIC demo experiment descriptor (KNOWN noise-free
   * surfaces), and the scope statement (the mandate's own distinction
   * table).
   */
  async rsmRegistry(): Promise<RsmRegistryResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/rsm/registry",
    );
    return parseRsmRegistryResponse(raw, "rsm registry");
  },

  /**
   * `POST /api/v1/rsm/design` — generate a design of experiments over
   * the real backend: the CCD (rotatable / faced / custom alpha),
   * Box-Behnken, or the full 2^k factorial, with coded AND actual
   * run tables, the two-block CCD split, seeded within-block
   * randomisation, the run-type census, and the honest assumptions.
   */
  async rsmDesign(
    params: Record<string, unknown>,
  ): Promise<RsmDesignResponse> {
    const raw = (await api.post(
      "/api/v1/rsm/design",
      params,
    )) as Record<string, unknown>;
    return parseRsmDesignResponse(raw, "rsm design");
  },

  /**
   * `POST /api/v1/rsm/fit` — import the observed results and fit the
   * full second-order model for every response column: a registered
   * typed dataset (the request's own declared factor ranges) or the
   * seeded SYNTHETIC demo experiment (labelled).  The answer carries
   * the coefficient tables, the ANOVA partition (pure error /
   * lack-of-fit only where replicated design points exist — nulls
   * elsewhere), the diagnostics block, the hierarchy notes (flagged,
   * never auto-removed), and the aliasing disclosure.  The fitted
   * bundle registers in the session model store and the run records
   * in the §15.2 lineage as family ml.rsm.
   */
  async rsmFit(
    params: Record<string, unknown>,
  ): Promise<RsmFitResponse> {
    const raw = (await api.post(
      "/api/v1/rsm/fit",
      params,
    )) as Record<string, unknown>;
    return parseRsmFitResponse(raw, "rsm fit");
  },

  /**
   * `POST /api/v1/rsm/predict` — point predictions in ACTUAL units
   * with the mean CI and the new-observation PI, and the honest zone
   * vocabulary per point (extrapolation flagged, never silent).
   */
  async rsmPredict(
    params: Record<string, unknown>,
  ): Promise<RsmPredictResponse> {
    const raw = (await api.post(
      "/api/v1/rsm/predict",
      params,
    )) as Record<string, unknown>;
    return parseRsmPredictResponse(raw, "rsm predict");
  },

  /**
   * `POST /api/v1/rsm/contour` — the two-factor grid payload for the
   * Plotly lanes: a chosen response's prediction OR the combined
   * desirability under the request's goals, with every other factor
   * held at explicitly displayed fixed values.
   */
  async rsmContour(
    params: Record<string, unknown>,
  ): Promise<RsmContourResponse> {
    const raw = (await api.post(
      "/api/v1/rsm/contour",
      params,
    )) as Record<string, unknown>;
    return parseRsmContourResponse(raw, "rsm contour");
  },

  /**
   * `POST /api/v1/rsm/optimize` — the Derringer-Suich multi-response
   * optimisation over the fitted surface: the goals (maximize /
   * minimize / target / range) with weights and exponents, optional
   * linear coded-factor constraints, the deterministic multi-start
   * candidate confined to the declared ranges, the per-factor
   * sensitivity sweep, and the curated Pareto alternatives.
   */
  async rsmOptimize(
    params: Record<string, unknown>,
  ): Promise<RsmOptimizeResponse> {
    const raw = (await api.post(
      "/api/v1/rsm/optimize",
      params,
    )) as Record<string, unknown>;
    return parseRsmOptimizeResponse(raw, "rsm optimize");
  },

  /**
   * `GET /api/v1/rsm/models` — the session's RSM models (the
   * comparison surface: every fitted surface side by side).
   */
  async rsmModels(): Promise<{ models: readonly RsmModelView[] }> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/rsm/models",
    );
    const r = expectObject(raw, "rsm models");
    const modelsRaw = Array.isArray(r["models"])
      ? (r["models"] as unknown[])
      : [];
    return {
      models: modelsRaw.map((row, i) =>
        parseRsmModelView(row, `rsm models[${i}]`),
      ),
    };
  },

  /**
   * `POST /api/v1/artifacts/model` — the RSM lane's export: the
   * fitted ResponseSurfaceBundle persisted through the REAL
   * `save_model` as a digest-verified joblib artifact (the reload
   * path is the house `load_model`).
   */
  async rsmExport(modelId: string): Promise<ArtifactView> {
    const raw = (await api.post(
      "/api/v1/artifacts/model",
      { model_id: modelId, filename: `rsm-${modelId}` },
    )) as Record<string, unknown>;
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the model lane answered without an artifact");
    }
    return parseArtifactView(artifact, "rsm model artifact");
  },

  /**
   * `GET /api/v1/workflow/steps` — the §8 workflow engine's typed
   * vocabulary, verbatim: the engine block's own decision text, the
   * closed port-kind list, and every step's ports, dials, REAL
   * backend, and honest notes (part13 Phase 6's GUI lane over the
   * r322 route family).
   */
  async workflowSteps(): Promise<WorkflowStepsResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/workflow/steps",
    );
    return parseWorkflowStepsResponse(raw, "workflow steps");
  },

  /**
   * `GET /api/v1/workflow/templates` — the saved templates (§8's own
   * example chain with its honest leakage reorder + the minimal
   * 6-node core), each carrying the full serialisable spec the
   * builder edits.
   */
  async workflowTemplates(): Promise<WorkflowTemplatesResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/workflow/templates",
    );
    return parseWorkflowTemplatesResponse(raw, "workflow templates");
  },

  /**
   * `POST /api/v1/workflow/validate` — the pre-execution dependency
   * validation over the spec as edited: every finding at once, the
   * topological order the executor would walk, the canonical digest.
   * The body IS the spec dict (the route's own shape — no wrapper).
   */
  async workflowValidate(
    spec: Record<string, unknown>,
  ): Promise<WorkflowValidateResponse> {
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/workflow/validate",
      spec,
    );
    return parseWorkflowValidateResponse(raw, "workflow validate");
  },

  /**
   * `POST /api/v1/workflow/run` — the real round trip over the
   * engine's REAL backend runners: the spec as edited, the route's
   * own seeded SYNTHETIC stand-in as the default dataset, sync mode
   * (bounded — the run record inline, partial failures carried
   * verbatim with downstream SKIPPED and completed artifacts
   * surviving) or job mode over the EXISTING JobManager (progress
   * per node, cooperative cancellation between nodes, the SQLite
   * history). Every metric a statistic over the synthetic rows.
   */
  async workflowRun(
    params: Partial<WorkflowRunRequest> = {},
  ): Promise<WorkflowRunResponse> {
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/workflow/run",
      params,
    );
    return parseWorkflowRunResponse(raw, "workflow run");
  },

  /**
   * `GET /api/v1/workflow/runs` — the bounded in-session execution
   * history with the honest lifetime note (job-mode runs persist
   * through the JobManager's own SQLite history).
   */
  async workflowRuns(): Promise<WorkflowRunsResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/workflow/runs",
    );
    return parseWorkflowRunsResponse(raw, "workflow runs");
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
    residual_diagnostics: parseResidualDiagnostics(
      r["residual_diagnostics"],
      `${what}: residual_diagnostics`,
    ),
    class_metrics: parseClassMetrics(
      r["class_metrics"],
      `${what}: class_metrics`,
    ),
    roc_curves: parseCurveList(
      r["roc_curves"],
      `${what}: roc_curves`,
      (series, what2) => parseRocSeries(series, what2),
    ),
    pr_curves: parseCurveList(
      r["pr_curves"],
      `${what}: pr_curves`,
      (series, what2) => parsePrSeries(series, what2),
    ),
    calibration: parseCalibration(
      r["calibration"],
      `${what}: calibration`,
    ),
  };
}

/** A strictly-optional field: absent stays null (never fabricated). */
function optional<T>(raw: unknown, parse: (raw: unknown, what: string) => T, what: string): T | null {
  return raw === undefined || raw === null ? null : parse(raw, what);
}

function numberArray(raw: unknown, what: string): readonly number[] {
  if (!Array.isArray(raw) || raw.some((v) => typeof v !== "number")) {
    throw new Error(`${what}: expected a number array`);
  }
  return raw as readonly number[];
}

function parseResidualDiagnostics(
  raw: unknown,
  what: string,
): ResidualDiagnostics | null {
  return optional(raw, (r, w) => {
    const o = expectObject(r, w);
    const histogramRaw = o["histogram"];
    const statsRaw = o["stats"];
    if (histogramRaw === undefined || statsRaw === undefined) {
      throw new Error(`${w}: histogram/stats are required`);
    }
    const h = expectObject(histogramRaw, `${w}: histogram`);
    const s = expectObject(statsRaw, `${w}: stats`);
    const qq = optional(o["qq"], (q, wq) => {
      const oq = expectObject(q, wq);
      return {
        theoretical: numberArray(oq["theoretical"], `${wq}: theoretical`),
        empirical: numberArray(oq["empirical"], `${wq}: empirical`),
      };
    }, `${w}: qq`);
    return {
      histogram: {
        bin_edges: numberArray(h["bin_edges"], `${w}: bin_edges`),
        counts: numberArray(h["counts"], `${w}: counts`),
      },
      stats: {
        mean: asNumber(s["mean"]) ?? 0,
        median: asNumber(s["median"]) ?? 0,
        std: asNumber(s["std"]) ?? 0,
        min: asNumber(s["min"]) ?? 0,
        max: asNumber(s["max"]) ?? 0,
        n: asNumber(s["n"]) ?? 0,
      },
      qq,
    };
  }, what);
}

function parseClassMetrics(raw: unknown, what: string): ClassMetrics | null {
  return optional(raw, (r, w) => {
    const o = expectObject(r, w);
    const rowsRaw = o["rows"];
    if (!Array.isArray(rowsRaw)) {
      throw new Error(`${w}: rows must be an array`);
    }
    const rows: ClassMetricRow[] = [];
    for (const rowRaw of rowsRaw) {
      const row = expectObject(rowRaw, `${w}: a metrics row`);
      rows.push({
        label: String(row["label"]),
        precision: asNumber(row["precision"]) ?? 0,
        recall: asNumber(row["recall"]) ?? 0,
        f1: asNumber(row["f1"]) ?? 0,
        support: asNumber(row["support"]) ?? 0,
      });
    }
    const macroRaw = expectObject(o["macro"], `${w}: macro`);
    return {
      rows,
      macro: {
        precision: asNumber(macroRaw["precision"]) ?? 0,
        recall: asNumber(macroRaw["recall"]) ?? 0,
        f1: asNumber(macroRaw["f1"]) ?? 0,
      },
      accuracy: asNumber(o["accuracy"]) ?? 0,
      n: asNumber(o["n"]) ?? 0,
    };
  }, what);
}

function parseCurveList<T>(
  raw: unknown,
  what: string,
  parseSeries: (raw: unknown, what: string) => T,
): readonly T[] | null {
  if (raw === undefined || raw === null) {
    return null;
  }
  if (!Array.isArray(raw)) {
    throw new Error(`${what}: expected an array of curves`);
  }
  return raw.map((series, i) =>
    parseSeries(series, `${what}[${i}]`),
  );
}

function parseRocSeries(raw: unknown, what: string): RocCurveSeries {
  const o = expectObject(raw, what);
  return {
    label: String(o["label"]),
    fpr: numberArray(o["fpr"], `${what}: fpr`),
    tpr: numberArray(o["tpr"], `${what}: tpr`),
    thresholds: numberArray(o["thresholds"], `${what}: thresholds`),
    auc: asNumber(o["auc"]) ?? 0,
    n_pos: asNumber(o["n_pos"]) ?? 0,
    n_neg: asNumber(o["n_neg"]) ?? 0,
  };
}

function parsePrSeries(raw: unknown, what: string): PrCurveSeries {
  const o = expectObject(raw, what);
  return {
    label: String(o["label"]),
    recall: numberArray(o["recall"], `${what}: recall`),
    precision: numberArray(o["precision"], `${what}: precision`),
    thresholds: numberArray(o["thresholds"], `${what}: thresholds`),
    average_precision: asNumber(o["average_precision"]) ?? 0,
    prevalence: asNumber(o["prevalence"]) ?? 0,
    n_pos: asNumber(o["n_pos"]) ?? 0,
  };
}

function parseCalibration(raw: unknown, what: string): CalibrationDiagram | null {
  return optional(raw, (r, w) => {
    const o = expectObject(r, w);
    const binsRaw = o["bins"];
    if (!Array.isArray(binsRaw)) {
      throw new Error(`${w}: bins must be an array`);
    }
    const bins: CalibrationBin[] = [];
    for (const binRaw of binsRaw) {
      const bin = expectObject(binRaw, `${w}: a bin`);
      bins.push({
        mean_predicted: asNumber(bin["mean_predicted"]) ?? 0,
        observed: asNumber(bin["observed"]) ?? 0,
        count: asNumber(bin["count"]) ?? 0,
      });
    }
    return {
      bins,
      n: asNumber(o["n"]) ?? 0,
      positive_class: String(o["positive_class"]),
    };
  }, what);
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

/**
 * Narrow the ML-3 carve block (the train answer's split half, or the
 * §15.2 record's refused echo): every field surfaced as it arrives, the
 * strategy validated against the engine's own literal set — an unknown
 * strategy throws rather than rendering as if it were honest.
 */
export function parseTrainSplitBlock(
  raw: unknown,
  what: string,
): TrainSplitBlock | null {
  if (raw === undefined || raw === null) {
    return null;
  }
  const r = expectObject(raw, what);
  const strategy = r["strategy"];
  if (
    typeof strategy !== "string" ||
    !(TRAIN_SPLIT_STRATEGY_LITERALS as readonly string[]).includes(strategy)
  ) {
    throw new Error(
      `${what}: split strategy is not one of the engine's literals`,
    );
  }
  const sidesRaw = r["sides"];
  const sides = expectObject(sidesRaw, `${what} sides`);
  const train = expectObject(sides["train"], `${what} train side`);
  const heldout = expectObject(sides["heldout"], `${what} heldout side`);
  if (
    typeof train["rows"] !== "number" ||
    typeof heldout["rows"] !== "number"
  ) {
    throw new Error(`${what}: side rows are not numbers`);
  }
  const censusRaw = r["group_census"];
  let census: TrainSplitBlock["group_census"] = null;
  if (typeof censusRaw === "object" && censusRaw !== null) {
    const censusObj = censusRaw as Record<string, unknown>;
    const trainSide = expectObject(censusObj["train"], `${what} census train`);
    const heldoutSide = expectObject(
      censusObj["heldout"],
      `${what} census heldout`,
    );
    const trainGroups = expectObject(
      trainSide["top_groups"],
      `${what} census train top_groups`,
    );
    const heldoutGroups = expectObject(
      heldoutSide["top_groups"],
      `${what} census heldout top_groups`,
    );
    census = {
      train: {
        n_groups:
          typeof trainSide["n_groups"] === "number"
            ? trainSide["n_groups"]
            : 0,
        top_groups: (Array.isArray(trainGroups) ? trainGroups : []).map(
          (entry) =>
            Array.isArray(entry) &&
            typeof entry[0] === "string" &&
            typeof entry[1] === "number"
              ? ([entry[0], entry[1]] as const)
              : (["", 0] as const),
        ),
      },
      heldout: {
        n_groups:
          typeof heldoutSide["n_groups"] === "number"
            ? heldoutSide["n_groups"]
            : 0,
        top_groups: (Array.isArray(heldoutGroups) ? heldoutGroups : []).map(
          (entry) =>
            Array.isArray(entry) &&
            typeof entry[0] === "string" &&
            typeof entry[1] === "number"
              ? ([entry[0], entry[1]] as const)
              : (["", 0] as const),
        ),
      },
      shared_groups: Array.isArray(censusObj["shared_groups"])
        ? (censusObj["shared_groups"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
    };
  }
  const nRows = r["n_rows"];
  if (typeof nRows !== "number") {
    throw new Error(`${what}: n_rows is not a number`);
  }
  return {
    strategy: strategy as SplitStrategy,
    smiles_column: typeof r["smiles_column"] === "string" ? r["smiles_column"] : null,
    group_column: typeof r["group_column"] === "string" ? r["group_column"] : null,
    test_size: typeof r["test_size"] === "number" ? r["test_size"] : 0,
    random_state: typeof r["random_state"] === "number" ? r["random_state"] : 0,
    n_rows: nRows,
    sides: {
      train: { rows: train["rows"] as number },
      heldout: { rows: heldout["rows"] as number },
    },
    group_census: census,
    error: typeof r["error"] === "string" ? r["error"] : null,
  };
}

export function parseTrainResult(raw: unknown, what: string): {
  model: TrainedModelView;
  split: TrainSplitBlock | null;
} {
  const r = expectObject(raw, what);
  return {
    model: parseTrainedModelView(r["model"], `${what} model`),
    split: parseTrainSplitBlock(r["split"], `${what} split`),
  };
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

// -- the DTA lane's §8.1 runtime parsers --------------------------------

function parseDtaProvenance(raw: unknown, what: string): DtaProvenance {
  const r = expectObject(raw, `${what} provenance`);
  const backend = r["backend"];
  const source = r["source"];
  if (typeof backend !== "string" || typeof source !== "string") {
    throw new Error(`${what} provenance: backend/source are not strings`);
  }
  return {
    backend,
    source,
    n_alphabets: asNumber(r["n_alphabets"]) ?? 0,
    shipped_checkpoints: asNumber(r["shipped_checkpoints"]) ?? 0,
    shipped_checkpoints_note:
      asString(r["shipped_checkpoints_note"]) ?? "",
    disclaimer: asString(r["disclaimer"]) ?? "",
  };
}

/** Narrow one alphabet block — the vendored symbol→code table. */
function parseDtaAlphabetBlock(
  raw: unknown,
  what: string,
): DtaAlphabetBlock {
  const r = expectObject(raw, what);
  const symbolsRaw = expectObject(r["symbols"], `${what} symbols`);
  const symbols: Record<string, number> = {};
  for (const [symbol, code] of Object.entries(symbolsRaw)) {
    if (typeof code === "number") {
      symbols[symbol] = code;
    }
  }
  return {
    n_symbols: asNumber(r["n_symbols"]) ?? Object.keys(symbols).length,
    padding_code: asNumber(r["padding_code"]) ?? 0,
    symbols,
    encoding: asString(r["encoding"]) ?? "",
    source: asString(r["source"]) ?? "",
  };
}

/** Narrow the registry answer. */
function parseDtaRegistryResponse(
  raw: unknown,
  what: string,
): DtaRegistryResponse {
  const r = expectObject(raw, what);
  const alphabetsRaw = expectObject(r["alphabets"], `${what} alphabets`);
  const proteinFeaturesRaw = expectObject(
    r["protein_features"],
    `${what} protein_features`,
  );
  const ligandFeaturesRaw = expectObject(
    r["ligand_features"],
    `${what} ligand_features`,
  );
  const pairedWidthRaw = expectObject(r["paired_width"], `${what} paired_width`);
  const demoRaw = expectObject(r["demo_frame"], `${what} demo_frame`);
  return {
    alphabets: {
      protein: parseDtaAlphabetBlock(
        alphabetsRaw["protein"],
        `${what} alphabets.protein`,
      ),
      smiles_isomeric: parseDtaAlphabetBlock(
        alphabetsRaw["smiles_isomeric"],
        `${what} alphabets.smiles_isomeric`,
      ),
      smiles_canonical: parseDtaAlphabetBlock(
        alphabetsRaw["smiles_canonical"],
        `${what} alphabets.smiles_canonical`,
      ),
    },
    amino_acids: Array.isArray(r["amino_acids"])
      ? (r["amino_acids"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
    protein_features: {
      n_base: asNumber(proteinFeaturesRaw["n_base"]) ?? 0,
      base_columns: Array.isArray(proteinFeaturesRaw["base_columns"])
        ? (proteinFeaturesRaw["base_columns"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      n_dipeptide_extras:
        asNumber(proteinFeaturesRaw["n_dipeptide_extras"]) ?? 0,
      n_with_dipeptides:
        asNumber(proteinFeaturesRaw["n_with_dipeptides"]) ?? 0,
      note: asString(proteinFeaturesRaw["note"]) ?? "",
    },
    ligand_features: {
      family: asString(ligandFeaturesRaw["family"]) ?? "",
      n_bits: asNumber(ligandFeaturesRaw["n_bits"]) ?? 0,
      source: asString(ligandFeaturesRaw["source"]) ?? "",
    },
    paired_width: {
      base: asNumber(pairedWidthRaw["base"]) ?? 0,
      with_dipeptides: asNumber(pairedWidthRaw["with_dipeptides"]) ?? 0,
    },
    metrics: Array.isArray(r["metrics"])
      ? (r["metrics"] as unknown[]).map((m, i) => {
          const mm = expectObject(m, `${what} metrics[${i}]`);
          return {
            name: asString(mm["name"]) ?? "",
            description: asString(mm["description"]) ?? "",
          };
        })
      : [],
    demo_frame: {
      rows: asNumber(demoRaw["rows"]) ?? 0,
      note: asString(demoRaw["note"]) ?? "",
    },
    provenance: parseDtaProvenance(r["provenance"], what),
    features_url: asString(r["features_url"]) ?? "",
    note: asString(r["note"]) ?? "",
  };
}

/** Narrow one row's protein composition — the backend's own columns. */
function parseDtaProteinRow(raw: unknown, what: string): DtaProteinRow {
  const r = expectObject(raw, `${what} protein_row`);
  const aacRaw = expectObject(r["aac"], `${what} protein_row.aac`);
  const aac: Record<string, number> = {};
  for (const [symbol, value] of Object.entries(aacRaw)) {
    if (typeof value === "number") {
      aac[symbol] = value;
    }
  }
  return {
    prot_length: asNumber(r["prot_length"]) ?? 0,
    prot_standard_fraction: asNumber(r["prot_standard_fraction"]) ?? 0,
    prot_nonstandard_fraction: asNumber(r["prot_nonstandard_fraction"]) ?? 0,
    aac,
  };
}

/** Narrow the features answer. */
function parseDtaFeaturesResponse(
  raw: unknown,
  what: string,
): DtaFeaturesResponse {
  const r = expectObject(raw, what);
  const rows = r["rows"];
  const droppedRows = r["dropped"];
  const nPairs = r["n_pairs"];
  const nKept = r["n_kept"];
  const nDropped = r["n_dropped"];
  if (
    !Array.isArray(rows) ||
    !Array.isArray(droppedRows) ||
    typeof nPairs !== "number" ||
    typeof nKept !== "number" ||
    typeof nDropped !== "number"
  ) {
    throw new Error(
      `${what}: n_pairs/n_kept/n_dropped/rows/dropped are malformed`,
    );
  }
  const affinityRaw = r["affinity"];
  let affinity: DtaFeaturesResponse["affinity"] = null;
  if (affinityRaw !== null && affinityRaw !== undefined) {
    const a = expectObject(affinityRaw, `${what} affinity`);
    affinity = {
      present: a["present"] === true,
      n: asNumber(a["n"]) ?? 0,
      min: asNumber(a["min"]) ?? 0,
      max: asNumber(a["max"]) ?? 0,
      mean: asNumber(a["mean"]) ?? 0,
    };
  }
  return {
    n_pairs: nPairs,
    n_kept: nKept,
    n_dropped: nDropped,
    kept_indices: Array.isArray(r["kept_indices"])
      ? (r["kept_indices"] as unknown[]).filter(
          (v): v is number => typeof v === "number",
        )
      : [],
    dropped: droppedRows.map((row, i) => {
      const d = expectObject(row, `${what} dropped[${i}]`);
      return {
        index: asNumber(d["index"]) ?? 0,
        smiles: asString(d["smiles"]) ?? "",
        reason: asString(d["reason"]) ?? "",
      };
    }),
    n_ligand_features: asNumber(r["n_ligand_features"]) ?? 0,
    n_protein_features: asNumber(r["n_protein_features"]) ?? 0,
    n_features: asNumber(r["n_features"]) ?? 0,
    dipeptides: r["dipeptides"] === true,
    rows: rows.map((row, i) => {
      const c = expectObject(row, `${what} rows[${i}]`);
      return {
        index: asNumber(c["index"]) ?? i,
        smiles: asString(c["smiles"]) ?? "",
        protein: asString(c["protein"]) ?? "",
        valid: c["valid"] === true,
        protein_row: parseDtaProteinRow(c["protein_row"], `${what} rows[${i}]`),
        affinity: asNumber(c["affinity"]),
      };
    }),
    affinity,
    provenance: parseDtaProvenance(r["provenance"], what),
    note: asString(r["note"]) ?? "",
  };
}

// -- the repurposing lane's §8.1 runtime parsers ------------------------

/** Narrow a record whose values are all numbers (the request echo /
 * the screen defaults). */
function parseNumberRecord(
  raw: unknown,
  what: string,
): Record<string, number> {
  const r = expectObject(raw, what);
  const out: Record<string, number> = {};
  for (const [key, value] of Object.entries(r)) {
    if (typeof value === "number") {
      out[key] = value;
    }
  }
  return out;
}

/** Narrow a record whose values are all strings (the screen bounds). */
function parseStringRecord(
  raw: unknown,
  what: string,
): Record<string, string> {
  const r = expectObject(raw, what);
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(r)) {
    if (typeof value === "string") {
      out[key] = value;
    }
  }
  return out;
}

function parseRepurposingProvenance(
  raw: unknown,
  what: string,
): RepurposingProvenance {
  const r = expectObject(raw, `${what} provenance`);
  const backend = r["backend"];
  const source = r["source"];
  if (typeof backend !== "string" || typeof source !== "string") {
    throw new Error(`${what} provenance: backend/source are not strings`);
  }
  return {
    backend,
    source,
    n_repositories: asNumber(r["n_repositories"]) ?? 0,
    shipped_checkpoints: asNumber(r["shipped_checkpoints"]) ?? 0,
    shipped_checkpoints_note:
      asString(r["shipped_checkpoints_note"]) ?? "",
    screen: asString(r["screen"]) ?? "",
    disclaimer: asString(r["disclaimer"]) ?? "",
  };
}

/** Narrow the registry answer. */
function parseRepurposingRegistryResponse(
  raw: unknown,
  what: string,
): RepurposingRegistryResponse {
  const r = expectObject(raw, what);
  const strategiesRaw = expectObject(
    r["split_strategies"],
    `${what} split_strategies`,
  );
  const modelRaw = expectObject(r["model"], `${what} model`);
  const hyperRaw = expectObject(
    r["hyperparameters"],
    `${what} hyperparameters`,
  );
  const screenRaw = expectObject(r["screen"], `${what} screen`);
  const repurposingRaw = expectObject(
    r["repurposing"],
    `${what} repurposing`,
  );
  const limitsRaw = expectObject(
    repurposingRaw["response_limits"],
    `${what} repurposing.response_limits`,
  );
  const metricsRaw = expectObject(r["metrics"], `${what} metrics`);
  return {
    split_strategies: {
      within_table: Array.isArray(strategiesRaw["within_table"])
        ? (strategiesRaw["within_table"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      cross_domain: asString(strategiesRaw["cross_domain"]) ?? "",
      source: asString(strategiesRaw["source"]) ?? "",
    },
    model: {
      family: asString(modelRaw["family"]) ?? "",
      formulation: asString(modelRaw["formulation"]) ?? "",
      direction: asString(modelRaw["direction"]) ?? "",
      telemetry_fields: Array.isArray(modelRaw["telemetry_fields"])
        ? (modelRaw["telemetry_fields"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      history_note: asString(modelRaw["history_note"]) ?? "",
    },
    hyperparameters: {
      n_factors: parseNumberRecord(
        hyperRaw["n_factors"],
        `${what} hyperparameters.n_factors`,
      ),
      lr: parseNumberRecord(
        hyperRaw["lr"],
        `${what} hyperparameters.lr`,
      ),
      max_epochs: parseNumberRecord(
        hyperRaw["max_epochs"],
        `${what} hyperparameters.max_epochs`,
      ),
      seed: parseNumberRecord(
        hyperRaw["seed"],
        `${what} hyperparameters.seed`,
      ),
      note: asString(hyperRaw["note"]) ?? "",
    },
    screen: {
      defaults: parseNumberRecord(
        screenRaw["defaults"],
        `${what} screen.defaults`,
      ),
      bounds: parseStringRecord(
        screenRaw["bounds"],
        `${what} screen.bounds`,
      ),
      label: asString(screenRaw["label"]) ?? "",
    },
    repurposing: {
      schema: Array.isArray(repurposingRaw["schema"])
        ? (repurposingRaw["schema"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      ordering: asString(repurposingRaw["ordering"]) ?? "",
      source: asString(repurposingRaw["source"]) ?? "",
      response_limits: {
        profiles: asString(limitsRaw["profiles"]) ?? "",
        top_k_per_profile: asString(limitsRaw["top_k_per_profile"]) ?? "",
      },
    },
    metrics: {
      keys: Array.isArray(metricsRaw["keys"])
        ? (metricsRaw["keys"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      threshold: asNumber(metricsRaw["threshold"]) ?? 0,
      note: asString(metricsRaw["note"]) ?? "",
    },
    provenance: parseRepurposingProvenance(r["provenance"], what),
    screen_url: asString(r["screen_url"]) ?? "",
    note: asString(r["note"]) ?? "",
  };
}

/** Narrow the screen answer's metric contract (null, never a lie). */
function parseMetricRecord(
  raw: unknown,
  what: string,
): Record<string, number | null> {
  const r = expectObject(raw, what);
  const out: Record<string, number | null> = {};
  for (const [key, value] of Object.entries(r)) {
    if (typeof value === "number") {
      out[key] = value;
    } else {
      out[key] = null;
    }
  }
  return out;
}

/** Narrow one ranked candidate row. */
function parseRepurposingCandidateRow(
  raw: unknown,
  what: string,
): RepurposingCandidateRow {
  const r = expectObject(raw, what);
  const drug = r["drug"];
  const rank = r["rank"];
  const predicted = r["predicted_ic50"];
  if (
    typeof drug !== "string" ||
    typeof rank !== "number" ||
    typeof predicted !== "number"
  ) {
    throw new Error(`${what}: drug/predicted_ic50/rank are malformed`);
  }
  return { drug, predicted_ic50: predicted, rank };
}

/** Narrow the screen answer. */
function parseRepurposingScreenResponse(
  raw: unknown,
  what: string,
): RepurposingScreenResponse {
  const r = expectObject(raw, what);
  const screenRaw = expectObject(r["screen"], `${what} screen`);
  const trainingRaw = expectObject(r["training"], `${what} training`);
  const candidatesRaw = expectObject(r["candidates"], `${what} candidates`);
  const perProfileRaw = expectObject(
    candidatesRaw["per_profile"],
    `${what} candidates.per_profile`,
  );
  const perProfile: Record<string, RepurposingCandidateRow[]> = {};
  for (const [cellLine, rows] of Object.entries(perProfileRaw)) {
    if (!Array.isArray(rows)) {
      continue;
    }
    perProfile[cellLine] = rows.map((row, i) =>
      parseRepurposingCandidateRow(row, `${what} per_profile[${cellLine}][${i}]`),
    );
  }
  return {
    request: parseNumberRecord(r["request"], `${what} request`),
    screen: {
      cells: Array.isArray(screenRaw["cells"])
        ? (screenRaw["cells"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      drugs: Array.isArray(screenRaw["drugs"])
        ? (screenRaw["drugs"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      features: Array.isArray(screenRaw["features"])
        ? (screenRaw["features"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      n_cells: asNumber(screenRaw["n_cells"]) ?? 0,
      n_drugs: asNumber(screenRaw["n_drugs"]) ?? 0,
      n_features: asNumber(screenRaw["n_features"]) ?? 0,
      n_observed: asNumber(screenRaw["n_observed"]) ?? 0,
      n_entries: asNumber(screenRaw["n_entries"]) ?? 0,
      observed_fraction: asNumber(screenRaw["observed_fraction"]) ?? 0,
      label: asString(screenRaw["label"]) ?? "",
    },
    training: {
      epochs_run: asNumber(trainingRaw["epochs_run"]) ?? 0,
      max_epochs: asNumber(trainingRaw["max_epochs"]) ?? 0,
      stopped_early: trainingRaw["stopped_early"] === true,
      observed_rmse: asNumber(trainingRaw["observed_rmse"]),
      first_epoch_rmse: asNumber(trainingRaw["first_epoch_rmse"]),
      first_epoch: asNumber(trainingRaw["first_epoch"]) ?? 0,
      first_epoch_ndcg: asNumber(trainingRaw["first_epoch_ndcg"]),
      n_factors: asNumber(trainingRaw["n_factors"]) ?? 0,
      note: asString(trainingRaw["note"]) ?? "",
    },
    metrics: parseMetricRecord(r["metrics"], `${what} metrics`),
    candidates: {
      n_profiles: asNumber(candidatesRaw["n_profiles"]) ?? 0,
      n_ranked_rows_total:
        asNumber(candidatesRaw["n_ranked_rows_total"]) ?? 0,
      top_k: asNumber(candidatesRaw["top_k"]) ?? 0,
      per_profile: perProfile,
      ordering: asString(candidatesRaw["ordering"]) ?? "",
    },
    provenance: parseRepurposingProvenance(r["provenance"], what),
    note: asString(r["note"]) ?? "",
  };
}

// -- the drug-response lane parsers (§8.1) ------------------------------

function parseDrugResponseProvenance(
  raw: unknown,
  what: string,
): DrugResponseProvenance {
  const r = expectObject(raw, `${what} provenance`);
  const backend = r["backend"];
  const source = r["source"];
  if (typeof backend !== "string" || typeof source !== "string") {
    throw new Error(`${what} provenance: backend/source are not strings`);
  }
  return {
    backend,
    source,
    n_repositories: asNumber(r["n_repositories"]) ?? 0,
    shipped_checkpoints: asNumber(r["shipped_checkpoints"]) ?? 0,
    shipped_checkpoints_note:
      asString(r["shipped_checkpoints_note"]) ?? "",
    screen: asString(r["screen"]) ?? "",
    disclaimer: asString(r["disclaimer"]) ?? "",
  };
}

/** Narrow the drug-response registry answer. */
function parseDrugResponseRegistryResponse(
  raw: unknown,
  what: string,
): DrugResponseRegistryResponse {
  const r = expectObject(raw, what);
  const modelRaw = expectObject(r["model"], `${what} model`);
  const hyperRaw = expectObject(
    r["hyperparameters"],
    `${what} hyperparameters`,
  );
  const screenRaw = expectObject(r["screen"], `${what} screen`);
  const rankingRaw = expectObject(
    r["sensitivity_ranking"],
    `${what} sensitivity_ranking`,
  );
  const limitsRaw = expectObject(
    rankingRaw["response_limits"],
    `${what} sensitivity_ranking.response_limits`,
  );
  const biomarkersRaw = expectObject(r["biomarkers"], `${what} biomarkers`);
  const biomarkerLimitsRaw = expectObject(
    biomarkersRaw["response_limits"],
    `${what} biomarkers.response_limits`,
  );
  const cvRaw = expectObject(
    r["cross_validation"],
    `${what} cross_validation`,
  );
  const metricsRaw = expectObject(r["metrics"], `${what} metrics`);
  return {
    model: {
      family: asString(modelRaw["family"]) ?? "",
      formulation: asString(modelRaw["formulation"]) ?? "",
      direction: asString(modelRaw["direction"]) ?? "",
      telemetry_fields: Array.isArray(modelRaw["telemetry_fields"])
        ? (modelRaw["telemetry_fields"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      history_note: asString(modelRaw["history_note"]) ?? "",
    },
    hyperparameters: {
      n_factors: parseNumberRecord(
        hyperRaw["n_factors"],
        `${what} hyperparameters.n_factors`,
      ),
      lr: parseNumberRecord(hyperRaw["lr"], `${what} hyperparameters.lr`),
      epochs: parseNumberRecord(
        hyperRaw["epochs"],
        `${what} hyperparameters.epochs`,
      ),
      l2: parseNumberRecord(hyperRaw["l2"], `${what} hyperparameters.l2`),
      seed: parseNumberRecord(
        hyperRaw["seed"],
        `${what} hyperparameters.seed`,
      ),
      note: asString(hyperRaw["note"]) ?? "",
    },
    screen: {
      defaults: parseNumberRecord(
        screenRaw["defaults"],
        `${what} screen.defaults`,
      ),
      bounds: parseStringRecord(screenRaw["bounds"], `${what} screen.bounds`),
      label: asString(screenRaw["label"]) ?? "",
      scale: asString(screenRaw["scale"]) ?? "",
    },
    sensitivity_ranking: {
      schema: Array.isArray(rankingRaw["schema"])
        ? (rankingRaw["schema"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      ordering: asString(rankingRaw["ordering"]) ?? "",
      source: asString(rankingRaw["source"]) ?? "",
      response_limits: {
        profiles: asString(limitsRaw["profiles"]) ?? "",
        top_k_per_profile: asString(limitsRaw["top_k_per_profile"]) ?? "",
      },
    },
    biomarkers: {
      schema: Array.isArray(biomarkersRaw["schema"])
        ? (biomarkersRaw["schema"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      ordering: asString(biomarkersRaw["ordering"]) ?? "",
      source: asString(biomarkersRaw["source"]) ?? "",
      response_limits: {
        top_k_per_drug: asString(biomarkerLimitsRaw["top_k_per_drug"]) ?? "",
      },
    },
    cross_validation: {
      n_folds: asString(cvRaw["n_folds"]) ?? "",
      rule: asString(cvRaw["rule"]) ?? "",
      summary: asString(cvRaw["summary"]) ?? "",
    },
    metrics: {
      keys: Array.isArray(metricsRaw["keys"])
        ? (metricsRaw["keys"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      note: asString(metricsRaw["note"]) ?? "",
    },
    provenance: parseDrugResponseProvenance(r["provenance"], what),
    analyze_url: asString(r["analyze_url"]) ?? "",
    note: asString(r["note"]) ?? "",
  };
}

/** Narrow one ranked sensitivity row. */
function parseDrugResponseRankRow(
  raw: unknown,
  what: string,
): DrugResponseRankRow {
  const r = expectObject(raw, what);
  const drug = r["drug"];
  const rank = r["rank"];
  const predicted = r["predicted_score"];
  if (
    typeof drug !== "string" ||
    typeof rank !== "number" ||
    typeof predicted !== "number"
  ) {
    throw new Error(`${what}: drug/predicted_score/rank are malformed`);
  }
  return { drug, predicted_score: predicted, rank };
}

/** Narrow one biomarker weight row. */
function parseDrugResponseBiomarkerRow(
  raw: unknown,
  what: string,
): DrugResponseBiomarkerRow {
  const r = expectObject(raw, what);
  const feature = r["feature"];
  const weight = r["weight"];
  if (typeof feature !== "string" || typeof weight !== "number") {
    throw new Error(`${what}: feature/weight are malformed`);
  }
  return { feature, weight };
}

/** Narrow one cross-cell-line validation fold. */
function parseDrugResponseCvFold(
  raw: unknown,
  what: string,
): DrugResponseCvFold {
  const r = expectObject(raw, what);
  return {
    fold: asNumber(r["fold"]) ?? 0,
    n_test_cells: asNumber(r["n_test_cells"]) ?? 0,
    n_observed: asNumber(r["n_observed"]) ?? 0,
    metrics: parseMetricRecord(r["metrics"], `${what} metrics`),
  };
}

/** Narrow the analyze answer. */
function parseDrugResponseAnalyzeResponse(
  raw: unknown,
  what: string,
): DrugResponseAnalyzeResponse {
  const r = expectObject(raw, what);
  const screenRaw = expectObject(r["screen"], `${what} screen`);
  const trainingRaw = expectObject(r["training"], `${what} training`);
  const rankingRaw = expectObject(r["ranking"], `${what} ranking`);
  const perCellRaw = expectObject(
    rankingRaw["per_cell"],
    `${what} ranking.per_cell`,
  );
  const perCell: Record<string, DrugResponseRankRow[]> = {};
  for (const [cellLine, rows] of Object.entries(perCellRaw)) {
    if (!Array.isArray(rows)) {
      continue;
    }
    perCell[cellLine] = rows.map((row, i) =>
      parseDrugResponseRankRow(
        row,
        `${what} per_cell[${cellLine}][${i}]`,
      ),
    );
  }
  const biomarkersRaw = expectObject(r["biomarkers"], `${what} biomarkers`);
  const perDrugRaw = expectObject(
    biomarkersRaw["per_drug"],
    `${what} biomarkers.per_drug`,
  );
  const perDrug: Record<string, DrugResponseBiomarkerRow[]> = {};
  for (const [drug, rows] of Object.entries(perDrugRaw)) {
    if (!Array.isArray(rows)) {
      continue;
    }
    perDrug[drug] = rows.map((row, i) =>
      parseDrugResponseBiomarkerRow(
        row,
        `${what} per_drug[${drug}][${i}]`,
      ),
    );
  }
  const cvRaw = expectObject(r["cross_validation"], `${what} cross_validation`);
  const foldsRaw = Array.isArray(cvRaw["folds"])
    ? (cvRaw["folds"] as unknown[])
    : [];
  return {
    request: parseNumberRecord(r["request"], `${what} request`),
    screen: {
      cells: Array.isArray(screenRaw["cells"])
        ? (screenRaw["cells"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      drugs: Array.isArray(screenRaw["drugs"])
        ? (screenRaw["drugs"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      features: Array.isArray(screenRaw["features"])
        ? (screenRaw["features"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      n_cells: asNumber(screenRaw["n_cells"]) ?? 0,
      n_drugs: asNumber(screenRaw["n_drugs"]) ?? 0,
      n_features: asNumber(screenRaw["n_features"]) ?? 0,
      n_observed: asNumber(screenRaw["n_observed"]) ?? 0,
      n_entries: asNumber(screenRaw["n_entries"]) ?? 0,
      observed_fraction: asNumber(screenRaw["observed_fraction"]) ?? 0,
      label: asString(screenRaw["label"]) ?? "",
    },
    training: {
      epochs_run: asNumber(trainingRaw["epochs_run"]) ?? 0,
      max_epochs: asNumber(trainingRaw["max_epochs"]) ?? 0,
      final_loss: asNumber(trainingRaw["final_loss"]),
      first_loss: asNumber(trainingRaw["first_loss"]),
      n_observed: asNumber(trainingRaw["n_observed"]) ?? 0,
      pearson_r: asNumber(trainingRaw["pearson_r"]),
      n_factors: asNumber(trainingRaw["n_factors"]) ?? 0,
      note: asString(trainingRaw["note"]) ?? "",
    },
    metrics: parseMetricRecord(r["metrics"], `${what} metrics`),
    ranking: {
      n_profiles: asNumber(rankingRaw["n_profiles"]) ?? 0,
      n_ranked_rows_total: asNumber(rankingRaw["n_ranked_rows_total"]) ?? 0,
      top_k: asNumber(rankingRaw["top_k"]) ?? 0,
      per_cell: perCell,
      ordering: asString(rankingRaw["ordering"]) ?? "",
    },
    biomarkers: {
      n_drugs: asNumber(biomarkersRaw["n_drugs"]) ?? 0,
      top_k: asNumber(biomarkersRaw["top_k"]) ?? 0,
      per_drug: perDrug,
      ordering: asString(biomarkersRaw["ordering"]) ?? "",
    },
    cross_validation: {
      n_folds: asNumber(cvRaw["n_folds"]) ?? 0,
      folds: foldsRaw.map((fold, i) =>
        parseDrugResponseCvFold(fold, `${what} folds[${i}]`),
      ),
      summary: parseMetricRecord(cvRaw["summary"], `${what} summary`),
      note: asString(cvRaw["note"]) ?? "",
    },
    provenance: parseDrugResponseProvenance(r["provenance"], what),
    note: asString(r["note"]) ?? "",
  };
}

/** Narrow the pharmacogenomics provenance block. */
function parsePharmacogenomicsProvenance(
  raw: unknown,
  what: string,
): PharmacogenomicsProvenance {
  const r = expectObject(raw, `${what} provenance`);
  return {
    backend: asString(r["backend"]) ?? "",
    source: asString(r["source"]) ?? "",
    n_repositories: asNumber(r["n_repositories"]) ?? 0,
    shipped_tables: asNumber(r["shipped_tables"]) ?? 0,
    shipped_tables_note: asString(r["shipped_tables_note"]) ?? "",
    shipped_checkpoints: asNumber(r["shipped_checkpoints"]) ?? 0,
    shipped_checkpoints_note: asString(r["shipped_checkpoints_note"]) ?? "",
    screen: asString(r["screen"]) ?? "",
    disclaimer: asString(r["disclaimer"]) ?? "",
  };
}

function parsePharmacogenomicsRegistryResponse(
  raw: unknown,
  what: string,
): PharmacogenomicsRegistryResponse {
  const r = expectObject(raw, what);
  const associationsRaw = expectObject(r["associations"], `${what} associations`);
  const gatesRaw = expectObject(
    associationsRaw["feasibility_gates"],
    `${what} associations.feasibility_gates`,
  );
  const effectRaw = expectObject(r["effect_sizes"], `${what} effect_sizes`);
  const multipleRaw = expectObject(
    r["multiple_testing"],
    `${what} multiple_testing`,
  );
  const volcanoRaw = expectObject(r["volcano"], `${what} volcano`);
  const evaluationRaw = expectObject(r["evaluation"], `${what} evaluation`);
  const vocabularyRaw = expectObject(r["vocabulary"], `${what} vocabulary`);
  const cnaRaw = expectObject(
    vocabularyRaw["cna_values"],
    `${what} vocabulary.cna_values`,
  );
  const screenRaw = expectObject(r["screen"], `${what} screen`);
  const geometryRaw = expectObject(
    screenRaw["geometry"],
    `${what} screen.geometry`,
  );
  return {
    associations: {
      family: asString(associationsRaw["family"]) ?? "",
      design_matrix: asString(associationsRaw["design_matrix"]) ?? "",
      feasibility_gates: parseStringRecord(
        gatesRaw,
        `${what} associations.feasibility_gates`,
      ),
      row_schema: Array.isArray(associationsRaw["row_schema"])
        ? (associationsRaw["row_schema"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      ordering: asString(associationsRaw["ordering"]) ?? "",
    },
    effect_sizes: {
      keys: Array.isArray(effectRaw["keys"])
        ? (effectRaw["keys"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      note: asString(effectRaw["note"]) ?? "",
    },
    multiple_testing: {
      methods: Array.isArray(multipleRaw["methods"])
        ? (multipleRaw["methods"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      bh_fdr: asString(multipleRaw["bh_fdr"]) ?? "",
      qvalue: asString(multipleRaw["qvalue"]) ?? "",
      percent_convention: asString(multipleRaw["percent_convention"]) ?? "",
    },
    volcano: {
      transforms: asString(volcanoRaw["transforms"]) ?? "",
      hit_rule: asString(volcanoRaw["hit_rule"]) ?? "",
      verdicts: asString(volcanoRaw["verdicts"]) ?? "",
      additional_fdr_lines: Array.isArray(volcanoRaw["additional_fdr_lines"])
        ? (volcanoRaw["additional_fdr_lines"] as unknown[]).filter(
            (v): v is number => typeof v === "number",
          )
        : [],
    },
    evaluation: {
      ruler: asString(evaluationRaw["ruler"]) ?? "",
      guarded_metrics: Array.isArray(evaluationRaw["guarded_metrics"])
        ? (evaluationRaw["guarded_metrics"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      guards: asString(evaluationRaw["guards"]) ?? "",
      normalized_metric: asString(evaluationRaw["normalized_metric"]) ?? "",
    },
    vocabulary: {
      cna_values: parseStringRecord(cnaRaw, `${what} vocabulary.cna_values`),
      diploid_zscores: asString(vocabularyRaw["diploid_zscores"]) ?? "",
      coexpression: asString(vocabularyRaw["coexpression"]) ?? "",
      enrichment_dispatchers:
        asString(vocabularyRaw["enrichment_dispatchers"]) ?? "",
    },
    screen: {
      geometry: parseNumberRecord(
        geometryRaw,
        `${what} screen.geometry`,
      ),
      planted_associations: asString(screenRaw["planted_associations"]) ?? "",
      label: asString(screenRaw["label"]) ?? "",
      scale: asString(screenRaw["scale"]) ?? "",
      missing: asString(screenRaw["missing"]) ?? "",
    },
    provenance: parsePharmacogenomicsProvenance(r["provenance"], what),
    analyze_url: asString(r["analyze_url"]) ?? "",
    note: asString(r["note"]) ?? "",
  };
}

/** Narrow one top-association row. */
function parsePharmacogenomicsAssociationRow(
  raw: unknown,
  what: string,
): PharmacogenomicsAssociationRow {
  const r = expectObject(raw, what);
  const drug = r["drug"];
  const feature = r["feature"];
  const nPos = r["n_pos"];
  const nNeg = r["n_neg"];
  if (
    typeof drug !== "string" ||
    typeof feature !== "string" ||
    typeof nPos !== "number" ||
    typeof nNeg !== "number"
  ) {
    throw new Error(`${what}: drug/feature/n_pos/n_neg are malformed`);
  }
  return {
    drug,
    feature,
    n_pos: nPos,
    n_neg: nNeg,
    mean_pos: asNumber(r["mean_pos"]),
    mean_neg: asNumber(r["mean_neg"]),
    delta_mean: asNumber(r["delta_mean"]),
    effect_size: asNumber(r["effect_size"]),
    glass_pos: asNumber(r["glass_pos"]),
    glass_neg: asNumber(r["glass_neg"]),
    t_pvalue: asNumber(r["t_pvalue"]),
    pvalue: asNumber(r["pvalue"]),
    qvalue: asNumber(r["qvalue"]),
    fdr_percent: asNumber(r["fdr_percent"]),
    tissue_pvalue: asNumber(r["tissue_pvalue"]),
    msi_pvalue: asNumber(r["msi_pvalue"]),
    volcano_x: asNumber(r["volcano_x"]),
    volcano_y: asNumber(r["volcano_y"]),
    verdict: asString(r["verdict"]),
  };
}

function parsePharmacogenomicsAnalyzeResponse(
  raw: unknown,
  what: string,
): PharmacogenomicsAnalyzeResponse {
  const r = expectObject(raw, what);
  const requestRaw = expectObject(r["request"], `${what} request`);
  const screenRaw = expectObject(r["screen"], `${what} screen`);
  const tissuesRaw = expectObject(screenRaw["tissues"], `${what} screen.tissues`);
  const msiRaw = expectObject(screenRaw["msi"], `${what} screen.msi`);
  const associationsRaw = expectObject(r["associations"], `${what} associations`);
  const topRaw = Array.isArray(associationsRaw["top"])
    ? (associationsRaw["top"] as unknown[])
    : [];
  const volcanoRaw = expectObject(r["volcano"], `${what} volcano`);
  const evaluationRaw = expectObject(r["evaluation"], `${what} evaluation`);
  return {
    request: {
      seed: asNumber(requestRaw["seed"]) ?? 0,
      top_k: asNumber(requestRaw["top_k"]) ?? 0,
      include_msi: requestRaw["include_msi"] === true,
    },
    screen: {
      n_cells: asNumber(screenRaw["n_cells"]) ?? 0,
      n_drugs: asNumber(screenRaw["n_drugs"]) ?? 0,
      n_features: asNumber(screenRaw["n_features"]) ?? 0,
      drugs: Array.isArray(screenRaw["drugs"])
        ? (screenRaw["drugs"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      features: Array.isArray(screenRaw["features"])
        ? (screenRaw["features"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      tissues: parseNumberRecord(tissuesRaw, `${what} screen.tissues`),
      msi: {
        msi_positive: asNumber(msiRaw["msi_positive"]) ?? 0,
        msi_negative: asNumber(msiRaw["msi_negative"]) ?? 0,
      },
      n_observed: asNumber(screenRaw["n_observed"]) ?? 0,
      n_entries: asNumber(screenRaw["n_entries"]) ?? 0,
      observed_fraction: asNumber(screenRaw["observed_fraction"]) ?? 0,
      label: asString(screenRaw["label"]) ?? "",
    },
    associations: {
      n_rows: asNumber(associationsRaw["n_rows"]) ?? 0,
      n_drugs_tested: asNumber(associationsRaw["n_drugs_tested"]) ?? 0,
      n_features_tested: asNumber(associationsRaw["n_features_tested"]) ?? 0,
      top_k: asNumber(associationsRaw["top_k"]) ?? 0,
      top: topRaw.map((row, i) =>
        parsePharmacogenomicsAssociationRow(
          row,
          `${what} associations.top[${i}]`,
        ),
      ),
      ordering: asString(associationsRaw["ordering"]) ?? "",
    },
    volcano: {
      n_sensitive: asNumber(volcanoRaw["n_sensitive"]) ?? 0,
      n_resistant: asNumber(volcanoRaw["n_resistant"]) ?? 0,
      n_not_significant: asNumber(volcanoRaw["n_not_significant"]) ?? 0,
      hit_rule: asString(volcanoRaw["hit_rule"]) ?? "",
    },
    evaluation: {
      n_observed: asNumber(evaluationRaw["n_observed"]) ?? 0,
      ruler: asString(evaluationRaw["ruler"]) ?? "",
      guarded: parseMetricRecord(evaluationRaw["guarded"], `${what} evaluation.guarded`),
      naive_drug_mean: parseMetricRecord(
        evaluationRaw["naive_drug_mean"],
        `${what} evaluation.naive_drug_mean`,
      ),
      naive_cell_mean: parseMetricRecord(
        evaluationRaw["naive_cell_mean"],
        `${what} evaluation.naive_cell_mean`,
      ),
      normalized_beyond_drug_mean: parseMetricRecord(
        evaluationRaw["normalized_beyond_drug_mean"],
        `${what} evaluation.normalized_beyond_drug_mean`,
      ),
      guards_note: asString(evaluationRaw["guards_note"]) ?? "",
      normalized_note: asString(evaluationRaw["normalized_note"]) ?? "",
    },
    provenance: parsePharmacogenomicsProvenance(r["provenance"], what),
    note: asString(r["note"]) ?? "",
  };
}

// -- the multi-objective lane's §8.1 runtime parsers ---------------------

/** Narrow the multi-objective provenance + scope block. */
function parseMultiobjectiveProvenance(
  raw: unknown,
  what: string,
): MultiobjectiveProvenance {
  const r = expectObject(raw, `${what} provenance`);
  return {
    backend: asString(r["backend"]) ?? "",
    source: asString(r["source"]) ?? "",
    n_repositories: asNumber(r["n_repositories"]) ?? 0,
    transfers_exactly: asString(r["transfers_exactly"]) ?? "",
    stays_upstream: asString(r["stays_upstream"]) ?? "",
    shipped_checkpoints: asNumber(r["shipped_checkpoints"]) ?? 0,
    shipped_checkpoints_note: asString(r["shipped_checkpoints_note"]) ?? "",
    dataset: asString(r["dataset"]) ?? "",
    disclaimer: asString(r["disclaimer"]) ?? "",
  };
}

/** Narrow the tunable-objective ledger (cross_validate key + sign flip). */
function parseObjectiveLedger(
  raw: unknown,
  what: string,
): Record<string, { cross_validate_key: string; sign: number }> {
  const r = expectObject(raw, what);
  const out: Record<string, { cross_validate_key: string; sign: number }> = {};
  for (const [label, value] of Object.entries(r)) {
    if (typeof value === "object" && value !== null) {
      const entry = value as Record<string, unknown>;
      const key = asString(entry["cross_validate_key"]);
      const sign = asNumber(entry["sign"]);
      if (key !== null && sign !== null) {
        out[label] = { cross_validate_key: key, sign };
      }
    }
  }
  return out;
}

function parseMultiobjectiveRegistryResponse(
  raw: unknown,
  what: string,
): MultiobjectiveRegistryResponse {
  const r = expectObject(raw, what);
  const orderRaw = expectObject(r["order"], `${what} order`);
  const indicatorRaw = expectObject(r["indicator"], `${what} indicator`);
  const spreadRaw = expectObject(r["spread"], `${what} spread`);
  const decompositionRaw = expectObject(
    r["decomposition"],
    `${what} decomposition`,
  );
  const studyRaw = expectObject(r["study"], `${what} study`);
  const tunableRaw = expectObject(
    r["tunable_objectives"],
    `${what} tunable_objectives`,
  );
  const reportRaw = expectObject(r["report"], `${what} report`);
  const requestRaw = expectObject(r["request"], `${what} request`);
  const datasetRaw = expectObject(r["dataset"], `${what} dataset`);
  return {
    order: {
      convention: asString(orderRaw["convention"]) ?? "",
      dominates: asString(orderRaw["dominates"]) ?? "",
      pareto_indices: asString(orderRaw["pareto_indices"]) ?? "",
      complexity: asString(orderRaw["complexity"]) ?? "",
    },
    indicator: {
      family: asString(indicatorRaw["family"]) ?? "",
      reference_contract: asString(indicatorRaw["reference_contract"]) ?? "",
      report_reference: asString(indicatorRaw["report_reference"]) ?? "",
      curve: asString(indicatorRaw["curve"]) ?? "",
      two_d_only: asString(indicatorRaw["two_d_only"]) ?? "",
    },
    spread: {
      family: asString(spreadRaw["family"]) ?? "",
      rule: asString(spreadRaw["rule"]) ?? "",
      json_note: asString(spreadRaw["json_note"]) ?? "",
    },
    decomposition: {
      family: asString(decompositionRaw["family"]) ?? "",
      form: asString(decompositionRaw["form"]) ?? "",
      rule: asString(decompositionRaw["rule"]) ?? "",
    },
    study: {
      samplers: Array.isArray(studyRaw["samplers"])
        ? (studyRaw["samplers"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      samplers_note: asString(studyRaw["samplers_note"]) ?? "",
      weight_modes: parseStringRecord(
        studyRaw["weight_modes"],
        `${what} study.weight_modes`,
      ),
      n_objectives: asString(studyRaw["n_objectives"]) ?? "",
      drift_note: asString(studyRaw["drift_note"]) ?? "",
    },
    tunable_objectives: {
      ledger: parseObjectiveLedger(
        tunableRaw["ledger"],
        `${what} tunable_objectives.ledger`,
      ),
      note: asString(tunableRaw["note"]) ?? "",
      bounds: asString(tunableRaw["bounds"]) ?? "",
    },
    report: {
      schema: Array.isArray(reportRaw["schema"])
        ? (reportRaw["schema"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      best_for: asString(reportRaw["best_for"]) ?? "",
      prune: asString(reportRaw["prune"]) ?? "",
    },
    request: {
      defaults:
        typeof r["request"] === "object" && requestRaw["defaults"] !== null
          ? (requestRaw["defaults"] as Record<string, unknown>)
          : {},
      bounds: parseStringRecord(requestRaw["bounds"], `${what} request.bounds`),
      dimension_kinds: Array.isArray(requestRaw["dimension_kinds"])
        ? (requestRaw["dimension_kinds"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      model_note: asString(requestRaw["model_note"]) ?? "",
    },
    dataset: {
      geometry: parseNumberRecord(
        datasetRaw["geometry"],
        `${what} dataset.geometry`,
      ),
      label: asString(datasetRaw["label"]) ?? "",
    },
    provenance: parseMultiobjectiveProvenance(r["provenance"], what),
    optimize_url: asString(r["optimize_url"]) ?? "",
    note: asString(r["note"]) ?? "",
  };
}

/** Narrow one MOTrial row (front/shortlist crowding optional). */
function parseMultiobjectiveTrialRow(
  raw: unknown,
  what: string,
): MultiobjectiveTrialRow {
  const r = expectObject(raw, what);
  const number = r["number"];
  if (typeof number !== "number") {
    throw new Error(`${what}: the trial number is malformed`);
  }
  const paramsRaw = expectObject(r["params"], `${what} params`);
  const params: Record<string, number> = {};
  for (const [key, value] of Object.entries(paramsRaw)) {
    if (typeof value === "number") {
      params[key] = value;
    }
  }
  const values = Array.isArray(r["values"])
    ? (r["values"] as unknown[]).filter(
        (v): v is number => typeof v === "number",
      )
    : [];
  const weights = Array.isArray(r["weights"])
    ? (r["weights"] as unknown[]).filter(
        (v): v is number => typeof v === "number",
      )
    : [];
  const row: MultiobjectiveTrialRow = {
    number,
    params,
    values,
    weights,
    scalar: asNumber(r["scalar"]),
    ...("crowding" in r
      ? {
          crowding: asNumber(r["crowding"]),
          crowding_is_boundary: r["crowding_is_boundary"] === true,
        }
      : {}),
  };
  return row;
}

function parseMultiobjectiveOptimizeResponse(
  raw: unknown,
  what: string,
): MultiobjectiveOptimizeResponse {
  const r = expectObject(raw, what);
  const requestRaw = expectObject(r["request"], `${what} request`);
  const dimensionsRaw = Array.isArray(requestRaw["dimensions"])
    ? (requestRaw["dimensions"] as unknown[])
    : [];
  const datasetRaw = expectObject(r["dataset"], `${what} dataset`);
  const studyRaw = expectObject(r["study"], `${what} study`);
  const frontRaw = Array.isArray(r["front"]) ? (r["front"] as unknown[]) : [];
  const indicatorRaw = expectObject(r["indicator"], `${what} indicator`);
  const curveRaw = Array.isArray(indicatorRaw["hypervolume_curve"])
    ? (indicatorRaw["hypervolume_curve"] as unknown[])
    : [];
  const refRaw = indicatorRaw["ref"];
  const bestRaw = expectObject(r["best_for"], `${what} best_for`);
  const shortlistRaw = expectObject(r["shortlist"], `${what} shortlist`);
  const shortlistTrialsRaw = Array.isArray(shortlistRaw["trials"])
    ? (shortlistRaw["trials"] as unknown[])
    : [];
  return {
    request: {
      model: asString(requestRaw["model"]) ?? "",
      task: asString(requestRaw["task"]) ?? "",
      objectives: Array.isArray(requestRaw["objectives"])
        ? (requestRaw["objectives"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      dimensions: dimensionsRaw.map((dim, i) => {
        const d = expectObject(dim, `${what} request.dimensions[${i}]`);
        return {
          name: asString(d["name"]) ?? "",
          kind: asString(d["kind"]) ?? "",
          low: asNumber(d["low"]) ?? 0,
          high: asNumber(d["high"]) ?? 0,
        };
      }),
      n_trials: asNumber(requestRaw["n_trials"]) ?? 0,
      cv: asNumber(requestRaw["cv"]) ?? 0,
      sampler: asString(requestRaw["sampler"]) ?? "",
      seed: asNumber(requestRaw["seed"]) ?? 0,
      weight_mode: asString(requestRaw["weight_mode"]) ?? "",
      n_weight_vectors: asNumber(requestRaw["n_weight_vectors"]) ?? 0,
      shortlist_keep: asNumber(requestRaw["shortlist_keep"]) ?? 0,
    },
    dataset: {
      n_samples: asNumber(datasetRaw["n_samples"]) ?? 0,
      n_features: asNumber(datasetRaw["n_features"]) ?? 0,
      task: asString(datasetRaw["task"]) ?? "",
      label: asString(datasetRaw["label"]) ?? "",
    },
    study: {
      n_trials: asNumber(studyRaw["n_trials"]) ?? 0,
      n_objectives: asNumber(studyRaw["n_objectives"]) ?? 0,
      objective_names: Array.isArray(studyRaw["objective_names"])
        ? (studyRaw["objective_names"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      sampler: asString(studyRaw["sampler"]) ?? "",
      weight_mode: asString(studyRaw["weight_mode"]) ?? "",
      n_weight_vectors: asNumber(studyRaw["n_weight_vectors"]) ?? 0,
      n_front: asNumber(studyRaw["n_front"]) ?? 0,
      objectives_note: asString(studyRaw["objectives_note"]) ?? "",
    },
    front: frontRaw.map((row, i) =>
      parseMultiobjectiveTrialRow(row, `${what} front[${i}]`),
    ),
    crowding_note: asString(r["crowding_note"]) ?? "",
    indicator: {
      ref: Array.isArray(refRaw)
        ? (refRaw as unknown[]).filter(
            (v): v is number => typeof v === "number",
          )
        : null,
      hypervolume: asNumber(indicatorRaw["hypervolume"]),
      hypervolume_curve: curveRaw.filter(
        (v): v is number => typeof v === "number",
      ),
      two_d_only: asString(indicatorRaw["two_d_only"]) ?? "",
      curve_note: asString(indicatorRaw["curve_note"]) ?? "",
    },
    best_for: Object.fromEntries(
      Object.entries(bestRaw).map(([name, row]) => [
        name,
        parseMultiobjectiveTrialRow(row, `${what} best_for ${name}`),
      ]),
    ),
    best_for_note: asString(r["best_for_note"]) ?? "",
    shortlist: {
      keep: asNumber(shortlistRaw["keep"]) ?? 0,
      n_front: asNumber(shortlistRaw["n_front"]) ?? 0,
      n_trials: asNumber(shortlistRaw["n_trials"]) ?? 0,
      trials: shortlistTrialsRaw.map((row, i) =>
        parseMultiobjectiveTrialRow(row, `${what} shortlist.trials[${i}]`),
      ),
      note: asString(shortlistRaw["note"]) ?? "",
    },
    provenance: parseMultiobjectiveProvenance(r["provenance"], what),
    timings_note: asString(r["timings_note"]) ?? "",
    note: asString(r["note"]) ?? "",
  };
}

// -- the multiresponse lane parsers (part17 §5 — the M1 lane;
// row-validated, never trusted) ------------------------------------------

/** Narrow one strategy row of the multiresponse registry. */
function parseMultiresponseStrategy(
  raw: unknown,
  what: string,
): MultiresponseStrategy {
  const r = expectObject(raw, what);
  const key = r["key"];
  const label = r["label"];
  const mechanism = r["mechanism"];
  if (
    typeof key !== "string" ||
    typeof label !== "string" ||
    typeof mechanism !== "string" ||
    !["native", "wrapper", "chain"].includes(mechanism)
  ) {
    throw new Error(`${what}: key/label/mechanism are not the registry shape`);
  }
  return {
    key,
    label,
    mechanism: mechanism as MultiresponseStrategy["mechanism"],
    native_multi_output: r["native_multi_output"] === true,
    sample_weight_support: r["sample_weight_support"] === true,
    notes: asString(r["notes"]) ?? "",
    base_keys: Array.isArray(r["base_keys"])
      ? (r["base_keys"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
    default_base: asString(r["default_base"]),
  };
}

/** Narrow one per-target metric block. */
function parseMultiresponsePerTargetMetrics(
  raw: unknown,
  what: string,
): MultiresponsePerTargetMetrics {
  const r = expectObject(raw, what);
  return {
    r2: asNumber(r["r2"]) ?? 0,
    rmse: asNumber(r["rmse"]) ?? 0,
    mae: asNumber(r["mae"]) ?? 0,
    mse: asNumber(r["mse"]) ?? 0,
    explained_variance: asNumber(r["explained_variance"]) ?? 0,
    n_test_rows: asNumber(r["n_test_rows"]) ?? 0,
  };
}

/** Narrow the multiresponse model view (the comparison surface row). */
function parseMultiresponseModelView(
  raw: unknown,
  what: string,
): MultiresponseModelView {
  const r = expectObject(raw, what);
  const mrRaw = expectObject(r["multiresponse"], `${what} multiresponse`);
  const perTargetRaw = mrRaw["per_target_metrics"];
  const aggRaw = mrRaw["aggregate_metrics"];
  const perTarget: MultiresponseModelView["multiresponse"]["per_target_metrics"] =
    perTargetRaw && typeof perTargetRaw === "object"
      ? Object.fromEntries(
          Object.entries(
            perTargetRaw as Record<string, unknown>,
          ).map(([name, row]) => [
            name,
            parseMultiresponsePerTargetMetrics(
              row,
              `${what} per_target_metrics ${name}`,
            ),
          ]),
        )
      : null;
  const aggUniform: Record<string, number> = {};
  const aggWeighted: Record<string, number> = {};
  if (aggRaw && typeof aggRaw === "object") {
    for (const [k, v] of Object.entries(aggRaw as Record<string, unknown>)) {
      if (k === "uniform" && v && typeof v === "object") {
        for (const [mk, mv] of Object.entries(v as Record<string, unknown>)) {
          if (typeof mv === "number") aggUniform[mk] = mv;
        }
      }
      if (k === "weighted" && v && typeof v === "object") {
        for (const [mk, mv] of Object.entries(v as Record<string, unknown>)) {
          if (typeof mv === "number") aggWeighted[mk] = mv;
        }
      }
    }
  }
  const modelId = r["model_id"];
  if (typeof modelId !== "string") {
    throw new Error(`${what}: model_id is not a string`);
  }
  return {
    model_id: modelId,
    name: asString(r["name"]) ?? "",
    task: asString(r["task"]) ?? "",
    dataset_id: asString(r["dataset_id"]),
    target: asString(r["target"]) ?? "",
    features: Array.isArray(r["features"])
      ? (r["features"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
    n_train_rows: asNumber(r["n_train_rows"]) ?? 0,
    cv_mean: asNumber(r["cv_mean"]),
    fit_seconds: asNumber(r["fit_seconds"]) ?? 0,
    created_at: asString(r["created_at"]) ?? "",
    multiresponse: {
      target_names: Array.isArray(mrRaw["target_names"])
        ? (mrRaw["target_names"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      target_order: Array.isArray(mrRaw["target_order"])
        ? (mrRaw["target_order"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      mechanism: asString(mrRaw["mechanism"]) ?? "",
      missing_strategy: asString(mrRaw["missing_strategy"]) ?? "",
      per_target_metrics: perTarget,
      aggregate_metrics:
        Object.keys(aggUniform).length > 0
          ? { uniform: aggUniform, weighted: aggWeighted }
          : null,
      n_test_rows: asNumber(mrRaw["n_test_rows"]),
    },
  };
}

/** Narrow the registry answer. */
function parseMultiresponseRegistryResponse(
  raw: unknown,
  what: string,
): MultiresponseRegistryResponse {
  const r = expectObject(raw, what);
  const demoRaw = expectObject(r["demo_dataset"], `${what} demo_dataset`);
  const strategiesRaw = Array.isArray(r["strategies"])
    ? (r["strategies"] as unknown[])
    : [];
  const missingRaw =
    r["missing_strategies"] && typeof r["missing_strategies"] === "object"
      ? (r["missing_strategies"] as Record<string, unknown>)
      : {};
  const missing: MultiresponseRegistryResponse["missing_strategies"] =
    Object.fromEntries(
      Object.entries(missingRaw).map(([name, block]) => {
        const b = expectObject(block, `${what} missing_strategies ${name}`);
        return [
          name,
          {
            label: asString(b["label"]) ?? "",
            description: asString(b["description"]) ?? "",
            mechanisms: Array.isArray(b["mechanisms"])
              ? (b["mechanisms"] as unknown[]).filter(
                  (v): v is string => typeof v === "string",
                )
              : [],
          },
        ];
      }),
    );
  const bounds: Record<string, number | readonly number[]> = {};
  if (r["bounds"] && typeof r["bounds"] === "object") {
    for (const [k, v] of Object.entries(
      r["bounds"] as Record<string, unknown>,
    )) {
      if (typeof v === "number") bounds[k] = v;
      else if (Array.isArray(v)) bounds[k] = v.filter((n): n is number => typeof n === "number");
    }
  }
  return {
    strategies: strategiesRaw.map((row, i) =>
      parseMultiresponseStrategy(row, `${what} strategies[${i}]`),
    ),
    missing_strategies: missing,
    metrics: Array.isArray(r["metrics"])
      ? (r["metrics"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
    max_sample_rows: asNumber(r["max_sample_rows"]) ?? 0,
    demo_dataset: {
      features: Array.isArray(demoRaw["features"])
        ? (demoRaw["features"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      targets: Array.isArray(demoRaw["targets"])
        ? (demoRaw["targets"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      n_rows: asNumber(demoRaw["n_rows"]) ?? 0,
      synthetic: demoRaw["synthetic"] === true,
      seed: asNumber(demoRaw["seed"]) ?? 0,
      note: asString(demoRaw["note"]) ?? "",
    },
    bounds,
    scope_statement: asString(r["scope_statement"]) ?? "",
  };
}

/** Narrow the train answer (the full §5.4/§5.5/§5.6 round trip). */
function parseMultiresponseTrainResponse(
  raw: unknown,
  what: string,
): MultiresponseTrainResponse {
  const r = expectObject(raw, what);
  const model = parseMultiresponseModelView(r["model"], `${what} model`);
  const provRaw = expectObject(r["provenance"], `${what} provenance`);
  const cfgRaw = expectObject(r["configuration"], `${what} configuration`);
  const dataRaw = expectObject(r["data"], `${what} data`);
  const splitRaw = expectObject(dataRaw["split"], `${what} data.split`);
  const perTargetRaw = expectObject(
    r["per_target_metrics"],
    `${what} per_target_metrics`,
  );
  const aggRaw = expectObject(
    r["aggregate_metrics"],
    `${what} aggregate_metrics`,
  );
  const resRaw = expectObject(r["residuals"], `${what} residuals`);
  const resPerRaw =
    resRaw["per_target"] && typeof resRaw["per_target"] === "object"
      ? (resRaw["per_target"] as Record<string, unknown>)
      : {};
  const corrRaw =
    r["target_correlation_matrix"] &&
    typeof r["target_correlation_matrix"] === "object"
      ? (r["target_correlation_matrix"] as Record<string, unknown>)
      : {};
  const corr: MultiresponseTrainResponse["target_correlation_matrix"] =
    Object.fromEntries(
      Object.entries(corrRaw).map(([col, row]) => {
        const rowObj =
          row && typeof row === "object" ? (row as Record<string, unknown>) : {};
        return [
          col,
          Object.fromEntries(
            Object.entries(rowObj).map(([k, v]) => [
              k,
              typeof v === "number" ? v : null,
            ]),
          ),
        ];
      }),
    );
  const cvRaw = expectObject(r["cv"], `${what} cv`);
  const cvPerRaw =
    cvRaw["per_target"] && typeof cvRaw["per_target"] === "object"
      ? (cvRaw["per_target"] as Record<string, unknown>)
      : {};
  const aggUniform: Record<string, number> = {};
  const aggWeighted: Record<string, number> = {};
  for (const [k, v] of Object.entries(aggRaw)) {
    if (k === "uniform" && v && typeof v === "object") {
      for (const [mk, mv] of Object.entries(v as Record<string, unknown>)) {
        if (typeof mv === "number") aggUniform[mk] = mv;
      }
    }
    if (k === "weighted" && v && typeof v === "object") {
      for (const [mk, mv] of Object.entries(v as Record<string, unknown>)) {
        if (typeof mv === "number") aggWeighted[mk] = mv;
      }
    }
  }
  const usable: Record<string, number> = {};
  const missingCounts: Record<string, number> = {};
  for (const [k, v] of Object.entries(
    expectObject(dataRaw["usable_rows_per_target"], `${what} usable`),
  )) {
    if (typeof v === "number") usable[k] = v;
  }
  for (const [k, v] of Object.entries(
    expectObject(
      dataRaw["missing_counts_per_target"],
      `${what} missing_counts`,
    ),
  )) {
    if (typeof v === "number") missingCounts[k] = v;
  }
  return {
    model,
    run_id: asString(r["run_id"]) ?? "",
    provenance: {
      source: asString(provRaw["source"]) ?? "",
      dataset_id: asString(provRaw["dataset_id"]),
      dataset_name: asString(provRaw["dataset_name"]) ?? "",
      synthetic: provRaw["synthetic"] === true,
      note: asString(provRaw["note"]) ?? undefined,
    },
    configuration: {
      strategy: asString(cfgRaw["strategy"]) ?? "",
      mechanism: asString(cfgRaw["mechanism"]) ?? "",
      params: (cfgRaw["params"] && typeof cfgRaw["params"] === "object"
        ? cfgRaw["params"]
        : {}) as Record<string, unknown>,
      preprocess: cfgRaw["preprocess"] === true,
      target_scaling: cfgRaw["target_scaling"] === true,
      test_size: asNumber(cfgRaw["test_size"]) ?? 0.25,
      cv_folds: asNumber(cfgRaw["cv_folds"]) ?? 5,
      missing: asString(cfgRaw["missing"]) ?? "complete_case",
      group_column: asString(cfgRaw["group_column"]),
      sample_weight_column: asString(cfgRaw["sample_weight_column"]),
      target_weights: Array.isArray(cfgRaw["target_weights"])
        ? (cfgRaw["target_weights"] as unknown[]).filter(
            (v): v is number => typeof v === "number",
          )
        : null,
      random_state: asNumber(cfgRaw["random_state"]) ?? 42,
    },
    data: {
      n_rows: asNumber(dataRaw["n_rows"]) ?? 0,
      usable_rows_per_target: usable,
      missing_counts_per_target: missingCounts,
      dropped_train_rows: asNumber(dataRaw["dropped_train_rows"]) ?? 0,
      split: {
        strategy: asString(splitRaw["strategy"]) ?? "",
        train_rows: asNumber(splitRaw["train_rows"]) ?? 0,
        test_rows: asNumber(splitRaw["test_rows"]) ?? 0,
        n_train_groups: asNumber(splitRaw["n_train_groups"]),
        n_test_groups: asNumber(splitRaw["n_test_groups"]),
        group_overlap: Array.isArray(splitRaw["group_overlap"])
          ? (splitRaw["group_overlap"] as unknown[]).filter(
              (v): v is string => typeof v === "string",
            )
          : [],
        test_size: asNumber(splitRaw["test_size"]) ?? 0,
        random_state: asNumber(splitRaw["random_state"]) ?? 0,
      },
    },
    per_target_metrics: Object.fromEntries(
      Object.entries(perTargetRaw).map(([name, row]) => [
        name,
        parseMultiresponsePerTargetMetrics(
          row,
          `${what} per_target_metrics ${name}`,
        ),
      ]),
    ),
    aggregate_metrics: { uniform: aggUniform, weighted: aggWeighted },
    residuals: {
      per_target: Object.fromEntries(
        Object.entries(resPerRaw).map(([name, row]) => {
          const b = expectObject(row, `${what} residuals ${name}`);
          const nums = (key: string): number[] =>
            Array.isArray(b[key])
              ? (b[key] as unknown[]).filter(
                  (v): v is number => typeof v === "number",
                )
              : [];
          return [
            name,
            {
              actual: nums("actual"),
              predicted: nums("predicted"),
              residual: nums("residual"),
              n_rows: asNumber(b["n_rows"]) ?? 0,
              n_sampled: asNumber(b["n_sampled"]) ?? 0,
            },
          ];
        }),
      ),
      max_rows: asNumber(resRaw["max_rows"]) ?? 0,
    },
    target_correlation_matrix: corr,
    cv: {
      folds: asNumber(cvRaw["folds"]) ?? 0,
      per_target: Object.fromEntries(
        Object.entries(cvPerRaw).map(([name, row]) => {
          const b = expectObject(row, `${what} cv ${name}`);
          return [
            name,
            {
              r2_mean: asNumber(b["r2_mean"]) ?? 0,
              r2_std: asNumber(b["r2_std"]) ?? 0,
              r2_folds: Array.isArray(b["r2_folds"])
                ? (b["r2_folds"] as unknown[]).filter(
                    (v): v is number => typeof v === "number",
                  )
                : [],
            },
          ];
        }),
      ),
      aggregate: {
        r2_uniform: asNumber(
          (cvRaw["aggregate"] && typeof cvRaw["aggregate"] === "object"
            ? (cvRaw["aggregate"] as Record<string, unknown>)["r2_uniform"]
            : null),
        ) ?? 0,
        r2_weighted: asNumber(
          (cvRaw["aggregate"] && typeof cvRaw["aggregate"] === "object"
            ? (cvRaw["aggregate"] as Record<string, unknown>)["r2_weighted"]
            : null),
        ) ?? 0,
      },
      folds_table: Array.isArray(cvRaw["folds_table"])
        ? (cvRaw["folds_table"] as unknown[]).filter(
            (v): v is Record<string, number> =>
              typeof v === "object" && v !== null,
          )
        : [],
    },
    contract: (r["contract"] && typeof r["contract"] === "object"
      ? r["contract"]
      : {}) as Record<string, unknown>,
    fit_seconds: asNumber(r["fit_seconds"]) ?? 0,
    max_sample_rows: asNumber(r["max_sample_rows"]) ?? 0,
  };
}

/** Narrow the predict answer. */
function parseMultiresponsePredictResponse(
  raw: unknown,
  what: string,
): MultiresponsePredictResponse {
  const r = expectObject(raw, what);
  const predictionsRaw = Array.isArray(r["predictions"])
    ? (r["predictions"] as unknown[])
    : [];
  const units: Record<string, string> = {};
  if (r["target_units"] && typeof r["target_units"] === "object") {
    for (const [k, v] of Object.entries(
      r["target_units"] as Record<string, unknown>,
    )) {
      if (typeof v === "string") units[k] = v;
    }
  }
  return {
    model_id: asString(r["model_id"]) ?? "",
    target_names: Array.isArray(r["target_names"])
      ? (r["target_names"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
    target_units: units,
    predictions: predictionsRaw.map((row, i) => {
      const obj =
        row && typeof row === "object" ? (row as Record<string, unknown>) : {};
      const out: Record<string, number> = {};
      for (const [k, v] of Object.entries(obj)) {
        if (typeof v === "number") out[k] = v;
      }
      if (Object.keys(out).length === 0) {
        throw new Error(`${what}: predictions[${i}] is not a numeric row`);
      }
      return out;
    }),
    n_rows: asNumber(r["n_rows"]) ?? 0,
    note: asString(r["note"]) ?? "",
  };
}

/** Narrow the models listing answer. */
function parseMultiresponseModelsResponse(
  raw: unknown,
  what: string,
): {
  models: readonly MultiresponseModelView[];
  metrics: readonly string[];
} {
  const r = expectObject(raw, what);
  const modelsRaw = Array.isArray(r["models"])
    ? (r["models"] as unknown[])
    : [];
  return {
    models: modelsRaw.map((row, i) =>
      parseMultiresponseModelView(row, `${what} models[${i}]`),
    ),
    metrics: Array.isArray(r["metrics"])
      ? (r["metrics"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
  };
}

// -- RSM lane parsers (part17 §6 — the R1 lane; row-validated, never
// trusted) -----------------------------------------------------------

/** Narrow one factor view of the RSM answers. */
function parseRsmFactorView(raw: unknown, what: string): RsmFactorView {
  const r = expectObject(raw, what);
  const name = r["name"];
  if (typeof name !== "string") {
    throw new Error(`${what}: name is not a string`);
  }
  const low = asNumber(r["low"]);
  const high = asNumber(r["high"]);
  if (low === null || high === null) {
    throw new Error(`${what}: low/high are not numbers`);
  }
  return {
    name,
    low,
    high,
    unit: asString(r["unit"]),
    center: asNumber(r["center"]) ?? (low + high) / 2,
    half_range: asNumber(r["half_range"]) ?? (high - low) / 2,
  };
}

/** Narrow the RSM registry answer. */
function parseRsmRegistryResponse(
  raw: unknown,
  what: string,
): RsmRegistryResponse {
  const r = expectObject(raw, what);
  const modelRaw = expectObject(r["model"], `${what} model`);
  const demoRaw = expectObject(r["demo_dataset"], `${what} demo_dataset`);
  const designsRaw = Array.isArray(r["designs"])
    ? (r["designs"] as unknown[])
    : [];
  return {
    designs: designsRaw.map((row, i) => {
      const d = expectObject(row, `${what} designs[${i}]`);
      return {
        key: (asString(d["key"]) ?? "ccd") as RsmDesignInfo["key"],
        label: asString(d["label"]) ?? "",
        alpha_types: Array.isArray(d["alpha_types"])
          ? (d["alpha_types"] as unknown[]).filter(
              (v): v is string => typeof v === "string",
            )
          : null,
        min_factors: asNumber(d["min_factors"]) ?? 2,
        max_factors: asNumber(d["max_factors"]) ?? 10,
        runs_formula: asString(d["runs_formula"]) ?? "",
        blocking: d["blocking"] === true,
        note: asString(d["note"]) ?? "",
      };
    }),
    goals: Array.isArray(r["goals"])
      ? (r["goals"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
    model: {
      order: asString(modelRaw["order"]) ?? "",
      terms: asString(modelRaw["terms"]) ?? "",
      term_count_formula: asString(modelRaw["term_count_formula"]) ?? "",
    },
    bounds:
      r["bounds"] && typeof r["bounds"] === "object"
        ? (r["bounds"] as Record<string, unknown>)
        : {},
    demo_dataset: {
      name: asString(demoRaw["name"]) ?? "",
      synthetic: demoRaw["synthetic"] === true,
      factors: Array.isArray(demoRaw["factors"])
        ? (demoRaw["factors"] as unknown[]).map((f, i) =>
            parseRsmFactorView(f, `${what} demo factors[${i}]`),
          )
        : [],
      responses: Array.isArray(demoRaw["responses"])
        ? (demoRaw["responses"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      runs: asNumber(demoRaw["runs"]) ?? 0,
      design: asString(demoRaw["design"]) ?? "",
      noise: demoRaw["noise"] === true,
      note: asString(demoRaw["note"]) ?? "",
    },
    scope_statement: asString(r["scope_statement"]) ?? "",
  };
}

/** Narrow the design-generation answer. */
function parseRsmDesignResponse(
  raw: unknown,
  what: string,
): RsmDesignResponse {
  const r = expectObject(raw, what);
  const designRaw = expectObject(r["design"], `${what} design`);
  const runsRaw = Array.isArray(r["runs"]) ? (r["runs"] as unknown[]) : [];
  const counts: Record<string, number> = {};
  if (r["run_type_counts"] && typeof r["run_type_counts"] === "object") {
    for (const [k, v] of Object.entries(
      r["run_type_counts"] as Record<string, unknown>,
    )) {
      if (typeof v === "number") counts[k] = v;
    }
  }
  return {
    factors: Array.isArray(r["factors"])
      ? (r["factors"] as unknown[]).map((f, i) =>
          parseRsmFactorView(f, `${what} factors[${i}]`),
        )
      : [],
    design: {
      type: asString(designRaw["type"]) ?? "ccd",
      alpha: asNumber(designRaw["alpha"]),
      alpha_type: asString(designRaw["alpha_type"]),
      n_factors: asNumber(designRaw["n_factors"]) ?? 0,
      runs: asNumber(designRaw["runs"]) ?? 0,
      center_points: asNumber(designRaw["center_points"]) ?? 0,
      replicates: asNumber(designRaw["replicates"]) ?? 1,
      randomized: designRaw["randomized"] === true,
      seed: asNumber(designRaw["seed"]) ?? 42,
      blocks: designRaw["blocks"] === true,
    },
    runs: runsRaw.map((row, i) => {
      const run = expectObject(row, `${what} runs[${i}]`);
      return {
        run_id: asString(run["run_id"]) ?? `r${i + 1}`,
        run_order: asNumber(run["run_order"]) ?? i + 1,
        block: asNumber(run["block"]) ?? 1,
        run_type: asString(run["run_type"]) ?? "",
        coded: numberRecord(run["coded"], `${what} runs[${i}] coded`),
        actual: numberRecord(run["actual"], `${what} runs[${i}] actual`),
      };
    }),
    run_type_counts: counts,
    assumptions: Array.isArray(r["assumptions"])
      ? (r["assumptions"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
    export_note: asString(r["export_note"]) ?? "",
  };
}

/** Narrow one record of string -> number (a coded/actual level row). */
function numberRecord(
  raw: unknown,
  what: string,
): Readonly<Record<string, number>> {
  const r = expectObject(raw, what);
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(r)) {
    if (typeof v === "number" && Number.isFinite(v)) {
      out[k] = v;
    }
  }
  return out;
}

/** Narrow one coefficient row of the fit answer. */
function parseRsmCoefficientRow(
  raw: unknown,
  what: string,
): RsmCoefficientRow {
  const r = expectObject(raw, what);
  const term = asString(r["term"]);
  if (term === null) {
    throw new Error(`${what}: term is not a string`);
  }
  return {
    term,
    label: asString(r["label"]) ?? term,
    kind: asString(r["kind"]) ?? "",
    estimate: asNumber(r["estimate"]) ?? 0,
    std_error: asNumber(r["std_error"]),
    t: asNumber(r["t"]),
    p: asNumber(r["p"]),
    ci_low: asNumber(r["ci_low"]),
    ci_high: asNumber(r["ci_high"]),
  };
}

/** Narrow one fitted-vs-observed diagnostics row. */
function parseRsmDiagnosticsRow(
  raw: unknown,
  what: string,
): RsmDiagnosticsRow {
  const r = expectObject(raw, what);
  return {
    run: asNumber(r["run"]) ?? 0,
    observed: asNumber(r["observed"]) ?? 0,
    fitted: asNumber(r["fitted"]) ?? 0,
    residual: asNumber(r["residual"]) ?? 0,
    studentized: asNumber(r["studentized"]) ?? 0,
    leverage: asNumber(r["leverage"]) ?? 0,
    cooks_distance: asNumber(r["cooks_distance"]) ?? 0,
    outlier_flag: r["outlier_flag"] === true,
    influential_flag: r["influential_flag"] === true,
  };
}

/** Narrow one RSM model view (the comparison surface row). */
function parseRsmModelView(raw: unknown, what: string): RsmModelView {
  const r = expectObject(raw, what);
  const rsmRaw = expectObject(r["rsm"], `${what} rsm`);
  return {
    model_id: asString(r["model_id"]) ?? "",
    name: asString(r["name"]) ?? "",
    task: asString(r["task"]) ?? "",
    dataset_id: asString(r["dataset_id"]),
    target: asString(r["target"]) ?? "",
    features: Array.isArray(r["features"])
      ? (r["features"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
    n_train_rows: asNumber(r["n_train_rows"]) ?? 0,
    cv_mean: asNumber(r["cv_mean"]),
    fit_seconds: asNumber(r["fit_seconds"]) ?? 0,
    created_at: asString(r["created_at"]) ?? "",
    rsm: {
      kind: asString(rsmRaw["kind"]) ?? "rsm",
      factors: Array.isArray(rsmRaw["factors"])
        ? (rsmRaw["factors"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      responses: Array.isArray(rsmRaw["responses"])
        ? (rsmRaw["responses"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      n_terms: asNumber(rsmRaw["n_terms"]) ?? 0,
      design_meta:
        rsmRaw["design_meta"] && typeof rsmRaw["design_meta"] === "object"
          ? (rsmRaw["design_meta"] as Record<string, unknown>)
          : {},
      df_residual: asNumber(rsmRaw["df_residual"]) ?? 0,
      rank: asNumber(rsmRaw["rank"]) ?? 0,
    },
  };
}

/** Narrow the fit answer (the full second-order surface). */
function parseRsmFitResponse(raw: unknown, what: string): RsmFitResponse {
  const r = expectObject(raw, what);
  const dataRaw = expectObject(r["data"], `${what} data`);
  const configRaw = expectObject(r["configuration"], `${what} configuration`);
  const coefficients: Record<string, readonly RsmCoefficientRow[]> = {};
  if (r["coefficients"] && typeof r["coefficients"] === "object") {
    for (const [response, rows] of Object.entries(
      r["coefficients"] as Record<string, unknown>,
    )) {
      coefficients[response] = Array.isArray(rows)
        ? (rows as unknown[]).map((row, i) =>
            parseRsmCoefficientRow(row, `${what} coefficients[${i}]`),
          )
        : [];
    }
  }
  const diagnostics: Record<string, readonly RsmDiagnosticsRow[]> = {};
  if (r["diagnostics"] && typeof r["diagnostics"] === "object") {
    for (const [response, rows] of Object.entries(
      r["diagnostics"] as Record<string, unknown>,
    )) {
      diagnostics[response] = Array.isArray(rows)
        ? (rows as unknown[]).map((row, i) =>
            parseRsmDiagnosticsRow(row, `${what} diagnostics[${i}]`),
          )
        : [];
    }
  }
  const fitStats: Record<string, RsmFitStatistics> = {};
  if (r["fit_statistics"] && typeof r["fit_statistics"] === "object") {
    for (const [response, blockRaw] of Object.entries(
      r["fit_statistics"] as Record<string, unknown>,
    )) {
      const s = expectObject(blockRaw, `${what} fit_statistics ${response}`);
      fitStats[response] = {
        n_runs: asNumber(s["n_runs"]) ?? 0,
        df_residual: asNumber(s["df_residual"]) ?? 0,
        rank: asNumber(s["rank"]) ?? 0,
        r2: asNumber(s["r2"]) ?? 0,
        adj_r2: asNumber(s["adj_r2"]) ?? 0,
        pred_r2: asNumber(s["pred_r2"]) ?? 0,
        rmse: asNumber(s["rmse"]) ?? 0,
        press: asNumber(s["press"]) ?? 0,
        confidence: asNumber(s["confidence"]) ?? 0.95,
        outliers_flagged: asNumber(s["outliers_flagged"]) ?? 0,
        influential_flagged: asNumber(s["influential_flagged"]) ?? 0,
        cooks_threshold: asNumber(s["cooks_threshold"]) ?? 1,
        machine_precision_fit: s["machine_precision_fit"] === true,
        diagnostics_note: asString(s["diagnostics_note"]),
        thresholds_note: asString(s["thresholds_note"]) ?? "",
      };
    }
  }
  const anova: Record<string, Readonly<Record<string, unknown>>> = {};
  if (r["anova"] && typeof r["anova"] === "object") {
    for (const [response, block] of Object.entries(
      r["anova"] as Record<string, unknown>,
    )) {
      anova[response] =
        block && typeof block === "object"
          ? (block as Record<string, unknown>)
          : {};
    }
  }
  const aliasingRaw = dataRaw["aliasing"];
  return {
    model: parseRsmModelView(r["model"], `${what} model`),
    run_id: asString(r["run_id"]) ?? "",
    provenance: {
      source: asString((r["provenance"] as Record<string, unknown>)?.["source"]) ?? "",
      dataset_id: asString((r["provenance"] as Record<string, unknown>)?.["dataset_id"]),
      dataset_name:
        asString((r["provenance"] as Record<string, unknown>)?.["dataset_name"]) ?? "",
      synthetic:
        (r["provenance"] as Record<string, unknown>)?.["synthetic"] === true,
      note: asString((r["provenance"] as Record<string, unknown>)?.["note"]) ?? undefined,
    },
    configuration: {
      factors: Array.isArray(configRaw["factors"])
        ? (configRaw["factors"] as unknown[]).map((f, i) =>
            parseRsmFactorView(f, `${what} factors[${i}]`),
          )
        : [],
      response_columns: Array.isArray(configRaw["response_columns"])
        ? (configRaw["response_columns"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      design_meta:
        configRaw["design_meta"] &&
        typeof configRaw["design_meta"] === "object"
          ? (configRaw["design_meta"] as Record<string, unknown>)
          : {},
      confidence: asNumber(configRaw["confidence"]) ?? 0.95,
    },
    data: {
      n_runs: asNumber(dataRaw["n_runs"]) ?? 0,
      dropped_rows: asNumber(dataRaw["dropped_rows"]) ?? 0,
      terms: Array.isArray(dataRaw["terms"])
        ? (dataRaw["terms"] as unknown[]).map((t) => {
            const tr = expectObject(t, `${what} term`);
            return {
              key: asString(tr["key"]) ?? "",
              kind: asString(tr["kind"]) ?? "",
              label: asString(tr["label"]) ?? "",
            };
          })
        : [],
      hierarchy_notes: Array.isArray(dataRaw["hierarchy_notes"])
        ? (dataRaw["hierarchy_notes"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      aliasing:
        aliasingRaw && typeof aliasingRaw === "object"
          ? {
              rank:
                asNumber((aliasingRaw as Record<string, unknown>)["rank"]) ?? 0,
              columns:
                asNumber((aliasingRaw as Record<string, unknown>)["columns"]) ?? 0,
              aliased: Array.isArray(
                (aliasingRaw as Record<string, unknown>)["aliased"],
              )
                ? ((aliasingRaw as Record<string, unknown>)["aliased"] as unknown[]).filter(
                    (v): v is string => typeof v === "string",
                  )
                : null,
              note:
                asString((aliasingRaw as Record<string, unknown>)["note"]) ?? "",
            }
          : null,
    },
    coefficients,
    anova,
    diagnostics,
    fit_statistics: fitStats,
    fit_seconds: asNumber(r["fit_seconds"]) ?? 0,
  };
}

/** Narrow the prediction answer (the zone vocabulary rides each row). */
function parseRsmPredictResponse(
  raw: unknown,
  what: string,
): RsmPredictResponse {
  const r = expectObject(raw, what);
  const rowsRaw = Array.isArray(r["rows"]) ? (r["rows"] as unknown[]) : [];
  const extrapolationRaw = expectObject(
    r["extrapolation"],
    `${what} extrapolation`,
  );
  return {
    model_id: asString(r["model_id"]) ?? "",
    rows: rowsRaw.map((row, i) => {
      const o = expectObject(row, `${what} rows[${i}]`);
      const zone = asString(o["zone"]) ?? "inside_ranges";
      return {
        run: asNumber(o["run"]) ?? i + 1,
        zone: (["inside_ranges", "axial_region", "outside_design"].includes(zone)
          ? zone
          : "inside_ranges") as "inside_ranges" | "axial_region" | "outside_design",
        coded: numberRecord(o["coded"], `${what} rows[${i}] coded`),
        actual: numberRecord(o["actual"], `${what} rows[${i}] actual`),
        ...Object.fromEntries(
          Object.entries(o).filter(
            ([k]) => !["run", "zone", "coded", "actual"].includes(k),
          ),
        ),
      };
    }) as RsmPredictResponse["rows"],
    responses: Array.isArray(r["responses"])
      ? (r["responses"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
    confidence: asNumber(r["confidence"]) ?? 0.95,
    df_residual: asNumber(r["df_residual"]) ?? 0,
    extrapolation: {
      count: asNumber(extrapolationRaw["count"]) ?? 0,
      runs: Array.isArray(extrapolationRaw["runs"])
        ? (extrapolationRaw["runs"] as unknown[]).filter(
            (v): v is number => typeof v === "number",
          )
        : [],
      vocabulary:
        extrapolationRaw["vocabulary"] &&
        typeof extrapolationRaw["vocabulary"] === "object"
          ? (extrapolationRaw["vocabulary"] as Record<string, string>)
          : null,
    },
    note: asString(r["note"]) ?? "",
  };
}

/** Narrow the contour answer (the two-factor grid payload). */
function parseRsmContourResponse(
  raw: unknown,
  what: string,
): RsmContourResponse {
  const r = expectObject(raw, what);
  const fxRaw = expectObject(r["factor_x"], `${what} factor_x`);
  const fyRaw = expectObject(r["factor_y"], `${what} factor_y`);
  const axis = (
    a: Record<string, unknown>,
    label: string,
  ): RsmContourResponse["factor_x"] => ({
    name: asString(a["name"]) ?? label,
    unit: asString(a["unit"]),
    coded: Array.isArray(a["coded"])
      ? (a["coded"] as unknown[]).filter(
          (v): v is number => typeof v === "number",
        )
      : [],
    actual: Array.isArray(a["actual"])
      ? (a["actual"] as unknown[]).filter(
          (v): v is number => typeof v === "number",
        )
      : [],
  });
  const zRaw = Array.isArray(r["z"]) ? (r["z"] as unknown[]) : [];
  const fixed: Record<string, { actual: number; coded: number }> = {};
  if (r["fixed"] && typeof r["fixed"] === "object") {
    for (const [name, blockRaw] of Object.entries(
      r["fixed"] as Record<string, unknown>,
    )) {
      const b = expectObject(blockRaw, `${what} fixed ${name}`);
      fixed[name] = {
        actual: asNumber(b["actual"]) ?? 0,
        coded: asNumber(b["coded"]) ?? 0,
      };
    }
  }
  return {
    model_id: asString(r["model_id"]) ?? "",
    factor_x: axis(fxRaw, "x"),
    factor_y: axis(fyRaw, "y"),
    fixed,
    grid_points: asNumber(r["grid_points"]) ?? 0,
    z_name: asString(r["z_name"]) ?? "",
    z: zRaw.map((row, i) =>
      Array.isArray(row)
        ? (row as unknown[]).filter(
            (v): v is number => typeof v === "number",
          )
        : (console.warn(`${what}: z[${i}] is not an array`), []),
    ),
    note: asString(r["note"]) ?? "",
    units_note: asString(r["units_note"]) ?? "",
  };
}

/** Narrow the optimisation answer (the desirability optimum). */
function parseRsmOptimizeResponse(
  raw: unknown,
  what: string,
): RsmOptimizeResponse {
  const r = expectObject(raw, what);
  const candidateRaw = expectObject(r["candidate"], `${what} candidate`);
  const responsesRaw = expectObject(
    candidateRaw["responses"],
    `${what} candidate responses`,
  );
  const responses: Record<string, RsmOptimumResponseBlock> = {};
  for (const [response, blockRaw] of Object.entries(responsesRaw)) {
    const b = expectObject(blockRaw, `${what} responses ${response}`);
    const bounds: Record<string, number> = {};
    if (b["bounds"] && typeof b["bounds"] === "object") {
      for (const [k, v] of Object.entries(b["bounds"] as Record<string, unknown>)) {
        if (typeof v === "number") bounds[k] = v;
      }
    }
    responses[response] = {
      goal: asString(b["goal"]) ?? "",
      prediction: asNumber(b["prediction"]) ?? 0,
      individual_desirability: asNumber(b["individual_desirability"]) ?? 0,
      weight: asNumber(b["weight"]) ?? 1,
      exponent: asNumber(b["exponent"]) ?? 1,
      bounds,
      bounds_source: asString(b["bounds_source"]) ?? "request",
    };
  }
  const sensitivity: RsmOptimizeResponse["sensitivity"] = Array.isArray(
    r["sensitivity"],
  )
    ? (r["sensitivity"] as unknown[]).map((row, i) => {
        const s = expectObject(row, `${what} sensitivity[${i}]`);
        const lowRaw = expectObject(s["low"], `${what} sensitivity low`);
        const highRaw = expectObject(s["high"], `${what} sensitivity high`);
        return {
          factor: asString(s["factor"]) ?? "",
          step_coded: asNumber(s["step_coded"]) ?? 0,
          low: {
            coded: asNumber(lowRaw["coded"]) ?? 0,
            combined_desirability:
              asNumber(lowRaw["combined_desirability"]) ?? 0,
            delta: asNumber(lowRaw["delta"]) ?? 0,
          },
          high: {
            coded: asNumber(highRaw["coded"]) ?? 0,
            combined_desirability:
              asNumber(highRaw["combined_desirability"]) ?? 0,
            delta: asNumber(highRaw["delta"]) ?? 0,
          },
        };
      })
    : [];
  const alternatives: RsmParetoAlternative[] = Array.isArray(
    r["pareto_alternatives"],
  )
    ? (r["pareto_alternatives"] as unknown[]).map((row, i) => {
        const a = expectObject(row, `${what} pareto[${i}]`);
        const predictions: Record<string, number> = {};
        if (a["predictions"] && typeof a["predictions"] === "object") {
          for (const [k, v] of Object.entries(
            a["predictions"] as Record<string, unknown>,
          )) {
            if (typeof v === "number") predictions[k] = v;
          }
        }
        return {
          kind:
            asString(a["kind"]) === "weight_tilted"
              ? "weight_tilted"
              : "individual",
          response: asString(a["response"]) ?? "",
          goal: asString(a["goal"]) ?? "",
          combined_desirability: asNumber(a["combined_desirability"]),
          note: asString(a["note"]) ?? "",
          x_coded: numberRecord(a["x_coded"], `${what} pareto[${i}] x_coded`),
          x_actual: numberRecord(
            a["x_actual"],
            `${what} pareto[${i}] x_actual`,
          ),
          predictions,
        };
      })
    : [];
  const feasibleRaw = r["feasible_grid"];
  const weights: Record<string, number> = {};
  if (r["weights"] && typeof r["weights"] === "object") {
    for (const [k, v] of Object.entries(r["weights"] as Record<string, unknown>)) {
      if (typeof v === "number") weights[k] = v;
    }
  }
  return {
    model_id: asString(r["model_id"]) ?? "",
    candidate: {
      x_coded: numberRecord(candidateRaw["x_coded"], `${what} x_coded`),
      x_actual: numberRecord(candidateRaw["x_actual"], `${what} x_actual`),
      combined_desirability:
        asNumber(candidateRaw["combined_desirability"]) ?? 0,
      responses,
    },
    weights,
    total_weight: asNumber(r["total_weight"]) ?? 0,
    constraints: Array.isArray(r["constraints"])
      ? (r["constraints"] as unknown[]).map((row) => {
          const c = expectObject(row, `${what} constraint`);
          return {
            coefficients: Array.isArray(c["coefficients"])
              ? (c["coefficients"] as unknown[]).filter(
                  (v): v is number => typeof v === "number",
                )
              : [],
            sense: asString(c["sense"]) ?? "<=",
            bound: asNumber(c["bound"]) ?? 0,
          };
        })
      : [],
    sensitivity,
    sensitivity_note: asString(r["sensitivity_note"]) ?? "",
    pareto_alternatives: alternatives,
    pareto_note: asString(r["pareto_note"]) ?? "",
    feasible_grid:
      feasibleRaw && typeof feasibleRaw === "object"
        ? (() => {
            const f = expectObject(feasibleRaw, `${what} feasible_grid`);
            return {
              factor_x: asString(f["factor_x"]) ?? "",
              factor_y: asString(f["factor_y"]) ?? "",
              grid_points: asNumber(f["grid_points"]) ?? 0,
              other_factors_at: numberRecord(
                f["other_factors_at"],
                `${what} other_factors_at`,
              ),
              feasible: Array.isArray(f["feasible"])
                ? (f["feasible"] as unknown[]).map((row) =>
                    Array.isArray(row)
                      ? (row as unknown[]).map((v): boolean => v === true)
                      : [],
                  )
                : [],
              note: asString(f["note"]) ?? "",
            };
          })()
        : null,
    region_note: asString(r["region_note"]) ?? "",
    bounds_source_note: asString(r["bounds_source_note"]) ?? "",
    goals_vocabulary: Array.isArray(r["goals_vocabulary"])
      ? (r["goals_vocabulary"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
  };
}

// -- workflow lane parsers (part13 Phase 6, §8 — the round-323 GUI
// leg; row-validated, never trusted) -------------------------------------

/** Narrow one port of a workflow step. */
function parseWorkflowPortView(
  raw: unknown,
  what: string,
): WorkflowPortView {
  const r = expectObject(raw, what);
  const name = r["name"];
  const kind = r["kind"];
  if (typeof name !== "string" || typeof kind !== "string") {
    throw new Error(`${what}: name/kind are not strings`);
  }
  return {
    name,
    kind,
    required: r["required"] === true,
    description: asString(r["description"]) ?? "",
  };
}

/** Narrow one parameter dial of a workflow step. */
function parseWorkflowParamView(
  raw: unknown,
  what: string,
): WorkflowParamView {
  const r = expectObject(raw, what);
  const name = r["name"];
  if (typeof name !== "string") {
    throw new Error(`${what}: name is not a string`);
  }
  return {
    name,
    default: r["default"],
    description: asString(r["description"]) ?? "",
  };
}

/** Narrow one step of the registry (the typed contract verbatim). */
function parseWorkflowStepView(
  raw: unknown,
  what: string,
): WorkflowStepView {
  const r = expectObject(raw, what);
  const stepId = r["step_id"];
  const title = r["title"];
  const backend = r["backend"];
  if (
    typeof stepId !== "string" ||
    typeof title !== "string" ||
    typeof backend !== "string"
  ) {
    throw new Error(`${what}: step_id/title/backend are not strings`);
  }
  return {
    step_id: stepId,
    title,
    description: asString(r["description"]) ?? "",
    backend,
    workspace: asString(r["workspace"]) ?? "",
    inputs: Array.isArray(r["inputs"])
      ? (r["inputs"] as unknown[]).map((p, i) =>
          parseWorkflowPortView(p, `${what} inputs[${i}]`),
        )
      : [],
    outputs: Array.isArray(r["outputs"])
      ? (r["outputs"] as unknown[]).map((p, i) =>
          parseWorkflowPortView(p, `${what} outputs[${i}]`),
        )
      : [],
    parameters: Array.isArray(r["parameters"])
      ? (r["parameters"] as unknown[]).map((p, i) =>
          parseWorkflowParamView(p, `${what} parameters[${i}]`),
        )
      : [],
    notes: asString(r["notes"]) ?? "",
  };
}

/** Narrow the `GET /steps` answer (the engine block + the registry). */
export function parseWorkflowStepsResponse(
  raw: unknown,
  what: string,
): WorkflowStepsResponse {
  const r = expectObject(raw, what);
  const engineRaw = r["engine"];
  const stepsRaw = r["steps"];
  if (typeof engineRaw !== "object" || engineRaw === null) {
    throw new Error(`${what}: engine block has the wrong shape`);
  }
  const engine = engineRaw as Record<string, unknown>;
  if (typeof engine["layer"] !== "string") {
    throw new Error(`${what}: engine.layer is not a string`);
  }
  if (!Array.isArray(stepsRaw)) {
    throw new Error(`${what}: steps is not an array`);
  }
  return {
    engine: {
      layer: engine["layer"] as string,
      decision: asString(engine["decision"]) ?? "",
      port_kinds: Array.isArray(engine["port_kinds"])
        ? (engine["port_kinds"] as unknown[]).filter(
            (v): v is string => typeof v === "string",
          )
        : [],
      job_integration: asString(engine["job_integration"]) ?? "",
    },
    steps: (stepsRaw as unknown[]).map((s, i) =>
      parseWorkflowStepView(s, `${what} steps[${i}]`),
    ),
  };
}

/** Narrow one saved template (the full serialisable spec included). */
function parseWorkflowTemplateView(
  raw: unknown,
  what: string,
): WorkflowTemplateView {
  const r = expectObject(raw, what);
  const templateId = r["template_id"];
  const name = r["name"];
  if (typeof templateId !== "string" || typeof name !== "string") {
    throw new Error(`${what}: template_id/name are not strings`);
  }
  const nodesRaw = r["nodes"];
  const specRaw = r["spec"];
  return {
    template_id: templateId,
    name,
    description: asString(r["description"]) ?? "",
    node_count: asNumber(r["node_count"]) ?? 0,
    nodes: Array.isArray(nodesRaw)
      ? (nodesRaw as unknown[])
          .map((n, i) => {
            const nr = expectObject(n, `${what} nodes[${i}]`);
            return {
              node_id: asString(nr["node_id"]) ?? "",
              step_id: asString(nr["step_id"]) ?? "",
            };
          })
          .filter((n) => n.node_id !== "" && n.step_id !== "")
      : [],
    metadata:
      typeof r["metadata"] === "object" && r["metadata"] !== null
        ? (r["metadata"] as Record<string, unknown>)
        : {},
    spec:
      typeof specRaw === "object" && specRaw !== null
        ? (specRaw as Record<string, unknown>)
        : {},
  };
}

/** Narrow the `GET /templates` answer. */
export function parseWorkflowTemplatesResponse(
  raw: unknown,
  what: string,
): WorkflowTemplatesResponse {
  const r = expectObject(raw, what);
  const templatesRaw = r["templates"];
  if (!Array.isArray(templatesRaw)) {
    throw new Error(`${what}: templates is not an array`);
  }
  return {
    templates: (templatesRaw as unknown[]).map((t, i) =>
      parseWorkflowTemplateView(t, `${what} templates[${i}]`),
    ),
  };
}

/** Narrow the `POST /validate` answer (the pre-execution gate). */
export function parseWorkflowValidateResponse(
  raw: unknown,
  what: string,
): WorkflowValidateResponse {
  const r = expectObject(raw, what);
  const valid = r["valid"];
  const errorsRaw = r["errors"];
  if (typeof valid !== "boolean" || !Array.isArray(errorsRaw)) {
    throw new Error(`${what}: valid/errors have wrong shapes`);
  }
  const orderRaw = r["topological_order"];
  return {
    valid,
    errors: (errorsRaw as unknown[]).filter(
      (e): e is string => typeof e === "string",
    ),
    topological_order: Array.isArray(orderRaw)
      ? (orderRaw as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : null,
    spec_digest: asString(r["spec_digest"]) ?? "",
    node_count: asNumber(r["node_count"]) ?? 0,
  };
}

/** Narrow one artifact + its honest lineage. */
function parseWorkflowArtifactView(
  raw: unknown,
  what: string,
): WorkflowArtifactView {
  const r = expectObject(raw, what);
  const node = r["node"];
  const port = r["port"];
  const digest = r["digest"];
  if (
    typeof node !== "string" ||
    typeof port !== "string" ||
    typeof digest !== "string"
  ) {
    throw new Error(`${what}: node/port/digest are not strings`);
  }
  const lineageRaw = r["lineage"];
  return {
    node,
    port,
    kind: asString(r["kind"]) ?? "",
    digest,
    digest_kind: asString(r["digest_kind"]) ?? "",
    lineage: Array.isArray(lineageRaw)
      ? (lineageRaw as unknown[])
          .map((l, i) => {
            const lr = expectObject(l, `${what} lineage[${i}]`);
            return {
              node: asString(lr["node"]) ?? "",
              port: asString(lr["port"]) ?? "",
            };
          })
          .filter((l) => l.node !== "" && l.port !== "")
      : [],
  };
}

/** Narrow one node's execution verdict (the partial-failure ledger). */
function parseWorkflowNodeResultView(
  raw: unknown,
  what: string,
): WorkflowNodeResultView {
  const r = expectObject(raw, what);
  const nodeId = r["node_id"];
  const stepId = r["step_id"];
  const state = r["state"];
  if (
    typeof nodeId !== "string" ||
    typeof stepId !== "string" ||
    typeof state !== "string"
  ) {
    throw new Error(`${what}: node_id/step_id/state are not strings`);
  }
  const artifactsRaw = r["artifacts"];
  return {
    node_id: nodeId,
    step_id: stepId,
    state,
    started_at: asString(r["started_at"]),
    duration_ms: asNumber(r["duration_ms"]),
    error: asString(r["error"]),
    skipped_reason: asString(r["skipped_reason"]),
    artifacts: Array.isArray(artifactsRaw)
      ? (artifactsRaw as unknown[]).map((a, i) =>
          parseWorkflowArtifactView(a, `${what} artifacts[${i}]`),
        )
      : [],
  };
}

/** Narrow one run record (the engine's own ledger — the reproducibility
 * block carries the honest determinism scope verbatim). */
export function parseWorkflowRunRecord(
  raw: unknown,
  what: string,
): WorkflowRunRecordView {
  const r = expectObject(raw, what);
  const runId = r["run_id"];
  const state = r["state"];
  const seed = r["seed"];
  if (
    typeof runId !== "string" ||
    typeof state !== "string" ||
    typeof seed !== "number"
  ) {
    throw new Error(`${what}: run_id/state/seed have wrong shapes`);
  }
  const nodesRaw = r["nodes"];
  if (!Array.isArray(nodesRaw)) {
    throw new Error(`${what}: nodes is not an array`);
  }
  const reproducibilityRaw = r["reproducibility"];
  const reproducibility =
    typeof reproducibilityRaw === "object" && reproducibilityRaw !== null
      ? (reproducibilityRaw as Record<string, unknown>)
      : {};
  return {
    run_id: runId,
    workflow_name: asString(r["workflow_name"]) ?? "",
    spec_digest: asString(r["spec_digest"]) ?? "",
    state,
    seed,
    started_at: asString(r["started_at"]) ?? "",
    completed_at: asString(r["completed_at"]),
    reproducibility: {
      toolkit_version: asString(reproducibility["toolkit_version"]) ?? "",
      python: asString(reproducibility["python"]) ?? "",
      implementation: asString(reproducibility["implementation"]) ?? "",
      platform: asString(reproducibility["platform"]) ?? "",
      seed: asNumber(reproducibility["seed"]) ?? seed,
      spec_digest: asString(reproducibility["spec_digest"]) ?? "",
      determinism_note:
        asString(reproducibility["determinism_note"]) ?? "",
    },
    node_counts:
      typeof r["node_counts"] === "object" && r["node_counts"] !== null
        ? (r["node_counts"] as Record<string, number>)
        : {},
    nodes: (nodesRaw as unknown[]).map((n, i) =>
      parseWorkflowNodeResultView(n, `${what} nodes[${i}]`),
    ),
  };
}

/** Narrow the `POST /run` answer — sync (the run record inline) or
 * job mode (the JobManager lane's job id + the honest note). */
export function parseWorkflowRunResponse(
  raw: unknown,
  what: string,
): WorkflowRunResponse {
  const r = expectObject(raw, what);
  const mode = r["mode"];
  if (mode === "sync") {
    if (typeof r["run"] !== "object" || r["run"] === null) {
      throw new Error(`${what}: the sync answer has no run record`);
    }
    return {
      mode: "sync",
      run: parseWorkflowRunRecord(r["run"], `${what} run`),
    };
  }
  if (mode === "job") {
    const jobId = r["job_id"];
    if (typeof jobId !== "string") {
      throw new Error(`${what}: the job answer has no job_id`);
    }
    return {
      mode: "job",
      job_id: jobId,
      job_state: asString(r["job_state"]) ?? "",
      note: asString(r["note"]) ?? "",
    };
  }
  throw new Error(`${what}: unknown mode ${JSON.stringify(mode)}`);
}

/** Narrow the `GET /runs` answer (the bounded in-session ledger). */
export function parseWorkflowRunsResponse(
  raw: unknown,
  what: string,
): WorkflowRunsResponse {
  const r = expectObject(raw, what);
  const runsRaw = r["runs"];
  if (!Array.isArray(runsRaw)) {
    throw new Error(`${what}: runs is not an array`);
  }
  return {
    runs: (runsRaw as unknown[]).map((row, i) => {
      const rr = expectObject(row, `${what} runs[${i}]`);
      const runId = rr["run_id"];
      const state = rr["state"];
      if (typeof runId !== "string" || typeof state !== "string") {
        throw new Error(
          `${what} runs[${i}]: run_id/state are not strings`,
        );
      }
      return {
        run_id: runId,
        workflow_name: asString(rr["workflow_name"]) ?? "",
        state,
        seed: asNumber(rr["seed"]) ?? 0,
        started_at: asString(rr["started_at"]) ?? "",
        completed_at: asString(rr["completed_at"]),
        node_counts:
          typeof rr["node_counts"] === "object" &&
          rr["node_counts"] !== null
            ? (rr["node_counts"] as Record<string, number>)
            : {},
      };
    }),
    note: asString(r["note"]) ?? "",
  };
}

// ---------------------------------------------------------------------------
// ML-1 — the model-evaluation family client (the four conformal routes,
// the roadmap's row 3: the finished backend family with zero consumers
// gets its typed frontend lane)
// ---------------------------------------------------------------------------

/** The conditional-mode Literal the classification route carries. */
export const MODEL_EVALUATION_CONDITIONAL_LITERALS = [
  "split",
  "label",
] as const;

/** `GET /api/v1/model-evaluation/catalog` — the family's own census. */
export interface ModelEvaluationCatalog {
  readonly entry_points: readonly {
    route: string;
    backend: string;
    defaults: Record<string, unknown>;
  }[];
  readonly interval_columns: readonly string[];
  readonly set_columns: readonly string[];
  readonly coverage_columns: readonly string[];
  readonly notes: readonly string[];
}

/** One conformal interval row (input order: prediction, lower, upper). */
export interface ConformalIntervalRow {
  readonly prediction: number;
  readonly lower: number;
  readonly upper: number;
}

/** The empirical block — present only when independent labels rode the
 * request (empirical, never the conformal guarantee). */
export interface ConformalEmpirical {
  readonly n_test: number;
  readonly n_covered: number | null;
  readonly coverage: number | null;
}

/** One Wilson-bounded coverage row (the group / scaffold table). */
export interface CoverageGroupRow {
  readonly group: string | number;
  readonly n_test: number;
  readonly n_covered: number;
  readonly coverage: number | null;
  readonly lower: number | null;
  readonly upper: number | null;
  readonly mean_size: number | null;
}

/** The coverage-diagnostics block — the shared shape of the standalone
 * route's answer, the regression lane's `coverage_diagnostics` and the
 * classification lane's `per_class`. */
export interface CoverageDiagnosticsBlock {
  readonly mode: string;
  readonly confidence: number;
  readonly overall: CoverageGroupRow;
  readonly groups: readonly CoverageGroupRow[];
}

/** `POST /api/v1/model-evaluation/conformal-regression` — the
 * split-conformal interval lane's full answer. */
export interface ConformalRegressionResult {
  readonly alpha: number;
  readonly radius: number;
  readonly n_calibration: number;
  readonly rank: number;
  readonly mean_width: number;
  readonly feature_names: readonly string[];
  readonly intervals: readonly ConformalIntervalRow[] | null;
  readonly empirical: ConformalEmpirical | null;
  readonly coverage_diagnostics: CoverageDiagnosticsBlock | null;
}

/** One prediction-set row (input order; `covered` only with labels). */
export interface ConformalSetRow {
  readonly prediction_set: readonly unknown[];
  readonly set_size: number;
  readonly covered: boolean | null;
}

/** `POST /api/v1/model-evaluation/conformal-classification` — the
 * label-set lane's full answer (marginal split or label-conditional). */
export interface ConformalClassificationResult {
  readonly alpha: number;
  readonly conditional: string;
  readonly classes: readonly unknown[];
  readonly n_calibration: number;
  readonly threshold: number | null;
  readonly class_thresholds: Readonly<Record<string, number>> | null;
  readonly calibration_summary: readonly Record<string, unknown>[] | null;
  readonly sets: readonly ConformalSetRow[] | null;
  readonly empirical: {
    readonly n_test: number;
    readonly n_covered: number | null;
    readonly coverage: number | null;
    readonly mean_set_size: number;
  } | null;
  readonly per_class: CoverageDiagnosticsBlock | null;
}

function _coverageGroupRow(
  raw: unknown,
  what: string,
): CoverageGroupRow {
  const r = expectObject(raw, what);
  const group = r["group"];
  if (
    typeof group !== "string" &&
    typeof group !== "number"
  ) {
    throw new Error(`${what}: group is not a string or number`);
  }
  const nTest = r["n_test"];
  const nCovered = r["n_covered"];
  if (typeof nTest !== "number" || typeof nCovered !== "number") {
    throw new Error(`${what}: n_test/n_covered are not numbers`);
  }
  return {
    group,
    n_test: nTest,
    n_covered: nCovered,
    coverage: asNumber(r["coverage"]),
    lower: asNumber(r["lower"]),
    upper: asNumber(r["upper"]),
    mean_size: asNumber(r["mean_size"]),
  };
}

/** Narrow the shared coverage-diagnostics block (the standalone route,
 * the regression lane's `coverage_diagnostics`, the classification
 * lane's `per_class` — one parser, three call sites). */
export function parseCoverageDiagnosticsBlock(
  raw: unknown,
  what: string,
): CoverageDiagnosticsBlock {
  const r = expectObject(raw, what);
  const mode = r["mode"];
  const confidence = r["confidence"];
  if (typeof mode !== "string" || typeof confidence !== "number") {
    throw new Error(`${what}: mode/confidence are malformed`);
  }
  const overall = _coverageGroupRow(
    r["overall"],
    `${what} overall`,
  );
  const groupsRaw = r["groups"];
  if (!Array.isArray(groupsRaw)) {
    throw new Error(`${what}: groups is not an array`);
  }
  return {
    mode,
    confidence,
    overall,
    groups: groupsRaw.map((row, i) =>
      _coverageGroupRow(row, `${what} groups[${i}]`),
    ),
  };
}

/** Narrow the regression lane's answer field-by-field — a malformed
 * payload fails loudly at the parse boundary, never renders as if it
 * were honest. */
export function parseConformalRegressionResult(
  raw: unknown,
  what: string,
): ConformalRegressionResult {
  const r = expectObject(raw, what);
  const alpha = r["alpha"];
  const radius = r["radius"];
  const nCalibration = r["n_calibration"];
  const rank = r["rank"];
  const meanWidth = r["mean_width"];
  if (
    typeof alpha !== "number" ||
    typeof radius !== "number" ||
    typeof nCalibration !== "number" ||
    typeof rank !== "number" ||
    typeof meanWidth !== "number"
  ) {
    throw new Error(
      `${what}: alpha/radius/n_calibration/rank/mean_width are not numbers`,
    );
  }
  const featureNames = r["feature_names"];
  if (!Array.isArray(featureNames)) {
    throw new Error(`${what}: feature_names is not an array`);
  }
  const intervalsRaw = r["intervals"];
  let intervals: readonly ConformalIntervalRow[] | null = null;
  if (Array.isArray(intervalsRaw)) {
    intervals = intervalsRaw.map((row, i) => {
      const rr = expectObject(row, `${what} intervals[${i}]`);
      const prediction = rr["prediction"];
      const lower = rr["lower"];
      const upper = rr["upper"];
      if (
        typeof prediction !== "number" ||
        typeof lower !== "number" ||
        typeof upper !== "number"
      ) {
        throw new Error(
          `${what} intervals[${i}]: bounds are not numbers`,
        );
      }
      return { prediction, lower, upper };
    });
  }
  const empiricalRaw = r["empirical"];
  let empirical: ConformalEmpirical | null = null;
  if (typeof empiricalRaw === "object" && empiricalRaw !== null) {
    const er = expectObject(empiricalRaw, `${what} empirical`);
    const nTest = er["n_test"];
    if (typeof nTest !== "number") {
      throw new Error(`${what} empirical: n_test is not a number`);
    }
    empirical = {
      n_test: nTest,
      n_covered: asNumber(er["n_covered"]),
      coverage: asNumber(er["coverage"]),
    };
  }
  const diagnosticsRaw = r["coverage_diagnostics"];
  const coverageDiagnostics =
    diagnosticsRaw === null || diagnosticsRaw === undefined
      ? null
      : parseCoverageDiagnosticsBlock(
          diagnosticsRaw,
          `${what} coverage_diagnostics`,
        );
  return {
    alpha,
    radius,
    n_calibration: nCalibration,
    rank,
    mean_width: meanWidth,
    feature_names: featureNames.map((name) => String(name)),
    intervals,
    empirical,
    coverage_diagnostics: coverageDiagnostics,
  };
}

/** Narrow the classification lane's answer field-by-field. */
export function parseConformalClassificationResult(
  raw: unknown,
  what: string,
): ConformalClassificationResult {
  const r = expectObject(raw, what);
  const alpha = r["alpha"];
  const conditional = r["conditional"];
  const nCalibration = r["n_calibration"];
  if (
    typeof alpha !== "number" ||
    typeof conditional !== "string" ||
    typeof nCalibration !== "number"
  ) {
    throw new Error(
      `${what}: alpha/conditional/n_calibration are malformed`,
    );
  }
  if (
    !(MODEL_EVALUATION_CONDITIONAL_LITERALS as readonly string[]).includes(
      conditional,
    )
  ) {
    throw new Error(
      `${what}: conditional is not one of the route's literals`,
    );
  }
  const classesRaw = r["classes"];
  if (!Array.isArray(classesRaw)) {
    throw new Error(`${what}: classes is not an array`);
  }
  const setsRaw = r["sets"];
  let sets: readonly ConformalSetRow[] | null = null;
  if (Array.isArray(setsRaw)) {
    sets = setsRaw.map((row, i) => {
      const rr = expectObject(row, `${what} sets[${i}]`);
      const predictionSet = rr["prediction_set"];
      const setSize = rr["set_size"];
      if (!Array.isArray(predictionSet) || typeof setSize !== "number") {
        throw new Error(
          `${what} sets[${i}]: prediction_set/set_size are malformed`,
        );
      }
      const covered = rr["covered"];
      return {
        prediction_set: predictionSet,
        set_size: setSize,
        covered:
          covered === undefined || covered === null
            ? null
            : asBoolean(covered),
      };
    });
  }
  const empiricalRaw = r["empirical"];
  let empirical: ConformalClassificationResult["empirical"] = null;
  if (typeof empiricalRaw === "object" && empiricalRaw !== null) {
    const er = expectObject(empiricalRaw, `${what} empirical`);
    const nTest = er["n_test"];
    const meanSetSize = er["mean_set_size"];
    if (
      typeof nTest !== "number" ||
      typeof meanSetSize !== "number"
    ) {
      throw new Error(
        `${what} empirical: n_test/mean_set_size are not numbers`,
      );
    }
    empirical = {
      n_test: nTest,
      n_covered: asNumber(er["n_covered"]),
      coverage: asNumber(er["coverage"]),
      mean_set_size: meanSetSize,
    };
  }
  const perClassRaw = r["per_class"];
  const perClass =
    perClassRaw === null || perClassRaw === undefined
      ? null
      : parseCoverageDiagnosticsBlock(perClassRaw, `${what} per_class`);
  const classThresholdsRaw = r["class_thresholds"];
  const classThresholds =
    typeof classThresholdsRaw === "object" && classThresholdsRaw !== null
      ? (Object.fromEntries(
          Object.entries(classThresholdsRaw as Record<string, unknown>).map(
            ([key, value]) => [key, Number(value)],
          ),
        ) as Record<string, number>)
      : null;
  const summaryRaw = r["calibration_summary"];
  const calibrationSummary = Array.isArray(summaryRaw)
    ? (summaryRaw as Record<string, unknown>[])
    : null;
  return {
    alpha,
    conditional,
    classes: classesRaw,
    n_calibration: nCalibration,
    threshold: asNumber(r["threshold"]),
    class_thresholds: classThresholds,
    calibration_summary: calibrationSummary,
    sets,
    empirical,
    per_class: perClass,
  };
}

/** Narrow the catalog answer (the family's own honesty notes ride in). */
export function parseModelEvaluationCatalog(
  raw: unknown,
  what: string,
): ModelEvaluationCatalog {
  const r = expectObject(raw, what);
  const entryRaw = r["entry_points"];
  if (!Array.isArray(entryRaw)) {
    throw new Error(`${what}: entry_points is not an array`);
  }
  return {
    entry_points: entryRaw.map((row, i) => {
      const rr = expectObject(row, `${what} entry_points[${i}]`);
      const route = rr["route"];
      const backend = rr["backend"];
      if (typeof route !== "string" || typeof backend !== "string") {
        throw new Error(
          `${what} entry_points[${i}]: route/backend are not strings`,
        );
      }
      const defaults =
        typeof rr["defaults"] === "object" && rr["defaults"] !== null
          ? (rr["defaults"] as Record<string, unknown>)
          : {};
      return { route, backend, defaults };
    }),
    interval_columns: Array.isArray(r["interval_columns"])
      ? (r["interval_columns"] as unknown[]).map(String)
      : [],
    set_columns: Array.isArray(r["set_columns"])
      ? (r["set_columns"] as unknown[]).map(String)
      : [],
    coverage_columns: Array.isArray(r["coverage_columns"])
      ? (r["coverage_columns"] as unknown[]).map(String)
      : [],
    notes: Array.isArray(r["notes"])
      ? (r["notes"] as unknown[]).filter(
          (note): note is string => typeof note === "string",
        )
      : [],
  };
}

/** The regression lane's request payload (the wire shape the backend's
 * `ConformalRegressionRequest` defines — field-for-field). */
export interface ConformalRegressionParams {
  readonly modelId: string;
  readonly xCal: readonly (readonly number[])[];
  readonly yCal: readonly number[];
  readonly xTest: readonly (readonly number[])[] | null;
  readonly yTest: readonly number[] | null;
  readonly alpha: number;
  readonly groups: readonly (string | number)[] | null;
  readonly smiles: readonly string[] | null;
}

/** The classification lane's request payload. */
export interface ConformalClassificationParams {
  readonly modelId: string;
  readonly xCal: readonly (readonly number[])[];
  readonly yCal: readonly (number | string)[];
  readonly xTest: readonly (readonly number[])[] | null;
  readonly yTest: readonly (number | string)[] | null;
  readonly alpha: number;
  readonly conditional: "split" | "label";
}

/** The standalone coverage-diagnostics request payload. */
export interface CoverageDiagnosticsParams {
  readonly covered: readonly boolean[];
  readonly groups: readonly (string | number)[] | null;
  readonly smiles: readonly string[] | null;
  readonly sizes: readonly number[] | null;
  readonly confidence: number;
}

/** `modelEvaluationApi` — the four typed operations over the finished
 * model-evaluation route family (ML-1: the routes existed with zero
 * consumers; this client is the consumer, every answer narrowed through
 * the strict parsers above). */
export const modelEvaluationApi = {
  /** `GET /api/v1/model-evaluation/catalog` — the family census. */
  async catalog(): Promise<ModelEvaluationCatalog> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/model-evaluation/catalog",
    );
    return parseModelEvaluationCatalog(raw, "model-evaluation catalog");
  },

  /** `POST /api/v1/model-evaluation/conformal-regression` — the
   * split-conformal interval lane (sync by design: one predict pass +
   * a quantile). */
  async conformalRegression(
    params: ConformalRegressionParams,
  ): Promise<ConformalRegressionResult> {
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/model-evaluation/conformal-regression",
      {
        model_id: params.modelId,
        x_cal: params.xCal,
        y_cal: params.yCal,
        x_test: params.xTest,
        y_test: params.yTest,
        alpha: params.alpha,
        groups: params.groups,
        smiles: params.smiles,
      },
    );
    return parseConformalRegressionResult(
      raw,
      "conformal-regression result",
    );
  },

  /** `POST /api/v1/model-evaluation/conformal-classification` — the
   * marginal / label-conditional label-set lane. */
  async conformalClassification(
    params: ConformalClassificationParams,
  ): Promise<ConformalClassificationResult> {
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/model-evaluation/conformal-classification",
      {
        model_id: params.modelId,
        x_cal: params.xCal,
        y_cal: params.yCal,
        x_test: params.xTest,
        y_test: params.yTest,
        alpha: params.alpha,
        conditional: params.conditional,
      },
    );
    return parseConformalClassificationResult(
      raw,
      "conformal-classification result",
    );
  },

  /** `POST /api/v1/model-evaluation/coverage-diagnostics` — the
   * Wilson-bounded coverage report over a covered vector. */
  async coverageDiagnostics(
    params: CoverageDiagnosticsParams,
  ): Promise<CoverageDiagnosticsBlock> {
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/model-evaluation/coverage-diagnostics",
      {
        covered: params.covered,
        groups: params.groups,
        smiles: params.smiles,
        sizes: params.sizes,
        confidence: params.confidence,
      },
    );
    return parseCoverageDiagnosticsBlock(
      raw,
      "coverage-diagnostics result",
    );
  },
};
