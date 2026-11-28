/**
 * The DL client — part11 §11.3's Deep Learning workflow service (the
 * torch-network catalog / train surface over the REAL backend engines).
 *
 * The backend contract (mirrored field-for-field from the route source
 * `routes/dl.py`; the request bodies come from the generated OpenAPI
 * types — the same one-DTO-source rule the Python drift guards in
 * `tests/test_dl_workspace.py` pin for this client):
 *
 * * `GET /api/v1/dl/networks` — the live network catalog
 *   (`_network_catalog`: one entry per tabular torch family —
 *   DeepMLP / ResidualMLP, classifier + regressor — with the defaults
 *   read off the class signatures: one source of truth, never
 *   re-declared), plus `predict_url` + the single-prediction-path
 *   note. Since part13 Phase 5 the catalog also carries the live GNN
 *   vocabulary (`gnn.convs` / `gnn.pools` read off the backend's own
 *   CONV_NAMES / POOL_NAMES) — the graph lane's select controls read
 *   it from here, never re-declared.
 * * `POST /api/v1/dl/train` — `DLTrainRequest` → `{mode: "job",
 *   job: JobView}` (the default lane here); the COMPLETED job's REST
 *   result payload is `{model: TrainedModelView + engine +
 *   predict_url}` — the page relays the authoritative REST result
 *   through the jobs client (§13), never the WS event payload alone.
 *
 * Predictions for a DL-trained model ride the SINGLE session
 * prediction path — `POST /api/v1/ml/predict` through the ml client
 * (`mlApi.predict`): the model registers in the same session model
 * store, so there is exactly one prediction path for every trainer,
 * never two. Checkpoint persistence rides `POST /api/v1/artifacts/model`
 * (the joblib bundle with the full metadata sidecar); the loss-curve
 * figure rides `POST /api/v1/artifacts/figure` — both are posted by
 * the page from the generated types (the workspace row's own
 * surface), parsed here with `parseArtifactView`.
 *
 * The GNN/MPNN lane (part13 Phase 5's frontend leg — the graph
 * models' own request/response contract, mirrored field-for-field
 * from `routes/dl.py`'s GnnTrainRequest / GnnPredictRequest):
 *
 * * `POST /api/v1/dl/gnn-train` — the REAL `GraphClassifier` over
 *   inline graph specs or the backend's synthetic motif fixture,
 *   sync or managed job. The job's progress is HONEST stage-level
 *   telemetry (the torch GNN trainer exposes `loss_history_` after
 *   the fit — no live per-epoch callback exists, so never a
 *   fabricated epoch stream); the result payload carries the model
 *   view (a GRAPH model with its own predict_url) plus
 *   `train_accuracy`, the `loss_history_tail`, the classes and the
 *   conv/pool actually used.
 * * `POST /api/v1/dl/gnn-predict` — the graph models' own
 *   prediction path (labels + optional class probabilities); the
 *   tabular `POST /api/v1/ml/predict` route refuses graph models up
 *   front.
 * * `POST /api/v1/dl/gnn-explain` — DL-2 (the roadmap's row 14): the
 *   graph models' edge attribution over the REAL `gnnxai` engine
 *   (sync by design): the chosen explainer's per-edge scores aligned
 *   with each graph's own edge_index, the fidelity_plus/minus /
 *   sparsity verdicts, and the deterministic circular node layout
 *   (parse with parseGnnExplainResult).
 *
 * The training telemetry (the Monitor stage's live loss/validation
 * curves) arrives as `training.metrics` envelopes on the WS bus —
 * `_training_progress_hook`'s per-epoch dict:
 * `{epoch, epochs, train_loss, val_loss, best_val, stale}` — narrowed
 * here by `parseTrainingMetricsPayload` (§8.1 runtime validation on
 * every untrusted payload, exactly like the REST lanes).
 */

import { api } from "@/services/api/client";
import { parseArtifactView, type ArtifactView } from "@/services/artifacts/client";
import {
  parseTrainedModelView,
  type TrainedModelView,
} from "@/services/ml/client";
import { type JobView, parseJobView } from "@/services/jobs/types";
import type { components } from "@/types/api.d";

type DLTrainRequest = components["schemas"]["DLTrainRequest"];
type FigureRequest = components["schemas"]["FigureRequest"];
type ModelArtifactRequest = components["schemas"]["ModelArtifactRequest"];
type GnnTrainRequest = components["schemas"]["GnnTrainRequest"];
type GnnPredictRequest = components["schemas"]["GnnPredictRequest"];
type GnnGraphSpec = components["schemas"]["GnnGraphSpec"];
type CnnTrainRequest = components["schemas"]["CnnTrainRequest"];
type CnnPredictRequest = components["schemas"]["CnnPredictRequest"];
type AutoencoderTrainRequest =
  components["schemas"]["AutoencoderTrainRequest"];
type AutoencoderEncodeRequest =
  components["schemas"]["AutoencoderEncodeRequest"];

/**
 * The conv tokens the GNN trainer actually accepts — the backend's
 * own `CONV_NAMES` tuple, as the live catalog's `gnn.convs` serves
 * it (one source of truth; the generated DTO's Literal union is the
 * compile-time guard).
 */
export type GnnConv = GnnTrainRequest["conv"];

/**
 * The pool tokens — the backend's own `POOL_NAMES` (the generated
 * DTO's Literal union is the compile-time guard).
 */
export type GnnPool = GnnTrainRequest["pool"];

/**
 * The network token union — the four literals the backend's own
 * pydantic Literal pins (`routes/dl.py`'s `_NETWORKS` keys), derived
 * from the generated DTO (never hand-duplicated).
 */
export type DLNetworkName = DLTrainRequest["network"];

/**
 * The backend's own hyperparameter vocabulary (`routes/dl.py`'s
 * `_HYPERPARAM_SPEC`) — mirrored once beside the route so the page
 * can state the doomed request honestly instead of firing it: an
 * unknown key is the backend's honest 422.
 */
export const DL_HYPERPARAM_KEYS: readonly string[] = [
  "hidden_layers",
  "dropout",
  "activation",
  "batchnorm",
  "epochs",
  "batch_size",
  "learning_rate",
  "weight_decay",
  "patience",
  "val_split",
  "random_state",
  "device",
  // the tabular Transformer family (part18 §5.4 — the route's own spec)
  "d_model",
  "heads",
  "n_blocks",
  "dim_feedforward",
] as const;

/**
 * The activation tokens the torch families actually accept
 * (`neural.py`'s own `_ACTIVATIONS` tuple — mirrored for the page's
 * select control, never a free-text guess).
 */
export const DL_ACTIVATIONS: readonly string[] = [
  "relu",
  "tanh",
  "gelu",
  "sigmoid",
] as const;

/** One `_network_catalog` entry — the catalog's own keys. */
export interface NetworkCatalogEntry {
  readonly network: DLNetworkName;
  readonly class: string;
  readonly module: string;
  readonly task: string;
  readonly engine: string;
  /** The live defaults read off the class signature. */
  readonly defaults: Readonly<Record<string, unknown>>;
  readonly train_url: string;
}

/**
 * The catalog's live GNN vocabulary block — `gnn.convs` /
 * `gnn.pools` read off the backend's own tuples, plus its honest
 * defaults and the graph lane's own route pointers.
 */
export interface NetworkCatalogGnn {
  readonly convs: readonly string[];
  readonly pools: readonly string[];
  readonly defaults: Readonly<Record<string, unknown>>;
  readonly train_url: string;
  readonly predict_url: string;
  readonly note: string;
}

export interface NetworkCatalogResponse {
  readonly networks: readonly NetworkCatalogEntry[];
  readonly predict_url: string;
  readonly note: string;
  /** The live GNN vocabulary (part13 Phase 5) — null if absent. */
  readonly gnn: NetworkCatalogGnn | null;
}

/**
 * The DL-trained model's view — `TrainedModelRecord.view()` plus the
 * route-added `engine` ("torch") and `predict_url` keys.
 */
export interface DLModelView extends TrainedModelView {
  readonly engine: string;
  readonly predict_url: string;
}

/** DL-1 — one trial row of the search report (the engine's own row). */
export interface DLTuneTrial {
  readonly number: number;
  readonly params: Readonly<Record<string, unknown>>;
  /** The negated smaller-is-better objective (the optuna rule). */
  readonly value: number;
}

/**
 * DL-1 — the search report: the ML lane's TuneReport surface (the
 * trial table + the best configuration + the best score) plus the
 * honest extras — the refit's comparable score on the same carve,
 * and the searched block echoed.  `best_score`/`refit_score` are
 * bigger-is-better (accuracy / R², negated back from the study's
 * convention); the trial `value` rides the study's own convention.
 */
export interface DLTuneReport {
  readonly best_params: Readonly<Record<string, unknown>>;
  readonly best_score: number;
  readonly n_trials: number;
  readonly task: string;
  readonly trials: readonly DLTuneTrial[];
  readonly refit_score: number;
  readonly sampler: string;
  readonly seed: number;
  readonly searched: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
}

/** The job-mode submission answer. */
export interface DLJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

/** The sync-mode train answer (`{mode: "sync", model: view}`). */
export interface DLTrainSyncResponse {
  readonly mode: "sync";
  readonly model: DLModelView;
  /** DL-1 — the search report (null on the fixed-config lane). */
  readonly tune: DLTuneReport | null;
}

export type DLTrainResponse = DLJobResponse | DLTrainSyncResponse;

/** The gnn-train job-mode submission answer. */
export interface GnnJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

/** The gnn-train sync answer — the train body spread directly. */
export interface GnnTrainSyncResponse {
  readonly mode: "sync";
  readonly result: GnnTrainResult;
}

export type GnnTrainResponse = GnnJobResponse | GnnTrainSyncResponse;

/** The GNN-trained model view — a GRAPH model with its own path. */
export interface GnnModelView extends DLModelView {
  readonly graph_model: boolean;
}

/** The gnn-train result payload (the COMPLETED job / sync answer). */
export interface GnnTrainResult {
  readonly model: GnnModelView;
  readonly train_accuracy: number;
  /** The last five `loss_history_` entries — the honest tail. */
  readonly loss_history_tail: readonly number[];
  readonly n_graphs: number;
  readonly classes: readonly (number | string)[];
  readonly conv: string;
  readonly pool: string;
}

/** The gnn-predict answer (the graph models' own path). */
export interface GnnPredictResponse {
  readonly model_id: string;
  readonly n_graphs: number;
  readonly predictions: readonly (number | string)[];
  readonly classes: readonly (number | string)[];
  readonly probabilities: readonly number[][] | null;
}

/** DL-2 — one explained graph's edge-attribution row. */
export interface GnnExplainGraphRow {
  readonly index: number;
  readonly n_nodes: number;
  readonly n_edges: number;
  readonly target_class: number;
  readonly method: string;
  readonly scores: readonly number[];
  readonly top_edges: readonly number[];
  readonly fidelity_plus: number;
  readonly fidelity_minus: number;
  readonly sparsity: number;
  readonly extra: Readonly<Record<string, unknown>>;
  readonly edge_index: readonly (readonly number[])[];
  readonly layout: {
    readonly kind: string;
    readonly nodes: readonly (readonly number[])[];
  };
}

/** DL-2 — the explain lane's request options (sync by design). */
export interface GnnExplainOptions {
  readonly modelId: string;
  readonly graphs: readonly GnnGraphRow[];
  readonly method: "gnnexplainer" | "pgexplainer" | "attention" | "subgraphx";
  readonly topK: number;
  readonly seed: number;
  readonly target: number | null;
  readonly labels: readonly number[] | null;
  readonly iters: number | null;
}

/**
 * DL-4 (the roadmap's row 16) — the D-MPNN train options (mirrored
 * from `routes/dl.py`'s MpnnTrainRequest: dataset + SMILES column).
 */
export interface MpnnTrainOptions {
  readonly datasetId: string;
  readonly smilesColumn: string;
  /** optional echo check — must be the dataset's own target */
  readonly targetColumn?: string | null;
  readonly hiddenSize?: number;
  readonly depth?: number;
  readonly ffnHidden?: number | null;
  readonly aggregation?: "norm" | "sum" | "mean";
  readonly aggregationNorm?: number;
  readonly dropout?: number;
  readonly epochs?: number;
  readonly lr?: number;
  readonly batchSize?: number | null;
  readonly valSplit?: number;
  readonly standardize?: boolean;
  readonly seed?: number;
  readonly mode?: "job" | "sync";
}

/** The D-MPNN-trained model view — a MOLECULAR model with its own path. */
export interface MpnnModelView extends DLModelView {
  readonly molecular_model: boolean;
}

/** One rejected row's honest accounting (train and predict lanes). */
export interface MpnnRejectedRow {
  readonly index: number;
  readonly smiles: string | null;
  readonly error: string;
}

/** The mpnn-train result payload (the COMPLETED job / sync answer). */
export interface MpnnTrainResult {
  readonly model: MpnnModelView;
  /** the held-out carve's RMSE — the roadmap's own validation step */
  readonly val_rmse: number;
  readonly val_n: number;
  readonly val_correlation: number | null;
  /** the last five fit-loss entries of both fits — the honest tails */
  readonly train_history_tail: readonly number[];
  readonly refit_history_tail: readonly number[];
  readonly n_rows: number;
  readonly n_rejected: number;
  readonly rejected: readonly MpnnRejectedRow[];
  readonly n_parameters: number;
  readonly held_out_note: string;
}

/** The mpnn-train job-mode submission answer. */
export interface MpnnJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

/** The mpnn-train sync answer — the train body spread directly. */
export interface MpnnTrainSyncResponse {
  readonly mode: "sync";
  readonly result: MpnnTrainResult;
}

export type MpnnTrainResponse = MpnnJobResponse | MpnnTrainSyncResponse;

/** One valid prediction row (aligned with the served index). */
export interface MpnnPredictionRow {
  readonly index: number;
  readonly smiles: string;
  readonly prediction: number;
}

/** The mpnn-predict answer (the molecular models' own path). */
export interface MpnnPredictResponse {
  readonly model_id: string;
  readonly n_valid: number;
  readonly predictions: readonly MpnnPredictionRow[];
  readonly n_rejected: number;
  readonly rejected: readonly MpnnRejectedRow[];
  readonly note: string;
}

/** DL-4 — the predict lane's request options. */
export interface MpnnPredictOptions {
  readonly modelId: string;
  readonly smiles: readonly string[];
}

/** DL-2 — the `POST /api/v1/dl/gnn-explain` answer (sync). */
export interface GnnExplainResult {
  readonly mode: "sync";
  readonly model_id: string;
  readonly method: string;
  readonly top_k: number;
  readonly seed: number;
  readonly graphs: readonly GnnExplainGraphRow[];
  readonly note: string;
}

/**
 * DL-3 (the roadmap's row 15) — the efficiency lab's request body
 * (mirrored from `routes/dl.py`'s DLEfficiencyRequest).
 */
export interface DLEfficiencyOptions {
  readonly modelId: string;
  readonly mode: "quantize_int8" | "quantize_nf4" | "prune";
  /** the prune lane's target sparsity [0, 1) — required on prune */
  readonly level: number | null;
  /** the magnitude engine's own scope vocabulary (prune only) */
  readonly scope: "local" | "global";
  /** the report rides as a downloadable table artifact */
  readonly artifact: boolean;
}

/** One quantize-mode layer row (the engine's own report fields). */
export interface DLEfficiencyQuantRow {
  readonly kind: "quantize";
  readonly layer: string;
  readonly shape: readonly number[];
  readonly nWeights: number;
  readonly method: string;
  readonly bitsPerWeight: number;
  readonly rmse: number;
  readonly relativeError: number;
}

/** One prune-mode layer row (the magnitude engine's own counts). */
export interface DLEfficiencyPruneRow {
  readonly kind: "prune";
  readonly layer: string;
  readonly shape: readonly number[];
  readonly nWeights: number;
  readonly nZeroed: number;
  readonly nKept: number;
  readonly sparsity: number;
}

/** The prune lane's metric block (scored on a pruned COPY). */
export interface DLEfficiencyMetric {
  readonly kind: string;
  readonly before: number | null;
  readonly after: number | null;
  readonly delta: number | null;
  readonly note: string;
}

/** DL-3 — the `POST /api/v1/dl/efficiency` answer (sync). */
export interface DLEfficiencyResult {
  readonly mode: "sync";
  readonly modelId: string;
  readonly network: string;
  readonly efficiencyMode: "quantize_int8" | "quantize_nf4" | "prune";
  readonly nLayers: number;
  readonly nWeights: number;
  readonly layers: readonly (DLEfficiencyQuantRow | DLEfficiencyPruneRow)[];
  readonly artifact: ArtifactView | null;
  // the quantize totals (null on the prune lane)
  readonly method: string | null;
  readonly bitsPerWeight: number | null;
  readonly originalBits: number | null;
  readonly quantizedBits: number | null;
  readonly memorySavings: number | null;
  // the prune totals (null on the quantize lanes)
  readonly targetSparsity: number | null;
  readonly scope: string | null;
  readonly nZeroed: number | null;
  readonly achievedSparsity: number | null;
  readonly metric: DLEfficiencyMetric | null;
  readonly note: string;
}

/**
 * One per-epoch `training.metrics` envelope payload — the REAL
 * `on_epoch` telemetry dict `_training_progress_hook` publishes
 * (the §11.3 Monitor stage's loss/validation curve points).
 */
export interface TrainingMetricsPoint {
  readonly epoch: number;
  readonly epochs: number;
  readonly train_loss: number | null;
  readonly val_loss: number | null;
  readonly best_val: number | null;
  readonly stale: number;
}

/** The `training.progress` envelope payload (progress %, stage). */
export interface TrainingProgressPayload {
  readonly progress: number;
  readonly stage: string;
}

export interface DLTrainOptions {
  readonly datasetId: string;
  /** The catalog row's own network token (the pydantic Literal union). */
  readonly network: DLNetworkName;
  /** Overrides for the signature defaults (null = the defaults). */
  readonly hyperparams: Record<string, unknown> | null;
  readonly mode?: "job" | "sync";
  /**
   * DL-1 (the roadmap's row 13) — the search lane: null/undefined =
   * the fixed-config lane; given, the request carries `tune: true`
   * plus the bounded per-key search block (each entry `choices` or
   * `low`/`high` — `log` optional on the float lane).
   */
  readonly tune?: {
    readonly search: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
    readonly nTrials: number;
    readonly sampler: "tpe" | "random";
    readonly seed: number;
  } | null;
}

/** The tune block's own shape (DL-1) — for pages that build it separately. */
export type DLTrainOptionsTune = NonNullable<DLTrainOptions["tune"]>;

/** One inline graph row — node features + the 2×E endpoints. */
export interface GnnGraphRow {
  readonly x: readonly (readonly number[])[];
  readonly edge_index: readonly (readonly number[])[];
}

/** The inline-graph training source: graphs + one label each. */
export interface GnnInlineSource {
  readonly graphs: readonly GnnGraphRow[];
  /** One class per graph — null in predict mode (no labels there). */
  readonly labels: readonly number[] | null;
}

export interface GnnTrainOptions {
  /** The synthetic motif fixture's row count (10..500). */
  readonly syntheticN?: number;
  readonly seed?: number | null;
  /** Inline graphs + labels (mutually exclusive with syntheticN). */
  readonly inline?: GnnInlineSource | null;
  readonly conv: GnnConv;
  readonly pool: GnnPool;
  readonly hidden: number;
  readonly layers: number;
  readonly lr: number;
  readonly epochs: number;
  readonly weightDecay: number;
  readonly heads: number;
  readonly ratio: number;
  readonly mode?: "job" | "sync";
}

export interface GnnPredictOptions {
  readonly modelId: string;
  readonly graphs: readonly GnnGraphRow[];
  readonly probabilities?: boolean;
}

export interface DLFigureOptions {
  /** The per-epoch points the curve draws (≥ 2 for a line). */
  readonly trainLoss: readonly (number | null)[];
  readonly valLoss: readonly (number | null)[];
  readonly format: "PNG" | "SVG" | "TIFF" | "JPEG";
}

export const dlApi = {
  /**
   * `GET /api/v1/dl/networks` — the live catalog: defaults read off
   * the class signatures, one entry per tabular torch family.
   */
  async networks(): Promise<NetworkCatalogResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/dl/networks",
    );
    const networks = raw["networks"];
    if (!Array.isArray(networks)) {
      return {
        networks: [],
        predict_url:
          typeof raw["predict_url"] === "string" ? raw["predict_url"] : "",
        note: typeof raw["note"] === "string" ? raw["note"] : "",
        gnn: parseGnnCatalogBlock(raw["gnn"]),
      };
    }
    return {
      networks: networks.map((item, i) =>
        parseNetworkCatalogEntry(item, `networks[${i}]`),
      ),
      predict_url:
        typeof raw["predict_url"] === "string" ? raw["predict_url"] : "",
      note: typeof raw["note"] === "string" ? raw["note"] : "",
      gnn: parseGnnCatalogBlock(raw["gnn"]),
    };
  },

  /**
   * `POST /api/v1/dl/train` — one torch network trained by its REAL
   * `fit(X, y, on_epoch=…)` on a registered dataset. Job mode (the
   * default) answers `{mode: "job", job: JobView}`; the COMPLETED
   * result payload is `{model: DLModelView}` (parseDLTrainResult).
   */
  async train(options: DLTrainOptions): Promise<DLTrainResponse> {
    const request: DLTrainRequest = {
      dataset_id: options.datasetId,
      network: options.network,
      hyperparams: options.hyperparams,
      mode: options.mode ?? "job",
      // DL-1 — the search lane's fields always ride the request (the
      // generated DTO marks them required; tune=false keeps the
      // fixed-config lane byte-identical server-side)
      tune: options.tune != null,
      search: options.tune != null
        ? (options.tune.search as unknown as DLTrainRequest["search"])
        : null,
      n_trials: options.tune?.nTrials ?? 12,
      sampler: options.tune?.sampler ?? "tpe",
      seed: options.tune?.seed ?? 0,
    };
    const raw = (await api.post(
      "/api/v1/dl/train",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        model: parseDLModelView(raw["model"], "dl.train(sync)"),
        tune: parseDLTuneReport(raw["tune"], "dl.train(sync) tune"),
      };
    }
    return { mode: "job", job: parseJobView(raw["job"], "dl.train(job)") };
  },

  /**
   * `POST /api/v1/dl/gnn-train` — the REAL `GraphClassifier` over
   * inline graph specs or the backend's own synthetic motif fixture
   * (part13 Phase 5's graph lane). Job mode (the default) answers
   * `{mode: "job", job: JobView}`; the COMPLETED result payload is
   * the full `GnnTrainResult` (parseGnnTrainResult). The job's
   * progress is HONEST stage-level telemetry — the torch GNN trainer
   * exposes no live per-epoch callback, so this lane never promises
   * one (no fabricated epoch stream).
   */
  async gnnTrain(options: GnnTrainOptions): Promise<GnnTrainResponse> {
    const graphs: GnnGraphSpec[] | null = options.inline
      ? options.inline.graphs.map((row) => ({
          x: row.x.map((node) => [...node]),
          edge_index: row.edge_index.map((endpoints) => [...endpoints]),
        }))
      : null;
    const request: GnnTrainRequest = {
      graphs,
      labels:
        graphs !== null && options.inline?.labels
          ? [...options.inline.labels]
          : null,
      synthetic_n:
        graphs === null ? (options.syntheticN ?? undefined) : undefined,
      seed: options.seed ?? undefined,
      conv: options.conv,
      pool: options.pool,
      hidden: options.hidden,
      layers: options.layers,
      lr: options.lr,
      epochs: options.epochs,
      weight_decay: options.weightDecay,
      heads: options.heads,
      ratio: options.ratio,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/dl/gnn-train",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      // the sync answer spreads the SAME train body the job lane
      // relays on completion — narrowed once by the shared parser
      return { mode: "sync", result: parseGnnTrainResult(raw, "dl.gnn-train(sync)") };
    }
    return { mode: "job", job: parseJobView(raw["job"], "dl.gnn-train(job)") };
  },

  /**
   * `POST /api/v1/dl/gnn-predict` — the graph models' own prediction
   * path: label sets (and optional class probabilities) for inline
   * graph rows from a session-trained GNN. The tabular
   * `POST /api/v1/ml/predict` route refuses graph models up front —
   * one path per model class, never two.
   */
  async gnnPredict(
    options: GnnPredictOptions,
  ): Promise<GnnPredictResponse> {
    const request: GnnPredictRequest = {
      model_id: options.modelId,
      graphs: options.graphs.map((row) => ({
        x: row.x.map((node) => [...node]),
        edge_index: row.edge_index.map((endpoints) => [...endpoints]),
      })),
      probabilities: options.probabilities ?? false,
    };
    const raw = (await api.post(
      "/api/v1/dl/gnn-predict",
      request,
    )) as Record<string, unknown>;
    return parseGnnPredictResult(raw, "dl.gnn-predict");
  },

  /**
   * `POST /api/v1/dl/gnn-explain` — DL-2 (the roadmap's row 14):
   * the graph models' edge attribution over the REAL gnnxai engine
   * (the learned mask / the Shapley search / the GAT attention /
   * the parametric scorer trained over the request's own labels).
   * Sync by design (masked forward passes, seconds per graph).
   */
  async gnnExplain(options: GnnExplainOptions): Promise<GnnExplainResult> {
    const request: Record<string, unknown> = {
      model_id: options.modelId,
      graphs: options.graphs.map((row) => ({
        x: row.x.map((node) => [...node]),
        edge_index: row.edge_index.map((endpoints) => [...endpoints]),
      })),
      method: options.method,
      top_k: options.topK,
      seed: options.seed,
      target: options.target ?? undefined,
      labels: options.labels ?? undefined,
      iters: options.iters ?? undefined,
    };
    const raw = (await api.post(
      "/api/v1/dl/gnn-explain",
      request,
    )) as Record<string, unknown>;
    return parseGnnExplainResult(raw, "dl.gnn-explain");
  },

  /**
   * `POST /api/v1/dl/efficiency` — DL-3 (the roadmap's row 15): the
   * weight-efficiency lab over the REAL quantization/pruning engines
   * (int8 absmax / the NF4 block pack / the magnitude masks), the
   * report registered as a downloadable table artifact. Sync by
   * design (one matrix sweep per mode); the persisted estimator is
   * never mutated (the prune lane scores a pruned COPY).
   */
  async efficiency(options: DLEfficiencyOptions): Promise<DLEfficiencyResult> {
    const request: Record<string, unknown> = {
      model_id: options.modelId,
      mode: options.mode,
      level: options.level ?? undefined,
      scope: options.scope,
      artifact: options.artifact,
    };
    const raw = (await api.post(
      "/api/v1/dl/efficiency",
      request,
    )) as Record<string, unknown>;
    return parseDLEfficiencyResult(raw, "dl.efficiency");
  },

  /**
   * `POST /api/v1/dl/mpnn-train` — DL-4 (the roadmap's row 16): the
   * REAL numpy D-MPNN (chemprop's directed message passing) over a
   * registered dataset's own SMILES column: the column re-read
   * server-side, the honest rejected-rows accounting, the seeded
   * held-out carve the roadmap's own validation step names, and the
   * shipped model as the FULL-corpus refit (job or sync).
   */
  async mpnnTrain(options: MpnnTrainOptions): Promise<MpnnTrainResponse> {
    const request: Record<string, unknown> = {
      dataset_id: options.datasetId,
      smiles_column: options.smilesColumn,
      target_column: options.targetColumn ?? undefined,
      hidden_size: options.hiddenSize ?? 64,
      depth: options.depth ?? 3,
      ffn_hidden: options.ffnHidden ?? undefined,
      aggregation: options.aggregation ?? "norm",
      aggregation_norm: options.aggregationNorm ?? 100.0,
      dropout: options.dropout ?? 0.0,
      epochs: options.epochs ?? 200,
      lr: options.lr ?? 1e-3,
      batch_size: options.batchSize ?? undefined,
      val_split: options.valSplit ?? 0.2,
      standardize: options.standardize ?? true,
      seed: options.seed ?? 0,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/dl/mpnn-train",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        result: parseMpnnTrainResult(raw, "dl.mpnn-train(sync)"),
      };
    }
    return {
      mode: "job",
      job: parseJobView(raw["job"], "dl.mpnn-train(job)"),
    };
  },

  /**
   * `POST /api/v1/dl/mpnn-predict` — the molecular models' own
   * prediction path over raw SMILES (the predictions in the original
   * target units, the rejected rows riding their own accounting).
   */
  async mpnnPredict(
    options: MpnnPredictOptions,
  ): Promise<MpnnPredictResponse> {
    const request: Record<string, unknown> = {
      model_id: options.modelId,
      smiles: [...options.smiles],
    };
    const raw = (await api.post(
      "/api/v1/dl/mpnn-predict",
      request,
    )) as Record<string, unknown>;
    return parseMpnnPredictResult(raw, "dl.mpnn-predict");
  },

  /**
   * `POST /api/v1/artifacts/figure` — the loss/validation curve as a
   * REAL figure artifact: the per-epoch train/val series as one
   * scatter figure through the §17 four-format export stack. The
   * page owns this lane (the workspace row's own surface), so the
   * request body rides the generated `FigureRequest` type.
   */
  async lossCurveFigure(options: DLFigureOptions): Promise<ArtifactView> {
    if (options.trainLoss.length !== options.valLoss.length) {
      throw new Error(
        "the train and validation loss series must carry one point per epoch",
      );
    }
    const epochs = options.trainLoss.length;
    if (epochs < 2) {
      throw new Error(
        "a loss curve needs at least two epochs of telemetry",
      );
    }
    const points: { x: number; train: number | null; val: number | null }[] =
      [];
    for (let i = 0; i < epochs; i++) {
      points.push({
        x: i + 1,
        train: options.trainLoss[i] ?? null,
        val: options.valLoss[i] ?? null,
      });
    }
    // two traces in one figure: x = the epoch axis, y = the stacked
    // series, color = the series label (the backend's own scatter
    // contract: parallel x/y/color arrays)
    const x: (number | string)[] = [];
    const y: number[] = [];
    const color: string[] = [];
    for (const point of points) {
      if (point.train !== null) {
        x.push(point.x);
        y.push(point.train);
        color.push("train_loss");
      }
      if (point.val !== null) {
        x.push(point.x);
        y.push(point.val);
        color.push("val_loss");
      }
    }
    const request: FigureRequest = {
      kind: "scatter",
      x,
      y,
      color,
      title: "Deep Learning — train/validation loss per epoch",
      format: options.format,
      dpi: 150,
      filename: "dl-loss-curve",
    };
    const raw = (await api.post(
      "/api/v1/artifacts/figure",
      request,
    )) as Record<string, unknown>;
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the figure lane answered without an artifact");
    }
    return parseArtifactView(artifact, "loss curve figure");
  },

  /**
   * `POST /api/v1/artifacts/model` — the §11.3 checkpoint
   * persistence: the trained estimator saved as the standard joblib
   * bundle with the full provenance metadata sidecar (features,
   * params, metrics, versions — a model loaded two years later
   * still knows what it is).
   */
  async saveCheckpoint(
    modelId: string,
    metrics?: Record<string, unknown> | null,
  ): Promise<ArtifactView> {
    const request: ModelArtifactRequest = {
      model_id: modelId,
      metrics: metrics ?? null,
      filename: `dl-${modelId}-checkpoint`,
    };
    const raw = (await api.post(
      "/api/v1/artifacts/model",
      request,
    )) as Record<string, unknown>;
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the model lane answered without an artifact");
    }
    return parseArtifactView(artifact, "checkpoint artifact");
  },

  /**
   * `POST /api/v1/dl/cnn-train` — the REAL SequenceCNN (part18 §5.2)
   * over inline 3-D sequences + labels or the backend's own
   * deterministic motif fixture. Job mode (the default) answers
   * `{mode: "job", job: JobView}`; the COMPLETED result payload is
   * the `CnnTrainResult`. The trainer's `on_epoch` drives the same
   * per-epoch `training.metrics` stream as the tabular lane (real
   * telemetry — the estimator's own callback, never fabricated).
   */
  async cnnTrain(options: CnnTrainOptions): Promise<CnnTrainResponse> {
    const sequences: CnnTrainRequest["sequences"] = options.inline
      ? options.inline.sequences.map((steps) => ({
          steps: steps.map((step) => [...step]),
        }))
      : null;
    const labels: CnnTrainRequest["labels"] =
      sequences !== null && options.inline?.labels
        ? [...options.inline.labels]
        : null;
    const request: CnnTrainRequest = {
      sequences,
      labels,
      synthetic_n:
        sequences === null ? (options.syntheticN ?? undefined) : undefined,
      seed: options.seed ?? undefined,
      task: options.task,
      channels: options.channels,
      kernel: options.kernel,
      n_layers: options.nLayers,
      epochs: options.epochs,
      batch_size: options.batchSize,
      learning_rate: options.learningRate,
      weight_decay: options.weightDecay,
      patience: options.patience,
      val_split: options.valSplit,
      random_state: options.randomState,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/dl/cnn-train",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        result: parseCnnTrainResult(raw, "dl.cnn-train(sync)"),
      };
    }
    return {
      mode: "job",
      job: parseJobView(raw["job"], "dl.cnn-train(job)"),
    };
  },

  /**
   * `POST /api/v1/dl/cnn-predict` — the sequence models' OWN
   * prediction path: label sets (and optional class probabilities)
   * for inline sequence rows from a session-trained CNN. The
   * tabular `POST /api/v1/ml/predict` route refuses sequence models
   * up front — one path per model class, never two.
   */
  async cnnPredict(
    options: CnnPredictOptions,
  ): Promise<CnnPredictResponse> {
    const request: CnnPredictRequest = {
      model_id: options.modelId,
      sequences: options.sequences.map((steps) => ({
        steps: steps.map((step) => [...step]),
      })),
      probabilities: options.probabilities ?? false,
    };
    const raw = (await api.post(
      "/api/v1/dl/cnn-predict",
      request,
    )) as Record<string, unknown>;
    return parseCnnPredictResult(raw, "dl.cnn-predict");
  },

  /**
   * `POST /api/v1/dl/autoencoder-train` — the REAL
   * `TabularAutoencoder` (part18 §5.5) over a registered dataset's
   * feature matrix: representation/reconstruction/anomaly lane,
   * unsupervised (the model registers with task `unsupervised` and
   * its own encode URL — never the supervised predict path).
   */
  async autoencoderTrain(
    options: AutoencoderTrainOptions,
  ): Promise<AutoencoderTrainResponse> {
    const request: AutoencoderTrainRequest = {
      dataset_id: options.datasetId,
      hidden_layers: [...options.hiddenLayers],
      latent_dim: options.latentDim,
      epochs: options.epochs,
      batch_size: options.batchSize,
      learning_rate: options.learningRate,
      weight_decay: options.weightDecay,
      patience: options.patience,
      val_split: options.valSplit,
      random_state: options.randomState,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/dl/autoencoder-train",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        result: parseAutoencoderTrainResult(
          raw,
          "dl.autoencoder-train(sync)",
        ),
      };
    }
    return {
      mode: "job",
      job: parseJobView(raw["job"], "dl.autoencoder-train(job)"),
    };
  },

  /**
   * `POST /api/v1/dl/autoencoder-encode` — the fitted encoder's
   * lane: the latent coordinates, the per-row reconstruction errors
   * and the honest outlier ranking (a heuristic, never ground
   * truth — the note rides every response).
   */
  async autoencoderEncode(
    options: AutoencoderEncodeOptions,
  ): Promise<AutoencoderEncodeResponse> {
    const request: AutoencoderEncodeRequest = {
      model_id: options.modelId,
      dataset_id: options.datasetId ?? undefined,
      rows: options.rows ? [...options.rows] : undefined,
      top_k: options.topK,
    };
    const raw = (await api.post(
      "/api/v1/dl/autoencoder-encode",
      request,
    )) as Record<string, unknown>;
    return parseAutoencoderEncodeResult(raw, "dl.autoencoder-encode");
  },
};

// -- validation helpers (§8.1) ------------------------------------------

function expectObject(raw: unknown, what: string): Record<string, unknown> {
  if (typeof raw !== "object" || raw === null) {
    throw new Error(`${what}: expected an object`);
  }
  return raw as Record<string, unknown>;
}

function asNumber(raw: unknown): number | null {
  return typeof raw === "number" ? raw : null;
}

/** Narrow an untrusted payload to one catalog entry. */
function parseNetworkCatalogEntry(
  raw: unknown,
  what: string,
): NetworkCatalogEntry {
  const r = expectObject(raw, what);
  const network = r["network"];
  const task = r["task"];
  if (typeof network !== "string" || typeof task !== "string") {
    throw new Error(`${what}: network/task are not strings`);
  }
  const defaults = r["defaults"];
  return {
    // the backend's own catalog rows are exactly the four tokens
    // the pydantic Literal pins — a drifted row would surface as
    // the route's honest 422 at train time, never silently accepted
    network: network as DLNetworkName,
    class: typeof r["class"] === "string" ? r["class"] : "",
    module: typeof r["module"] === "string" ? r["module"] : "",
    task,
    engine: typeof r["engine"] === "string" ? r["engine"] : "",
    defaults:
      typeof defaults === "object" && defaults !== null
        ? (defaults as Record<string, unknown>)
        : {},
    train_url: typeof r["train_url"] === "string" ? r["train_url"] : "",
  };
}

/** Narrow the DL model view (the trained view + engine + predict_url). */
function parseDLModelView(raw: unknown, what: string): DLModelView {
  const r = expectObject(raw, what);
  return {
    ...parseTrainedModelView(raw, what),
    engine: typeof r["engine"] === "string" ? r["engine"] : "",
    predict_url: typeof r["predict_url"] === "string" ? r["predict_url"] : "",
  };
}

/**
 * Narrow the train job's result payload: `{model: DLModelView}`
 * (the authoritative REST relay — `parseTrainedModelView` keeps the
 * base fields honest, the engine keys surface as they arrive).
 */
export function parseDLTrainResult(
  raw: unknown,
  what: string,
): DLModelView {
  const r = expectObject(raw, what);
  return parseDLModelView(r["model"], `${what} model`);
}

/**
 * DL-1 — narrow the search report. A missing trial row value throws
 * (a numberless verdict would fabricate a ranking); the fixed-config
 * lane passes null and gets null back — the absence stays honest.
 */
export function parseDLTuneReport(
  raw: unknown,
  what: string,
): DLTuneReport | null {
  if (raw === null || raw === undefined) {
    return null;
  }
  const r = expectObject(raw, what);
  const bestScore = asNumber(r["best_score"]);
  const nTrials = r["n_trials"];
  const refitScore = asNumber(r["refit_score"]);
  const task = r["task"];
  const sampler = r["sampler"];
  const seed = asNumber(r["seed"]);
  if (
    bestScore === null ||
    typeof nTrials !== "number" ||
    refitScore === null ||
    typeof task !== "string" ||
    typeof sampler !== "string" ||
    seed === null
  ) {
    throw new Error(
      `${what}: best_score/n_trials/refit_score/task/sampler/seed ` +
        "are not the report's own types",
    );
  }
  const trialsRaw = r["trials"];
  if (!Array.isArray(trialsRaw)) {
    throw new Error(`${what}: trials is not an array`);
  }
  const trials: DLTuneTrial[] = [];
  for (const trialRaw of trialsRaw as unknown[]) {
    const row = expectObject(trialRaw, `${what} trial row`);
    const value = asNumber(row["value"]);
    const number = asNumber(row["number"]);
    if (value === null || number === null) {
      throw new Error(
        `${what}: a trial row lost its number/value pair (the ` +
          "ranking would be fabricated)",
      );
    }
    const params = row["params"];
    if (typeof params !== "object" || params === null) {
      throw new Error(`${what}: a trial row lost its params`);
    }
    trials.push({
      number,
      params: params as Record<string, unknown>,
      value,
    });
  }
  const bestParams = r["best_params"];
  if (typeof bestParams !== "object" || bestParams === null) {
    throw new Error(`${what}: best_params is not an object`);
  }
  const searched = r["searched"];
  if (typeof searched !== "object" || searched === null) {
    throw new Error(`${what}: searched is not an object`);
  }
  return {
    best_params: bestParams as Record<string, unknown>,
    best_score: bestScore,
    n_trials: nTrials,
    task,
    trials,
    refit_score: refitScore,
    sampler,
    seed,
    searched: searched as Record<string, Record<string, unknown>>,
  };
}

/**
 * DL-1 — narrow the tune lane's full result (the sync answer's own
 * halves, or the job COMPLETED result): `{model, tune}` with the
 * tune block null on the fixed-config lane.
 */
export function parseDLTuneResult(
  raw: unknown,
  what: string,
): { model: DLModelView; tune: DLTuneReport | null } {
  const r = expectObject(raw, what);
  return {
    model: parseDLModelView(r["model"], `${what} model`),
    tune: parseDLTuneReport(r["tune"], `${what} tune`),
  };
}

/**
 * Narrow one `training.metrics` envelope payload to the per-epoch
 * telemetry point (the Monitor stage's curve input). The payload is
 * untrusted wire data — every key surfaces as it arrives, a missing
 * loss stays null rather than being invented.
 */
export function parseTrainingMetricsPayload(
  raw: unknown,
  what: string,
): TrainingMetricsPoint {
  const r = expectObject(raw, what);
  const epoch = r["epoch"];
  const epochs = r["epochs"];
  if (typeof epoch !== "number" || typeof epochs !== "number") {
    throw new Error(`${what}: epoch/epochs are not numbers`);
  }
  return {
    epoch,
    epochs,
    train_loss: asNumber(r["train_loss"]),
    val_loss: asNumber(r["val_loss"]),
    best_val: asNumber(r["best_val"]),
    stale: typeof r["stale"] === "number" ? r["stale"] : 0,
  };
}

/**
 * Narrow one `training.progress` envelope payload (progress % +
 * the epoch stage string).
 */
export function parseTrainingProgressPayload(
  raw: unknown,
  what: string,
): TrainingProgressPayload {
  const r = expectObject(raw, what);
  const progress = r["progress"];
  if (typeof progress !== "number") {
    throw new Error(`${what}: progress is not a number`);
  }
  return {
    progress,
    stage: typeof r["stage"] === "string" ? r["stage"] : "",
  };
}

// -- the GNN lane's parsers (part13 Phase 5, §8.1 runtime narrowing) --

/** Narrow the catalog's live GNN vocabulary block (null if absent). */
function parseGnnCatalogBlock(raw: unknown): NetworkCatalogGnn | null {
  if (typeof raw !== "object" || raw === null) {
    return null;
  }
  const r = raw as Record<string, unknown>;
  const convs = r["convs"];
  const pools = r["pools"];
  if (!Array.isArray(convs) || !Array.isArray(pools)) {
    return null;
  }
  const asStrings = (items: unknown[]): string[] =>
    items.every((token) => typeof token === "string")
      ? (items as string[])
      : [];
  const defaults = r["defaults"];
  return {
    convs: asStrings(convs),
    pools: asStrings(pools),
    defaults:
      typeof defaults === "object" && defaults !== null
        ? (defaults as Record<string, unknown>)
        : {},
    train_url: typeof r["train_url"] === "string" ? r["train_url"] : "",
    predict_url: typeof r["predict_url"] === "string" ? r["predict_url"] : "",
    note: typeof r["note"] === "string" ? r["note"] : "",
  };
}

/** One class label as the backend serves it (number or string). */
function asLabel(raw: unknown): number | string {
  return typeof raw === "number" || typeof raw === "string" ? raw : String(raw);
}

/**
 * Narrow the gnn-train result payload (the COMPLETED job's REST
 * relay and the sync answer's own body — the SAME keys): the
 * graph model view + `train_accuracy` + the honest `loss_history_tail`
 * + the classes + the conv/pool actually used. A missing accuracy
 * or tail stays what it is — never invented.
 */
export function parseGnnTrainResult(
  raw: unknown,
  what: string,
): GnnTrainResult {
  const r = expectObject(raw, what);
  const model = parseGnnModelView(r["model"], `${what} model`);
  const trainAccuracy = r["train_accuracy"];
  const nGraphs = r["n_graphs"];
  if (typeof trainAccuracy !== "number" || typeof nGraphs !== "number") {
    throw new Error(`${what}: train_accuracy/n_graphs are not numbers`);
  }
  const tail = r["loss_history_tail"];
  const classes = r["classes"];
  return {
    model,
    train_accuracy: trainAccuracy,
    loss_history_tail: Array.isArray(tail)
      ? tail.filter((v): v is number => typeof v === "number")
      : [],
    n_graphs: nGraphs,
    classes: Array.isArray(classes) ? classes.map(asLabel) : [],
    conv: typeof r["conv"] === "string" ? r["conv"] : "",
    pool: typeof r["pool"] === "string" ? r["pool"] : "",
  };
}

/** Narrow the GNN model view (the trained view + the graph keys). */
function parseGnnModelView(raw: unknown, what: string): GnnModelView {
  const r = expectObject(raw, what);
  return {
    ...parseDLModelView(raw, what),
    graph_model: r["graph_model"] === true,
  };
}

/**
 * DL-2 — narrow the gnn-explain answer: every per-graph row must
 * carry its scores aligned with the served edge_index (a row whose
 * counts disagree throws — the scatter would fabricate geometry),
 * and the fidelity verdicts stay floats (never nulls — an absent
 * verdict is never an implied zero).
 */
export function parseGnnExplainResult(
  raw: unknown,
  what: string,
): GnnExplainResult {
  const r = expectObject(raw, what);
  const modelId = r["model_id"];
  const method = r["method"];
  const topK = r["top_k"];
  const seed = r["seed"];
  const note = r["note"];
  if (
    typeof modelId !== "string" ||
    typeof method !== "string" ||
    typeof topK !== "number" ||
    typeof seed !== "number" ||
    typeof note !== "string"
  ) {
    throw new Error(
      `${what}: model_id/method/top_k/seed/note are not the answer's own types`,
    );
  }
  const graphsRaw = r["graphs"];
  if (!Array.isArray(graphsRaw)) {
    throw new Error(`${what}: graphs is not an array`);
  }
  const graphs: GnnExplainGraphRow[] = [];
  for (const rowRaw of graphsRaw as unknown[]) {
    const row = expectObject(rowRaw, `${what} graph row`);
    const index = asNumber(row["index"]);
    const nNodes = asNumber(row["n_nodes"]);
    const nEdges = asNumber(row["n_edges"]);
    const targetClass = asNumber(row["target_class"]);
    const rowMethod = row["method"];
    const fidelityPlus = asNumber(row["fidelity_plus"]);
    const fidelityMinus = asNumber(row["fidelity_minus"]);
    const sparsityValue = asNumber(row["sparsity"]);
    if (
      index === null || nNodes === null || nEdges === null ||
      targetClass === null || typeof rowMethod !== "string" ||
      fidelityPlus === null || fidelityMinus === null ||
      sparsityValue === null
    ) {
      throw new Error(
        `${what}: a graph row lost its counts/verdicts (an absent ` +
          "fidelity is never an implied zero)",
      );
    }
    const scores = row["scores"];
    const topEdges = row["top_edges"];
    const edgeIndex = row["edge_index"];
    if (
      !Array.isArray(scores) || !Array.isArray(topEdges) ||
      !Array.isArray(edgeIndex)
    ) {
      throw new Error(
        `${what}: a graph row lost its scores/top_edges/edge_index`,
      );
    }
    if (scores.length !== edgeIndex.length || scores.length !== nEdges) {
      throw new Error(
        `${what}: a graph row's scores (${scores.length}) disagree ` +
          `with its edge_index (${edgeIndex.length}) — the scatter ` +
          "geometry would be fabricated",
      );
    }
    const layoutRaw = row["layout"];
    const layoutRow = expectObject(layoutRaw, `${what} layout`);
    const layoutKind = layoutRow["kind"];
    const layoutNodes = layoutRow["nodes"];
    if (
      typeof layoutKind !== "string" ||
      !Array.isArray(layoutNodes) ||
      layoutNodes.length !== nNodes
    ) {
      throw new Error(
        `${what}: a graph row's layout lost its node positions`,
      );
    }
    const extra = row["extra"];
    graphs.push({
      index,
      n_nodes: nNodes,
      n_edges: nEdges,
      target_class: targetClass,
      method: rowMethod,
      scores: scores.filter((s): s is number => typeof s === "number"),
      top_edges: topEdges.filter(
        (e): e is number => typeof e === "number",
      ),
      fidelity_plus: fidelityPlus,
      fidelity_minus: fidelityMinus,
      sparsity: sparsityValue,
      extra:
        typeof extra === "object" && extra !== null
          ? (extra as Record<string, unknown>)
          : {},
      edge_index: edgeIndex as readonly (readonly number[])[],
      layout: {
        kind: layoutKind,
        nodes: layoutNodes as readonly (readonly number[])[],
      },
    });
  }
  return {
    mode: "sync",
    model_id: modelId,
    method,
    top_k: topK,
    seed,
    graphs,
    note,
  };
}

/**
 * Narrow the DL-3 efficiency answer: the mode-specific totals and
 * per-layer rows, with the quantize/prune lanes' own fields never
 * implied (an absent total is null, never zero — the honest parser
 * rule every lane here follows).
 */
export function parseDLEfficiencyResult(
  raw: unknown,
  what: string,
): DLEfficiencyResult {
  const r = expectObject(raw, what);
  const modelId = r["model_id"];
  const network = r["network"];
  const efficiencyModeRaw = r["efficiency_mode"];
  const nLayers = asNumber(r["n_layers"]);
  const nWeights = asNumber(r["n_weights"]);
  const note = r["note"];
  if (
    typeof modelId !== "string" ||
    typeof network !== "string" ||
    typeof efficiencyModeRaw !== "string" ||
    nLayers === null ||
    nWeights === null ||
    typeof note !== "string"
  ) {
    throw new Error(
      `${what}: model_id/network/efficiency_mode/n_weights/note are not ` +
        "the answer's own types",
    );
  }
  const efficiencyMode = efficiencyModeRaw as DLEfficiencyResult["efficiencyMode"];
  const layersRaw = r["layers"];
  if (!Array.isArray(layersRaw) || layersRaw.length !== nLayers) {
    throw new Error(
      `${what}: layers is not the answer's own per-layer table`,
    );
  }
  const layers: (DLEfficiencyQuantRow | DLEfficiencyPruneRow)[] = [];
  for (const rowRaw of layersRaw as unknown[]) {
    const row = expectObject(rowRaw, `${what} layer row`);
    const layer = row["layer"];
    const shapeRaw = row["shape"];
    const rowN = asNumber(row["n_weights"]);
    if (
      typeof layer !== "string" ||
      !Array.isArray(shapeRaw) ||
      rowN === null
    ) {
      throw new Error(`${what}: a layer row lost its identity/shape/count`);
    }
    const shape = shapeRaw.filter(
      (d): d is number => typeof d === "number",
    );
    if (efficiencyMode === "prune") {
      const nZeroed = asNumber(row["n_zeroed"]);
      const nKept = asNumber(row["n_kept"]);
      const sparsity = asNumber(row["sparsity"]);
      if (nZeroed === null || nKept === null || sparsity === null) {
        throw new Error(
          `${what}: a prune row lost its counts (an absent count is ` +
            "never an implied zero)",
        );
      }
      layers.push({
        kind: "prune",
        layer,
        shape,
        nWeights: rowN,
        nZeroed,
        nKept,
        sparsity,
      });
    } else {
      const method = row["method"];
      const bitsPerWeight = asNumber(row["bits_per_weight"]);
      const rmse = asNumber(row["rmse"]);
      const relativeError = asNumber(row["relative_error"]);
      if (
        typeof method !== "string" ||
        bitsPerWeight === null ||
        rmse === null ||
        relativeError === null
      ) {
        throw new Error(
          `${what}: a quantize row lost the engine's own report fields`,
        );
      }
      layers.push({
        kind: "quantize",
        layer,
        shape,
        nWeights: rowN,
        method,
        bitsPerWeight,
        rmse,
        relativeError,
      });
    }
  }
  // the mode-specific totals: null on the lanes they do not ride
  const quantTotals =
    efficiencyMode === "prune"
      ? null
      : (() => {
          const method = r["method"];
          const bitsPerWeight = asNumber(r["bits_per_weight"]);
          const originalBits = asNumber(r["original_bits"]);
          const quantizedBits = asNumber(r["quantized_bits"]);
          const memorySavings = asNumber(r["memory_savings"]);
          if (
            typeof method !== "string" ||
            bitsPerWeight === null ||
            originalBits === null ||
            quantizedBits === null ||
            memorySavings === null
          ) {
            throw new Error(
              `${what}: the quantize totals lost the engine's own fields`,
            );
          }
          return {
            method,
            bitsPerWeight,
            originalBits,
            quantizedBits,
            memorySavings,
          };
        })();
  const pruneTotals =
    efficiencyMode === "prune"
      ? (() => {
          const targetSparsity = asNumber(r["target_sparsity"]);
          const scope = r["scope"];
          const nZeroed = asNumber(r["n_zeroed"]);
          const achievedSparsity = asNumber(r["achieved_sparsity"]);
          if (
            targetSparsity === null ||
            typeof scope !== "string" ||
            nZeroed === null ||
            achievedSparsity === null
          ) {
            throw new Error(
              `${what}: the prune totals lost the engine's own counts`,
            );
          }
          return { targetSparsity, scope, nZeroed, achievedSparsity };
        })()
      : null;
  // the metric block rides the prune lane only
  let metric: DLEfficiencyMetric | null = null;
  if (efficiencyMode === "prune") {
    const metricRaw = r["metric"];
    if (typeof metricRaw !== "object" || metricRaw === null) {
      throw new Error(`${what}: the prune lane lost its metric block`);
    }
    const m = expectObject(metricRaw, `${what} metric`);
    const kind = m["kind"];
    const metricNote = m["note"];
    if (typeof kind !== "string" || typeof metricNote !== "string") {
      throw new Error(`${what}: the metric block lost its kind/note`);
    }
    metric = {
      kind,
      before: asNumber(m["before"]),
      after: asNumber(m["after"]),
      delta: asNumber(m["delta"]),
      note: metricNote,
    };
  }
  // the artifact: null when the lane was asked to skip it (the
  // request's own flag — never a silent default)
  const artifactRaw = r["artifact"];
  const artifact =
    typeof artifactRaw === "object" && artifactRaw !== null
      ? parseArtifactView(artifactRaw, `${what} artifact`)
      : null;
  return {
    mode: "sync",
    modelId,
    network,
    efficiencyMode,
    nLayers,
    nWeights,
    layers,
    artifact,
    method: quantTotals?.method ?? null,
    bitsPerWeight: quantTotals?.bitsPerWeight ?? null,
    originalBits: quantTotals?.originalBits ?? null,
    quantizedBits: quantTotals?.quantizedBits ?? null,
    memorySavings: quantTotals?.memorySavings ?? null,
    targetSparsity: pruneTotals?.targetSparsity ?? null,
    scope: pruneTotals?.scope ?? null,
    nZeroed: pruneTotals?.nZeroed ?? null,
    achievedSparsity: pruneTotals?.achievedSparsity ?? null,
    metric,
    note,
  };
}

/**
 * DL-4 — narrow the mpnn-train answer: the held-out carve's own
 * verdicts (val_rmse always a float; val_correlation null when the
 * carve was too small or degenerate — never an implied zero) plus
 * the honest rejected-rows accounting.
 */
export function parseMpnnTrainResult(
  raw: unknown,
  what: string,
): MpnnTrainResult {
  const r = expectObject(raw, what);
  const model = parseMpnnModelView(r["model"], `${what} model`);
  const valRmse = asNumber(r["val_rmse"]);
  const valN = asNumber(r["val_n"]);
  const nRows = asNumber(r["n_rows"]);
  const nRejected = asNumber(r["n_rejected"]);
  const nParameters = asNumber(r["n_parameters"]);
  const note = r["held_out_note"];
  if (
    valRmse === null ||
    valN === null ||
    nRows === null ||
    nRejected === null ||
    nParameters === null ||
    typeof note !== "string"
  ) {
    throw new Error(
      `${what}: val_rmse/val_n/n_rows/n_rejected/n_parameters lost ` +
        "the route's own types",
    );
  }
  return {
    model,
    val_rmse: valRmse,
    val_n: valN,
    val_correlation: asNumber(r["val_correlation"]),
    train_history_tail: asNumberArray(r["train_history_tail"], `${what} train tail`),
    refit_history_tail: asNumberArray(r["refit_history_tail"], `${what} refit tail`),
    n_rows: nRows,
    n_rejected: nRejected,
    rejected: parseMpnnRejected(r["rejected"], what),
    n_parameters: nParameters,
    held_out_note: note,
  };
}

/** Narrow the D-MPNN model view (the trained view + the molecular keys). */
function parseMpnnModelView(raw: unknown, what: string): MpnnModelView {
  const r = expectObject(raw, what);
  return {
    ...parseDLModelView(raw, what),
    molecular_model: r["molecular_model"] === true,
  };
}

/** The rejected rows' own accounting (index + smiles + error). */
function parseMpnnRejected(raw: unknown, what: string): MpnnRejectedRow[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const rows: MpnnRejectedRow[] = [];
  for (const rowRaw of raw as unknown[]) {
    const row = expectObject(rowRaw, `${what} rejected row`);
    const index = asNumber(row["index"]);
    const smiles = row["smiles"];
    const error = row["error"];
    if (index === null || typeof error !== "string") {
      throw new Error(
        `${what}: a rejected row lost its index/error (a row dropped ` +
          "silently is never the honest accounting)",
      );
    }
    rows.push({
      index,
      smiles: typeof smiles === "string" ? smiles : null,
      error,
    });
  }
  return rows;
}

function asNumberArray(raw: unknown, what: string): number[] {
  if (!Array.isArray(raw)) {
    throw new Error(`${what} is not an array`);
  }
  return raw.filter((v): v is number => typeof v === "number");
}

/**
 * DL-4 — narrow the mpnn-predict answer: one prediction per VALID
 * row (aligned with the served index), the rejected rows riding
 * their own accounting.
 */
export function parseMpnnPredictResult(
  raw: unknown,
  what: string,
): MpnnPredictResponse {
  const r = expectObject(raw, what);
  const modelId = r["model_id"];
  const nValid = asNumber(r["n_valid"]);
  const nRejected = asNumber(r["n_rejected"]);
  const note = r["note"];
  if (
    typeof modelId !== "string" ||
    nValid === null ||
    nRejected === null ||
    typeof note !== "string"
  ) {
    throw new Error(
      `${what}: model_id/n_valid/n_rejected/note lost the route's own types`,
    );
  }
  const predictionsRaw = r["predictions"];
  if (!Array.isArray(predictionsRaw)) {
    throw new Error(`${what}: predictions is not an array`);
  }
  const predictions: MpnnPredictionRow[] = [];
  for (const rowRaw of predictionsRaw as unknown[]) {
    const row = expectObject(rowRaw, `${what} prediction row`);
    const index = asNumber(row["index"]);
    const smiles = row["smiles"];
    const prediction = asNumber(row["prediction"]);
    if (
      index === null ||
      typeof smiles !== "string" ||
      prediction === null
    ) {
      throw new Error(
        `${what}: a prediction row lost its index/smiles/prediction`,
      );
    }
    predictions.push({ index, smiles, prediction });
  }
  if (predictions.length !== nValid) {
    throw new Error(
      `${what}: predictions (${predictions.length}) disagree with the ` +
        `served n_valid (${nValid})`,
    );
  }
  return {
    model_id: modelId,
    n_valid: nValid,
    predictions,
    n_rejected: nRejected,
    rejected: parseMpnnRejected(r["rejected"], what),
    note,
  };
}

/**
 * Narrow the gnn-predict answer: label sets (+ optional class
 * probabilities) for the graph rows — the graph models' own path.
 */
export function parseGnnPredictResult(
  raw: unknown,
  what: string,
): GnnPredictResponse {
  const r = expectObject(raw, what);
  const modelId = r["model_id"];
  const nGraphs = r["n_graphs"];
  const predictions = r["predictions"];
  const classes = r["classes"];
  if (
    typeof modelId !== "string" ||
    typeof nGraphs !== "number" ||
    !Array.isArray(predictions) ||
    !Array.isArray(classes)
  ) {
    throw new Error(
      `${what}: model_id/n_graphs/predictions/classes are malformed`,
    );
  }
  const probabilities = r["probabilities"];
  return {
    model_id: modelId,
    n_graphs: nGraphs,
    predictions: predictions.map(asLabel),
    classes: classes.map(asLabel),
    probabilities: Array.isArray(probabilities)
      ? probabilities.map((row) =>
          Array.isArray(row)
            ? row.filter((v): v is number => typeof v === "number")
            : [],
        )
      : null,
  };
}

/**
 * Parse + validate the page's inline-graph JSON (the textarea
 * lane): `{graphs: [{x, edge_index}, …], labels: […]}` in train
 * mode, `{graphs: […]}` (no labels) in predict mode — every row
 * validated BEFORE any request fires (the backend’s own 422s still
 * rule the wire; this gate keeps malformed JSON client-side).
 */
export function parseGnnInlineGraphs(
  text: string,
  mode: "train" | "predict" = "train",
): GnnInlineSource | null {
  const trimmed = text.trim();
  if (trimmed === "") {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    throw new Error(
      "the inline graphs block is not valid JSON — expected " +
        '{"graphs": [{"x": [[…]], "edge_index": [[…], […]}], "labels": […]}',
    );
  }
  const r = expectObject(parsed, "inline graphs");
  const graphs = r["graphs"];
  if (!Array.isArray(graphs)) {
    throw new Error('the inline graphs block needs a "graphs" array');
  }
  const labels = r["labels"];
  if (mode === "train") {
    if (!Array.isArray(labels)) {
      throw new Error(
        'the inline training block needs its "labels" array ' +
          "(one label per graph)",
      );
    }
    if (graphs.length !== labels.length) {
      throw new Error(
        `graphs has ${graphs.length} rows; labels has ${labels.length} ` +
          "(one label per graph)",
      );
    }
    if (graphs.length < 2) {
      throw new Error("inline training needs at least two graphs");
    }
  } else if (Array.isArray(labels) && labels.length > 0) {
    throw new Error(
      "the predict block carries no labels — predictions are the " +
        "output, never an input",
    );
  }
  const rows: GnnGraphRow[] = [];
  for (const [i, graph] of graphs.entries()) {
    const g = expectObject(graph, `graphs[${i}]`);
    const x = g["x"];
    const edgeIndex = g["edge_index"];
    if (!Array.isArray(x) || !Array.isArray(edgeIndex)) {
      throw new Error(
        `graphs[${i}]: needs both "x" and "edge_index" arrays`,
      );
    }
    if (x.length === 0) {
      throw new Error(`graphs[${i}]: a graph needs at least one node`);
    }
    const width = Array.isArray(x[0]) ? x[0].length : -1;
    for (const [j, node] of x.entries()) {
      if (
        !Array.isArray(node) ||
        node.length === 0 ||
        !node.every((v) => typeof v === "number")
      ) {
        throw new Error(
          `graphs[${i}].x[${j}]: node features must be a non-empty ` +
            "number array",
        );
      }
      if (node.length !== width) {
        throw new Error(
          `graphs[${i}].x[${j}]: node width ${node.length} ≠ the ` +
            `graph's ${width} — all nodes in the batch must share one width`,
        );
      }
    }
    for (const [j, endpoints] of edgeIndex.entries()) {
      if (
        !Array.isArray(endpoints) ||
        !endpoints.every(
          (v) => typeof v === "number" && Number.isInteger(v) && v >= 0,
        )
      ) {
        throw new Error(
          `graphs[${i}].edge_index[${j}]: endpoints must be non-negative ` +
            "integer node indices",
        );
      }
    }
    if (
      edgeIndex.length > 0 &&
      edgeIndex.some(
        (endpoints) =>
          Array.isArray(endpoints) &&
          endpoints.some((v) => Number(v) >= x.length),
      )
    ) {
      throw new Error(
        `graphs[${i}]: an edge endpoint exceeds the graph's ` +
          `${x.length} nodes`,
      );
    }
    rows.push({
      x: x.map((node) => [...(node as number[])]),
      edge_index: edgeIndex.map((endpoints) => [...(endpoints as number[])]),
    });
  }
  const labelNumbers =
    mode === "train" && Array.isArray(labels)
      ? labels.map((v) => {
          if (typeof v !== "number" || !Number.isInteger(v)) {
            throw new Error("labels must be integers (one class per graph)");
          }
          return v;
        })
      : null;
  if (rows.length < 1) {
    throw new Error("the block carries no graphs");
  }
  return { graphs: rows, labels: labelNumbers };
}

// -- the sequence-CNN + autoencoder lanes (part18 §5.2 / §5.5) ------------

/**
 * The sequence-CNN train options — the inline source carries the
 * rectangular 3-D sequences (`(seq_len, channels)` step lists) plus
 * one label per sequence; the synthetic source rides the backend's
 * own deterministic motif fixture.
 */
export interface CnnTrainOptions {
  syntheticN?: number;
  seed?: number;
  inline?: {
    sequences: readonly (readonly (readonly number[])[])[];
    labels: readonly number[];
  };
  task: "classification" | "regression";
  channels: number;
  kernel: number;
  nLayers: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  weightDecay: number;
  patience: number;
  valSplit: number;
  randomState: number;
  mode?: "job" | "sync";
}

/** The sequence model's own view fields (its predict path is its own). */
export interface CnnModelView extends DLModelView {
  sequenceModel: boolean;
}

/** The `POST /api/v1/dl/cnn-train` result body. */
export interface CnnTrainResult {
  model: CnnModelView;
  seqLen: number;
  nChannels: number;
  nIter: number;
  classes: readonly (number | string)[] | null;
  trainAccuracy: number | null;
  trainMae: number | null;
}

export interface CnnJobResponse {
  mode: "job";
  job: JobView;
}

export interface CnnTrainSyncResponse {
  mode: "sync";
  result: CnnTrainResult;
}

export type CnnTrainResponse = CnnJobResponse | CnnTrainSyncResponse;

/** The sequence models' own prediction options. */
export interface CnnPredictOptions {
  modelId: string;
  sequences: readonly (readonly (readonly number[])[])[];
  probabilities?: boolean;
}

/** The `POST /api/v1/dl/cnn-predict` result body. */
export interface CnnPredictResponse {
  modelId: string;
  nSequences: number;
  predictions: readonly (number | string)[];
  probabilities: readonly (readonly number[])[] | null;
  classes: readonly (number | string)[] | null;
}

/** The autoencoder train options (unsupervised over a dataset). */
export interface AutoencoderTrainOptions {
  datasetId: string;
  hiddenLayers: readonly number[];
  latentDim: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  weightDecay: number;
  patience: number;
  valSplit: number;
  randomState: number;
  mode?: "job" | "sync";
}

/** The autoencoder model view (task `unsupervised`, its own encode URL). */
export interface AutoencoderModelView extends DLModelView {
  encodeUrl: string;
}

/** The `POST /api/v1/dl/autoencoder-train` result body. */
export interface AutoencoderTrainResult {
  model: AutoencoderModelView;
  latentDim: number;
  nIter: number;
  meanReconstructionError: number;
}

export interface AutoencoderJobResponse {
  mode: "job";
  job: JobView;
}

export interface AutoencoderTrainSyncResponse {
  mode: "sync";
  result: AutoencoderTrainResult;
}

export type AutoencoderTrainResponse =
  | AutoencoderJobResponse
  | AutoencoderTrainSyncResponse;

/** The encode lane's options (a dataset or inline rows, never both). */
export interface AutoencoderEncodeOptions {
  modelId: string;
  datasetId?: string;
  rows?: readonly Record<string, number>[];
  topK: number;
}

/** One honest outlier ranking row (index + reconstruction error). */
export interface AutoencoderOutlier {
  index: number;
  error: number;
}

/** The `POST /api/v1/dl/autoencoder-encode` result body. */
export interface AutoencoderEncodeResponse {
  modelId: string;
  nRows: number;
  latentDim: number;
  latent: readonly (readonly number[])[];
  reconstructionErrors: readonly number[];
  meanReconstructionError: number;
  topOutliers: readonly AutoencoderOutlier[];
  note: string;
}

/**
 * Parse + validate the cnn-train result (the sync body and the job
 * result relay share one shape — §8.1 runtime validation).
 */
export function parseCnnTrainResult(
  raw: unknown,
  what: string,
): CnnTrainResult {
  const r = expectObject(raw, what);
  const model = parseDLModelView(r["model"], `${what} model`);
  const modelRaw = expectObject(r["model"], `${what} model`);
  const classes = Array.isArray(r["classes"])
    ? r["classes"].map(asLabel)
    : null;
  const trainAccuracy = asNumber(r["train_accuracy"]);
  const trainMae = asNumber(r["train_mae"]);
  return {
    model: {
      ...model,
      sequenceModel: modelRaw["sequence_model"] === true,
    },
    seqLen: asNumber(r["seq_len"]) ?? 0,
    nChannels: asNumber(r["n_channels"]) ?? 0,
    nIter: asNumber(r["n_iter"]) ?? 0,
    classes,
    trainAccuracy,
    trainMae,
  };
}

/** Parse + validate the cnn-predict result. */
export function parseCnnPredictResult(
  raw: unknown,
  what: string,
): CnnPredictResponse {
  const r = expectObject(raw, what);
  const predictions = Array.isArray(r["predictions"])
    ? r["predictions"].map(asLabel)
    : [];
  const classes = Array.isArray(r["classes"])
    ? r["classes"].map(asLabel)
    : null;
  const probabilities = Array.isArray(r["probabilities"])
    ? r["probabilities"].map((row: unknown) =>
        Array.isArray(row)
          ? row.filter((v): v is number => typeof v === "number")
          : [],
      )
    : null;
  return {
    modelId: typeof r["model_id"] === "string" ? r["model_id"] : "",
    nSequences: asNumber(r["n_sequences"]) ?? 0,
    predictions,
    probabilities,
    classes,
  };
}

/** Parse + validate the autoencoder-train result. */
export function parseAutoencoderTrainResult(
  raw: unknown,
  what: string,
): AutoencoderTrainResult {
  const r = expectObject(raw, what);
  const model = parseDLModelView(r["model"], `${what} model`);
  const modelRaw = expectObject(r["model"], `${what} model`);
  return {
    model: {
      ...model,
      encodeUrl:
        typeof modelRaw["encode_url"] === "string"
          ? modelRaw["encode_url"]
          : "/api/v1/dl/autoencoder-encode",
    },
    latentDim: asNumber(r["latent_dim"]) ?? 0,
    nIter: asNumber(r["n_iter"]) ?? 0,
    meanReconstructionError:
      asNumber(r["mean_reconstruction_error"]) ?? Number.NaN,
  };
}

/** Parse + validate the autoencoder-encode result. */
export function parseAutoencoderEncodeResult(
  raw: unknown,
  what: string,
): AutoencoderEncodeResponse {
  const r = expectObject(raw, what);
  const latent = Array.isArray(r["latent"])
    ? r["latent"].map((row: unknown) =>
        Array.isArray(row)
          ? row.filter((v): v is number => typeof v === "number")
          : [],
      )
    : [];
  const reconstructionErrors = Array.isArray(r["reconstruction_errors"])
    ? r["reconstruction_errors"].filter(
        (v: unknown): v is number => typeof v === "number",
      )
    : [];
  const topOutliers = Array.isArray(r["top_outliers"])
    ? r["top_outliers"].flatMap((entry: unknown) => {
        if (typeof entry !== "object" || entry === null) {
          return [];
        }
        const record = entry as Record<string, unknown>;
        const index = asNumber(record["index"]);
        const error = asNumber(record["error"]);
        if (index === null || error === null) {
          return [];
        }
        return [{ index, error }];
      })
    : [];
  return {
    modelId: typeof r["model_id"] === "string" ? r["model_id"] : "",
    nRows: asNumber(r["n_rows"]) ?? 0,
    latentDim: asNumber(r["latent_dim"]) ?? 0,
    latent,
    reconstructionErrors,
    meanReconstructionError:
      asNumber(r["mean_reconstruction_error"]) ?? Number.NaN,
    topOutliers,
    note: typeof r["note"] === "string" ? r["note"] : "",
  };
}
