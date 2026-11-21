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

/** The job-mode submission answer. */
export interface DLJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

/** The sync-mode train answer (`{mode: "sync", model: view}`). */
export interface DLTrainSyncResponse {
  readonly mode: "sync";
  readonly model: DLModelView;
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
}

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
    };
    const raw = (await api.post(
      "/api/v1/dl/train",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        model: parseDLModelView(raw["model"], "dl.train(sync)"),
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
