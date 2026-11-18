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
 *   note.
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

export interface NetworkCatalogResponse {
  readonly networks: readonly NetworkCatalogEntry[];
  readonly predict_url: string;
  readonly note: string;
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
      };
    }
    return {
      networks: networks.map((item, i) =>
        parseNetworkCatalogEntry(item, `networks[${i}]`),
      ),
      predict_url:
        typeof raw["predict_url"] === "string" ? raw["predict_url"] : "",
      note: typeof raw["note"] === "string" ? raw["note"] : "",
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
