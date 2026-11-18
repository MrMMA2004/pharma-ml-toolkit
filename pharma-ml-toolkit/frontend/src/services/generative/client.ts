/**
 * The Generative AI client — part11 §11.4's workflow service (the
 * prior/sample + genetic-optimize surface over the REAL backend
 * engines).
 *
 * The backend contract (mirrored field-for-field from the route source
 * `routes/generative.py`; the request bodies come from the generated
 * OpenAPI types — the same one-DTO-source rule the Python drift guards
 * in `tests/test_generative_workspace.py` pin for this client):
 *
 * * `GET /api/v1/generative/objectives` — the scoring-oracle catalog
 *   (`_OBJECTIVE_CATALOG`: the nine objective kinds — QED, SAS, the
 *   LogP/MW/TPSA bands, similarity, substructure, novelty, and the
 *   weighted multi — with each objective's own parameter schema).
 * * `POST /api/v1/generative/rnn/train` —
 *   `PriorTrainRequest` → `{mode: "job", job: JobView}` (the default
 *   lane here) / `{mode: "sync", prior: PriorView}`; the COMPLETED
 *   job's REST result payload is `{prior: PriorView}` — the page relays
 *   the authoritative REST result through the jobs client (§13), never
 *   the WS event payload alone. The prior's REAL per-epoch loss curve
 *   is published as one `training.metrics` envelope when training ends
 *   (`{stage, epochs, losses, final_loss}` — narrowed here by
 *   `parseTrainingMetricsPayload`, §8.1 runtime validation).
 * * `POST /api/v1/generative/rnn/sample` — `SampleRequest` → the
 *   sampled SMILES off a session prior (this lane is sync by design:
 *   the route declares no job mode — stated honestly in the page, the
 *   Generate stage notes it, no pretend tracking).
 * * `POST /api/v1/generative/optimize` — `OptimizeRequest` →
 *   `{mode: "job", job: JobView}` (the default) /
 *   `{mode: "sync", optimization: OptimizationView}`; the COMPLETED
 *   job's REST result payload is `{optimization: OptimizationView}`
 *   (the REAL evolution: the optimizer's own progress callback drives
 *   one `iteration.update` envelope per generation — `{progress,
 *   stage}`, narrowed by `parseIterationUpdatePayload`).
 *
 * The properties lane (Validate + Calculate Properties) rides the
 * chemistry client's own `descriptors` surface — never re-declared
 * here. The figure lanes (the prior's REAL loss curve, the ranked
 * score scatter) ride `POST /api/v1/artifacts/figure`; the export
 * rides the rows-based `POST /api/v1/visualization/table` — both
 * posted from the generated types (the workspace row's own surface),
 * parsed here with `parseArtifactView`.
 */

import { api } from "@/services/api/client";
import { parseArtifactView, type ArtifactView } from "@/services/artifacts/client";
import { type JobView, parseJobView } from "@/services/jobs/types";
import type { components } from "@/types/api.d";

type PriorTrainRequest = components["schemas"]["PriorTrainRequest"];
type SampleRequest = components["schemas"]["SampleRequest"];
type OptimizeRequest = components["schemas"]["OptimizeRequest"];
type FigureRequest = components["schemas"]["FigureRequest"];
type VisualizationTableRequest =
  components["schemas"]["VisualizationTableRequest"];

/**
 * The objective-kind vocabulary — the nine kinds the backend's own
 * `_OBJECTIVE_CATALOG` rows carry (mirrored once beside the route so
 * the page can build the spec editor honestly: an unknown kind is the
 * route's own 422, never a silent client guess).
 */
export const OBJECTIVE_KINDS: readonly string[] = [
  "qed",
  "sas",
  "logp_band",
  "mw_band",
  "tpsa_band",
  "similarity",
  "substructure",
  "novelty",
  "multi",
] as const;

/** One `_OBJECTIVE_CATALOG` row — the catalog's own keys. */
export interface ObjectiveCatalogEntry {
  readonly kind: string;
  readonly class: string;
  /** The objective's own parameter schema (name → example default). */
  readonly params: Readonly<Record<string, unknown>>;
  readonly description: string;
}

export interface ObjectivesResponse {
  readonly objectives: readonly ObjectiveCatalogEntry[];
}

/** `PriorRecord.view()` — the session prior registry's own shape. */
export interface PriorView {
  readonly prior_id: string;
  readonly name: string;
  readonly kind: string;
  readonly params: Readonly<Record<string, unknown>>;
  readonly n_train_smiles: number;
  readonly n_epochs: number;
  readonly final_loss: number | null;
  readonly losses: readonly (number | null)[];
  readonly corpus_source: Readonly<Record<string, unknown>>;
  readonly created_at: string;
  readonly sample_url: string;
}

export interface PriorsResponse {
  readonly priors: readonly PriorView[];
}

/** The prior-lane job/sync submission answer. */
export interface PriorTrainJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

export interface PriorTrainSyncResponse {
  readonly mode: "sync";
  readonly prior: PriorView;
}

export type PriorTrainResponse = PriorTrainJobResponse | PriorTrainSyncResponse;

/** `POST /rnn/sample` — the sampled SMILES off a session prior. */
export interface SampleResponse {
  readonly prior_id: string;
  readonly n_requested: number;
  readonly n_sampled: number;
  readonly unique: number;
  readonly samples: readonly string[];
  readonly params: {
    readonly temperature: number;
    readonly top_k: number | null;
    readonly seed: number | null;
    readonly max_length: number;
  };
}

/** One ranked optimize row — `top`'s own keys. */
export interface OptimizationTopRow {
  readonly smiles: string;
  readonly score: number | null;
  readonly breakdown: Readonly<Record<string, unknown>>;
}

/** The `_optimize_body` result — the REAL evolution's own shape. */
export interface OptimizationView {
  readonly best: string | null;
  readonly best_score: number | null;
  readonly generations: number;
  readonly history: readonly unknown[];
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly top: readonly OptimizationTopRow[];
}

export interface OptimizeJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

export interface OptimizeSyncResponse {
  readonly mode: "sync";
  readonly optimization: OptimizationView;
}

export type OptimizeResponse = OptimizeJobResponse | OptimizeSyncResponse;

/**
 * One `training.metrics` envelope payload — the prior lane's REAL loss
 * curve, published once when the prior's training ends (the route's own
 * `{stage, epochs, losses, final_loss}` dict).
 */
export interface TrainingMetricsPayload {
  readonly stage: string;
  readonly epochs: number;
  readonly losses: readonly (number | null)[];
  readonly final_loss: number | null;
}

/** The `iteration.update` envelope payload (progress %, stage). */
export interface IterationUpdatePayload {
  readonly progress: number;
  readonly stage: string;
}

export interface PriorTrainOptions {
  /** Inline corpus SMILES (≥ 5, the route's own rule) or null. */
  readonly smiles: readonly string[] | null;
  /** A registered dataset's id + SMILES column (or null). */
  readonly datasetId: string | null;
  readonly column: string | null;
  readonly name: string;
  readonly embeddingDim: number;
  readonly hiddenDim: number;
  readonly nLayers: number;
  readonly epochs: number;
  readonly batchSize: number;
  readonly lr: number;
  readonly gradClip: number;
  readonly seed: number;
  readonly mode?: "job" | "sync";
}

export interface SampleOptions {
  readonly priorId: string;
  readonly n: number;
  readonly temperature: number;
  readonly topK: number | null;
  readonly seed: number | null;
  readonly maxLength: number;
}

export interface OptimizeOptions {
  /** The objective spec (the catalog kind + its own params). */
  readonly objective: Record<string, unknown>;
  /** Seed SMILES (≤ 200) and/or a dataset corpus. */
  readonly seedSmiles: readonly string[] | null;
  readonly datasetId: string | null;
  readonly column: string | null;
  readonly populationSize: number;
  readonly generations: number;
  readonly mutationRate: number;
  readonly crossoverRate: number;
  readonly elitism: number;
  readonly tournamentK: number;
  readonly maxSmilesLength: number;
  readonly seed: number;
  readonly topN: number;
  readonly mode?: "job" | "sync";
}

export interface TableExportOptions {
  /** The ranked/filtered rows the export carries. */
  readonly rows: Readonly<Record<string, unknown>[]>;
  /** The backend's own export enum (visualization.py's own literals). */
  readonly format: "csv" | "tsv" | "xlsx" | "md" | "json" | "html";
  readonly filename: string;
}

export const generativeApi = {
  /**
   * `GET /api/v1/generative/objectives` — the nine-objective scoring
   * catalog, each row the backend's own parameter schema.
   */
  async objectives(): Promise<ObjectivesResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/generative/objectives",
    );
    const objectives = raw["objectives"];
    if (!Array.isArray(objectives)) {
      return { objectives: [] };
    }
    return {
      objectives: objectives.map((item, i) =>
        parseObjectiveCatalogEntry(item, `objectives[${i}]`),
      ),
    };
  },

  /** `GET /api/v1/generative/priors` — the session prior registry. */
  async priors(): Promise<PriorsResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/generative/priors",
    );
    const priors = raw["priors"];
    if (!Array.isArray(priors)) {
      return { priors: [] };
    }
    return {
      priors: priors.map((item, i) => parsePriorView(item, `priors[${i}]`)),
    };
  },

  /** `GET /api/v1/generative/priors/{prior_id}` — one prior's view. */
  async priorView(priorId: string): Promise<PriorView> {
    const raw = await api.get<Record<string, unknown>>(
      `/api/v1/generative/priors/${encodeURIComponent(priorId)}`,
    );
    return parsePriorView(raw, "prior view");
  },

  /**
   * `POST /api/v1/generative/rnn/train` — the REAL
   * `SmilesRnn.train_prior` over an inline corpus or a registered
   * dataset's SMILES column. Job mode (the default) answers
   * `{mode: "job", job: JobView}`; the COMPLETED result payload is
   * `{prior: PriorView}` (parsePriorTrainResult).
   */
  async trainPrior(options: PriorTrainOptions): Promise<PriorTrainResponse> {
    const request: PriorTrainRequest = {
      smiles: options.smiles === null ? null : [...options.smiles],
      dataset_id: options.datasetId,
      column: options.column,
      name: options.name,
      embedding_dim: options.embeddingDim,
      hidden_dim: options.hiddenDim,
      n_layers: options.nLayers,
      epochs: options.epochs,
      batch_size: options.batchSize,
      lr: options.lr,
      grad_clip: options.gradClip,
      seed: options.seed,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/generative/rnn/train",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        prior: parsePriorView(raw["prior"], "rnn.train(sync)"),
      };
    }
    return { mode: "job", job: parseJobView(raw["job"], "rnn.train(job)") };
  },

  /**
   * `POST /api/v1/generative/rnn/sample` — sampled SMILES off a
   * session prior (temperature / top-k / seed / max length passed
   * through verbatim). Sync by design: the route declares no job mode.
   */
  async sample(options: SampleOptions): Promise<SampleResponse> {
    const request: SampleRequest = {
      prior_id: options.priorId,
      n: options.n,
      temperature: options.temperature,
      top_k: options.topK,
      seed: options.seed,
      max_length: options.maxLength,
    };
    const raw = (await api.post(
      "/api/v1/generative/rnn/sample",
      request,
    )) as Record<string, unknown>;
    return parseSampleResponse(raw, "rnn.sample");
  },

  /**
   * `POST /api/v1/generative/optimize` — the REAL evolution
   * (`GeneticOptimizer.optimize` with the optimizer's own per-
   * generation `iteration.update` telemetry; the native cancel path
   * lands the job in CANCELLED, never FAILED). Job mode (the default)
   * answers `{mode: "job", job: JobView}`; the COMPLETED result
   * payload is `{optimization: OptimizationView}`.
   */
  async optimize(options: OptimizeOptions): Promise<OptimizeResponse> {
    const request: OptimizeRequest = {
      objective: options.objective,
      seed_smiles: options.seedSmiles === null ? null : [...options.seedSmiles],
      dataset_id: options.datasetId,
      column: options.column,
      population_size: options.populationSize,
      generations: options.generations,
      mutation_rate: options.mutationRate,
      crossover_rate: options.crossoverRate,
      elitism: options.elitism,
      tournament_k: options.tournamentK,
      max_smiles_length: options.maxSmilesLength,
      seed: options.seed,
      top_n: options.topN,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/generative/optimize",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        optimization: parseOptimizationView(
          raw["optimization"],
          "optimize(sync)",
        ),
      };
    }
    return { mode: "job", job: parseJobView(raw["job"], "optimize(job)") };
  },

  /**
   * `POST /api/v1/artifacts/figure` — the prior's REAL per-epoch loss
   * curve as a figure artifact (one scatter trace: x = epoch, y = the
   * loss the backend actually recorded). The curve needs at least two
   * epochs — the honest minimum, stated before the request fires.
   */
  async priorLossFigure(
    losses: readonly (number | null)[],
    format: "PNG" | "SVG" | "TIFF" | "JPEG",
  ): Promise<ArtifactView> {
    const epochs = losses.length;
    if (epochs < 2) {
      throw new Error(
        "a prior loss curve needs at least two epochs of telemetry",
      );
    }
    const x: (number | string)[] = [];
    const y: number[] = [];
    const color: string[] = [];
    for (let i = 0; i < epochs; i++) {
      const loss = losses[i];
      if (typeof loss === "number" && Number.isFinite(loss)) {
        x.push(i + 1);
        y.push(loss);
        // the route's own validator: color must be the same length as x
        color.push("prior_loss");
      }
    }
    if (y.length < 2) {
      throw new Error(
        "a prior loss curve needs at least two finite loss points",
      );
    }
    const request: FigureRequest = {
      kind: "scatter",
      x,
      y,
      color,
      title: "Generative AI — prior training loss per epoch",
      format,
      dpi: 150,
      filename: "generative-prior-loss",
    };
    const raw = (await api.post(
      "/api/v1/artifacts/figure",
      request,
    )) as Record<string, unknown>;
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the figure lane answered without an artifact");
    }
    return parseArtifactView(artifact, "prior loss figure");
  },

  /**
   * `POST /api/v1/artifacts/figure` — the ranked molecules' REAL
   * value profile as a figure artifact (one scatter: x = the rank
   * position, y = the ranking key's value — the objective score the
   * evolution assigned or the property the Rank stage selected —
   * color = the kept / filtered lane the Filter stage chose; the
   * axis label names the real key, never an invented one).
   */
  async rankFigure(
    rows: readonly { readonly score: number | null; readonly kept: boolean }[],
    format: "PNG" | "SVG" | "TIFF" | "JPEG",
    scoreLabel: string,
  ): Promise<ArtifactView> {
    const x: (number | string)[] = [];
    const y: number[] = [];
    const color: string[] = [];
    rows.forEach((row, i) => {
      if (typeof row.score === "number" && Number.isFinite(row.score)) {
        x.push(i + 1);
        y.push(row.score);
        color.push(row.kept ? "kept" : "filtered");
      }
    });
    if (y.length < 2) {
      throw new Error(
        "a score profile needs at least two scored molecules",
      );
    }
    const request: FigureRequest = {
      kind: "scatter",
      x,
      y,
      color,
      title: `Generative AI — ranked molecules by ${scoreLabel}`,
      format,
      dpi: 150,
      filename: "generative-ranked-scores",
    };
    const raw = (await api.post(
      "/api/v1/artifacts/figure",
      request,
    )) as Record<string, unknown>;
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the figure lane answered without an artifact");
    }
    return parseArtifactView(artifact, "ranked score figure");
  },

  /**
   * `POST /api/v1/visualization/table` — the ranked/filtered result
   * table as a REAL artifact (the §17 four-format export stack). The
   * caller carries the backend's own 5000-row cap — mirrored at the
   * page, enforced before the request fires (the error names both
   * numbers, never a silent truncation).
   */
  async exportTable(options: TableExportOptions): Promise<ArtifactView> {
    if (options.rows.length === 0) {
      throw new Error("the export carries no rows");
    }
    const request: VisualizationTableRequest = {
      rows: [...options.rows],
      format: options.format,
      filename: options.filename,
    };
    const raw = (await api.post(
      "/api/v1/visualization/table",
      request,
    )) as Record<string, unknown>;
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the table lane answered without an artifact");
    }
    return parseArtifactView(artifact, "table export");
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

/** Narrow an untrusted payload to one objective catalog row. */
function parseObjectiveCatalogEntry(
  raw: unknown,
  what: string,
): ObjectiveCatalogEntry {
  const r = expectObject(raw, what);
  const kind = r["kind"];
  if (typeof kind !== "string") {
    throw new Error(`${what}: kind is not a string`);
  }
  const params = r["params"];
  return {
    kind,
    class: typeof r["class"] === "string" ? r["class"] : "",
    params:
      typeof params === "object" && params !== null
        ? (params as Record<string, unknown>)
        : {},
    description:
      typeof r["description"] === "string" ? r["description"] : "",
  };
}

/**
 * Narrow an untrusted payload to `PriorRecord.view()`'s own shape (the
 * registry's view — the losses stay exactly as they arrived, nulls
 * never invented).
 */
export function parsePriorView(raw: unknown, what: string): PriorView {
  const r = expectObject(raw, what);
  const priorId = r["prior_id"];
  const name = r["name"];
  if (typeof priorId !== "string" || typeof name !== "string") {
    throw new Error(`${what}: prior_id/name are not strings`);
  }
  const params = r["params"];
  const corpusSource = r["corpus_source"];
  const losses = r["losses"];
  return {
    prior_id: priorId,
    name,
    kind: typeof r["kind"] === "string" ? r["kind"] : "",
    params:
      typeof params === "object" && params !== null
        ? (params as Record<string, unknown>)
        : {},
    n_train_smiles:
      typeof r["n_train_smiles"] === "number" ? r["n_train_smiles"] : 0,
    n_epochs: typeof r["n_epochs"] === "number" ? r["n_epochs"] : 0,
    final_loss: asNumber(r["final_loss"]),
    losses: Array.isArray(losses) ? losses.map((v) => asNumber(v)) : [],
    corpus_source:
      typeof corpusSource === "object" && corpusSource !== null
        ? (corpusSource as Record<string, unknown>)
        : {},
    created_at: typeof r["created_at"] === "string" ? r["created_at"] : "",
    sample_url: typeof r["sample_url"] === "string" ? r["sample_url"] : "",
  };
}

/** Narrow the sample answer (the samples stay strings, verbatim). */
export function parseSampleResponse(
  raw: unknown,
  what: string,
): SampleResponse {
  const r = expectObject(raw, what);
  const samples = r["samples"];
  if (!Array.isArray(samples)) {
    throw new Error(`${what}: samples is not an array`);
  }
  const params = expectObject(r["params"], `${what} params`);
  const topK = params["top_k"];
  const seed = params["seed"];
  return {
    prior_id: typeof r["prior_id"] === "string" ? r["prior_id"] : "",
    n_requested: typeof r["n_requested"] === "number" ? r["n_requested"] : 0,
    n_sampled: typeof r["n_sampled"] === "number" ? r["n_sampled"] : 0,
    unique: typeof r["unique"] === "number" ? r["unique"] : 0,
    samples: samples.map((v) => (typeof v === "string" ? v : String(v))),
    params: {
      temperature: asNumber(params["temperature"]) ?? 0,
      top_k: typeof topK === "number" ? topK : null,
      seed: typeof seed === "number" ? seed : null,
      max_length: asNumber(params["max_length"]) ?? 0,
    },
  };
}

/**
 * Narrow the optimize job's result payload: `{optimization:
 * OptimizationView}` (the authoritative REST relay — the top rows keep
 * the smiles/score/breakdown exactly as the evolution produced them).
 */
export function parseOptimizeResult(
  raw: unknown,
  what: string,
): OptimizationView {
  const r = expectObject(raw, what);
  return parseOptimizationView(r["optimization"], `${what} optimization`);
}

/** Narrow one `_optimize_body` answer (sync mode / the job result). */
export function parseOptimizationView(
  raw: unknown,
  what: string,
): OptimizationView {
  const r = expectObject(raw, what);
  const top = r["top"];
  const history = r["history"];
  const metadata = r["metadata"];
  return {
    best: typeof r["best"] === "string" ? r["best"] : null,
    best_score: asNumber(r["best_score"]),
    generations: typeof r["generations"] === "number" ? r["generations"] : 0,
    history: Array.isArray(history) ? history : [],
    metadata:
      typeof metadata === "object" && metadata !== null
        ? (metadata as Record<string, unknown>)
        : {},
    top: Array.isArray(top)
      ? top.map((row, i) => parseOptimizationTopRow(row, `${what} top[${i}]`))
      : [],
  };
}

/** Narrow one ranked row (`top`'s own keys). */
function parseOptimizationTopRow(
  raw: unknown,
  what: string,
): OptimizationTopRow {
  const r = expectObject(raw, what);
  const smiles = r["smiles"];
  if (typeof smiles !== "string") {
    throw new Error(`${what}: smiles is not a string`);
  }
  const breakdown = r["breakdown"];
  return {
    smiles,
    score: asNumber(r["score"]),
    breakdown:
      typeof breakdown === "object" && breakdown !== null
        ? (breakdown as Record<string, unknown>)
        : {},
  };
}

/**
 * Narrow one `training.metrics` envelope payload — the prior lane's
 * REAL loss curve dict (`{stage, epochs, losses, final_loss}`,
 * published once when the prior's training ends).
 */
export function parseTrainingMetricsPayload(
  raw: unknown,
  what: string,
): TrainingMetricsPayload {
  const r = expectObject(raw, what);
  const epochs = r["epochs"];
  if (typeof epochs !== "number") {
    throw new Error(`${what}: epochs is not a number`);
  }
  const losses = r["losses"];
  return {
    stage: typeof r["stage"] === "string" ? r["stage"] : "",
    epochs,
    losses: Array.isArray(losses) ? losses.map((v) => asNumber(v)) : [],
    final_loss: asNumber(r["final_loss"]),
  };
}

/** Narrow one `iteration.update` envelope payload (progress % + stage). */
export function parseIterationUpdatePayload(
  raw: unknown,
  what: string,
): IterationUpdatePayload {
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

/**
 * Narrow the prior-train job's result payload: `{prior: PriorView}`
 * (the authoritative REST relay).
 */
export function parsePriorTrainResult(
  raw: unknown,
  what: string,
): PriorView {
  const r = expectObject(raw, what);
  return parsePriorView(r["prior"], `${what} prior`);
}
