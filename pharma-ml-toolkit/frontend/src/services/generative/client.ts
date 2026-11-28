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
type GenerationMetricsRequest = components["schemas"]["GenerationMetricsRequest"];
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
  /** GEN-1: the operator modes the GA engine validates (the literals
   * the generated OptimizeRequest schema carries). */
  readonly mutationMode: "uniform" | "scaffold" | "topology" | "bioisostere";
  readonly crossoverMode: "splice" | "scaffold_hop";
  /** the scaffold-hop child gate (1 = off — the engine's own default). */
  readonly minHeavyAtoms: number;
  readonly seed: number;
  readonly topN: number;
  readonly mode?: "job" | "sync";
}

/**
 * GEN-3 (the roadmap's row 17) — the RL lane's request options
 * (mirrored from `routes/generative.py`'s RlOptimizeRequest: the
 * REINVENT-style optimizer over a prior or the on-the-fly corpus).
 */
export interface RlOptimizeOptions {
  /** The objective spec (the SAME vocabulary the GA lane rides). */
  readonly objective: Record<string, unknown>;
  /** a session-trained molecular prior, or null for the engine's own */
  readonly priorId: string | null;
  /** an inline on-the-fly prior corpus (or null for DEFAULT_CORPUS) */
  readonly smiles: readonly string[] | null;
  readonly priorEpochs: number;
  readonly iterations: number;
  readonly batchSize: number;
  readonly temperature: number;
  readonly lr: number;
  readonly anchorWeight: number;
  readonly entropyBonus: number;
  readonly replaySize: number;
  readonly patience: number;
  readonly maxSmilesLength: number;
  readonly seed: number;
  readonly topN: number;
  readonly mode?: "job" | "sync";
}

/** GEN-3 — one RL history row (the engine's own introspection). */
export interface RlIterationRow {
  readonly iteration: number;
  readonly best_score: number | null;
  readonly mean_score: number | null;
  readonly valid_fraction: number | null;
  readonly drift: number | null;
  readonly entropy: number | null;
  readonly loss: number | null;
}

/** GEN-3 — the rl-optimize answer's optimization block. */
export interface RlOptimizationView {
  readonly best: string | null;
  readonly best_score: number | null;
  readonly iterations: number;
  readonly history: readonly RlIterationRow[];
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly top: readonly OptimizationTopRow[];
}

/** GEN-3 — the rl-optimize sync answer (the job relay carries the same). */
export interface RlOptimizeResult {
  readonly mode: "sync";
  readonly optimization: RlOptimizationView;
  readonly prior: PriorView | null;
  readonly note: string;
}

/** The RL lane's submission answer (job mode). */
export interface RlJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

export type RlOptimizeResponse = RlJobResponse | RlOptimizeResult;

/** The engine's mutation-mode vocabulary (the route Literal's own set). */
export const GA_MUTATION_LITERALS = [
  "uniform",
  "scaffold",
  "topology",
  "bioisostere",
] as const;

/** The engine's crossover-mode vocabulary (the route Literal's own set). */
export const GA_CROSSOVER_LITERALS = ["splice", "scaffold_hop"] as const;

export interface TableExportOptions {
  /** The ranked/filtered rows the export carries. */
  readonly rows: Readonly<Record<string, unknown>[]>;
  /** The backend's own export enum (visualization.py's own literals). */
  readonly format: "csv" | "tsv" | "xlsx" | "md" | "json" | "html";
  readonly filename: string;
}

// ---------------------------------------------------------------------------
// GEN-2 — the MOSES generation-metrics lane (the roadmap's row 5:
// validity / uniqueness / novelty / the internal diversities + the
// property-shift table over the engine's own audit)
// ---------------------------------------------------------------------------

/** One property-shift row (MW / MolLogP / TPSA / QED). */
export interface PropertyShiftRow {
  readonly property: string;
  readonly generated_mean: number | null;
  readonly reference_mean: number | null;
  readonly shift: number | null;
}

/** `POST /api/v1/generative/metrics` — the MOSES-lite audit answer. */
export interface GenerationMetricsResult {
  readonly n_generated: number;
  readonly n_valid: number;
  readonly valid_fraction: number;
  readonly n_unique: number;
  readonly unique_fraction: number;
  readonly n_novel: number;
  readonly novel_fraction: number;
  readonly internal_diversity_1: number;
  readonly internal_diversity_2: number;
  readonly scaffold_diversity: number;
  readonly n_scaffolds: number;
  readonly pairwise_truncated: boolean;
  readonly property_shifts: readonly PropertyShiftRow[];
  readonly reference: { readonly source: string | null; readonly n: number };
}

/** Narrow the metrics answer field-by-field (malformed throws, the
 * honest nulls — the NaN reference means — stay null). */
export function parseGenerationMetricsResult(
  raw: unknown,
  what: string,
): GenerationMetricsResult {
  const r = expectObject(raw, what);
  const numeric: Array<keyof GenerationMetricsResult> = [
    "n_generated",
    "n_valid",
    "valid_fraction",
    "n_unique",
    "unique_fraction",
    "n_novel",
    "novel_fraction",
    "internal_diversity_1",
    "internal_diversity_2",
    "scaffold_diversity",
    "n_scaffolds",
  ];
  const counts: Record<string, number | null> = {};
  for (const key of numeric) {
    const value = asNumber(r[key]);
    if (value === null) {
      throw new Error(`${what}: ${String(key)} is not a number`);
    }
    counts[key] = value;
  }
  const shiftsRaw = r["property_shifts"];
  if (!Array.isArray(shiftsRaw)) {
    throw new Error(`${what}: property_shifts is not an array`);
  }
  const propertyShifts: PropertyShiftRow[] = shiftsRaw.map((row, i) => {
    const rr = expectObject(row, `${what} property_shifts[${i}]`);
    const property = rr["property"];
    if (typeof property !== "string") {
      throw new Error(
        `${what} property_shifts[${i}]: property is not a string`,
      );
    }
    return {
      property,
      generated_mean: asNumber(rr["generated_mean"]),
      reference_mean: asNumber(rr["reference_mean"]),
      shift: asNumber(rr["shift"]),
    };
  });
  const refRaw = r["reference"];
  const ref = expectObject(refRaw, `${what} reference`);
  const source = ref["source"];
  if (source !== null && typeof source !== "string") {
    throw new Error(`${what} reference.source is malformed`);
  }
  const n = asNumber(ref["n"]);
  if (n === null) {
    throw new Error(`${what} reference.n is not a number`);
  }
  return {
    n_generated: Math.round(counts["n_generated"] ?? 0),
    n_valid: Math.round(counts["n_valid"] ?? 0),
    valid_fraction: counts["valid_fraction"] ?? 0,
    n_unique: Math.round(counts["n_unique"] ?? 0),
    unique_fraction: counts["unique_fraction"] ?? 0,
    n_novel: Math.round(counts["n_novel"] ?? 0),
    novel_fraction: counts["novel_fraction"] ?? 0,
    internal_diversity_1: counts["internal_diversity_1"] ?? 0,
    internal_diversity_2: counts["internal_diversity_2"] ?? 0,
    scaffold_diversity: counts["scaffold_diversity"] ?? 0,
    n_scaffolds: Math.round(counts["n_scaffolds"] ?? 0),
    pairwise_truncated: r["pairwise_truncated"] === true,
    property_shifts: propertyShifts,
    reference: { source, n: Math.round(n) },
  };
}

/** The metrics request payload (the wire shape the backend's
 * `GenerationMetricsRequest` defines). */
export interface MetricsOptions {
  readonly generated: readonly string[];
  readonly reference: readonly string[] | null;
  readonly datasetId: string | null;
  readonly column: string | null;
}

export const generativeApi = {
  /**
   * `POST /api/v1/generative/metrics` — the MOSES-lite audit over a
   * generated SMILES list (GEN-2, the roadmap's row 5): the full
   * metrics dict + the property-shift table, novelty an honest 0.0
   * "not measured" without a reference (never a fake 1.0).
   */
  async metrics(options: MetricsOptions): Promise<GenerationMetricsResult> {
    const request: GenerationMetricsRequest = {
      generated: [...options.generated],
      reference: options.reference === null ? null : [...options.reference],
      dataset_id: options.datasetId,
      column: options.column,
    };
    const raw = (await api.post(
      "/api/v1/generative/metrics",
      request,
    )) as Record<string, unknown>;
    return parseGenerationMetricsResult(raw, "generative metrics result");
  },

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
      mutation: options.mutationMode,
      crossover: options.crossoverMode,
      min_heavy_atoms: options.minHeavyAtoms,
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
   * `POST /api/v1/generative/rl-optimize` — GEN-3 (the roadmap's
   * row 17): the REINVENT-style molecular RL optimizer over the REAL
   * engine (the policy-gradient fine-tuning of a SMILES RNN prior
   * against the scoring objective, with the prior-drift anchor and
   * the experience-replay buffer) — a managed job with per-step
   * telemetry and cancellation BETWEEN iterations, or sync.
   */
  async rlOptimize(
    options: RlOptimizeOptions,
  ): Promise<RlOptimizeResponse> {
    const request: Record<string, unknown> = {
      objective: options.objective,
      prior_id: options.priorId ?? undefined,
      smiles: options.smiles === null ? undefined : [...options.smiles],
      prior_epochs: options.priorEpochs,
      iterations: options.iterations,
      batch_size: options.batchSize,
      temperature: options.temperature,
      lr: options.lr,
      anchor_weight: options.anchorWeight,
      entropy_bonus: options.entropyBonus,
      replay_size: options.replaySize,
      patience: options.patience,
      max_smiles_length: options.maxSmilesLength,
      seed: options.seed,
      top_n: options.topN,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/generative/rl-optimize",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      const parsed = parseRlOptimizeView(raw, "rl-optimize(sync)");
      return parsed;
    }
    return {
      mode: "job",
      job: parseJobView(raw["job"], "rl-optimize(job)"),
    };
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
  /**
   * `POST /api/v1/generative/gan/train` — the REAL
   * `FeatureSpaceGAN` (part18 §5.6) over a registered dataset's
   * numeric feature matrix. FEATURE-space, not molecules — the view
   * says so and the sample lane carries the honest distributional
   * parity report. Job mode (the default) answers
   * `{mode: "job", job: JobView}`; the sync answer is
   * `{mode: "sync", prior: FeaturePriorView}`.
   */
  async ganTrain(options: GanTrainOptions): Promise<GanTrainResponse> {
    const request: GanTrainRequest = {
      dataset_id: options.datasetId,
      name: options.name,
      latent_dim: options.latentDim,
      hidden_layers: [...options.hiddenLayers],
      d_steps: options.dSteps,
      epochs: options.epochs,
      batch_size: options.batchSize,
      learning_rate: options.learningRate,
      weight_decay: options.weightDecay,
      random_state: options.randomState,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/generative/gan/train",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return { mode: "sync", prior: parseFeaturePriorView(raw, "gan train") };
    }
    return {
      mode: "job",
      job: parseJobView(raw["job"], "gan train"),
    };
  },

  /**
   * `POST /api/v1/generative/gan/sample` — synthetic feature rows off
   * a session feature GAN, with the parity report against the
   * training distribution (a REPORT, never a real-data claim).
   */
  async ganSample(options: FeatureSampleOptions): Promise<FeatureSampleResponse> {
    const request: GanSampleRequest = {
      prior_id: options.priorId,
      n: options.n,
      seed: options.seed ?? undefined,
    };
    const raw = (await api.post(
      "/api/v1/generative/gan/sample",
      request,
    )) as Record<string, unknown>;
    return parseFeatureSampleResult(raw, "gan sample");
  },

  /**
   * `POST /api/v1/generative/diffusion/train` — the REAL
   * `FeatureSpaceDiffusion` (part18 §5.6): the DDPM machinery (linear
   * schedule, noise prediction, the full reverse chain) over a
   * registered dataset's feature matrix. Same honest scope as the GAN
   * lane — feature rows, not molecules.
   */
  async diffusionTrain(
    options: DiffusionTrainOptions,
  ): Promise<DiffusionTrainResponse> {
    const request: DiffusionTrainRequest = {
      dataset_id: options.datasetId,
      name: options.name,
      timesteps: options.timesteps,
      hidden_layers: [...options.hiddenLayers],
      epochs: options.epochs,
      batch_size: options.batchSize,
      learning_rate: options.learningRate,
      weight_decay: options.weightDecay,
      random_state: options.randomState,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/generative/diffusion/train",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        prior: parseFeaturePriorView(raw, "diffusion train"),
      };
    }
    return {
      mode: "job",
      job: parseJobView(raw["job"], "diffusion train"),
    };
  },

  /**
   * `POST /api/v1/generative/diffusion/sample` — the reverse chain
   * run for `n` synthetic feature rows, with the parity report.
   */
  async diffusionSample(
    options: FeatureSampleOptions,
  ): Promise<FeatureSampleResponse> {
    const request: DiffusionSampleRequest = {
      prior_id: options.priorId,
      n: options.n,
      seed: options.seed ?? undefined,
    };
    const raw = (await api.post(
      "/api/v1/generative/diffusion/sample",
      request,
    )) as Record<string, unknown>;
    return parseFeatureSampleResult(raw, "diffusion sample");
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

/**
 * GEN-3 — narrow the rl-optimize answer: the optimization block (the
 * engine's own history rows with the drift diagnostics), the prior
 * view (null on the on-the-fly lane), and the note.  The history
 * rows' absent diagnostics stay null, never implied zeros.
 */
export function parseRlOptimizeView(
  raw: unknown,
  what: string,
): RlOptimizeResult {
  const r = expectObject(raw, what);
  const optimizationRaw = r["optimization"];
  if (typeof optimizationRaw !== "object" || optimizationRaw === null) {
    throw new Error(`${what}: the optimization block is missing`);
  }
  const optimization = expectObject(optimizationRaw, `${what} optimization`);
  const top = optimization["top"];
  const historyRaw = optimization["history"];
  const iterations = asNumber(optimization["iterations"]);
  const note = r["note"];
  if (iterations === null || typeof note !== "string") {
    throw new Error(
      `${what}: the optimization block lost its iterations/note`,
    );
  }
  const history: RlIterationRow[] = [];
  if (Array.isArray(historyRaw)) {
    for (const rowRaw of historyRaw as unknown[]) {
      const row = expectObject(rowRaw, `${what} history row`);
      history.push({
        iteration: asNumber(row["iteration"]) ?? 0,
        best_score: asNumber(row["best_score"]),
        mean_score: asNumber(row["mean_score"]),
        valid_fraction: asNumber(row["valid_fraction"]),
        drift: asNumber(row["drift"]),
        entropy: asNumber(row["entropy"]),
        loss: asNumber(row["loss"]),
      });
    }
  }
  const priorRaw = r["prior"];
  const prior =
    typeof priorRaw === "object" && priorRaw !== null
      ? parsePriorView(priorRaw, `${what} prior`)
      : null;
  return {
    mode: "sync",
    optimization: {
      best: typeof optimization["best"] === "string"
        ? optimization["best"]
        : null,
      best_score: asNumber(optimization["best_score"]),
      iterations,
      history,
      metadata:
        typeof optimization["metadata"] === "object" &&
        optimization["metadata"] !== null
          ? (optimization["metadata"] as Record<string, unknown>)
          : {},
      top: Array.isArray(top)
        ? top.map((row, i) =>
            parseOptimizationTopRow(row, `${what} top[${i}]`),
          )
        : [],
    },
    prior,
    note,
  };
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


// -- the feature-space GAN / Diffusion lanes (part18 §5.6) -------------------

type GanTrainRequest = components["schemas"]["GanTrainRequest"];
type GanSampleRequest = components["schemas"]["GanSampleRequest"];
type DiffusionTrainRequest = components["schemas"]["DiffusionTrainRequest"];
type DiffusionSampleRequest =
  components["schemas"]["DiffusionSampleRequest"];

/**
 * A trained FEATURE-space generative prior (the tabular analogue of
 * the SMILES RNN prior): a fitted FeatureSpaceGAN or
 * FeatureSpaceDiffusion with its per-epoch loss telemetry. The
 * `molecular: false` field pins the honest scope — these models
 * generate synthetic feature ROWS, never molecules.
 */
export interface FeaturePriorView {
  priorId: string;
  name: string;
  kind: "feature_gan" | "feature_diffusion" | string;
  params: Record<string, unknown>;
  nTrainRows: number;
  nEpochs: number;
  finalLoss: number | null;
  finalDLoss: number | null;
  losses: readonly (number | null)[];
  datasetSource: string;
  createdAt: string;
  sampleUrl: string | null;
  molecular: boolean;
  note: string | null;
}

/** The feature GAN's train options. */
export interface GanTrainOptions {
  datasetId: string;
  name: string;
  latentDim: number;
  hiddenLayers: readonly number[];
  dSteps: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  weightDecay: number;
  randomState: number;
  mode?: "job" | "sync";
}

export interface GanJobResponse {
  mode: "job";
  job: JobView;
}

export interface GanTrainSyncResponse {
  mode: "sync";
  prior: FeaturePriorView;
}

export type GanTrainResponse = GanJobResponse | GanTrainSyncResponse;

/** The DDPM's train options. */
export interface DiffusionTrainOptions {
  datasetId: string;
  name: string;
  timesteps: number;
  hiddenLayers: readonly number[];
  epochs: number;
  batchSize: number;
  learningRate: number;
  weightDecay: number;
  randomState: number;
  mode?: "job" | "sync";
}

export interface DiffusionJobResponse {
  mode: "job";
  job: JobView;
}

export interface DiffusionTrainSyncResponse {
  mode: "sync";
  prior: FeaturePriorView;
}

export type DiffusionTrainResponse =
  | DiffusionJobResponse
  | DiffusionTrainSyncResponse;

/** The shared sampler options (one path per model kind). */
export interface FeatureSampleOptions {
  priorId: string;
  n: number;
  seed?: number;
}

/** The honest distributional parity report (a REPORT, not real data). */
export interface FeatureQualityReport {
  nRef: number;
  meanDelta: readonly number[];
  stdDelta: readonly number[];
  corrL1: number;
}

/** The shared sample result body. */
export interface FeatureSampleResponse {
  priorId: string;
  nRequested: number;
  nSampled: number;
  rows: readonly (readonly number[])[];
  quality: FeatureQualityReport;
  seed: number | null;
  note: string;
}

/** Parse + validate a feature-space prior view (§8.1). */
export function parseFeaturePriorView(
  raw: unknown,
  what: string,
): FeaturePriorView {
  const r = expectObject(raw, what);
  const priorId = r["prior_id"];
  const name = r["name"];
  if (typeof priorId !== "string" || typeof name !== "string") {
    throw new Error(`${what}: prior_id/name are not strings`);
  }
  const params = r["params"];
  const losses = r["losses"];
  return {
    priorId,
    name,
    kind: typeof r["kind"] === "string" ? r["kind"] : "",
    params:
      typeof params === "object" && params !== null
        ? (params as Record<string, unknown>)
        : {},
    nTrainRows: asNumber(r["n_train_rows"]) ?? 0,
    nEpochs: asNumber(r["n_epochs"]) ?? 0,
    finalLoss: asNumber(r["final_loss"]),
    finalDLoss: asNumber(r["final_d_loss"]),
    losses: Array.isArray(losses)
      ? losses.map((v) => asNumber(v))
      : [],
    datasetSource:
      typeof r["dataset_source"] === "string" ? r["dataset_source"] : "",
    createdAt: typeof r["created_at"] === "string" ? r["created_at"] : "",
    sampleUrl: typeof r["sample_url"] === "string" ? r["sample_url"] : null,
    molecular: r["molecular"] === true,
    note: typeof r["note"] === "string" ? r["note"] : null,
  };
}

/** Parse + validate the shared feature-sample result body. */
export function parseFeatureSampleResult(
  raw: unknown,
  what: string,
): FeatureSampleResponse {
  const r = expectObject(raw, what);
  const rowsRaw = r["rows"];
  const rows = Array.isArray(rowsRaw)
    ? rowsRaw.map((row: unknown) =>
        Array.isArray(row)
          ? row.filter((v): v is number => typeof v === "number")
          : [],
      )
    : [];
  const qualityRaw = r["quality"];
  const quality: FeatureQualityReport = {
    nRef: 0,
    meanDelta: [],
    stdDelta: [],
    corrL1: Number.NaN,
  };
  if (typeof qualityRaw === "object" && qualityRaw !== null) {
    const q = qualityRaw as Record<string, unknown>;
    quality.nRef = asNumber(q["n_ref"]) ?? 0;
    quality.meanDelta = Array.isArray(q["mean_delta"])
      ? q["mean_delta"].filter(
          (v): v is number => typeof v === "number",
        )
      : [];
    quality.stdDelta = Array.isArray(q["std_delta"])
      ? q["std_delta"].filter(
          (v): v is number => typeof v === "number",
        )
      : [];
    quality.corrL1 = asNumber(q["corr_l1"]) ?? Number.NaN;
  }
  return {
    priorId: typeof r["prior_id"] === "string" ? r["prior_id"] : "",
    nRequested: asNumber(r["n_requested"]) ?? 0,
    nSampled: asNumber(r["n_sampled"]) ?? 0,
    rows,
    quality,
    seed: asNumber(r["params"] === undefined ? null : (r["params"] as Record<string, unknown>)["seed"]),
    note: typeof r["note"] === "string" ? r["note"] : "",
  };
}
/**
 * Narrow the feature-prior train job's result payload:
 * `{prior: FeaturePriorView}` (the authoritative REST relay — the
 * envelope-extracting sibling of `parsePriorTrainResult`).
 */
export function parseFeaturePriorTrainResult(
  raw: unknown,
  what: string,
): FeaturePriorView {
  const r = expectObject(raw, what);
  return parseFeaturePriorView(r["prior"], `${what} prior`);
}
