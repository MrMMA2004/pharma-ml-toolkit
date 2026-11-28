/**
 * The part18 §4.3/§4.5/§6 lanes' API clients — the unsupervised
 * clustering families (DBSCAN / mean shift / spectral), Isolation
 * Forest anomaly detection, and the EXPERIMENTAL reinforcement-
 * learning lane over `routes/unsupervised.py` / `routes/reinforce.py`.
 *
 * The DTO surface comes from the generated contract
 * (`src/types/api.d.ts`, §12.2's one DTO source — no hand duplicates);
 * the runtime parsers narrow every answer with the same
 * expect-object/typed-field discipline the ml client uses, so a
 * malformed payload fails loudly at the parse boundary instead of
 * rendering as a blank card.
 */

import { api } from "@/services/api/client";
import { type JobView, parseJobView } from "@/services/jobs/types";

// ---------------------------------------------------------------------------
// shared narrowing helpers (the ml client's own discipline, local copy)
// ---------------------------------------------------------------------------
function expectObject(raw: unknown, what: string): Record<string, unknown> {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error(`${what}: expected an object`);
  }
  return raw as Record<string, unknown>;
}

function asNumber(raw: unknown): number | null {
  return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
}

function asInt(raw: unknown): number | null {
  const value = asNumber(raw);
  return value === null ? null : Math.round(value);
}

function asString(raw: unknown): string | null {
  return typeof raw === "string" ? raw : null;
}

function asBoolean(raw: unknown): boolean {
  return raw === true;
}

// ---------------------------------------------------------------------------
// the unsupervised views
// ---------------------------------------------------------------------------

/** `_diagnostics_view` — the shared clustering metric block. */
export interface ClusterDiagnosticsView {
  readonly n_clusters: number;
  readonly silhouette: number | null;
  readonly davies_bouldin: number | null;
  readonly calinski_harabasz: number | null;
  readonly noise_count: number;
  readonly noise_ratio: number | null;
  readonly cluster_sizes: Readonly<Record<string, number>>;
}

/** The clustering answer's `unsupervised` view (family-specific keys). */
export interface ClusterResultView {
  readonly method: string;
  readonly dataset_id: string;
  readonly n_samples: number;
  readonly diagnostics: ClusterDiagnosticsView;
  readonly labels: {
    readonly n: number;
    readonly preview: readonly number[];
    readonly unique: readonly number[];
  };
  readonly summary: string;
  // DBSCAN
  readonly eps: number | null;
  readonly eps_source: string | null;
  readonly min_samples: number | null;
  // mean shift
  readonly bandwidth: number | null;
  readonly bandwidth_source: string | null;
  readonly n_modes: number | null;
  // spectral
  readonly n_clusters: number | null;
  readonly affinity: string | null;
  readonly gamma: number | null;
  readonly n_neighbors: number | null;
  // k-means (ML-4)
  readonly k: number | null;
  readonly k_source: string | null;
  readonly kmeans_method: string | null;
  readonly inertia: number | null;
  readonly kmeans_silhouette: number | null;
  readonly scan: readonly KmeansScanRow[];
  readonly sizes: Readonly<Record<string, number>> | null;
  // GMM (ML-4)
  readonly n_components: number | null;
  readonly covariance_type: string | null;
  readonly chosen_by: string | null;
  readonly criterion: string | null;
  readonly converged: boolean;
  readonly log_likelihood: number | null;
  readonly weights: Readonly<Record<string, number>> | null;
  readonly bic_table: readonly GmmBicRow[];
  // hierarchical (ML-4)
  readonly linkage: string | null;
  readonly merge_heights: readonly number[];
}

/** One k-means scan row (the elbow + silhouette sweep the engine ran). */
export interface KmeansScanRow {
  readonly k: number;
  readonly inertia: number | null;
  readonly silhouette: number | null;
}

/** One GMM sweep row (the AIC/BIC model-selection table). */
export interface GmmBicRow {
  readonly n_components: number;
  readonly aic: number | null;
  readonly bic: number | null;
}

/** The anomaly answer's `anomaly` view. */
export interface AnomalyResultView {
  readonly method: string;
  readonly dataset_id: string;
  readonly contamination: string;
  readonly n_estimators: number;
  readonly seed: number;
  readonly n_samples: number;
  readonly n_features: number;
  readonly n_outliers: number;
  readonly anomaly_fraction: number | null;
  readonly score_threshold: number | null;
  readonly outlier_positions: readonly number[];
  readonly scores_preview: readonly number[];
  readonly summary: string;
  readonly honesty_note: string;
  readonly feature_report: readonly {
    readonly feature: string;
    readonly inlier_mean: number | null;
    readonly outlier_mean: number | null;
    readonly shift_sigma: number | null;
  }[];
}

/**
 * The PCA answer's `pca` view (ML-4's sibling route over the real
 * `pca_reduce` engine): the explained-variance ladder, the loadings
 * preview (the "what does PC1 mean" table), and the scores preview.
 */
export interface PcaResultView {
  readonly method: string;
  readonly dataset_id: string;
  readonly n_components: number;
  readonly scaled: boolean;
  readonly n_samples: number;
  readonly n_features: number;
  readonly explained_variance_ratio: readonly number[];
  readonly cumulative_variance: readonly number[];
  readonly total_variance_explained: number | null;
  readonly column_names: readonly string[];
  readonly loadings_preview: readonly Record<string, unknown>[];
  readonly scores_preview: readonly Record<string, unknown>[];
  readonly summary: string;
  readonly honesty_note: string;
}

// ---------------------------------------------------------------------------
// the reinforcement views
// ---------------------------------------------------------------------------

/** `_run_view` — the RL training answer's `reinforcement` view. */
export interface ReinforceRunView {
  readonly method: string;
  readonly env: string;
  readonly success_rate: number | null;
  readonly reward_curve: {
    readonly episodes: number;
    readonly first_20_mean: number | null;
    readonly last_20_mean: number | null;
    readonly max: number | null;
    readonly min: number | null;
  };
  readonly convergence_episode: number | null;
  readonly hyperparams: Readonly<Record<string, unknown>>;
  readonly policy: Readonly<Record<string, number>>;
  readonly value_table_preview: readonly number[];
  readonly notes: readonly string[];
  readonly experimental: boolean;
}

/** The evaluate answer's `evaluation` view. */
export interface ReinforceEvaluationView {
  readonly method: string;
  readonly env: string;
  readonly episodes: number;
  readonly mean_reward: number | null;
  readonly termination_rate: number | null;
  readonly experimental: boolean;
}

// ---------------------------------------------------------------------------
// the parsers
// ---------------------------------------------------------------------------

function parseDiagnostics(raw: unknown, what: string): ClusterDiagnosticsView {
  const r = expectObject(raw, what);
  const sizesRaw = r["cluster_sizes"];
  const sizes: Record<string, number> = {};
  if (typeof sizesRaw === "object" && sizesRaw !== null) {
    for (const [key, value] of Object.entries(sizesRaw as Record<string, unknown>)) {
      const count = asInt(value);
      if (count !== null) {
        sizes[key] = count;
      }
    }
  }
  const nClusters = asInt(r["n_clusters"]);
  if (nClusters === null) {
    throw new Error(`${what}: diagnostics.n_clusters is not a number`);
  }
  return {
    n_clusters: nClusters,
    silhouette: asNumber(r["silhouette"]),
    davies_bouldin: asNumber(r["davies_bouldin"]),
    calinski_harabasz: asNumber(r["calinski_harabasz"]),
    noise_count: asInt(r["noise_count"]) ?? 0,
    noise_ratio: asNumber(r["noise_ratio"]),
    cluster_sizes: sizes,
  };
}

function numberList(raw: unknown): number[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw
    .map((value) => asNumber(value))
    .filter((value): value is number => value !== null);
}

function parseClusterResult(raw: unknown, what: string): ClusterResultView {
  const r = expectObject(raw, `${what} (unsupervised view)`);
  const method = asString(r["method"]);
  const datasetId = asString(r["dataset_id"]);
  const nSamples = asInt(r["n_samples"]);
  const summary = asString(r["summary"]);
  if (method === null || datasetId === null || nSamples === null || summary === null) {
    throw new Error(`${what}: method/dataset_id/n_samples/summary have wrong shapes`);
  }
  const labelsRaw = expectObject(r["labels"], `${what} labels`);
  // ML-4: the k-means elbow/silhouette sweep + the GMM AIC/BIC table
  const scanRaw = r["scan"];
  const scan: KmeansScanRow[] = [];
  if (Array.isArray(scanRaw)) {
    for (const rowRaw of scanRaw) {
      const row = expectObject(rowRaw, `${what} scan row`);
      const k = asInt(row["k"]);
      if (k === null) {
        continue;
      }
      scan.push({
        k,
        inertia: asNumber(row["inertia"]),
        silhouette: asNumber(row["silhouette"]),
      });
    }
  }
  const bicRaw = r["bic_table"];
  const bicTable: GmmBicRow[] = [];
  if (Array.isArray(bicRaw)) {
    for (const rowRaw of bicRaw) {
      const row = expectObject(rowRaw, `${what} bic row`);
      const n = asInt(row["n_components"]);
      if (n === null) {
        continue;
      }
      bicTable.push({
        n_components: n,
        aic: asNumber(row["aic"]),
        bic: asNumber(row["bic"]),
      });
    }
  }
  const numberRecord = (key: string): Record<string, number> | null => {
    const dict = r[key];
    if (typeof dict !== "object" || dict === null) {
      return null;
    }
    const clean: Record<string, number> = {};
    for (const [name, value] of Object.entries(dict as Record<string, unknown>)) {
      const numeric = asNumber(value);
      if (numeric !== null) {
        clean[name] = numeric;
      }
    }
    return clean;
  };
  return {
    method,
    dataset_id: datasetId,
    n_samples: nSamples,
    diagnostics: parseDiagnostics(r["diagnostics"], what),
    labels: {
      n: asInt(labelsRaw["n"]) ?? 0,
      preview: numberList(labelsRaw["preview"]),
      unique: numberList(labelsRaw["unique"]),
    },
    summary,
    eps: asNumber(r["eps"]),
    eps_source: asString(r["eps_source"]),
    min_samples: asInt(r["min_samples"]),
    bandwidth: asNumber(r["bandwidth"]),
    bandwidth_source: asString(r["bandwidth_source"]),
    n_modes: asInt(r["n_modes"]),
    n_clusters: asInt(r["n_clusters"]),
    affinity: asString(r["affinity"]),
    gamma: asNumber(r["gamma"]),
    n_neighbors: asInt(r["n_neighbors"]),
    k: asInt(r["k"]),
    k_source: asString(r["k_source"]),
    kmeans_method: asString(r["kmeans_method"]),
    inertia: asNumber(r["inertia"]),
    kmeans_silhouette: asNumber(r["kmeans_silhouette"]),
    scan,
    sizes: numberRecord("sizes"),
    n_components: asInt(r["n_components"]),
    covariance_type: asString(r["covariance_type"]),
    chosen_by: asString(r["chosen_by"]),
    criterion: asString(r["criterion"]),
    converged: asBoolean(r["converged"]),
    log_likelihood: asNumber(r["log_likelihood"]),
    weights: numberRecord("weights"),
    bic_table: bicTable,
    linkage: asString(r["linkage"]),
    merge_heights: numberList(r["merge_heights"]),
  };
}

function parsePcaResult(raw: unknown, what: string): PcaResultView {
  const r = expectObject(raw, `${what} (pca view)`);
  const method = asString(r["method"]);
  const datasetId = asString(r["dataset_id"]);
  const nComponents = asInt(r["n_components"]);
  if (method === null || datasetId === null || nComponents === null) {
    throw new Error(`${what}: method/dataset_id/n_components have wrong shapes`);
  }
  const rows = (key: string): Record<string, unknown>[] => {
    const list = r[key];
    if (!Array.isArray(list)) {
      return [];
    }
    return list.filter(
      (row): row is Record<string, unknown> =>
        typeof row === "object" && row !== null && !Array.isArray(row),
    );
  };
  return {
    method,
    dataset_id: datasetId,
    n_components: nComponents,
    scaled: asBoolean(r["scaled"]),
    n_samples: asInt(r["n_samples"]) ?? 0,
    n_features: asInt(r["n_features"]) ?? 0,
    explained_variance_ratio: numberList(r["explained_variance_ratio"]),
    cumulative_variance: numberList(r["cumulative_variance"]),
    total_variance_explained: asNumber(r["total_variance_explained"]),
    column_names: Array.isArray(r["column_names"])
      ? (r["column_names"] as unknown[]).filter(
          (name): name is string => typeof name === "string",
        )
      : [],
    loadings_preview: rows("loadings_preview"),
    scores_preview: rows("scores_preview"),
    summary: asString(r["summary"]) ?? "",
    honesty_note: asString(r["honesty_note"]) ?? "",
  };
}

function parseAnomalyResult(raw: unknown, what: string): AnomalyResultView {
  const r = expectObject(raw, `${what} (anomaly view)`);
  const method = asString(r["method"]);
  const datasetId = asString(r["dataset_id"]);
  const nOutliers = asInt(r["n_outliers"]);
  if (method === null || datasetId === null || nOutliers === null) {
    throw new Error(`${what}: method/dataset_id/n_outliers have wrong shapes`);
  }
  const featureReportRaw = r["feature_report"];
  const featureReport: {
    feature: string;
    inlier_mean: number | null;
    outlier_mean: number | null;
    shift_sigma: number | null;
  }[] = [];
  if (Array.isArray(featureReportRaw)) {
    for (const rowRaw of featureReportRaw) {
      const row = expectObject(rowRaw, `${what} feature_report row`);
      const feature = asString(row["feature"]);
      if (feature === null) {
        continue;
      }
      featureReport.push({
        feature,
        inlier_mean: asNumber(row["inlier_mean"]),
        outlier_mean: asNumber(row["outlier_mean"]),
        shift_sigma: asNumber(row["shift_sigma"]),
      });
    }
  }
  return {
    method,
    dataset_id: datasetId,
    contamination: asString(r["contamination"]) ?? "auto",
    n_estimators: asInt(r["n_estimators"]) ?? 0,
    seed: asInt(r["seed"]) ?? 0,
    n_samples: asInt(r["n_samples"]) ?? 0,
    n_features: asInt(r["n_features"]) ?? 0,
    n_outliers: nOutliers,
    anomaly_fraction: asNumber(r["anomaly_fraction"]),
    score_threshold: asNumber(r["score_threshold"]),
    outlier_positions: numberList(r["outlier_positions"]),
    scores_preview: numberList(r["scores_preview"]),
    summary: asString(r["summary"]) ?? "",
    honesty_note: asString(r["honesty_note"]) ?? "",
    feature_report: featureReport,
  };
}

function parseReinforceRun(raw: unknown, what: string): ReinforceRunView {
  const r = expectObject(raw, `${what} (reinforcement view)`);
  const method = asString(r["method"]);
  if (method === null) {
    throw new Error(`${what}: method is not a string`);
  }
  const curveRaw = expectObject(r["reward_curve"], `${what} reward_curve`);
  const policyRaw = r["policy"];
  const policy: Record<string, number> = {};
  if (typeof policyRaw === "object" && policyRaw !== null) {
    for (const [key, value] of Object.entries(policyRaw as Record<string, unknown>)) {
      const action = asInt(value);
      if (action !== null) {
        policy[key] = action;
      }
    }
  }
  const notesRaw = r["notes"];
  const hyperparamsRaw = r["hyperparams"];
  return {
    method,
    env: asString(r["env"]) ?? "",
    success_rate: asNumber(r["success_rate"]),
    reward_curve: {
      episodes: asInt(curveRaw["episodes"]) ?? 0,
      first_20_mean: asNumber(curveRaw["first_20_mean"]),
      last_20_mean: asNumber(curveRaw["last_20_mean"]),
      max: asNumber(curveRaw["max"]),
      min: asNumber(curveRaw["min"]),
    },
    convergence_episode: asInt(r["convergence_episode"]),
    hyperparams:
      typeof hyperparamsRaw === "object" && hyperparamsRaw !== null
        ? hyperparamsRaw as Record<string, unknown>
        : {},
    policy,
    value_table_preview: numberList(r["value_table_preview"]),
    notes: Array.isArray(notesRaw) ? notesRaw.filter((n): n is string => typeof n === "string") : [],
    experimental: asBoolean(r["experimental"]),
  };
}

function parseReinforceEvaluation(
  raw: unknown,
  what: string,
): ReinforceEvaluationView {
  const r = expectObject(raw, `${what} (evaluation view)`);
  const method = asString(r["method"]);
  if (method === null) {
    throw new Error(`${what}: method is not a string`);
  }
  return {
    method,
    env: asString(r["env"]) ?? "",
    episodes: asInt(r["episodes"]) ?? 0,
    mean_reward: asNumber(r["mean_reward"]),
    termination_rate: asNumber(r["termination_rate"]),
    experimental: asBoolean(r["experimental"]),
  };
}

// ---------------------------------------------------------------------------
// the clients
// ---------------------------------------------------------------------------

export const unsupervisedApi = {
  /** `GET /api/v1/unsupervised/methods` — the capability listing. */
  async methods(): Promise<{ clustering: readonly string[] }> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/unsupervised/methods",
    );
    const clustering: string[] = [];
    const rawClustering = raw["clustering"];
    if (Array.isArray(rawClustering)) {
      for (const entry of rawClustering) {
        const obj = expectObject(entry, "methods clustering entry");
        const method = asString(obj["method"]);
        if (method !== null) {
          clustering.push(method);
        }
      }
    }
    return { clustering };
  },

  /** `POST /api/v1/unsupervised/cluster` — the clustering families. */
  async cluster(
    body: Record<string, unknown>,
  ): Promise<{ job?: JobView; result?: ClusterResultView }> {
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/unsupervised/cluster",
      body,
    );
    if (raw["mode"] === "job") {
      return { job: parseJobView(raw["job"], "cluster job") };
    }
    return { result: parseClusterResult(raw["unsupervised"], "cluster") };
  },

  /** `POST /api/v1/unsupervised/anomaly` — the Isolation Forest lane. */
  async anomaly(
    body: Record<string, unknown>,
  ): Promise<{ job?: JobView; result?: AnomalyResultView }> {
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/unsupervised/anomaly",
      body,
    );
    if (raw["mode"] === "job") {
      return { job: parseJobView(raw["job"], "anomaly job") };
    }
    return { result: parseAnomalyResult(raw["anomaly"], "anomaly") };
  },

  /**
   * `POST /api/v1/unsupervised/pca` — ML-4's PCA sibling over the
   * real `pca_reduce` engine (the explained-variance ladder + the
   * loadings preview).
   */
  async pca(
    body: Record<string, unknown>,
  ): Promise<{ job?: JobView; result?: PcaResultView }> {
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/unsupervised/pca",
      body,
    );
    if (raw["mode"] === "job") {
      return { job: parseJobView(raw["job"], "pca job") };
    }
    return { result: parsePcaResult(raw["pca"], "pca") };
  },
};

export const reinforceApi = {
  /** `GET /api/v1/reinforce/methods` — the experimental listing. */
  async methods(): Promise<{ experimental: boolean; agents: readonly string[] }> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/reinforce/methods",
    );
    const agents: string[] = [];
    const rawAgents = raw["agents"];
    if (Array.isArray(rawAgents)) {
      for (const entry of rawAgents) {
        const obj = expectObject(entry, "reinforce methods entry");
        const method = asString(obj["method"]);
        if (method !== null) {
          agents.push(method);
        }
      }
    }
    return { experimental: asBoolean(raw["status"] === undefined ? true : raw["status"] === "experimental"), agents };
  },

  /** `POST /api/v1/reinforce/train` — the experimental training lane. */
  async train(
    body: Record<string, unknown>,
  ): Promise<{ job?: JobView; result?: ReinforceRunView }> {
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/reinforce/train",
      body,
    );
    if (raw["mode"] === "job") {
      return { job: parseJobView(raw["job"], "reinforce job") };
    }
    return { result: parseReinforceRun(raw["reinforcement"], "reinforce") };
  },

  /** `POST /api/v1/reinforce/evaluate` — the greedy-policy rollouts. */
  async evaluate(
    body: Record<string, unknown>,
  ): Promise<ReinforceEvaluationView> {
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/reinforce/evaluate",
      body,
    );
    return parseReinforceEvaluation(raw["evaluation"], "reinforce evaluate");
  },
};
