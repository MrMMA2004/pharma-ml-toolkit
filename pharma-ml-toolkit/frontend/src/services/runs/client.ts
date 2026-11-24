/**
 * The experiment-run REST client — part16 §15.2 (the Phase-12 lane).
 *
 * Two GET operations, exactly the backend's `/api/v1/runs` surface:
 * the lineage table (filterable summaries, newest first) and the full
 * §15.2 record per run. Read-only by design — runs are recorded at the
 * training points (`routes/ml.py`), never hand-POSTed; the record is
 * the system's, not the user's claim. Every response is narrowed
 * through the parse guards — an unexpected backend shape surfaces as
 * `ApiSchemaError`, not as silent `any`.
 */

import { api } from "@/services/api/client";
import {
  parseExperimentRun,
  parseRunSummary,
  type ExperimentRun,
  type RunListResponse,
  type RunStatus,
} from "./types";

export type {
  ExperimentRun,
  RunListResponse,
  RunStatus,
  RunSummary,
  RunDatasetBlock,
  RunFeatureConfigurationBlock,
  RunPreprocessingBlock,
  RunAlgorithmBlock,
  RunHyperparametersBlock,
  RunParallelismAdvisoryBlock,
  RunCrossValidationBlock,
  RunEvaluationBlock,
  RunModelArtifactBlock,
  RunEnvironmentBlock,
  RunLogsBlock,
  RunMode,
} from "./types";

/** The lineage-table filters (the route's own query parameters). */
export interface RunsListParams {
  /** The run family (e.g. "ml.train"). */
  readonly family?: string;
  /** Filter to one dataset's runs. */
  readonly datasetId?: string;
  /** Filter to one model name. */
  readonly model?: string;
  /** Filter to one of the three honest states. */
  readonly status?: RunStatus;
  /** The listing bound (the backend refuses outside 1..200). */
  readonly limit?: number;
}

export const runsApi = {
  /** `GET /api/v1/runs` — the lineage table (newest first). */
  async list(params?: RunsListParams): Promise<RunListResponse> {
    const parts: string[] = [];
    if (params?.family !== undefined) {
      parts.push(`family=${encodeURIComponent(params.family)}`);
    }
    if (params?.datasetId !== undefined) {
      parts.push(`dataset_id=${encodeURIComponent(params.datasetId)}`);
    }
    if (params?.model !== undefined) {
      parts.push(`model=${encodeURIComponent(params.model)}`);
    }
    if (params?.status !== undefined) {
      parts.push(`status=${encodeURIComponent(params.status)}`);
    }
    if (params?.limit !== undefined) {
      parts.push(`limit=${encodeURIComponent(String(params.limit))}`);
    }
    const path =
      parts.length === 0 ? "/api/v1/runs" : `/api/v1/runs?${parts.join("&")}`;
    const raw = await api.get<Record<string, unknown>>(path);
    const items = raw["runs"];
    if (!Array.isArray(items)) {
      return { runs: [], count: 0 };
    }
    const runs = items.map((item, i) => parseRunSummary(item, `runs[${i}]`));
    const count = typeof raw["count"] === "number" ? raw["count"] : runs.length;
    return { runs, count };
  },

  /** `GET /api/v1/runs/{run_id}` — the full §15.2 record. */
  async detail(runId: string): Promise<ExperimentRun> {
    const raw = await api.get<Record<string, unknown>>(
      `/api/v1/runs/${encodeURIComponent(runId)}`,
    );
    return parseExperimentRun(raw, `run ${runId}`);
  },
};
