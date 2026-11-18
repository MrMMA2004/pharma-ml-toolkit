/**
 * The job REST client — part11 §13 ("REST status queries must remain
 * available") on the generated-type foundation (§12.2 / §5.2).
 *
 * Five operations, exactly the backend's `/api/v1/jobs` surface:
 * list (optionally by workspace), the authoritative status fetch, the
 * completed-result fetch (409 until COMPLETED) and cancellation.
 * Every response is narrowed through `parseJobView` — an unexpected
 * backend shape surfaces as `ApiSchemaError`, not as silent `any`.
 */

import { api } from "@/services/api/client";
import {
  ApiSchemaError,
  type JobCancelledView,
  type JobListResponse,
  type JobResultResponse,
  type JobStatusView,
  type JobView,
  parseJobView,
} from "./types";

export const jobsApi = {
  /** `GET /api/v1/jobs?workspace=…` — the session's job ledger. */
  async list(workspace?: string): Promise<JobListResponse> {
    const path =
      workspace === undefined
        ? "/api/v1/jobs"
        : `/api/v1/jobs?workspace=${encodeURIComponent(workspace)}`;
    const raw = await api.get<Record<string, unknown>>(path);
    const items = raw["jobs"];
    if (!Array.isArray(items)) {
      return { jobs: [] };
    }
    return {
      jobs: items.map((item, i) => parseJobView(item, `jobs[${i}]`)),
    };
  },

  /** `GET /api/v1/jobs/{id}` — the authoritative status path (§13). */
  async status(jobId: string): Promise<JobStatusView> {
    const raw = await api.get<Record<string, unknown>>(
      `/api/v1/jobs/${encodeURIComponent(jobId)}`,
    );
    const view = parseJobView(raw, `job ${jobId}`);
    return { ...view, result_ready: raw["result_ready"] === true };
  },

  /** `GET /api/v1/jobs/{id}/result` — 409 until the job COMPLETED. */
  async result(jobId: string): Promise<JobResultResponse> {
    const raw = await api.get<Record<string, unknown>>(
      `/api/v1/jobs/${encodeURIComponent(jobId)}/result`,
    );
    if (raw["job_id"] !== jobId) {
      throw new Error(`result: job_id mismatch for ${jobId}`);
    }
    return { job_id: jobId, result: raw["result"] };
  },

  /** `POST /api/v1/jobs/{id}/cancel` — real cooperative cancellation. */
  async cancel(jobId: string): Promise<JobCancelledView> {
    const raw = await api.post<Record<string, unknown>>(
      `/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`,
    );
    const view = parseJobView(raw, `cancel ${jobId}`);
    if (raw["cancel_acknowledged"] !== true) {
      throw new ApiSchemaError(
        `cancel ${jobId}: the backend did not acknowledge the cancellation`,
      );
    }
    return { ...view, cancel_acknowledged: true };
  },

  /** Parse one submission answer (`{mode: "job", job: …}`). */
  parseSubmission(raw: Record<string, unknown>): JobView {
    return parseJobView(raw["job"], "submission");
  },
};
