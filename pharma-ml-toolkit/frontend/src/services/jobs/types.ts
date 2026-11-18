/**
 * The job DTOs — part11 §13/§18 (Phase Three services, part two).
 *
 * Field-for-field mirrors of the backend's `JobRecord.public()`
 * (`pharmaml.api.jobs`) plus the route-added flags — the Python drift
 * guards in `tests/test_frontend_services.py` pin the vocabulary, so
 * a backend rename turns this suite red before the frontend drifts.
 */

/** The JobManager's state machine (mirrors `api.jobs.JobState`). */
export const JOB_STATES = [
  "QUEUED",
  "RUNNING",
  "CANCELLATION_REQUESTED",
  "CANCELLED",
  "COMPLETED",
  "FAILED",
] as const;

export type JobState = (typeof JOB_STATES)[number];

/** Terminal states: no further transitions are possible. */
export function isTerminalJobState(state: JobState): boolean {
  return state === "COMPLETED" || state === "CANCELLED" || state === "FAILED";
}

/** The REST-facing view of one managed job (`JobRecord.public()`). */
export interface JobView {
  readonly job_id: string;
  readonly operation: string;
  readonly workspace: string;
  readonly state: JobState;
  readonly created_at: string;
  readonly started_at: string | null;
  readonly completed_at: string | null;
  readonly progress: number;
  readonly stage: string;
  readonly configuration: Readonly<Record<string, unknown>>;
  readonly cancellation: string;
  readonly error: string | null;
}

/** `GET /api/v1/jobs/{id}` — the authoritative status (adds the flag). */
export interface JobStatusView extends JobView {
  readonly result_ready: boolean;
}

/** `POST /api/v1/jobs/{id}/cancel` — the acknowledgement wrapper. */
export interface JobCancelledView extends JobView {
  readonly cancel_acknowledged: true;
}

/** `GET /api/v1/jobs` */
export interface JobListResponse {
  readonly jobs: readonly JobView[];
}

/** `GET /api/v1/jobs/{id}/result` (409 while not COMPLETED). */
export interface JobResultResponse {
  readonly job_id: string;
  readonly result: unknown;
}

/** The job-submission answer used across the domain routes. */
export interface JobSubmissionResponse {
  readonly mode: "job";
  readonly job: JobView;
}

/** Thrown when a response cannot be narrowed to the wire DTO (§8.1). */
export class ApiSchemaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiSchemaError";
  }
}

/**
 * Validate an untrusted payload as a JobView (§8.1: validate API
 * responses where appropriate — the state string is checked against
 * the drifted vocabulary, the invariant fields against their types).
 */
export function parseJobView(raw: unknown, what: string): JobView {
  if (typeof raw !== "object" || raw === null) {
    throw new ApiSchemaError(`${what}: expected an object`);
  }
  const r = raw as Record<string, unknown>;
  const required: Array<readonly [string, "string" | "number"]> = [
    ["job_id", "string"],
    ["operation", "string"],
    ["workspace", "string"],
    ["state", "string"],
    ["created_at", "string"],
    ["progress", "number"],
    ["stage", "string"],
    ["cancellation", "string"],
  ];
  for (const [field, kind] of required) {
    if (typeof r[field] !== kind) {
      throw new ApiSchemaError(`${what}: field ${field} is not a ${kind}`);
    }
  }
  if (!(JOB_STATES as readonly string[]).includes(r["state"] as string)) {
    throw new ApiSchemaError(
      `${what}: unknown job state ${JSON.stringify(r["state"])}`,
    );
  }
  return {
    job_id: r["job_id"] as string,
    operation: r["operation"] as string,
    workspace: r["workspace"] as string,
    state: r["state"] as JobState,
    created_at: r["created_at"] as string,
    started_at: (r["started_at"] as string | null) ?? null,
    completed_at: (r["completed_at"] as string | null) ?? null,
    progress: r["progress"] as number,
    stage: r["stage"] as string,
    configuration: (r["configuration"] as Record<string, unknown>) ?? {},
    cancellation: r["cancellation"] as string,
    error: (r["error"] as string | null) ?? null,
  };
}
