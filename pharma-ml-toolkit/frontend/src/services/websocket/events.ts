/**
 * The typed WebSocket event envelope — part11 §13.
 *
 * The single wire DTO mirrors the backend's
 * `pharmaml.api.events.EventEnvelope` exactly (field-for-field,
 * including the 20-event vocabulary below). The REST request/response
 * types come from the OpenAPI contract generation
 * (`npm run generate:types` — one DTO source, §12.2); the WS envelope
 * is typed here because the backend serves it as plain `dict` over
 * the socket — this file is the one hand-maintained definition, and
 * `tests/test_frontend_foundation.py` drift-guards the vocabulary
 * against `pharmaml.api.events.EventType` so the two sides cannot
 * diverge silently.
 */

/** The backend's event vocabulary (mirrors `api.events.EventType`). */
export const EVENT_TYPES = [
  "job.queued",
  "job.started",
  "job.progress",
  "job.stage",
  "job.completed",
  "job.failed",
  "job.cancel_requested",
  "job.cancelled",
  "result.ready",
  "training.progress",
  "training.metrics",
  "iteration.update",
  "descriptor.progress",
  "file.progress",
  "log.record",
  "warning",
  "error",
  "heartbeat",
  "bus.overflow",
  "shutdown",
] as const;

export type EventType = (typeof EVENT_TYPES)[number];

/** The single wire DTO for every backend event (part11 §13). */
export interface EventEnvelope {
  /** Generated unique event id. */
  readonly event_id: string;
  /** Strictly increasing bus-wide sequence number. */
  readonly sequence: number;
  /** The owning job, or null for session-level events. */
  readonly job_id: string | null;
  readonly event_type: EventType;
  /** UTC ISO-8601 timestamp with the trailing `Z`. */
  readonly timestamp: string;
  /** The typed-per-event payload (narrowed at the consumer). */
  readonly payload: Readonly<Record<string, unknown>>;
}

/** The `hello` welcome frame sent right after the WS opens. */
export interface HelloMessage {
  readonly kind: "hello";
  /** The backend's current bus sequence (replay-since cursor). */
  readonly sequence: number;
  readonly version: string;
}

/** Payload shapes the job-facing consumers narrow to (§13). */
export interface JobProgressPayload {
  readonly progress: number;
  readonly stage: string;
}
