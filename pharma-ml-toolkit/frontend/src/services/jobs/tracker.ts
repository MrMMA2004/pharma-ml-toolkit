/**
 * The job tracker — part11 §13's frontend half (Phase Three services,
 * part two): the WebSocket stream drives live updates, REST stays the
 * authoritative recovery path.
 *
 * Rules (all pinned by `tests/test_frontend_services.py`):
 *
 * * WS events overlay the tracked view optimistically — `job.started`
 *   -> RUNNING, `job.progress`/`training.progress`/`descriptor.progress`
 *   /`file.progress` -> progress+stage, `job.completed` -> COMPLETED,
 *   `result.ready` -> result_ready, `job.failed` -> error, the two
 *   cancellation transitions — but timestamps/configuration stay from
 *   REST (the events do not carry them; nothing is fabricated);
 * * while a tracked job is non-terminal, the tracker polls REST
 *   (`pollMs`, default 1500) — a dropped WS frame costs at most one
 *   poll interval of staleness;
 * * on every stream (re)open, all non-terminal tracked jobs are
 *   refreshed through REST immediately (§13: recovery of job status
 *   after reconnection);
 * * on `bus.overflow` the tracker reconciles through REST too — the
 *   backend's own backpressure contract says REST is authoritative;
 * * a terminal REST refresh stops that job's polling (one final
 *   authoritative read, then the entry is quiescent).
 */

import { type EventEnvelope } from "@/services/websocket/events";
import {
  EventStreamClient,
  getEventStream,
} from "@/services/websocket/client";
import { jobsApi } from "./client";
import {
  isTerminalJobState,
  type JobStatusView,
  type JobState,
} from "./types";

/** The per-job tracked state the React layer renders. */
export interface TrackedJob {
  readonly job_id: string;
  /** The authoritative REST view (null before the first fetch lands). */
  readonly view: JobStatusView | null;
  /** The newest accepted envelope for this job (§13 diagnostics). */
  readonly lastEvent: EventEnvelope | null;
  readonly lastError: string | null;
}

export interface JobTrackerOptions {
  /** REST reconciliation interval for non-terminal jobs. */
  readonly pollMs?: number;
  /** Terminal jobs are kept this many extra polls, then pruned. */
  readonly keepTerminalPolls?: number;
}

const DEFAULTS: Required<JobTrackerOptions> = {
  pollMs: 1_500,
  keepTerminalPolls: 2,
};

/** The event-driven state overlay mapping (mirrors the JobManager). */
function overlayFromEvent(
  view: JobStatusView,
  envelope: EventEnvelope,
): JobStatusView {
  switch (envelope.event_type) {
    case "job.queued":
      return { ...view, state: "QUEUED" as JobState };
    case "job.started":
      return { ...view, state: "RUNNING" as JobState };
    case "job.progress":
    case "training.progress":
    case "descriptor.progress":
    case "file.progress": {
      const progress = envelope.payload["progress"];
      const stage = envelope.payload["stage"];
      return {
        ...view,
        progress: typeof progress === "number" ? progress : view.progress,
        stage: typeof stage === "string" && stage !== "" ? stage : view.stage,
        state: "RUNNING" as JobState,
      };
    }
    case "job.completed":
      return {
        ...view,
        state: "COMPLETED" as JobState,
        progress: 100,
        result_ready: true,
      };
    case "result.ready":
      return { ...view, result_ready: true };
    case "job.failed": {
      const error = envelope.payload["error"];
      return {
        ...view,
        state: "FAILED" as JobState,
        error: typeof error === "string" ? error : view.error,
      };
    }
    case "job.cancel_requested":
      return { ...view, state: "CANCELLATION_REQUESTED" as JobState };
    case "job.cancelled":
      return {
        ...view,
        state: "CANCELLED" as JobState,
        result_ready: false,
      };
    default:
      return view; // log/warning/etc. do not change the state machine
  }
}

interface Entry {
  readonly jobId: string;
  snapshot: TrackedJob;
  terminalPollsRemaining: number;
  inflightRefresh: boolean;
  /** Live `track()` handles (refcount: the entry dies with the last). */
  owners: number;
  readonly subscribers: Set<() => void>;
}

export class JobTracker {
  private readonly stream: EventStreamClient;
  private readonly opts: Required<JobTrackerOptions>;
  private readonly entries = new Map<string, Entry>();
  private unsubscribeStream: (() => void) | null = null;
  private unsubscribeState: (() => void) | null = null;
  private pollTimer: ReturnType<typeof setInterval> | null = null;
  private running = false;

  constructor(stream: EventStreamClient, options?: JobTrackerOptions) {
    this.stream = stream;
    this.opts = { ...DEFAULTS, ...options };
  }

  /** Begin tracking (idempotent). */
  start(): void {
    if (this.running) {
      return;
    }
    this.running = true;
    this.unsubscribeStream = this.stream.subscribe((envelope) => {
      this.onEnvelope(envelope);
    });
    this.unsubscribeState = this.stream.onStateChange((status) => {
      if (status.state === "open") {
        void this.refreshNonTerminal(); // §13 reconnection recovery
      }
    });
    this.pollTimer = setInterval(() => {
      void this.pollTick();
    }, this.opts.pollMs);
  }

  /** Stop tracking (idempotent; the stream itself is untouched). */
  stop(): void {
    this.running = false;
    this.unsubscribeStream?.();
    this.unsubscribeStream = null;
    this.unsubscribeState?.();
    this.unsubscribeState = null;
    if (this.pollTimer !== null) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    for (const entry of this.entries.values()) {
      entry.subscribers.clear();
    }
    this.entries.clear();
  }

  /**
   * Track a job: seeds a snapshot, kicks the authoritative REST read,
   * and returns the untrack function. Refcounted — several components
   * may render the same job; the entry dies with the last untrack.
   */
  track(jobId: string): () => void {
    let entry = this.entries.get(jobId);
    if (entry === undefined) {
      entry = {
        jobId,
        snapshot: {
          job_id: jobId,
          view: null,
          lastEvent: null,
          lastError: null,
        },
        terminalPollsRemaining: this.opts.keepTerminalPolls,
        inflightRefresh: false,
        owners: 0,
        subscribers: new Set(),
      };
      this.entries.set(jobId, entry);
      void this.refresh(jobId);
    }
    entry.owners += 1;
    return () => {
      const tracked = this.entries.get(jobId);
      if (tracked === undefined) {
        return;
      }
      tracked.owners -= 1;
      if (tracked.owners <= 0) {
        tracked.subscribers.clear();
        this.entries.delete(jobId);
      }
    };
  }

  /** The referentially-stable snapshot (useSyncExternalStore-ready). */
  snapshot(jobId: string): TrackedJob | null {
    return this.entries.get(jobId)?.snapshot ?? null;
  }

  /** Subscribe to this job's snapshot changes. */
  subscribe(jobId: string, cb: () => void): () => void {
    const entry = this.entries.get(jobId);
    if (entry === undefined) {
      return () => undefined;
    }
    entry.subscribers.add(cb);
    return () => {
      entry.subscribers.delete(cb);
    };
  }

  /** The authoritative REST read for one job (also the reconnect fix). */
  async refresh(jobId: string): Promise<void> {
    const entry = this.entries.get(jobId);
    if (entry === undefined || entry.inflightRefresh) {
      return;
    }
    entry.inflightRefresh = true;
    try {
      const view = await jobsApi.status(jobId);
      const live = this.entries.get(jobId);
      if (live === undefined) {
        return; // untracked while the read was in flight
      }
      live.snapshot = { ...live.snapshot, view, lastError: null };
      if (isTerminalJobState(view.state)) {
        live.terminalPollsRemaining = Math.min(
          live.terminalPollsRemaining,
          this.opts.keepTerminalPolls,
        );
      }
      this.notify(live);
    } catch (error) {
      const live = this.entries.get(jobId);
      if (live === undefined) {
        return;
      }
      live.snapshot = {
        ...live.snapshot,
        lastError: error instanceof Error ? error.message : String(error),
      };
      this.notify(live);
    } finally {
      const live = this.entries.get(jobId);
      if (live !== undefined) {
        live.inflightRefresh = false;
      }
    }
  }

  /** Refresh every non-terminal tracked job (reconnect/overflow). */
  async refreshNonTerminal(): Promise<void> {
    const ids = [...this.entries.keys()].filter((jobId) => {
      const view = this.entries.get(jobId)?.snapshot.view ?? null;
      return view === null || !isTerminalJobState(view.state);
    });
    await Promise.allSettled(ids.map((jobId) => this.refresh(jobId)));
  }

  // -- internals ---------------------------------------------------------

  private onEnvelope(envelope: EventEnvelope): void {
    if (envelope.event_type === "bus.overflow") {
      void this.refreshNonTerminal(); // REST is authoritative (§13)
      return;
    }
    const jobId = envelope.job_id;
    if (jobId === null) {
      return; // session-level events do not belong to a job entry
    }
    const entry = this.entries.get(jobId);
    if (entry === undefined) {
      return; // not tracked: the REST ledger remains the source
    }
    entry.snapshot = {
      ...entry.snapshot,
      lastEvent: envelope,
    };
    const view = entry.snapshot.view;
    if (view !== null && !isTerminalJobState(view.state)) {
      entry.snapshot = {
        ...entry.snapshot,
        view: overlayFromEvent(view, envelope),
      };
    }
    this.notify(entry);
  }

  private async pollTick(): Promise<void> {
    if (!this.running) {
      return;
    }
    const live: string[] = [];
    for (const [jobId, entry] of this.entries) {
      if (entry.subscribers.size === 0) {
        continue; // nobody renders this job; the entry is dormant
      }
      const view = entry.snapshot.view;
      if (view === null || !isTerminalJobState(view.state)) {
        live.push(jobId);
      } else if (entry.terminalPollsRemaining > 0) {
        entry.terminalPollsRemaining -= 1;
        live.push(jobId); // a couple of confirming reads, then quiet
      }
    }
    await Promise.allSettled(live.map((jobId) => this.refresh(jobId)));
  }

  private notify(entry: Entry): void {
    for (const cb of entry.subscribers) {
      try {
        cb();
      } catch (error) {
        console.error("job snapshot listener failed", error);
      }
    }
  }
}

// -- the application singleton -------------------------------------------

let tracker: JobTracker | null = null;

/**
 * The application's one tracker, bound to the singleton event stream
 * (§5.2's Frontend Application Services block: REST client | WS
 * client | Job Manager). Lazy: the handshake may not have landed yet
 * — `getEventStream()` throws the documented configure error until
 * the Tauri host completes the startup handshake (part11 §14.1).
 */
export function getJobTracker(): JobTracker {
  if (tracker === null) {
    tracker = new JobTracker(getEventStream());
  }
  return tracker;
}
