/**
 * The WebSocket client service — part11 §13 (Phase Three services,
 * part two: hello/replay/heartbeats/reconnection).
 *
 * Wire contract (mirrors `pharmaml.api.routes.events` exactly — the
 * Python drift guards in `tests/test_frontend_services.py` pin it):
 *
 * * the socket lives at `/api/v1/events?token=<session>&since=<seq>`
 *   (browsers cannot set headers on the WebSocket constructor, so the
 *   backend accepts the token as a query parameter; the token is the
 *   per-startup, memory-only credential — not a long-lived secret, so
 *   §15's "no long-lived credentials in query strings" holds);
 * * the FIRST frame is the hello envelope — a `heartbeat` envelope
 *   whose payload is `{hello: true, last_sequence, replayed}`;
 * * every idle period carries `heartbeat` envelopes (the backend
 *   sends one after 15 s without events) — the client treats any
 *   received frame as liveness and closes+reconnects when nothing
 *   arrives within `heartbeatTimeoutMs` (dead-peer detection);
 * * reconnects pass `since=<last sequence seen>` so the handshake
 *   replays exactly the missed tail (duplicate-event protection is
 *   ALSO enforced client-side: envelopes with `sequence <= cursor`
 *   never reach listeners);
 * * `shutdown` means the backend is going down — the client closes
 *   and does NOT reconnect (graceful shutdown);
 * * the client may send the literal text `ping`; the backend answers
 *   with a heartbeat envelope carrying `payload.pong = true`.
 *
 * REST stays the authoritative job-status path (`/api/v1/jobs/{id}`):
 * this stream is the convenience view, and the JobTracker reconciles
 * through REST on every reconnect (§13's recovery rule).
 */

import { currentBackend } from "@/services/api/client";
import {
  type EventEnvelope,
  type EventType,
  type HelloPayload,
  isHelloEnvelope,
} from "./events";

export type WsState =
  | "idle" // constructed, never connected
  | "connecting" // the socket is opening (first attempt or a retry)
  | "open" // hello received; events flowing
  | "reconnecting" // a drop was detected; waiting out the backoff
  | "closed"; // terminal: user close() or the backend's shutdown

export interface EventStreamStatus {
  readonly state: WsState;
  /** The last sequence the client accepted (the replay cursor). */
  readonly lastSequence: number;
  /** Reconnect attempts since the last successful hello. */
  readonly attempts: number;
  /** The latest hello frame's summary (null before the first open). */
  readonly hello: HelloPayload | null;
}

export interface EventStreamOptions {
  /**
   * Liveness window: close and reconnect when no frame (including
   * heartbeats) arrives within this many milliseconds. The backend
   * heartbeats every 15 s, so 40 s means roughly two missed beats.
   */
  readonly heartbeatTimeoutMs?: number;
  /** Maximum reconnect attempts before giving up (default 12). */
  readonly maxReconnectAttempts?: number;
  /** First reconnect delay; doubles each attempt (default 500 ms). */
  readonly reconnectBaseMs?: number;
  /** Backoff ceiling (default 8_000 ms). */
  readonly reconnectMaxMs?: number;
}

/** A listener receives every accepted envelope, in bus order. */
export type EventListener = (envelope: EventEnvelope) => void;

const DEFAULTS: Required<EventStreamOptions> = {
  heartbeatTimeoutMs: 40_000,
  maxReconnectAttempts: 12,
  reconnectBaseMs: 500,
  reconnectMaxMs: 8_000,
};

/**
 * The framework-free WebSocket client. One instance per application
 * (the `getEventStream()` singleton); React binds through
 * `useSyncExternalStore` in `hooks/useJobTracking.ts`.
 */
export class EventStreamClient {
  private readonly token: string;
  /** The ws:// URL base derived from the REST handshake endpoint. */
  private readonly wsBase: string;
  private readonly opts: Required<EventStreamOptions>;

  private _state: WsState = "idle";
  private _lastSequence = -1;
  private _attempts = 0;
  private _hello: HelloPayload | null = null;
  private socket: WebSocket | null = null;
  private userClosed = false;
  private shutdownSeen = false;
  private lastFrameAt = 0;
  private livenessTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly listeners = new Set<EventListener>();
  private readonly stateListeners = new Set<
    (status: EventStreamStatus) => void
  >();

  constructor(baseUrl: string, token: string, options?: EventStreamOptions) {
    this.wsBase = toWebSocketBase(baseUrl);
    this.token = token;
    this.opts = { ...DEFAULTS, ...options };
  }

  /** Open the stream (idempotent while a connection is live). */
  connect(): void {
    if (this.userClosed || this._state === "connecting" || this._state === "open") {
      return;
    }
    this.userClosed = false;
    this.shutdownSeen = false;
    this.openSocket();
  }

  /** Terminal close: no reconnect is scheduled afterwards. */
  close(): void {
    this.userClosed = true;
    this.teardownTimers();
    this.setState("closed");
    if (this.socket !== null) {
      try {
        this.socket.close(1000, "client close");
      } catch {
        // already closing — nothing to do
      }
      this.socket = null;
    }
  }

  /** Protocol keepalive probe: the backend replies with a pong beat. */
  ping(): void {
    if (this.socket !== null && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send("ping");
    }
  }

  get status(): EventStreamStatus {
    return {
      state: this._state,
      lastSequence: this._lastSequence,
      attempts: this._attempts,
      hello: this._hello,
    };
  }

  /** Subscribe to accepted envelopes. Returns the unsubscribe fn. */
  subscribe(listener: EventListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /** Subscribe to connection-state changes. Returns the unsubscribe fn. */
  onStateChange(cb: (status: EventStreamStatus) => void): () => void {
    this.stateListeners.add(cb);
    return () => {
      this.stateListeners.delete(cb);
    };
  }

  // -- internals ---------------------------------------------------------

  private openSocket(): void {
    this.setState("connecting");
    const url = new URL("/api/v1/events", this.wsBase);
    url.searchParams.set("token", this.token);
    if (this._lastSequence >= 0) {
      // reconnect: ask for exactly the missed tail (§13 replay)
      url.searchParams.set("since", String(this._lastSequence));
    }
    let socket: WebSocket;
    try {
      socket = new WebSocket(url.toString());
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;
    this.lastFrameAt = Date.now();

    socket.onmessage = (event: MessageEvent<string>) => {
      this.lastFrameAt = Date.now();
      this.handleFrame(event.data);
    };
    socket.onerror = () => {
      // onclose follows; nothing to do here
    };
    socket.onclose = () => {
      if (this.socket === socket) {
        this.socket = null;
      }
      if (this.userClosed || this.shutdownSeen) {
        this.teardownTimers();
        this.setState("closed");
        return;
      }
      this.scheduleReconnect();
    };
    this.startLiveness();
  }

  private handleFrame(raw: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return; // not an envelope — the protocol only speaks JSON
    }
    const envelope = parseEnvelope(parsed);
    if (envelope === null) {
      return; // malformed frame: drop, do not kill the stream
    }
    if (isHelloEnvelope(envelope)) {
      // the control frame: the cursor stays where it was so the
      // replayed tail right behind the hello still flows through
      this._hello = readHello(envelope);
      this._attempts = 0;
      this.setState("open");
      return;
    }
    if (envelope.sequence <= this._lastSequence) {
      return; // duplicate-event protection (§13)
    }
    this._lastSequence = envelope.sequence;
    if (envelope.event_type === "shutdown") {
      this.shutdownSeen = true; // graceful: close, never reconnect
      this.setState("closed");
      try {
        this.socket?.close(1000, "backend shutdown");
      } catch {
        // closing already
      }
      this.teardownTimers();
    }
    for (const listener of this.listeners) {
      try {
        listener(envelope);
      } catch (error) {
        // a broken consumer must not break the stream
        console.error("event listener failed", error);
      }
    }
    this.notifyState(); // lastSequence moved; status readers refresh
  }

  private startLiveness(): void {
    this.teardownTimers();
    this.livenessTimer = setInterval(() => {
      if (
        this._state !== "open" &&
        this._state !== "connecting" &&
        this._state !== "reconnecting"
      ) {
        return;
      }
      if (Date.now() - this.lastFrameAt > this.opts.heartbeatTimeoutMs) {
        // dead peer: the backend missed two heartbeat windows
        try {
          this.socket?.close(4008, "liveness timeout");
        } catch {
          // closing already
        }
        if (this.socket !== null && this.socket.readyState === WebSocket.CLOSED) {
          this.socket = null;
          this.scheduleReconnect();
        }
      }
    }, 5_000);
  }

  private scheduleReconnect(): void {
    if (this.userClosed || this.shutdownSeen) {
      this.setState("closed");
      return;
    }
    if (this._attempts >= this.opts.maxReconnectAttempts) {
      this.teardownTimers();
      this.setState("closed");
      return;
    }
    const delay = Math.min(
      this.opts.reconnectBaseMs * 2 ** this._attempts,
      this.opts.reconnectMaxMs,
    );
    this._attempts += 1;
    this.setState("reconnecting");
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.openSocket();
    }, delay);
  }

  private teardownTimers(): void {
    if (this.livenessTimer !== null) {
      clearInterval(this.livenessTimer);
      this.livenessTimer = null;
    }
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private setState(state: WsState): void {
    if (this._state === state) {
      this.notifyState();
      return;
    }
    this._state = state;
    this.notifyState();
  }

  private notifyState(): void {
    const status = this.status;
    for (const cb of this.stateListeners) {
      try {
        cb(status);
      } catch (error) {
        console.error("state listener failed", error);
      }
    }
  }
}

/** Narrow an unknown parsed frame to the envelope DTO (validation). */
function parseEnvelope(parsed: unknown): EventEnvelope | null {
  if (typeof parsed !== "object" || parsed === null) {
    return null;
  }
  const record = parsed as Record<string, unknown>;
  const { event_id, sequence, job_id, event_type, timestamp, payload } =
    record;
  if (typeof event_id !== "string" || typeof sequence !== "number") {
    return null;
  }
  if (typeof event_type !== "string" || typeof timestamp !== "string") {
    return null;
  }
  if (job_id !== null && typeof job_id !== "string") {
    return null;
  }
  if (typeof payload !== "object" || payload === null) {
    return null;
  }
  if (!EVENT_TYPE_SET.has(event_type as EventType)) {
    return null; // an event outside the drifted vocabulary: reject
  }
  return {
    event_id,
    sequence,
    job_id: (job_id as string | null) ?? null,
    event_type: event_type as EventType,
    timestamp,
    payload: payload as Readonly<Record<string, unknown>>,
  };
}

const EVENT_TYPE_SET = new Set<string>([
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
]);

function readHello(envelope: EventEnvelope): HelloPayload | null {
  const last = envelope.payload["last_sequence"];
  const replayed = envelope.payload["replayed"];
  if (typeof last !== "number" || typeof replayed !== "number") {
    return null;
  }
  return { hello: true, last_sequence: last, replayed };
}

/** `http(s)://host` -> `ws(s)://host` (the loopback handshake base). */
function toWebSocketBase(baseUrl: string): string {
  if (baseUrl.startsWith("https://")) {
    return `wss://${baseUrl.slice("https://".length)}`;
  }
  if (baseUrl.startsWith("http://")) {
    return `ws://${baseUrl.slice("http://".length)}`;
  }
  return baseUrl;
}

// -- the application singleton -------------------------------------------

let singleton: EventStreamClient | null = null;

/**
 * The application's one event stream, wired from the startup
 * handshake's endpoint (the same source the REST client trusts).
 * Throws while the handshake has not completed — the Tauri host owns
 * the bootstrap order (part11 §14.1).
 */
export function getEventStream(): EventStreamClient {
  if (singleton === null) {
    const backend = currentBackend();
    if (backend === null) {
      throw new Error(
        "the backend endpoint is not configured yet (the startup handshake has not completed)",
      );
    }
    singleton = new EventStreamClient(backend.baseUrl, backend.token);
  }
  return singleton;
}
