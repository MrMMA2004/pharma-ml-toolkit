/**
 * The startup bridge — part11 §14.1 steps 9–12 (the wiring the Phase
 * Sixteen audit surfaced as the last real GUI gap: `configureBackend`
 * and `stream.connect` previously had no caller inside `frontend/src`
 * — only the Node e2e harnesses called them, so the browser app could
 * never reach its own backend without an injected shim).
 *
 * The endpoint + session token reach this module through exactly two
 * channels — never a query string, a log, or a persisted setting:
 *
 * 1. the Tauri IPC bridge: `invoke("backend_endpoint")` returns the
 *    handshake the Rust host parsed from the sidecar's single
 *    `PHARMAML-READY {port, token, pid, version}` stdout line
 *    (§14.1 steps 4–8; `src-tauri/src/commands.rs`);
 * 2. a DEV-ONLY same-origin fallback: a plain browser dev session
 *    rides the Vite proxy (`/api` lanes, REST + WS) with the token
 *    supplied through `VITE_PHARMAML_TOKEN` at dev-server start.
 *    Production builds never see this path — the `import.meta.env.DEV`
 *    guard below pins that.
 *
 * The sequence (§14.1 steps 10–12): wire the typed clients with the
 * handshake endpoint → verify backend health via REST → open the
 * WebSocket event stream → the workspaces mount. A failed handshake,
 * health probe, or stream open surfaces as a typed `StartupError`
 * with the failing stage — the gate renders it honestly with a
 * retry (§14.2's controlled restart policy), never a silent fake.
 */

import { invoke } from "@tauri-apps/api/core";

import { api, ApiError, configureBackend } from "./client";
import { getEventStream } from "@/services/websocket/client";

/** Which startup channel delivered the handshake (displayed by the shell). */
export type StartupSource = "tauri-ipc" | "dev-same-origin";

/** The §14.1 boot outcome the shell consumes. */
export interface StartupInfo {
  readonly source: StartupSource;
  /** The backend's own reported version (`/api/v1/health`). */
  readonly version: string;
  /** Whether the scientific core (RDKit) imported cleanly. */
  readonly scientificCore: boolean;
}

/**
 * Why the startup sequence failed — the honest classification of the
 * part14 §3.1 item-12 incident (D-3): an unreachable/CORS-blocked
 * probe and an authorization rejection were previously all wrapped
 * into "the backend health probe failed", which the gate rendered
 * under the "The backend did not start" headline — mislabeling the
 * packaged app's origin-policy 403s and the browser's CORS-blocked
 * fetch rejections as startup crashes.
 */
export type StartupFailureCause =
  | "handshake" // the endpoint/token acquisition itself failed
  | "network" // fetch rejected — unreachable or CORS-blocked
  | "auth" // the backend answered 401/403 — token/origin rejected
  | "http" // the backend answered another non-ok status
  | "protocol"; // the backend answered, but not with the payload the contract expects

/** A startup failure carrying the failing §14.1 stage. */
export class StartupError extends Error {
  readonly stage: "acquire" | "health" | "stream";
  /** Why the sequence failed (narrows the base `Error.cause`). */
  override readonly cause: StartupFailureCause;

  constructor(
    stage: "acquire" | "health" | "stream",
    message: string,
    cause: StartupFailureCause = "handshake",
  ) {
    super(message);
    this.name = "StartupError";
    this.stage = stage;
    this.cause = cause;
  }
}

/** The handshake-derived settings the typed clients need. */
interface HandshakeEndpoint {
  readonly baseUrl: string;
  readonly token: string;
  readonly version: string;
}

/** The health probe's response shape (`routes/health.py`). */
interface HealthReport {
  readonly status: string;
  readonly version: string;
  readonly scientific_core: boolean;
}

/** The Tauri runtime injects this before any frontend code runs. */
function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/**
 * §14.1 steps 4–8: the Tauri host already launched the sidecar and
 * parsed its handshake — fetch that endpoint over the protected IPC
 * bridge (the ONE application command; no filesystem, no shell).
 */
async function acquireFromTauriIpc(): Promise<HandshakeEndpoint> {
  // the Rust `BackendEndpoint` serializes snake_case
  const raw = await invoke<{ base_url: string; token: string; version: string }>(
    "backend_endpoint",
  );
  return {
    baseUrl: raw.base_url,
    token: raw.token,
    version: raw.version,
  };
}

/**
 * The dev-only fallback: same-origin REST/WS lanes through the Vite
 * proxy (`PHARMAML_DEV_BACKEND_PORT`), token injected at dev-server
 * start via `VITE_PHARMAML_TOKEN`.  Guarded to `import.meta.env.DEV`
 * so production builds never take this path.
 */
function acquireFromDevFallback(): HandshakeEndpoint {
  const token = import.meta.env.VITE_PHARMAML_TOKEN;
  if (typeof token !== "string" || token.length === 0) {
    throw new StartupError(
      "acquire",
      "no dev startup channel: start the Vite dev server with VITE_PHARMAML_TOKEN set " +
        "(the production frontend instead boots inside the Tauri 2 host, which " +
        "delivers the handshake over the backend_endpoint IPC bridge)",
    );
  }
  return { baseUrl: window.location.origin, token, version: "" };
}

/**
 * The §14.1 steps 10–12 boot sequence.  Idempotent per process: the
 * gate calls it once; a retry re-runs the health/stream stages after
 * re-acquiring the endpoint.
 */
export async function initializeBackend(): Promise<StartupInfo> {
  // steps 4–8 (acquire the handshake's endpoint + token)
  let endpoint: HandshakeEndpoint;
  let source: StartupSource;
  if (isTauriRuntime()) {
    try {
      endpoint = await acquireFromTauriIpc();
    } catch (err) {
      throw new StartupError(
        "acquire",
        "the Tauri host's backend_endpoint bridge failed: " +
          (err instanceof Error ? err.message : String(err)),
      );
    }
    source = "tauri-ipc";
  } else if (import.meta.env.DEV) {
    endpoint = acquireFromDevFallback();
    source = "dev-same-origin";
  } else {
    throw new StartupError(
      "acquire",
      "no startup channel: the production frontend boots inside the Tauri 2 " +
        "host (the dev fallback exists only in dev builds)",
    );
  }

  // step 10 — wire the typed REST client with the handshake endpoint
  configureBackend({ baseUrl: endpoint.baseUrl, token: endpoint.token });

  // step 10 (verify backend health BEFORE enabling the workspaces)
  let health: HealthReport;
  try {
    health = await api.get<HealthReport>("/api/v1/health");
  } catch (err) {
    throw classifyHealthFailure(err);
  }
  if (health.status !== "ok") {
    throw new StartupError(
      "health",
      `the backend reported status=${health.status} instead of ok`,
      "protocol",
    );
  }

  // step 11 — initialize the WebSocket client (the event stream the
  // JobTracker overlays; REST stays the authoritative recovery path)
  try {
    getEventStream().connect();
  } catch (err) {
    throw new StartupError(
      "stream",
      "the WebSocket event stream failed to open: " +
        (err instanceof Error ? err.message : String(err)),
    );
  }

  // step 12 — the workspaces may mount
  return {
    source,
    version: health.version || endpoint.version,
    scientificCore: health.scientific_core,
  };
}

/** Pull the backend's `detail` payload out of an ApiError, if any. */
function apiErrorDetail(detail: unknown): string {
  if (detail === null || detail === undefined) {
    return "";
  }
  if (typeof detail === "object" && detail !== null && "detail" in detail) {
    return String((detail as { detail: unknown }).detail);
  }
  return String(detail);
}

/**
 * The D-3 fix (part14 §3.1 item 12): classify the health probe's
 * failure honestly —
 *
 * * a 401/403 is an AUTHORIZATION rejection (the backend is up; the
 *   token or this app's origin was not accepted);
 * * a fetch TypeError is a REACHABILITY/CORS block (the browser either
 *   could not connect or refused the cross-origin request — the
 *   packaged-app origin incident class);
 * * another HTTP status is an honest "answered, but not ok";
 * * anything else is a protocol-level surprise.
 *
 * The gate renders the cause, so "The backend did not start" stays
 * reserved for failures that actually are startup failures.
 */
function classifyHealthFailure(err: unknown): StartupError {
  if (err instanceof ApiError) {
    const detail = apiErrorDetail(err.detail);
    if (err.status === 401 || err.status === 403) {
      return new StartupError(
        "health",
        `the backend rejected the health probe with HTTP ${err.status}` +
          (detail ? ` (${detail})` : "") +
          " — the session token or this app's origin was not accepted; " +
          "this is an authorization failure, not a backend startup failure",
        "auth",
      );
    }
    return new StartupError(
      "health",
      `the backend health probe failed with HTTP ${err.status}` +
        (detail ? ` (${detail})` : "") +
        " — the backend answered, but not with ok",
      "http",
    );
  }
  const reason = err instanceof Error ? err.message : String(err);
  if (err instanceof TypeError || /fetch/i.test(reason)) {
    return new StartupError(
      "health",
      `the health probe could not reach the backend (${reason}) — the ` +
        "browser either could not connect or blocked the cross-origin " +
        "request (network/CORS); whether the backend itself started is " +
        "not established by this failure",
      "network",
    );
  }
  return new StartupError(
    "health",
    "the backend health probe failed: " + reason,
    "protocol",
  );
}
