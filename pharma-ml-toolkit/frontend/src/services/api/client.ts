/**
 * The typed REST client — part11 §5.2 (frontend application services)
 * and §15 (local API security).
 *
 * The base URL and the session token reach this client ONLY through
 * the startup handshake channel (`pharmaml.api.server` prints the
 * single `PHARMAML-READY {port, token, pid, version}` line that the
 * Tauri host forwards over the protected IPC bridge) — never from a
 * query string, a log, or a persisted setting. Every request goes to
 * the loopback host the handshake named; the token rides the
 * `X-Session-Token` header the backend's guard accepts.
 */

/** The handshake-derived connection settings (set once at startup). */
export interface BackendEndpoint {
  /** e.g. `http://127.0.0.1:54127` (the handshake's actual port). */
  readonly baseUrl: string;
  /** The per-startup session token (memory-only; never persisted). */
  readonly token: string;
}

let endpoint: BackendEndpoint | null = null;

/** Wire the client (called once from the startup handshake bridge). */
export function configureBackend(endpoint_: BackendEndpoint): void {
  endpoint = { baseUrl: endpoint_.baseUrl, token: endpoint_.token };
}

/** The wired endpoint, or null before the handshake completes. */
export function currentBackend(): BackendEndpoint | null {
  return endpoint;
}

/** The typed API error — status + the backend's `detail` payload. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown) {
    super(`API ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (endpoint === null) {
    throw new Error(
      "the backend endpoint is not configured yet (the startup handshake has not completed)",
    );
  }
  const response = await fetch(`${endpoint.baseUrl}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-Session-Token": endpoint.token,
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let detail: unknown = null;
    try {
      detail = await response.json();
    } catch {
      detail = null;
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }),
  put: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PUT", body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};
