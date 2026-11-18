/**
 * The artifact registry client — part11 §21's persistence surface.
 *
 * The generic registry lanes every workspace row reuses: list, view,
 * delete, and the digest-verified download. The creation routes
 * (`POST /figure`, `POST /table`, `POST /model`) are the workspace
 * rows' own surfaces — each rides its own workspace client (the
 * generative/ML/DL clients post their own figures), never this one.
 *
 * The backend contract (mirrored field-for-field from the route
 * source `routes/artifacts.py` and the store's `ArtifactRecord.view()`):
 *
 * * `GET /api/v1/artifacts` — `{artifacts: ArtifactView[]}` (the
 *   session's registry, newest-first).
 * * `GET /api/v1/artifacts/{artifact_id}` — one `ArtifactView`
 *   (`artifact_id`, `kind`, `filename`, `content_type`, `size_bytes`,
 *   `sha256`, `created_at`, `meta`, `download_url`); 404 when unknown.
 * * `DELETE /api/v1/artifacts/{artifact_id}` — `{evicted: true,
 *   artifact_id}` (the bounded registry's own evict lane).
 * * `GET /api/v1/artifacts/{artifact_id}/download` — the bytes as an
 *   attachment with the verification headers (`ETag` = the quoted
 *   sha256, `X-Artifact-SHA256`, `X-Artifact-Kind`).
 *
 * §15: the browser cannot put the session token header on a bare
 * `<a href>` — the download rides the client's raw lane (the header
 * applied), the digest is verified locally through
 * `crypto.subtle.digest` when the context is secure, and an insecure
 * context reports `null` honestly (never a fabricated verification):
 * the route's `X-Artifact-SHA256` was received either way, the local
 * re-computation is the unavailable half.
 */

import { api } from "@/services/api/client";

/** `ArtifactRecord.view()` — the registry's own shape (mirrored once). */
export interface ArtifactView {
  readonly artifact_id: string;
  readonly kind: string;
  readonly filename: string;
  readonly content_type: string;
  readonly size_bytes: number;
  readonly sha256: string;
  readonly created_at: string;
  readonly meta: Readonly<Record<string, unknown>>;
  readonly download_url: string;
}

interface ArtifactsResponse {
  readonly artifacts: readonly unknown[];
}

/** `DELETE /{artifact_id}` — the bounded registry's evict answer. */
export interface ArtifactEvicted {
  readonly evicted: boolean;
  readonly artifact_id: string;
}

/**
 * The digest-verified download result: the bytes as a `Blob` plus the
 * route's promise (`sha256`) and the local re-computation
 * (`actual_sha256`, `digest_verified`) — `null` when the local
 * re-computation is unavailable (an insecure context), `false` only
 * on a real mismatch.
 */
export interface ArtifactDownload {
  readonly blob: Blob;
  readonly filename: string;
  readonly sha256: string;
  readonly actual_sha256: string | null;
  readonly digest_verified: boolean | null;
}

function _field(raw: unknown, key: string, label: string): unknown {
  if (typeof raw !== "object" || raw === null) {
    throw new Error(`${label} is not an object`);
  }
  const value = (raw as Record<string, unknown>)[key];
  if (value === undefined) {
    throw new Error(`${label} is missing the ${key} field`);
  }
  return value;
}

function _string(raw: unknown, key: string, label: string): string {
  const value = _field(raw, key, label);
  if (typeof value !== "string") {
    throw new Error(`${label}.${key} is not a string`);
  }
  return value;
}

function _number(raw: unknown, key: string, label: string): number {
  const value = _field(raw, key, label);
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label}.${key} is not a finite number`);
  }
  return value;
}

function _meta(raw: unknown, label: string): Readonly<Record<string, unknown>> {
  const value = _field(raw, "meta", label);
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label}.meta is not an object`);
  }
  return value as Readonly<Record<string, unknown>>;
}

/**
 * Narrow one untrusted payload to `ArtifactView` (§8.1: runtime
 * validation — the label names the lane in the thrown error).
 */
export function parseArtifactView(raw: unknown, label: string): ArtifactView {
  return {
    artifact_id: _string(raw, "artifact_id", label),
    kind: _string(raw, "kind", label),
    filename: _string(raw, "filename", label),
    content_type: _string(raw, "content_type", label),
    size_bytes: _number(raw, "size_bytes", label),
    sha256: _string(raw, "sha256", label),
    created_at: _string(raw, "created_at", label),
    meta: _meta(raw, label),
    download_url: _string(raw, "download_url", label),
  };
}

async function _sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  const bytes = new Uint8Array(digest);
  let hex = "";
  for (const byte of bytes) {
    hex += byte.toString(16).padStart(2, "0");
  }
  return hex;
}

export const artifactsApi = {
  /** `GET /api/v1/artifacts` — the session's registry (newest-first). */
  async list(): Promise<readonly ArtifactView[]> {
    const raw = await api.get<ArtifactsResponse>("/api/v1/artifacts");
    const items = _field(raw, "artifacts", "the artifact registry") as unknown[];
    return items.map((item, index) =>
      parseArtifactView(item, `the registry's row ${index}`),
    );
  },

  /** `GET /api/v1/artifacts/{artifact_id}` — one registry row. */
  async view(artifactId: string): Promise<ArtifactView> {
    const raw = await api.get<unknown>(
      `/api/v1/artifacts/${encodeURIComponent(artifactId)}`,
    );
    return parseArtifactView(raw, `artifact ${artifactId}`);
  },

  /** `DELETE /api/v1/artifacts/{artifact_id}` — the evict lane. */
  async delete(artifactId: string): Promise<ArtifactEvicted> {
    const raw = await api.delete<unknown>(
      `/api/v1/artifacts/${encodeURIComponent(artifactId)}`,
    );
    const evicted = _field(raw, "evicted", `the evict answer for ${artifactId}`);
    if (typeof evicted !== "boolean") {
      throw new Error("the evict answer's evicted field is not a boolean");
    }
    const artifact_id = _string(raw, "artifact_id", "the evict answer");
    return { evicted, artifact_id };
  },

  /**
   * `GET /api/v1/artifacts/{artifact_id}/download` — the bytes as an
   * attachment, the route's `ETag`/`X-Artifact-SHA256` consumed (never
   * discarded), and the local digest re-computed when the context is
   * secure (§15's token-header raw lane: a bare `<a href>` could never
   * carry the session header).
   */
  async download(artifactId: string): Promise<ArtifactDownload> {
    const response = await api.raw(
      `/api/v1/artifacts/${encodeURIComponent(artifactId)}/download`,
    );
    const filename =
      (response.headers.get("content-disposition") ?? "")
        .split("filename=")[1]
        ?.replace(/^"|"$/g, "") ?? `${artifactId}.bin`;
    const sha256 =
      response.headers.get("x-artifact-sha256") ??
      (response.headers.get("etag") ?? "").replace(/^"|"$/g, "");
    const blob = await response.blob();
    let actual_sha256: string | null = null;
    if (typeof crypto !== "undefined" && crypto.subtle !== undefined) {
      try {
        const buffer = await blob.arrayBuffer();
        actual_sha256 = await _sha256Hex(buffer);
      } catch {
        // an insecure context (or a revoked buffer) — the local half
        // of the verification is honestly unavailable
        actual_sha256 = null;
      }
    }
    const digest_verified =
      actual_sha256 === null ? null : actual_sha256 === sha256;
    return { blob, filename, sha256, actual_sha256, digest_verified };
  },
};
