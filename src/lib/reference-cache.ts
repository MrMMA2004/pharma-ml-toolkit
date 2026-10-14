/**
 * Shared, module-level cache for the public-API reference payload so the
 * API-Reference tab and the command palette can both use one fetch.
 */

import type { ReferenceResponse } from "@/lib/toolkit-types";

let cache: ReferenceResponse | null = null;
let inflight: Promise<ReferenceResponse> | null = null;

/** Synchronously peek at the cached reference payload (null until loaded). */
export function peekReference(): ReferenceResponse | null {
  return cache;
}

/**
 * Fetch (and memoize) the public-API reference payload exactly once.
 * Concurrent callers share the same in-flight promise; failures reset the
 * promise so a later call can retry.
 */
export function getReference(): Promise<ReferenceResponse> {
  if (cache) return Promise.resolve(cache);
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const res = await fetch("/api/toolkit/reference");
      const body: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof body === "object" && body !== null && "detail" in body
            ? String((body as { detail: unknown }).detail)
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      cache = body as ReferenceResponse;
      return cache;
    } finally {
      if (!cache) inflight = null; // allow retry after a failed load
    }
  })();
  return inflight;
}
