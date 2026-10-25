/* ------------------------------------------------------------------ */
/* The pinned-trace store (100x-136) — the web twin of the desktop      */
/* playground's report_page_order session-file key.                     */
/*                                                                      */
/* A pin deliberately held by the reader (Enter on a focused trace      */
/* group in the report-preview sketch) survives the dialog closing      */
/* and the page reloading, remembered per browser.  The map is keyed    */
/* by the figure section's title (the collector's stable identity       */
/* for it) and rides localStorage — exactly the way the desktop's       */
/* order memory rides the auto-saved session file.                      */
/*                                                                      */
/* The store is wrapped as a React external store (the TSX side         */
/* subscribes through useSyncExternalStore, so the component needs NO   */
/* restore effect at all): reads ride an in-memory mirror hydrated      */
/* once from the key (getSnapshot must stay cheap and stable between    */
/* writes), writes update the mirror AND the key and then notify       */
/* every subscriber.  SSR hydrates against the server snapshot (null —  */
/* the honest nothing), and the client's stored pin lands in the        */
/* follow-up render React runs after subscribing, so there is never a   */
/* hydration mismatch.                                                  */
/*                                                                      */
/* Every read is failure-tolerant (a corrupt payload is treated as      */
/* empty — never a broken report) and every write is best-effort (a     */
/* blocked localStorage still updates the mirror: the pin lives for     */
/* this page even when the browser refuses to remember it).             */
/* ------------------------------------------------------------------ */

/** localStorage key for the per-figure pinned trace map. */
export const PINNED_TRACE_STORAGE_KEY = "pharmaml.reportPinnedTrace";

/** The store's subscribers — notified on every write. */
const pinListeners = new Set<() => void>();

/** The in-memory mirror of the stored map (loaded lazily, written
 * through on every save — the client's source of truth between
 * writes, so getSnapshot stays cheap and referentially stable). */
const pinMirror = new Map<string, number>();

/** Whether the mirror has been hydrated from the key yet. */
let pinMirrorLoaded = false;

/** Titles whose pin was written THIS browser session — a pin read
 * from storage (never touched) confesses its provenance in the
 * readout; a fresh keystroke pin is the reader's own doing. */
const pinTouched = new Set<string>();

/** Read the stored map (client only, failure-tolerant: a corrupt
 * payload is treated as empty — never a broken report).  Reads
 * whatever localStorage holds RIGHT NOW (no caching), so it is also
 * the honest probe surface for the tolerance contract. */
export function readPinnedTraceMap(): Record<string, number> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(PINNED_TRACE_STORAGE_KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return {};
    }
    return parsed as Record<string, number>;
  } catch {
    return {};
  }
}

/** The store's client snapshot for one figure — the mirror loads
 * lazily on first read (only non-negative integers survive the load;
 * anything else in the payload is dropped, the honest fallback) and
 * stays stable until a write. */
export function pinnedTraceSnapshot(title: string): number | null {
  if (typeof window === "undefined") return null;
  if (!pinMirrorLoaded) {
    for (const [key, value] of Object.entries(readPinnedTraceMap())) {
      if (typeof value === "number" && Number.isInteger(value) && value >= 0) {
        pinMirror.set(key, value);
      }
    }
    pinMirrorLoaded = true;
  }
  return pinMirror.get(title) ?? null;
}

/** Subscribe to the store — the useSyncExternalStore contract.  The
 * returned unsubscribe removes the listener. */
export function subscribeToPinnedTraces(onStoreChange: () => void): () => void {
  pinListeners.add(onStoreChange);
  return () => pinListeners.delete(onStoreChange);
}

/** Was this figure's pin written THIS browser session?  A pin that
 * was only ever READ from storage returns false — the component
 * renders its "pinned from your last visit" confession for it. */
export function pinWasTouched(title: string): boolean {
  return pinTouched.has(title);
}

/** The write path — pin or unpin one figure: the key and the mirror
 * move together, an empty map removes the whole key (the stored
 * payload stays minimal — a browser that never pinned anything never
 * writes storage, the walking-order key's discipline), and every
 * subscriber is notified so React re-renders what changed.  A blocked
 * localStorage still updates the mirror: the pin lives for this page
 * even when the browser refuses to remember it. */
export function savePinnedTrace(title: string, traceIndex: number | null): void {
  if (typeof window === "undefined") return;
  try {
    const map = readPinnedTraceMap();
    if (traceIndex === null) {
      delete map[title];
    } else {
      map[title] = traceIndex;
    }
    if (Object.keys(map).length === 0) {
      window.localStorage.removeItem(PINNED_TRACE_STORAGE_KEY);
    } else {
      window.localStorage.setItem(
        PINNED_TRACE_STORAGE_KEY,
        JSON.stringify(map),
      );
    }
  } catch {
    /* a blocked localStorage just loses the preference */
  }
  if (traceIndex === null) {
    pinMirror.delete(title);
  } else {
    pinMirror.set(title, traceIndex);
  }
  pinTouched.add(title);
  for (const listener of pinListeners) listener();
}
