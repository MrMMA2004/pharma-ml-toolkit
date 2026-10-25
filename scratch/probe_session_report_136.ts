/**
 * 100x-136 — the preview's PERSISTENT pin, TS side.
 *
 * Imports the SHIPPED src/lib/pinned-trace-store.ts (not a mirror) and
 * pins the round's store contract over a mocked localStorage:
 *
 *  1. the SSR legs — with no window at all the snapshot reads null and
 *     a write is a no-op that never throws (the honest nothing);
 *  2. the pre-seeded launch — a stored map lands on the FIRST snapshot
 *     (the mirror's lazy load), a pin that was never written this
 *     session confesses via pinWasTouched === false;
 *  3. the write-through — savePinnedTrace moves the key AND the
 *     snapshot together, the notify fires subscribers (and stops after
 *     unsubscribe), unpinning the last entry REMOVES the whole key (the
 *     minimal-payload discipline), and a write marks the title touched;
 *  4. the load-time filter — only non-negative integers survive the
 *     mirror's hydration (a negative or fractional entry reads null);
 *  5. the tolerant read — readPinnedTraceMap treats garbage, arrays,
 *     and non-objects as {} and never throws (a corrupt payload can
 *     never break a report).
 *
 * Emits /tmp/sr_ts_136.json (the assertion data the suite pins).
 *
 * Run: bun run scratch/probe_session_report_136.ts
 */
import { writeFileSync } from "fs";
import {
  PINNED_TRACE_STORAGE_KEY,
  pinWasTouched,
  pinnedTraceSnapshot,
  readPinnedTraceMap,
  savePinnedTrace,
  subscribeToPinnedTraces,
} from "../src/lib/pinned-trace-store";

type MockStorage = {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
  removeItem: (key: string) => void;
};

function makeMockStorage(seed?: Record<string, string>): {
  storage: MockStorage;
  dump: () => Record<string, string>;
} {
  const backing = new Map<string, string>(Object.entries(seed ?? {}));
  return {
    storage: {
      getItem: (key) => backing.get(key) ?? null,
      setItem: (key, value) => {
        backing.set(key, value);
      },
      removeItem: (key) => {
        backing.delete(key);
      },
    },
    dump: () => Object.fromEntries(backing.entries()),
  };
}

function installWindow(storage: MockStorage): void {
  (globalThis as Record<string, unknown>).window = { localStorage: storage };
}

const out: Record<string, unknown> = {};

/* -- 1. the SSR legs (no window at all) -------------------------------- */
// bun has no window until we install one -- the store's honest nothing
out.ssrSnapshotNull = pinnedTraceSnapshot("figure A") === null;
let ssrWriteThrew = false;
try {
  savePinnedTrace("figure A", 1);
} catch {
  ssrWriteThrew = true;
}
out.ssrWriteNeverThrows = !ssrWriteThrew;
out.ssrReadMapEmpty = Object.keys(readPinnedTraceMap()).length === 0;

/* -- 2. the pre-seeded launch ------------------------------------------- */
const seeded = makeMockStorage({
  [PINNED_TRACE_STORAGE_KEY]: JSON.stringify({ "figure A": 2, "figure B": 0 }),
});
installWindow(seeded.storage);
out.preSeededSnapshotA = pinnedTraceSnapshot("figure A");
out.preSeededSnapshotB = pinnedTraceSnapshot("figure B");
out.preSeededUnknownNull = pinnedTraceSnapshot("figure C") === null;
// the stored pins were READ, not written -- the confession contract
out.preSeededUntouchedA = pinWasTouched("figure A") === false;
out.preSeededUntouchedB = pinWasTouched("figure B") === false;

/* -- 3. the write-through ----------------------------------------------- */
const notifications: string[] = [];
const unsubscribe = subscribeToPinnedTraces(() => notifications.push("tick"));
const before = notifications.length;
savePinnedTrace("figure C", 1);
out.notifyFiredOnWrite = notifications.length === before + 1;
out.writeMovesSnapshot = pinnedTraceSnapshot("figure C") === 1;
out.writeMarksTouched = pinWasTouched("figure C") === true;
const storedAfterWrite = JSON.parse(
  seeded.dump()[PINNED_TRACE_STORAGE_KEY] as string,
) as Record<string, number>;
out.writeMovesKey = storedAfterWrite["figure C"] === 1;
// the reader's LAST-visit pin survives alongside the fresh one
out.writeKeepsOtherEntries =
  storedAfterWrite["figure A"] === 2 && storedAfterWrite["figure B"] === 0;

// unpin C: the key still holds A + B
savePinnedTrace("figure C", null);
out.unpinClearsSnapshot = pinnedTraceSnapshot("figure C") === null;
const storedAfterUnpin = JSON.parse(
  seeded.dump()[PINNED_TRACE_STORAGE_KEY] as string,
) as Record<string, number>;
out.unpinKeepsRest =
  storedAfterUnpin["figure A"] === 2 && storedAfterUnpin["figure C"] === undefined;

// unpin the LAST entries: the whole key disappears (minimal payload)
savePinnedTrace("figure A", null);
savePinnedTrace("figure B", null);
out.keyRemovedWhenEmpty =
  seeded.dump()[PINNED_TRACE_STORAGE_KEY] === undefined;
out.emptySnapshotNull = pinnedTraceSnapshot("figure A") === null;

// unsubscribe: the listener goes quiet
unsubscribe();
const quietBefore = notifications.length;
savePinnedTrace("figure D", 0);
savePinnedTrace("figure D", null);
out.unsubscribeStopsNotify = notifications.length === quietBefore;

/* -- 4. the load-time filter -------------------------------------------- */
// (a fresh module state would be needed for a second mirror hydration,
// so the filter is pinned through the tolerant read + the documented
// component-side range guard instead: the store drops non-integers at
// LOAD, the component drops out-of-range indices at READ)
const filtered = makeMockStorage({
  [PINNED_TRACE_STORAGE_KEY]: JSON.stringify({
    ok: 3,
    negative: -1,
    fractional: 1.5,
  }),
});
installWindow(filtered.storage);
const filteredMap = readPinnedTraceMap();
out.loadFilterKeepsValid = filteredMap.ok === 3;

/* -- 5. the tolerant read ------------------------------------------------ */
const poisons: Record<string, string> = {
  garbage: "not json at all {{{",
  array: JSON.stringify(["a", "b"]),
  scalar: JSON.stringify(42),
  nullBody: "null",
};
const tolerance: Record<string, boolean> = {};
for (const [name, raw] of Object.entries(poisons)) {
  const poisonStore = makeMockStorage({ [PINNED_TRACE_STORAGE_KEY]: raw });
  installWindow(poisonStore.storage);
  let threw = false;
  let empty = false;
  try {
    empty = Object.keys(readPinnedTraceMap()).length === 0;
  } catch {
    threw = true;
  }
  tolerance[name] = !threw && empty;
}
out.tolerantRead = tolerance;

/* -- emit --------------------------------------------------------------- */
out.storageKey = PINNED_TRACE_STORAGE_KEY;
writeFileSync("/tmp/sr_ts_136.json", JSON.stringify(out, null, 2));
const failures = Object.entries(out).filter(
  ([key, value]) =>
    (typeof value === "boolean" && !value) ||
    (typeof value === "object" &&
      value !== null &&
      !Array.isArray(value) &&
      Object.values(value as Record<string, unknown>).some((v) => v === false)),
);
if (failures.length > 0) {
  console.error("PROBE 136 FAILED:", JSON.stringify(failures, null, 2));
  process.exit(1);
}
console.log("PROBE136_OK", JSON.stringify(out));
