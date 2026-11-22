/**
 * The React binding for the service layer — part11 §8.2/§5.2
 * (Phase Three services, part two).
 *
 * Two hooks over the framework-free singletons, both through
 * `useSyncExternalStore` (no extra state-management dependency —
 * §8.2's "appropriate state management without unnecessary
 * abstraction"):
 *
 * * `useEventStreamState()` — the WS connection state for the shell
 *   (a reconnect banner, the offline badge, …);
 * * `useJob(jobId)` — one tracked job: the WS-driven view with the
 *   REST reconciliation the JobTracker owns (§13: REST stays the
 *   authoritative recovery path), plus its `cancel`/`refresh`
 *   actions.
 *
 * Both hooks assume the startup handshake completed (the Tauri host
 * wires `configureBackend` before the workspaces mount — §14.1);
 * before that, `getEventStream()` throws the documented configure
 * error, loudly, instead of rendering a silent fake.
 */

import { useCallback, useEffect, useMemo, useRef, useSyncExternalStore } from "react";

import { jobsApi } from "@/services/jobs/client";
import {
  getJobTracker,
  type TrackedJob,
} from "@/services/jobs/tracker";
import {
  getEventStream,
  type EventStreamStatus,
} from "@/services/websocket/client";

/** The live WS connection state (idle/connecting/open/…/closed). */
export function useEventStreamState(): EventStreamStatus {
  const stream = getEventStream();
  // The snapshot cache must be a ref: `EventStreamStatus` is an object
  // (a fresh literal per state transition), so `getSnapshot` needs a
  // stable reference between notifications — and a ref is the one
  // render-scope slot React sanctions for reassignment after render.
  const cache = useRef(stream.status);
  const [subscribe, getSnapshot] = useMemo(() => {
    const get = () => cache.current;
    const sub = (cb: () => void) => {
      const unsubscribe = stream.onStateChange((status) => {
        cache.current = status;
        cb();
      });
      // Resync from the live store on (re)subscribe — a transition that
      // lands before the passive effect attaches (the interleaving window
      // the CI flake in run 45056299925 exposed) is not lost: React's
      // post-subscribe snapshot consistency check sees the fresh value
      // and re-renders the banner to the store's true state.
      cache.current = stream.status;
      return unsubscribe;
    };
    return [sub, get] as const;
  }, [stream]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export interface UseJobResult {
  /** The tracked view (null until the tracker entry exists). */
  readonly snapshot: TrackedJob | null;
  /** POST the cooperative cancellation; the tracker follows the event. */
  readonly cancel: () => Promise<void>;
  /** Force the authoritative REST read now (§13's recovery path). */
  readonly refresh: () => Promise<void>;
}

/** Track one job through its whole lifecycle (QUEUED → terminal). */
export function useJob(jobId: string | null): UseJobResult {
  const tracker = getJobTracker();

  // the singleton starts exactly once, lazily, on first consumer
  useEffect(() => {
    tracker.start();
  }, [tracker]);

  const [subscribe, getSnapshot] = useMemo(() => {
    if (jobId === null) {
      const noop = () => () => undefined;
      return [noop, () => null] as const;
    }
    const sub = (cb: () => void) => {
      // track() seeds the tracker entry (and kicks the authoritative
      // REST read) BEFORE subscribe() registers this callback — the
      // entry must exist when the callback is added, or the async
      // refresh's notify() would reach an empty subscriber set and
      // the store would never re-render the card (the REST-poll
      // recovery path depends on this order as much as the WS path).
      const untrack = tracker.track(jobId);
      const unlisten = tracker.subscribe(jobId, cb);
      return () => {
        unlisten();
        untrack();
      };
    };
    const get = () => tracker.snapshot(jobId);
    return [sub, get] as const;
  }, [tracker, jobId]);

  const snapshot = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  const cancel = useCallback(async () => {
    if (jobId === null) {
      return;
    }
    await jobsApi.cancel(jobId);
    await tracker.refresh(jobId);
  }, [jobId, tracker]);

  const refresh = useCallback(async () => {
    if (jobId === null) {
      return;
    }
    await tracker.refresh(jobId);
  }, [jobId, tracker]);

  return { snapshot, cancel, refresh };
}
