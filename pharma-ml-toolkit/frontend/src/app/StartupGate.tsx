/**
 * The §14.1 startup gate — the shell mounts the workspaces only after
 * the backend handshake completes, health is verified, and the event
 * stream opens (part11 Phase Fourteen; the Phase Sixteen audit's
 * central fix: previously the pages mounted unconditionally and every
 * service call would throw pre-handshake).
 *
 * Three honest states, never a silent fake:
 * - `connecting` — the handshake is in flight (backend launch,
 *   health probe, WS open);
 * - `failed` — the typed `StartupError` stage + message verbatim,
 *   with a Retry (§14.2's controlled restart, not a page reload);
 * - `ready` — the workspaces mount; the connection banner below the
 *   header consumes `useEventStreamState` (its first consumer: a
 *   subtle strip whenever the WS lane degrades, silent when healthy).
 */

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import { useEventStreamState } from "@/hooks/useJobTracking";
import {
  initializeBackend,
  StartupError,
  type StartupInfo,
  type StartupFailureCause,
} from "@/services/api/startup";

type Phase = "connecting" | "failed" | "ready";

interface Failure {
  readonly stage: StartupError["stage"];
  readonly cause: StartupFailureCause;
  readonly message: string;
}

/**
 * The honest headline per failure cause (part14 D-3 — §3.1 item
 * 12): "The backend did not start" is reserved for failures that
 * actually are startup failures; authorization rejections and
 * reachability/CORS blocks get named for what they are.
 */
function headlineFor(cause: StartupFailureCause): string {
  switch (cause) {
    case "auth":
      return "The backend rejected this app's request";
    case "network":
      return "The backend could not be reached";
    case "http":
    case "protocol":
      return "The backend is not healthy";
    default:
      return "The backend did not start";
  }
}

export function StartupGate({ children }: { children: ReactNode }) {
  const [phase, setPhase] = useState<Phase>("connecting");
  const [info, setInfo] = useState<StartupInfo | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const booted = useRef(false);

  // The boot coroutine: every state transition inside it happens
  // after an await (the handshake is genuinely async), so nothing
  // synchronous-in-effect ever touches setState — the effect below
  // only kicks it, and the Retry button re-arms the `connecting`
  // state from the click handler (an event, the sanctioned site).
  const run = useCallback(async () => {
    try {
      const started = await initializeBackend();
      setInfo(started);
      setPhase("ready");
    } catch (err) {
      const failure: Failure =
        err instanceof StartupError
          ? { stage: err.stage, cause: err.cause, message: err.message }
          : {
              stage: "acquire",
              cause: "handshake",
              message: err instanceof Error ? err.message : String(err),
            };
      setFailure(failure);
      setPhase("failed");
    }
  }, []);

  const retry = useCallback(() => {
    setPhase("connecting");
    setFailure(null);
    void run();
  }, [run]);

  useEffect(() => {
    if (booted.current) {
      // React StrictMode mounts effects twice in dev — the §14.1
      // sequence runs exactly once per process
      return;
    }
    booted.current = true;
    // The §14.1 boot is exactly the "synchronize React with an
    // external system" case effects exist for (the external system
    // is the Python sidecar handshake). Every setState in `run`
    // happens strictly after `await initializeBackend()` — the
    // async boundary the set-state-in-effect heuristic cannot see,
    // so this one-time external-sync call is suppressed, not
    // restructured into a fake.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void run();
  }, [run]);

  if (phase === "connecting") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--color-background)] p-6">
        <div className="w-full max-w-md rounded-xl border border-[var(--color-border)] bg-[var(--color-card)] p-6 text-center shadow-sm">
          <div className="mx-auto mb-4 h-8 w-8 animate-pulse rounded-full bg-[var(--color-primary)]/20" aria-hidden="true" />
          <h1 className="text-base font-semibold text-[var(--color-foreground)]">
            Starting the local scientific backend
          </h1>
          <p className="mt-2 text-sm text-[var(--color-muted-foreground)]">
            Completing the secure startup handshake: launching the Python
            sidecar, verifying backend health, and opening the live event
            stream. The nine workspaces unlock when the backend is ready.
          </p>
          <p className="sr-only" role="status">
            Backend startup in progress
          </p>
        </div>
      </div>
    );
  }

  if (phase === "failed" && failure !== null) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--color-background)] p-6">
        <div className="w-full max-w-md rounded-xl border border-red-300 bg-[var(--color-card)] p-6 shadow-sm">
          <h1 className="text-base font-semibold text-[var(--color-foreground)]">
            {headlineFor(failure.cause)}
          </h1>
          <p className="mt-1 text-xs font-medium uppercase tracking-wide text-[var(--color-muted-foreground)]">
            stage: {failure.stage} · cause: {failure.cause}
          </p>
          <p className="mt-2 break-words text-sm text-[var(--color-foreground)]">
            {failure.message}
          </p>
          <button
            type="button"
            onClick={retry}
            className="mt-4 inline-flex h-9 items-center justify-center rounded-md bg-[var(--color-primary)] px-4 text-sm font-medium text-[var(--color-primary-foreground)] transition-colors hover:opacity-90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
          >
            Retry the startup handshake
          </button>
        </div>
      </div>
    );
  }

  // phase === "ready"
  return (
    <>
      {info !== null && (
        <span className="sr-only" data-testid="startup-ready">
          backend {info.version} ready via {info.source}
        </span>
      )}
      <ConnectionBanner />
      {children}
    </>
  );
}

/**
 * The live-update health strip — the shell's honest WS indicator
 * (silent while the stream is open or has not yet been asked to
 * connect; a visible strip the moment it degrades, reconnects, or
 * closes).  REST stays the authoritative path, so this is a notice,
 * never a blocker: job progress falls back to polling.
 */
function ConnectionBanner() {
  const ws = useEventStreamState();
  if (ws.state === "open" || ws.state === "idle") {
    return null;
  }
  const label =
    ws.state === "connecting"
      ? "connecting the live event stream"
      : ws.state === "reconnecting"
        ? `reconnecting the live event stream (attempt ${ws.attempts})`
        : "live updates closed — job progress falls back to REST polling";
  return (
    <div
      role="status"
      className="border-b border-[var(--color-border)] bg-[var(--color-muted)] px-4 py-1.5 text-center text-xs text-[var(--color-muted-foreground)]"
    >
      {label}
    </div>
  );
}
