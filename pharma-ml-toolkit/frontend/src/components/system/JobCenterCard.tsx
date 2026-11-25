/**
 * part17 §3.2 (Round 360, the G1 lane) — the global Job Center.
 *
 * The mandate's own words: "The existing JobRunCard and per-workflow
 * useJob are not automatically equivalent to a global Job Center."
 * This card IS that surface: the live cross-workspace job ledger over
 * `GET /api/v1/jobs` (every managed job of the CURRENT app session —
 * the list lane that had a typed, API-tested client and no consumer),
 * with the authoritative per-row actions:
 *
 * * the result lane — `GET /api/v1/jobs/{id}/result`, offered only
 *   for COMPLETED rows (the backend answers 409 for anything else;
 *   a non-completed row never offers the button, and a 409 that does
 *   arrive is surfaced verbatim, never swallowed);
 * * the cancellation lane — `POST /api/v1/jobs/{id}/cancel`, offered
 *   only for the live states (QUEUED / RUNNING /
 *   CANCELLATION_REQUESTED) behind a two-step confirmation: the row
 *   action only ARMS, the confirm button fires, and the honest 404
 *   (already-gone) reconciles the ledger instead of hiding;
 * * the refresh lane — the manual re-read of the ledger.
 *
 * The honest split this card must state plainly: this is the LIVE
 * in-memory session ledger; the PERSISTED history that survives
 * restarts is the studio database card's own table (GET
 * /api/v1/database/jobs) — the two views never pretend to be one.
 */

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  LoaderCircle,
  RefreshCw,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { jobsApi } from "@/services/jobs/client";
import { ApiError } from "@/services/api/client";
import type { JobView } from "@/services/jobs/types";

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    const detail = error.detail;
    if (
      typeof detail === "object" &&
      detail !== null &&
      "detail" in detail &&
      typeof (detail as Record<string, unknown>)["detail"] === "string"
    ) {
      return (detail as Record<string, string>)["detail"] ?? `HTTP ${error.status}`;
    }
    if (typeof detail === "string") {
      return detail;
    }
    return `HTTP ${error.status}`;
  }
  return error instanceof Error ? error.message : String(error);
}

/** The live-state vocabulary — only these rows offer cancellation. */
const CANCELLABLE_STATES: readonly string[] = [
  "QUEUED",
  "RUNNING",
  "CANCELLATION_REQUESTED",
];

function stateTone(state: string): string {
  if (state === "COMPLETED") {
    return "text-emerald-600 dark:text-emerald-400";
  }
  if (state === "FAILED") {
    return "text-red-600 dark:text-red-400";
  }
  if (state === "RUNNING" || state === "CANCELLATION_REQUESTED") {
    return "text-amber-600 dark:text-amber-400";
  }
  return "text-[var(--color-muted-foreground)]";
}

function JobStateChip({ state }: { state: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] font-medium ${stateTone(state)}`}
    >
      {state}
    </span>
  );
}

/** The result payload rendered as bounded JSON (never a fake summary). */
function ResultPayload({ payload }: { payload: unknown }) {
  let text: string;
  try {
    text = JSON.stringify(payload, null, 2);
  } catch {
    text = String(payload);
  }
  const bounded = text.length > 4000 ? `${text.slice(0, 4000)}…` : text;
  return (
    <pre className="max-h-72 overflow-auto rounded-md border border-[var(--color-border)] bg-[var(--color-muted)] p-3 font-mono text-[11px] leading-relaxed text-[var(--color-foreground)]">
      {bounded}
    </pre>
  );
}

export function JobCenterCard() {
  const [jobs, setJobs] = useState<readonly JobView[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  // the two-step cancellation gate: the row action only arms it
  const [cancelPending, setCancelPending] = useState<JobView | null>(null);
  const [cancelBusy, setCancelBusy] = useState(false);
  // the result lane (one open row at a time — the honest 409/404
  // failures surface verbatim, never swallowed)
  const [resultFor, setResultFor] = useState<JobView | null>(null);
  const [resultBusy, setResultBusy] = useState(false);
  const [resultPayload, setResultPayload] = useState<unknown>(null);
  const [resultError, setResultError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await jobsApi.list();
      setJobs(response.jobs);
      // the armed confirmation can never outlive its row: a job
      // that left the ledger disarms the gate
      setCancelPending((pending) =>
        pending === null
          ? null
          : response.jobs.find(
              (job) => job.job_id === pending.job_id,
            ) ?? null,
      );
    } catch (err) {
      setError(describeError(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const requestCancel = useCallback((job: JobView) => {
    setCancelPending(job);
    setNotice(null);
    setError(null);
  }, []);

  const cancelRequest = useCallback(() => {
    setCancelPending(null);
  }, []);

  const confirmCancel = useCallback(
    async (job: JobView) => {
      setCancelBusy(true);
      setError(null);
      setNotice(null);
      try {
        const view = await jobsApi.cancel(job.job_id);
        setCancelPending(null);
        setNotice(
          `Cancellation acknowledged for ${view.operation} (${view.state}).`,
        );
        await refresh();
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          // the honest already-gone lane: the job left the ledger —
          // reconcile and say so, never a silent failure
          setCancelPending(null);
          await refresh();
          setNotice(
            `${job.operation} (${job.job_id}) was already gone (404) — ledger refreshed.`,
          );
          return;
        }
        setError(describeError(err));
      } finally {
        setCancelBusy(false);
      }
    },
    [refresh],
  );

  const openResult = useCallback(async (job: JobView) => {
    setResultFor(job);
    setResultPayload(null);
    setResultError(null);
    setResultBusy(true);
    try {
      const response = await jobsApi.result(job.job_id);
      setResultPayload(response.result);
    } catch (err) {
      // a 409 (not completed anymore) or a 404 (already gone) is
      // the backend's own honest answer — render it verbatim
      setResultError(describeError(err));
    } finally {
      setResultBusy(false);
    }
  }, []);

  const closeResult = useCallback(() => {
    setResultFor(null);
    setResultPayload(null);
    setResultError(null);
  }, []);

  return (
    <section className="mt-12" aria-label="Job Center">
      <div className="flex items-center gap-2">
        <Activity aria-hidden className="h-5 w-5 text-[var(--color-primary)]" />
        <h2 className="text-xl font-semibold text-[var(--color-foreground)]">
          The Job Center
        </h2>
      </div>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[var(--color-muted-foreground)]">
        The live cross-workspace job ledger — every managed job of the
        current session over <code className="font-mono text-xs">GET /api/v1/jobs</code>,
        with the authoritative result and cancellation routes. This is the
        in-session operational view; the persisted history that survives
        restarts is the studio database card&rsquo;s own table below.
      </p>

      {error ? (
        <div
          role="alert"
          className="mt-4 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
        >
          {error}
        </div>
      ) : null}
      {notice ? (
        <p role="status" className="mt-4 text-sm text-[var(--color-muted-foreground)]">
          {notice}
        </p>
      ) : null}

      <Card className="mt-4">
        <CardHeader>
          <CardTitle className="text-base">The session ledger</CardTitle>
          <CardDescription>
            {loading
              ? "Reading the ledger…"
              : jobs === null
                ? "The ledger is unavailable."
                : `${jobs.length} job${jobs.length === 1 ? "" : "s"} in the current session.`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => void refresh()}
              disabled={loading}
            >
              {loading ? (
                <LoaderCircle aria-hidden className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw aria-hidden className="h-4 w-4" />
              )}
              Refresh
            </Button>
          </div>
          {jobs !== null && jobs.length > 0 ? (
            <div className="mt-4 max-h-96 space-y-2 overflow-y-auto">
              {jobs.map((job) => (
                <div
                  key={job.job_id}
                  className="rounded-md border border-[var(--color-border)] p-3"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <JobStateChip state={job.state} />
                        <p className="truncate font-mono text-xs font-medium text-[var(--color-foreground)]">
                          {job.operation}
                        </p>
                      </div>
                      <p className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
                        {job.workspace} · progress {job.progress}% ·{" "}
                        {job.created_at.slice(0, 19).replace("T", " ")}
                        {job.completed_at !== null
                          ? ` → ${job.completed_at.slice(0, 19).replace("T", " ")}`
                          : job.started_at !== null
                            ? ` · started ${job.started_at.slice(0, 19).replace("T", " ")}`
                            : ""}{" "}
                        · id {job.job_id}
                      </p>
                      {job.error !== null ? (
                        <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">
                          {job.error}
                        </p>
                      ) : null}
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {job.state === "COMPLETED" ? (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            resultFor?.job_id === job.job_id
                              ? closeResult()
                              : void openResult(job)
                          }
                          aria-label={`Fetch the result of ${job.operation}`}
                        >
                          {resultFor?.job_id === job.job_id
                            ? "Hide result"
                            : "Result"}
                        </Button>
                      ) : null}
                      {CANCELLABLE_STATES.includes(job.state) ? (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => requestCancel(job)}
                          aria-label={`Cancel ${job.operation}`}
                        >
                          <XCircle aria-hidden className="h-4 w-4" />
                          Cancel…
                        </Button>
                      ) : null}
                    </div>
                  </div>
                  {resultFor?.job_id === job.job_id ? (
                    <div className="mt-3">
                      {resultBusy ? (
                        <p className="text-xs text-[var(--color-muted-foreground)]">
                          fetching the completed result…
                        </p>
                      ) : resultError !== null ? (
                        <p className="text-xs text-red-600 dark:text-red-400">
                          {resultError}
                        </p>
                      ) : (
                        <ResultPayload payload={resultPayload} />
                      )}
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          ) : jobs !== null && jobs.length === 0 ? (
            <p className="mt-4 text-sm text-[var(--color-muted-foreground)]">
              No jobs this session yet — run a computation in any workspace
              and it lands here.
            </p>
          ) : null}

          {cancelPending !== null ? (
            <div className="mt-4 rounded-md border border-[var(--color-destructive)]/40 p-4">
              <p className="text-sm font-semibold text-[var(--color-foreground)]">
                <XCircle aria-hidden className="mr-1 inline h-4 w-4" />
                Cancel {cancelPending.operation} ({cancelPending.state})?
              </p>
              <p className="mt-2 text-sm text-[var(--color-muted-foreground)]">
                The job&rsquo;s cooperative cancellation is requested through
                POST /api/v1/jobs/{cancelPending.job_id}/cancel — a running
                computation stops at its next checkpoint. Completed jobs and
                the persisted history stay untouched.
              </p>
              <div className="mt-3 flex gap-2">
                <Button variant="outline" size="sm" onClick={cancelRequest} disabled={cancelBusy}>
                  Keep it running
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => void confirmCancel(cancelPending)}
                  disabled={cancelBusy}
                >
                  {cancelBusy ? "Cancelling…" : "Confirm cancel"}
                </Button>
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </section>
  );
}
