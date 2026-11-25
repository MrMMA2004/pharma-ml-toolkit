import { useCallback, useEffect, useState } from "react";
import {
  ArchiveRestore,
  Database,
  Download,
  FlaskConical,
  HardDrive,
  LoaderCircle,
  RefreshCw,
  WifiOff,
  Workflow,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { DemoDataSection } from "@/components/demo/DemoDataSection";
import { DatabasePreferences } from "@/components/system/DatabasePreferences";
import { DiagnosticsCard } from "@/components/system/DiagnosticsCard";
import { JobCenterCard } from "@/components/system/JobCenterCard";
import { WORKSPACES } from "@/app/workspaces";
import { ApiError } from "@/services/api/client";
import { artifactsApi } from "@/services/artifacts/client";
import type { ArtifactView } from "@/services/artifacts/client";
import {
  databaseApi,
  type DatabaseStatus,
  type JobHistoryRow,
} from "@/services/database/client";

/**
 * The Introduction page — part11 §10.2.
 *
 * Professional and informative; explains what PharmaML Studio is, its
 * computational pharmaceutical purpose, the supported scientific
 * areas, the general workflow, the offline nature and the local
 * backend architecture. It must NOT become a second dashboard of nine
 * large duplicate workspace cards — the workspace list below is a
 * concise directory (label + one line), not a card grid.
 *
 * (part12 Phase 3) The page now also carries the ONE live surface it
 * always claimed: the studio database card below fetches the real
 * `/api/v1/database` status and the persisted job history, and drives
 * the backup/restore routes — before this, "backup and restore built
 * in" was a description with no GUI consumer anywhere in the app.
 */
export function IntroductionPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-10">
      <div className="max-w-3xl">
        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[var(--color-border)] bg-[var(--color-muted)] px-3 py-1 text-xs font-medium text-[var(--color-muted-foreground)]">
          <WifiOff aria-hidden className="h-3.5 w-3.5" />
          Fully offline — your data never leaves this machine
        </div>
        <h1 className="text-4xl font-bold tracking-tight text-[var(--color-foreground)]">
          PharmaML Studio
        </h1>
        <p className="mt-4 text-lg leading-relaxed text-[var(--color-muted-foreground)]">
          An offline desktop workbench for computational pharmacy: QSAR
          machine learning over molecular descriptors and fingerprints,
          deep learning, semi-supervised and generative molecular design,
          docking, PK/PD, molecular-dynamics analysis, and a
          visualization workspace — over one Python backend, driven
          through a typed local REST + WebSocket API.
        </p>
      </div>

      <div className="mt-10 grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <FlaskConical aria-hidden className="h-5 w-5 text-[var(--color-primary)]" />
            <CardTitle className="text-base">The scientific backend stays Python</CardTitle>
            <CardDescription>
              Chemistry, ML, DL, generative, docking, PK/PD, MD analysis
              and applicability-domain logic run in the packaged Python
              backend — the UI never reimplements science.
            </CardDescription>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader>
            <Workflow aria-hidden className="h-5 w-5 text-[var(--color-primary)]" />
            <CardTitle className="text-base">Sequential workflows, real jobs</CardTitle>
            <CardDescription>
              Every workspace follows its end-to-end workflow; long
              computations run as managed background jobs with live
              progress over WebSocket and honest cancellation.
            </CardDescription>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader>
            <HardDrive aria-hidden className="h-5 w-5 text-[var(--color-primary)]" />
            <CardTitle className="text-base">Local, durable, yours</CardTitle>
            <CardDescription>
              Datasets, models, artifacts and job history persist in a
              local SQLite database plus a backend-managed file vault —
              with backup and restore built in (the live status, history
              and backup controls live in the database card below).
            </CardDescription>
          </CardHeader>
        </Card>
      </div>

      <div className="mt-12">
        <h2 className="text-xl font-semibold text-[var(--color-foreground)]">
          The nine workspaces
        </h2>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[var(--color-muted-foreground)]">
          Navigation runs left-to-right in a fixed order — the workflow
          from data to insight. Each workspace exposes its complete
          backend configuration; work continues in later campaign rows.
        </p>
        <ol className="mt-6 divide-y divide-[var(--color-border)] overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]">
          {WORKSPACES.map((workspace, index) => (
            <li key={workspace.id} className="flex gap-4 px-5 py-4">
              <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-[var(--color-muted)] text-xs font-semibold text-[var(--color-muted-foreground)]">
                {index + 1}
              </span>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-medium text-[var(--color-foreground)]">
                    {workspace.label}
                  </span>
                  {workspace.state === "shell" ? (
                    <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] font-medium text-[var(--color-muted-foreground)]">
                      workflow row queued
                    </span>
                  ) : null}
                </div>
                <p className="mt-1 text-sm leading-relaxed text-[var(--color-muted-foreground)]">
                  {workspace.summary}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </div>

      {/* part16 §8 (Phase 5): the global demonstration entry point —
          every shipped sample, loadable per its target workflow */}
      <div className="mt-12">
        <DemoDataSection
          heading="Demo data — every workspace"
        />
      </div>

      <StudioDatabaseSection />

      {/* part17 §3.2 (the G1 lane): the durable user-preference
          manager — the /database/preferences quartet's first consumer
          (the backend surface that had no client lane at all) */}
      <DatabasePreferences />

      {/* part17 §3.2 (the G1 lane): the global Job Center — the live
          cross-workspace session ledger over GET /api/v1/jobs with the
          authoritative result + cancellation routes (the JobRunCard
          per-workflow lane was never this surface) */}
      <JobCenterCard />

      {/* part16 §15.5 (Phase 12): the diagnostics/recovery surface —
          dependency + scientific-tool availability, storage health,
          crash diagnostics, logs, recoverable job state, and the safe
          cache cleanup, beside the database/backup half it completes */}
      <div className="mt-12">
        <DiagnosticsCard />
      </div>
    </div>
  );
}

/** The persisted job-state chip (the history view of JobRunCard's states). */
function JobStateChip({ state }: { state: string }) {
  const tone =
    state === "COMPLETED"
      ? "text-emerald-600 dark:text-emerald-400"
      : state === "FAILED"
        ? "text-red-600 dark:text-red-400"
        : "text-[var(--color-muted-foreground)]";
  return (
    <span
      className={`inline-flex items-center rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] font-medium ${tone}`}
    >
      {state}
    </span>
  );
}

function _errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    return `The backend answered ${err.status}: ${err.message}`;
  }
  return err instanceof Error ? err.message : String(err);
}

/**
 * The studio database card — the live persistence surface.
 *
 * Fetches `GET /api/v1/database` (status + row counts) and
 * `GET /api/v1/database/jobs?limit=6` (the persisted history), and
 * drives the backup (`POST /database/backup` → a `database`-kind
 * artifact, downloadable through the digest-verified artifact lane)
 * and restore (`POST /database/restore`) routes. Restore replaces
 * the live registry with the backup's registry — the confirm step
 * states that plainly instead of hiding it.
 */
function StudioDatabaseSection() {
  const [status, setStatus] = useState<DatabaseStatus | null>(null);
  const [ephemeralReason, setEphemeralReason] = useState<string | null>(null);
  const [jobs, setJobs] = useState<readonly JobHistoryRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<"backup" | "restore" | null>(null);
  const [backupArtifact, setBackupArtifact] = useState<ArtifactView | null>(null);
  const [restoreId, setRestoreId] = useState("");
  const [confirmingRestore, setConfirmingRestore] = useState(false);
  const [restoreNote, setRestoreNote] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await databaseApi.status();
      if (next.persistent) {
        setStatus(next);
        setEphemeralReason(null);
        try {
          setJobs(await databaseApi.jobHistory(6));
        } catch (err) {
          // a durable instance should answer the history lane; a
          // non-409 failure here is a real error worth showing
          if (err instanceof ApiError && err.status === 409) {
            setJobs(null);
          } else {
            throw err;
          }
        }
      } else {
        setStatus(null);
        setJobs(null);
        setEphemeralReason(next.reason);
      }
    } catch (err) {
      setError(_errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function handleBackup() {
    setBusy("backup");
    setError(null);
    setRestoreNote(null);
    try {
      const result = await databaseApi.backup();
      setBackupArtifact(result.artifact);
      setRestoreId((current) => current || result.artifact.artifact_id);
      await refresh();
    } catch (err) {
      setError(_errorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  async function handleRestore() {
    if (!restoreId.trim()) return;
    setBusy("restore");
    setError(null);
    setRestoreNote(null);
    try {
      await databaseApi.restore(restoreId.trim());
      setRestoreNote(
        "Database restored — the live registry was rebuilt from the backup.",
      );
      setBackupArtifact(null);
      setConfirmingRestore(false);
      await refresh();
    } catch (err) {
      setError(_errorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  async function handleDownloadBackup() {
    if (!backupArtifact) return;
    setBusy("backup");
    setError(null);
    try {
      const download = await artifactsApi.download(backupArtifact.artifact_id);
      const url = URL.createObjectURL(download.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = download.filename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(_errorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="mt-12" aria-label="Studio database">
      <div className="flex items-center gap-2">
        <Database aria-hidden className="h-5 w-5 text-[var(--color-primary)]" />
        <h2 className="text-xl font-semibold text-[var(--color-foreground)]">
          The studio database
        </h2>
      </div>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[var(--color-muted-foreground)]">
        Live status, the persisted job history and the backup/restore
        controls — the same SQLite database and file vault the
        workspaces write into, read back over the local REST API.
      </p>

      {error ? (
        <div
          role="alert"
          className="mt-4 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
        >
          {error}
        </div>
      ) : null}

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card className="p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-base font-semibold text-[var(--color-foreground)]">
                Status
              </h3>
              <p className="mt-1 text-sm text-[var(--color-muted-foreground)]">
                {loading
                  ? "Reading the local database…"
                  : status
                    ? `Durable — ${status.database} in ${status.data_dir} (schema v${status.schema_version}, ${status.journal_mode} journal, integrity: ${status.integrity}).`
                    : ephemeralReason
                      ? `Non-durable instance — ${ephemeralReason}`
                      : "Status unavailable."}
              </p>
            </div>
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
          {status ? (
            <dl className="mt-4 grid grid-cols-2 gap-2 text-sm sm:grid-cols-3">
              {(
                [
                  ["datasets", "Datasets"],
                  ["trained_models", "Trained models"],
                  ["generative_priors", "Generative priors"],
                  ["jobs", "Job records"],
                  ["artifacts", "Artifacts"],
                  ["user_preferences", "Preferences"],
                ] as const
              ).map(([key, label]) => (
                <div
                  key={key}
                  className="rounded-md border border-[var(--color-border)] bg-[var(--color-muted)] px-3 py-2"
                >
                  <dt className="text-xs text-[var(--color-muted-foreground)]">
                    {label}
                  </dt>
                  <dd className="font-semibold text-[var(--color-foreground)]">
                    {status.tables[key] ?? 0}
                  </dd>
                </div>
              ))}
            </dl>
          ) : null}
        </Card>

        <Card className="p-5">
          <h3 className="text-base font-semibold text-[var(--color-foreground)]">
            Backup &amp; restore
          </h3>
          <p className="mt-1 text-sm text-[var(--color-muted-foreground)]">
            A backup snapshots the SQLite database plus the vault
            payloads into one registered artifact. Restoring replaces
            the live registry with the backup&rsquo;s registry.
          </p>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              onClick={() => void handleBackup()}
              disabled={busy !== null || !status}
            >
              {busy === "backup" ? (
                <LoaderCircle aria-hidden className="h-4 w-4 animate-spin" />
              ) : (
                <ArchiveRestore aria-hidden className="h-4 w-4" />
              )}
              Create backup
            </Button>
            {backupArtifact ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => void handleDownloadBackup()}
                disabled={busy !== null}
              >
                <Download aria-hidden className="h-4 w-4" />
                Download {backupArtifact.size_bytes.toLocaleString()} B
              </Button>
            ) : null}
          </div>
          {backupArtifact ? (
            <p className="mt-2 font-mono text-xs text-[var(--color-muted-foreground)]">
              {backupArtifact.artifact_id} · sha256 {backupArtifact.sha256.slice(0, 16)}…
            </p>
          ) : null}
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <label className="sr-only" htmlFor="restore-artifact-id">
              Artifact id to restore
            </label>
            <input
              id="restore-artifact-id"
              value={restoreId}
              onChange={(event) => setRestoreId(event.target.value)}
              placeholder="backup artifact id"
              className="min-w-0 flex-1 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-1.5 font-mono text-xs text-[var(--color-foreground)] placeholder:text-[var(--color-muted-foreground)]"
            />
            {confirmingRestore ? (
              <Button
                size="sm"
                onClick={() => void handleRestore()}
                disabled={busy !== null || !restoreId.trim()}
              >
                {busy === "restore" ? (
                  <LoaderCircle aria-hidden className="h-4 w-4 animate-spin" />
                ) : null}
                Confirm replace
              </Button>
            ) : (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setConfirmingRestore(true)}
                disabled={busy !== null || !restoreId.trim()}
              >
                Restore…
              </Button>
            )}
          </div>
          {confirmingRestore ? (
            <p className="mt-2 text-xs text-red-600 dark:text-red-400">
              Restoring replaces the session&rsquo;s current datasets,
              models, priors and artifacts with the backup&rsquo;s
              registry.
            </p>
          ) : null}
          {restoreNote ? (
            <p className="mt-2 text-xs text-emerald-600 dark:text-emerald-400">
              {restoreNote}
            </p>
          ) : null}
        </Card>
      </div>

      <div className="mt-4 overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]">
        <div className="border-b border-[var(--color-border)] px-5 py-3">
          <h3 className="text-sm font-semibold text-[var(--color-foreground)]">
            Recent jobs (persisted history)
          </h3>
        </div>
        {jobs && jobs.length > 0 ? (
          <table className="w-full text-left text-sm">
            <thead className="bg-[var(--color-muted)] text-xs text-[var(--color-muted-foreground)]">
              <tr>
                <th scope="col" className="px-5 py-2 font-medium">State</th>
                <th scope="col" className="px-5 py-2 font-medium">Operation</th>
                <th scope="col" className="px-5 py-2 font-medium">Workspace</th>
                <th scope="col" className="px-5 py-2 font-medium">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--color-border)]">
              {jobs.map((job) => (
                <tr key={job.job_id} className="align-top">
                  <td className="px-5 py-2.5">
                    <JobStateChip state={job.state} />
                  </td>
                  <td className="px-5 py-2.5 font-mono text-xs text-[var(--color-foreground)]">
                    {job.operation}
                  </td>
                  <td className="px-5 py-2.5 text-[var(--color-muted-foreground)]">
                    {job.workspace}
                  </td>
                  <td className="px-5 py-2.5 text-xs text-[var(--color-muted-foreground)]">
                    {job.created_at}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="px-5 py-4 text-sm text-[var(--color-muted-foreground)]">
            {loading
              ? "Reading the job history…"
              : "No persisted jobs yet — run a computation in any workspace and it lands here."}
          </p>
        )}
      </div>
    </section>
  );
}
