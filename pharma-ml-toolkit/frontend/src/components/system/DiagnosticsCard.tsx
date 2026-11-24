/**
 * part16 §15.5 — the Diagnostics card (the Phase-12 frontend lane).
 *
 * The §15.5 "Diagnostics and Recovery" surface over
 * `GET /api/v1/system/diagnostics`, mounted in the Introduction
 * workspace beside the studio-database section (which already owns
 * the backup/restore + job-history half of §15.5): the dependency
 * availability table (the requirements.txt vocabulary with the
 * required/optional tiers), the scientific-tool probes (rdkit /
 * torch / selfies with their functional results), the storage
 * health (disk, the SQLite database, the vault), the crash
 * diagnostics (the recent FAILED jobs with their errors), the log
 * configuration report, and the recoverable job state (the two
 * views honestly split).
 *
 * The one mutation control is the safe cache cleanup — §15.5's own
 * lane: `POST /api/v1/system/cache/cleanup` clears ONLY the
 * enumerated derived cache, and the card renders the preserved
 * guarantee verbatim with the result. Nothing else here mutates
 * anything: the diagnostics are read-only by design.
 */

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  CheckCircle2,
  CircleAlert,
  HardDrive,
  LoaderCircle,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { systemApi, type DiagnosticsSnapshot } from "@/services/system/client";

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatBytes(bytes: number): string {
  if (bytes >= 1_073_741_824) {
    return `${(bytes / 1_073_741_824).toFixed(2)} GB`;
  }
  if (bytes >= 1_048_576) {
    return `${(bytes / 1_048_576).toFixed(1)} MB`;
  }
  if (bytes >= 1_024) {
    return `${(bytes / 1_024).toFixed(1)} kB`;
  }
  return `${bytes} B`;
}

/** One availability row: importable version or the honest error. */
function DependencyLine({
  name,
  version,
  error,
  tier,
}: {
  name: string;
  version: string | null;
  error: string | null;
  tier: "required" | "optional";
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5">
      <span className="flex items-baseline gap-2">
        <span className="font-mono text-xs text-[var(--color-foreground)]">
          {name}
        </span>
        <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted-foreground)]">
          {tier}
        </span>
      </span>
      {error !== null ? (
        <span className="text-right text-[11px] text-[var(--color-destructive)]">
          unavailable ({error})
        </span>
      ) : (
        <span className="text-right font-mono text-[11px] text-[var(--color-muted-foreground)]">
          {version || "—"}
        </span>
      )}
    </div>
  );
}

/** The scientific-tool probe row (null probe = the absent tool). */
function ToolLine({
  name,
  version,
  functionalProbe,
}: {
  name: string;
  version: string | null;
  functionalProbe: boolean | null;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5">
      <span className="flex items-baseline gap-2">
        {functionalProbe === true ? (
          <CheckCircle2
            aria-hidden
            className="size-3 text-[var(--color-primary)]"
          />
        ) : functionalProbe === false ? (
          <CircleAlert
            aria-hidden
            className="size-3 text-[var(--color-destructive)]"
          />
        ) : (
          <span aria-hidden className="size-3 text-center text-[10px] text-[var(--color-muted-foreground)]">
            —
          </span>
        )}
        <span className="font-mono text-xs text-[var(--color-foreground)]">
          {name}
        </span>
      </span>
      <span className="text-right font-mono text-[11px] text-[var(--color-muted-foreground)]">
        {version ?? "absent"}
      </span>
    </div>
  );
}

/**
 * The §15.5 diagnostics card. Self-contained by design: it loads
 * its own data through `systemApi.diagnostics()`, no global state,
 * no context — the host page only positions it.
 */
export function DiagnosticsCard() {
  const [snapshot, setSnapshot] = useState<DiagnosticsSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cleanupBusy, setCleanupBusy] = useState(false);
  const [cleanupNote, setCleanupNote] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSnapshot(await systemApi.diagnostics());
    } catch (requestError) {
      setError(describeError(requestError));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const runCleanup = useCallback(async () => {
    if (cleanupBusy) {
      return;
    }
    setCleanupBusy(true);
    setError(null);
    setCleanupNote(null);
    try {
      const answer = await systemApi.cleanupCache();
      setCleanupNote(
        `Cleared ${answer.cleared.samples_stats_entries} cached sample stat${
          answer.cleared.samples_stats_entries === 1 ? "" : "s"
        } — user data preserved.`,
      );
    } catch (requestError) {
      setError(describeError(requestError));
    } finally {
      setCleanupBusy(false);
    }
  }, [cleanupBusy]);

  return (
    <Card id="diagnostics-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Activity aria-hidden className="size-4" />
          System Diagnostics &amp; Recovery
        </CardTitle>
        <CardDescription>
          The §15.5 surface: dependency and scientific-tool availability,
          storage health, crash diagnostics, logs, recoverable job state —
          and the safe cache cleanup.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => void refresh()}
            disabled={loading}
            aria-label="Refresh the diagnostics snapshot"
          >
            {loading ? (
              <LoaderCircle aria-hidden className="size-4 animate-spin" />
            ) : (
              <RefreshCw aria-hidden className="size-4" />
            )}
            Refresh
          </Button>
          {snapshot !== null && (
            <span className="text-xs text-[var(--color-muted-foreground)]">
              backend <span className="font-mono">{snapshot.backend.version}</span>{" "}
              · Python {snapshot.backend.python}
            </span>
          )}
        </div>

        {error !== null && (
          <div
            role="alert"
            className="rounded-md border border-[var(--color-destructive)] bg-[var(--color-destructive)]/10 px-3 py-2 text-sm"
          >
            <span className="font-medium">Diagnostics failed:</span>{" "}
            <span className="text-[var(--color-muted-foreground)]">{error}</span>
          </div>
        )}

        {loading && snapshot === null && (
          <p role="status" className="text-sm text-[var(--color-muted-foreground)]">
            Loading the diagnostics snapshot…
          </p>
        )}

        {snapshot !== null && (
          <>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
                  Scientific tools
                </h4>
                <div className="rounded-md border border-[var(--color-border)] p-2">
                  {snapshot.scientific_tools.map((tool) => (
                    <ToolLine
                      key={tool.name}
                      name={tool.name}
                      version={tool.version}
                      functionalProbe={tool.functional_probe}
                    />
                  ))}
                </div>
                <p className="mt-1 text-[11px] leading-snug text-[var(--color-muted-foreground)]">
                  Functional probes: {snapshot.scientific_tools
                    .map((t) => t.probe)
                    .join("; ")}.
                </p>
              </div>

              <div>
                <h4 className="mb-1 flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
                  <HardDrive aria-hidden className="size-3" />
                  Storage health
                </h4>
                <div className="rounded-md border border-[var(--color-border)] p-2 text-xs">
                  <div className="flex items-baseline justify-between gap-3 py-0.5">
                    <span className="text-[var(--color-muted-foreground)]">disk free</span>
                    <span className="font-mono text-[var(--color-foreground)]">
                      {formatBytes(snapshot.storage.disk.free)} /{" "}
                      {formatBytes(snapshot.storage.disk.total)}
                    </span>
                  </div>
                  {snapshot.storage.database !== null ? (
                    <>
                      <div className="flex items-baseline justify-between gap-3 py-0.5">
                        <span className="text-[var(--color-muted-foreground)]">database</span>
                        <span className="font-mono text-[var(--color-foreground)]">
                          {formatBytes(snapshot.storage.database.size_bytes ?? 0)} ·{" "}
                          {snapshot.storage.integrity ?? "—"}
                        </span>
                      </div>
                      <div className="flex items-baseline justify-between gap-3 py-0.5">
                        <span className="text-[var(--color-muted-foreground)]">vault</span>
                        <span className="font-mono text-[var(--color-foreground)]">
                          {formatBytes(snapshot.storage.vault_bytes ?? 0)}
                        </span>
                      </div>
                    </>
                  ) : (
                    <p className="py-0.5 text-[11px] leading-snug text-[var(--color-muted-foreground)]">
                      Ephemeral session — {snapshot.storage.reason ?? "no data_dir"}.
                    </p>
                  )}
                </div>
              </div>
            </div>

            <details className="rounded-md border border-[var(--color-border)] p-2">
              <summary className="cursor-pointer text-xs text-[var(--color-muted-foreground)]">
                The dependency availability table ({snapshot.dependencies.length} probes)
              </summary>
              <div className="mt-2 max-h-64 overflow-y-auto">
                {snapshot.dependencies.map((row) => (
                  <DependencyLine
                    key={row.name}
                    name={row.name}
                    version={row.version}
                    error={row.error}
                    tier={row.tier}
                  />
                ))}
              </div>
            </details>

            <div>
              <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
                Crash diagnostics
              </h4>
              {snapshot.crash.recent_failed.length === 0 ? (
                <p className="rounded-md border border-[var(--color-border)] bg-[var(--color-muted)] p-2 text-xs text-[var(--color-muted-foreground)]">
                  No failed jobs recorded ({snapshot.crash.source}).
                </p>
              ) : (
                <ul className="space-y-1">
                  {snapshot.crash.recent_failed.map((row, index) => (
                    <li
                      key={row.job_id ?? index}
                      className="rounded-md border border-[var(--color-border)] p-2 text-xs"
                    >
                      <div className="flex flex-wrap items-baseline gap-2">
                        <span className="font-mono text-[10px] text-[var(--color-muted-foreground)]">
                          {row.job_id ?? "—"}
                        </span>
                        <span className="text-[var(--color-foreground)]">
                          {row.operation ?? "—"}
                        </span>
                        <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted-foreground)]">
                          {row.workspace ?? "—"}
                        </span>
                        <span className="ml-auto text-[10px] text-[var(--color-muted-foreground)]">
                          {row.completed_at ?? ""}
                        </span>
                      </div>
                      {row.error !== null && (
                        <p className="mt-1 font-mono text-[11px] text-[var(--color-destructive)]">
                          {row.error}
                        </p>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
                  Logs (configuration)
                </h4>
                <div className="rounded-md border border-[var(--color-border)] p-2 text-xs">
                  <div className="flex items-baseline justify-between gap-3 py-0.5">
                    <span className="text-[var(--color-muted-foreground)]">level</span>
                    <span className="font-mono text-[var(--color-foreground)]">
                      {snapshot.logs.level}
                    </span>
                  </div>
                  <div className="flex items-baseline justify-between gap-3 py-0.5">
                    <span className="text-[var(--color-muted-foreground)]">handlers</span>
                    <span className="font-mono text-[var(--color-foreground)]">
                      {snapshot.logs.handlers
                        .map((h) => h.kind)
                        .join(", ") || "none configured"}
                    </span>
                  </div>
                </div>
              </div>
              <div>
                <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
                  Recoverable job state
                </h4>
                <div className="rounded-md border border-[var(--color-border)] p-2 text-xs">
                  <div className="flex items-baseline justify-between gap-3 py-0.5">
                    <span className="text-[var(--color-muted-foreground)]">persisted</span>
                    <span className="font-mono text-[var(--color-foreground)]">
                      {Object.entries(snapshot.recovery.persisted_by_state)
                        .map(([state, n]) => `${state}:${n}`)
                        .join(" ") || "none"}
                    </span>
                  </div>
                  <div className="flex items-baseline justify-between gap-3 py-0.5">
                    <span className="text-[var(--color-muted-foreground)]">in-session</span>
                    <span className="font-mono text-[var(--color-foreground)]">
                      {Object.entries(snapshot.recovery.in_session_by_state)
                        .filter(([, n]) => n > 0)
                        .map(([state, n]) => `${state}:${n}`)
                        .join(" ") || "idle"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="border-t border-[var(--color-border)] pt-3">
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => void runCleanup()}
                  disabled={cleanupBusy}
                  aria-label="Clear the derived cache (user data preserved)"
                >
                  {cleanupBusy ? (
                    <LoaderCircle aria-hidden className="size-4 animate-spin" />
                  ) : (
                    <Trash2 aria-hidden className="size-4" />
                  )}
                  Clear derived cache
                </Button>
                {cleanupNote !== null && (
                  <span role="status" className="text-xs text-[var(--color-foreground)]">
                    {cleanupNote}
                  </span>
                )}
              </div>
              <p className="mt-1 text-[11px] leading-snug text-[var(--color-muted-foreground)]">
                §15.5: "Do not delete user data during diagnostic or cleanup
                operations." The cleared list is the complete inventory of the
                backend's derived caches — datasets, models, priors, artifacts,
                preferences, the job history, and the run lineage are preserved.
              </p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
