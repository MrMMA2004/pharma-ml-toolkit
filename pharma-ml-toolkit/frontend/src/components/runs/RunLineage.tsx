/**
 * part16 §15.2 — the Experiment log card (the Phase-12 frontend lane).
 *
 * The query surface over the run store, mounted in the Compare stage:
 * the lineage table (the `/api/v1/runs` summaries, newest first) with
 * the status badge, the model/dataset identity, the CV mean (the honest
 * dash when absent), the mode, the seed and the creation stamp. A row
 * click fetches the full §15.2 record on demand (`/api/v1/runs/{id}`)
 * and expands the definition-style lineage blocks — the dataset
 * lineage, the algorithm + toolkit version, the preprocessing
 * pipeline, the requested AND tuned hyperparameters, the
 * cross-validation configuration, the evaluation metrics, the model
 * artifact reference with its fetch hint, the environment snapshot and
 * the job reference — cached per run_id so a re-expand never
 * refetches. Failed runs carry their typed error verbatim; nothing is
 * ever fabricated (absent numbers render as the honest dash).
 *
 * Self-contained by design: it loads its own data through `runsApi`
 * with a refresh callback, no global state, no context, no settings
 * dependency — the host stage only positions it.
 */

import {
  Fragment,
  useCallback,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { History, LoaderCircle, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { runsApi } from "@/services/runs/client";
import type {
  ExperimentRun,
  RunStatus,
  RunSummary,
} from "@/services/runs/types";

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatBytes(size: number | null): string {
  if (size === null) {
    return "—";
  }
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KiB`;
  }
  return `${(size / (1024 * 1024)).toFixed(2)} MiB`;
}

/** The status badge vocabulary — one chip per honest terminal state. */
const STATUS_BADGE: Record<RunStatus, string> = {
  completed:
    "border-[var(--color-primary)]/40 bg-[var(--color-primary)]/10 text-[var(--color-primary)]",
  failed:
    "border-[var(--color-destructive)]/40 bg-[var(--color-destructive)]/10 text-[var(--color-destructive)]",
  cancelled:
    "border-[var(--color-border)] bg-[var(--color-muted)] text-[var(--color-muted-foreground)]",
};

/** A dismissible-free error banner — MLPage's own markup, inline. */
function ErrorBanner({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="rounded-md border border-[var(--color-destructive)] bg-[var(--color-destructive)]/10 px-3 py-2 text-sm text-[var(--color-foreground)]"
    >
      <span className="font-medium">Request failed:</span>{" "}
      <span className="text-[var(--color-muted-foreground)]">{message}</span>
    </div>
  );
}

function DetailRow({
  label,
  value,
}: {
  label: string;
  value: ReactNode;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5">
      <span className="text-[11px] text-[var(--color-muted-foreground)]">
        {label}
      </span>
      <span className="text-right font-mono text-[11px] text-[var(--color-foreground)]">
        {value}
      </span>
    </div>
  );
}

/** One labeled lineage block of the §15.2 definition-style detail. */
function DetailSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section aria-label={title} className="min-w-0">
      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
        {title}
      </div>
      {children}
    </section>
  );
}

/** The honest JSON view for the requested/tuned parameter sets. */
function JsonBlock({ value }: { value: Readonly<Record<string, unknown>> | null }) {
  return (
    <pre className="max-h-40 overflow-auto rounded-md border border-[var(--color-border)] bg-[var(--color-muted)] p-2 font-mono text-[11px] leading-snug text-[var(--color-foreground)]">
      {value === null ? "—" : JSON.stringify(value, null, 2)}
    </pre>
  );
}

function packageLine(run: ExperimentRun): string {
  const entries = Object.entries(run.environment.packages);
  if (entries.length === 0) {
    return "—";
  }
  return entries
    .map(([name, version]) => `${name} ${version ?? "—"}`)
    .join(" · ");
}

/** The full §15.2 record as the expanded lineage blocks. */
function RunDetail({ run }: { run: ExperimentRun }) {
  return (
    <div className="grid gap-x-6 gap-y-3 rounded-md border border-[var(--color-border)] bg-[var(--color-muted)]/30 p-3 md:grid-cols-2">
      <DetailSection title="Dataset lineage">
        <DetailRow label="Name" value={run.dataset.name} />
        <DetailRow
          label="Shape"
          value={`${run.dataset.shape[0]} × ${run.dataset.shape[1]}`}
        />
        <DetailRow label="Target" value={run.dataset.target ?? "—"} />
        <DetailRow label="Source file" value={run.dataset.source_filename} />
        <DetailRow label="Size" value={formatBytes(run.dataset.size_bytes)} />
        <DetailRow label="Features" value={run.dataset.n_features} />
        <DetailRow label="Created" value={run.dataset.created_at} />
      </DetailSection>

      <DetailSection title="Algorithm">
        <DetailRow label="Name" value={run.algorithm.name} />
        <DetailRow label="Task" value={run.algorithm.task} />
        <DetailRow label="Toolkit" value={run.algorithm.toolkit_version} />
      </DetailSection>

      <DetailSection title="Preprocessing">
        <DetailRow label="Enabled" value={run.preprocessing.enabled ? "on" : "off"} />
        <p className="mt-1 break-words text-[11px] leading-snug text-[var(--color-foreground)]">
          {run.preprocessing.pipeline}
        </p>
      </DetailSection>

      <DetailSection title="Hyperparameters">
        <DetailRow
          label="Tuned"
          value={run.hyperparameters.tuned ? "yes" : "no"}
        />
        <p className="mt-1 text-[10px] text-[var(--color-muted-foreground)]">
          requested
        </p>
        <JsonBlock value={run.hyperparameters.requested} />
        <p className="mt-1 text-[10px] text-[var(--color-muted-foreground)]">
          best_params
        </p>
        <JsonBlock value={run.hyperparameters.best_params} />
      </DetailSection>

      <DetailSection title="Cross-validation">
        <DetailRow label="Random seed" value={run.random_seed} />
        <DetailRow label="Folds" value={run.cross_validation.cv_folds} />
        <DetailRow
          label="Tuning"
          value={run.cross_validation.tuning ? "on" : "off"}
        />
        <DetailRow
          label="n_jobs"
          value={`${run.cross_validation.n_jobs_requested} requested → ${
            run.cross_validation.n_jobs_effective ?? "—"
          } effective`}
        />
        <DetailRow
          label="Parallelism"
          value={
            run.cross_validation.parallelism_advisory?.reason ?? "unclamped"
          }
        />
      </DetailSection>

      <DetailSection title="Evaluation">
        <DetailRow
          label="CV mean"
          value={run.evaluation.cv_mean === null ? "—" : run.evaluation.cv_mean.toFixed(4)}
        />
        <DetailRow
          label="CV std"
          value={run.evaluation.cv_std === null ? "—" : run.evaluation.cv_std.toFixed(4)}
        />
        <DetailRow
          label="CV scores"
          value={
            run.evaluation.cv_scores === null
              ? "—"
              : run.evaluation.cv_scores.map((score) => score.toFixed(3)).join(", ")
          }
        />
        <DetailRow label="Primary metric" value={run.evaluation.primary_metric ?? "—"} />
      </DetailSection>

      <DetailSection title="Model artifact">
        {run.model_artifact === null ? (
          <p className="text-[11px] leading-snug text-[var(--color-muted-foreground)]">
            No model artifact — the run never produced one (never invented).
          </p>
        ) : (
          <>
            <DetailRow label="model_id" value={run.model_artifact.model_id} />
            <DetailRow label="Name" value={run.model_artifact.name} />
            <DetailRow
              label="Train rows"
              value={run.model_artifact.n_train_rows ?? "—"}
            />
            <DetailRow
              label="Fit seconds"
              value={
                run.model_artifact.fit_seconds === null
                  ? "—"
                  : run.model_artifact.fit_seconds.toFixed(2)
              }
            />
            <DetailRow label="Fetch" value={run.model_artifact.fetch} />
          </>
        )}
      </DetailSection>

      <DetailSection title="Environment">
        <DetailRow label="Toolkit" value={run.environment.toolkit} />
        <DetailRow label="Python" value={run.environment.python} />
        <DetailRow label="Platform" value={run.environment.platform} />
        <DetailRow label="Machine" value={run.environment.machine} />
        <p className="mt-1 break-words text-[11px] leading-snug text-[var(--color-foreground)]">
          {packageLine(run)}
        </p>
      </DetailSection>

      <DetailSection title="Job reference">
        <DetailRow label="Job" value={run.job_id ?? "sync mode"} />
        <DetailRow label="History" value={run.logs.history} />
      </DetailSection>

      {run.error !== null ? (
        <div
          role="alert"
          className="rounded-md border border-[var(--color-destructive)] bg-[var(--color-destructive)]/10 px-3 py-2 text-xs text-[var(--color-destructive)] md:col-span-2"
        >
          {run.error}
        </div>
      ) : null}
    </div>
  );
}

export interface RunLineageProps {
  /** Extra classes for the root card (the host stage positions it). */
  readonly className?: string;
}

export function RunLineage({ className }: RunLineageProps) {
  const [runs, setRuns] = useState<readonly RunSummary[] | null>(null);
  const [count, setCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [openRunId, setOpenRunId] = useState<string | null>(null);
  const [details, setDetails] = useState<Readonly<Record<string, ExperimentRun>>>({});
  const [detailBusyRunId, setDetailBusyRunId] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const listing = await runsApi.list();
      setRuns(listing.runs);
      setCount(listing.count);
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // one fetch per run_id (the §15.2 record caches per run); a failed
  // fetch never caches, so a re-open retries the record
  const toggleRun = useCallback(
    async (runId: string) => {
      if (openRunId === runId) {
        setOpenRunId(null);
        return;
      }
      setOpenRunId(runId);
      if (details[runId] !== undefined) {
        return;
      }
      setDetailError(null);
      setDetailBusyRunId(runId);
      try {
        const detail = await runsApi.detail(runId);
        setDetails((prev) => ({ ...prev, [runId]: detail }));
        setDetailBusyRunId(null);
      } catch (err) {
        setDetailError(describeError(err));
        setDetailBusyRunId(null);
      }
    },
    [openRunId, details],
  );

  const detailCell = (runId: string): ReactNode => {
    const detail = details[runId];
    if (detailBusyRunId === runId && detail === undefined) {
      return (
        <p className="text-xs text-[var(--color-muted-foreground)]">
          <LoaderCircle aria-hidden className="mr-1 inline h-3.5 w-3.5 animate-spin" />
          loading the run record…
        </p>
      );
    }
    if (detail === undefined) {
      if (detailError !== null) {
        return <ErrorBanner message={detailError} />;
      }
      return null;
    }
    return <RunDetail run={detail} />;
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">
          <History aria-hidden className="h-4 w-4 text-[var(--color-primary)]" />
          Experiment log
          <span
            className="rounded border border-[var(--color-border)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--color-muted-foreground)]"
            aria-label="run count"
          >
            {busy ? "…" : `${count} ${count === 1 ? "run" : "runs"}`}
          </span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void refresh()}
            disabled={busy}
            aria-label="refresh the experiment log"
            className="ml-auto"
          >
            <RotateCw aria-hidden className="h-3.5 w-3.5" />
          </Button>
        </CardTitle>
        <CardDescription>
          Every training run&apos;s lineage record — the dataset, the
          configuration, the environment and the model artifact joined
          per run (the record writes itself when a model trains).
        </CardDescription>
      </CardHeader>
      <CardContent>
        {error !== null ? <ErrorBanner message={error} /> : null}
        {runs === null ? (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            <LoaderCircle aria-hidden className="mr-1 inline h-3.5 w-3.5 animate-spin" />
            loading the experiment log…
          </p>
        ) : runs.length === 0 ? (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            No runs recorded yet — train a model in stage 7 and the lineage
            records itself.
          </p>
        ) : (
          <div className="overflow-hidden rounded-lg border border-[var(--color-border)]">
            <div className="max-h-96 overflow-auto">
              <table
                className="w-full border-collapse text-xs"
                aria-label="Experiment runs"
              >
                <thead className="sticky top-0 bg-[var(--color-muted)]">
                  <tr>
                    {["Status", "Model", "Dataset", "CV mean", "Mode", "Seed", "Created"].map(
                      (header) => (
                        <th
                          key={header}
                          scope="col"
                          className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                        >
                          {header}
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <Fragment key={run.run_id}>
                      <tr
                        onClick={() => void toggleRun(run.run_id)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            void toggleRun(run.run_id);
                          }
                        }}
                        tabIndex={0}
                        aria-label={`run ${run.run_id}`}
                        className={`cursor-pointer border-t border-[var(--color-border)] focus:outline-none focus-visible:bg-[var(--color-accent)] ${
                          openRunId === run.run_id
                            ? "bg-[var(--color-accent)]"
                            : "bg-[var(--color-card)] hover:bg-[var(--color-accent)]/50"
                        }`}
                      >
                        <td className="whitespace-nowrap px-3 py-1.5">
                          <span
                            className={`rounded-full border px-1.5 py-0.5 text-[10px] font-semibold ${STATUS_BADGE[run.status]}`}
                          >
                            {run.status}
                          </span>
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">
                          {run.model_name}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {run.dataset_name}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {run.metric === null ? "—" : run.metric.toFixed(4)}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {run.mode}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {run.random_seed}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {run.created_at}
                        </td>
                      </tr>
                      {openRunId === run.run_id ? (
                        <tr className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                          <td colSpan={7} className="px-3 py-3">
                            {detailCell(run.run_id)}
                          </td>
                        </tr>
                      ) : null}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {runs !== null && runs.length > 0 ? (
          <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
            The lineage table (newest first) — select a row for the run&apos;s
            full record; failed runs carry their typed error, and the
            fit-only fields stay honestly absent.
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}
