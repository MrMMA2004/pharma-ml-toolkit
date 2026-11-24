/**
 * part16 §15.3 (Phase 12) — the Model registry card (the GUI projection).
 *
 * The registry's two joined views over the EXISTING backend surface (§8's
 * integration rule — the ml client's own operations, never a parallel
 * registry architecture):
 *
 *   the catalog — every trainable algorithm the registry ships
 *                 (`GET /api/v1/ml/models` with the availability
 *                 stated per row: the unavailable entries carry their
 *                 `requires` line, never a silent omission), the task
 *                 filter and the availability filter applied over the
 *                 one full listing;
 *
 *   the trained registry — the session's trained models (`GET
 *                 /api/v1/ml/trained`), newest first, with the derived
 *                 version column (the created_at order within each
 *                 model+dataset identity — derived, stated as
 *                 derived, never invented), the on-demand detail
 *                 (`GET /api/v1/ml/trained/{model_id}` cached per
 *                 model_id), the export action (`POST
 *                 /api/v1/artifacts/model` — the REAL backward-
 *                 compatible joblib export, downloaded through the
 *                 artifacts client's digest-verified lane) and the
 *                 deletion action (`DELETE /api/v1/ml/trained/{model_id}`
 *                 — the bounded store's own evict; the §15.2 lineage and
 *                 the exported artifacts outlive the row).
 *
 * The honest surfaces throughout: absent numbers render as the dash,
 * the bounded store's own 16-model contract is stated where it acts,
 * and every failure surfaces through the error banner — never a
 * fabricated row. Self-contained by design: it loads its own data
 * through mlApi/artifactsApi with a refresh callback; the host stage
 * only positions it.
 */

import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { Boxes, Download, LoaderCircle, RotateCw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { artifactsApi } from "@/services/artifacts/client";
import { mlApi } from "@/services/ml/client";
import type {
  ModelCatalogEntry,
  TrainedModelView,
} from "@/services/ml/client";

/** The store's own bound (routes consume `MAX_TRAINED_MODELS` = 16). */
const TRAINED_STORE_BOUND = 16;

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

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

type CatalogFilter = "all" | "classification" | "regression" | "both";

const CATALOG_FILTERS: ReadonlyArray<{
  readonly key: CatalogFilter;
  readonly label: string;
}> = [
  { key: "all", label: "All" },
  { key: "classification", label: "Classification" },
  { key: "regression", label: "Regression" },
  { key: "both", label: "Dual-task" },
];

function entryMatchesFilter(
  entry: ModelCatalogEntry,
  filter: CatalogFilter,
  availableOnly: boolean,
): boolean {
  if (availableOnly && !entry.available) {
    return false;
  }
  if (filter === "all") {
    return true;
  }
  // the backend's own rule: task="classification" keeps the
  // classification entries AND the dual-task ("both") entries
  return entry.task === filter || entry.task === "both";
}

/** The catalog section — the registry of what can be trained. */
function CatalogSection({
  entries,
  busy,
  error,
  filter,
  onFilter,
  availableOnly,
  onAvailableOnly,
}: {
  entries: readonly ModelCatalogEntry[] | null;
  busy: boolean;
  error: string | null;
  filter: CatalogFilter;
  onFilter: (filter: CatalogFilter) => void;
  availableOnly: boolean;
  onAvailableOnly: (value: boolean) => void;
}) {
  const visible = useMemo(
    () =>
      (entries ?? []).filter((entry) =>
        entryMatchesFilter(entry, filter, availableOnly),
      ),
    [entries, filter, availableOnly],
  );
  const total = entries?.length ?? 0;
  return (
    <section aria-label="Registry catalog" className="min-w-0">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-[var(--color-foreground)]">
          Catalog
        </span>
        <span
          className="rounded border border-[var(--color-border)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--color-muted-foreground)]"
          aria-label="catalog entry count"
        >
          {busy && entries === null ? "…" : `${visible.length} of ${total}`}
        </span>
        <div
          className="flex flex-wrap gap-1"
          role="group"
          aria-label="catalog task filter"
        >
          {CATALOG_FILTERS.map(({ key, label }) => (
            <Button
              key={key}
              size="sm"
              variant={filter === key ? "default" : "outline"}
              onClick={() => onFilter(key)}
              aria-pressed={filter === key}
              className="h-6 px-2 text-[11px]"
            >
              {label}
            </Button>
          ))}
        </div>
        <label className="ml-auto flex cursor-pointer items-center gap-1.5 text-[11px] text-[var(--color-muted-foreground)]">
          <input
            type="checkbox"
            checked={!availableOnly}
            onChange={(event) => onAvailableOnly(!event.target.checked)}
            aria-label="include unavailable catalog entries"
          />
          show unavailable
        </label>
      </div>
      {error !== null ? <ErrorBanner message={error} /> : null}
      {error !== null ? null : entries === null ? (
        busy ? (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            <LoaderCircle aria-hidden className="mr-1 inline h-3.5 w-3.5 animate-spin" />
            loading the registry catalog…
          </p>
        ) : null
      ) : (
        <div className="overflow-hidden rounded-lg border border-[var(--color-border)]">
          <div className="max-h-72 overflow-auto">
            <table
              className="w-full border-collapse text-xs"
              aria-label="Model registry catalog"
            >
              <thead className="sticky top-0 bg-[var(--color-muted)]">
                <tr>
                  {["Model", "Task", "Available", "Notes"].map((header) => (
                    <th
                      key={header}
                      scope="col"
                      className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {visible.map((entry) => (
                  <tr
                    key={entry.name}
                    className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                  >
                    <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">
                      {entry.name}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                      {entry.task}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5">
                      {entry.available ? (
                        <span className="rounded-full border border-[var(--color-primary)]/40 bg-[var(--color-primary)]/10 px-1.5 py-0.5 text-[10px] font-semibold text-[var(--color-primary)]">
                          available
                        </span>
                      ) : (
                        <span
                          className="rounded-full border border-[var(--color-border)] bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--color-muted-foreground)]"
                          title={
                            entry.requires === null
                              ? "optional backend missing"
                              : `requires ${entry.requires}`
                          }
                        >
                          requires {entry.requires ?? "—"}
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                      {entry.notes || "—"}
                    </td>
                  </tr>
                ))}
                {visible.length === 0 ? (
                  <tr>
                    <td
                      colSpan={4}
                      className="px-3 py-4 text-center text-[var(--color-muted-foreground)]"
                    >
                      No catalog entries under this filter.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
        The catalog is the backend registry&apos;s own projection — one source
        of truth (`GET /api/v1/ml/models`); the unavailable entries carry
        their missing dependency, never a silent omission.
      </p>
    </section>
  );
}

/**
 * The derived version index: the created_at order within each
 * model+task+dataset identity. Derived from the registry's own listing
 * (stated as derived — the store holds ids, not version numbers).
 */
function versionIndex(
  model: TrainedModelView,
  all: readonly TrainedModelView[],
): { readonly version: number; readonly total: number } {
  const identity = `${model.name}|${model.task}|${model.dataset_id}`;
  const siblings = all
    .filter((m) => `${m.name}|${m.task}|${m.dataset_id}` === identity)
    .sort((a, b) => a.created_at.localeCompare(b.created_at));
  const version = siblings.findIndex((m) => m.model_id === model.model_id) + 1;
  return { version, total: siblings.length };
}

function JsonBlock({ value }: { value: Readonly<Record<string, unknown>> }) {
  return (
    <pre className="max-h-40 overflow-auto rounded-md border border-[var(--color-border)] bg-[var(--color-muted)] p-2 font-mono text-[11px] leading-snug text-[var(--color-foreground)]">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

function DetailRow({ label, value }: { label: string; value: ReactNode }) {
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

/** One trained model's expanded registry record. */
function TrainedDetail({ model }: { model: TrainedModelView }) {
  return (
    <div className="grid gap-x-6 gap-y-3 rounded-md border border-[var(--color-border)] bg-[var(--color-muted)]/30 p-3 md:grid-cols-2">
      <div aria-label="registry identity" className="min-w-0">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
          Identity
        </div>
        <DetailRow label="model_id" value={model.model_id} />
        <DetailRow label="Dataset" value={model.dataset_id || "—"} />
        <DetailRow label="Target" value={model.target || "—"} />
        <DetailRow label="Train rows" value={model.n_train_rows} />
        <DetailRow label="Fit seconds" value={model.fit_seconds.toFixed(2)} />
        <DetailRow label="Created" value={model.created_at} />
      </div>
      <div aria-label="registry evaluation" className="min-w-0">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
          Evaluation
        </div>
        <DetailRow
          label="CV mean"
          value={model.cv_mean === null ? "—" : model.cv_mean.toFixed(4)}
        />
        <DetailRow
          label="CV std"
          value={model.cv_std === null ? "—" : model.cv_std.toFixed(4)}
        />
        <DetailRow
          label="CV scores"
          value={
            model.cv_scores === null
              ? "—"
              : model.cv_scores.map((score) => score.toFixed(3)).join(", ")
          }
        />
        <DetailRow label="Features" value={`${model.features.length}`} />
      </div>
      <div aria-label="registry configuration" className="min-w-0 md:col-span-2">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
          Training configuration (best_params)
        </div>
        <JsonBlock value={model.best_params} />
        {model.features.length > 0 ? (
          <p className="mt-1 break-words text-[11px] leading-snug text-[var(--color-muted-foreground)]">
            {model.features.join(" · ")}
          </p>
        ) : null}
      </div>
    </div>
  );
}

export interface ModelRegistryProps {
  /** Extra classes for the root card (the host stage positions it). */
  readonly className?: string;
}

export function ModelRegistry({ className }: ModelRegistryProps) {
  // the catalog (one full fetch — the filters derive over it)
  const [catalog, setCatalog] = useState<readonly ModelCatalogEntry[] | null>(
    null,
  );
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogBusy, setCatalogBusy] = useState(false);
  const [filter, setFilter] = useState<CatalogFilter>("all");
  const [availableOnly, setAvailableOnly] = useState(false);

  // the trained registry
  const [trained, setTrained] = useState<readonly TrainedModelView[] | null>(
    null,
  );
  const [trainedError, setTrainedError] = useState<string | null>(null);
  const [trainedBusy, setTrainedBusy] = useState(false);
  const [openModelId, setOpenModelId] = useState<string | null>(null);
  const [details, setDetails] = useState<
    Readonly<Record<string, TrainedModelView>>
  >({});
  const [detailBusyId, setDetailBusyId] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  // the export / delete action lanes (per-model status lines)
  const [actionBusyId, setActionBusyId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionStatus, setActionStatus] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setCatalogBusy(true);
    setTrainedBusy(true);
    setCatalogError(null);
    setTrainedError(null);
    try {
      // the full §15.3 evaluation listing — availability stated per row
      const [catalogResponse, trainedResponse] = await Promise.all([
        mlApi.catalog(undefined, false),
        mlApi.trained(),
      ]);
      setCatalog(catalogResponse.models);
      setTrained(trainedResponse.models);
    } catch (err) {
      setCatalogError(describeError(err));
      setTrainedError(describeError(err));
    } finally {
      setCatalogBusy(false);
      setTrainedBusy(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // one fetch per model_id (the registry record caches per model);
  // a failed fetch never caches, so a re-open retries
  const toggleModel = useCallback(
    async (modelId: string) => {
      if (openModelId === modelId) {
        setOpenModelId(null);
        return;
      }
      setOpenModelId(modelId);
      if (details[modelId] !== undefined) {
        return;
      }
      setDetailError(null);
      setDetailBusyId(modelId);
      try {
        const detail = await mlApi.trainedView(modelId);
        setDetails((prev) => ({ ...prev, [modelId]: detail }));
        setDetailBusyId(null);
      } catch (err) {
        setDetailError(describeError(err));
        setDetailBusyId(null);
      }
    },
    [openModelId, details],
  );

  const exportModel = useCallback(async (model: TrainedModelView) => {
    setActionError(null);
    setActionStatus(null);
    setActionBusyId(model.model_id);
    try {
      const artifact = await mlApi.exportTrained(model.model_id);
      const download = await artifactsApi.download(artifact.artifact_id);
      const url = URL.createObjectURL(download.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = download.filename;
      anchor.click();
      URL.revokeObjectURL(url);
      setActionStatus(
        `exported ${download.filename} (${artifact.size_bytes} B) — ${
          download.digest_verified === true
            ? `digest verified: ${download.actual_sha256?.slice(0, 16)}…`
            : download.digest_verified === null
              ? "local digest unavailable (insecure context) — the route's X-Artifact-SHA256 was received"
              : "DIGEST MISMATCH — the downloaded bytes differ from the route's promise"
        }`,
      );
    } catch (err) {
      setActionError(describeError(err));
    } finally {
      setActionBusyId(null);
    }
  }, []);

  const evictModel = useCallback(
    async (model: TrainedModelView) => {
      setActionError(null);
      setActionStatus(null);
      setActionBusyId(model.model_id);
      setConfirmDeleteId(null);
      try {
        const answer = await mlApi.evictTrained(model.model_id);
        setTrained((prev) =>
          prev === null
            ? prev
            : prev.filter((m) => m.model_id !== model.model_id),
        );
        setOpenModelId((prev) => (prev === model.model_id ? null : prev));
        setActionStatus(
          `evicted ${model.name} (${answer.evicted}) — ${answer.remaining} remaining in the session registry`,
        );
      } catch (err) {
        // a raced eviction (the id already gone) refreshes honestly
        setActionError(describeError(err));
        await refresh();
      } finally {
        setActionBusyId(null);
      }
    },
    [refresh],
  );

  const trainedNewestFirst = useMemo(
    () => (trained === null ? [] : [...trained].reverse()),
    [trained],
  );

  const detailCell = (modelId: string): ReactNode => {
    const detail = details[modelId];
    if (detailBusyId === modelId && detail === undefined) {
      return (
        <p className="text-xs text-[var(--color-muted-foreground)]">
          <LoaderCircle aria-hidden className="mr-1 inline h-3.5 w-3.5 animate-spin" />
          loading the registry record…
        </p>
      );
    }
    if (detail === undefined) {
      if (detailError !== null) {
        return <ErrorBanner message={detailError} />;
      }
      return null;
    }
    return <TrainedDetail model={detail} />;
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">
          <Boxes aria-hidden className="h-4 w-4 text-[var(--color-primary)]" />
          Model registry
          <span
            className="rounded border border-[var(--color-border)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--color-muted-foreground)]"
            aria-label="trained model count"
          >
            {trainedBusy
              ? "…"
              : `${trained === null ? 0 : trained.length} / ${TRAINED_STORE_BOUND}`}
          </span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void refresh()}
            disabled={catalogBusy || trainedBusy}
            aria-label="refresh the model registry"
            className="ml-auto"
          >
            <RotateCw aria-hidden className="h-3.5 w-3.5" />
          </Button>
        </CardTitle>
        <CardDescription>
          The registry joined in one view — the catalog of trainable models
          with their availability, and the session&apos;s trained models with
          the derived versions, the on-demand record, the joblib export and
          the evict lane.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <CatalogSection
          entries={catalog}
          busy={catalogBusy}
          error={catalogError}
          filter={filter}
          onFilter={setFilter}
          availableOnly={availableOnly}
          onAvailableOnly={setAvailableOnly}
        />

        <section aria-label="Trained model registry" className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-[var(--color-foreground)]">
              Trained models
            </span>
            <span
              className="rounded border border-[var(--color-border)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--color-muted-foreground)]"
              aria-label="trained count against the store bound"
            >
              {trained === null ? "…" : `${trained.length} of ${TRAINED_STORE_BOUND}`}
            </span>
          </div>
          {trainedError !== null ? <ErrorBanner message={trainedError} /> : null}
          {actionError !== null ? <ErrorBanner message={actionError} /> : null}
          {actionStatus !== null ? (
            <p
              role="status"
              className="mb-2 rounded-md border border-[var(--color-border)] bg-[var(--color-muted)]/40 px-3 py-1.5 text-xs text-[var(--color-foreground)]"
            >
              {actionStatus}
            </p>
          ) : null}
          {trained === null ? (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              <LoaderCircle aria-hidden className="mr-1 inline h-3.5 w-3.5 animate-spin" />
              loading the trained-model registry…
            </p>
          ) : trainedNewestFirst.length === 0 ? (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              No trained models in this session yet — train one in stage 7
              and the registry holds it here.
            </p>
          ) : (
            <div className="overflow-hidden rounded-lg border border-[var(--color-border)]">
              <div className="max-h-96 overflow-auto">
                <table
                  className="w-full border-collapse text-xs"
                  aria-label="Trained models"
                >
                  <thead className="sticky top-0 bg-[var(--color-muted)]">
                    <tr>
                      {[
                        "Version",
                        "Model",
                        "Task",
                        "Dataset",
                        "CV mean ± std",
                        "Train rows",
                        "Created",
                        "Actions",
                      ].map((header) => (
                        <th
                          key={header}
                          scope="col"
                          className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                        >
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {trainedNewestFirst.map((model) => {
                      const { version, total } = versionIndex(
                        model,
                        trainedNewestFirst,
                      );
                      const busy = actionBusyId === model.model_id;
                      return (
                        <Fragment key={model.model_id}>
                          <tr
                            onClick={() => void toggleModel(model.model_id)}
                            onKeyDown={(event) => {
                              if (event.key === "Enter" || event.key === " ") {
                                event.preventDefault();
                                void toggleModel(model.model_id);
                              }
                            }}
                            tabIndex={0}
                            aria-label={`trained model ${model.model_id}`}
                            className={`cursor-pointer border-t border-[var(--color-border)] focus:outline-none focus-visible:bg-[var(--color-accent)] ${
                              openModelId === model.model_id
                                ? "bg-[var(--color-accent)]"
                                : "bg-[var(--color-card)] hover:bg-[var(--color-accent)]/50"
                            }`}
                          >
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              v{version} of {total}
                            </td>
                            <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">
                              {model.name}
                            </td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {model.task}
                            </td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {model.dataset_id || "—"}
                            </td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {model.cv_mean === null
                                ? "—"
                                : `${model.cv_mean.toFixed(4)}${
                                    model.cv_std === null
                                      ? ""
                                      : ` ± ${model.cv_std.toFixed(4)}`
                                  }`}
                            </td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {model.n_train_rows}
                            </td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {model.created_at}
                            </td>
                            <td className="whitespace-nowrap px-3 py-1.5">
                              <span className="flex items-center gap-1">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  disabled={busy}
                                  aria-label={`export ${model.name} as a joblib artifact`}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    void exportModel(model);
                                  }}
                                  className="h-6 px-2"
                                >
                                  <Download aria-hidden className="h-3.5 w-3.5" />
                                </Button>
                                <Button
                                  size="sm"
                                  variant={
                                    confirmDeleteId === model.model_id
                                      ? "destructive"
                                      : "outline"
                                  }
                                  disabled={busy}
                                  aria-label={`delete ${model.name} from the session registry`}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    if (confirmDeleteId === model.model_id) {
                                      void evictModel(model);
                                    } else {
                                      setConfirmDeleteId(model.model_id);
                                    }
                                  }}
                                  className="h-6 px-2"
                                >
                                  <Trash2 aria-hidden className="h-3.5 w-3.5" />
                                  {confirmDeleteId === model.model_id
                                    ? "Confirm"
                                    : ""}
                                </Button>
                              </span>
                            </td>
                          </tr>
                          {openModelId === model.model_id ? (
                            <tr className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                              <td colSpan={8} className="px-3 py-3">
                                {detailCell(model.model_id)}
                              </td>
                            </tr>
                          ) : null}
                        </Fragment>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
            The trained registry is the backend store&apos;s own projection —
            the newest {TRAINED_STORE_BOUND} held, the oldest evicted first
            (the store&apos;s bound, not a UI choice). The version column is
            derived from the created_at order within each model+dataset
            identity; the export writes the toolkit&apos;s own
            backward-compatible joblib format (an independent copy that
            outlives the row); the delete evicts the session row only — the
            experiment lineage and the exported artifacts survive it.
          </p>
        </section>
      </CardContent>
    </Card>
  );
}
